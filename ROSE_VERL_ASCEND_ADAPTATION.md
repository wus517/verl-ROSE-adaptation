# ROSE 在 VeRL 上的适配与 Ascend NPU 实施方案

> 论文：**Reinforced Efficient Reasoning via Semantically Diverse Exploration**  
> 本地论文：[paper/ROSE.pdf](paper/ROSE.pdf)  
> VeRL 基线：`main@00cd5b44`  
> 目标平台：华为 Ascend NPU，优先采用 `vLLM + vllm-ascend` rollout 以及 FSDP2 或 Megatron 训练后端  
> 文档性质：实施设计，不代表当前仓库已经完成这些代码改造

## 1. 目标与结论

ROSE 不是简单替换 GRPO advantage 的算法。它同时改变了两个阶段：

1. **Rollout 阶段**：同一个 prompt 的多条 response 不再完全独立生成，而是形成一棵由语义熵选择分叉点的推理树。
2. **Advantage 阶段**：不再把同一个 response-level advantage 均匀广播给整条回答，而是根据树节点 value 的差值生成 segment-level advantage，并对过长的正确回答做长度校准。

在当前 VeRL V1 架构中，建议采用以下总体方案：

```text
训练 prompt
   │
   ▼
自定义 RoseAgentLoopManagerTQ / RoseAgentLoopWorkerTQ
   │
   ├── vllm-ascend 在 NPU 上生成完整 continuation
   │        └── 返回 sampled logprob + 每个位置 top-20 token/logprob
   │
   ├── host CPU 计算 semantic entropy 并选择下一分叉点
   │
   ├── 重复生成，直到每个 prompt 得到 G 条 leaf trajectory
   │
   └── reward function 为每条 leaf 计算 binary reward
            │
            ▼
V1 Trainer 从 TransferQueue 读取 leaf、reward 和 tree metadata
            │
            ▼
ROSE tree advantage estimator
   ├── node value
   ├── segment advantage
   └── length-aware calibration
            │
            ▼
Ascend NPU 上执行 PPO + KL 更新
```

核心结论如下：

- ROSE 的树采样必须放在自定义 V1 AgentLoop manager 中，不能只改 `adv_estimator`。
- 当前 async vLLM server 会把整数 `logprobs=20` 错误转换成 `logprobs=0`，必须修复。
- semantic entropy 首版建议在 **NPU 服务器本机的 host CPU** 上批量计算，而不是侵入 vllm-ascend worker。
- tree advantage 和 length calibration 放在 trainer/controller CPU 上计算即可。
- actor、reference、rollout、old/ref logprob、PPO forward/backward 仍然运行在 NPU 上。
- FSDP2、Megatron、rollout TP 的选择只影响部署，不改变 ROSE 算法。
- 第一版不要启用 fully async、off-policy 或 prefill/decode disaggregation；一棵树必须由同一个 policy version 完整生成。

## 2. ROSE 算法拆解

### 2.1 Generation entropy

对于问题 `q`、第 `i` 条回答 `o_i` 和生成位置 `k`，论文定义：

```text
H_k = - Σ_{v∈V} pθ(v | q, o_i,<k) log pθ(v | q, o_i,<k)
```

完整计算需要整个 vocabulary 的分布。论文为语义分歧选择 top-20 token；作者公开实现也使用 vLLM 返回的有限 top-logprobs 近似 generation entropy。因此在 VeRL 适配中建议提供两种模式：

- `strict_reference`：完全复现作者公开代码，只对 vLLM 返回集合计算 `H_k`，不重新归一化概率。
- `normalized_topk`：对 top-K 概率重新归一化后计算熵，仅用于消融实验，不作为默认复现配置。

默认采用 `strict_reference`。

### 2.2 Semantic divergence

在位置 `k` 选择概率最高的 top-20 token，记为 `V_k`。从模型 input embedding 中取得每个候选 token 的向量 `e_v`，论文定义语义分歧：

```text
SD_k = - Σ p_i p_j cos(e_i, e_j)
```

论文公式书写为候选 token 两两求和；作者公开实现会排除对角项。对归一化 embedding 矩阵 `E ∈ R^(K×H)` 和概率向量 `p ∈ R^K`，作者实现等价于：

```text
y    = E^T p
pair = ||y||² - ||p||²
SD_k = -pair
```

其中 `||p||²` 用来减去 `cos(e_i, e_i)=1` 的对角项。适配时应把这一行为写成明确的兼容选项，默认使用作者实现的 `exclude_diagonal=true`。

### 2.3 Semantic entropy

论文用 generation entropy 和 semantic divergence 的乘积作为分叉指标：

```text
SE_k = H_k × SD_k
```

每生成一条新 response，计算其中所有新 token 位置的 `SE_k`，然后在当前已经存在的全部 trajectory 中寻找全局最大值。

### 2.4 ε-exploration 和树形 rollout

对于每个 prompt，需要生成 `G` 条 leaf trajectory：

1. 第 0 条 trajectory 从 root 完整生成。
2. 对后续每条 trajectory：
   - 以概率 `ε` 从 root 独立生成；
   - 以概率 `1-ε` 从当前所有 trajectory 中 `SE_k` 最大的位置分叉。
3. 从 pivot 分叉时保留 pivot 前的 prefix，pivot token 本身重新采样。
4. 生成新的 continuation，加入候选树。
5. 重复直到得到 `G` 条 leaf。

论文默认：

```text
G = 8
ε = 0.5
top_k = 20
```

需要特别区分：

- `ε=1`：所有回答从 root 独立生成，退化为类似 Dr.GRPO 的探索形式。
- `ε=0`：完全局部分叉，容易过度聚焦在少数局部路径。
- `0<ε<1`：同时保留树的深度探索和 root 级广度探索。

### 2.5 Node value

一条 leaf response 上的 root、所有 pivot 和 terminal 将回答划分为若干 segment。对于任意树节点 `b_j`，定义经过该节点的 leaf 集合为 `Ω_bj`：

```text
V(b_j) = mean_{o_m ∈ Ω_bj} reward(o_m)
```

论文使用可验证的 binary reward：

```text
correct   -> 1
incorrect -> 0
```

因此节点 value 可以解释为“从该节点继续推理最终得到正确答案的经验概率”。

### 2.6 Segment advantage

若某条 leaf 的相邻路径节点为 `b_{j-1}` 和 `b_j`，那么两个节点之间所有 token 的 advantage 为：

```text
A_t = V(b_j) - V(b_{j-1})
```

这意味着：

- 正确的中间推理 segment 即使最终回答错误，也可能获得正 advantage。
- 导致错误分支的 segment 会得到负 advantage。
- 同一条 response 的不同 segment 可以获得不同 credit。

### 2.7 Length-aware calibration

对同一 prompt 的所有正确 leaf：

1. 找到最短正确回答 `o_s`。
2. 对其他正确回答 `o_c`，找到它与 `o_s` 的真实最后公共树节点 `b_c`。
3. 对 `o_c` 中位于 `b_c` 之后的 token，执行：

```text
A_t ← A_t - |A_t| × (1 - ((|o_s|-b_c) / (|o_c|-b_c))^α)
```

其中 `α` 控制长度惩罚强度。论文搜索范围为：

```text
α ∈ {0.5, 1, 2, 3}
```

建议初始使用 `α=1`，并记录 accuracy/length 的 Pareto 曲线。

### 2.8 PPO objective

ROSE 采用 Dr.GRPO 风格的 PPO objective，并加入 reference KL：

```text
L = - mean(min(r_t A_t, clip(r_t) A_t)) + β KL(πθ || πref)
```

论文主要参数：

```text
learning_rate = 1e-6
clip_ratio    = 0.2
kl_coef       = 0.001
ppo_epochs    = 1（建议保持严格 on-policy）
```

VeRL 已有 PPO loss、old logprob、reference logprob 和 KL loss，无需重新实现完整 actor loss。

## 3. 当前 VeRL 架构与缺口

### 3.1 V1 AgentLoop 扩展点

当前默认使用 V1 trainer。`verl/trainer/main_ppo.py` 支持通过配置加载自定义 manager：

```yaml
actor_rollout_ref.rollout.agent.agent_loop_manager_class: recipe.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ
```

对应入口：

- `verl/trainer/main_ppo.py:111`
- `verl/workers/config/rollout.py:79`

自定义 manager 需要：

1. 实现 `generate_sequences()`。
2. 将生成结果写入 TransferQueue。
3. 保持 V1 trainer 期望的 key 和字段约定。

### 3.2 默认多 trajectory 行为不满足 ROSE

默认 `AgentLoopManagerTQ` 会把同一 prompt 的 `n` 个 session 作为相互独立的生成任务。ROSE 要求同一 prompt 内存在顺序依赖：第 `r` 条 trajectory 的 prefix 取决于前 `r` 条 trajectory 的 semantic entropy，因此不能直接使用默认的 `n` 次并发独立采样。

正确的并发粒度是：

- **prompt 之间并行**；
- **同一个 prompt 的树内生成按 trajectory 顺序执行**。

### 3.3 当前 async vLLM server 只支持 sampled-token logprob

`verl/workers/rollout/vllm_rollout/vllm_async_server.py:620` 当前逻辑：

```python
sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
```

这会把整数 `20` 当成 truthy 值并转换为 `0`，导致无法获得 top-20。

`verl/workers/rollout/vllm_rollout/vllm_async_server.py:713` 当前只提取 sampled token 的 logprob，没有把完整 top-K 暴露给 AgentLoop。

### 3.4 Advantage 入口缺少 tree metadata

V1 trainer 在 `verl/trainer/ppo/v1/trainer_base.py:1716` 读取固定字段：

```text
uid
response_mask
rm_scores
rollout_log_probs
old_log_probs
ref_log_prob
values
```

ROSE estimator 还需要 tree metadata。建议把 trainer 改造成通用扩展：允许通过 `algorithm.advantage_extra_fields` 指定额外字段，而不是在 trainer 中硬编码 ROSE 字段。

### 3.5 Advantage registry 可以复用

`verl/trainer/ppo/core_algos.py:113` 已有 `ADV_ESTIMATOR_REGISTRY`。ROSE 可以在 recipe 中注册：

```python
@register_adv_est("rose_tree")
def compute_rose_tree_advantage(...):
    ...
```

不要求把算法实现直接堆进 `core_algos.py`。自定义 manager 模块加载时可以导入 `tree_advantage.py` 完成注册。

## 4. 推荐目录与文件改造

建议新增：

```text
recipe/rose/
├── __init__.py
├── rose_agent_loop.py
├── rose_agent_loop_tq.py
├── semantic_entropy.py
├── tree_advantage.py
├── prepare_embeddings.py
├── config/
│   └── rose_npu.yaml
└── run_qwen3_rose_npu.sh
```

建议修改：

| 文件 | 改动目的 |
|---|---|
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 支持整数 top-K logprobs，并返回紧凑 top-K 数据 |
| `verl/trainer/ppo/v1/trainer_base.py` | 允许 estimator 读取配置指定的额外 metadata |
| `verl/trainer/config/algorithm.py` | 可选：正式定义 `advantage_extra_fields` 配置字段 |
| `verl/trainer/config/ppo_trainer.yaml` | 可选：增加空的默认 `advantage_extra_fields` |
| `tests/...` | 增加 top-K、semantic entropy、tree advantage 和 V1 集成测试 |

不建议直接复制作者仓库中的旧版 `vllm_rollout_spmd.py`。作者实现基于旧同步 rollout，而当前 VeRL 主分支默认使用 V1 async AgentLoop 与 TransferQueue，直接搬运会绕过当前权重同步、调度和 replay buffer 逻辑。

## 5. 配置设计

建议新增以下配置。新增 Hydra 字段在命令行中需要使用 `+` 或 `++`。

```yaml
actor_rollout_ref:
  rollout:
    n: 8
    name: vllm
    agent:
      agent_loop_manager_class: recipe.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ
    rose:
      enable: true
      num_trajectories: 8
      epsilon: 0.5
      top_k: 20
      branch_metric: semantic_entropy
      generation_entropy_mode: strict_reference
      exclude_semantic_diagonal: true
      embedding_path: /local_nvme/rose/input_embeddings.f16.npy
      embedding_meta_path: /local_nvme/rose/input_embeddings.json
      semantic_device: cpu
      semantic_chunk_size: 256
      cache_node_entropy: true
      validation_mode: independent

algorithm:
  adv_estimator: rose_tree
  advantage_extra_fields:
    - rose_tree_metadata
  rose:
    length_calibration_alpha: 1.0
    require_binary_reward: true
    fail_on_invalid_tree: true
    normalize_advantage: false
```

约束校验：

- `num_trajectories == actor_rollout_ref.rollout.n`。
- `0 <= epsilon <= 1`。
- `top_k >= 2`。
- `semantic_device ∈ {cpu, npu}`。
- 训练时 `validation_mode=independent`，不要让验证指标依赖训练树策略。
- 如果 `require_binary_reward=true`，reward 必须接近 `0/1`，否则 fail closed。

## 6. vLLM/vllm-ascend top-K 接口改造

### 6.1 请求侧兼容

修改 `vllm_async_server.py`，保留布尔接口的历史行为，同时允许整数：

```python
requested_logprobs = sampling_params.pop("logprobs", None)

if isinstance(requested_logprobs, bool):
    requested_logprobs = 0 if requested_logprobs else None
elif requested_logprobs is not None:
    requested_logprobs = int(requested_logprobs)
    if requested_logprobs < 0:
        raise ValueError("logprobs must be non-negative")

sampling_params["logprobs"] = requested_logprobs
```

语义：

- `False` 或 `None`：不返回 generation logprob。
- `True`：保持旧行为，只确保 sampled token logprob 可用，相当于 `0`。
- `20`：请求 top-20，同时 vLLM 可能额外返回 sampled token。

### 6.2 返回侧提取

对于每一个生成位置，vLLM 的 `logprobs_dict` 可能包含 K 或 K+1 项。必须：

1. 通过 `Logprob.rank` 筛选真正的 top-K。
2. sampled token logprob 继续写入 `TokenOutput.log_probs`。
3. top-K token ID 使用 `int32`。
4. top-K logprob 使用 `float32`。
5. 不依赖 Python 字典迭代顺序。
6. 检查 token ID 是否位于 embedding vocabulary 范围内。

建议临时返回：

```python
extra_fields["rose_topk_token_ids"] = np.ndarray((L, K), dtype=np.int32)
extra_fields["rose_topk_logprobs"] = np.ndarray((L, K), dtype=np.float32)
extra_fields["rose_topk_valid_mask"] = np.ndarray((L, K), dtype=np.bool_)
```

如果某个位置少于 K 项，使用 mask，而不是伪造 token 或概率。

### 6.3 生命周期

top-K 数据属于 rollout 控制面的临时数据：

```text
vLLM server -> RoseAgentLoop -> semantic scorer -> 删除
```

不应进入最终训练 batch，也不应长期放在 TransferQueue。最终 leaf 只保留：

- sampled response token IDs；
- sampled-token rollout logprobs；
- response mask；
- reward；
- compact tree metadata。

### 6.4 vllm-ascend smoke test

正式接入前必须在目标环境验证：

```python
SamplingParams(
    max_tokens=16,
    temperature=1.0,
    top_p=1.0,
    logprobs=20,
)
```

断言：

- 每个生成位置都有 logprobs。
- 每个位置返回 20 或 21 项。
- sampled token 总能在字典中找到。
- `rank` 的含义与上游 vLLM 一致。
- token ID 可以安全转换为 `int`。
- TP=1 和目标生产 TP 下行为一致。

## 7. Semantic entropy 实现

### 7.1 为什么首版放在 host CPU

这里的 CPU 是 Ascend 服务器本机 CPU。理由不是 CPU 理论算力高于 NPU，而是当前数据边界决定的：

1. vllm-ascend 已经把生成结果封装为 host 侧 `RequestOutput`。
2. AgentLoop 的树控制逻辑本来就在 host CPU。
3. 在 NPU 上重新计算需要把 top-K 再传回设备，或者侵入 sampler 内部。
4. TP>1 时 input embedding 可能按 vocabulary 分片，直接从 worker 取完整 embedding 会引入 HCCL gather。
5. 额外持有完整 embedding 会占用 rollout NPU HBM，并与 sleep/wakeup、weight update 交互。

首版应优先保证算法正确和 vllm-ascend 版本稳定性。后续根据 profiling 决定是否增加独立 NPU scorer。

### 7.2 离线准备 embedding

`prepare_embeddings.py` 建议执行：

1. 从与 rollout 完全相同的 Hugging Face checkpoint 加载 input embedding。
2. 读取 `model.get_input_embeddings().weight`。
3. 裁剪或校验到 tokenizer vocabulary size。
4. 按行做 L2 normalize。
5. 保存为连续 FP16 文件。
6. 保存 metadata 和校验摘要。

metadata 示例：

```json
{
  "model_path": "/models/Qwen3-8B",
  "vocab_size": 151936,
  "hidden_size": 4096,
  "dtype": "float16",
  "normalized": true,
  "tokenizer_hash": "...",
  "embedding_hash": "..."
}
```

启动训练时必须检查：

- tokenizer vocabulary size 一致；
- embedding 第一维覆盖所有有效 token ID；
- hidden size 与模型配置一致；
- hash 或模型路径匹配；
- 文件不是网络文件系统上的远程随机读热点。

推荐把文件放到每台服务器的本地 NVMe 或 `/dev/shm`。不要让所有节点从同一个 NFS mmap 文件进行随机读取。

### 7.3 CPU scorer

建议接口：

```python
class SemanticEntropyScorer:
    def score(
        self,
        topk_token_ids: np.ndarray,
        topk_logprobs: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Return semantic entropy with shape [response_length]."""
```

向量化实现：

```python
for start in range(0, response_length, chunk_size):
    stop = min(start + chunk_size, response_length)

    ids = topk_token_ids[start:stop]
    logp = topk_logprobs[start:stop]
    mask = valid_mask[start:stop]

    probabilities = np.where(mask, np.exp(logp), 0.0).astype(np.float32)
    entropy = -np.sum(probabilities * np.where(mask, logp, 0.0), axis=-1)

    embeddings = embedding_mmap[ids].astype(np.float32)
    weighted = np.einsum("lk,lkh->lh", probabilities, embeddings)
    pairwise = np.sum(weighted * weighted, axis=-1)

    if exclude_diagonal:
        pairwise -= np.sum(probabilities * probabilities, axis=-1)

    semantic_divergence = -pairwise
    result[start:stop] = entropy * semantic_divergence
```

注意：

- embedding 应在离线阶段完成归一化。
- 计算累加使用 FP32，存储可以使用 FP16。
- 不要构造 `[L,K,K]` cosine matrix。
- 不要写三层 Python 循环。
- 如果有无效 token ID，应 fail closed 或显式 mask；不能静默替换成 pad token。

### 7.4 Prefix 复用

从 parent trajectory 的位置 `k` 分叉时：

```text
new_response = parent_response[:k] + new_continuation
```

prefix 部分已经有 semantic entropy，必须直接复用：

```text
new_entropy[:k] = parent_entropy[:k]
new_entropy[k:] = score(new_continuation_topk)
```

否则树越深，CPU 会反复计算相同 prefix，开销可能呈二次增长。

### 7.5 CPU 性能估算

embedding 读取量近似：

```text
bytes ≈ generated_tokens × top_k × hidden_size × embedding_bytes
```

以 `K=20`、`H=4096`、FP16 为例：

| 新生成长度 | embedding 读取量/trajectory |
|---:|---:|
| 2K | 约 320 MB |
| 8K | 约 1.25 GB |
| 32K | 约 5 GB |

需要记录：

```text
rose/semantic_score_seconds
rose/semantic_score_tokens_per_second
rose/semantic_score_fraction_of_rollout
```

决策建议：

- `<5% rollout 时间`：保留 CPU。
- `5%~10%`：优化 mmap、chunk、NUMA、top-K 压缩和缓存。
- `>10%`：评估独立 NPU scorer。
- `>20%`：CPU 已明显限制 rollout 吞吐，应迁移或近似计算。

### 7.6 可选 NPU scorer

如果 CPU 成为瓶颈，增加独立 `RoseSemanticScorer` Ray Actor，而不是直接侵入 vllm-ascend：

```text
vllm-ascend -> compact top-K batch -> NPU scorer -> SE[L] -> AgentLoop
```

要求：

- 按整条 response 或多条 response batch 调用，禁止逐 token RPC。
- 每个节点只加载一份 normalized embedding。
- 尽量使用预留 NPU，避免与 rollout engine 抢同一设备。
- 记录 RPC、H2D、embedding gather 和 kernel 时间。
- scorer 故障必须使当前 prompt 失败并重采样，不能退化成随机 pivot 而不记录。

只有在拥有稳定 vllm-ascend 扩展能力时，才考虑把 semantic entropy 融入 sampler。该方案理论传输最少，但需要处理 TP vocabulary shard、HCCL、vLLM 调度和版本兼容，不适合作为第一版。

## 8. 树形 AgentLoop 设计

### 8.1 Prompt 级状态

建议每个 prompt 使用独立状态对象：

```python
@dataclass
class RosePromptState:
    uid: str
    prompt_ids: list[int]
    trajectories: list[RoseTrajectory]
    next_node_id: int
    rng: random.Random
```

trajectory：

```python
@dataclass
class RoseTrajectory:
    trajectory_id: int
    parent_trajectory_id: int | None
    branch_pos: int
    root_restart: bool
    response_ids: list[int]
    response_logprobs: list[float]
    semantic_entropy: np.ndarray
    path_node_ids: list[int]
    path_ends: list[int]
    reward: float | None
```

约定：

- `branch_pos` 是新 trajectory 中重新采样 token 的位置。
- 保留 prefix 范围为 `[0, branch_pos)`。
- `path_node_ids` 同时包含 root、沿途 pivot 和 terminal node。
- `path_ends` 是对应 node 在 response 中的位置，与 `path_node_ids` 等长。
- root node 的位置为 `0`，即 `path_ends[0] == 0`。
- terminal node 的位置为真实 response length。
- segment `j` 的半开区间为 `[path_ends[j-1], path_ends[j])`。
- terminal node 必须是 leaf 专属 node。

### 8.2 稳定 node ID

不要使用 `(prompt, token_position)` 作为 node identity，因为相同深度可能存在多个不同 prefix。建议 node ID 在创建 branch/leaf 时单调分配：

```text
tree_id + local_node_id
```

例如：

```text
prompt-42/node-0   root
prompt-42/node-1   first leaf
prompt-42/node-2   branch at trajectory 0, position 113
prompt-42/node-3   second leaf
```

每条 leaf 保存完整 `path_node_ids` 和对应 `path_ends`。这样 estimator 不需要根据 batch 顺序猜树结构，也不会把同深度的不同节点错误合并。

分叉可能发生在一条尚未包含显式 pivot node 的已有树边内部。例如第一条 leaf 最初只有 `root -> terminal`，第二条 trajectory 在位置 100 分叉后，这条边必须被拆成 `root -> pivot@100 -> old_terminal`。因此推荐在所有 `G` 条 trajectory 生成完成后执行一次 **tree finalization**：

1. rollout 阶段先记录不可变的 branch event：`(new_trajectory_id, parent_trajectory_id, branch_pos, root_restart)`。
2. finalization 从 root 开始重放 branch event。
3. 如果 pivot 落在已有 edge 内，则创建 pivot node 并拆分该 edge。
4. 将 pivot 插入所有真实经过该 edge 和该 prefix 的已有 leaf path。
5. 如果后续 trajectory 在同一个真实 node 再次分叉，则复用 node ID，不重复创建。
6. 为每条 leaf 创建独占 terminal node。
7. 最后统一导出等长的 `path_node_ids` 和 `path_ends`。

不要一边生成一边把早期 leaf 的 path 当成最终不可变结构，否则后创建的 pivot 可能不会出现在旧 leaf 中，进而导致 node membership 和 value 计算错误。

### 8.3 训练 rollout 伪代码

```python
async def generate_tree(prompt, config):
    state = RosePromptState(...)

    first = await generate_from_prefix(
        prompt_ids=prompt.ids,
        prefix_ids=[],
        logprobs=config.top_k,
    )
    add_root_trajectory(state, first)

    for trajectory_id in range(1, config.num_trajectories):
        if state.rng.random() < config.epsilon:
            parent_id = None
            branch_pos = 0
            prefix_ids = []
            prefix_logprobs = []
            prefix_entropy = []
            root_restart = True
        else:
            parent_id, branch_pos = find_global_max_entropy(state)
            parent = state.trajectories[parent_id]
            prefix_ids = parent.response_ids[:branch_pos]
            prefix_logprobs = parent.response_logprobs[:branch_pos]
            prefix_entropy = parent.semantic_entropy[:branch_pos]
            root_restart = False

        continuation = await generate_from_prefix(
            prompt_ids=prompt.ids + prefix_ids,
            prefix_ids=prefix_ids,
            max_tokens=config.response_length - len(prefix_ids),
            logprobs=config.top_k,
        )

        entropy = concat(
            prefix_entropy,
            scorer.score(continuation.topk_ids, continuation.topk_logprobs),
        )

        add_trajectory(
            state,
            parent_id=parent_id,
            branch_pos=branch_pos,
            root_restart=root_restart,
            response_ids=prefix_ids + continuation.token_ids,
            response_logprobs=prefix_logprobs + continuation.sampled_logprobs,
            semantic_entropy=entropy,
        )

    outputs = []
    for trajectory in state.trajectories:
        trajectory.reward = await score_leaf(prompt, trajectory)
        outputs.append(to_agent_loop_output(trajectory))
    return outputs
```

### 8.4 全局最大 pivot

`find_global_max_entropy()` 必须搜索所有已生成 trajectory 的所有有效位置，而不是只搜索最后一条 trajectory：

```python
best = max(
    (score, trajectory_id, position)
    for trajectory in state.trajectories
    for position, score in enumerate(trajectory.semantic_entropy)
    if is_valid_pivot(trajectory, position)
)
```

需要明确的 tie-break：

1. semantic entropy 更大者优先；
2. 相同 score 时 position 更靠前者优先，或使用固定 seed 随机；
3. 再相同时 trajectory ID 更小者优先。

tie-break 必须确定性，以便复现实验。

### 8.5 有效 pivot

至少排除：

- padding 位置；
- EOS 之后的位置；
- 无 top-K 数据的位置；
- NaN/Inf semantic entropy；
- 导致剩余 generation budget 小于 1 的位置；
- 配置要求下的特殊控制 token。

如果没有有效 pivot，应记录指标并从 root 生成，而不是崩溃或选 padding：

```text
rose/fallback_root_no_valid_pivot
```

### 8.6 Prefix cache

ROSE 反复使用相同 prompt 和 prefix，建议启用：

```yaml
actor_rollout_ref.rollout.enable_prefix_caching: true
```

但 prefix cache 只是性能优化，不能作为正确性依赖。每次 policy 权重更新后必须遵循现有 VeRL/vLLM 的 cache 生命周期和权重同步规则。

### 8.7 Validation

验证时不要使用训练树，因为它会改变 pass@k 的采样分布和不同 baseline 的比较口径。建议：

```text
validate=true -> 普通独立采样 n=val_kwargs.n
train         -> ROSE tree sampling G=n
```

论文评估使用独立的 `pass@8`。

## 9. AgentLoop 输出与 TransferQueue

### 9.1 Key 约定

建议每条 leaf 使用独立 session：

```text
{uid}_{trajectory_id}_0
```

例如：

```text
abc123_0_0
abc123_1_0
...
abc123_7_0
```

这样 V1 的 session key 规则保持成立，同时 estimator 不依赖 leaf 在 batch 中连续排列。

### 9.2 输出字段

每条 leaf 的 `AgentLoopOutput`：

```python
AgentLoopOutput(
    prompt_ids=prompt_ids,
    response_ids=trajectory.response_ids,
    response_mask=[1] * len(trajectory.response_ids),
    response_logprobs=trajectory.response_logprobs,
    reward_score=trajectory.reward,
    metrics=metrics,
    extra_fields={
        "rose_tree_metadata": serialize_tree_metadata(...),
    },
)
```

当前 `AgentLoopOutput.extra_fields` 会在聚合时展开为 non-tensor 字段，可以用它携带 tree metadata。metadata 应保持紧凑，建议使用 msgpack/JSON 字符串或结构稳定的小字典；不要写入 embedding、top-K logprobs 或整棵树的重复副本。

### 9.3 Metadata schema

推荐每条 leaf 保存：

```json
{
  "schema_version": 1,
  "tree_id": "abc123",
  "trajectory_id": 3,
  "parent_trajectory_id": 1,
  "branch_pos": 257,
  "root_restart": false,
  "path_node_ids": [0, 4, 9, 12],
  "path_ends": [0, 112, 257, 846]
}
```

约束：

- `len(path_ends) == len(path_node_ids)`。
- `path_ends[0] == 0`。
- `path_ends` 严格递增。
- `path_ends[-1] == response_mask.sum()`。
- `path_node_ids[0]` 是 root，`path_node_ids[-1]` 是当前 leaf 的 terminal node。
- terminal node 不能被其他非相同 leaf 误用。
- `tree_id` 必须与原始 prompt `uid` 对应。

## 10. Trainer 数据通路改造

### 10.1 通用额外字段配置

建议新增：

```yaml
algorithm:
  advantage_extra_fields: []
```

在 V1 `_compute_advantage()` 中：

```python
fields = [
    "uid",
    "response_mask",
    "rm_scores",
    "rollout_log_probs",
    "old_log_probs",
    "ref_log_prob",
    "values",
]
fields.extend(self.config.algorithm.get("advantage_extra_fields", []))
```

在 `to_padded_tensor()` 之前取出 non-tensor metadata：

```python
extra_non_tensor = {}
for field in advantage_extra_fields:
    value = data.pop(field)
    extra_non_tensor[field] = np.asarray(value.tolist(), dtype=object)

data = DataProto(
    batch=data.to_padded_tensor(),
    non_tensor_batch=extra_non_tensor,
)
```

具体实现需要遵循 TransferQueue 当前返回类型；关键原则是不能尝试把可变长度 path metadata pad 成训练 tensor。

### 10.2 Multi-trajectory wrapper

`verl/trainer/ppo/v1/utils.py:148` 当前对 GRPO 做特殊 final-session 处理，其他 estimator 会委托给通用 `compute_advantage()`。注册名 `rose_tree` 不应伪装成 `GRPO`，这样完整 leaf batch 会传给 ROSE estimator。

ROSE estimator 必须使用 `uid/tree_id` 分组，不依赖：

- batch 排序；
- 每个 prompt 恰好在相邻行；
- prompt-major 或 trajectory-major flatten；
- padding row 的位置。

## 11. Tree advantage estimator

### 11.1 输入

函数至少需要：

```text
token_level_rewards or rm_scores
response_mask
uid
rose_tree_metadata
config.algorithm.rose
```

reward 取每条 leaf 的有效 token reward 总和：

```python
leaf_reward = token_level_rewards[row].sum()
```

不要假设 reward 一定写在固定的最后一个 padding index；应通过 `response_mask` 确定有效范围。

### 11.2 分组与校验

对每个 `tree_id`：

1. 排除 `response_mask.sum()==0` 的自动 padding trajectory。
2. 检查 trajectory ID 唯一。
3. 检查所有 parent 都存在，或者为 root restart。
4. 检查 parent graph 无环。
5. 检查 `branch_pos` 合法。
6. 检查 path node 与 path end 数量匹配。
7. 检查相同 node ID 在不同 leaf 上有一致的 segment end/prefix 语义。
8. 如果启用 binary reward，检查 reward 接近 0 或 1。

生产训练建议 fail closed。树 metadata 错误会静默污染 credit assignment，比丢弃一个 prompt 更危险。

### 11.3 Node membership 和 value

建立：

```python
node_to_leaf_rows: dict[NodeId, list[int]]
```

对每条 leaf 的所有 `path_node_ids` 注册 membership，然后：

```python
node_value[node_id] = mean(leaf_reward[row] for row in member_rows)
```

root node 包含当前 prompt 的全部 leaf。leaf terminal node 只包含当前 leaf，因此：

```text
V(terminal_leaf) = reward(leaf)
```

### 11.4 Segment advantage 填充

对每条 leaf：

```python
for node_index in range(1, len(path_node_ids)):
    start = path_ends[node_index - 1]
    stop = path_ends[node_index]
    parent_node = path_node_ids[node_index - 1]
    child_node = path_node_ids[node_index]
    advantage = node_value[child_node] - node_value[parent_node]
    advantages[row, start:stop] = advantage
```

该 schema 固定把 root 和 terminal 都放入 `path_node_ids`；不要混用“位置节点”和“segment 节点”两套定义。

最后执行：

```python
advantages *= response_mask
returns = advantages.clone()
```

ROSE 是 outcome supervision、无 critic 的主要配置，因此一般不使用 GAE value tensor。

### 11.5 Length-aware calibration

对每棵树：

```python
correct_rows = [row for row in rows if leaf_reward[row] == 1]
```

如果正确 leaf 少于 2 条，不做校准。

选择最短正确回答时使用真实有效长度：

```python
length = int(response_mask[row].sum())
```

并使用固定 tie-break，例如 trajectory ID 最小者。

对其他正确 leaf，找到两条 `path_node_ids` 的最长公共前缀，得到真实最后公共 node；再由 `path_ends` 得到 `b_c`。不要仅用相同 token position 推断 LCA，也不要因为两个独立 root rollout 偶然生成相同文本 prefix 就合并树节点。

数值保护：

- 如果 `|o_c|-b_c <= 0`，视为 metadata 错误。
- ratio clamp 到 `[0,1]`。
- 使用 FP32 计算幂。
- calibration 后再次乘 `response_mask`。
- 记录校准前后 advantage 均值和绝对值。

### 11.6 是否做 advantage normalization

论文使用 Dr.GRPO 风格，默认不按 group std 归一化。建议：

```yaml
algorithm.norm_adv_by_std_in_grpo: false
algorithm.rose.normalize_advantage: false
```

不要复用 GRPO 的 group normalization，否则会改变 segment value difference 的尺度和长度校准效果。

## 12. Dynamic sampling / DAPO filtering

论文会删除同一 prompt 下 reward 完全相同的 group，因为这些树通常缺乏有效相对训练信号。VeRL V1 已有 group filtering：

```yaml
algorithm:
  filter_groups:
    enable: true
    metric: reward
    max_inflight_gen_batches: 1

trainer:
  v1:
    sampler:
      sync_refill_failed_groups: true
```

注意：

- V1 内置 replay buffer 使用 `max_inflight_gen_batches`。
- 旧配置中的 `max_num_gen_batches` 在该路径会被忽略。
- filtering 单位必须是完整 prompt group/完整树，不能只删除某些 leaf。
- reward filtering 读取 canonical `rm_scores`，因此每条 leaf 都必须已经完成 reward。
- validation 不应启用 group filtering。

## 13. PPO 与训练配置

推荐初始配置：

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: vanilla
    loss_agg_mode: seq-mean-token-sum-norm
    loss_scale_factor: 4096
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    entropy_coeff: 0.0
    ppo_epochs: 1
    shuffle: false

algorithm:
  use_kl_in_reward: false
  norm_adv_by_std_in_grpo: false
```

原因：

- `vanilla` PPO 最接近论文公式。
- `seq-mean-token-sum-norm` 对应 Dr.GRPO 风格的固定长度尺度，而不是按实际回答长度做 token mean。
- `loss_scale_factor` 建议与最大 response length 一致，论文为 4096。
- KL 通过 actor loss 加入，不要同时在 reward 和 loss 中重复加入。
- `ppo_epochs=1` 降低 policy staleness。
- `shuffle=false` 有利于调试和确定性；算法稳定后可以验证开启 shuffle 是否完全不改变 estimator 结果。

需要根据当前 VeRL PPO clip 配置把 `clip_ratio=0.2` 显式设置到对应 actor 字段。

## 14. Ascend NPU 专项设计

### 14.1 推荐软件栈

以当前仓库 Ascend 安装文档为准，vLLM 路径推荐组合为：

```text
CANN          9.1.0
Python        3.12
torch         2.10.0
torch_npu     2.10.0.post4
vLLM          0.23.0
vLLM-Ascend   0.23.0
triton-ascend 3.2.2
```

优先使用仓库发布的 Ascend 镜像或 `scripts/install_vllm_mcore_npu.sh`。NPU 示例使用 ambient Python 环境，不要直接套用 GPU 的 `uv --extra vllm` 环境。

### 14.2 环境变量

基础建议：

```bash
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
```

多机时还要正确配置：

```text
ASCEND_RT_VISIBLE_DEVICES
HCCL_SOCKET_IFNAME
GLOO_SOCKET_IFNAME
Ray head/worker 地址和 NPU resources
```

### 14.3 设备配置

显式设置：

```yaml
trainer.device: npu
```

即使当前 VeRL 可以通过 `torch_npu` 自动识别，显式配置仍然更利于脚本可读性和排错。

配置名 `gpu_memory_utilization` 和 `trainer.n_gpus_per_node` 是 VeRL/vLLM 的通用历史命名，在 NPU 上继续使用，不要自行改名。

### 14.4 FSDP2 与 Megatron

推荐：

| 模型规模/结构 | 训练后端 | rollout 建议 |
|---|---|---|
| 0.6B～8B dense | FSDP2 | vLLM-Ascend TP=1/2/4 |
| 30B/32B dense | FSDP2 或 Megatron | TP=4 起步，结合序列长度调优 |
| 30B+ MoE | Megatron + TP/EP | 沿用已有 Ascend Megatron 脚本 |
| 100B+ / 超长上下文 | Megatron | 单独做容量与通信规划 |

`tensor_model_parallel_size=1` 只适合小模型、短上下文的算法验证，不是 ROSE 要求。CPU semantic scorer 与 rollout TP 无关。

### 14.5 NPU 推荐开关

初始建议：

```yaml
actor_rollout_ref:
  actor:
    use_torch_compile: false
    ppo_epochs: 1
  ref:
    use_torch_compile: false
  rollout:
    name: vllm
    enable_prefix_caching: true
    enable_chunked_prefill: true
    free_cache_engine: true
    checkpoint_engine:
      update_weights_bucket_megabytes: 4096
```

`free_cache_engine`、`enforce_eager`、编译配置和 vllm-ascend 优化环境变量需要根据目标镜像版本实测，不应从 CUDA 示例无条件复制 cudagraph 参数。

### 14.6 不支持或首版禁用

首版禁用：

- vLLM prefill/decode disaggregation：当前代码明确标记 NPU 未验证。
- SGLang PD/disaggregation：同样未验证。
- fully async policy training。
- 一棵树跨 policy version 生成。
- multi-turn/tool calling。
- multimodal prompt。
- speculative decoding/MTP，除非先证明 top-K logprobs 与无 speculative decoding 完全一致。
- 在 vllm-ascend worker 中 all-gather 完整 vocabulary embedding。

## 15. NPU 启动脚本示例

以下是方向性模板；其中 `+...rose.*` 和 `algorithm.advantage_extra_fields` 需要前述代码/配置改造完成后才能使用。

```bash
#!/usr/bin/env bash
set -xeuo pipefail

export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=1
export HYDRA_FULL_ERROR=1

MODEL_PATH=${MODEL_PATH:-/models/Qwen3-4B-Base}
TRAIN_FILE=${TRAIN_FILE:-/data/math/train.parquet}
VAL_FILE=${VAL_FILE:-/data/math/test.parquet}
EMBEDDING_PATH=${EMBEDDING_PATH:-/dev/shm/qwen3_4b_embeddings.f16.npy}

python3 -m verl.trainer.main_ppo \
    trainer.device=npu \
    trainer.use_v1=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor=4096 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_prefix_caching=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ \
    +actor_rollout_ref.rollout.rose.enable=true \
    +actor_rollout_ref.rollout.rose.num_trajectories=8 \
    +actor_rollout_ref.rollout.rose.epsilon=0.5 \
    +actor_rollout_ref.rollout.rose.top_k=20 \
    +actor_rollout_ref.rollout.rose.semantic_device=cpu \
    +actor_rollout_ref.rollout.rose.semantic_chunk_size=256 \
    +actor_rollout_ref.rollout.rose.embedding_path="${EMBEDDING_PATH}" \
    algorithm.adv_estimator=rose_tree \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=false \
    +algorithm.advantage_extra_fields='["rose_tree_metadata"]' \
    +algorithm.rose.length_calibration_alpha=1.0 \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=reward \
    algorithm.filter_groups.max_inflight_gen_batches=1 \
    trainer.v1.sampler.sync_refill_failed_groups=true \
    trainer.logger='["console"]' \
    trainer.project_name=rose-verl-npu \
    trainer.experiment_name=qwen3-4b-rose
```

大模型应从仓库已有 Ascend FSDP2/Megatron 脚本继承并只叠加 ROSE 参数，不要以这个 4B 模板覆盖其 TP/PP/EP/CP 和 offload 配置。

## 16. 测试方案

### 16.1 Semantic entropy 单元测试

覆盖：

1. 两个正交 embedding 的手算结果。
2. 两个相同 embedding 时 semantic divergence 接近零。
3. 对角项排除与包含模式。
4. valid mask 中不足 K 个候选。
5. 极小 logprob 下无 NaN。
6. token ID 越界时报错。
7. chunked 与非 chunked 结果一致。
8. NumPy CPU 与 Torch reference 结果一致。
9. prefix entropy 复用后结果与完整重算一致。

### 16.2 Top-K 提取测试

使用 mocked `RequestOutput`：

1. `logprobs=False/True/0/20` 兼容。
2. sampled token 位于 top-K。
3. sampled token 不在 top-K，返回 K+1 项。
4. 字典顺序被打乱。
5. rank 缺失或异常时 fail closed。
6. abort/empty output 不崩溃。
7. top-K 临时字段不会进入最终训练 batch。

### 16.3 Tree rollout 测试

使用 deterministic fake LLM server：

1. 第 0 条从 root 生成。
2. `ε=0` 总是从最大 SE 分叉。
3. `ε=1` 总是 root restart。
4. pivot token 被重新生成，prefix 不包含 pivot token。
5. prefix response IDs、sampled logprobs 和 entropy 对齐。
6. 最大 response length 正确截断。
7. EOS 后不参与 pivot。
8. 固定 seed 下树结构可复现。
9. validation 使用独立采样。

### 16.4 Tree advantage 手工测试

构造最小树：

```text
root V=0.5
├── node-A V=1.0 -> correct leaf
└── node-B V=0.0 -> incorrect leaf
```

断言：

- root 到 A segment advantage 为 `+0.5`。
- root 到 B segment advantage 为 `-0.5`。
- terminal value 等于 leaf reward。
- mask 外 advantage 为 0。

继续覆盖：

1. 三层树。
2. 同深度不同 prefix 的两个 node 不合并。
3. root restart leaf 只共享 root。
4. leaf 顺序打乱后结果不变。
5. batch 中不同 prompt 交错排列后结果不变。
6. 自动 padding row 被忽略。
7. 缺失 parent、环和非法 path end 报错。
8. 只有一条正确 leaf 时不做 length calibration。
9. 多条正确 leaf 时真实 LCA 正确。
10. `α=0/1/2/3` 数值符合公式。

### 16.5 NPU smoke test

建议使用 Qwen3-0.6B 或更小可用模型：

1. 单 NPU/最小 TP 验证 `logprobs=20`。
2. 生成一个 prompt、`G=2`，打印树 metadata。
3. 验证 reward 和 advantage shape。
4. 执行一个 actor update。
5. 验证 checkpoint 保存/恢复。
6. 再切换到生产 TP，重复 top-K 和一轮训练。

### 16.6 小规模端到端

推荐顺序：

```text
Qwen3-0.6B, G=2, response=256, 2 steps
Qwen3-0.6B, G=4, response=1024, 10 steps
目标模型, G=4, response=2048, 10 steps
目标模型, G=8, 正式 response length
```

每一级都比较：

- 普通 independent rollout 是否仍正常。
- ROSE reward distribution。
- tree depth/width。
- semantic scoring 开销。
- NPU 利用率。
- HCCL 稳定性。
- loss、KL、clip fraction 是否异常。

## 17. 监控指标

### 17.1 Rollout

```text
rose/tree_count
rose/leaves_per_tree
rose/root_restart_ratio
rose/tree_max_depth
rose/tree_mean_depth
rose/branch_position_mean
rose/branch_position_p50
rose/branch_position_p95
rose/semantic_entropy_mean
rose/semantic_entropy_max
rose/no_valid_pivot_count
rose/prefix_reuse_tokens
rose/new_generated_tokens
```

### 17.2 Reward 和 filtering

```text
rose/reward_mean
rose/all_correct_group_ratio
rose/all_incorrect_group_ratio
rose/mixed_reward_group_ratio
rose/filtered_group_count
rose/refilled_group_count
```

### 17.3 Advantage

```text
rose/node_value_mean
rose/segment_advantage_mean
rose/segment_advantage_abs_mean
rose/segment_advantage_positive_ratio
rose/length_calibration_delta
rose/correct_response_length_mean
rose/shortest_correct_length_mean
```

### 17.4 性能

```text
timing/rose_tree_generation
timing/rose_topk_pack
timing/rose_semantic_score
timing/rose_reward
timing/rose_advantage
rose/semantic_tokens_per_second
rose/topk_payload_bytes
```

必须把 semantic score 时间和 vLLM generation 时间分开，否则无法判断 CPU 是否成为瓶颈。

## 18. 与作者公开实现相比的有意改进

适配时建议保留算法，但修正以下工程问题：

### 18.1 Node identity

作者实现主要用 `(prompt, token_position)` 标识节点，可能把同一深度、不同 prefix 的节点合并。当前方案使用真实 node ID 和 path，避免错误 value aggregation。

### 18.2 真实 LCA

作者实现的长度校准主要基于共同 position。当前方案通过 node path 的最长公共前缀寻找真实 LCA，root restart 和多级分叉都能正确处理。

### 18.3 Batch 顺序无关

作者实现依赖固定的 prompt-major/trajectory-major 排列。当前 estimator 按 `tree_id` 分组，允许 replay buffer、padding 和 batch balance 改变行顺序。

### 18.4 V1 async 架构

不复制旧同步 `vllm_rollout_spmd.py`，而是使用当前 AgentLoop、async server、TransferQueue 和 V1 trainer 的正式扩展点。

### 18.5 NPU embedding 路径

不从每个 vllm-ascend TP worker all-gather 完整 embedding，而是使用 host mmap 或独立 NPU scorer，降低对 vllm-ascend 内部 API 的依赖。

这些改进应通过 `compatibility_mode` 或清晰实验记录区分“论文公式”“作者公开代码”和“更稳健的工程实现”。

## 19. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| vllm-ascend top-K 返回格式不一致 | 无法计算 SE | 在目标版本做 smoke test，按 rank 解析 |
| Python dict/Ray 序列化过重 | rollout 吞吐下降 | server 立即压缩成 dense ndarray，及时删除 |
| CPU embedding 随机读成为瓶颈 | NPU 空转 | 本地 mmap、chunk、缓存、NUMA；必要时 NPU scorer |
| TP embedding 分片 | 难以设备内打分 | 首版不进入 vLLM worker |
| 树内跨 policy version | PPO ratio 和树 value 不一致 | sync trainer，一棵树原子生成 |
| metadata 错误 | 静默污染 advantage | 严格 schema 和 fail-closed 校验 |
| padding trajectory 混入节点 value | value 偏差 | `response_mask.sum()==0` 必须跳过 |
| root restart 与普通 branch 混淆 | LCA 错误 | 显式 `root_restart`，只共享 root node |
| reward 非 binary | 偏离论文 | 默认校验 0/1；若扩展必须明确实验设定 |
| prefix cache 跨权重更新 | 使用陈旧 KV | 遵循 vLLM 权重同步后的 cache 清理机制 |
| fully async policy staleness | 同树 leaf 不同 policy | 第一版禁止 fully async |
| embedding 与 tokenizer 不匹配 | OOV/错误 cosine | 启动时校验 vocab、hash 和维度 |

## 20. 分阶段实施计划

### 阶段 A：纯算法单元测试

1. 实现 `semantic_entropy.py`。
2. 实现 `tree_advantage.py`。
3. 使用手工 tensor/树完成全部单元测试。
4. 不依赖 vLLM、不依赖 NPU。

完成标准：公式、mask、LCA、padding、shuffle invariance 全部通过。

### 阶段 B：vLLM top-K 通路

1. 修复整数 `logprobs`。
2. 实现 top-K dense packing。
3. mocked unit test。
4. GPU/CPU 可用环境 smoke test。
5. Ascend vllm-ascend smoke test。

完成标准：目标 TP 下每个位置稳定获得 top-20，并保留 sampled logprob。

### 阶段 C：单 prompt 树生成

1. 实现 `RoseAgentLoopWorkerTQ`。
2. `G=2/4` 生成树。
3. 验证 prefix、pivot、metadata、reward。
4. 验证 validation independent mode。

完成标准：固定 seed 下树结构确定，metadata 通过所有校验。

### 阶段 D：Trainer 集成

1. 增加 `advantage_extra_fields`。
2. 注册 `rose_tree` estimator。
3. 将 advantages/returns 写回 TransferQueue。
4. 跑一个 actor update。

完成标准：端到端一步训练，无 shape/mask/key 错误。

### 阶段 E：Dynamic sampling

1. 启用 reward group filtering。
2. 验证完整树作为过滤单位。
3. 验证 refill 与失败恢复。

完成标准：mixed-reward group 被保留，统一 reward group 被正确替换。

### 阶段 F：Ascend 性能调优

1. 目标模型和目标长度 profiling。
2. 调整 TP、prefix cache、chunk size、CPU NUMA。
3. 测量 CPU semantic score 比例。
4. 必要时实现独立 NPU scorer。

完成标准：NPU 利用率、吞吐和稳定性达到可接受水平，且算法指标无变化。

## 21. 验收标准

功能验收：

- `G` 条 leaf 构成合法树。
- `ε=0/1` 行为符合定义。
- top-20 semantic entropy 与 reference 实现数值一致。
- node value 与手算一致。
- segment advantage 与手算一致。
- length calibration 使用真实 LCA。
- batch shuffle、padding、balance 不影响结果。
- validation 仍为 independent pass@k。

训练验收：

- 在 Ascend 上连续运行至少 100 step 无 deadlock/OOM。
- policy version 在一棵树内一致。
- KL、clip fraction、gradient norm 无异常尖峰。
- checkpoint 可以恢复。
- dynamic filtering/refill 不丢失或混合 prompt group。

性能验收：

- top-K 不进入长期 replay buffer。
- semantic scorer 不产生每 worker 一份完整匿名内存副本。
- semantic score 时间有独立指标。
- 如果 CPU scoring 超过 rollout 的 10%，必须完成优化评估或 NPU scorer 方案评审。

实验验收：

- 至少包含 GRPO、Dr.GRPO 风格 independent rollout 和 ROSE 对照。
- 至少报告 reward/accuracy、pass@k、response length、tree depth 和训练吞吐。
- 对 `ε`、`α`、generation entropy/semantic divergence/semantic entropy 做消融。

## 22. 最小可行版本建议

第一版建议严格限制范围：

```text
trainer             V1 sync
task                单轮纯文本数学题
reward              binary rule-based reward
rollout             vLLM + vllm-ascend
semantic scorer     host CPU + local mmap
G                   4（验证）-> 8（正式）
top-K               20
validation          independent rollout
training backend    小模型 FSDP2，大模型沿用现有 Megatron 脚本
PD/disaggregation   disabled
multi-turn/tools    disabled
multimodal          disabled
spec decoding       disabled
```

这个范围已经覆盖 ROSE 的完整算法贡献，同时把 Ascend 风险集中在两个可独立验证的接口：

1. vllm-ascend top-20 logprobs。
2. V1 AgentLoop 到 tree advantage 的 metadata 通路。

完成最小版本并获得 profiling 数据后，再决定是否加入 NPU semantic scorer、multi-turn 或 fully async 支持。
