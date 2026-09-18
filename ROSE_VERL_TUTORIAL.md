# ROSE 如何适配到 verl：从框架入门到代码实现

本文是一份“边学 verl、边读 ROSE 适配”的教程。目标不是只告诉你如何启动脚本，而是解释：

1. verl 的一次强化学习训练 step 是怎样流动的；
2. ROSE 哪些算法步骤必须放在 rollout 阶段，哪些步骤属于 advantage 阶段；
3. 当前实现为什么选择 V1 AgentLoop、TransferQueue 和自定义 advantage estimator；
4. 每个改动解决了什么问题，数据在改动前后怎样传递；
5. 在 Ascend NPU 服务器上如何验证、调试和继续扩展。

本文对应当前工作区的实现。主要代码位于 `verl/experimental/rose/`，训练入口为根目录的
`run_rose.sh`。已有的设计决策和逐文件改动日志可作为补充阅读：

- `ROSE_VERL_ASCEND_ADAPTATION.md`：算法、NPU 部署和验收方案；
- `ROSE_IMPLEMENTATION_DECISIONS.md`：为什么作出这些工程取舍；
- `ROSE_CODE_CHANGES.md`：改动清单和测试记录。

> 重要边界：当前实现支持 **V1 + vLLM/vllm-ascend + 单轮纯文本 + binary reward**。
> 不要把它理解成已经支持所有 verl agent、tool、多模态或 fully async 场景。
> 目标 Ascend 环境仍需单独做真实 vLLM-Ascend 和 HCCL 验证。

---

## 1. 先建立总图：verl 在训练什么

verl 可以看成由四类组件组成的 RLHF/RLVR 系统：

```text
数据集
  │ prompt、ground truth、data_source
  ▼
Trainer / Dataloader
  │ 把 prompt 发给 rollout
  ▼
Rollout（vLLM/vllm-ascend）
  │ 生成 response、rollout logprob、额外 metadata
  ▼
Reward Manager
  │ 将 response 转成 reward
  ▼
Advantage Estimator
  │ reward + response_mask (+ 自定义 metadata)
  ▼
Actor / Reference / Critic Worker
  │ 在设备上计算 logprob、KL、loss、反向传播
  ▼
更新后的 policy
  │ 权重同步回 rollout
  └─────────────── 下一轮 rollout
```

普通 GRPO 大致只要求同一个 prompt 采样多条独立回答，然后根据组内 reward 计算 advantage。
ROSE 多做了一件关键的事：**同一个 prompt 的回答不是独立采样，而是共享 prefix、在语义熵高的
位置分叉，形成一棵树**。因此 ROSE 不能只替换 `adv_estimator`，还必须替换 rollout 的
调度和数据写入逻辑。

### 1.1 当前采用的 verl V1 路径

当前脚本设置了：

```bash
trainer.use_v1=true
trainer.device=npu
actor_rollout_ref.rollout.name=vllm
```

V1 训练器的入口链路是：

```text
verl/trainer/main_ppo.py
  └─ TaskRunnerV1.run()
      └─ trainer_cls(config).fit(agent_loop_manager)
          └─ PPOTrainer.step()
              ├─ prepare_step()                  # 提交 prompt 到 rollout
              ├─ replay_buffer.sample()           # 等待并取回完成的 group
              ├─ _compute_reward_colocate()       # 需要时计算 reward
              ├─ _compute_old_log_prob()          # policy 的 old logprob
              ├─ _compute_ref_log_prob()          # reference logprob（可选）
              ├─ _compute_values()                # critic（可选）
              ├─ _compute_advantage()             # ROSE 在这里进入
              └─ _update_actor()                  # NPU 上 PPO 更新
```

`TaskRunnerV1.init_agent_loop_manager()` 会读取：

```yaml
actor_rollout_ref.rollout.agent.agent_loop_manager_class
```

如果该字段为空，就使用默认的 `AgentLoopManagerTQ`；ROSE 脚本将它设为：

```text
verl.experimental.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ
```

这就是 ROSE 接入 V1 的主扩展点。

---

## 2. 一次 V1 训练 step：把每个对象弄明白

### 2.1 配置：Hydra + dataclass

verl 的配置来自 `verl/trainer/config/ppo_trainer.yaml`，启动命令可以通过 Hydra override
覆盖，例如：

```bash
algorithm.adv_estimator=rose_tree
data.train_batch_size=32
actor_rollout_ref.rollout.n=8
```

配置会被转换到 dataclass schema。ROSE 新增的两个 schema 位于：

- `verl/workers/config/rollout.py`：`RoseRolloutConfig`；
- `verl/trainer/config/algorithm.py`：`RoseAlgorithmConfig` 和
  `AlgoConfig.advantage_extra_fields`。

这样做的目的不是形式主义，而是让非法配置尽早失败。例如：

```python
if self.top_k < 2:
    raise ValueError(...)
if self.semantic_device != "cpu":
    raise ValueError(...)
if self.enable and not self.embedding_path:
    raise ValueError(...)
```

`rollout.rose.enable=true` 时，embedding 路径、top-K、epsilon 等配置在 worker 初始化阶段
就会检查，而不是等到生成半小时后才发现配置错误。

### 2.2 Ray：进程/节点编排，不是算法本身

verl 使用 Ray 管理多个角色：

- Trainer/TaskRunner：协调训练步骤；
- rollout worker：持有 vLLM client 或 rollout engine；
- actor worker：在 NPU 上计算 PPO loss 和梯度；
- reference/critic/reward worker：按配置启用。

ROSE 没有另起一套 Ray 训练框架，而是复用 V1 的 manager/worker：

```text
RoseAgentLoopManagerTQ
  └─ 多个 RoseAgentLoopWorkerTQ（Ray actors）
      └─ 每个 worker 中同时处理多个 prompt
          └─ 每个 prompt 内的 G 条 trajectory 按顺序生成
```

这里有一个非常重要的并行粒度：

- 不同 prompt 可以并行；
- 同一个 prompt 的树内 trajectory 不能像普通 `rollout.n` 那样全部并发。

如果把同一个 prompt 的分支全部并发，第二条 trajectory 还没看到第一条的 semantic entropy，
就无法决定正确的 pivot，算法会退化成独立采样。

### 2.3 TransferQueue：V1 的共享 rollout 缓冲区

V1 不直接把所有生成结果作为一个巨大的 Python 返回值交给 Trainer，而是将数据写入
TransferQueue（简称 TQ）。每个 prompt group 有一个 uid，每条 trajectory 使用形如：

```text
{uid}_{trajectory_id}_0
```

的 key。prompt 的状态通过 tag 管理：

```text
pending → running → finished
                    └→ failure
```

trajectory 的字段包括：

```text
prompts
responses
response_mask
rollout_log_probs
rm_scores
extra_fields
input_ids
position_ids
loss_mask
```

ROSE 最终将 `rose_tree_metadata` 放进 `extra_fields`，但不会把每个 token 的 top-K
候选长期保存到 TQ。top-K 只在 rollout worker 中用于选择 pivot，计算完 semantic entropy 后
立即删除。

### 2.4 ReplayBuffer：什么时候一个 batch 才算准备好

`verl/trainer/ppo/v1/replay_buffer.py` 负责观察 TQ 状态并取出训练 batch。对于 ROSE，
一个 prompt group 必须等到 G 条 leaf 都处理完，才可以标记为 `finished`。

这也是为什么 `RoseAgentLoopWorkerTQ` 不能让每个 leaf 单独把 uid 标记为 finished：
如果第一条 leaf 完成就标记，ReplayBuffer 可能提前取走不完整的树。

`algorithm.filter_groups.enable=true` 时，ReplayBuffer 还可能执行 DAPO 风格的 group 过滤。
如果同一 group 的所有 trajectory reward 完全相同，该 group 可能被丢弃并补采新的 prompt。
这会让日志中的 `finished` 超过 `data.train_batch_size`，并显著增加 ROSE 训练时间。

学习和排错时建议先关闭：

```bash
algorithm.filter_groups.enable=false
```

### 2.5 `DataProto`、TensorDict 和 non-tensor metadata

verl 的训练数据并不是一个简单的 `dict`。在传统 PPO 路径中，`DataProto` 通常包含：

```text
DataProto
├── batch: TensorDict              # token、mask、logprob 等 tensor
├── non_tensor_batch: dict         # uid、字符串、变长 Python 对象
└── meta_info: dict                # batch 级控制信息
```

V1 + TransferQueue 路径会在 TQ 中使用 nested/jagged tensor 保存不同长度的 prompt/response，
在真正送给 actor 前才通过 `to_padded_tensor()` 对齐成训练 batch。这个设计避免 rollout 阶段
为了 padding 大量短回答而浪费存储。

ROSE metadata 是树结构，天然属于 non-tensor 数据；它不能被强行堆成一个规则 tensor。当前
适配因此遵循：

```text
TQ extra_fields（每条 leaf 一个 mapping）
  → object ndarray
  → DataProto.non_tensor_batch["rose_tree_metadata"]
  → estimator kwargs
```

这也是为什么 `algorithm.advantage_extra_fields` 是一个字段名列表，而不是把 metadata 的
内部结构写死在 `DataProto` 或 trainer 中。

### 2.6 policy version：为什么树必须原子地使用同一版权重

V1 的训练循环在进入第一个 step 前会增加 `global_steps`，而 rollout server 的初始权重版本
通常从 0 开始。因此当前实现中，训练 step `N` 的 rollout 预期版本是 `N-1`。ROSE worker
会检查：

```text
每条 leaf: min_global_steps == max_global_steps
一棵树: 所有 leaf 的版本相同
整棵树: 版本 == 当前 prompt 预期版本
```

这不是多余的防御式代码。假设同一棵树的第一条 leaf 用旧 policy、第二条 leaf 用更新后的
policy，那么 semantic pivot、leaf reward 和 node value 就不再来自同一个采样分布，segment
advantage 的含义会被破坏。

排查 policy version 错误时，应该先核对：

1. 当前 trainer 的 `global_steps` 何时递增；
2. rollout server 初始化版本是多少；
3. prompt tag 中的 `global_steps`、`min_global_steps`、`max_global_steps` 是否一致；
4. 是否错误地把校验删除，而不是修正预期版本的 off-by-one。

---

## 3. ROSE 算法拆成两个必须分别实现的部分

ROSE 的工程适配可以先画成两个边界：

```text
                 rollout / data collection              training / credit assignment
                 ------------------------              ----------------------------
prompt ────────► 生成树 + top-K + semantic entropy ───► leaf reward
                         │                                  │
                         └── tree metadata ─────────────────┘
                                                            │
                                                            ▼
                                                   tree advantage + PPO update
```

### 3.1 Rollout 阶段做什么

对每个 prompt：

1. 从 root 生成第一条 trajectory；
2. 从 vLLM 取得每个 response token 的 sampled logprob 和 top-K logprob；
3. 用 input embedding 计算每个位置的 semantic entropy；
4. 按 epsilon 决定 root restart 或从已有 trajectory 分叉；
5. 重复直到获得 G 条 leaf；
6. 给每条 leaf 计算 reward；
7. 把 leaf 和树 metadata 写进 TQ。

### 3.2 Advantage 阶段做什么

对同一个树的所有 leaf：

1. 按 node 收集经过该 node 的 leaf；
2. 计算 node value，即经过该 node 的 leaf reward 均值；
3. 用相邻 node value 之差填充 segment advantage；
4. 对正确回答执行长度校准；
5. 将得到的 token-level advantage 交给现有 actor PPO loss。

因此：

- 只实现 `rose_tree` estimator，不会产生 ROSE 树；
- 只实现树 rollout，不实现 estimator，actor 仍会使用普通 GRPO/GAE 逻辑；
- 两部分通过 `rose_tree_metadata` 连接。

---

## 4. Rollout 端：从 vLLM top-K 到一棵树

### 4.1 为什么必须改 top-K logprob 接口

默认 rollout 只需要 sampled token 的 logprob，而 ROSE 需要每个位置的 top-K 分布近似：

```text
token_ids[L, K]
logprobs[L, K]
valid_mask[L, K]
```

改动前有一个容易忽略的问题：Python 中整数 `20` 是 truthy。如果服务端写成：

```python
sampling_params["logprobs"] = 0 if requested_logprobs else None
```

那么用户设置 `logprobs=20` 后反而会变成 `0`，只能得到 sampled token，拿不到 top-20。

当前实现将这部分逻辑提取到 `verl/workers/rollout/logprobs.py`：

- `normalize_requested_logprobs()`：区分 `False`、`True`、整数 K；
- `pack_topk_logprobs()`：将后端返回的 mapping 压成固定形状数组；
- 优先使用 vLLM `Logprob.rank`，rank 缺失时才按 logprob 排序补齐；
- token id 为 `int32`，logprob 为 `float32`，有效位置由 bool mask 表示。

`verl/workers/rollout/vllm_rollout/vllm_async_server.py` 只负责提取并暂存：

```text
rose_topk_token_ids
rose_topk_logprobs
rose_topk_valid_mask
```

### 4.2 top-K 为什么不能进入训练 batch

假设回答长度为 L、top-K 为 20、embedding hidden size 为 H，保存 top-K 会随 L×K 增长，
而树中每条 trajectory 都可能携带一份。把它写入 TQ 会增加：

- Ray object store 内存；
- TQ 序列化成本；
- checkpoint/replay 体积；
- 多机网络传输。

因此生命周期被限制为：

```text
vLLM server → RoseAgentLoop → CPU scorer → 删除
```

最终训练字段只保留 sampled response、sampled logprob、reward 和 compact tree metadata。

### 4.3 `RoseAgentLoop`：保证 prefix token 不被改写

文件：`verl/experimental/rose/rose_agent_loop.py`。

`RoseAgentLoop.generate_from_prefix()` 做四件事：

1. 将 `prompt_ids + prefix_ids` 直接发给 server；
2. 只请求剩余 response token；
3. 将 parent prefix 的 token/logprob 与新 continuation 拼接；
4. 从 `TokenOutput.extra_fields` 取走 top-K payload。

关键点是 **不 decode 再 tokenize**：

```text
new_response_ids = parent_response_ids[:branch_pos] + continuation_ids
new_logprobs     = parent_logprobs[:branch_pos] + continuation_logprobs
```

这样可以保证已经采样的 prefix token、token 边界和 logprob 完全不变。对聊天模板、多轮
tool observation 或多模态输入，简单拼接 token 可能不再成立，所以当前实现明确拒绝这些模式。

### 4.4 `RoseAgentLoopWorkerTQBase`：prompt 间并发、树内顺序

文件：`verl/experimental/rose/rose_agent_loop_tq.py`。

worker 初始化时检查：

```text
rose.enable == true
rollout.name == vllm
multi_turn.enable == false
semantic_device == cpu
embedding_path 非空
```

对训练 prompt，核心循环可概括为：

```python
trajectories = []
for trajectory_id in range(G):
    selected_branch = choose_branch(trajectories, rng, epsilon)

    if selected_branch is None:
        prefix = []                  # root restart
    else:
        parent_id, branch_pos = selected_branch
        prefix = trajectories[parent_id].response_ids[:branch_pos]

    output, topk_ids, topk_logprobs, mask = generate_from_prefix(...)
    entropy = scorer.score(topk_ids, topk_logprobs, mask)
    entropy = prefix_entropy + entropy
    trajectories.append(...)
```

分叉位置的选择逻辑是：

- 以 epsilon 概率 root restart；
- 否则在当前全部 trajectory 的有限 semantic entropy 中选最大值；
- 没有有效 entropy 时回退 root；
- prefix entropy 直接复用，不重复计算。

这段代码刻意没有使用默认 worker 的“同一 prompt 的 n 个 session 并发”行为，因为默认行为
不满足树的顺序依赖。

### 4.5 树 finalization：为什么不直接用 token position 当 node id

树节点不仅是一个 token position。后生成的 pivot 可能落在已有 edge 的中间：

```text
root ───────────── terminal
          ▲
       新 pivot
```

如果只按 position 记录，旧 leaf 的路径需要被拆边；不同 prefix 也可能在同一深度有不同节点。

`verl/experimental/rose/tree.py` 先记录不可变的 `BranchRecord`，全部 trajectory 完成后调用：

```python
finalize_tree(tree_id, records)
```

输出每条 leaf 的 compact metadata：

```python
{
    "schema_version": 1,
    "tree_id": uid,
    "trajectory_id": 3,
    "parent_trajectory_id": 1,
    "branch_pos": 128,
    "root_restart": False,
    "path_node_ids": [1, 2, 5],
    "path_ends": [0, 128, 512],
}
```

`path_ends` 表示每个 segment 的结束位置。它让 estimator 可以直接把某个 node-to-node
advantage 写入 `[start:stop]`，不需要猜测 batch 顺序。

### 4.6 为什么每个 leaf 必须独立 reward

默认 `AgentLoopWorkerTQ._agent_loop_postprocess()` 支持一个 session 返回多个阶段，并可能把
最终 reward 复制给前序 output。ROSE 的每个 leaf 都是一条独立回答，不能复用这种“同一 session
多个 output”的语义。

当前实现的顺序是：

```text
整棵树完成
  → finalize_tree()
  → 每个 leaf 单独 _agent_loop_postprocess()
  → 所有 leaf 完成后 uid=finished
```

这样 reward manager 会对每条 leaf 单独评分，tree advantage 才有真实的 leaf reward。

---

## 5. Semantic entropy：为什么在 NPU 服务器的 CPU 上

### 5.1 计算内容

每个 response position 使用 top-K 近似计算：

```text
generation entropy:  H = -Σ p_i log p_i
semantic divergence: SD = -Σ_{i≠j} p_i p_j cos(e_i, e_j)
semantic entropy:    SE = H × SD
```

embedding `e_i` 来自模型 input embedding，并在离线步骤中逐行 L2 normalize。

对于归一化 embedding，可以使用恒等式避免构造 `[L,K,K]` 的 cosine 矩阵：

```text
y = Σ_i p_i e_i
Σ_{i,j} p_i p_j cos(e_i,e_j) = ||y||²
Σ_i p_i²                  = 对角项
```

代码位于 `verl/experimental/rose/semantic_entropy.py`，使用：

- FP16 mmap 存储 embedding；
- FP32 概率和向量累加；
- chunked NumPy 计算；
- valid mask 处理不足 K 的位置；
- vocab 范围、文件大小、NaN/Inf 检查。

### 5.2 这里的 CPU 不是“把训练放到 CPU”

`semantic_device=cpu` 只表示这段控制面数学在 **Ascend 服务器 host CPU** 上运行：

```text
NPU：模型生成、sampled logprob、actor/reference forward、backward、optimizer
CPU：读取 top-K、semantic entropy、树 pivot、metadata、trainer 控制逻辑
```

原因是当前 vLLM-Ascend 已经把生成结果封装到 host 侧；把 scorer 放进 NPU sampler 会涉及
embedding 复制、TP vocabulary shard、HCCL gather 和设备占用。首版先保证接口稳定和算法正确。

这并不代表 CPU scorer 没有代价。长回答或大模型可能使 embedding 随机读取成为瓶颈，必须记录：

```text
rose_semantic_score_seconds
rose_generation_seconds
rose_semantic_score_fraction_of_generation
```

如果 profiler 显示 scorer 占 rollout 时间超过约 10%，再考虑独立的 NPU scorer，而不是直接
把 NumPy 代码改成 `.to("npu")`。

### 5.3 离线生成 embedding

训练前执行：

```bash
python3 -m verl.experimental.rose.prepare_embeddings \
  --model-path /models/Qwen3-4B-Base \
  --output-path /local_nvme/rose/qwen3_4b_input_embeddings.f16
```

会输出：

```text
qwen3_4b_input_embeddings.f16
qwen3_4b_input_embeddings.json
```

metadata 记录 vocab size、hidden size、dtype、normalized 标记和 hash。训练脚本
`run_rose.sh` 会自动检查并在缺失或模型路径不匹配时重新准备。生产多机时更推荐先在每台机器
本地 NVMe 准备一份，避免所有 worker 从同一个 NFS 随机 mmap。

---

## 6. Advantage 端：ROSE 如何进入 verl 的 PPO

### 6.1 verl 原有的 advantage 接口

verl 的 `compute_advantage()` 通常接收：

```text
token_level_rewards
response_mask
uid/index
config
```

然后根据 `algorithm.adv_estimator` 从 registry 找函数。ROSE 注册：

```python
@register_adv_est("rose_tree")
def compute_rose_tree_advantage(...):
    ...
```

实现文件：`verl/experimental/rose/tree_advantage.py`。

### 6.2 为什么 trainer 需要通用 metadata 通道

树结构不是普通 tensor，无法只靠 `rm_scores` 重建。ROSE estimator 需要每行对应的：

```text
tree_id
trajectory_id
parent_trajectory_id
branch_pos
path_node_ids
path_ends
root_restart
```

如果把 `rose_tree_metadata` 硬编码到 trainer，未来每加一个自定义 estimator 都要改 trainer。
当前实现增加通用配置：

```yaml
algorithm:
  advantage_extra_fields:
    - rose_tree_metadata
```

对应数据流：

```text
TQ extra_fields
  → trainer_base._extract_advantage_extra_fields()
  → DataProto.non_tensor_batch
  → ray_trainer.compute_advantage()
  → estimator kwargs
```

字段不存在、row 不是 mapping 或字段为空时 fail closed，而不是把 `None` 传给 estimator。

### 6.3 Node value

对一个 `tree_id`，每个 node 收集经过它的 leaf：

```text
V(node) = mean(reward of leaves through node)
```

binary reward 下，`V(node)` 可以解释成经验成功率。例如某 node 后面 4 条 leaf 中 3 条正确，
那么 `V(node)=0.75`。

### 6.4 Segment advantage

如果一条 leaf 的 path 是：

```text
root (0) → pivot (128) → terminal (512)
```

那么：

```text
response[0:128]    = V(pivot) - V(root)
response[128:512]  = V(terminal) - V(pivot)
```

这使得不同推理片段收到不同 credit，而不是把最终 reward 均匀复制到整条回答。

### 6.5 Length calibration

对同一树中多条正确 leaf，选择最短正确回答作为参照，并根据真实 LCA（最后公共节点）对
较长正确回答的后续 segment 做长度校准。当前 `tree_advantage.py` 会使用 finalization 后的
node path，而不是简单比较两个字符串的公共前缀，避免在 token edge 被拆分后算错校准起点。

`algorithm.rose.length_calibration_alpha` 控制强度，建议从：

```yaml
algorithm:
  rose:
    length_calibration_alpha: 1.0
```

开始。当前默认还会检查 reward 是否接近 0/1；如果你的 reward 不是 binary，需要明确关闭或
重新定义 estimator 语义，而不是只绕过异常。

### 6.6 Advantage 算完后谁负责训练

ROSE estimator 只产生：

```text
advantages
returns
```

随后仍由 verl 原有 actor worker 执行：

```text
old_log_prob
ref_log_prob
KL
PPO clipped policy loss
backward
optimizer.step
```

因此 ROSE 不是另写一套 PPO。它改变的是采样分布和 token-level credit assignment，设备上的
前向/反向基础设施继续复用 verl。

---

## 7. 逐文件学习：当前到底改了什么

### 7.1 `verl/experimental/rose/`

| 文件 | 责任 | 为什么独立出来 |
|---|---|---|
| `semantic_entropy.py` | top-K + embedding → semantic entropy | 算法数学与后端解耦，可 CPU 单测 |
| `prepare_embeddings.py` | checkpoint → normalized mmap embedding | 避免每个 rollout worker 重复加载完整模型 |
| `tree.py` | branch records → stable node/path metadata | 统一处理拆边、terminal、路径 |
| `rose_agent_loop.py` | token-in/token-out、prefix reuse、top-K 生命周期 | 不污染通用 single-turn agent |
| `rose_agent_loop_tq.py` | prompt 级并行、树内顺序、reward/TQ 写入 | V1 的 ROSE rollout 实现 |
| `tree_advantage.py` | node value、segment advantage、length calibration | 通过 estimator registry 接入 PPO |

### 7.2 `verl/workers/rollout/logprobs.py`

新增的是无 vLLM 依赖的纯数据 helper，而不是把打包逻辑写进 vLLM package。这样：

- CPU 单测不必导入完整 vLLM；
- vLLM、vllm-ascend 或未来其他后端可以复用同一数组协议；
- rank 和 fallback 排序逻辑集中在一个地方。

### 7.3 `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

只改接口适配层：

```text
整数 logprobs=K 不再被当作 bool
每个位置打包 top-K
sampled token logprob 仍按 verl 原协议返回
```

没有侵入 vLLM sampler，也没有复制一份 vLLM engine。这样对 vllm-ascend 版本变化更耐受。

### 7.4 `verl/trainer/ppo/v1/agent_loop_tq.py`

原始 `AgentLoopWorkerTQ` 是 Ray remote class，不能直接作为普通 Python 基类继承。当前改成：

```python
class AgentLoopWorkerTQBase(AgentLoopWorker):
    ...

AgentLoopWorkerTQ = ray.remote(AgentLoopWorkerTQBase)
```

于是默认 worker 的公共 remote API 保持不变，ROSE 可以继承 base，复用 TQ postprocess、reward
和字段写入逻辑，同时覆盖 `_run_prompt()`。

这是一个通用的 verl 扩展技巧：先把可复用逻辑放进普通 class，再在边界处进行 Ray remote 包装。

### 7.5 `verl/trainer/ppo/v1/trainer_base.py`

新增的是额外字段提取，不是 ROSE 特判：

```python
fields = ["uid", ..., "values"]
if advantage_extra_fields:
    fields.append("extra_fields")
```

trainer 负责把 TQ 数据搬进 `DataProto`，但不理解 tree metadata 的内部格式。

### 7.6 `verl/trainer/ppo/ray_trainer.py`

在通用 estimator dispatch 中，将 `algorithm.advantage_extra_fields` 指定的字段加入 kwargs。
所以未来可以写：

```yaml
algorithm.adv_estimator: my_estimator
algorithm.advantage_extra_fields: [my_metadata]
```

而无需再次修改 trainer。

### 7.7 配置文件和 generated YAML

新增配置必须同时更新：

```text
verl/workers/config/rollout.py
verl/trainer/config/algorithm.py
verl/trainer/config/rollout/rollout.yaml
verl/trainer/config/ppo_trainer.yaml
```

仓库还保存 generated trainer YAML，因此修改 schema 后需要运行项目配置生成器，确保：

```text
_generated_ppo_trainer.yaml
_generated_ppo_megatron_trainer.yaml
_generated_ppo_torchtitan_trainer.yaml
_generated_ppo_veomni_trainer.yaml
```

与源 YAML 一致。只改命令行脚本而不改 schema，会在 Hydra/dataclass 转换时得到难以排查的配置错误。

### 7.8 `run_rose.sh`

它是根目录的 canonical 入口，设计目标是用户只需修改：

```bash
BASE_PATH='...'
MODEL_PATH='...'
```

脚本自动完成：

1. 检查 train/validation/reward 文件；
2. 选择仓库 `.venv/bin/python`；
3. 检测可见 NPU 数量；
4. 准备或复用与模型绑定的 embedding mmap；
5. 注入 ROSE manager、`rose_tree` estimator 和 V1 配置；
6. 调用 `python -m verl.trainer.main_ppo`。

注意脚本中的 `trainer.n_gpus_per_node` 是 verl 现有配置字段名称，Ascend 上它表示参与该
resource pool 的设备数量，并不意味着实际使用 CUDA GPU。真正的设备类型由：

```bash
trainer.device=npu
```

控制。

---

## 8. 从小到大运行：建议的学习和验收顺序

不要直接用论文规模跑 250 steps。先把每个边界分别验证。

### 8.1 纯算法单测（不需要 Ray/NPU）

```bash
python -m pytest tests/experimental/rose -q
```

重点观察：

- semantic entropy 的 mask/chunk/OOV；
- tree 拆边和 path finalization；
- tree advantage 的 node value 和 length calibration；
- top-K rank/fallback packing；
- 配置 fail-closed。

### 8.2 V1 邻近回归

```bash
python -m pytest \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py \
  tests/trainer/test_multi_trajectories_advantage_on_cpu.py -q
```

如果本机 Ray/系统权限限制导致 replay buffer 测试失败，应记录环境原因，不要把它误判成 ROSE
算法失败。

### 8.3 Hydra 配置解析

先检查 override 能否被解析，再启动 Ray：

```bash
python -m verl.trainer.main_ppo --help
```

实际启动时把脚本参数缩小：

```bash
TRAIN_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=1 \
PPO_MICRO_BATCH_SIZE=1 \
ROLLOUT_N=2 \
MAX_RESPONSE_LENGTH=256 \
TOTAL_EPOCHS=1 \
./run_rose.sh \
  algorithm.filter_groups.enable=false \
  trainer.total_training_steps=1
```

`trainer.total_training_steps=1` 比只设置 `TOTAL_EPOCHS=1` 更明确，因为 dataset size 仍可能
让 epoch 对应很多 step。

### 8.4 Ascend 单 prompt smoke test

在服务器上先验证 vLLM top-K：

```text
SamplingParams(logprobs=20, max_tokens=16)
```

断言每个位置：

- 有 sampled logprob；
- top-K 数量符合预期；
- `rank` 语义和上游 vLLM 一致；
- token id 未超出 embedding vocab；
- TP=1 与生产 TP 都能返回相同结构。

然后跑 `G=2`、短 response、单步 actor update。只有当以下链路都成功，才放大到 G=8：

```text
generate → top-K → semantic score → tree metadata → reward
→ replay sample → old/ref logprob → rose_tree advantage → actor update
```

### 8.5 正式训练前的指标

建议在日志中重点查看：

```text
Training Progress
pending/running/finished/failure
rose_semantic_score_seconds
rose_generation_seconds
rose_prefix_reuse_tokens
rose_new_generated_tokens
actor loss / entropy / KL
reward mean / accuracy
```

异常判断：

| 现象 | 优先检查 |
|---|---|
| `failure > 0` | reward、top-K 字段、tree metadata、TQ postprocess |
| `finished` 超过 batch 很多 | `algorithm.filter_groups` 是否启用 |
| 长时间 `running=32, finished=0` | vLLM 生成、reward timeout、CPU scorer、Ray event loop |
| policy version 错误 | V1 step 与 rollout version 的约定，不能直接删除校验 |
| NPU 上出现 CPU fallback | jagged/nested tensor 算子支持情况和数据搬运 |
| loss/advantage NaN | binary reward、空 response、mask、树路径合法性 |
| 单步几十分钟 | G、response length、filter_groups、CPU semantic scoring |

---

## 9. 当前实现的限制和为什么要 fail closed

### 9.1 单轮纯文本

ROSE prefix branching 需要知道 token prefix 的精确语义。多轮/tool 模式中，response 里可能
混入 observation、tool response、额外 mask；多模态还会涉及 processor 输出和视觉 token。
当前实现遇到这些输入直接报错，而不是静默生成错误树。

### 9.2 Binary reward

当前 estimator 默认要求每条 leaf reward 接近 0 或 1，因为 node value 的“成功率”解释和论文
设置都基于 binary verifier。若要支持连续 reward，至少需要重新审查：

- node value 是否仍是均值；
- length calibration 是否仍成立；
- zero-signal filter 是否应该保留；
- reward scale 是否需要 normalization。

### 9.3 Policy version 原子性

一棵树内所有 leaf 必须由同一 policy version 生成。当前 worker 会校验：

```text
min_global_steps == max_global_steps
所有 leaf version 相同
该 version 与当前 prompt 预期 version 一致
```

不能为了绕过异常而删除校验，否则同一树的不同分支来自不同 policy，node value 和 advantage
会混合不同分布。

### 9.4 Fully async 不支持

当前树是严格 on-policy 的顺序采样。fully async 会让不同 leaf 跨 policy 更新，必须引入 off-policy
校正或整树冻结协议；这不是把配置改成 `mode=async` 就能解决的问题。

---

## 10. 想继续扩展时，应该从哪里下手

### 10.1 支持多模态

不能直接删除 `RoseAgentLoop.build_prompt()` 中的文本检查。需要先定义：

1. branch position 是否只允许落在纯文本 response token；
2. processor 输出如何复用；
3. prefix 的 image/video payload 如何随请求携带；
4. response mask 如何区分模型 token 和 observation token；
5. embedding vocab 是否覆盖视觉特殊 token。

### 10.2 支持 NPU semantic scorer

建议新增独立 scorer actor：

```text
RoseAgentLoopWorkerTQ
  → compact top-K batch
  → RoseSemanticScorerActor（专用 NPU）
  → semantic entropy
```

不要把完整 embedding 复制进每个 vLLM TP rank。需要单独 benchmark H2D、embedding gather、
kernel 和 RPC 开销。

### 10.3 支持连续 reward

先把 `require_binary_reward=false` 做成显式实验选项，再增加独立测试验证 node value、
filter_groups 和 length calibration。不要只修改一行配置就宣称论文复现。

### 10.4 支持更高吞吐

优先级通常是：

1. 减小 response length，确认是否大量生成到上限；
2. 优化 prompt 间 worker 数和 vLLM batch；
3. 保证 prefix entropy 复用；
4. 关闭实验阶段的 `filter_groups`；
5. 减少 top-K embedding 随机读取；
6. 最后才考虑 NPU scorer 或更复杂的异步树调度。

---

## 11. 最后用一句话记住这套适配

```text
verl 负责“训练系统”和 PPO 基础设施；
ROSE rollout 负责“如何生成一棵有语义分叉的树”；
ROSE estimator 负责“如何把树上的最终 reward 变成 segment-level advantage”；
TransferQueue + extra_fields 负责把两者安全地接起来。
```

如果你要读代码，推荐按这个顺序：

1. `verl/trainer/ppo/v1/trainer_base.py` 的 `fit()`、`step()`、`_step_once()`；
2. `verl/trainer/ppo/v1/agent_loop_tq.py` 的 worker 和 manager；
3. `verl/experimental/rose/rose_agent_loop_tq.py` 的 `_run_prompt()`；
4. `verl/experimental/rose/semantic_entropy.py` 和 `tree.py`；
5. `verl/trainer/ppo/ray_trainer.py` 的 `compute_advantage()`；
6. `verl/experimental/rose/tree_advantage.py`；
7. 最后看 `run_rose.sh`，把配置和代码路径对应起来。
