# ROSE + EA 详细适配实施方案

版本：v1.2。日期：2026-11-12。目标分支：`rose-plus-ea`。

**文档状态：P0-P5 工程实施与本地验证已完成；P6 NPU 验证和 P7 效果实验待服务器执行。第 0.4 节保留 v1.1 历史交接快照，最新事实以第 14 节及 WORKLOG 为准。**

本文件是实施的主入口和验收依据。算法背景见 [ROSE_EA_DESIGN.md](ROSE_EA_DESIGN.md)，执行记录见 [ROSE_EA_WORKLOG.md](ROSE_EA_WORKLOG.md)。若本文件与初步设计存在差异，以本文件明确给出的首版接口为准；实施后必须同步两者，不可悄悄改变算法。

## 0. 任务、基线与恢复步骤

### 0.1 用户目标

在当前 ROSE + VeRL V1 + Ascend NPU 训练链路上加入 Entropy Advantage（EA），保留 ROSE 树采样及优势估计，提供可关闭的增量实现、独立运行脚本、测试和持续更新的文档。

已按用户后续实施目标接续初步代码并完成本地适配，另提供基于 `run_rose_new.sh` 的 `run_rose_ea.sh`。不执行服务器部署/训练，不自动提交或推送。

### 0.2 已核查的基线

- 分支：`rose-plus-ea`。
- HEAD：`e77dceb55b68416c3d153c3b36a0f13c815733dd`。
- 运行脚本模板：根目录 `run_rose_new.sh`。
- 训练入口：`python3 -m verl.trainer.main_ppo`。
- rollout manager：`verl.experimental.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ`。
- advantage estimator：`rose_tree`。
- actor：FSDP2；当前模型 `use_remove_padding=False`，每卡 microbatch=1。
- 用户提供的 `paper/EA.pdf` 当前未跟踪；不可删除或覆盖。
- 本组四份文档当前也未跟踪，属于本任务工作成果；不能按“未跟踪垃圾”清理。

本地脚本当前默认：batch=128、mini=32、response=3072、epsilon=0.3、ROSE length alpha=0、KL=0.01、clip high=0.20、训练 temperature=1.0、top_p=0.99、entropy_coeff=0、calculate_entropy=False、greedy 单样本验证。没有显式启用 group filter 或 seq-mean-token-sum-norm。**此前对话中的建议不等于脚本已修改；服务器版本也可能不同。**

### 0.3 下一轮上下文恢复的固定步骤

1. 完整阅读根目录 `AGENTS.md`、本文件、`ROSE_EA_README.md` 和最新 WORKLOG。
2. 执行 `git branch --show-current`、`git status --short`、`git rev-parse HEAD`。
3. 重新定位本文件引用的函数，不依赖旧行号，检查用户是否改过相同文件。
4. 如果工作区变化不影响方案，保护用户修改后继续；如果产生实质冲突，向用户说明。
5. 本文件已经提供首版默认选择；用户要求按方案实施时，不因非阻塞超参数选择反复询问。
6. 每完成一个阶段，更新 WORKLOG 的已修改文件、真实测试命令/结果、未完成事项。
7. 不自动提交、不推送、不创建 PR。不修改 `AGENTS.md`。

### 0.4 v1.1 时的工作区快照（历史记录，不是最新状态）

本节区分“已有代码”和“已验证代码”。以下是本次 `git status` 与 diff 的实际核查，不代表测试通过。

| 文件 | 已有初步改动 | 下一轮必须核查/补齐 |
| --- | --- | --- |
| `verl/workers/config/actor.py` | EA dataclass、读取 helper、ActorConfig 字段 | Hydra 实例化、非法配置和 inactive 行为测试 |
| `verl/trainer/config/actor/actor.yaml` | 默认关闭的 EA 块 | 同步 generated YAML，验证正式配置组合 |
| `verl/utils/config.py` | active-only 兼容性校验 | 用真实 resolved config 验证字段、错误提示及无效组合 |
| `verl/trainer/ppo/v1/trainer_base.py` | 元信息、启动日志、EA sum/finalize 接口 | TQ/mini/microbatch 传递及多子更新聚合测试 |
| `verl/workers/engine/fsdp/transformer_impl.py` | dense/packed `ea_entropy` 输出 | token shift、非连续张量、现有 logprob 梯度回归 |
| `verl/workers/utils/losses.py` | PPO 前 shaping、EA raw stats | 原优势不被改写、参数关闭时等价、nested 对齐测试 |
| `verl/workers/utils/entropy_advantage.py`（未跟踪） | 公式、no_grad chunk entropy、统计 helper | 数学、边界、数值稳定性及分布式统计测试 |

上述 6 个 tracked 文件当前 diff 合计 158 行新增；这是交接快照，不是后续必须维持的行数。未发现 `run_rose_ea.sh`，generated configs 和 EA 专项测试仍待补齐。已有实现尚未获得本轮测试证据，不能直接作为服务器正式训练版本。

下一轮优先顺序：保留并审查这些改动 → 补测试找问题 → 修正实现 → 生成配置 → 新建脚本 → CPU 回归 → 交付 NPU 验证步骤。不要为了恢复“干净基线”删除初步代码或用户的 EA.pdf。

## 1. 算法合同：必须保持的数学定义

来源：`paper/EA.pdf`，Reasoning with Exploration: An Entropy Perspective，arXiv:2506.14758v4，2025-11-08，第 3.2 节、式 (4)-(7)。ROSE 定义来自 `paper/ROSE.pdf` 第 3 节以及当前实现。未核查 EA 作者官方仓库，不声称复刻其每个工程细节。

```text
A_base = ROSE 节点 value 差，再经过现有长度校准后的逐 token 优势
H_current = 当前 actor 在该位置的完整词表 Shannon 熵

B = min(ea_coef * stop_gradient(H_current), abs(A_base) / kappa)
A_used = A_base + B
```

约束：`ea_coef >= 0`，`kappa > 1`。正式计算限定有效 response token。零优势仍为零；正优势不变负，负优势不变正。kappa=2 时，正优势最多放大到 1.5 倍，负优势绝对值最少保留一半。

EA 论文默认：GRPO coef=0.4、PPO coef=0.1、kappa=2。本组合初试 coef=0.1，原因是 ROSE 优势不做 GRPO 的组内标准差归一化，不能认为 0.4 在两者中强度相同。

### 1.1 处理顺序

```text
ROSE rollout -> verifier -> tree finalization / metadata
-> 节点 value -> segment advantage -> ROSE 长度校准 -> A_base 写入 TQ
-> actor 当前模型前向 -> current entropy -> EA shaping
-> 现有 PPO clipping / IS weighting / loss aggregation -> 现有 KL -> backward
```

注意：tree finalization 在实际代码中于 leaf reward 后处理前完成；上图表示数据依赖模块，不要求重新安排现有 rollout 执行时序。

EA 只能在 actor loss 前应用一次。不得：

- 加到 reward、节点 value、returns 或 rollout 的 logprob 中。
- 在 trainer 先加一遍、actor 再加一遍。
- 以 `advantages += bonus` 覆盖 TQ 中的基础优势；后续 epoch 必须从 A_base 重算。
- 对 A_used 再做中心化/whitening。
- 用 sampled surprisal、top-20 熵或 ROSE semantic entropy 替代 H_current。
- 使用熵的梯度或通过 entropy_coeff>0 冒充 EA。

### 1.2 组合性质和限制

- 共享前缀保留 ROSE 原有按 leaf 计数方式，不新增节点去重。
- 零 advantage 的公共 segment 不会被 EA 激活。
- EA 可能部分抵消长度校准对正优势的下调，不能预先保证效果更好。
- 不修复 ROSE 相关采样或 pivot 选择造成的统计偏差，不宣称新的无偏性定理。
- 符号保证针对 A_used；不意味着 KL、clip、跨 token 梯度相加后的参数更新方向完全不变。

## 2. 首版范围与默认选择

首版支持：V1 sync、`rose_tree`、`RoseAgentLoopManagerTQ`、单轮纯文本、FSDP/FSDP2 actor、vanilla token PPO loss、无蒸馏、非 fused logits 路径、actor SP=1、训练 temperature=1.0。NPU 是目标环境；CPU 是纯算法与接口测试环境。

支持 dense（当前 `use_remove_padding=False`）和 packed entropy 的位置恢复，至少以 CPU/mock 对齐测试覆盖两者；NPU 首先验证当前 dense 路径。没有实测的 packed NPU 路径不能标记为已验证。

首版不支持：fully async/one-step-off、Megatron/VeOmni/TorchTitan、蒸馏、其他 policy loss、fused entropy、sequence/context/tensor-parallel actor 的额外组合、非纯文本 ROSE、多轮工具。

**限制只在 EA 实际 active 时检查，EA disabled 不能使原本支持的其他算法报错。**

```text
ea_active = entropy_advantage.enable and entropy_advantage.coef > 0
```

`enable=True, coef=0` 按 no-op 处理，不新增 EA entropy 请求。默认关闭 EA 时必须保持原先训练行为、计算开关和指标不变。

首版保持 entropy_coeff=0；EA active 且 entropy_coeff!=0 显式报错，避免未经测试的正则组合。实现仍须保持原有 entropy 通路的梯度，不全局 detach，为后续扩展保留空间。

保持 ROSE 基线所有非 EA 超参数；不因加入 EA 自动切到 EA 论文的 no-KL、clip=0.28、8K response、不同 batch 或不同 loss 聚合。若用户要同时复现论文配置，作为单独实验处理。

## 3. 正式配置合同（已实现并完成本地验证）

新增 `EntropyAdvantageConfig(BaseConfig)`，建议定义在 `verl/workers/config/actor.py` 并加入 `__all__`。`ActorConfig.entropy_advantage` 使用 `field(default_factory=EntropyAdvantageConfig)`。

```yaml
entropy_advantage:
  _target_: verl.workers.config.actor.EntropyAdvantageConfig
  enable: false
  coef: 0.1
  kappa: 2.0
  entropy_chunk_size: 64
```

字段合同：

| 字段 | 规则 | 作用 |
| --- | --- | --- |
| enable | bool，默认 false | 是否启用 EA |
| coef | 有限实数且 >=0，不接受 bool 冒充数值 | bonus 系数 |
| kappa | 有限实数且 >1 | bonus 上限 |
| entropy_chunk_size | 正整数，不接受 bool | 每次处理的 token 数，不是 top-K |

为避免 dense 路径现有 chunk 开关未生效的歧义，**首版使用独立 EA chunk 配置**，不要求用户修改旧 `entropy_from_logits_with_chunking`。默认 64，后续 NPU 测量比较 128；这不是论文超参数。

验证配置路径：

- dataclass 构造与直接 Python 实例化。
- OmegaConf/Hydra `_target_` 递归实例化。
- 默认 YAML、四份 generated YAML 与 dataclass 一致。
- 配置为 dict 的测试/辅助调用应转换或有明确读取 helper，不假设所有调用都已实例化。
- 不绕过 frozen dataclass 约束，不在训练期间直接赋值不允许修改的字段。

不添加可切换 entropy 来源字段：首版只实现 current full-vocab。以后若做 old-policy/top-K 近似，另起实验模式并明确与论文公式不同。

## 4. 数据接口与熵通道

### 4.1 为什么新增独立 `ea_entropy`

当前 `model_output['entropy']` 同时服务监控/entropy regularization。为防止 EA 将旧分支意外 detach，首版使用独立输出：

```text
model_output['log_probs'] : 原逻辑，保留梯度
model_output['entropy']  : 原有可选逻辑，保持既有语义
model_output['ea_entropy']: 新增，可选，必须无梯度
```

不新增 TQ field，不把 ea_entropy 保存为训练 batch 的持久字段。它只在当前 actor forward -> loss 之间短暂存在。

### 4.2 非张量元信息

V1 `_update_actor()` 在 active 时通过现有 `batch.extra_info` 通道传递：

```text
calculate_entropy_advantage = True
entropy_advantage_chunk_size = entropy_advantage.entropy_chunk_size
```

普通 `calculate_entropy` 保持原先 `actor.calculate_entropy or entropy_coeff != 0` 的表达式，**不因 EA 自动变 True**，避免既计算完整可微 H，又额外计算 EA H。

inactive 时不增加这两个元信息，engine 缺省读取为 False。元信息必须实际到达每个 microbatch：用已有 TQ decorator/worker 数据恢复链路核实，不能仅在 trainer 写字段就认为传递成功。

在 actor loss 中，配置 active 但 ea_entropy 缺失必须报清晰错误；不得默默退化为旧 H、零 H 或 top-K H。engine 收到错误 chunk 参数也报错。

## 5. 按文件实施内容

### 5.1 `verl/workers/config/actor.py` / actor YAML / generated YAML

1. 新增 typed config、验证和导出。
2. ActorConfig 新字段与文档说明。
3. `verl/trainer/config/actor/actor.yaml` 新增上面的块，遵守“注释在字段上方、字段之间空行”的仓库格式。
4. 配置生成通过 `scripts/generate_trainer_config.sh` 或其正式等价流程完成，不把 generated YAML 当配置唯一来源。
5. 该脚本末尾会因 generated 文件相对 git 已改变而 exit 1；区分“文件确实已生成”和“末尾要求提交导致的非零退出”，不能掩盖真实 Hydra 失败，也不能为了通过检查自动提交。

### 5.2 `verl/utils/config.py::validate_config`

加入可单测的 EA 兼容性校验 helper，只在 active 时调用；依据第 2 节拒绝不支持组合。优先在启动大模型前失败。

检查 trainer mode、adv estimator、manager、backend/strategy、loss mode、temperature、entropy_coeff、distillation、fused/SP 等。实际字段使用源码真实路径，不根据旧对话臆造字段。

仅校验配置能确定的属性；若 engine 实际类型/布局只能运行时获知，在 engine 再校验。报错写出字段和支持条件，不静默覆盖用户设置。

### 5.3 `verl/trainer/ppo/v1/trainer_base.py::_update_actor`

1. 读取 typed/dict EA config，确定 active。
2. active 时加入第 4.2 节元信息。
3. 不改 `_compute_advantage()` 的数学计算，不在这里 shaping。
4. 不改变 old-log-prob 阶段现有 entropy 存储或 actor/entropy 指标含义。
5. actor 返回后处理 EA raw stats；最终派生指标见第 8 节。

### 5.4 `verl/workers/utils/entropy_advantage.py`

建议提供以下纯 tensor 接口（命名可小幅调整，但行为合同不变，调整需记录）：

```python
shape_entropy_advantage(advantages, entropy, response_mask, *, coef, kappa)
    -> (shaped_advantages, bonus, stats)

entropy_advantage_from_logits(logits, *, chunk_size)
    -> entropy  # shape = logits.shape[:-1], detached FP32
```

shaping helper：

- 输入相同 `[B,R]` 形状、同设备；A/H 采用 FP32 数值运算。
- 以 `torch.where(mask, value, 0)` 先清除无效位置，再运算，不依赖 `NaN * 0`。
- 有效位置 A/H 非有限、H < -1e-5 报错；`[-1e-5,0)` 的 H 截为 0，并将容差写入测试。
- 只从只读的基础 A 生成新张量；输出、bonus 的 padding 必须为 0。
- A 视为固定 advantage，H detach；bonus 不携带梯度。
- 输出不出现符号反转；不得用 abs(A) 的阈值人为改变公式。统计 ratio 时才使用非零条件。
- 返回 stats 为 detached 设备标量，包含 sums/counts，不包含每 token 巨大统计数组。
- 全零 mask 返回有限零统计，ratio 的零分母在最终聚合时处理。

entropy helper 的数值与内存合同见下一节。它不得导入 vLLM、Ray 或 torch_npu，便于 CPU 单测；NPU 依赖由训练环境提供。

### 5.5 `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_outputs`

1. 读 EA 元信息，保持原 logprob/entropy/蒸馏通路行为不变。
2. EA active 且非 fused 路径，在与 actor logprob 相同的、温度缩放后的 logits 上计算 H。
3. packed 分支使用对应 logits 及已有 SP gather/unpad 流程，dense 分支使用 `[B,S,V]`。
4. 将 EA H 按与 logprob 相同的 token 布局恢复为 nested/dense 输出，键为 `ea_entropy`。
5. 当前支持 SP=1，不因为“写了 gather 代码”就声称 SP 已支持。
6. 若普通 entropy 已在同一前向且按同一分布计算，可复用其 detach 值；但默认脚本 calculate_entropy=False 时必须直接走 EA-only no_grad 分块路径，不创建 H 计算图。
7. 不额外 forward 模型，不向 CPU 搬词表，不改变 logits 或原 logprob 的梯度。
8. fused/其他不支持布局收到请求时显式失败。

### 5.6 `verl/workers/utils/losses.py::ppo_loss`

1. 与 log_probs 一样，用 `no_padding_2_padding` 将 ea_entropy 对齐到 response。
2. 保持原数据 select、IS weights、ref_log_prob 等字段。
3. 在读取 A、mask 后、调用 policy_loss_fn 前应用 shaping helper。
4. 将局部变量 A_used 传给现有 vanilla policy loss；原 A/TQ/returns 不改。
5. 原 entropy regularization 分支不读取 ea_entropy；EA stats 独立命名。
6. inactive 完全绕过 helper，不要求模型提供 ea_entropy，不新增指标或设备计算。
7. 返回 EA raw stats，按第 8 节聚合，不返回依赖 padding 或微批长度的均值。

### 5.7 `verl/workers/engine_workers.py` / V1 metrics 接口

当前 `Metric.aggregate_dp` 即使 SUM 也先对 DP ranks 做 mean，且 actor mini-batch 输出最后又经 `reduce_metrics` 做 mean。需要针对 EA raw stats 保留可加性：

- loss 内对每个 EA 本地 sum/count 返回 `Metric(SUM, local_stat * dp_size)`，补偿框架 DP mean；现有 loss/其他指标不动。
- worker 每个 mini-batch 的 EA raw stats 经过 DP 聚合后应为全局 sum。
- trainer 对 actor 返回的 EA raw-stat 列表先做 SUM 并移出普通 `reduce_metrics`，不能被其默认 mean。
- 多个 parameter_sync_step 子更新时，V1 MetricsAggregator 必须继续对 raw stats 求 SUM。
- 在最终日志写入前，按累计 raw stats 派生 ratios/means；不要提前平均各子更新的 ratio。
- 如实现采用另一种等价的计数通道，必须有不等长 microbatch、模拟多 DP、多个 mini-batch 的测试证明准确。不要全局修改 Metric.aggregate_dp 语义以修 EA。

必要改动可能位于 trainer_base 的 actor 指标归约和最终 `_compute_metrics`，或 `verl/trainer/ppo/v1/utils.py` 的专用 EA finalize helper。以最小局部实现为准，记录函数路径。

## 6. NPU 全词表熵：具体计算策略

### 6.1 计算设备和梯度

H 在 actor NPU 上计算，EA-only helper 内 `torch.no_grad()`。logits 不整体 cast 到 FP32，不调用 `.cpu()`/`.numpy()`。

每个 token chunk 才转 FP32。可用以下等价稳定形式：

```text
z = float32(logits_chunk)
z = z - max(z, vocabulary_dim)
p = softmax(z)
H = logsumexp(z) - sum(p * z)
```

平移不改变熵，便于控制数值误差；helper 最终返回每 token 一个 FP32 标量。Qwen 词表允许额外 embedding padding 行时，沿 actor 实际输出词表计算；不能自行把长度截到 tokenizer len 而改变策略分布。

### 6.2 内存布局

- 支持二维 `[N,V]` 和三维 `[B,S,V]`，返回 `[N]` 或 `[B,S]`。
- dense 可按 batch 与 token slice 循环，避免对非连续 logits 执行全量 `.contiguous()`/reshape 导致复制完整词表张量。
- 必须测试非连续输入。每个临时 FP32 张量的 token 维不超过 chunk_size（词表维完整）。
- 输出预分配在同设备，不保留每块概率列表，不缓存整份 softmax。
- chunk 减少额外 entropy 中间内存，不能消除模型原本的 logits 内存。
- 默认 64 token/块，约 24.8 万词表时一份 FP32 块约 61 MiB；多个中间张量会叠加，不能把 61 MiB 说成总峰值。
- 无必要不逐 chunk `.item()` 同步 NPU；在 helper 完成后合并检查数值。错误检查的同步成本需在 NPU profile 中记录。

### 6.3 对齐要求

完整 response 第一个 token 的 H 来自最后一个 prompt 位置的 logits，而不是 response 第一个 token 自身位置的 logits。复用 `no_padding_2_padding` 的一位 shift，不能自己按 response 长度粗略切尾。

EOS 若为有效 response token 则正常参与；EOS 后 padding 为零。prompt、padding、上采样空行不参与 shaping 和 stats。

## 7. 运行脚本方案

新增根目录 `run_rose_ea.sh`，不覆盖 `run_rose_new.sh`。以当前本地基线为默认模板；若用户实施前提供更新后的服务器脚本，优先用它。

保留原脚本的 BASE_PATH、MODEL_PATH、train/val 文件、reward 路径、多节点发现、CANN/NPU 环境、代码部署复制流程和训练入口。不执行脚本验证，因为它会部署环境并启动集群；只做静态检查与 mock/配置检查。

新增变量：

```bash
EA_ENABLE=${EA_ENABLE:-True}
EA_COEF=${EA_COEF:-0.1}
EA_KAPPA=${EA_KAPPA:-2.0}
EA_ENTROPY_CHUNK_SIZE=${EA_ENTROPY_CHUNK_SIZE:-64}
```

新增 Hydra overrides（实现后才能运行）：

```bash
actor_rollout_ref.actor.entropy_advantage.enable=${EA_ENABLE} \
actor_rollout_ref.actor.entropy_advantage.coef=${EA_COEF} \
actor_rollout_ref.actor.entropy_advantage.kappa=${EA_KAPPA} \
actor_rollout_ref.actor.entropy_advantage.entropy_chunk_size=${EA_ENTROPY_CHUNK_SIZE} \
```

保留 `entropy_coeff=0`、`calculate_entropy=False`。不要为了 EA 设置普通 calculate_entropy=True；自动 EA 通道负责计算。

新实验名和输出路径默认带 rose_ea，避免覆盖 ROSE 输出；原有用户显式路径覆盖应保留。记录 resolved EA config、代码文件来源和 commit，以防多节点复制后运行旧代码。

EA_ENABLE=False 在相同新脚本上即为无 EA 对照，不自动修改其他基线参数。验证 n=1 与 n=8 使用同一配置分别对照；现有 n=1 贪心指标不能宣称 pass@8，不能自动用树内 8 leaf 当独立验证采样。

不扩展训练任务为自动依赖安装或多种子调度；保留模板已有部署内容，不新增无关安装/下载。脚本头部修正遗留 GRPO/多模态说明，使其准确说明 ROSE+EA 与 NPU。

## 8. 可观测性与统计定义

### 8.1 必需 raw stats（内部名）

统一前缀 `actor/ea/raw/`（worker 在 trainer 外层加 actor 前缀时，loss 中只写 `ea/raw/`，避免 actor/actor 重复）。

```text
valid_tokens_sum, nonzero_adv_tokens_sum
positive_tokens_sum, negative_tokens_sum
entropy_sum, bonus_sum, bonus_ratio_sum
cap_hits_sum, sign_flips_sum
base_positive_sum, shaped_positive_sum
base_negative_sum, shaped_negative_sum
```

- ratio 逐 token 为 B/abs(A)，只计 A!=0；它的分母是 nonzero_adv_tokens。
- cap hit 指 `coef*H >= abs(A)/kappa`，且 A!=0；不把 A=0 的零 cap 计成饱和。
- sign_flips 指有效位置原 A 非零但 A_used 改变符号或变为零；正常应为 0。
- stats 表示训练 forward 的 token exposures，多 PPO epoch 会重复计数；不是唯一 rollout token 总量。

### 8.2 最终派生指标

```text
actor/ea/entropy_token_mean = entropy_sum / valid_tokens_sum
actor/ea/bonus_mean = bonus_sum / valid_tokens_sum
actor/ea/nonzero_adv_fraction = nonzero_adv_tokens_sum / valid_tokens_sum
actor/ea/bonus_ratio_mean = bonus_ratio_sum / nonzero_adv_tokens_sum
actor/ea/cap_fraction = cap_hits_sum / nonzero_adv_tokens_sum
actor/ea/sign_flips = sign_flips_sum
```

正/负优势分别除以对应 token count。分母为 0 时输出 0，并保留 count 供辨别“没有样本”和“均值为零”。raw stats 可在 debug 日志保留，至少最终的 counts 应可追溯。

这些是有效 token 加权均值，与当前固定长度聚合下的 actor/entropy 不必数值相同。不能用固定长度 loss 分母来计算 EA entropy_token_mean。

日志启动时输出一次 EA enable/active、coef、kappa、chunk、current/full-vocab/detached/NPU 计算模式。不要每 token print；短 smoke test 可额外输出少量 A/H/B 三元组，不默认保存巨量数组。

保留原 reward、输出 JSONL、response length、截断率、KL/clip、actor update 耗时和峰值显存；记录新增生成 token 与 leaf 训练 token 的区别。

## 9. 测试计划与预期结果

以下为测试行为合同。实际新增四份文件：config、on_cpu（合并公式与熵计算测试）、integration（含聚合）、script；没有额外创建独立 entropy 测试文件。实际命令和结果见 WORKLOG。使用仓库 uv 环境，不用系统 Python 3.14 跑要求 Python<3.13 的项目。

### T1：配置测试 `tests/experimental/rose/test_entropy_advantage_config_on_cpu.py`

- dataclass/Hydra/default YAML 一致，nested config 不是未实例化 dict。
- coef=0 允许；负数、NaN/Inf、kappa<=1、chunk<=0/非整数拒绝。
- active 时不支持模式失败；inactive 时原模式照常通过。
- active 的默认限制不会错误套用到 validation sampling temperature=0。

### T2：公式与梯度 `tests/experimental/rose/test_entropy_advantage_on_cpu.py`

- 手算正/负/零 A、cap 两侧、H=0 和大 H。
- H requires_grad=True 时，loss.backward 不给 H 梯度，但给 policy logprob 正确梯度。
- disabled/coef=0 等于基础 loss；不原地改 A。
- padding 中 NaN/Inf 清零；有效 NaN/Inf 报错；全 padding 安全。
- 负熵容差边界；shape/device mismatch 明确错误。
- 相同 A、不同 current H 多次调用不累积。

### T3：熵计算 `tests/experimental/rose/test_entropy_advantage_entropy_on_cpu.py`

- 2D/3D、连续/非连续、chunk=1/64/大于 token 数，与全 FP32 Shannon 熵一致。
- 平移 logits 不改变 H；大幅 logits、近确定分布、均匀分布（H=log V）。
- 返回 FP32、同设备、requires_grad=False/grad_fn=None。
- 验证没有全量 FP32 softmax 副本；可用局部 helper spy 检查每块 token 上界，结合 NPU 实测确认真实峰值。

### T4：loss 与数据链路 `tests/experimental/rose/test_entropy_advantage_integration_on_cpu.py`

- 使用短 prompt/不等长 response 构造 nested 数据，确认 shift、EOS 和 mask。
- mock policy loss 收到精确 A_used；TQ/basic advantage/returns 未修改。
- 缺少 ea_entropy 在 active 时失败，inactive 时不要求此字段。
- engine 元信息经过 mini/microbatch 分割仍存在；inactive 不新增请求。
- 当前 dense、packed 对齐结果一致，logprob 梯度不受 no_grad 污染。
- 普通 entropy 通道原有梯度行为保持；EA 不误开启正则。

### T5：聚合测试

- 不等长两 microbatch、两个模拟 DP ranks、多个 mini-batch 的 sums/counts 与直接拼接参考一致。
- 额外测试 parameter_sync_step>1 的最终 ratio，而不是子 step ratio 的均值。
- 原 `Metric` 和其他 actor 指标聚合回归不变。
- 固定 current logits、不改变模型参数时，改变 microbatch 切分得到相同梯度与 EA 统计。
- 不要求“多次 optimizer update 后”与单次大 batch 轨迹相同；那是不同优化过程。

### T6：现有 ROSE 回归

```bash
uv run --no-sync python -m pytest -q tests/experimental/rose
uv run --no-sync python -m pytest -q tests/trainer/ppo/test_dynamic_policy_losses_on_cpu.py
bash -n run_rose_ea.sh
git diff --check
```

根据真实测试依赖追加 metrics/padding 相关回归。若 uv 环境缺依赖，记录缺少项，按 AGENTS 使用 uv 安装最小测试依赖；不为了 CPU 单测无条件安装完整 vLLM/Ascend/GPU 堆栈。网络或 NPU 不可用时，报告验证边界，不将测试标为通过。

配置生成先使用 `uv run --no-sync python scripts/print_cfg.py --cfg job`（可追加四种 model_engine 参数核查），再按生成脚本同步；记录真实返回码与原因。

### T7：服务器 NPU 验收（本地无 NPU 时由用户执行）

1. 保存依赖版本、NPU型号、节点数、代码 commit/工作区补丁和 resolved config。
2. 同一新脚本 EA off，短样本 1 step：确认纯 ROSE 回归。
3. EA on，短序列 1 step：确认 ea_entropy、bonus、反向传播、无乱码/死锁。
4. 当前真实 response 长度运行 3-5 steps，chunk=64；记录 actor update 秒数、峰值显存、sign_flips=0、cap_fraction。
5. 若余量允许比较 chunk=128；不因 OOM 擅自降低 response 长度造成不公平比较。
6. 再运行 ROSE/ROSE+EA 效果实验；不承诺固定开销百分比或必然提分。

## 10. 实施阶段与完成条件

| 阶段 | 工作 | 完成条件 |
| --- | --- | --- |
| P0 | 重新核查分支/用户修改/基线 | WORKLOG 更新，不覆盖用户内容 |
| P1 | config 与 compatibility guard | T1 通过，默认关闭不影响原模式 |
| P2 | 纯公式与 chunk entropy helper | T2/T3 通过 |
| P3 | engine 元信息/ea_entropy/loss 接入 | T4 通过，真实 current H，无重复 shaping |
| P4 | metrics 可加聚合 | T5 通过，符号翻转为 0 |
| P5 | 独立脚本、generated configs、文档 | bash/config 检查与 T6 回归通过 |
| P6 | NPU smoke/性能 | T7 由有 NPU 的环境验证，缺硬件时列为待办 |
| P7 | 效果对照 | 相同配置/预算实验，单独报告，不与代码适配完成混淆 |

P5 本地完成可以交付“代码已适配、CPU 验证通过、NPU 待验证”，不能声称 P6/P7 已完成。工程实施完毕必须列出实际改动、参数、测试结果、未验证项和服务器运行方式。

## 11. 回退与失败处理

- EA_ENABLE=False 应恢复当前 ROSE 行为，无 EA entropy 开销；已有普通熵计算开销保持原状。
- 训练恢复同一个 checkpoint 时可以禁用 EA 做工程排障，但切换算法后的曲线不能混称同一对照实验；保存新 experiment/output 目录。
- OOM：先检查是否错误计算两套熵/保留梯度图，确认 chunk 实际执行，再缩小 EA chunk。不要先改 ROSE 轨迹数或训练长度掩盖问题。
- bonus=0：检查 active、有效 mask、A 是否全零、H 是否实际传入，不允许自动改公式给零优势加常数。
- cap_fraction 接近 1：记录优势尺度，尝试更小 coef；不说明为代码必然错误。
- 速度下降：分开测 actor 熵计算、长 response 带来的 rollout 变化和 DAPO 补采；不一概归因于 EA 的一行加法。
- NPU 不支持某算子：记录实际堆栈与 shape，优先替换成等价 NPU tensor 运算；不要默默搬全词表到 CPU。
- 不得使用 destructive git 命令回退用户修改。需要撤回任务改动时按文件精确处理并说明。

## 12. 文档同步合同

实施时同步维护：

1. 本文件：接口或范围改变时更新版本与理由，保留算法合同。
2. `ROSE_EA_DESIGN.md`：反映真实最终设计，不再留与实现冲突的草案。
3. `ROSE_EA_WORKLOG.md`：阶段、逐文件修改、实际测试命令、结果与阻塞项。
4. `ROSE_EA_README.md`：当前能否运行、如何启动、下一步、NPU 验证状态。

首版工程核对清单（P0-P5）：

- [x] 当前分支为 rose-plus-ea，用户改动已保护。
- [x] 首版是 ROSE + actor-side EA，不是改 rollout metric。
- [x] A_base 是长度校准后优势，H 是 current full-vocab，bonus detach。
- [x] 使用独立 ea_entropy 通道与独立 chunk 配置。
- [x] 默认关闭/coef=0 无额外熵计算；新脚本默认开启 EA。
- [x] 数据路径、多节点 NPU 环境保持基线；源码复制修正见第 14 节，集群实测待 T7。
- [x] CPU 公式/接口测试通过；服务器 NPU 仍待验证，不声称硬件测试通过。

**下一轮用户说“按方案实施”后，从 P0 开始，按阶段完成，不需要重新发散设计。遇到源码基线变化时以实际接口校正工程细节，并将任何实质算法变化先告知用户。**

上述为原实施约定；v1.2 已完成 P0-P5，后续从服务器验证接续，不重复创建实现。

## 13. 可直接执行的实施交接与审查样例

### 13.1 建议按以下顺序提交本地工作成果（不是自动 git commit）

1. **基线与配置**：记录当前 diff；验证 EA config，核对真实 trainer/engine 选择路径。不要只因 strategy 字段写着 fsdp2 就认定运行时 engine 必然匹配。
2. **纯函数测试**：先测现有 helper；失败时只修对应数学/边界，不动 rollout。测试代码使用短随机词表，无须加载 9B 权重。
3. **engine/loss 接口**：用最小 nested TensorDict 和 mock engine 执行真实入口，覆盖 dense/packed；对照普通 logprob 的反向梯度。
4. **元信息与统计**：验证 extra_info 经 TQ、mini/microbatch 保留；测试 DP mean 补偿、mini list 求和、parameter_sync_step 求和后再算 ratio。
5. **配置生成与脚本**：同步 generated YAML；从 `run_rose_new.sh` 新建脚本，仅增 EA、独立实验输出与必要日志，不顺手变更 ROSE 超参数。
6. **回归与交付**：记录测试命令、数量、失败/跳过原因；更新四份文档。没有 NPU 则明确交付“本地测试结果 + 服务器待验证清单”。

各步只有通过相应测试才可在 WORKLOG 打勾；“代码已经加进去”不算 P1-P4 完成。

### 13.2 精确 token 对齐样例

单条序列为 `[p0, p1, r0, r1, eos]` 时，模型位置与预测目标如下：

| logits/entropy 位置 | 预测目标 | 是否用于 response EA |
| --- | --- | --- |
| 0（p0） | p1 | 否 |
| 1（p1） | r0 | 是 |
| 2（r0） | r1 | 是 |
| 3（r1） | eos | 是，若 response_mask 含 EOS |
| 4（eos） | 下一 token | 否 |

令各位置 H 分别为 `[0.1, 0.2, 0.3, 0.4, 0.5]`，最终 response H 必须为 `[0.2, 0.3, 0.4]`，而不是 `[0.3, 0.4, 0.5]`。另加一条不同 prompt/response 长度的序列，验证不是统一粗略切尾。masked response padding 中放 NaN，确认被清零且不污染有效位置。

### 13.3 精确公式与梯度样例

设 `coef=0.1, kappa=2`：

| A_base | H | bonus | A_used |
| --- | --- | --- | --- |
| 0.2 | 0.5 | 0.05 | 0.25 |
| -0.2 | 0.5 | 0.05 | -0.15 |
| 0 | 10 | 0 | 0 |
| 0.2 | 10 | 0.1 | 0.3 |
| -0.2 | 10 | 0.1 | -0.1 |

先单测上述数值，再让 logprob 与 H 都 requires_grad，构造 policy loss 反向：logprob 必须有符合参考公式的梯度；H 不得有梯度。对同一 A 连续调用两次，第二次从 A_base 重新计算，不得从第一次 A_used 继续叠加。

### 13.4 统计加权反例

假设两个分片分别有 1 个和 9 个有效 token，熵均值分别为 1 和 3。正确全局均值为 `(1*1 + 9*3)/(1+9)=2.8`，不是 `(1+3)/2=2`。

测试须同时覆盖：microbatch→worker、DP→worker、mini-batch→trainer、多个 parameter_sync_step→最终 logger。某一级只要提前计算均值再平均，就可能错误。raw counts 是训练 token exposures，多 PPO epoch 计多次是预期行为。

### 13.5 最终交付物与禁止遗漏项

- 可审查的 Python/YAML 实现，默认 EA 关闭；有效配置下启用，禁用和 coef=0 不请求额外熵。
- `run_rose_ea.sh`：基于现有模板，模型/数据/reward/多节点路径保持；EA 参数可在脚本或环境变量中填写，默认输出目录独立。
- CPU 单测/集成测试及实际结果；脚本 `bash -n`、Hydra 配置组合、`git diff --check`。
- 更新后的四份 MD；WORKLOG 列出通过项和未通过/未运行项。
- 服务器运行说明必须包含 EA off/on 的相同基线对照、短序列单步、真实长度 3-5 步、耗时和显存检查。
- 不把纸面方案当成运行结果，不承诺速度/收益；不修改用户奖励、不将 EA 熵挪到 CPU、不自动部署或启动服务器。

## 14. v1.2 实施落点与交付核查

- typed config/YAML 默认关闭；active 定义和公式不变，训练 temperature=1、vanilla、entropy_coeff=0 等支持范围在启动时校验。
- V1 `_update_actor` 携带 EA 元信息，经实际 TQ 恢复和 mini/microbatch 测试；engine dense/packed 提供 detached `ea_entropy`，loss 复用现有 response shift。
- 纯 helper 不依赖 Ray/vLLM/NPU 库。全词表 softmax/reduction 在 logits 所在设备、no_grad、FP32 token chunk 中完成，不将 logits/概率搬 CPU。仅 13 个日志标量合并搬回 CPU；有效数值检查另有一次同步。
- 使用 `Metric(SUM, local_stat*dp_size)`，trainer 单独求和 raw lists，最终日志前 finalize ratios；用不等长分片、模拟 DP 与多子更新覆盖。
- 普通 entropy/regularizer 通道保持梯度；EA disabled/coef=0 无额外 entropy 请求，测试覆盖 engine 不调用 EA helper。
- 原 `run_rose_new.sh` 未修改。新脚本仅增加四个 EA overrides，非 EA Hydra 参数逐项对照一致；正式 `validate_config` 已对脚本 EA on/off 配置通过。
- 部署修正：显式复制 source contents 到 `/cache/algorithm`，避免目标已存在时嵌套 algorithm/algorithm；cd 与 PYTHONPATH 固定到部署源码；日志文件默认 train.log。均不改变算法或原模型/数据/reward 路径。
- 四份 generated configs 已由官方生成脚本生成；其中 dp 参考文件同时同步了源 YAML 先前已有的 `grad_sync_every_micro_batch=false`，不是新增训练开关。脚本末尾因 git diff 非零返回 1；另有四模式测试逐字验证生成内容一致。
- 现有 rollout 测试仅补齐 logger 需要的 mock stop_reason/tokenizer，未改生产 rollout 逻辑。
- NPU 9B 端到端训练、集群通信、NPU 算子实际支持、峰值显存/耗时、训练收益均未在本地验证；按 T7 执行后再记录。
- 最终本地证据：164 passed、1 skipped；ruff check/format、bash -n、git diff --check 与 Markdown 检查通过。具体命令、跳过原因见 WORKLOG。
