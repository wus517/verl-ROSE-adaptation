# ROSE + Entropy Advantage：算法与 Ascend 适配设计

更新日期：2026-11-12。状态：P0-P5 已实现并完成 CPU 验证；NPU smoke/性能与效果实验待服务器执行。逐文件接口、真实交接状态与验收以 [详细实施方案](ROSE_EA_IMPLEMENTATION_PLAN.md) v1.2 为准。

## 1. 来源与适用范围

EA 来源：[Reasoning with Exploration: An Entropy Perspective](paper/EA.pdf)，本地 PDF 为 arXiv:2506.14758v4，2025-11-08，重点依据正文第 3.2 节、式 (4)-(7)、第 4 节和附录 B。

ROSE 来源：[ROSE 论文](paper/ROSE.pdf)，重点依据第 3.1-3.3 节，以及本项目当前 `verl/experimental/rose/` 的实际实现。

本设计是将 EA 的 advantage shaping 迁移到 ROSE 的组合提案，不是宣称任一论文已经验证过 ROSE+EA。本轮未核查 EA 作者的官方代码，论文之外的实现细节以当前源码和明确标记的工程设计为准。

目标环境：VeRL V1 sync trainer、ROSE TransferQueue AgentLoop、FSDP2、vLLM/vllm-ascend、华为 Ascend NPU、单轮纯文本和二值数学奖励。

## 2. EA 做了什么

### 2.1 原始公式

对 response 的每个 token 位置 t，使用当前 actor 策略的全词表熵：

```text
H_t = -sum_{v in vocabulary} pi_theta(v | prompt, response_<t)
                              * log pi_theta(v | prompt, response_<t)

bonus_t = min(ea_coef * stop_gradient(H_t), abs(A_t) / kappa)
A_shaped_t = A_t + bonus_t
```

其中 `ea_coef >= 0`（0 用于禁用消融），`kappa > 1`。论文符号 alpha 在本设计中改名为 `ea_coef`，避免与 ROSE 长度校准 alpha 混淆。

论文所有实验 kappa=2，GRPO 的 ea_coef=0.4，PPO 的 ea_coef=0.1。这不是 ROSE 的已验证默认值。

### 2.2 不要误解为熵正则

- EA 的熵必须 detach。没有额外的 `grad(H)` 项。
- 不需要将 `entropy_coeff` 设为正数；首版保持 0。
- 不给 reward 加 entropy，不改变 verifier，不用熵修改树节点 value。
- 不把 `-log pi(sampled_token)` 当成全词表熵。后者只是 sampled surprisal，单个 token 不等于熵。
- 不保证策略熵会上升。论文中 EA 模型的熵也会下降，目标是探索行为与推理效果，而非维持某个熵数值。

### 2.3 对不同符号优势的作用

在 kappa=2、A=0.2、H=0.5、ea_coef=0.1 时：bonus=0.05，A_shaped=0.25。

在 A=-0.2、其余相同时：bonus=0.05，A_shaped=-0.15。负优势仍是惩罚，只是惩罚减弱。

在 A=0 时，无论熵多高：bonus=0，A_shaped=0。

因此 EA 并非对负优势乘更大的正系数。正确公式是给所有 token 加非负偏移。kappa=2 时，正优势最多变成 1.5 倍；负优势的绝对值最小为原来的 0.5 倍。符号保证只针对 policy advantage，不保证加上 KL 等项后的完整参数梯度方向不变。

## 3. 与 ROSE 的结合公式

先保持现有 ROSE 完整流程：

```text
树 rollout -> 二值 leaf reward -> 节点均值 value
-> 相邻节点 value 差形成 segment advantage
-> 对较长正确轨迹做长度校准
-> A_ROSE
```

再在 actor 当前前向与 policy loss 之间：

```text
bonus_t = min(ea_coef * H_current_t.detach(), abs(A_ROSE_t) / kappa)
A_combined_t = A_ROSE_t + bonus_t
```

最后将 `A_combined` 代入现有 PPO clipped surrogate；KL、IS weights 和 loss 聚合仍由已有实现处理。

### 3.1 推荐的顺序与不变项

- 推荐“长度校准之后再 EA”，并使用校准后优势确定 bonus 上限。
- 不在 trainer 和 actor 各加一次 EA；每次 actor forward 只从原始 `A_ROSE` 重新计算一次，不累积写回。
- 不修改 `tree_advantage.py` 的节点 value、树关系验证、返回的 returns。
- 不对 EA 后优势再做 whitening 或组内重新中心化，否则会改变 EA 公式及符号性质。
- 不改变每题 8 条轨迹、不增加树分支、不改变 prefix 复用和 reward 调用次数。
- 最小组合不等于理论无偏的独立采样估计：ROSE 分支已有相关性与选择偏差，EA 不会自动修复它们。

### 3.2 两种熵必须分开

| 内容 | ROSE 分支信号 | EA 优势信号 |
| --- | --- | --- |
| 来源 | vLLM 返回 top-K 与 input embeddings | actor 当前前向的全词表 logits |
| 形式 | 截断生成熵乘语义差异，可为负 | Shannon 熵，理论非负 |
| 位置 | rollout 控制端 | actor 更新端 |
| 设备 | 当前 host CPU 语义 scorer | actor 所在 NPU |
| 用途 | 选择 pivot | 有界地改变 token 优势 |

不能直接复用 `semantic_entropy`、`rose_semantic_entropy_max` 或 vLLM top-20 截断熵来声称实现了 EA 原公式。现有 trainer 存储的 `entropy` 是 old-policy 重算阶段的熵；首版也不以它替代 current-policy 熵。

当前引擎先按 rollout temperature 缩放 logits。首轮保持训练 temperature=1.0，可避免 current-policy 熵与温度变换口径混淆；若以后使用其他温度，需要同时定义 policy ratio 与 EA 熵的分布口径，而不是偷偷改变。

## 4. 已核查的源码接入点

### 4.1 Rollout 与优势保持原样

- `verl/experimental/rose/rose_agent_loop_tq.py`：树内按轨迹依赖顺序生成、树间并行，维护语义评分和 tree metadata；验证走独立生成分支。
- `verl/experimental/rose/rose_agent_loop.py`：token prefix 续写，不 decode 再 tokenize。
- `verl/experimental/rose/tree_advantage.py`：按真实节点身份计算 value 差及长度校准，默认检查 reward 为 0/1。
- `verl/trainer/ppo/v1/trainer_base.py::_compute_advantage`：先完成 ROSE 优势，写入 TQ。

首版不需要改上述 rollout 或 tree estimator 的算法。

### 4.2 必须接通 actor 熵

`trainer_base.py::_update_actor` 当前仅在 `actor.calculate_entropy` 或 `entropy_coeff != 0` 时请求普通训练熵。EA active 时通过新的 `calculate_entropy_advantage` 元信息独立请求熵，普通 `calculate_entropy` 表达式不变，避免计算两套熵。

`verl/workers/engine/fsdp/transformer_impl.py::prepare_model_outputs` 已有 `model_output['entropy']` 输出。首版新增独立 `model_output['ea_entropy']`，使用 EA-only detached/no_grad 分块计算，避免生成熵反传图后再 detach 浪费显存；复用既有布局恢复和 response 对齐接口。

首版 EA active 时不支持同时使用 entropy regularization，应明确报错。原有非 EA 正则分支对 H 的梯度必须保留；不能全局 detach 所有 entropy。

### 4.3 最小优势修正入口

`verl/workers/utils/losses.py::ppo_loss` 当前已：

1. 从模型输出取 log_probs 和 entropy；
2. 用 `no_padding_2_padding` 对齐到 response；
3. 读取 advantages、response_mask；
4. 调用注册的 policy loss。

在第 3 与第 4 步之间做 EA 最自然。不要仅修改旧版论文脚注中的 `workers/actor/dp_actor.py`；本项目当前 actor loss 不走那个旧路径。

建议新增纯 tensor helper `verl/workers/utils/entropy_advantage.py`：输入 A/H/mask/coef/kappa，返回新优势、bonus 和可聚合统计；不能原地覆盖 TQ 中原始优势。

要求：形状一致；只使用有效 response 位置；无效 padding 在参与数值运算前通过 where 清零；有效位置 NaN/Inf 和显著负熵 fail closed，允许在明确容差内修正浮点微小负值；原优势保持只读。

第一版限制到已验证的 vanilla token PPO loss 与 ROSE sync 路径。其他 policy loss、蒸馏、fused entropy、异步路径在测试前不能声称支持；不支持组合应显式报错。

## 5. NPU 与显存方案

EA 不增加 rollout 请求，但不是零开销。训练时额外的全词表熵需要 softmax/reduction，还可能因模型变长间接增加后续 rollout 成本。

尤其 Qwen3.5 词表约 24.8 万：即便按 4096 个位置计算，仅 FP32 概率临时张量也约 3.8 GiB；实际若包含 prompt 或其他中间结果，峰值会更高。detach 之后才计算完并不能消除前面的峰值。

当前代码的特殊风险：

- 脚本为 `model.use_remove_padding=False`。
- `prepare_model_outputs` 的 remove-padding=True 分支使用 chunk 配置；False 分支直接调用普通 `entropy_from_logits`。
- 所以仅将 `entropy_from_logits_with_chunking=True` 写进脚本，不保证当前路径生效。
- old-log-prob 阶段已经请求 entropy，但那是无训练梯度的单独阶段，不代表训练阶段额外熵计算免费。

实施建议：

1. 为 EA-only 在现有 actor forward 内，以 no_grad、FP32 reduction、按 token chunk 计算完整词表熵；不增加独立模型 forward。
2. 对 dense `[B,S,V]` 与 packed `[N,V]` 都正确处理，分块沿 token 维，不是把 top-K 调成更大。
3. 正确恢复 token 位置，复用 `no_padding_2_padding` 的一位预测偏移和 response mask。
4. 首轮 chunk 先试 64/128，按 NPU 显存和吞吐测量，不宣称最优。它与 ROSE 的 CPU semantic chunk 是两套参数。
5. 不额外把全词表 logits/概率搬到 CPU，不将词表张量写入 TQ。
6. chunk 只降低新增 entropy 临时内存，不能消除模型原本 materialize logits 的开销。

## 6. 已修改文件与职责

| 文件 | 实施内容 |
| --- | --- |
| `verl/workers/config/actor.py` | 增加 typed EA config，默认 disabled，校验 coef/kappa/支持范围 |
| `verl/trainer/config/actor/actor.yaml` | 增加默认 EA 配置块，与 schema 保持一致 |
| `verl/trainer/config/_generated_ppo*.yaml` | 按生成脚本同步，不手改作为唯一来源 |
| `verl/trainer/ppo/v1/trainer_base.py` | EA 开启时请求 current entropy，携带 EA-only 无梯度计算元信息 |
| `verl/workers/utils/entropy_advantage.py` | 新增公式 helper、mask/数值检查、统计 |
| `verl/workers/utils/losses.py` | actor loss 前应用 EA；保持原优势只读，区分熵监控与正则 |
| `verl/workers/engine/fsdp/transformer_impl.py` | EA-only detached chunked entropy，优先覆盖当前 dense NPU 路径 |
| 测试文件 | 公式、梯度、mask、配置、dense/packed 对齐、回归 |
| `run_rose_ea.sh` | 基于 run_rose_new.sh，保留模型/数据/reward/集群与非 EA 超参数，新增 EA、独立输出和源码检查 |

正式配置（已通过本地 Hydra/typed schema 测试；NPU 运行待验证）：

```yaml
actor_rollout_ref:
  actor:
    entropy_advantage:
      enable: false
      coef: 0.1
      kappa: 2.0
      entropy_chunk_size: 64
```

熵来源首版固定 current_full_vocab，不同时加入多种近似模式。详细方案确定使用独立 EA chunk 配置，避免当前 dense 分支忽略旧开关的歧义；通过独立 EA 元信息让 engine 计算并返回 ea_entropy。正式接口见实施方案第 3-6 节。

脚本变量为 `EA_ENABLE`、`EA_COEF`、`EA_KAPPA`、`EA_ENTROPY_CHUNK_SIZE`，与 `ROSE_ALPHA` 明确分离。配置默认关闭，新脚本默认开启；关闭不请求额外 EA entropy，已有普通 entropy 路径不变。

## 7. 组合风险与实验判断

### 7.1 长度目标张力

EA 论文鼓励更长、更深入的推理；ROSE 长度校准降低较长正确路径的优势。EA 在校准后添加 bonus 可能抵消一部分正优势下调，但受 cap 约束且不会让校准后的负优势变正。

不能保证两者必然互补。第一版先固定原长度 alpha，只开关 EA；再做长度 alpha 与 EA coef 的二维小规模消融。不要一开始同时改变长度、KL、clip、batch 和 EA，导致无法归因。

### 7.2 零优势与动态过滤

树公共前缀上，一些 segment 的节点 value 相同，A=0，EA 不会给这些位置创造梯度。全同 reward 的组在当前 0/1 奖励下也没有新增 EA 信号。

保留现有 group filter 与失败补采设置；EA 不应将 entropy 当作 filter metric，否则会改变“有 reward 学习信号”的定义。

### 7.3 共享前缀

首版保持 ROSE 按 leaf 计算训练 loss 的既有计数。非零优势的共享 segment 上，EA bonus 也会随 leaf 重复计入。这不是新增独立 rollout，也不是自动去重。

按节点去重、节点平均熵、只对 pivot 或新生成 suffix 应用 EA，都可作为后续新变体，但会改变原公式/训练权重，不混入最小首版。

### 7.4 优势尺度与 cap 饱和

EA 论文的 GRPO 优势经过组内标准差归一化；当前 ROSE 使用节点 value 差，无该归一化。相同 coef 不保证相同作用强度。

从 0.1 开始小试，随后比较 0.05/0.2/0.4；若大部分 token 达到 cap，继续调高 coef 意义有限。需要统计 bonus/abs(A) 及 cap 命中率，而非只看整体熵。

### 7.5 EA 论文超参数不整体照搬

论文附录 B 使用 token-mean、clip low/high=0.2/0.28、无 KL loss、8K response、512 prompts、mini-batch 32 prompts；正文 EA GRPO coef=0.4、kappa=2。

本项目已有 ROSE baseline 使用不同 loss/clip/KL/长度。这些不是 EA 公式必需项。第一版保留基线；clip-higher/no-KL/8K 作为独立实验，不能为了添加 EA 一次全改。

## 8. 指标与验收

基础指标：验证 pass@1/pass@8、reward、截断率、平均/分位 response 长度、重复率、每步时间、actor update 时间、NPU 峰值显存、实际新增生成 token 数和训练消耗 token 数。

EA 专属指标：有效 response token 的 Shannon 熵均值、bonus 均值、非零 A 占比、bonus/abs(A)（只统计非零 A）、cap 命中率、正/负优势分别的均值与变化、优势符号翻转数（必须为 0）。

各卡/各 microbatch 指标应聚合 sum 和 count，而不是盲目平均均值；不要将固定长度归一化后的 `actor/entropy` 当成无条件的真实 token-mean 熵。

测试要求：

- disabled/coef=0 恢复基线，并且 disabled 不新增 entropy 请求。
- 正/负/零 A、cap 两侧、kappa>1、无效配置 fail closed。
- padding 为零，prompt 不参与，EOS 作为有效 response token 按 mask 处理。
- 不累加改写原 advantages；多 epoch 每次从基础优势重算，使用当次 current H。
- H 无梯度，policy logprob 有梯度；仅 EA 时不生成 entropy 的反向图。
- dense/packed 的 response 位移一致，chunked entropy 与全词表 FP32 参考一致。
- 不同 microbatch 切分下梯度和指标一致；有效 NaN/Inf 报错。
- NPU 先短序列单步，再真实长度 3-5 步，记录显存、耗时、bonus、输出可读性和截断率。

公平比较先做：ROSE vs ROSE+EA（其余相同）。随后可做 GRPO vs GRPO+EA、非零长度校准消融及更大 pass@K。比较相同步数之外，还要报告相同生成 token/壁钟预算的结果。

## 9. 首版不做

不修改 ROSE pivot 排名，不增加 root restart 的自适应调度，不对零优势加常数，不把 entropy 写入 reward，不切换成 PPO critic，不引入第二套 rollout 服务，不默认关闭 KL，不以固定随机种子宣称严格复现。

当前 ROSE 随机分支 seed 还混入随机 UUID；并发和 NPU kernel 也可能非确定性。小样本 A/B 的波动不能自动解释为 EA 效果，多种子/重复实验应作为效果确认的一部分。
