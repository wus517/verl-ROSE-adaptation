# ROSE + LATR：算法与 VeRL / Ascend 初步适配设计

状态：原始初步提案，首版 A 随后已实现并通过 CPU/mock 回归。当前分支 `rose-plus-latr`，基线 `e77dceb55b68416c3d153c3b36a0f13c815733dd`。实际实现/测试见 WORKLOG，NPU 实机未验证。

后续已整理 [详细实施方案](ROSE_LATR_IMPLEMENTATION_PLAN.md)，作为首版 A 的执行规格。本文保留论文依据和方案比较；关于预算不足不缩短窗口、短终止去重范围、seed、路由及观测口径的细节，以详细实施方案为准。原文“拟修改”保留历史语境，实际实现以 WORKLOG 和源码为准。

## 1. 目标与边界

目标是让 ROSE 在选出分支位置后，不再立即把该分支生成到完整 response，而是先生成短段，通过 lookahead 观察它与原路径是否真正发生分歧；重复的短段被拒绝，预算留给其他候选。最终仍向 VeRL 提供每个 prompt 的 G 条完整轨迹，并使用 ROSE 的分段优势。

初版保持：纯文本、single-turn、同步 V1 trainer、vLLM-Ascend、host CPU semantic scorer、最终二值 reward、独立随机验证。不引入额外神经网络。

必须区分：

- ROSE 解决“在哪个已有响应的位置分支”，并改变 credit assignment。
- LATR 的 lookahead 解决“新分支短程是否重复”，不是未来 reward 预测或正确性判断。
- 剪掉相似路径并不保证留下的路径更正确。
- 组合方法只继承部分 LATR 机制时，名称应明确为 ROSE + lookahead pruning，而非原样复现 LATR。

## 2. 论文依据

### 2.1 ROSE

来源：`paper/ROSE.pdf`，§3.1–3.3、§4.3。

- 先生成完整响应，再对响应各位置计算语义熵。
- 对全部已有响应选高语义熵位置，保留其前缀，重新生成后缀。
- 以 epsilon 概率从 prompt 根部重新采样。
- 直到得到 G 条完整响应；动态采样可过滤同 reward 的组。
- 节点 value 是经过节点的最终响应 reward 均值；相邻节点 value 差构成 segment advantage。
- alpha 控制长正确响应的优势下调。

### 2.2 LATR

来源：`paper/LATR.pdf`，题为 Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards，ICLR 2026；重点为 §3、Algorithm 1、Appendix B.3 / C.2 / C.4。

论文机制：

1. 维护在线生成的部分序列树；主路径先按最高概率 token 扩展。
2. 为候选 c 建立分支的条件为 `P(c) > tau_abs` 且 `P(c*) - P(c) < tau_rel`，c* 为最高概率 token。
3. 子分支与父分支继续生成短程后，在相应窗口内计算 token ID 序列的归一化 Levenshtein 距离。
4. 距离小于阈值则剪掉子分支及其后代。
5. 树宽达到目标后，转为普通随机续写；最终不足 k 时论文 Algorithm 1 规定 padding。
6. 训练还使用混合 rollout：`k_latr = round(eta * k)`，其余独立随机，`eta_i = eta_0 * gamma^i`。

论文 Appendix B.3 的参数：

| 项目 | 值 |
|---|---|
| 每 prompt 轨迹数 | 8 |
| tau_abs / tau_rel | 0.25 / 0.15 |
| 编辑距离阈值 | 0.4 |
| 多窗口 | 20、30、50，全部条件满足才保留 |
| eta_0 | 1.0 |
| gamma | Countdown 0.985，Math 0.995 |
| 训练采样 | temperature=1、top-p=1、top-k=-1 |
| 验证采样 | temperature=0.6、top-p=0.95、top-k=20 |

注意：这里 top-k=20 的验证采样限制，不是 ROSE 用来返回候选概率的 top-20 payload，二者不能混淆。

论文主文用单窗口描述，附录说明实验用多窗口。EOS、pad 的具体工程处理，以及部分边界的实现顺序，单靠论文不够明确；本地没有检查 LATR 官方代码，不捏造其具体行为。

### 2.3 不直接继承的论文结论

LATR 的在线树描述为 backtracking-free，给出 O(nk) 前向次数上界。下文建议的离线 ROSE 选点后重试，是另一种控制流程，不能直接继承该上界。

论文报告单步 rollout 约慢 10%，收益主要体现为达到相同效果所需 step 更少。该数字来自其 GPU / 模型 / 数据配置，不能用于承诺当前 NPU 上的耗时。

## 3. 当前项目的真实运行链路

```text
V1 trainer 提交 prompt group
  -> RoseAgentLoopManagerTQ / RoseAgentLoopWorkerTQBase
  -> build_prompt：正常 chat template，仅一次 tokenization
  -> _choose_branch：epsilon root restart 或 _select_branch 全局语义熵最大值
  -> generate_from_prefix：prompt IDs + 已有 response 前缀
  -> vLLM generate 完整剩余 response
  -> top-k payload + CPU semantic scoring
  -> 重复至 G 条完整轨迹
  -> finalize_tree -> 每个 leaf 一份 metadata
  -> 各 leaf 独立 reward / TQ postprocess
  -> 可选 DAPO 同奖励组过滤及新 prompt 补采
  -> rose_tree estimator -> PPO actor update
```

关键源码：

| 文件 | 当前职责 / 限制 |
|---|---|
| `verl/experimental/rose/rose_agent_loop_tq.py` | prompt 级树控制，树内完整响应依次生成；不同 prompt 可并发 |
| `verl/experimental/rose/rose_agent_loop.py` | token 前缀续写，生成 helper 目前总是覆盖 max_tokens 为完整剩余长度 |
| `verl/experimental/rose/semantic_entropy.py` | CPU top-k + embedding 语义熵，分块处理 |
| `verl/experimental/rose/tree.py` | 完整响应 branch records 转稳定节点路径；不是可剪枝的在线部分树 |
| `verl/experimental/rose/tree_advantage.py` | 严格 metadata 校验、叶子 reward 均值、segment advantage 与 alpha |
| `verl/workers/rollout/logprobs.py` | 后端概率字典打包，当前 payload 计算完评分即丢弃 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 返回 TokenOutput；目前将 finish_reason=stop 和 length 都映射为 completed |
| `verl/workers/rollout/llm_server.py` | client 路由、生成请求、policy version 元数据；每次调用是独立请求 |
| `verl/trainer/ppo/v1/agent_loop_tq.py` | 完整输出转 TQ；不得把试跑当作完整训练样本提交 |
| `verl/trainer/ppo/v1/trainer_base.py` | metadata 通路、old logprob 重算、指标、rollout 保存 |

现有 `_select_branch` 没有候选位置的失败记忆或多样性约束；新增 lookahead 后必须避免持续选择同一失败位置。

当前语义熵评分后丢弃整段 top-k；若以后强制候选首 token，需要保留少量候选位置的 token ID / 概率，而不是假设仍能读取所有 top-k。

## 4. 三种组合方式与推荐顺序

| 方案 | 内容 | 取舍 |
|---|---|---|
| A：rose_lookahead，推荐先做 | ROSE 选点；随机生成短段；父子距离剪枝；通过后续写 | 改动可控，不强制 token；不是完整 LATR |
| B：rose_latr_candidates，第二阶段 | A + LATR 概率门控 + 不同首 token 候选控制 | 更接近 LATR，需要概率和强制 token 契约审计 |
| C：完整在线 LATR + ROSE advantage | 每步在线建树/剪枝，另行引入 ROSE 选点或评分 | 需要重构在线状态机及真实树 metadata；不建议首轮实施 |

不建议同时更换 rollout、优势、loss、KL 和 batch size；无法归因。新增开关关闭时必须保持当前 ROSE 路径。

## 5. 推荐首版 A：具体流程

### 5.1 每个 prompt 的主循环

1. 第一条响应沿用原 ROSE，从根部完整生成并计算语义熵。
2. 为下一个最终轨迹槽位只抽一次 epsilon：
   - root restart：直接独立完整采样，不做父子剪枝。
   - 分支：从已有完整轨迹选择合法的高语义熵 pivot。
3. 复制父轨迹 `response_ids[:b]`，b 表示第 b 个 token 之前的前缀长度。
4. 在此 prefix 上随机生成最多 50 个新 token，不做 decode/re-tokenize。
5. 将试跑新 token 与父轨迹 `response_ids[b:b+r]` 比较，按各窗口计算归一化编辑距离。
6. 未通过：丢弃试跑，不 postprocess，不 reward，不分配最终 leaf ID；记录候选尝试失败。
7. 通过：以 `prefix + lookahead` 为续写前缀，生成剩余 response；试跑 token 和对应数据直接拼入最终响应，不从 b 重采。
8. 对新完整响应记录 branch event，评分并加入树，继续生成下一个最终轨迹。
9. 达到 G 条完整响应后，才运行 finalize_tree、reward 和 TQ 写入。

首版父轨迹已经完整存在，因此不必再次生成父分支 50 token。这比完整在线 LATR 更适合现有 ROSE 架构，但父轨迹来自随机生成，而非 LATR 搜索期的贪心路径。

### 5.2 相似性定义

首版提议：用同一分支点之后的对齐 token ID 段，定义：

```text
child_r  = child_response[b:b+r]
parent_r = parent_response[b:b+r]
d_r      = Levenshtein(child_r, parent_r) / r
```

对 r=20、30、50，要求 `d_r >= 0.4` 全部成立才保留。举例：长度 50 的窗口只有 5 次编辑，d=0.1，视为重复；25 次编辑得到 d=0.5，通过该窗口。

这里以“从分支点起前 r 个新 token”为明确融合契约。论文按分支年龄与最近 r token 写法描述，多窗口的精确 off-by-one 还需与官方实现核对，不能声称当前定义逐 token 等价。

必须排除：prompt、共享 prefix、padding；不使用整段 response 做比较，否则公共前缀会淹没分支差异。

初版只比较父子。与所有同前缀兄弟比较可以作为后续更严格的去重消融，不默认启用，以免成本和选择偏差同时增加。

### 5.3 有界搜索与回退

建议调试起点：

- 单槽位每个 pivot 最多 2 次随机试跑。
- 单槽位最多尝试 4 个不同 prefix 的 pivot。
- group 全部试跑 token 预算初始 700，包含通过和拒绝的 probe；每次提交前预留完整最大窗口，预算不足则回退，不缩短窗口绕过检验。
- 达到任一硬上限后，该槽位从根部独立完整采样；这是本融合方法的工程选择，不是 LATR 的 padding 原文。

候选身份应用 prefix token 指纹 + b 标记，不能只用轨迹编号/position，因为同一 prefix 可能在多个已有轨迹中重复出现。仅对单次 prompt group 有效；固定 seed 时 tie-break 必须稳定。

无合法 pivot / 没有足够剩余长度 / 连续拒绝时，都要有明确回退原因，不能无限 while 重试。

硬预算只约束剪枝带来的尝试，不会终止已通过分支的合法完整续写；总 wall-clock 还受完整 rollout、DAPO 补采和失败重试影响。后端异常按失败处理，不伪装成相似度拒绝或悄悄复用别人的响应。

### 5.4 EOS 与短后缀

提案：

- 父后缀不足最大窗口时暂不选作需要完整多窗口检查的 pivot，转下一个候选。
- 子试跑提前 EOS：它已经是完整响应，不应为了凑 50 token 越过 EOS 续写。除全同重复外，先保留并标记 `short_terminal`，不谎称通过完整多窗口检验。
- 短终止规则是本地设计，需独立统计并讨论是否改成可用窗口规则；此处不声称是 LATR 论文规定。
- 达到 response token 预算但没有 EOS 的完整最终轨迹仍按原项目截断语义处理，保留 truncation 标志。
- 试跑因 50-token cap 停止不是最终轨迹截断，不能误报为完整 response 的 length clip。

## 6. 分段生成的数据契约：正确性优先

### 6.1 不改变训练 prompt

最终 `AgentLoopOutput.prompt_ids` 仍仅为原始用户 prompt。试跑 token 属于 response，不能因为下一次请求将其作为输入 prefix，就从训练 response 中消失。

```text
final_response_ids = inherited_prefix_ids + accepted_probe_ids + completion_ids
final_logprobs     = inherited_prefix_logprobs + probe_logprobs + completion_logprobs
final_entropy      = inherited_prefix_entropy + probe_entropy + completion_entropy
```

三个数组的 token 长度须严格一致。试跑只计算一次熵，续写后复用；失败试跑只进入调试与成本统计。

分段上限不计入多个独立 response budget：最终总 response 长度始终不超过 `rollout.response_length`。

### 6.2 helper 和服务端必须改的点

- 提取分段生成 helper，例如 `generate_segment(..., max_new_tokens)`，从配置剩余预算与调用方段长共同取 min；旧的完整续写调用保持兼容。
- 保留后端原始 `finish_reason` / `stop_reason` 于 extra_fields，至少能区分 EOS/stop、length-cap、abort。
- 不只靠“返回 token 数是否等于 50”判断 EOS：EOS 可能恰好落在最后一个 token；停止字符串也可能影响最后返回 token。
- 每段检查 `global_steps/min_global_steps/max_global_steps`；最终与复制的 prefix 所属版本一致。
- 构造最终输出时汇总所有段的耗时、preemption、缓存命中；不能仅保留最后一个请求的 metrics。
- 后端 request ID 每段唯一；首版正确性不依赖 cache / affinity。稳定 group affinity 留待实测 router 的 acquire/release 后优化，不可把 request ID 相同误当成真正 KV 恢复。

### 6.3 树 metadata

首版只有完整保留轨迹可以成为未来轨迹的 parent。失败 probe 从未生出已提交子树，所以没有递归删除后代的问题，保留轨迹可以继续使用现有 `BranchRecord/finalize_tree`。

通过后续写的原始 branch_pos 仍是 b，不是 b+50；试跑边界只是执行分段，不一定是 ROSE credit-assignment pivot，不能自动造一个新分支节点。

最终只向 estimator 提交 G 个保留 leaf；被拒绝试跑不得进入 value 的 reward 均值。parent ID 必须指向仍保留的完整轨迹。

若转向方案 C，剪枝可能删除已有父节点，必须另做 alive-leaf 投影/祖先保留，不能直接套用上述限制。

## 7. 第二阶段 B：候选 token 门控

在 ROSE 选定的 pivot 上，保留 top-k 概率候选，使用 LATR 条件：

```text
p(candidate) > 0.25
p(top1) - p(candidate) < 0.15
```

但与原 LATR 的贪心父路径不同，ROSE 的父 token 可能不是 top1。必须明确排除已走过的父 token；是否允许 top1 成为替代候选，作为显式 hybrid 规则记录，而不能直接照搬“排除 top1”。

候选门控后的低接受率不应通过把 top-20 概率重新归一化解决。top-k 是截断视图，概率应保留全分布口径；temperature/top-p/processed_logprobs 的语义须与目标 vLLM-Ascend 实测一致。

强制首 token 有额外契约：

- 该 token 必须属于 response，response mask 通常仍为 1。
- 不可因为后端约束只允许一个 token，就把它的原策略 logprob 写成 0。
- 可用同 prefix 下缓存的原始候选 logprob，但必须确认它和其他位置使用相同分布定义。
- 采样分布 q 被强制候选和剪枝改变；保存原策略 logprob 不等于完成行为分布的重要性校正。
- 当前默认 actor 重算 old logprob 可用，但它不消除搜索选择偏差；也不能直接打开现有 rollout correction 就声称精确校正。
- 不把强制首 token 的 mask 随意改成 0；那会变成新的 loss 定义，且需检查 response length / tree endpoints 一致性。

推荐以 A 为对照，再单独加概率门控与首 token 控制。没有合格 token 时转下一个 pivot，最终按预算回退。

## 8. eta 与 epsilon、DAPO 的关系

- epsilon：ROSE 每个最终响应槽位是否从 root 独立采样。
- eta：LATR 分配给树搜索的轨迹比例，并随 step 衰减。
- DAPO group filtering：G 条最终轨迹获得 reward 后，同 reward 组被移除并换新 prompt。

三者处于不同层级，不能把 eta 简单称为 `1-epsilon` 或同时叠加后不记录实际比例。

首版固定 epsilon，eta 不启用。后续可显式预留独立采样配额，再给分支槽位应用 lookahead；这属于 hybrid schedule 实验，要看当前总 step 数，不盲搬 500-step 实验的 gamma。

调试先固定同一批 prompt，关闭 DAPO 过滤或仅运行 rollout：便于看到原始接受率，不被新 prompt 补采掩盖。正式对照中 ROSE 与融合方法使用同样的 DAPO 开关。

当前 ROSE estimator 默认要求二值 reward。不能直接把 LATR Countdown 的 0.1 格式奖励复制进来；数学场景沿用现有 0/1 verifier。

## 9. Ascend 工程与成本

模型生成、embedding/attention 仍在 NPU；host 只做现有语义熵、几十 token 的编辑距离与树控制，不增加 CPU 神经网络推理。

主要耗时风险不是编辑距离，而是：

1. 每条保留 branch 从一次生成变成 probe+completion 两次 RPC。
2. 独立请求可能重复 prefill，共享 token prefix 不保证 KV cache 命中。
3. 被拒绝试跑、多个 pivot 尝试增加生成 token。
4. 保留随机父路径不保证试跑更有分歧，可能触发很多拒绝。
5. group filtering 会在其外再叠加新 prompt 补采成本。

当前 `run_rose_new.sh` 禁用 prefix caching；不能在没有目标 NPU/Qwen3.5 长前缀正确性测试时直接强制开启，尤其需检查模型架构/混合注意力支持。

先保持随机候选一条条试跑。需要优化时再在同 pivot 下做小并发候选，例如 2 个，并采用稳定选择；全部候选的成本都计入预算。不要把整棵树的 G 条同时独立生成，这会丢失 ROSE 顺序选点依赖。

## 10. 拟修改文件（未实施）

| 路径 | 计划 |
|---|---|
| `verl/workers/config/rollout.py` | 新增受校验的 lookahead 子配置，默认关闭 |
| `verl/trainer/config/rollout/rollout.yaml` | 同步默认配置，避免临时 `++` 绕开 schema |
| `verl/experimental/rose/lookahead.py`（新） | 纯 CPU 编辑距离、多窗口、候选去重、预算与拒绝记录 |
| `verl/experimental/rose/rose_agent_loop.py` | 分段生成与数据拼接，保持旧路径兼容 |
| `verl/experimental/rose/rose_agent_loop_tq.py` | probe/accept/reject/complete 控制，最终 G 叶子与 metadata |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 保留真实停止原因；B 阶段再考虑候选 logprob 通路 |
| `verl/trainer/ppo/v1/trainer_base.py` | 补充融合指标聚合和调试保存；不重写 trainer 主训练循环 |
| `tests/experimental/rose/` | CPU 距离、预算、mock rollout、metadata 和概率契约测试 |
| 根目录新启动脚本（名称待定） | 参考当前服务器脚本，仅增加融合开关和参数；不覆盖现用脚本 |

首版预期复用、不改算法公式：`tree.py`、`tree_advantage.py`、verifier、PPO loss。复用前后要跑其已有回归测试。

拟配置（只是提案，当前尚无这些 Hydra 字段）：

```yaml
actor_rollout_ref:
  rollout:
    rose:
      lookahead:
        enable: false
        mode: sampled_probe
        windows: [20, 30, 50]
        edit_distance_threshold: 0.4
        max_trials_per_pivot: 2
        max_pivots_per_slot: 4
        max_probe_tokens_per_group: 700
        fallback: root_restart
```

## 11. 观测与可复现性

建议 group 级记录：probe attempts、accepted / rejected、short terminals、不同 fallback 原因、各窗口距离、probe 总 token、被拒绝 token、保留 token、RPC 耗时、completion 耗时、cache hit、policy version、原始 stop reason。

每个 group 的统计只能聚合一次，不能把同一份总成本复制到 G 个叶子后再相加。最终 rollout JSONL 仍保存完整响应，另设有界 debug trace 保存被拒绝短段及原因；不要默认打印每个 top-k/所有 prompt 造成日志放大。

观察训练：reward 组方差、DAPO 过滤率、独立答案数、验证 pass@1/pass@8、平均长度、KL、clip fraction、生成 token/秒、step wall-clock 与达到同验证指标的总耗时。

当前 ROSE seed 混入随机 UUID；固定 `rollout.seed` 不保证跨运行相同分支。正式对照还需定义稳定 sample ID + dataset pass/visit + step/slot/trial 的 seed 派生，区分逻辑 group UID 和算法 RNG key。不在首轮文档任务中改代码。

## 12. 分阶段验证与停止条件

### 阶段 0：本轮文档

只完成源码/论文核查与方案讨论，不运行训练，不宣称 NPU 验证。

### 阶段 1：纯 CPU 和 mock server

- 距离 0/1、阈值边界、多窗口、空/短/EOS。
- 拒绝后仍有硬上限；候选重复 prefix 不重复无限尝试。
- 通过后 probe token/logprob/entropy 只拼一次；长度预算正确。
- 真实 EOS 不续写；length cap 可续写；abort 不提交训练 leaf。
- 被拒绝 probe 不计 reward、不改变最终树 node mean。
- G 个完整 leaf 且 parent 有效，policy version 一致。
- 关闭开关时与旧 ROSE 控制流程一致。

### 阶段 2：服务器 NPU smoke

固定小批 prompt，G=2，检查 probe、拼接 response 与后端停止原因；先确认纯 rollout 正常，再完成 1 个 actor update；最后增至 G=8 和生产 TP / 两节点。

### 阶段 3：消融与性能

- baseline ROSE。
- A：随机 probe + lookahead pruning。
- B：A + 概率门控/候选首 token（独立实验）。
- 可选后续：eta 衰减、兄弟间距离、alpha 开关。

除 step 数比较外，至少记录总生成 token 与 wall-clock；同一数据、模型、reward、DAPO 开关和其他训练超参。收益未知，短程 token 差异只是代理指标，不能当作语义或正确性的充分证明。

如果拒绝率或回退率持续接近 100%，先停大规模训练，检查概率口径、窗口规则、模型及数据差异，而不是无上限补采。若吞吐损失超过可接受范围，应先做预算/缓存/候选并发 profiling，再决定是否投入在线树重构。
