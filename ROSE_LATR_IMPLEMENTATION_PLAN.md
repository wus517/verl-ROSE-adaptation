# ROSE + LATR lookahead：详细适配与实施方案

> 状态更新：首版 A 已实现并通过本地 CPU/mock 回归；NPU 实机与训练效果待验证。本文保留原实施规格措辞，实际接口、差异和测试见 WORKLOG，运行见 RUN_GUIDE。
>
> 适用分支：`rose-plus-latr`；核对基线：`e77dceb55b68416c3d153c3b36a0f13c815733dd`。
>
> 本文将前面的讨论收敛成首版实施规则。下一轮获得实施授权后，先读本文，再核对工作区与源码，按第 15 节执行。本文中的代码、配置、指标和测试文件若标记为“拟新增”，当前还不能直接运行。

## 0. 文档优先级与交接规则

- 本文：首版 A 的实施规格、接口约束、测试与交付清单。
- [ROSE_LATR_DESIGN.md](ROSE_LATR_DESIGN.md)：论文依据、备选方案、融合方法的局限。
- [ROSE_LATR_WORKLOG.md](ROSE_LATR_WORKLOG.md)：决策变化、实际修改、真实测试结果。
- [ROSE_LATR_PROJECT.md](ROSE_LATR_PROJECT.md)：项目入口与当前进度。

旧设计中的“待讨论”默认值，由本文第 2 节收敛为下一轮可实施的起点；若用户另有要求，以最新明确要求为准，并同步修改文档。源码事实若变化，应重新核对，不能机械照搬旧行号。

最初版本只编写文档；随后已按用户指令实施首版。下文“拟新增”是原规格描述，不代表当前缺失；实际完成状态以第 17 节及 WORKLOG 为准。不得将 mock 通过写成 NPU 通过。

## 1. 一句话定义：我们究竟实现什么

保留 ROSE 的“完整父轨迹 → 全局高语义熵选点 → 分支 → 树分段优势”，在分支生成中插入 LATR 风格的短程多窗口 token 编辑距离筛选。

首版命名为 **ROSE + lookahead pruning（方案 A）**，不宣称完整复现 LATR。

```text
原 ROSE：选定父轨迹和分支点 b → 一次生成完整后缀 → 加入最终树
方案 A：选定父轨迹和分支点 b → 生成最多 50 token 的 probe
        → 与父后缀比较
          → 通过：保留 probe，继续生成剩余后缀，加入最终树
          → 拒绝：不训练、不算 reward；有界重试，最终从根部回退
```

“lookahead”检查的是短程分歧，不是未来 reward，不访问正确答案，也不保证答案更好。它可能增加探索，也可能增加耗时或引入不利选择偏差，必须实验验证。

### 1.1 明确不是逐 token RPC

一次 `vLLM.generate(max_tokens=50)` 内部自回归生成至多 50 个 token，返回后在 host CPU 比较窗口。通过后，最多再调用一次 vLLM 续写至最终上限。

首版不做 `max_tokens=1` 的外部循环，不改 vLLM / vLLM-Ascend 的逐 token scheduler，不接管 attention 或 NPU kernel。

多窗口检查是返回 50 token 后同时检查前 20 / 30 / 50 token；不是生成到第 20 token 就实时取消。不提前终止意味着可能浪费本来在 20-token 窗口即可拒绝的后续 token，这是首版降低集成复杂度的明确取舍。

## 2. 首版范围和默认决策

| 事项 | 首版规则 |
|---|---|
| rollout 环境 | 当前纯文本、single-turn、同步 V1 trainer、vLLM-Ascend |
| 第一个最终响应 | 原 ROSE 的根部完整采样 |
| 后续槽位 | 每槽位只抽一次 epsilon；根部独立或进行 lookahead 分支 |
| 分支位置 | 已保留完整轨迹上的全局最高合法语义熵位置 |
| probe 采样 | 与当前训练 sampling 参数一致；不强制首 token |
| 多窗口 | `[20, 30, 50]` |
| 接受阈值 | 每个窗口归一化编辑距离均 `>= 0.4` |
| 每 pivot 重试 | 每槽位最多 2 次 |
| 每槽位 pivot 数 | 最多 4 个不同前缀 |
| 每 group probe 预算 | 最多 700 个生成 token，包含通过和拒绝的 probe |
| 搜索失败 | 为当前槽位从根部独立完整采样 |
| 短 EOS / stop | 非空且不是已有完整响应的完全重复时保留，单独标记 |
| 最终输出 | 恰好 G 条合法最终响应；允许原有长度截断，不用重复响应凑数 |
| 验证集 | 继续走现有 independent 路径，不做 lookahead |
| 优势 / loss | 复用 `rose_tree`、alpha、既有 PPO loss |
| 默认开关 | 通用配置 `enable=false`；新增专用启动脚本显式开启 |
| 概率门控、强制首 token | 不在首版实现 |
| eta 衰减 / 兄弟间多窗口比较 | 不在首版实现 |
| prefix caching | 保留现有设置；不未经 NPU 正确性测试直接打开 |

以上预算是保守实验起点，不是论文规定，也不是最优值。700 限制的是全部 probe 成本，不等于“最多浪费 700 token”；根部生成、接受后的 completion、DAPO 补采均另计。

## 3. 源码基线：从哪里插入

下列位置按符号定位，不把当前行号作为长期接口。

| 文件 / 符号 | 已核实的现状 | 适配要求 |
|---|---|---|
| `verl/experimental/rose/rose_agent_loop_tq.py::_run_prompt` | G 次循环，每次产生完整轨迹后才参与下次选点 | 在训练分支插入槽位级搜索状态机 |
| 同文件 `_select_branch` / `_choose_branch` | 全局最大语义熵，没有失败 pivot 记忆 | 保留原函数用于关闭路径；新增合法候选选择逻辑 |
| `verl/experimental/rose/rose_agent_loop.py::generate_from_prefix` | 总将 `max_tokens` 设成完整剩余 response 长度 | 抽取分段 helper，旧接口作为兼容包装 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py::generate` | `stop`、`length` 都映射成 `completed` | 增加原始结束原因元数据，不破坏既有 `stop_reason` |
| `verl/workers/rollout/llm_server.py::LLMServerClient.generate` | 同步 client 直接返回 output，并添加 policy version 范围 | 确认新增 extra_fields 可透传；正确性首版无需改 client |
| 同文件 `FullyAsyncLLMServerClient` | 会重建 output / 合并 partial rollout | ROSE 明确拒绝此 client；首版不扩展 fully async |
| `verl/experimental/rose/tree.py::finalize_tree` | 输入为按顺序生成的完整 branch records | 只传最终保留轨迹；不改 schema |
| `verl/experimental/rose/tree_advantage.py` | 默认二值 reward，严格检查祖先/分段 | 不改公式；增加组合回归测试 |
| `verl/trainer/ppo/v1/trainer_base.py::_compute_rose_rollout_metrics` | 当前按最终 leaf 的 extra_fields 聚合 | 增加 lookahead 统计，避免 G 倍重复 group 成本 |
| 同文件 `_log_rollout_data` | 当前保存 prompt、最终 response、reward、uid | 如添加调试摘要，需要显式取出并排序 extra_fields，不能假定自动落盘 |
| `verl/workers/config/rollout.py::RoseRolloutConfig` | 没有 lookahead 子配置 | 新增受校验 dataclass |
| `verl/trainer/config/rollout/rollout.yaml::rose` | 没有 lookahead 子配置 | 新增完整默认值和 `_target_` |

不要把历史调试日志变化、服务器脚本改动当成当前本地已实现内容。实施前重新运行 `git status --short`，保留用户已有修改及 `paper/LATR.pdf`。

## 4. 配置设计

### 4.1 拟新增 dataclass

在 `verl/workers/config/rollout.py` 的 `RoseRolloutConfig` 之前定义 `RoseLookaheadConfig(BaseConfig)`；加入该文件 `__all__`，使 `verl.workers.config.RoseLookaheadConfig` 可导入。

`RoseRolloutConfig` 增加：

```python
lookahead: RoseLookaheadConfig = field(default_factory=RoseLookaheadConfig)
```

不运行中重写配置，不使用 `object.__setattr__` 修改搜索计数。计数和预算放在每个 group 的独立状态对象中，避免 BaseConfig 冻结约束及不同 prompt 串状态。

### 4.2 拟新增 YAML

```yaml
actor_rollout_ref:
  rollout:
    rose:
      lookahead:
        _target_: verl.workers.config.RoseLookaheadConfig
        enable: false
        mode: sampled_probe
        windows: [20, 30, 50]
        edit_distance_threshold: 0.4
        max_trials_per_pivot: 2
        max_pivots_per_slot: 4
        max_probe_tokens_per_group: 700
        fallback: root_restart
        short_terminal_policy: keep_nonduplicate
        request_timeout_seconds: 600.0
        trace_dir: null
        trace_max_events_per_group: 64
        trace_include_token_ids: false
```

实际 YAML 写在 `rollout.yaml` 的 `rose.lookahead` 下，上面的完整层级仅用于说明。不要把整个 `actor_rollout_ref` 再嵌进 `rollout.yaml`。

`windows` 使用 `field(default_factory=lambda: [20, 30, 50])` 或同等无共享可变默认值的定义；不要直接写共享 list 默认值。

### 4.3 校验规则

- `mode` 只接受 `sampled_probe`；不先放入尚未实现的模式名字。
- windows 非空、严格递增、无重复，元素为正整数；不接受 bool 冒充 int。
- threshold 为有限数且处于 `[0,1]`，不能让 NaN 绕过校验。
- 两项次数上限必须为正整数。
- probe 预算为非负整数；0 表示不做 probe，分支槽位直接回退。
- 预算不足 `max(windows)` 时不缩短完整窗口，直接回退，不能伪称做过完整检查。
- fallback 只接受 `root_restart`；short_terminal_policy 只接受 `keep_nonduplicate`。
- timeout 为有限正数，初始 600 秒是故障护栏，不是吞吐目标；服务器首轮 profiling 后可调。
- trace 事件上限为正整数，token ID 明细默认关闭。
- 启用 lookahead 但未启用 ROSE 时尽早报配置错误，不静默无效。
- 若 `response_length < max(windows)`，启动时明确警告所有普通分支将回退；仍允许短 response smoke，不能循环等待合法 pivot。
- 继续沿用现有 text-only / synchronous / vLLM / binary reward 限制。

配置 dataclass 和 Hydra 实例化都要测试，不能只证明直接 `RoseLookaheadConfig(...)` 能创建。

## 5. 核心纯函数与状态对象

新建 `verl/experimental/rose/lookahead.py`。该文件仅依赖标准库和必要的 NumPy，不导入 Ray、vLLM、torch_npu，不加载模型。

建议结构如下，名字允许小范围调整，但行为必须与本文一致：

| 拟定义对象 | 职责 |
|---|---|
| `levenshtein_distance(left, right)` | token ID 序列编辑距离；插入、删除、替换各成本 1 |
| `evaluate_probe(parent_suffix, probe_ids, windows, threshold)` | 返回接受/拒绝、各窗口距离和原因；不 decode |
| `PivotCandidate` | parent ID、b、score、prefix key |
| `rank_legal_pivots(...)` | 合法位置过滤、相同 prefix 去重、稳定排序 |
| `ProbeBudget` | group 总预算、预留/结算、使用 token 数 |
| `SlotSearchState` | 本槽位访问 pivot、每 pivot 尝试数、拒绝结果 |
| `LookaheadGroupStats` | 最终保留与搜索开销的计数/时间，不放入 tree schema |

### 5.1 距离定义

对每个窗口 r：

```python
parent_r = parent_response_ids[b : b + r]
child_r = probe_ids[:r]
distance_r = levenshtein_distance(child_r, parent_r) / r
accept = all(distance_r >= threshold for r in windows)
```

这是 token ID 编辑距离，不是 logprob 距离、embedding 距离或字符串字符距离。只比较 pivot 后的新 token，不包含 prompt / 公共 prefix / padding。

相等阈值保留，例如 r=20、编辑次数=8 时 d=0.4，通过此窗口；但仍须通过其余窗口。采用普通浮点比较并用边界测试锁定行为，不自行加未记录的 epsilon 改变阈值。

用滚动行动态规划，空间 O(r)，默认最大窗口 50 的 CPU 成本很小。可以复用一次 DP 的相关对角值或分别计算各窗口；先保证正确性，不新增 native 编辑距离依赖。

### 5.2 合法 pivot 与去重

位置 b 代表“复制父 response 的前 b 个 token，从第 b 个 token 重新采样”，b=0 合法，且仍然是有 parent 的分支，不等于 epsilon root restart。

候选必须满足：

1. 来自已保留的完整轨迹，且 score 有限。
2. `len(parent.response_ids) - b >= R`，其中 `R=max(windows)`。
3. `response_length - b >= R`。
4. 当前槽位对该 prefix 的尝试次数未耗尽。

按 score 降序，平分按 parent ID、b 升序，保持确定 tie-break。相同 token prefix+b 去重；取其中最高 score 的代表，平分取较早 parent。

prefix key 应稳定 hash 整数 token ID 序列和 b，不使用 Python 的进程随机 hash；可先用 digest 加速，必要时用原 prefix 精确核对。大量位置的 prefix 指纹可增量计算，避免对每个位置重复拷贝整段导致二次开销。

每槽位维护拒绝/次数状态；新槽位重新开始，不能无意把单槽位两次上限变成整棵树两次上限。group token 总预算则跨所有槽位共享。

同一 pivot 两次失败后转下一合法 pivot；最多访问 4 个不同 pivot。不同已有轨迹中相同 prefix 不算新的 pivot，防止共享前缀绕过重试上限。

## 6. 分段生成接口与输出契约

### 6.1 抽取 `generate_segment`

在 `rose_agent_loop.py` 提取不含前缀拼接的低层 helper，拟接口：

```python
async def generate_segment(
    *, request_id, prompt_ids, response_prefix_ids,
    sampling_params, max_new_tokens, priority=0,
) -> GenerationSegment:
    ...
```

`GenerationSegment` 是内部 dataclass，至少携带：

- 本段新生成的 `token_ids`、逐 token `log_probs`。
- packed top-k 三数组；形状 `[本段长度, K]`。
- `finish_reason`、后端 stop detail、实际 max_tokens、输入 token 数。
- policy version 范围、生成耗时、preemption、cache 命中（可为未知）。
- 其余必须保留的 server extra_fields。

`max_new_tokens` 为显式正整数，与 response 剩余预算取 min：

```text
remaining_response = response_length - len(response_prefix_ids)
requested = min(max_new_tokens, remaining_response)
server_input = original_prompt_ids + response_prefix_ids
```

`requested<=0` 时不调用 server，调用方应已将轨迹判为最终预算耗尽。不能把 0 传后端后被 clamp 成 1，产生越界 token。

服务端还会按模型上下文空间 clamp；必须返回实际采用的 `max_tokens` / context 空间，供上层区分“probe cap”与“上下文已经耗尽”。

helper 复制 sampling 参数后再设置长度，不修改调用者 dict；校验 sampled logprob 与全部 top-k 数组第一维、形状一致。缺失概率、非整数 token、异常长度都报错，不填 0、不静默截断掩盖协议错误。

原 `generate_from_prefix(...)` 保留原签名，内部调用 segment helper，段长用完整剩余长度，再构造原来的返回 tuple。关闭 lookahead 时，仍调用该兼容接口，保持原行为。

### 6.2 服务端停止信息：增加字段，不改变全局老语义

`vllm_async_server.py::generate` 的正常输出和空 abort 输出均携带统一字段：

```text
extra_fields.rollout_finish_reason       # stop / length / abort / unknown
extra_fields.rollout_stop_detail         # vLLM output.stop_reason，可能是 token ID / str / None
extra_fields.rollout_effective_max_tokens
extra_fields.rollout_input_tokens
extra_fields.rollout_context_limit
```

沿用现有顶层 `TokenOutput.stop_reason` 映射，避免普通 GRPO / multi-turn 调用者发生全局回归。ROSE 新 helper 从这些字段读取真实语义。

`finish_reason=stop` 是 terminal，可能由 EOS、stop token 或 stop string 导致；不能都叫“已看到 EOS token”。记录为自然停止即可。必须覆盖恰好第 50 个 token 为 EOS 的测试。

停止判定优先级：

1. abort / 超时 / 后端异常 → 当前 group 失败，不算相似度拒绝。
2. `stop` → terminal，不再续写。
3. `length` 且 response 或模型 context 预算耗尽 → 最终截断响应，不再请求。
4. `length` 且只是 probe 的 R-token cap → 可检查并继续。
5. 未知原因、length 却意外不足请求上限且未耗尽 context → fail closed，记录完整协议诊断。

lookahead 开启时缺失原始 reason 必须报错，不用“长度等于 50”猜测。旧路径不得因此要求现有所有后端和旧 mock 都立刻提供新字段；测试需要区分兼容包装与 lookahead 严格模式。

### 6.3 最终拼接：probe 只能出现一次

设父前缀为 P、通过 probe 为 Q、completion 为 C：

```text
final.prompt_ids         = 原始 prompt_ids
final.response_ids       = P.ids + Q.ids + C.ids
final.response_logprobs  = P.logprobs + Q.logprobs + C.logprobs
final.semantic_entropy   = P.entropy + Q.entropy + C.entropy
final.response_mask      = 全部 response token 为 1
final.branch_pos         = len(P) = b
```

completion 请求的模型输入是 `原始 prompt + P + Q`，最大生成数为 `response_length-b-len(Q)`；不能从 b 重新生成 Q，也不能把 Q 当成最终训练 prompt。

实例：response 上限 3072，b=1000，probe=50，则 completion 最多 2022 token。最终长度最多 `1000+50+2022=3072`，不是每段各给 3072。

probe 通过前不做语义熵 scorer：拒绝时直接释放 top-k，节省 CPU 和内存。通过后先评分 Q 并释放 Q 的 packed top-k，再请求 completion；完成后评分 C。每个保留新 token 恰好评分一次，prefix entropy 直接继承。

返回完整 AgentLoopOutput 前断言 ids/logprobs/mask/entropy 等长；top-k 有效候选少于 2 的位置仍设为 `-inf`，沿用现有分支选点约束。

### 6.4 策略版本与额外数组

对每段都检查版本，且与 expected policy version、父 prefix 版本一致；汇总 min/max 不能只用 completion 的最后版本。拒绝 probe 如果返回错误版本，也必须判失败，不允许当普通拒绝继续。

保留已有 `_validate_policy_versions` 做最终 group 再检查。不要因同步模式“理论上不会换权重”就删除它。

现有 helper 未提供完整的跨 prefix routed-experts 拼接契约。首版如果开启 `enable_rollout_routing_replay`，应在 lookahead 启动时明确拒绝，除非另行补齐并通过相关测试；不要悄悄丢弃该数组。标准当前 Qwen3.5-9B 配置不应因此新增 routing replay。

## 7. 槽位状态机：如何避免无限重试

训练 `_run_prompt` 中保留 validation 的最前面 bypass，再按开关选择旧 ROSE 或新控制流。

```text
for slot in range(G):
    if slot == 0:
        生成根部完整响应；保留
        continue

    只抽一次 epsilon
    if root_restart:
        生成根部完整响应；保留
        continue

    accepted = false
    建立本槽位候选 / 尝试状态
    依次尝试最多 4 个不同合法 pivot:
        每个 pivot 最多 2 次:
            若 group 预算不足 R: 退出搜索
            预留 R token
            生成一段 max_tokens=R 的 probe
            按实际生成数结算预算；检查版本 / 停止原因
            若有效短 terminal: 按短终止规则处理
            否则检查所有窗口
            拒绝: 只计成本，不 reward、不提交 leaf；继续
            接受: 保留 probe，必要时 completion；形成完整响应
                  记录 parent 和原始 b；退出搜索
    if not accepted:
        从根部完整采样填充该槽位

    只将本槽位最终完整响应加入 trajectories

全组一致性检查 → finalize_tree → 每 leaf reward / TQ postprocess
```

实际实现可以抽成 `_generate_lookahead_trajectory`，避免 `_run_prompt` 膨胀成难测试的长函数。epsilon 随机决策、候选选择、执行生成、结果提交应能分别 mock。

### 7.1 预算细则

- 每次请求前必须至少可预留 R；预算不足时，不生成 17 token 然后忽略 20/30/50 窗口。
- 单 prompt 内首版串行尝试，无须跨 prompt 锁；不同 prompt 各有自己的 700-token 预算。
- 生成成功后计实际 probe token 数，未使用的预留额度归还。
- 通过 probe 也消耗预算，不能通过后退款，导致无界请求。
- backend / timeout 失败即停止 group，不继续试下个 pivot；失败期间已消耗 token 不得在全 workload 报表中说成 0。
- max trial 与 max pivot 同时生效；预算充足也不能无限重试。
- group 预算耗尽后，剩余分支槽位都 root fallback，但 epsilon 决策仍单独记录。

### 7.2 回退与失败不是一回事

正常 fallback 原因：`no_legal_pivot`、`probe_budget_exhausted`、`pivot_trials_exhausted`。一次从根部完整生成成功后，该槽位照常保留，metadata 为 `parent=None, branch_pos=0, root_restart=True`，并另外记录 fallback reason。

异常失败原因：server error、abort、timeout、token/logprob 形状错误、policy version 错误、unknown finish reason。不要用 root fallback 掩盖基础设施或数据契约错误；沿用现有 group failure / trainer refill 通路。

请求 timeout 用异步有界等待，不额外做训练层重试。取消 Ray await 不保证后端 vLLM 请求立即被取消，日志必须注明；若后续实现 request abort，必须使用真实 backend request ID 并验证接口，不能凭空调用。

### 7.3 短终止与重复

父后缀不足 R 的 pivot 不选。子 probe 可能提前自然停止，首版规则：

- `0 < len(probe) < R` 且 finish=stop：形成 `prefix+probe` 的完整响应。
- 与已有任意完整 response 的 token IDs 完全相同时拒绝，计入尝试与预算。
- 否则保留，标记 `short_terminal`，不得报告为 full-window pass。
- 恰好长度 R 但 stop：可检查全部窗口；通过即直接保留，无 completion。
- 上下文/response 耗尽导致的短 length 与自然 stop 分开标记，不使用短 EOS 例外伪装接受。正常合法 pivot 应有至少 R 的预算；模型 context 不足是协议/配置诊断点。
- 根部正常采样和 fallback 允许抽到重复完整答案，不加去重循环；否则会偷偷改变基线分布并失去硬上限。

这一短终止例外是融合方案的工程定义，不声称是 LATR 官方实现。单独统计它，避免全部分支靠提前 EOS 绕过距离判据却误报高接受率。

## 8. 最终树、reward 和 PPO 如何保持不变

拒绝 probe 不分配最终 trajectory ID、不创建 BranchRecord、不运行 verifier、不记 reward=0，也不进入 TQ 样本。它是搜索成本，不是错误答案。

接受并补全后才加入 `_Trajectory`。最终 ID 连续为 `0..G-1`，parent 必须已在该列表中。后续分支可以基于任意已保留完整轨迹，包括刚刚完成的 lookahead 分支。

`branch_pos=b` 不变；probe 与 completion 的分界 b+R 不创建额外树节点。复用现有 `finalize_tree` 为所有最终轨迹构造路径，包括在旧父轨迹路径中插入 pivot 的行为。

树节点 value 仍由最终保留叶子的 reward 计算。剪掉的 probe 没有 reward，不能稀释节点均值。alpha、优势归一化和 loss mask 仍按现有逻辑。

最终 reward 后外层 DAPO 若启用，会照常移除同 reward group 并补新 prompt。这与 lookahead 拒绝是不同层级：前者依据完整 group reward，后者只看短 token 差异。两者叠加可能明显增加耗时，必须分别统计。

采样/拒绝改变行为分布。存好原 token 的 logprob、重算 old logprob 或沿用 PPO correction，均不意味着已严格校正了搜索分布偏差。首版继承现有 ROSE 训练约定，不新增“已无偏”的结论。

## 9. 验证路径、采样参数和随机种子

### 9.1 验证独立，避免评估口径改变

继续在 `trajectory['validate']` 时调用基类 independent `_run_prompt`。验证集不使用 probe、group 搜索预算或 tree metadata；`val_before_train` 只是控制先验证，与启用训练 lookahead 无关。

不要自动把当前 val n / temperature 改成论文设置。要做 pass@8 对照，应单独配置相同的 independent val 参数，两组实验保持一致。

### 9.2 不混淆两种 top-k

- `rose.top_k=20` 是语义熵返回概率候选数。
- sampling 的 `top_k` / `top_p` 是生成采样分布限制。

方案 A 沿用现有 sampling temperature / top_p / top_k；不因为 LATR 论文参数而在融合脚本里暗改训练分布。

### 9.3 首版随机性边界

当前 `_seed_for_prompt` 含 uid，uid 可能来自随机 UUID；固定 seed 不保证跨进程启动逐 token 重现。首版不同时重构 dataset ID / dataloader / checkpoint RNG。

开启 lookahead 时，在已有 prompt seed 基础上为 slot、pivot attempt、阶段派生不同的请求 seed（若调用者提供 seed，则以该 seed 为根），避免同一 prefix 多次 probe 因复用固定 seed 必然生成完全一样的内容。派生过程稳定、不用 Python hash，并限制在后端可接受整数范围。

关闭 lookahead 不注入新的 seed，不改变旧 RNG 消耗顺序。mock 测试使用固定 uid/seed，验证重试可重现和不同 trial seed 不同。服务器全流程确定性仍需独立验证，不能由这些测试推出。

## 10. 请求路由、NPU 与性能取舍

### 10.1 首版不依赖 KV 复用才正确

每段都显式提供完整 token 输入，因此即使 prefix caching 关闭，也应正确生成。模型推理仍在 NPU，CPU 只增加几十 token 的编辑距离与树控制。

vLLM 新请求不等于恢复同一条 KV 会话；将 prefix list 拷贝给请求只是逻辑前缀共享，不保证设备缓存共享或显存节省。

首版为每个 slot / probe / completion 生成唯一的逻辑 request ID，保证 full_determinism 模式也不出现同名请求冲突。同步 client 当前会把不同请求路由后直接透传 extra_fields，因此不为正确性强制改共享 client/router。

稳定 group affinity 是后续性能优化：只有实测默认 router 的 acquire/release 会保留何种 sticky 状态后才能决定实现，不能把“相同 request_id”当 cache resume。如果为 affinity 加新参数，必须在 client 消费，不泄漏到不接受该参数的 server API。

### 10.2 成本估算

假设 G=8、epsilon=0.3，粗略约有 `7*0.7=4.9` 个槽位进入分支搜索；这是概率期望，实际还受无合法 pivot / fallback 影响。

- 原 ROSE 每 prompt 通常 G 次完整生成调用。
- 首轮全部 probe 接受且非 terminal 时，每个分支比原来多一次 RPC；最终 token 里的 probe 被复用，不是完整后缀生成两遍。
- 拒绝 probe 额外消耗 token，completion 可能重复长前缀 prefill。
- 每个槽位最多 8 次 probe；默认 group 总预算 700 又进一步限制总量。
- probe 相似性 DP 通常远小于 NPU 生成和 prefill；不能预先说它是主要瓶颈。

首版不并行生成同 pivot 的多个候选，避免预算竞争、无效候选浪费及结果返回顺序影响选择。未来若做候选并发，全部已发请求成本都要计入，选择顺序必须稳定。

## 11. 观测、保存与成本口径

### 11.1 必需计数

每 group 独立记录：

```text
probe_attempts / probe_full_window_accepted / probe_rejected
short_terminal_kept / short_terminal_duplicate_rejected
epsilon_root_slots / accepted_branch_slots / fallback_root_slots
fallback_no_legal_pivot / fallback_probe_budget / fallback_trials
probe_generated_tokens / rejected_probe_tokens / accepted_probe_tokens
root_generated_tokens / completion_generated_tokens
probe_rpc_seconds / completion_rpc_seconds / root_rpc_seconds
semantic_score_seconds / group_wall_seconds
cached_tokens_known_sum / cached_tokens_observations
```

接受率分母使用 probe attempts，但 full-window pass、short terminal keep 分别报告；最终 full response 截断率不得把 probe 的 length cap 当成一次截断。

`num_preempted=None` 等未知值单独处理，不能将 -1 当成一次负抢占加总。cache 命中缺失不能记成命中数 0 来推导精确命中率。

### 11.2 如何放进 extra_fields 与 trainer

- 最终每 leaf 保存自身 `rose_lookahead_leaf` 摘要：来源、parent、b、最终原因、保留 probe 长度、各窗口距离。
- group 完成后，在每个最终 leaf 的 extra_fields 放入同一份只读 `rose_lookahead_group` 标量摘要，带 tree_id；便于任意保存路径提取。
- trainer 聚合按 tree_id 去重，检查同组摘要一致，只计一次；padding 不计。不能简单对 G 份 group 计数求和。
- 保留当前 ROSE leaf 指标口径，新增 `rose_lookahead/*`；新增指标明确以最终进入 batch 的 groups 为范围。
- 拒绝尝试的时间计入 group 成本，但不虚构成某个 retained leaf 的生成时间；leaf 可另记“为该槽位付出的搜索时间”。

重要限制：trainer 读到的是经过过滤/保留的最终 batch；被 DAPO 丢弃、失败、额外补采后未入 batch 的 group 不一定出现。因此该处不能宣称统计了整步全部生成成本。

为完整 workload profiling，lookahead 开启时每个 group 输出一条紧凑 `[ROSE_LOOKAHEAD_GROUP]` 摘要，包含 uid、step、finished/failure、上述成本及 failure reason；即使 group 后来被 DAPO 过滤也保留此日志。日志聚合与 trainer surviving-batch 指标分开命名，不混算。

### 11.3 有界调试 trace

`trace_dir=null` 时不写额外 trace 文件；开启时由 worker 写入 `hostname + pid + worker标识` 区分的 JSONL，避免多节点写同名文件冲突。每 group 最多保存配置数量的事件，另记录 dropped-event 数量。

事件包含：uid、step、slot、trial、parent、b、prefix digest、phase、segment 长度、真实 reason、各窗口距离、最终决定、seed、耗时。默认不保存 prompt 正文、response 正文或全部 top-k。

`trace_include_token_ids=true` 时仅增加有界的 probe / 对应父窗口 token IDs，方便离线 decode；属于敏感训练数据，目录权限和共享文件系统由运行者确认。不默认为了调试复制整个数据集。

trace I/O 属于可选观测：失败需明显报警但不改变算法接受/拒绝；测试覆盖异常路径，不让日志写满导致无限重试。group 最终摘要也必须在异常清理路径尽力输出。

### 11.4 最终 rollout 历史

继续使用现有 `trainer.rollout_data_dir`、`trainer.validation_data_dir` 保存最终 response。若要一起保存 lookahead 摘要，在 `_log_rollout_data` 中增加读取 output extra_fields，并与现有 sorted_indices 同步排序后传给 `_dump_generations`。

关闭 lookahead / 旧样本没有摘要时仍可保存，不能出现 missing-field crash。训练历史只保存 retained 完整轨迹，拒绝 probe 在单独 trace，不冒充训练样本。

## 12. 启动脚本如何适配

新增根目录 `run_rose_latr.sh`，基于当前 `run_rose_new.sh` 复制形成可独立运行脚本，不改用户正在使用的旧脚本、不只提供一个找不到实际训练命令的包装片段。

保持 BASE_PATH、MODEL_PATH、数据路径、reward、模型长度、batch、TP、节点数、NPU 环境、embedding 准备、Ray 生命周期与 baseline 一致。沿用原有 `SAVE_PATH` 体系，但使用新的实验名/输出子目录，避免混入 baseline。

拟新增可填写变量：

```bash
ROSE_LOOKAHEAD_ENABLE=${ROSE_LOOKAHEAD_ENABLE:-true}
ROSE_LOOKAHEAD_WINDOWS=${ROSE_LOOKAHEAD_WINDOWS:-'[20,30,50]'}
ROSE_LOOKAHEAD_THRESHOLD=${ROSE_LOOKAHEAD_THRESHOLD:-0.4}
ROSE_LOOKAHEAD_MAX_TRIALS=${ROSE_LOOKAHEAD_MAX_TRIALS:-2}
ROSE_LOOKAHEAD_MAX_PIVOTS=${ROSE_LOOKAHEAD_MAX_PIVOTS:-4}
ROSE_LOOKAHEAD_GROUP_BUDGET=${ROSE_LOOKAHEAD_GROUP_BUDGET:-700}
ROSE_LOOKAHEAD_REQUEST_TIMEOUT=${ROSE_LOOKAHEAD_REQUEST_TIMEOUT:-600}
```

训练命令相应追加，不使用 `++` 绕 schema：

```bash
actor_rollout_ref.rollout.rose.lookahead.enable="$ROSE_LOOKAHEAD_ENABLE" \
actor_rollout_ref.rollout.rose.lookahead.windows="$ROSE_LOOKAHEAD_WINDOWS" \
actor_rollout_ref.rollout.rose.lookahead.edit_distance_threshold="$ROSE_LOOKAHEAD_THRESHOLD" \
actor_rollout_ref.rollout.rose.lookahead.max_trials_per_pivot="$ROSE_LOOKAHEAD_MAX_TRIALS" \
actor_rollout_ref.rollout.rose.lookahead.max_pivots_per_slot="$ROSE_LOOKAHEAD_MAX_PIVOTS" \
actor_rollout_ref.rollout.rose.lookahead.max_probe_tokens_per_group="$ROSE_LOOKAHEAD_GROUP_BUDGET" \
actor_rollout_ref.rollout.rose.lookahead.request_timeout_seconds="$ROSE_LOOKAHEAD_REQUEST_TIMEOUT" \
```

脚本中真实训练入口仍为当前 `python -m ...` 所用入口，显式注释其位置。可选 trace path 通过 shell 参数数组按需添加，避免将空字符串误当 null；有空格路径需要使用与项目 Hydra 约定一致的引用。

对照跑法：同一个新脚本仅切换 `ROSE_LOOKAHEAD_ENABLE=false/true`，其他参数相同。暂不一起改 DAPO、alpha、clip high、loss_scale_factor、val sampling、prefix caching，以免无法归因。

## 13. 文件改动清单与最小职责

| 文件 | 首版改动 | 不应顺手做的事 |
|---|---|---|
| `verl/workers/config/rollout.py` | 新配置类、导出、嵌套字段和校验 | 放宽全部 BaseConfig 冻结规则 |
| `verl/trainer/config/rollout/rollout.yaml` | lookahead 默认配置 | 修改全部 rollout 默认采样参数 |
| `verl/experimental/rose/lookahead.py`（新） | 距离、候选、预算、统计纯逻辑 | 导入 backend / 加载 NPU 模型 |
| `verl/experimental/rose/rose_agent_loop.py` | segment helper、兼容包装、拼接契约 | 修改 tokenizer/template 以“适应”分支 |
| `verl/experimental/rose/rose_agent_loop_tq.py` | 开关、槽位搜索、seed、trace、最终统计 | 改验证算法、提交失败 probe |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 原始 reason、实际预算元数据 | 全局重定义 TokenOutput.stop_reason |
| `verl/trainer/ppo/v1/trainer_base.py` | 指标去重、历史保存摘要 | 重写 DAPO / dataloader / PPO 主循环 |
| `tests/experimental/rose/test_lookahead_on_cpu.py`（新） | DP / 窗口 / 候选 / 预算 |
| `tests/experimental/rose/test_lookahead_rollout_on_cpu.py`（新） | mock segment / 槽位 / group 集成 |
| 现有 ROSE config / helper 测试 | 兼容、schema、metrics、dump 回归 |
| 合适的 rollout server CPU/mock 测试 | raw stop metadata 透传 |
| `run_rose_latr.sh`（新） | 独立 NPU 运行脚本，保持 baseline |
| 本文与项目其余三份 md | 实际状态、测试和差异更新 |

默认不改：`tree.py`、`tree_advantage.py`、`semantic_entropy.py` 数学公式、reward verifier、actor loss、vLLM 第三方源码、共享 router 和 fully async client。若实现时发现必须越界，先说明证据和契约原因，不擅自将方案 A 扩成在线树重构。

## 14. 测试矩阵与验收标准

### 14.1 纯 CPU 单测

1. 编辑距离：相同=0，完全替换/插入/删除、空序列、重复 token、边界长度。
2. 多窗口：每个窗口全过才接受；20 失败即总体失败；阈值恰好相等保留。
3. 候选：过滤非有限分数、短父后缀、预算不足位置；全局排序和稳定 tie-break；同 prefix 多 parent 去重。
4. 状态：slot 重置但 group 预算不重置；预算 0、49、50、700；预留后按实际长度结算。
5. 配置：非法 mode/NaN/负数/乱序 windows；独立默认 list；Hydra YAML 到 dataclass 的真实实例化。
6. seed：同 uid/seed/slot/trial 可复现；不同 trial seed 不同；关闭路径不注入新 seed。

### 14.2 mock server / worker 集成

| 场景 | 必须断言 |
|---|---|
| probe 通过 + completion | 实际请求上限正确；P/Q/C 只拼一次；prompt 不变 |
| 连续拒绝 | 尝试/pivot/总预算均有上限；最终 root fallback |
| 同 prefix 在多个父中出现 | 不能靠换 parent 绕过 pivot 次数 |
| 无合法 pivot | 不调用 probe，直接回退 |
| probe 在 12 token 自然 stop | 不续写，按 short terminal 规则处理 |
| probe 恰好 50 token stop | 以 stop 优先，绝不误续写 |
| probe 50 token length | 通过后继续，未误记完整截断 |
| completion 达总长度上限 | 保留最终截断，长度不越界 |
| context clamp | 不超 context；无法解释的短 length 报错 |
| abort / timeout / missing reason | group failure，拒绝指标不增加 |
| 缺 logprob / top-k 形状错 | fail closed，不补 0 |
| probe 或 completion 版本变化 | 整组拒绝，不仅检查最后一段 |
| 只有 G=1 或 epsilon=1 | 没有 probe；仍恰好 G 个最终输出 |
| 中途失败 | 不提交未验证整树；TQ 状态终结且无悬空后台 postprocess |
| 最终树 | parent 都有效、branch_pos 是 b；probe 边界不造节点 |
| reward | 被拒绝 probe 从未触发 verifier；最终每 leaf 独立 reward |
| 指标 | G 份 group 摘要只计一次；padding 排除；旧样本不崩 |
| 保存 | 摘要和 uid/response 排序一致；trace 有界；I/O 异常不影响分支判据 |
| validation | 即使开关开启，仍调用 independent 原路径 |
| 关闭开关 | 请求数量/长度、RNG 消耗、prefix 和 tree metadata 与旧 ROSE 一致 |

关闭开关的“相同”指 mock 可观测契约与控制流程；不能用它承诺 NPU 不同调度下浮点/采样逐 bit 一致。

### 14.3 拟运行命令（实施后才运行）

遵循 AGENTS.md 使用 uv 管理环境；不在本次文档任务安装依赖。实施时先检查可用 uv / 现有环境，不覆盖用户环境。

```bash
uv run pytest -q tests/experimental/rose
uv run pytest -q tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py
uv run pytest -q tests/workers/rollout/test_llm_server_routed_experts_on_cpu.py
uv run pre-commit run --files <本轮实际修改的文件>
bash -n run_rose_latr.sh
git diff --check
```

新 server 元数据测试及实际有影响的测试要补入上述命令。若依赖缺失或本地导入 NPU/vLLM 失败，记录确切失败，不把 skip 当通过；纯函数测试尽量不依赖这些模块。

### 14.4 Ascend 服务器分层验收

1. 固定少量真实纯文本 prompt，G=2，epsilon=0，让新路径实际走分支；response 先用能容纳窗口的短上限。
2. 看见 `probe_start → reject/accept → completion/root_fallback → group_finished`；核对 token IDs 及 decode，无乱码、无 EOS 后续写。
3. 同一小批跑 enable=false/true，核对关闭路径、最终长度与 reward 通路正常。
4. 完成 1 个 actor update，检查 loss、KL、grad norm 有限，tree estimator 无 metadata 错误。
5. G=8，再测试当前 TP 和两节点配置；最后恢复生产 response/batch 上限。
6. prefix caching 单独做开关正确性/性能实验，不与首次算法 smoke 绑在一起。

不能承诺本机自动完成 NPU 验收。需要服务器证据的项目在工作日志中标为待验证，并交付命令、日志 marker 和预期。

### 14.5 训练消融

至少 baseline ROSE vs A，两者固定数据、模型、reward、训练/验证采样、batch、loss、alpha、epsilon、DAPO 和最大训练步数。条件允许再做多 seed。

同时记录：验证 pass@1/pass@8、reward、组内方差、不同答案数、最终长度/截断、KL/clip、拒绝率/回退率、总生成 token、有效 token/s、step wall-clock、达到目标验证指标的总用时。

若 full-window 接受接近 0、回退接近 100%、短 terminal 异常主导，先停大规模实验检查窗口/token/采样，不增加无上限补采。没有实测前不声称融合更快或更准。

## 15. 下一轮实施顺序与交付门槛

### S0：现场重核对

- 读 AGENTS.md、项目四份 md，检查分支 / HEAD / dirty files。
- 检查是否存在更深层 AGENTS.md；保留用户修改。
- 若 HEAD / 基线脚本已变化，写出差异再按现状实施，不覆盖。
- 本地没有检查 LATR 官方仓库；首版不以拉取官方代码为前置条件，也不声称原样复现。

### S1：纯逻辑和配置

- 新增 dataclass/YAML、纯 lookahead helper、相应单测。
- 验收：距离、预算、去重、schema 可独立测试。

### S2：后端元数据和 segment helper

- 添加兼容的真实停止字段；提取分段 helper，保留原完整生成包装。
- 验收：原 ROSE helper 回归；stop/length/abort、预算、shape、版本 mock 测试。

### S3：训练 worker 状态机

- 加开关、slot 搜索、probe 接受/拒绝、续写、回退、seed 派生。
- 验收：恰好 G、硬边界、树复用、拒绝不 reward、validation bypass、disabled parity。

### S4：观测与启动脚本

- 接入 group 日志、trace、trainer 去重指标、最终历史摘要。
- 新增 `run_rose_latr.sh`，只改变融合相关项及实验输出目录。
- 验收：脚本语法、关闭/开启配置解析、排序/指标/trace 测试。

### S5：回归与交接

- 跑能运行的全部相关 CPU/mock/lint 检查，区分失败原因与功能失败。
- 更新文件清单、配置表、实际接口名、真实测试结果。
- 交付服务器 smoke 命令/参数和 marker；说明未验证的 Ascend 部分。
- 不擅自 commit、push、开 PR 或启动昂贵服务器训练。

每完成 S1–S5 都更新 WORKLOG；只有真实完成后才勾选实现状态。不能因为已有本文就把所有阶段标记完成。

## 16. 后续方案 B 的明确隔离

方案 A 验证之后，才考虑在 ROSE pivot 上增加 LATR `p(candidate)>0.25` 和 `p(top1)-p(candidate)<0.15` 门控、候选首 token 控制。

届时需要另写增量规格，解决：

- 父轨迹不是 greedy，排除实际父 token 后是否允许 top1 作为候选。
- pivot 候选 payload 的缓存，不能假设已经丢弃的 top-k 还存在。
- raw / processed logprobs、temperature/top-p 的概率定义，不能将 top-20 重归一化来通过门控。
- 强制 token 的真实原策略 logprob，不能写成约束分布中的 0。
- 行为分布偏差、训练 mask、候选耗尽策略。

完整在线 LATR、eta 衰减、跨兄弟窗口去重、动态窗口提前取消、真实 KV continuation 和 NPU kernel 皆独立于本轮实施，不夹带进方案 A。

## 17. 实施前状态检查表

- [x] 已核对当前 ROSE worker / helper / server / tree / trainer / config 的关键接口。
- [x] 已将一次 50-token 调用而非逐 token RPC 写入规格。
- [x] 已明确首版短终止、预算、回退、树和概率偏差边界。
- [x] 已列出文件、接口、测试、NPU 验收和实施顺序。
- [x] 配置和算法代码实现，独立 `run_rose_latr.sh` 与运行指南已交付。
- [x] CPU / mock / Ruff / 格式 / 编译 / shell 语法通过；完整 pre-commit 环境初始化受网络限制，未声称全部 hook 通过。
- [ ] NPU rollout 与 actor-update smoke 通过。
- [ ] 真实训练消融和效果结论。
