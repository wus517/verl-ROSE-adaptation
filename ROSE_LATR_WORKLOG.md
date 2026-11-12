# ROSE + LATR 进度与决策日志

## 2026-11-12：首次论文与源码核查

状态：只完成分析与文档，没有算法实现，没有运行训练或算法测试。

### 基线

- 当前分支：`rose-plus-latr`。
- commit：`e77dceb55b68416c3d153c3b36a0f13c815733dd`。
- 开始时 `git status --short`：仅 `?? paper/LATR.pdf`。
- 已阅读根目录 AGENTS.md；不进行 PR、提交或推送。

### 本轮已做

1. 使用 PDF skill，读取 LATR 主要方法、Algorithm 1、消融、Appendix B/C，并渲染/查看第 4、5 页核对公式与算法流程。
2. 回看 ROSE §3.1–3.2，确认完整响应生成后选语义熵分支点的流程。
3. 核对当前 ROSE worker、token-prefix helper、树 finalization、优势 estimator、top-k payload、server/client、TQ 和已有测试。
4. 检查 `run_rose_new.sh` 当前默认参数；不把历史聊天中建议过的参数视为已经修改。
5. 新增三份项目文档，训练代码和脚本保持不变。

### 已确认的源码事实

- `_select_branch` 对所有已有完整轨迹取全局语义熵最大值，没有失败 pivot 状态。
- `generate_from_prefix` 覆盖 max_tokens 为完整剩余长度，需要提取分段 helper。
- 服务端将 `stop` 和 `length` 都转为 `completed`；需要保留真实停止原因才能安全续写。
- top-k payload 在 helper/worker 中被消费，后续强制候选需要另存紧凑候选池。
- 当前树记录与 estimator 可以复用，但前提是失败 probe 不进入最终训练树，未来 parent 只引用保留完整轨迹。
- 当前默认 `logprobs_mode=processed_logprobs`，概率门控和强制 token 的原策略 logprob 必须特别审计。
- 目前 seed 的 prompt key 包含随机 UUID，跨运行并非完整确定性。
- 当前 `run_rose_new.sh` 仍是 epsilon=0.3、alpha=0、train batch=128、response=3072、KL=0.01、单条贪心验证、prefix caching=False，未显式设置 group filtering 或 loss 聚合。服务器实际脚本可与本地不同。

### D001：首版保留 ROSE，仅插入 lookahead

提议，待用户讨论确认。

先做 `rose_lookahead`：ROSE 选点 + 随机短程试跑 + 父子编辑距离剪枝 + 试跑保留续写。保留 epsilon、树 segment advantage、alpha、loss 和外层 DAPO 逻辑。

理由：可以隔离 lookahead 剪枝的作用；不需要立即逐 token 接管 vLLM 的 decoding，也不引入强制候选的额外概率契约。

替代方案 B 是加入概率门控和不同首 token；完整在线 LATR 为更大重构，暂不优先。

### D002：明确这不是原样 LATR

事实与设计边界。

LATR 搜索期用 greedy 主路径/候选 token 在线扩树；首版依然使用已有随机完整父轨迹，并允许预算内重试。不能继承其 backtracking-free 复杂度结论，也不能承诺其论文的速度收益。

### D003：只让最终完整叶子进入 TQ 与 reward

提议。

拒绝的 probe 是搜索开销，不是失败答案，不给 0 reward，也不参与 node value；试跑中不生出正式子树，故现有 tree schema 有望不改。

### D004：预算和停止语义是实现前置条件

提议。

配置候选次数、不同 pivot 次数及 group probe token 硬预算；失败则 root fallback。先解决 EOS / length-cap / abort 区分，禁止把 EOS 后继续生成当作正常 response。

### D005：不把 token 差异说成正确性信号

事实与实验约束。

LATR lookahead 的判据是短段 token 距离，不依赖标准答案，不预测 reward。编辑距离只是代理指标，保留分支不一定更正确；实际收益通过受控训练和独立验证评估。

### 待讨论

- 是否接受 A 为首版、B 为单独消融？
- 初版是否采用 [20,30,50] 与 0.4 的论文起始设置？短 EOS 的保留策略是否合适？
- 新启动脚本名称及服务器实际 baseline 超参；不要同时更换多项算法配置。
- 以后是否拉取 LATR 官方代码核对窗口 off-by-one、候选 probability 和 padding 细节？当前未检查官方仓库。
- alpha、DAPO 开关、验证采样方式需固定后才做融合对比。

### 验证记录

- 分支 / commit / git status：已读取。
- 论文：文本提取；LATR 方法页 4–5 已视觉核对。
- 文档：新增后检查路径引用和 whitespace diff。
- 单测、NPU smoke、训练效果：未运行；本轮无训练代码变动。

## 2026-11-12：详细实施规格落盘（仍未实现）

用户要求：现在生成详细适配方案，下一步再按方案实施。

### 本轮操作与范围

- 重新检查分支 / HEAD / worktree、根目录 AGENTS.md，以及现有三份设计文档。
- 核对 RoseRolloutConfig、YAML、worker、生成 helper、vLLM server、同步/异步 client、tree、trainer metrics / rollout dump 和现有测试。
- 新增 `ROSE_LATR_IMPLEMENTATION_PLAN.md`；更新导航和初步设计的细节说明。
- 未修改 Python / YAML / shell，未启动训练，未提交或推送；用户的 `paper/LATR.pdf` 保持不动。
- 本轮没有重新提取 PDF；论文依据沿用前次已核对记录，不声称检查过 LATR 官方代码。

### D006：首版实施默认值收敛

状态：作为下一轮实施起点，未实现；用户可继续调整。

- 一次 vLLM 请求最多 50 token，返回后检查 20/30/50，接受后复用 probe 续写；不逐 token 外部 RPC。
- 每槽位 epsilon 只抽一次；每 pivot 2 次、4 个 pivot，group 全部 probe 共 700 token。
- 对 D004 / 初步设计预算细化：每次需预留完整最大窗口，预算不足直接回退，不缩短窗口假装通过全部检查。
- 短自然 stop 的非重复完整响应保留并单列，不作为 full-window pass；不越过 EOS/stop 续写。
- 请求超时暂定 600 秒可配置；超时/abort/协议错误属于 group failure，不伪装成普通剪枝拒绝。

### D007：首版不为正确性重构路由

状态：取代初步设计中把稳定 group affinity 列为必须项的描述。

完整 token 输入保证分段正确性；独立唯一 request ID 先实现，不承诺 KV 命中。affinity 属于 profiling 后优化，需先核实 router 会话生命周期。当前同步 client 直接透传新增 extra_fields，无需仅为 stop metadata 改共享 client；fully async 仍不支持。

### D008：成本观测区分训练保留样本与全部 workload

最终 leaf 可以携带重复 group 摘要，但 trainer 必须按 tree_id 去重。经过 DAPO 过滤/失败/多余补采而没有进入 batch 的 group，不在 retained-batch 指标中；每 group 紧凑日志和可选 worker trace 才提供更完整的 workload 证据，不能混称整步总成本。

### D009：重试 seed 与关闭路径隔离

开启 lookahead 时为 slot/trial/phase 派生不同请求 seed，避免相同 prefix + 固定 seed 必然重复。关闭时不改变旧 RNG 流程。不在首版顺手改 dataset ID / UID 体系，明确全流程跨运行确定性仍有限。

### 下一步

用户授权实施后，按详细方案 S0–S5 顺序：现场重核对 → 配置与纯逻辑 → raw stop 与 segment helper → worker 状态机 → 观测和独立脚本 → CPU 回归及服务器 NPU 交接。候选概率门控 / 强制 token 属第二阶段，不夹带进首版。

### 本轮验证状态

- 文档内容与关键源码契约：已核对。
- 四份项目 Markdown 目标文件存在性：通过 shell `test -f` 检查。
- `git diff --check`：无输出；四份未跟踪 md 另用 `git diff --no-index --check /dev/null <文件>` 检查，均无 whitespace 报告（新增文件比较的退出码 1 表示存在 diff，不是空白错误）。
- `git status --short`：仅四份项目 md 和用户原有 `paper/LATR.pdf` 未跟踪，无受跟踪训练代码/配置/脚本改动。
- CPU 单测、NPU smoke、actor update、训练效果：未运行，不适用本轮文档交付。

## 2026-11-12：首版 A 实施与本地回归

用户已授权按详细方案实施，并以 `run_rose_new.sh` 为基线生成新服务器脚本。本轮属于实际进展，不是等待或仅重述计划。

### 已实现

1. `RoseLookaheadConfig` 及嵌套 YAML：默认关闭、严格 windows / budget / timeout 等校验、可选 trace；真实 Hydra 和 typed conversion 都已测试。
2. `lookahead.py`：滚动行 token 编辑距离、多窗口判定、增量 prefix SHA 去重排序、group 预算预留/结算、slot/trial/phase seed 派生。
3. `rose_agent_loop.py`：新增 `GenerationSegment`、`generate_segment`；原 `generate_from_prefix` 保留兼容包装。检查 ids/logprob/top-k 长度、实际请求预算和原始 stop metadata。
4. `vllm_async_server.py`：正常及空 abort 都保留原始结束原因、stop detail、effective max_tokens、input/context 数；原 TokenOutput.stop_reason 映射不变。
5. `lookahead_rollout.py`：每 group 独立的搜索对象；每槽位 epsilon 一次、全局合法高熵 pivot、2 次/4 pivot/700 token 边界、拒绝不评分/不 reward、接受 probe 复用、短自然 stop 去重例外、root fallback、版本检查、请求超时。
6. `rose_agent_loop_tq.py`：接入开关；验证 independent bypass；最终 G 个保留叶子才 finalize / postprocess；group summary 在循环外一次快照，避免不同 leaf 的 wall 时间不同导致一致性错误。
7. `trainer_base.py`：按 tree_id 去重 retained-batch 指标、检查摘要一致；最终历史增加 leaf/group 摘要并与 response 同序排序。拒绝/失败/被 DAPO 过滤的全部 workload 另用 group 终结日志记录。
8. 有界 per-worker JSONL、请求开始/返回/判定 marker、未知失败 token 成本计数。RPC await 取消不保证引擎取消，日志明确告知。
9. 新增可执行 `run_rose_latr.sh`，独立包含环境准备、embedding、Ray 集群与实际训练入口；新实验输出子目录；原 `run_rose_new.sh` 经 diff 确认未改。
10. 新增 `ROSE_LATR_RUN_GUIDE.md`，包含完整部署、1-step smoke、查日志、保存路径、控制实验及未验证范围。

### 实施细化 / 与原规格的命名差异

- 使用 `LookaheadRollout` 对象和有界 for 循环管理 slot/pivot/trial，没有另外新增冗余的 `SlotSearchState` 类；行为相同，试次数每槽位重置，budget 跨槽位共享。
- segment 参数沿用原 helper 的 `prefix_ids`，未命名为 response_prefix_ids；含义不变，只包含 response 前缀。
- summary 采用 Counter + 固定零值指标而非额外 stats dataclass，避免零尝试的 step 图表丢字段。
- fallback 计数名字是 `fallback_no_legal_pivot` / `fallback_probe_budget_exhausted` / `fallback_pivot_trials_exhausted`，与真实原因字符串一致。
- seed 是稳定 unsigned 32-bit 派生；显式 sampling seed 为根，未提供时用已有 prompt seed。当前 UUID/dataloader 全流程可复现性限制未改变。
- routing replay 开启时明确不支持；没有扩展 fully async，也没改共享 client/router/第三方 vLLM。同步 client 透传字段已有 mock 证据。
- `TOTAL_TRAINING_STEPS` 是新增脚本的可选 smoke 上限，不设时保留 baseline epoch 语义。
- 四份 `_generated_ppo*_trainer.yaml` 用仓库生成器更新；其中普通 PPO 引用配置额外出现 3 个原 schema 已有但旧引用漏掉的 `grad_sync_every_micro_batch: false`。这是生成器同步的 reference-only 字段，不是运行默认值改动。新增测试逐字比对四份生成结果。

### 真实本地测试记录

使用已有 `.venv`，所有 Python 命令经 `UV_CACHE_DIR=/tmp/rose-latr-uv-cache uv run --no-sync ...`，没有安装或替换服务器依赖。

```text
pytest -q tests/experimental/rose
  tests/workers/rollout/test_rose_stop_metadata_on_cpu.py
  tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py
  tests/workers/rollout/test_llm_server_routed_experts_on_cpu.py
=> 135 passed, 6 warnings（可选训练引擎缺失 / Ray deprecation），无 skip
```

覆盖：默认 20/30/50、50-token 恰好 stop、G=8、P/Q/C 拼接、连续拒绝、budget 跨 slot、pivot/trial 上限、短重复、root fallback、错误 metadata、timeout/abort、segment/整树版本、最终 TQ 叶子数、验证 bypass、disabled 不注 seed、accepted-child 可作新 parent、新树 metadata 进入原 ROSE 优势估计器、指标去重、dump 排序、trace I/O、Hydra、原脚本 training override 等价及四份生成配置。

server 测试从实际 `vLLMHttpServer.generate` AST 编译该方法，替换引擎依赖后执行实际预算/结果提取/stop mapping；不是 NPU 运行，也不是只比对文本。真实 Ray、vLLM-Ascend 引擎和分布式 I/O 仍需服务器验证。

其他检查：

- Ruff lint / Ruff format --check（修改及新增 Python）：通过。
- compileall（相关实现和测试）：通过。
- `bash -n run_rose_latr.sh`：通过；脚本已设可执行权限。
- 新增 Python 文件 license 检查：通过。
- device API usage 检查：通过。
- `git diff --check`：通过；新增/未跟踪代码、脚本和五份 md 逐一执行 `git diff --no-index --check /dev/null <文件>`，均无空白错误。新增 diff 的退出码 1 不代表检查错误。
- 实际 `print_cfg.py --cfg job` 加新 overrides 和 `trainer.total_training_steps=1`：通过，不启动训练。

### 不应误报为通过的检查

- 完整 pre-commit 首先遇到默认 cache 目录权限；改用 `/tmp` 后，hook 环境初始化遇到 `Could not resolve host: github.com`。因此没运行完完整 hook 集合，本地 Ruff 等是单独执行。
- `scripts/generate_trainer_config.sh` 成功重新生成四份引用配置后，以工作树相对 HEAD 有 diff 为由按原脚本约定退出 1；不是生成失败。新增 4 个 CPU 测试对比实时生成输出与文件，全部通过。
- 全仓 `validate_structure.py` 报三个既有根目录测试布局问题：`test_run_debug_tokenizer_rollout_script_on_cpu.py`、`test_debug_tokenizer_rollout_on_cpu.py`、`test_verify_threaded_on_cpu.py`。本轮新测试位于正确子目录，没有顺手改旧问题。
- 没有目标 Ascend 设备，NPU rollout、actor update、两节点实训、速度/准确率结论均未验证。服务器命令及验收门槛见 RUN_GUIDE。

### 当前交付范围

首版 A 代码、独立服务器脚本、本地可运行验收和服务器 smoke 指南已落盘；不声称完成 NPU 实机验收或完整在线 LATR。未 commit、push 或开 PR，用户论文保持不动。

## 后续条目模板

每次实现追加：日期、基线、用户已确认目标、修改路径、数据契约、测试命令/结果、服务器证据、未解决问题、下一步。不能把计划中的测试填写为通过。
