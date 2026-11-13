# ROSE + EA 实施与验证日志

## 记录规则

- 每次工作结束同步记录日期、分支/基线、做了什么、没做什么、测试与待办。
- “论文事实”“源码事实”“工程提案”“实测结果”分开写。
- 尚未实现的配置不能标为可运行，尚未执行的测试不能标为通过。
- 保留用户输入与无关修改；不自动提交或推送。

## 2026-11-12：论文核查与首轮方案

### 输入与基线

- 用户请求：在 ROSE 上结合 EA，先读论文/项目并讨论初步适配方案，同时维护根目录 Markdown 记录。
- 分支：`rose-plus-ea`。
- HEAD：`e77dceb55b68416c3d153c3b36a0f13c815733dd`，最近提交为 `fix verl:grad_sync_every_micro_batch`。
- 初始 `git status --short`：仅 `?? paper/EA.pdf`。没有将历史对话中的服务器改动假定为本地已完成。
- 已阅读根目录 `AGENTS.md` 和 PDF 阅读技能；本轮不涉及 PR 或训练代码写入。

### 论文事实

- EA 标题为 Reasoning with Exploration: An Entropy Perspective，本地 PDF v4，2025-11-08，共 19 页。
- 详细核对第 4-9 页方法/实验/分析、第 11 页总结、第 15 页附录超参数；渲染检查第 5 页，确认式 (4)-(7) 中的 min、abs(A)/kappa、detach 和加号。
- EA 使用 current-policy 全词表熵；优势修正为 `A + min(coef * H.detach(), abs(A)/kappa)`。
- kappa=2，GRPO coef=0.4，PPO coef=0.1；论文未验证 ROSE+EA。
- EA 不等于 entropy loss；论文实验无 entropy/KL loss，clip high=0.28，但这些不自动成为本组合必改项。
- EA 可能增加推理长度，且不以提高整体 entropy 为目标。
- ROSE 第 3.1-3.3 节与当前树优势实现再次核对：树采样、value 差、长度校准、动态过滤与 actor 更新可分离。

### 源码事实与证据定位

| 文件/位置（初始行号） | 核查结论 |
| --- | --- |
| `verl/experimental/rose/rose_agent_loop_tq.py:131` | 验证走独立生成，训练走树 rollout |
| 同文件 `:171` 附近 | 单 prompt 内轨迹顺序生成，使用前序语义分数选分支 |
| `verl/experimental/rose/tree_advantage.py:210-248` | value 差后做长度校准，可将结果作为 EA 基础优势 |
| `verl/trainer/ppo/v1/trainer_base.py:1689` | old-log-prob 阶段当前已请求 entropy |
| 同文件 `:1877` | actor update 的熵开关由 calculate_entropy/entropy_coeff 控制，尚无 EA |
| `verl/workers/utils/losses.py:57-112` | actor loss 读取并对齐 current entropy，在调用 policy loss 前可接 shaping |
| `verl/workers/engine/fsdp/transformer_impl.py:1394` | remove-padding 分支支持已有 chunk 开关 |
| 同文件 `:1486` | 当前 dense 分支直接调用非分块 entropy 函数，需要处理显存风险 |
| `verl/workers/utils/padding.py::no_padding_2_padding` | response 预测需一位位移，复用既有对齐函数 |
| `verl/utils/torch_functional.py:248` | 已有二维 token/vocab 的 FP32 chunk entropy helper，但不直接适配所有 dense 场景 |

本地脚本核查：`run_rose_new.sh` 为 batch=128、mini=32、prompt=1500、response=3072、epsilon=0.3、length alpha=0、KL=0.01、entropy_coeff=0、calculate_entropy=False、use_remove_padding=False、microbatch=1、clip high=0.20、单样本贪心验证，未显式设置 loss 聚合和 group filtering。仅作为本地状态记录，不作为服务器最终配置保证。

### 本轮提案

- 推荐保留 ROSE rollout、树优势及长度校准；在 actor policy loss 前应用 EA。
- current-full-vocab entropy，EA-only 在 NPU 上 detached、no_grad、分块计算；不搬词表张量到 CPU。
- 默认关闭 EA，保留现有基线；初试 coef=0.1/kappa=2，不宣称等于论文 GRPO 最优配置。
- 不将 top-20 semantic entropy 当成 EA 熵，不将 bonus 注入 reward/value/returns。
- 明确零优势不激活、共享前缀计数、长度目标张力和 cap 饱和风险。

### 本轮实际修改

仅新增：

1. `ROSE_EA_README.md`：状态/索引/待讨论事项。
2. `ROSE_EA_DESIGN.md`：算法与工程方案。
3. `ROSE_EA_WORKLOG.md`：本日志。

没有修改任何 Python、YAML、训练脚本或用户 PDF。临时渲染图位于 `/tmp/rose-ea-page5.png`，不是交付物。

### 验证

- 已执行只读 git 分支/状态核查、`rg`/源码阅读、`pdfinfo`、`pdftotext` 和单页渲染检查。
- 文档创建后执行 `git diff --check`，通过；另检查三份未跟踪新文档的行尾空白与仓库相对引用，未发现问题。
- 未执行 CPU 测试、NPU 训练和性能测试，因为训练代码尚未变更。

## 2026-11-12：详细实施方案 v1.0

- 用户要求将方案写到详细 MD，下一轮直接依方案实施；本轮仍不改训练代码。
- 新增 `ROSE_EA_IMPLEMENTATION_PLAN.md`，定义 P0-P7 阶段、逐文件修改、配置和元信息合同、公式与数值检查、CPU/NPU 测试、指标聚合、脚本与回退。
- 进一步核查当前 `Metric.aggregate_dp` 的 DP mean 语义、worker mini-batch 列表与 trainer `reduce_metrics`/MetricsAggregator；明确 EA sums/counts 必须专门保持可加性，不能简单平均各卡均值。
- 明确两个工程选择：新增独立 `ea_entropy` 输出通道，不改普通 entropy 请求；新增 EA `entropy_chunk_size=64`，不依赖 dense 分支未生效的旧 chunk 开关。
- 确定首版 active 范围、coef=0 no-op、unsupported fail closed、保留 ROSE 基线、不自动改 KL/clip/长度等参数。
- 更新 README/算法设计与详细方案一致。下一轮用户要求实施后从 P0 开始，不重复发散已确定的设计。
- 检查：仅文档变更；完成 `git diff --check`、新文档空白/相对链接与必需章节检查。未执行任何训练、依赖安装或 CPU/NPU 算法测试。

## 2026-11-12：v1.1 交接状态核实与方案补全

- 当前请求：生成详细适配方案，供下一步直接实施。本次只更新文档，不继续修改 Python/YAML 或运行脚本。
- 重新执行 `git status --short`、分支核查和 `git diff` 阅读，确认当前仍为 `rose-plus-ea`。
- 工作区已存在之前工作留下的初步改动：actor typed config/YAML、兼容性校验、V1 请求与指标、FSDP entropy 输出、PPO shaping，共 6 个 tracked 文件新增 158 行；另有未跟踪 `verl/workers/utils/entropy_advantage.py`。
- 这些改动尚无本次可报告的测试结果；generated configs、EA 专项测试与 `run_rose_ea.sh` 未补齐。没有回退或覆盖它们。
- 纠正文档中“当前只有文档修改”的过期描述，保留前面历史记录作为当时状态。
- 方案升到 v1.1：增加真实代码快照、下一轮接续顺序、精确 token shift/公式/统计样例和最终交付清单。
- 下轮必须先审查现有代码，按测试驱动补齐，不能直接跑正式服务器训练或将已有初稿标为 P1-P4 完成。
- 文档验证：`git diff --check` 通过；用 `UV_CACHE_DIR=/tmp/verl-rose-ea-uv-cache uv run --no-sync python -c ...` 检查四份 MD 的行尾空白、本地链接和代码围栏，全部通过。本次没有执行算法测试或训练。

## 2026-11-12：P0-P5 实施与本地验证

### 实际改动

- `actor.py`/actor YAML：新增默认关闭的 typed EntropyAdvantageConfig；验证有限 coef>=0、kappa>1、正整数 chunk；支持 DictConfig/dict/dataclass 读取。
- `utils/config.py`：active-only 首版兼容性校验，包含 V1 sync、ROSE、FSDP、vanilla、temperature=1、entropy_coeff=0、非 fused/SP=1 等。
- `entropy_advantage.py`：纯公式与 no_grad 全词表分块熵；mask/数值检查、新优势不覆盖基础 A；raw sums/counts 和最终指标。无 DP 时框架返回的 Metric、常规 mini list 和标量均可处理。
- `transformer_impl.py`：dense/packed 独立 ea_entropy，复用 logprob 的布局恢复；不改变既有 entropy 梯度，不新增模型 forward。
- `losses.py`：从当前 model_output 对齐 H 后 shaping 一次，局部 A_used；13 个统计标量合并传 CPU，不传词表或 token 数组；本地 stats 乘 dp_size 补偿框架 DP mean。
- `trainer_base.py`：EA 请求元信息、启动说明、actor raw stats 求和、最终 logger 前派生 token 加权指标。
- 新增 `run_rose_ea.sh`：四个 EA overrides；原模型/数据/reward/多节点与非 EA Hydra 参数保留，独立输出目录。修正内容复制避免嵌套目录，固定部署源码 PYTHONPATH，补日志文件默认名和源码 SHA256 检查。
- 官方生成脚本已同步四份 generated configs；dp 文件额外同步源 YAML 已有的 grad_sync_every_micro_batch=false，未新增其运行逻辑。
- 四份新增测试覆盖数学、梯度、配置、chunk/dtype/非连续、dense/packed/尾部 padding、response shift、真实 loss/engine、TQ/mini/micro 元信息、多层指标、脚本命令与四模式 generated 输出。
- 现有 rollout helper 测试补 mock stop_reason/tokenizer，匹配生产 logger 所需接口；未改生产 rollout。

### 验证与边界

本地使用已有 Python 3.12 uv 环境（`UV_CACHE_DIR=/tmp/verl-rose-ea-uv-cache` 避免默认 cache 权限问题），未安装 GPU/NPU 堆栈。

```bash
UV_CACHE_DIR=/tmp/verl-rose-ea-uv-cache uv run --no-sync python -m pytest -q \
  tests/experimental/rose \
  tests/trainer/ppo/test_dynamic_policy_losses_on_cpu.py \
  tests/trainer/ppo/v1/test_metrics_aggregator_on_cpu.py \
  tests/utils/test_padding_on_cpu.py \
  tests/utils/test_transferqueue_utils_on_cpu.py \
  tests/workers/test_fsdp_temperature_scaling_on_cpu.py --tb=short -rs
```

最终结果见下方收尾记录。中间测试暴露的是测试构造与真实框架的差异：Hydra 要使用项目 `_convert_=partial` 的正式转换、rollout.name 必填、Metric.extend 而非 list 构造、无 DP 返回 Metric 而非 list；均已修正并加入覆盖，不降低测试断言。

- 正式脚本 Hydra EA on/off 经完整 `validate_config` 通过；不执行任何部署/训练 shell。
- `UV_CACHE_DIR=/tmp/verl-rose-ea-uv-cache uv run --no-sync bash scripts/generate_trainer_config.sh`：四份生成成功；返回 1 来自末尾检查未提交 git diff，不是 Hydra 失败。另有 4 项测试逐字验证当前 generated 文件与 print_cfg 输出相同。
- ruff check/format、`bash -n run_rose_ea.sh` 和 `git diff --check` 执行；最终状态见收尾记录。
- 无 NPU 本地硬件；未验证 9B NPU 训练、真实 HCCL、多节点一致性、耗时/显存和收益。不宣称完成 P6/P7。
- 未改原 run_rose_new.sh、奖励、论文、ROSE rollout/tree advantage；未提交、推送或执行服务器脚本。

## 完成与后续清单

收尾验证（2026-11-12）：上述完整 pytest 命令 **164 passed, 1 skipped**，10.88s。唯一跳过为 macOS 缺少 Linux `/proc/self/fd` 的现有资源检查；6 个 warning 为 Ray 废弃接口与未安装的非目标 engine。ruff check 全通过、10 个改动/新增 Python 文件格式检查通过、bash -n 与 git diff --check 通过；四份 MD 的空白/本地链接/代码围栏检查通过。git diff 验证原 run_rose_new.sh 和 ROSE 生产代码未改变。没有运行 NPU 训练。

- [x] 完成可执行的首版组合与本地脚本基线方案；服务器不同版本可在实施前替换模板。
- [x] 根据实施目标接续并完成 P0-P5。
- [x] 实施 typed EA config 与 YAML/schema 测试；同步 generated configs。
- [x] 实施无原地修改的 shaping helper 和公式/梯度/mask 单测。
- [x] 接通 actor current entropy 请求和 loss 入口，默认关闭不影响 ROSE。
- [x] 实施 EA-only no_grad chunk entropy，dense/packed CPU 对齐验证完成；NPU 实测见后续项。
- [x] 接入正确的跨卡/跨 microbatch 统计，明确不支持的组合；模拟 DP 验证完成。
- [x] 新增独立运行脚本，保留用户模型/数据/奖励/多节点部署路径。
- [x] CPU 回归，记录实际命令和结果。
- [ ] 服务器 NPU 逐步 smoke test，记录 resolved config 与 commit。
- [ ] ROSE/ROSE+EA 对照及长度校准消融，按 token 和时间预算评估。

## 上下文恢复提醒

当前工程适配与本地验证已完成，入口为 run_rose_ea.sh。后续从服务器 NPU smoke test 接续，不重复实施 P0-P5，也不把 CPU 验证当成服务器训练通过。不要混用 old-policy/current-policy entropy。先从 `ROSE_EA_README.md` 进入，并重新核查工作区。

## 2026-11-12：新增算法对比文档

- 按用户要求在分支根目录新增 `ROSE_EA_ALGORITHM_COMPARISON.md`，整理 ROSE/EA 论文算法、当前组合公式、工程实现与近似、运行脚本相对论文的配置差异。
- 明确区分 ROSE top-K 语义评分与 EA 当前策略全词表熵，说明默认 alpha=0、同奖励组过滤关闭，以及失败补采不等于动态过滤。
- README 增加入口；未改 Python、YAML、运行脚本或论文，没有重新运行算法测试、部署或训练。

## 2026-11-13：同步 main 的 rollout 调试输出修改

- 来源：本地 `main` 的 `afaf1a3d`；同步用户指定的 `agent_loop.py`、`single_turn_agent_loop.py`、`rose_agent_loop_tq.py`、`rose_agent_loop.py`，未合并其他文件或切换分支。
- 关闭 prompt 的直接打印；将 NORMAL_RAW_OUTPUT、ROSE_PROMPT_PATH、ROSE_PROMPT、ROSE_RAW_OUTPUT、ROSE_PROGRESS 的 warning 调试块同步为 main 中已注释的 print 块。保留 main 的关闭状态，没有重新启用 print。
- 不引入 main 中重复的 `import time` 和新增行尾空格，保留文件末尾换行。ROSE 采样、树优势、EA 实现和运行脚本未改变。
- 验证：`UV_CACHE_DIR=/tmp/verl-rose-ea-uv-cache uv run --no-sync python -m pytest -q tests/experimental/rose --tb=short`，结果 **128 passed**；`git diff --check` 通过。未运行 NPU 训练；未提交或推送。

## 2026-11-13：新增开启动态组过滤的运行脚本

- 新增根目录 `run_rose_ea_filter_groups.sh`，完整复制 `run_rose_ea.sh`，仅增加 `algorithm.filter_groups.enable=True`、`algorithm.filter_groups.metric=reward` 两行。原脚本、训练代码、数据/模型/reward 路径及其余超参数不变。
- 保留 `sync_refill_failed_groups=True`，它处理失败组，不替代同奖励组过滤。README 增加新脚本入口，避免把原脚本的关闭过滤状态误认为新脚本状态。
- `bash -n run_rose_ea_filter_groups.sh` 通过；与原脚本比较仅上述两行不同。未执行脚本部署或 NPU 训练。
