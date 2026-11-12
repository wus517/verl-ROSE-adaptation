# ROSE + LATR lookahead：运行和验收

## 1. 已实现的范围

这是 [详细方案](ROSE_LATR_IMPLEMENTATION_PLAN.md) 的首版 A：ROSE 高语义熵选点，先生成最多 50 个 token，以 20/30/50 多窗口 token 编辑距离筛掉相似短段，通过后复用 probe 继续生成。不是完整在线 LATR，不含概率门控或强制首 token。

通用默认开关关闭；新脚本默认开启。原 `run_rose_new.sh` 保持不变，验证集仍独立采样，树优势、alpha、loss、reward 和外层 DAPO 不改。

新增运行脚本：[run_rose_latr.sh](run_rose_latr.sh)。它是独立的调度平台 NPU 脚本，包含原脚本的环境准备、embedding 准备、Ray 多节点启动和实际 `python3 -m verl.trainer.main_ppo` 命令，不是单机通用启动器。

## 2. 部署与正常启动

把整个修改后的项目同步到服务器，不要只复制新脚本。尤其要包含：

- `verl/experimental/rose/lookahead.py`、`lookahead_rollout.py`、两个已修改的 ROSE agent loop。
- rollout dataclass/YAML、trainer metrics/dump 和 vLLM server 的修改。
- `run_rose_latr.sh`。

脚本仍会从调度平台源目录复制项目到 `/cache/algorithm`；必须更新真正的源目录，防止 Ray worker 导入旧代码。

沿用原环境里 `lhx_sh`、`lhx_save`、`MA_*`、`VC_*` 等设置。可在脚本内填写，也可按节点提交环境变量：

```bash
BASE_PATH=/your/data/base \
SAVE_PATH=/your/output/base \
MODEL_PATH=/your/model/Qwen3_5_9B \
EXP_NAME=qwen35_rose_latr_trial1 \
bash run_rose_latr.sh
```

这些只是路径占位符；节点启动方式与原脚本一致，不要在已有生产 Ray 集群里盲目重复启动。

保留原 train/val/reward 路径和默认模型、batch、采样超参；新增默认实验名/输出子目录为 `rose_latr`，避免与原 ROSE 混淆。外部显式传相同 `EXP_NAME`、checkpoint 目录时仍需自行避免覆盖。

## 3. 新参数

| shell 变量 | 默认值 | 含义 |
|---|---|---|
| `ROSE_LOOKAHEAD_ENABLE` | true | 关闭后走原 ROSE 控制流 |
| `ROSE_LOOKAHEAD_WINDOWS` | `[20,30,50]` | 每个窗口都须达到距离阈值 |
| `ROSE_LOOKAHEAD_THRESHOLD` | 0.4 | token 编辑距离 / 窗口长度的下限 |
| `ROSE_LOOKAHEAD_MAX_TRIALS` | 2 | 单槽位每 pivot 最多 probe 次数 |
| `ROSE_LOOKAHEAD_MAX_PIVOTS` | 4 | 单槽位不同前缀 pivot 上限 |
| `ROSE_LOOKAHEAD_GROUP_BUDGET` | 700 | 每 prompt group 所有 probe 的生成 token 上限 |
| `ROSE_LOOKAHEAD_REQUEST_TIMEOUT` | 600 | 单次 RPC 秒数上限，包括排队时间 |
| `ROSE_LOOKAHEAD_TRACE_DIR` | 空/不写 | 有界 per-worker JSONL 的服务器路径 |
| `ROSE_LOOKAHEAD_TRACE_TOKEN_IDS` | false | 是否记录有界 probe / 父窗口 token IDs |
| `TOTAL_TRAINING_STEPS` | 不设 | 可选训练 step 上限，供 smoke 用 |

预算包括通过的 probe，不仅是拒绝浪费；预算不足完整最大窗口时直接 root fallback，不缩短窗口。不限制后续 completion / 根部 response / 外层 DAPO 补采成本。

根部 epsilon 采样不做 lookahead，短自然 stop 的非重复响应按设计保留并单列。出现 abort、超时、缺原始停止原因、概率形状或版本错误会将 group 标记失败，不作为“低多样性”重试。

`TOTAL_TRAINING_STEPS` 只给 trainer 设上限，不改变 dataset 及 epoch 语义。生产环境保持不设即可沿用原 total_epochs。

## 4. 推荐第一次先跑 1 step

在原有多节点调度任务中，对所有节点传入一致参数。两节点每节点 8 卡、TP=2 时，下例的 `train_batch=16`、`n=2`、mini=8 可作为起点；若卡数/并行度不同，先检查 batch 整除关系。

```bash
EXP_NAME=rose_latr_smoke \
ROSE_LOOKAHEAD_ENABLE=true \
ROSE_EPSILON=0 \
ROLLOUT_N=2 \
TRAIN_BATCH_SIZE=16 \
PPO_MINI_BATCH_SIZE=8 \
MAX_RESPONSE_LENGTH=256 \
TOTAL_TRAINING_STEPS=1 \
ROSE_LOOKAHEAD_TRACE_DIR=/your/output/base/rose_latr_smoke_trace \
bash run_rose_latr.sh
```

必须沿用可用的 `MODEL_PATH`、数据和 reward。response=256 只是 smoke，可能显著降低数学答案完整性；不是效果实验超参。若父轨迹少于 50 token、语义熵没有合法位置或 probe 太相似，允许正常 fallback；应同时看原因计数。

成功标准：至少一个真实 probe，明确 accept/reject，恰好 G 个最终叶子进入 reward/TQ，一次 actor update 无非有限 loss / KL / grad、无 tree metadata 错误。随后 n=8、恢复生产 response，再恢复生产 batch。

本机没有 Ascend 设备，未执行上述服务器训练。CPU/mock 通过不能证明 Qwen3.5 在特定 vLLM-Ascend 版本的长前缀或缓存正确性。

## 5. 查日志与看输出

训练日志使用原脚本的 `LOG_PATH`。替换下面文件名：

```bash
grep -nE '\[ROSE_LOOKAHEAD_PROGRESS\]|\[ROSE_LOOKAHEAD_GROUP\]' train.log
grep -nE 'probe_start|probe_finished|"phase": "accept"|"phase": "reject"|root_fallback' train.log
grep -nE 'ERROR|Traceback|policy version|stopping metadata|timeout|cancellation' train.log
```

`slot` 从 0 开始；`probe_start` 说明已提交短程生成，`probe_finished` 是返回；`accept` 后如果尚未终止，应出现 `completion_start/finished`。日志在 RPC 前就打印，能区分是等待生成还是在做下一步。

新 marker 使用 warning。若平台屏蔽 warning，优先检查 Ray worker 日志或启用 `ROSE_LOOKAHEAD_TRACE_DIR`；不要只在 driver stdout 里判断代码没执行。trace 在 group 结束时落盘，不是实时逐 token 进度。

最终 prompt/response/reward 仍自动保存在：

```text
${SAVE_PATH}/rollout_history/rose_latr/${EXP_NAME}_${TIMESTAMP}/train/<step>.jsonl
${SAVE_PATH}/rollout_history/rose_latr/${EXP_NAME}_${TIMESTAMP}/val/<step>.jsonl
```

训练 JSONL 额外包含 `rose_lookahead_leaf`、`rose_lookahead_group` 摘要。拒绝的 probe 不在训练 response 历史中；需要检查时启用可选 trace 和 token IDs，并注意敏感数据权限。

可选 trace 文件按 hostname/pid 区分，group 内事件有上限；终结摘要不受事件上限丢弃。worker 串行文件写入的异常报警但不改变算法决定。

## 6. 指标和故障解释

- `probe_full_window_accepted`：通过全部窗口的 probe。
- `short_terminal_kept`：短自然停止例外保留，不等价于完整多窗口通过。
- `probe_rejected` / `rejected_probe_tokens`：搜索额外成本，不是 reward=0 的训练轨迹。
- `fallback_root_slots` 与具体原因：预算/次数/无合法 pivot 回退。
- `failed_requests_unknown_tokens`：后端未返回可计量结果，已用 token 未知，不能当成零成本。
- `rose_lookahead/retained_*`：只统计最终进入训练 batch 的 group，按 tree_id 去重。
- `[ROSE_LOOKAHEAD_GROUP]`：包括 worker 已完成/失败的 group，即使之后被 DAPO 过滤；不等于上面的 retained-batch 指标。

超时包含后端排队；600 秒是可调护栏。先看 NPU 利用率、调度和同批请求时延，不应无限增加超时来掩盖卡住。取消 await 不保证后端请求立刻取消，发生大量 timeout 时应终止该测试任务并按平台流程清理，而不是在同一集群不断叠加重试。

缺 `rollout_finish_reason` 多为只部署了 worker、没部署 server，或跑到了旧代码；检查 worker/server 实际导入路径。策略版本错误不能跳过校验。

不要在首次 smoke 同时打开 prefix caching。分段本身用完整 token 输入即可正确执行，缓存支持和性能需要在目标 Ascend/模型组合上单独测试。

## 7. 对照与本地验证

同样参数下设置 `ROSE_LOOKAHEAD_ENABLE=false` 得到原 ROSE 控制；用不同 EXP_NAME / 输出目录。不要同时修改 epsilon、alpha、DAPO、loss、batch 和验证采样再归因 lookahead。

本地检查使用项目已有 uv 环境：

```bash
uv run --no-sync pytest -q tests/experimental/rose \
  tests/workers/rollout/test_rose_stop_metadata_on_cpu.py \
  tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py \
  tests/workers/rollout/test_llm_server_routed_experts_on_cpu.py
bash -n run_rose_latr.sh
git diff --check
```

本地以上回归共 135 项通过；Ruff、格式、编译、脚本语法和配置解析也已通过。完整 pre-commit 因无法访问 GitHub 未运行完。实现和测试证据见 [WORKLOG](ROSE_LATR_WORKLOG.md)。未做训练效果或性能提升承诺，需比较总生成 token、wall-clock 和同口径独立验证结果。
