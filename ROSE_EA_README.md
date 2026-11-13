# ROSE + EA 工作入口与上下文索引

更新日期：2026-11-12。分支：`rose-plus-ea`。

## 当前状态

- 当前阶段：P0-P5 工程适配与本地验证已完成，方案版本 v1.2。
- 已实现：typed config、active-only 校验、独立 current-policy `ea_entropy` 通道、无梯度分块熵、actor-side shaping、token 加权指标。
- 新脚本：[run_rose_ea.sh](run_rose_ea.sh)，基于 `run_rose_new.sh`；原脚本、奖励函数、ROSE rollout/tree advantage 和两篇论文未改。
- 已同步四份 generated configs，新增四份 EA 测试文件。实际验证命令/结果见 WORKLOG。
- 本地回归：164 passed、1 skipped（macOS 不具备 Linux /proc）；ruff、脚本语法及差异检查通过。
- 本机没有 NPU，未执行服务器部署或训练；P6 NPU smoke/性能和 P7 效果实验仍待服务器验证。不能将 CPU mock 验证当成 9B NPU 训练通过。

## 服务器运行

将**整份修改后的项目**部署到原平台 `/opt/huawei/schedule-train/algorithm`，不是只上传脚本。新脚本沿用原 CANN/代理/accelerate 安装/Ray 集群环境和每节点 8 张 NPU 假设，仍依赖原平台的 `MA_*`、`VC_*` 等变量；不是任意服务器的通用集群启动器。

在原来启动 `run_rose_new.sh` 的相同平台入口，改为启动：

```bash
BASE_PATH=/你的数据根目录 \
SAVE_PATH=/你的保存根目录 \
MODEL_PATH=/你的模型目录 \
EXP_NAME=rose_ea_trial \
bash run_rose_ea.sh
```

默认沿用 `${BASE_PATH}/xyi/train_verl.json`、`high_pass_val_verl.json`、`low_pass_val_verl.json` 和 `verify_choice_answer_new_v4.py`；原变量覆盖方式保留。EA 默认 `enable=True, coef=0.1, kappa=2, chunk=64`，原 KL/clip/长度/batch/ROSE 参数不自动改变。

同一脚本做无 EA 对照：

```bash
EA_ENABLE=False EXP_NAME=rose_control bash run_rose_ea.sh
```

如需与开启动态组过滤的完整 ROSE 基线对齐，使用 [run_rose_ea_filter_groups.sh](run_rose_ea_filter_groups.sh)。它仅在上述脚本基础上增加 `algorithm.filter_groups.enable=True` 和 `algorithm.filter_groups.metric=reward`，其余配置不变；EA off/on 对照应统一使用这个新脚本，并设置不同 EXP_NAME/输出目录。原 `run_rose_ea.sh` 保留未过滤版本，本文及算法对比文档中关于该原脚本默认关闭过滤的描述仍然适用。

上例需要已设置 BASE_PATH/SAVE_PATH 或原 `lhx_sh/lhx_save` 环境。所有节点应使用相同代码与 EA 设置。日志开头有 `[ROSE_EA]` 配置、代码来源和 helper SHA256；archive 没有 `.git` 时 commit 显示 not available 是预期行为。

默认 checkpoint、log、TensorBoard、rollout history 使用 `rose_ea` 子目录，路径可覆盖。没有指定 `SAVELOG_FILE_NAME` 时用 `train.log`。`EA_ENABLE=False` 用不同 EXP_NAME，避免混入同一实验 checkpoint。

关注 `actor/ea/entropy_token_mean`、`bonus_ratio_mean`、`cap_fraction` 和 `sign_flips`（应为 0）。验证阶段没有 actor update，所以不会产生验证专属 EA shaping 指标；验证采样仍为基线 n=1 贪心，不是 pass@8。

首次运行按方案 T7：EA off 短跑→EA on 短序列单步→真实长度 3-5 步；记录 NPU 显存和 actor update 耗时。chunk 太大时先减为 32；不要同时改 response 长度、轨迹数和 EA 参数做效果归因。

## 下一次工作从这里开始

1. 阅读本文件，确认当前分支与 `git status`。
2. 完整阅读 [详细实施方案](ROSE_EA_IMPLEMENTATION_PLAN.md)，这是下一轮实施的主入口；算法背景见 [算法与适配设计](ROSE_EA_DESIGN.md)。
3. 阅读 [实施与验证日志](ROSE_EA_WORKLOG.md)，按最新记录接续，不能把计划当成已经完成的代码。
4. 本地实施已完成，不重复从零编写。下一步收集服务器 T7 的 resolved config、日志、耗时和显存，再决定是否调整实现或参数。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `ROSE_EA_README.md` | 当前状态、上下文恢复入口、待讨论事项 |
| `ROSE_EA_IMPLEMENTATION_PLAN.md` | 实施主合同：接口、逐文件改动、阶段、测试与验收 |
| `ROSE_EA_DESIGN.md` | 论文事实、组合公式、设计理由、改动范围、配置与实验设计 |
| [ROSE_EA_ALGORITHM_COMPARISON.md](ROSE_EA_ALGORITHM_COMPARISON.md) | ROSE/EA 原算法、最终组合、工程等价优化与近似、实际脚本和论文配置差异 |
| `ROSE_EA_WORKLOG.md` | 已执行工作、证据、逐文件实施进度、测试结果与风险 |

已有 `ROSE_VERL_ASCEND_ADAPTATION.md`、`ROSE_VERL_TUTORIAL.md`、`ROSE_IMPLEMENTATION_DECISIONS.md`、`ROSE_CODE_CHANGES.md` 作为 ROSE 历史背景保留。本组文档不将历史配置自动视为本分支运行事实。

## 任务边界

用户希望在已有 ROSE 上加入 EA，并提供基于原脚本的 Ascend 运行入口。本组文档记录设计、已实现内容及测试边界，不记录未经验证的推测为结论。

## 初步推荐

- ROSE 仍负责树式 rollout、语义分支选择、树优势和长度校准。
- EA 使用 actor 当前策略的全词表 Shannon 熵，对 ROSE 最终优势做有界、停止梯度的正向偏移。
- 首版不增加 rollout 次数，不修改奖励或树节点价值，不把语义熵当成 EA 的熵。
- 首版保留 ROSE 基线的 KL、clip、loss 聚合、动态过滤等设置，以单变量对照判断 EA 的作用。
- EA-only 熵在 actor 设备上用无梯度、分块方式计算，优先验证当前 `use_remove_padding=False` 路径。
- 首版确定使用独立 `model_output['ea_entropy']`、EA 配置中的 `entropy_chunk_size`（默认 64），不改变普通 entropy/entropy regularization 通道。

## 实施默认选择与后续实验

1. 首版采用上述最小组合，不改变分支评分或给零优势额外奖励。
2. 长度校准系数：沿用服务器当前基线做 A/B；若基线为 0，再单独安排非零长度校准消融。
3. EA 强度：论文 GRPO 为 0.4，ROSE 无标准差归一化，不能认为同值同强度；建议先 0.1 小试，再比较 0.05/0.2/0.4。
4. 验证预算：先固定 pass@1/pass@8 与速度、长度，再决定是否扩展到更大的 K。

## 重要上下文

- 初始核查 HEAD：`e77dceb55b68416c3d153c3b36a0f13c815733dd`。
- 初始工作区仅 `paper/EA.pdf` 为未跟踪文件；该文件是用户提供的输入，不能删除。
- 本地 `run_rose_new.sh` 仍为 epsilon=0.3、长度 alpha=0、KL=0.01、贪心单样本验证，未显式打开 group filter 或固定长度 loss；此前聊天中的修改建议不能视为已经应用。
- 本地脚本未必等于服务器版本。做基线前必须保存服务器的最终 resolved config、代码 commit 和启动参数。
