# ROSE 适配实施：代码改动日志

本文记录 `ROSE-adaptation` 分支为实现 ROSE 所做的逐文件改动、接口变化、验证命令和结果。它描述“改了什么”；设计取舍见 `ROSE_IMPLEMENTATION_DECISIONS.md`。

## 基线与范围

- 分支：`ROSE-adaptation`
- 初始 commit：`f7d6513d`
- 目标：VeRL V1 sync trainer + vLLM/vllm-ascend + Ascend NPU
- 首版范围：单轮纯文本、binary rule-based reward、树内顺序生成、prompt 间并行
- 显式不支持：fully async、multi-turn/tool、multimodal、speculative decoding、NPU semantic scorer

## 文档

### `ROSE_VERL_ASCEND_ADAPTATION.md`

- 记录论文算法、VeRL 扩展点、CPU/NPU 边界、配置、部署、测试和验收方案。
- 更新为当前分支实际路径和配置字段。
- 增加实现状态表，区分本地已验证项与 Ascend 待验证项。

### `ROSE_IMPLEMENTATION_DECISIONS.md`

- 记录每一步的背景、候选方案、最终决策和验证边界。
- 重点记录 V1 AgentLoop 选型、host CPU scorer、tree metadata、gitlink 迁移和 fail-closed 策略。

### `ROSE_CODE_CHANGES.md`

- 记录本文所列的逐文件改动与测试结果。

## Rollout top-K 通路

### `verl/workers/rollout/logprobs.py`

- 新增 `normalize_requested_logprobs()`：兼容 `None/False/True/0/K`，保留整数 top-K 请求。
- 新增 `pack_topk_logprobs()`：把每位置 mapping 压成固定形状数组。
- 优先按 vLLM `Logprob.rank` 写槽位；rank 缺失时按 logprob 降序补齐。
- 输出类型固定为 token ID `int32`、logprob `float32`、valid mask `bool`。
- helper 放在通用 rollout 层，使 CPU 测试不需要导入 vLLM package。

### `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

- 修复整数 `logprobs=20` 被 truthy 判断降级为 `0` 的问题。
- sampled-token logprob 行为保持不变。
- 当请求值大于 0 时，把 top-K dense arrays 写入临时 `TokenOutput.extra_fields`。
- 不改 vLLM-Ascend sampler 内部逻辑，保持对后端版本的低侵入性。

## ROSE 算法模块

### `verl/experimental/rose/semantic_entropy.py`

- 新增 normalized embedding mmap loader。
- 按 chunk 读取 top-K embedding，使用 FP32 累加计算 generation entropy、semantic divergence 和 semantic entropy。
- 使用加权 embedding 范数恒等式，避免构造 `[L,K,K]` cosine tensor。
- 支持 valid mask、OOV 检查、NaN/Inf 检查、embedding 文件大小检查。
- 提供 `from_array()` 供纯算法单测使用。

### `verl/experimental/rose/prepare_embeddings.py`

- 从与 rollout 相同的 Hugging Face checkpoint 读取 input embedding。
- 以 FP16 加载模型，embedding 转 FP32 做逐行 L2 normalize，再保存为 flat FP16。
- 输出 JSON metadata：模型路径、vocab size、hidden size、dtype、normalized 标志和 SHA256。

### `verl/experimental/rose/tree.py`

- 新增 `BranchRecord` 和 `finalize_tree()`。
- 为 root、pivot、terminal 分配稳定 node ID。
- 支持后生成 pivot 落在已有 edge 内时拆边，并更新所有经过该 edge 的旧 leaf path。
- 输出每条 leaf 的 parent、branch、root restart、`path_node_ids` 和 `path_ends`。

### `verl/experimental/rose/tree_advantage.py`

- 注册自定义 estimator 名称 `rose_tree`。
- 以经过 node 的 leaf reward 均值计算 node value。
- 用相邻 node value 差填充 segment-level advantage。
- 对正确 leaf 使用真实 node-path LCA 做 length-aware calibration。
- 跳过零 response mask 的 padding row；默认强制 binary reward。
- 校验 tree ID/uid、路径端点、node 位置、node ancestry、parent 存在性、parent graph 无环、branch prefix、root 和 terminal 唯一性。

### `verl/experimental/rose/rose_agent_loop.py`

- 新增单轮纯文本 token-in/token-out helper。
- 初始 prompt 复用 Continuous Token builder。
- 分叉续写输入为 `prompt_ids + prefix_ids`，不做 decode/re-tokenize。
- prefix sampled logprob 原样复用，continuation logprob 与 token 严格对齐。
- 读取并立即移除临时 top-K payload，保证其不进入 TransferQueue。

### `verl/experimental/rose/rose_agent_loop_tq.py`

- 新增 ROSE V1 manager/worker。
- prompt 之间保留 Ray worker 并行；单 prompt 内按 trajectory 顺序生成。
- 实现 `ε` root restart、全局最大有限 semantic entropy pivot 和无有效 pivot 时 root fallback。
- prefix token/logprob/entropy 直接复用。
- 一棵树生成完成后统一 finalization，再逐 leaf 独立 reward/postprocess/TQ 写入。
- 校验每条 trajectory 不跨 policy version，整棵树版本一致且等于当前 step。
- 拒绝 fully async client、非 vLLM backend、multi-turn 和非 CPU semantic scorer。

## VeRL 数据通路

### `verl/trainer/ppo/v1/agent_loop_tq.py`

- 将原 worker 类体提取为可继承的 `AgentLoopWorkerTQBase`。
- 用 `AgentLoopWorkerTQ = ray.remote(AgentLoopWorkerTQBase)` 保留原公共 remote symbol。
- 默认 manager 只在子类未指定 worker class 时设置原 worker，默认行为保持不变。

### `verl/trainer/ppo/v1/trainer_base.py`

- `_compute_advantage()` 按 `algorithm.advantage_extra_fields` 额外读取 TQ nested `extra_fields`。
- 新增 `_extract_advantage_extra_fields()`，把可变结构保存为一维 object ndarray。
- 任意非 mapping row 或缺失字段立即抛错，不把 `None` 静默传给 estimator。

### `verl/trainer/ppo/ray_trainer.py`

- 通用 custom-estimator dispatch 把配置指定的 non-tensor 字段加入 estimator kwargs。
- 字段不存在时 fail closed。

## 配置

### `verl/workers/config/rollout.py`

- 新增 `RoseRolloutConfig`：enable、trajectory 数、epsilon、top-K、embedding 路径、CPU chunk、对角项和 validation mode。
- 挂载为 `RolloutConfig.rose`。

### `verl/trainer/config/algorithm.py`

- 新增 `RoseAlgorithmConfig`：length calibration alpha、binary reward 开关和容差。
- 新增通用 `AlgoConfig.advantage_extra_fields`。
- 挂载为 `AlgoConfig.rose`。

### 默认与 generated YAML

- `verl/trainer/config/rollout/rollout.yaml`：增加 custom manager 和 ROSE rollout 默认字段。
- `verl/trainer/config/ppo_trainer.yaml`：增加 advantage metadata 和 ROSE algorithm 默认字段。
- 四份 `_generated_ppo*.yaml` 使用 `scripts/generate_trainer_config.sh` 重新生成；生成结果与先前手工同步内容逐字一致。

## Ascend 示例

### `examples/ascend_extras/rose/README.md`

- 说明支持范围、embedding 准备、环境变量、启动方式和 CPU/NPU 边界。

### `examples/ascend_extras/rose/rose_npu.yaml`

- 记录 canonical ROSE/Ascend overrides，作为配置审阅参考。

### `examples/ascend_extras/rose/run_qwen3_4b_fsdp2.sh`

- 提供 Qwen3-4B-Base、FSDP2、vLLM-Ascend 的启动模板。
- 默认论文参数：`G=8`、`epsilon=0.5`、top-K=20、LR `1e-6`、clip `0.2`、KL `0.001`。
- 支持通过环境变量缩小为 `G=2`、短 response 的首轮 smoke test。

## 测试

### `tests/experimental/rose/test_semantic_entropy_on_cpu.py`

- 覆盖 bool/int logprobs、rank/fallback packing、pairwise reference、chunking、mask、OOV、无效 chunk 和 mmap 文件大小。

### `tests/experimental/rose/test_tree_advantage_on_cpu.py`

- 覆盖拆边、root restart、segment value、padding、batch reorder、真实 LCA calibration、binary reward、parent 关系、cycle 和通用 estimator metadata forwarding。

### `tests/experimental/rose/test_rollout_helpers_on_cpu.py`

- 覆盖全局 pivot 选择、无有限 pivot、policy-version 原子性、TQ worker base API、extra-field 提取、fake-server prefix/top-K 生命周期和 ROSE rollout 性能指标。

### `tests/experimental/rose/test_rose_config_on_cpu.py`

- 覆盖合法 CPU 配置，以及 trajectory 数、epsilon、top-K、semantic device/chunk、validation mode、embedding 路径和 algorithm 数值的 fail-closed 校验。

## 验证记录

使用官方 `uv` 在项目 `.venv` 创建 Python 3.12.14 环境。当前机器没有 NPU/vLLM-Ascend，以下均为 CPU/静态验证。

```bash
env PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest \
  tests/experimental/rose \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  tests/trainer/test_multi_trajectories_advantage_on_cpu.py \
  tests/test_base_config_on_cpu.py \
  tests/utils/test_config_on_cpu.py -q
```

结果：ROSE 目标与相邻回归测试通过，`49 passed`。

```bash
.venv/bin/python -m pytest \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py \
  tests/trainer/ppo/v1/test_metrics_aggregator_on_cpu.py \
  tests/trainer/ppo/v1/test_rollout_profiling_on_cpu.py \
  tests/workers/rollout/rollout_vllm/test_mtp_hybrid_sleep_acceptance_on_cpu.py -q
```

结果：trainer/rollout metrics 与 speculative decoding 相邻回归通过，`36 passed, 1 skipped`。

```bash
.venv/bin/python -m pytest \
  tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py \
  tests/trainer/ppo/v1/test_compute_reward_colocate_on_cpu.py -q
```

结果：沙箱内 Ray/psutil 因 macOS `sysctl` 权限失败；在沙箱外原命令 `57 passed`。

其他验证：

- ROSE 目录目标测试：`40 tests collected`，全部通过。
- `tests/special_sanity/validate_structure.py`：通过。
- `tests/special_sanity/check_example_naming.py --root examples`：104 个示例脚本全部通过。
- Ruff lint：通过。
- Ruff format check：通过。
- `python3 -m compileall`：通过。
- `bash -n examples/ascend_extras/rose/run_qwen3_4b_fsdp2.sh`：通过。
- `git diff --check`：通过。
- Hydra 使用实际 ROSE overrides 组合成功。
- 四份 generated config 与生成器输出一致。

## 尚需目标 Ascend 环境验证

- vLLM-Ascend 在 TP=1 和生产 TP 下 `logprobs=20` 的返回结构与 rank 语义。
- 单 prompt、`G=2`、短 response 的真实 tree metadata 和一步 actor update。
- checkpoint 保存/恢复、目标 TP、多机 HCCL 和至少 100 step 稳定性。
- host CPU semantic scorer 占 rollout 时间的比例；超过 10% 时再评估独立 NPU scorer。

这些项目不能在当前 macOS 工作区内伪造通过，必须在用户的 Ascend 服务器上执行。
