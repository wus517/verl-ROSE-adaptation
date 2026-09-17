# ROSE 适配实施：思考与决策日志

本文记录在 `ROSE-adaptation` 分支上实现 ROSE 时的关键分析、选择、备选方案和验证依据。它描述“为什么这样做”，不替代用户文档或代码注释。

## 记录约定

- 每项决策包含背景、候选方案、结论和后续验证。
- 以当前工作区为准；基线变化时重新核对，不沿用未经验证的旧结论。
- 算法定义优先参考 `paper/ROSE.pdf`，工程接口优先参考当前 VeRL 源码。
- 对论文公式、作者公开实现和本项目工程增强进行明确区分。

## D001：以当前分支源码重新审计，而不是直接应用旧设计

**状态：已决定**

当前分支为 `ROSE-adaptation`，初始基线为 `f7d6513d`。设计文档最初基于更早的主分支快照，因此所有接口必须重新核对。

已确认：

- V1 自定义 manager 入口仍为 `actor_rollout_ref.rollout.agent.agent_loop_manager_class`。
- 默认 `AgentLoopWorkerTQ._run_prompt()` 会并发启动同一 prompt 的 `n` 个 session，不满足 ROSE 树内顺序依赖。
- `AgentLoopOutput.extra_fields` 仍可承载动态 metadata。
- rollout 和 algorithm 配置已使用 dataclass schema；正式字段应进入 schema 和默认 YAML。

结论：实现基于当前 V1 AgentLoop/TransferQueue，不复制旧同步 rollout。

## D002：维护独立的实现决策与代码改动文档

**状态：已决定**

用户要求工作过程中维护两个 Markdown 文档。采用：

- `ROSE_IMPLEMENTATION_DECISIONS.md`：记录分析、取舍和验证结论。
- `ROSE_CODE_CHANGES.md`：记录逐文件修改、接口变化和测试结果。

两份文档都放在仓库根目录，并随实现同步更新。

## D003：树内顺序、树间并行

**状态：已决定**

ROSE 第 `r` 条 trajectory 的生成 prefix 依赖前面 trajectory 的 semantic entropy。候选方案：

1. 修改默认 manager，让所有 session 都串行。
2. 自定义 ROSE manager/worker，只让单个 prompt 内串行，不同 prompt 继续并行。

选择方案 2。原因：保持默认行为不变，并保留 prompt 级并发吞吐。

## D004：首版支持单轮纯文本，其他模式显式拒绝

**状态：已决定**

ROSE prefix branching 需要精确控制 response token prefix。multi-turn、tool observation 和 multimodal processor 会引入非模型 token、动态模板和额外 mask 语义。

首版约束：

- 只支持 `single_turn_agent`。
- 只支持纯文本 prompt。
- validation 使用默认独立 rollout。
- 对 multi-turn、tool 和 multimodal 输入 fail closed，而不是静默产生错误树。

后续扩展必须单独定义 pivot 能否落在 observation/tokenizer expansion 区域。

## D005：semantic entropy 默认在服务器 host CPU 计算

**状态：已决定，待性能验证**

理由：

- vllm-ascend 已把生成结果包装到 host 侧。
- 树控制位于 AgentLoop worker。
- 避免依赖 vllm-ascend 模型内部结构和 TP vocabulary shard。
- 避免在每个 rollout rank 上复制完整 embedding。

实现要求：normalized FP16 mmap embedding、FP32 累加、chunked vectorization、prefix score 复用。若 profiling 表明 scoring 超过 rollout 时间 10%，再实现独立 NPU scorer。

## D006：top-K 是临时控制数据，不进入训练 batch

**状态：已决定**

vLLM server 将每个位置的 top-K 压缩为连续数组，通过 `TokenOutput.extra_fields` 返回。ROSE worker 计算 semantic entropy 后立即移除这些字段。最终 TransferQueue 只保存 response、sampled logprob、reward 和紧凑 tree metadata。

这样避免长回答的 top-K 数据占据 Ray object store 和 replay buffer。

## D007：树节点使用稳定 node ID，并在生成完成后 finalization

**状态：已决定**

仅使用 token position 会合并同深度不同 prefix。并且后续分叉可能落在早期 leaf 的一条已有 edge 内，需要把该 edge 拆分并更新旧 leaf path。

因此 rollout 先记录不可变 branch event，生成完成后统一构建最终树：

- root、pivot、terminal 都有稳定 node ID。
- 每条 leaf 导出等长的 `path_node_ids` 和 `path_ends`。
- estimator 只消费 finalization 后的路径，不猜测 batch 排列。

## D008：tree advantage 作为注册 estimator，trainer 只提供通用 metadata 通路

**状态：已决定**

不把 ROSE 算法硬编码进 V1 trainer。新增通用 `algorithm.advantage_extra_fields`，把配置指定的 non-tensor 字段传给 estimator；`verl.experimental.rose.tree_advantage` 注册 `rose_tree`。

这样 trainer 改动保持通用，其他树算法也可复用。

## D009：每条 leaf 是独立 TQ session，不能复用多输出 reward 语义

**状态：已决定并实施**

默认 `AgentLoopWorkerTQ._agent_loop_postprocess()` 把 list output 解释为同一 session 的多个阶段，reward 只计算最终 output 并复制给前序 output。ROSE 的每条 leaf 有独立 reward，因此：

- 树在 worker 内完整生成。
- 每条 leaf 分别调用一次 `_agent_loop_postprocess()`。
- key 使用 `{uid}_{trajectory_id}_0`。
- 只有所有 leaf 后处理结束后，prompt group 才标记为 `finished`。

## D010：为 TQ worker 提取可继承基类，同时保留公共远程类

**状态：已决定并实施**

Ray 的 `@ray.remote` 装饰结果不能作为普通 Python 基类。为避免复制整段 TQ 写入逻辑，将原 worker 类体改为 `AgentLoopWorkerTQBase`，随后用：

```python
AgentLoopWorkerTQ = ray.remote(AgentLoopWorkerTQBase)
```

恢复原公共符号。默认 manager 仍使用相同的 `AgentLoopWorkerTQ`，行为不变；ROSE worker 继承 base 并单独包装为 Ray actor。

## D011：纯文本 prefix 直接按 token ID 拼接

**状态：已决定并实施**

初始 prompt 仍使用 VeRL Continuous Token builder 生成。分叉后给 vLLM 的输入为：

```text
initial_prompt_ids + parent_response_ids[:branch_pos]
```

返回 continuation 后，训练 response 为 prefix 与 continuation 的 token ID 直接拼接。该路径仅对单轮纯文本开放；不经过文本 decode/re-tokenize，因此不会改变已经采样的 prefix token。

## D012：没有有效 semantic pivot 时回退 root

**状态：已决定并实施**

top-K 有效候选少于 2 的位置被标记为不可选。如果当前树不存在任何有限 semantic entropy，下一条 trajectory 从 root 重新生成，并保留指标扩展空间。这样不会选 padding/OOV 位置，也不会让整个 prompt 因后端少量缺失 top-K 数据失败。

## D013：主实现不能放在 `recipe/` gitlink 中

**状态：已决定并实施**

当前仓库的 `recipe` 是未初始化的 git submodule，而不是根仓普通目录。把新文件写入 `recipe/rose/` 会导致它们无法被根仓 git 跟踪，最终提交也不会包含实现。

结论：

- 主实现放到 `verl/experimental/rose/`。
- Ascend 配置和启动脚本放到 `examples/ascend_extras/rose/`。
- 测试放在与源码顶层模块对应的 `tests/experimental/rose/`，import 指向 `verl.experimental.rose`。
- 删除本轮误放到 gitlink 中的源码和编译缓存。

## D014：top-K 纯数据 helper 放在通用 rollout 层

**状态：已决定并实施**

最初把 helper 放在 `verl/workers/rollout/vllm_rollout/logprobs.py`。CPU 测试导入这个子模块时，会先执行 `vllm_rollout/__init__.py`，从而在未安装 vLLM 的算法测试环境中失败。

选择把无后端依赖的 bool/int 规范化和 dense packing 移到 `verl/workers/rollout/logprobs.py`。vLLM server 仍是当前唯一调用方，但纯 NumPy helper 可以独立测试，也为后续其他 backend 返回同一 ROSE payload 留出复用空间。

## D015：tree metadata 和 policy version 必须 fail closed

**状态：已决定并实施**

仅校验 `path_node_ids/path_ends` 的形状不足以防止错误 credit assignment。estimator 现在还校验：

- parent 必须存在且 parent graph 无环；
- `root_restart`、parent 和 `branch_pos` 语义一致；
- child 与 parent 在 branch node 之前拥有相同真实 node path；
- 所有 leaf 共享唯一 root，terminal node 不复用；
- 同一 node ID 的位置和祖先签名一致。

rollout 侧同时校验每条 leaf 的 `min_global_steps == max_global_steps`，整棵树只有一个 version，且该 version 等于当前 prompt 的 `global_steps`。fully async client 在 worker 初始化时直接拒绝。

## D016：embedding 保持显式离线步骤

**状态：已决定并实施**

不在训练启动脚本内隐式加载完整 checkpoint 并生成 embedding。原因是多机训练时这会在每个节点重复占用 CPU 内存并延长 Ray 启动，且容易在共享 NFS 上形成热点。

采用显式命令：

```bash
python3 -m verl.experimental.rose.prepare_embeddings \
    --model-path /models/Qwen3-4B-Base \
    --output-path /local_nvme/rose/qwen3_4b_embeddings.f16
```

工具使用 FP16 加载、FP32 归一化并输出 flat FP16 文件和 JSON metadata。训练节点通过只读 mmap 加载，文件长度必须与 metadata 中的 shape/dtype 完全一致。

## D017：验证分层为本地 CPU 证据和目标 NPU 证据

**状态：已决定，本地部分已完成**

本地 macOS 可以证明公式、树结构、metadata 转发、默认 V1 行为和 replay/TQ 逻辑，但不能证明 vLLM-Ascend 的 `Logprob.rank`、HCCL、NPU 内存或生产 TP 行为。

因此验收分两层：

1. 本地 Python 3.12 `.venv`：ROSE 目标测试、相邻配置/advantage/TQ 测试、Ruff、compileall、脚本语法和配置生成。
2. Ascend 服务器：top-20 smoke test、`G=2` 一步 actor update、目标 TP、多机和至少 100 step 稳定性测试。

不得用本地测试结果替代第二层证据。

## 待决策/验证

- 需要在实际 vllm-ascend 上验证当前策略：优先使用 `Logprob.rank`，缺失 rank 时按 logprob 降序补齐。
- 当前 TransferQueue/replay buffer 回归测试已通过；仍需在真实 Ray + vLLM-Ascend rollout 中验证 `extra_fields.rose_tree_metadata` 的端到端序列化。
- NPU 环境下目标模型与生产 TP 的 top-20 smoke test。
- 根据 Ascend profiling 决定是否实现独立 NPU semantic scorer；首版没有实现该可选路径。
