# ROSE、EA 与当前 ROSE+EA 实现：算法与工程差异

更新日期：2026-11-12。适用分支：`rose-plus-ea`。

本文回答三个问题：ROSE 原来做了什么，EA 原来做了什么，我们最终把两者组合成了什么。并区分论文定义、代码支持的能力、运行脚本实际启用的配置。

论文来源：[ROSE](paper/ROSE.pdf) 第 3 节及实验配置；[EA：Reasoning with Exploration: An Entropy Perspective](paper/EA.pdf) 第 3.2 节、实验部分及附录 B。代码与配置以本分支为准，服务器另行修改后的版本不自动包含在本文结论中。

## 1. 先给结论

当前实现可以准确概括为：

> 使用 ROSE 的树式采样和分段树优势，在 actor 更新时，使用当前策略的全词表熵，对 ROSE 优势做 EA 的有界、停止梯度的正向偏移，然后交给现有 verl 策略损失更新。

这里存在三层不同的工作：

1. **算法组合**：将 EA 从其论文使用的 PPO/GRPO 优势，扩展到 ROSE 的分段树优势。这是本项目的组合方案，不是两篇论文已经验证过的结果。
2. **工程适配**：接通 verl V1、TransferQueue（下文简称 TQ）、FSDP/FSDP2、当前 actor 熵和 NPU 上的分块计算。
3. **实验选择**：当前脚本保留原 ROSE 基线的大部分设置，方便只改变 EA 做对照，并没有自动改成论文的全部超参数。

因此，“已经实现 ROSE+EA”不等于“已经完整复现 ROSE 论文”，也不等于“已经证明在 NPU 上效果更好”。

## 2. 三种算法分别改变哪里

| 维度 | ROSE | EA | 当前 ROSE+EA |
| --- | --- | --- | --- |
| rollout | 从高语义不确定性位置分支，形成轨迹树 | 核心方法不要求改变 rollout | 保留已有 ROSE 树式 rollout |
| 基础优势 | 根据树节点价值差给不同片段分配优势 | 接受 PPO/GRPO 等已有优势 | 使用 ROSE 分段优势，不换回 GRPO 优势 |
| 长度处理 | 对较长正确轨迹的后缀做长度校准 | 没有 ROSE 的这项校准 | 代码保留，脚本默认 alpha=0，实际关闭 |
| 熵的用途 | 决定从哪里继续探索 | 调整 actor 更新时的优势 | 两种熵分别存在，互不替代 |
| 新增训练偏移 | 无 EA 偏移 | 对优势加一个非负且有上限的 bonus | 在 ROSE 基础优势之后加 EA bonus |
| 新增 critic | 不需要学习一个树价值网络 | 取决于基础算法 | 不因 EA 新增 critic |
| 额外完整轨迹 | 每题按设定生成 G 条树叶轨迹 | EA 本身不增加轨迹 | 仍然 G 条，不因 EA 变成 2G 条 |

## 3. ROSE 原算法

### 3.1 先理解它和普通 GRPO 的采样差别

普通 GRPO 对一个 prompt 独立采样 G 条完整 response，再根据这些 response 的奖励计算组内优势。

ROSE 对一个 prompt 的采样过程是：

1. 先从头生成第一条完整 response。
2. 根据生成时的候选 token 概率和 embedding，为各位置计算语义分支分数。
3. 生成下一条时，以 epsilon 的概率从 prompt 根部重新采样。
4. 否则，从已有轨迹中选择语义分数最高的可用位置，保留该位置之前的前缀，重新生成后缀。
5. 把新轨迹加入已有集合，继续到总共 G 条。

例如 G=8，最终仍有 8 条完整训练轨迹，但它们可能共享 response 前缀，通常不是 8 次独立采样。

如果第二条选择从头生成，第三条也不是固定只参考第一条或第二条。当前实现从已有轨迹集合中寻找候选分支，因此两条都可能成为来源。

同一棵树的后续分支依赖此前生成结果；不同 prompt 的树可以并行。这解释了为什么不能直接把 ROSE 的 G 条轨迹当成 GRPO 那样一次独立并发完成。

这里没有引入按 reward 排序的 beam search、MCTS 或额外 critic。分支依据是语义分数，不是先知道哪条答案正确。

### 3.2 ROSE 的“语义熵”是什么

论文首先定义下一个 token 分布的 Shannon 熵：

```text
H_t = - sum_{v in vocabulary} p_t(v) * log p_t(v)
```

然后使用该位置概率最高的候选 token 的 embedding 衡量语义关系。论文式 (2) 可写为：

```text
SD_t = - sum_{i,j in top-K} p_i * p_j * cos(e_i, e_j)
SE_t = H_t * SD_t
```

论文取 K=20。这里的 top-K 是“某个位置的下一个 token 分布中概率最高的 K 个候选”，不是“整条 response 里 logprob 最高的 K 个已生成 token”。

`SE_t` 是供分支排序使用的组合分数，不是标准 Shannon 熵，不应假设它必定非负，也不应该拿它替代 EA 的全词表策略熵。

### 3.3 树节点价值与分段优势

完整 response 是树的叶子。根节点、分叉节点和终止节点共同组成优势计算所需的树结构。

节点价值由经过该节点的叶子奖励求均值得到：

```text
V(node) = mean(reward of descendant leaf responses)
```

对于两个相邻节点之间的 response 片段：

```text
A_segment = V(end_node) - V(start_node)
```

该片段中的 token 使用相同的基础优势。例如某段从价值 0.25 的节点走到价值 0.75 的节点，该段的 token 优势都是 +0.5。

与普通 GRPO 的区别是：

- 不是把同一个 response-level 组内优势复制到整条 response。
- 不使用 GRPO 的组内奖励标准差归一化。
- 节点价值来自当前采样叶子的奖励统计，不是额外训练出来的价值模型。

当前实现默认要求二值奖励 0/1。若要改成连续、多项或过程奖励，不能只替换 verifier 就认为算法含义完全不变。

### 3.4 长度校准：不是简单地缩小所有优势

ROSE 还对较长的正确轨迹施加长度校准。设同组最短正确轨迹长度为 `L_short`，当前正确轨迹长度为 `L_current`，二者最后公共节点的位置为 `b`：

```text
r = (L_short - b) / (L_current - b)
A_calibrated = A - abs(A) * (1 - r^alpha)
```

校准作用于分叉后的后缀，不是把共享前缀也统一惩罚。

当 `0 < r < 1` 时，alpha 越大，减去的非负项越大。具体而言：

- A 为正：正优势变小。
- A 为负：负优势变得更负，而不是绝对值变小。
- alpha=0：校准项为 0，关闭该功能。
- 正确轨迹不足两条：没有上述正确轨迹间比较，不执行该校准。

因此“alpha 越大，对较长正确轨迹的优势下调越强”中的“下调”指代数值减小，不是所有 token 的训练作用都减弱。

### 3.5 ROSE 和 DAPO 的关系

ROSE 论文的训练使用动态采样：剔除同一 prompt 下全部 reward 相同的组，并继续取新 prompt 补足有效组数。这是 DAPO 中常讨论的组件之一。

但不能因此把 ROSE 简写为“完整使用 DAPO 的全部设置”。论文还讨论 Dr.GRPO 风格的损失修改与 KL；分支采样和树优势本身也不是 DAPO 的普通组内优势。

不开组过滤时，全同奖励的组仍然可以被传入训练，但整棵树的节点价值相同，ROSE 的策略优势为 0。它不会自动被 EA 变成有正向策略信号的组。

## 4. EA 原算法

### 4.1 它改变优势，不改变答案奖励

EA 的核心是在已有优势上加一个由当前策略熵决定的非负偏移：

```text
H_current_t = - sum_v pi_theta(v | prompt, response_<t)
                       * log pi_theta(v | prompt, response_<t)

B_t = min(coef * stop_gradient(H_current_t), abs(A_t) / kappa)
A_used_t = A_t + B_t
```

这里 `kappa > 1`，`coef >= 0`。论文的 EA 系数与 ROSE 长度校准都使用过 alpha 符号，因此本项目把 EA 系数命名为 `coef`，避免混淆。

EA 不直接修改 verifier 的 0/1 reward，也不把熵加到 reward 中重新计算树价值。

### 4.2 必须是当前 actor 的全词表熵

这里的熵来自 actor 计算训练 loss 时的当前参数，而不是以下量：

- rollout 时返回的旧策略熵。
- ROSE 的语义分支分数。
- top-20 候选的部分熵。
- 单个采样 token 的 `-logprob`。

同一批数据若经历多次 actor forward，应使用相应 forward 的当前熵重新计算 bonus，而不是不断把 bonus 累加到存储的基础优势上。

### 4.3 为什么既增强正优势，又减弱负优势

因为 bonus 始终非负，且上限为 `abs(A)/kappa`。当 kappa=2：

- 正优势最多变为原来的 1.5 倍。
- 负优势最多向 0 靠近原绝对值的一半，不会翻成正优势。
- 原优势为 0 时，上限也是 0，EA 不产生额外策略信号。

以 coef=0.1、kappa=2 为例：

| 基础 A | 当前熵 H | bonus | 最终 A_used |
| --- | --- | --- | --- |
| +0.2 | 0.5 | 0.05 | +0.25 |
| -0.2 | 0.5 | 0.05 | -0.15 |
| +0.2 | 10 | 0.10，触及上限 | +0.30 |
| -0.2 | 10 | 0.10，触及上限 | -0.10 |
| 0 | 任意有效非负熵 | 0 | 0 |

“不翻转符号”只针对 policy advantage，不保证加上 KL 等项后完整参数梯度的方向保持不变。

### 4.4 为什么它不是 entropy loss

EA 将熵停止梯度，把它作为优势权重使用，梯度仍然通过原策略损失中的 logprob/ratio 路径传播。

它不是往 loss 中加入 `-entropy_coeff * H` 并对 H 求导。当前脚本 `entropy_coeff=0` 并不意味着 EA 被关闭。

EA 也不保证监测到的策略熵一定上升；更新会改变 token 置信度和后续采样，实际熵、长度和正确率都需要实验观察。

## 5. 最终组合算法

### 5.1 依赖关系

```text
prompt
  -> ROSE 树式 rollout，产生 G 条叶子轨迹和树元信息
  -> verifier 给完整轨迹打分
  -> 树节点价值和分段优势
  -> 可选长度校准，得到基础 A_ROSE
  -> 当前 actor forward，得到 logits 和 logprob
  -> 无梯度计算全词表 H_current
  -> A_used = A_ROSE + min(coef * H_current, abs(A_ROSE) / kappa)
  -> 现有 verl 策略损失及原有 KL 项
  -> backward / optimizer update
```

上图表示算法依赖，不要求把生产代码强行改成同样的串行顺序。实际 rollout 中树元信息的 finalize、奖励调度和数据后处理有各自的生命周期。

### 5.2 EA 放在哪里，哪些地方没有改

EA 放在 actor 的 policy-loss 入口，在基础优势已经确定之后执行：

- 不改 ROSE 的分支概率和分支排序。
- 不改变生成轨迹数。
- 不改变 reward、树节点价值、基础 returns。
- 不原地覆盖 TQ 中保存的基础优势。
- 不新增一次模型 forward；复用当前 actor forward 的 logits。
- 不因 EA 合并或去重共享前缀 token 的训练项。

同一 ROSE 片段内的 `A_ROSE` 相同，但各 token 的当前熵可能不同，因此加 EA 后的 `A_used` 可以不同；如果都达到相同上限，则也可能相同。

共享前缀中原优势为 0 的位置，加 EA 后仍为 0。全同 reward 的组也是如此。不过保留的 KL 等训练项仍可能对这些位置产生作用。

### 5.3 这是不是论文里的完整 ROSE 或 EA

它保留了 ROSE 核心的树采样/树优势，并使用 EA 的核心 shaping 公式，但组合后的算法不是任一论文单独报告的实验设置。

尤其 EA 在 GRPO 中使用的基础优势通常经过组内标准差归一化；本项目用的是未做该归一化的树价值差。即使两边都设 coef=0.4，也不代表作用强度相同。

当前 coef=0.1 是保守的组合初试值，不能因为数值等于 EA 论文某个 PPO 设置，就把当前算法称作 PPO。

## 6. 为工程实现做了什么，以及是否改变算法

### 6.1 ROSE 分支分数：存在明确的近似和约定

对应代码：[semantic_entropy.py](verl/experimental/rose/semantic_entropy.py)。

当前通过 vLLM 返回的有限 top-K 数据计算：

```text
p_i = exp(logprob_i)
H_approx = -sum_{i in returned top-K} p_i * logprob_i
```

这里没有把 top-K 概率重新归一化。它不是论文式 (1) 的全词表 H，截断后的乘积分支分数可能改变 pivot 排名，而不只是统一缩放。

当前语义项默认排除对角线：

```text
SD_impl = -sum_{i != j} p_i * p_j * cos(e_i, e_j)
```

论文印出的双重求和没有明确写排除 `i=j`，因此不能将二者直接说成逐字一致。历史适配文档将排除对角线记录为兼容官方实现的选择；本次没有重新核验该官方源码，不据此扩大一致性结论。

为了避免显式构造 K×K 相似度矩阵，代码利用归一化 embedding：

```text
m = sum_i p_i * e_i
SD_impl = -(dot(m, m) - sum_i p_i^2)
```

这个向量化恒等变换相对于“排除对角线”的约定是数学等价优化；top-K 截断、对角线约定则不是同一种性质的问题。

对应分支调度：[rose_agent_loop_tq.py](verl/experimental/rose/rose_agent_loop_tq.py)。当前扫描已有轨迹中的候选位置，严格更大才更新最优项；同分保留先遇到的候选。有效候选不足两个的位置不作为有效分支，没有有限有效分数时退回根部采样。这些是边界处理约定。

### 6.2 ROSE embedding：静态语义评分表，不是模型输入替代品

对应代码：[prepare_embeddings.py](verl/experimental/rose/prepare_embeddings.py)。

当前流程为：

1. 从指定 checkpoint 读取 input embedding。
2. 按 tokenizer 长度截取所需词表行。
3. 在 FP32 下归一化，以 FP16 mmap 和 JSON 元信息保存。
4. rollout worker 在 CPU 上读取候选 token 的 embedding，计算分支分数。

表来自准备时的 checkpoint，不随每次 actor 参数更新刷新。如果训练改变了 embedding 权重，评分表仍然是原来的表。论文未在此处给出明确的逐步刷新要求，因此应记录这个选择，而不是无证据地断言官方一定每步刷新。

FP16 存储会带来数值近似。CPU 计算位置本身不改变公式，只影响数据传输、吞吐及并发资源。

最重要的是：这张离线表只用于选择分支，不替换 vLLM 的模型 embedding，不承担 tokenizer/decode，也不把语义 embedding 当作 prompt 输入。

### 6.3 前缀复用：复用 token，不等于保证复用 KV cache

分支通过保留 response 前缀的 token IDs，把它们接到原 prompt 后请求 vLLM 继续生成。不是先 decode 成文本再 tokenize。

共享 token 前缀与推理引擎的 KV block 共享是不同层面。当前脚本 `enable_prefix_caching=False`，不能仅凭轨迹有公共前缀就承诺节省 KV 显存、避免重复 prefill 或比 GRPO 更快。

ROSE 的树内依赖、分支请求、top-K 数据返回和 CPU 评分都会影响耗时；实际收益取决于生成长度、并行度和推理引擎缓存行为。

### 6.4 树结构与 TQ：保存信用分配所需信息

对应代码：[rose_agent_loop_tq.py](verl/experimental/rose/rose_agent_loop_tq.py)、[tree_advantage.py](verl/experimental/rose/tree_advantage.py)。

每条叶子轨迹保留自己的训练数据、奖励和 TQ 会话，树元信息记录稳定节点 ID、路径和片段边界。优势计算据此恢复节点成员关系并求价值差。

同步策略版本检查、树路径一致性检查用于避免把不同策略或错误树结构拼在一起。这些属于维护数据契约的工程步骤，不是新增奖励或新价值估计器。

### 6.5 EA 全词表熵：分块计算，但没有截断词表

对应代码：[entropy_advantage.py](verl/workers/utils/entropy_advantage.py)。

EA 熵与 ROSE 的 top-K 近似不同：

- 在 logits 所在设备计算；服务器 actor 在 NPU 时就在 NPU 上算。
- 使用 actor 输出的完整词表维度，不按 top-K 筛选，也不按 tokenizer 长度截断模型输出维度。
- 使用 `torch.no_grad()`，不保留熵计算的反向图。
- 沿 token 维度分块，默认每块 64 个位置；每个位置仍覆盖全词表。
- 块内转换 FP32，并用平移后的稳定公式计算熵。

```text
z = float(logits_chunk) - max(logits_chunk)
H = logsumexp(z) - sum(softmax(z) * z)
```

这与全词表 Shannon 熵在数学上等价，仅有通常的浮点误差。chunk 不是把词表切掉，也不是每 64 个 token 才取一个熵。

该方式限制新增 FP32 中间张量的峰值，但不消除模型本身的完整 logits，也不意味着没有额外 softmax/reduction 开销。

### 6.6 熵必须与“被预测的 response token”对齐

对应代码：[transformer_impl.py](verl/workers/engine/fsdp/transformer_impl.py)、[losses.py](verl/workers/utils/losses.py)。

因果模型位置 t 的 logits 预测的是下一个 token。例如：

```text
输入 token:       p0  p1  r0  r1  eos
logits 位置:       0   1   2   3    4
response 目标:           r0  r1  eos
应取熵的位置:             1   2    3
```

不能直接取 r0、r1、eos 所在输入位置的熵。当前 EA 通道复用 logprob 的布局恢复及 response 对齐，覆盖 dense/packed 路径；EOS 是否计入取决于有效 response mask。

padding 上的优势和熵先清零，有效位置上的 NaN/Inf 则报错，不静默参与 loss。

### 6.7 独立 EA 通道，避免改变既有熵正则

引擎返回独立的 `ea_entropy`，不借用原有普通 entropy 通道改变其梯度语义。loss 入口构造局部 `A_used`，而不是覆盖基础 A。

只有 EA enable 且 coef>0 时请求该额外通道；关闭 EA 或 coef=0 可作为无 EA 对照。当前 active 路径支持范围限定为 V1 同步采样、FSDP/FSDP2、actor SP=1、非 fused、纯文本单轮、vanilla policy loss、训练 temperature=1、entropy_coeff=0 等，具体由配置校验约束。

这不是对 verl 所有 engine、异步策略、多模态或所有 loss 的通用适配。

### 6.8 指标与分布式汇总

对应代码：[entropy_advantage.py](verl/workers/utils/entropy_advantage.py)、[losses.py](verl/workers/utils/losses.py)、[trainer_base.py](verl/trainer/ppo/v1/trainer_base.py)。

指标先保存 token sums/counts，在跨卡、microbatch、minibatch 汇总后再求均值，不直接平均不同长度 batch 的平均值。框架 DP 默认平均的地方做相应补偿。

完整词表 logits/概率不搬 CPU、不写入 TQ；日志统计将 13 个 detached 标量合并搬回 CPU。数值检查仍存在设备同步，不能称为完全没有同步成本。

## 7. 当前运行脚本与论文实验设置的差异

本节的“当前”指 [run_rose_ea.sh](run_rose_ea.sh) 默认值与其继承的配置，不代表用户通过环境变量或命令行覆盖后的最终值。原 [run_rose_new.sh](run_rose_new.sh) 没有因加入 EA 被自动改成论文设置。

| 项目 | 论文设置/定义 | 当前默认与含义 |
| --- | --- | --- |
| 每题轨迹数 | 两篇相关训练实验均使用 8 条 | 8；ROSE 叶子相关，不是独立 GRPO 采样 |
| ROSE 根部重采样 epsilon | ROSE 为 0.5 | 0.3 |
| ROSE 长度 alpha | ROSE 比较 0.5、1、2、3 等值 | 0，实际关闭长度校准 |
| 同奖励组过滤 | ROSE 使用动态采样 | 继承 `filter_groups.enable=False`，未开启 |
| 失败组补采 | 不等同于同奖励组过滤 | `sync_refill_failed_groups=True`，只针对失败组 |
| EA coef | EA：GRPO 0.4，PPO 0.1 | 0.1，组合初试值，不声称同强度 |
| EA kappa | EA 为 2 | 2 |
| KL | ROSE 报告 0.001；EA 附录实验不加 KL loss | 保留基线 KL loss，系数 0.01 |
| 熵正则 | EA 不是普通 entropy loss | `entropy_coeff=0`，独立开启 EA |
| ratio clip | ROSE 0.2/0.2；EA 0.2/0.28 | 0.2/0.2，另有 verl 默认 dual clip |
| loss 聚合 | ROSE 使用 Dr.GRPO 风格修改；EA 附录为 token-mean | 当前继承 token-mean，非自动复刻 ROSE 固定长度缩放 |
| train batch | 相关论文实验为 512 prompts | 128 prompts |
| actor mini batch | EA 附录为 32 prompts | 32 prompts，对应展开后的多条轨迹 |
| 最大 response | ROSE 4096；EA 8192 | 3072 |
| 最大 prompt | ROSE 2048 | 1500 |
| 训练采样 | EA temperature=1；不要把 ROSE 验证参数误作训练要求 | temperature=1，top_p=0.99 |
| 训练数据轮数 | ROSE 最多 8 epochs | 3 epochs |
| 验证 | ROSE 独立采样 8 条，T=0.6、top_p=0.95，关注 pass@8 | n=1、do_sample=False、T=0、top_p=1，贪心单样本 |
| 模型/数据 | 两篇论文使用各自的 Qwen/Llama 模型和数学基准 | Qwen3.5-9B、当前自定义数据和 verifier |
| 硬件 | ROSE 报告 8×A800 80GB | 目标为 Huawei NPU；尚待实测 |

EA 论文的评估还包含重复 pass@1，以及 AIME pass@256、AMC pass@128、MATH500 pass@16 等预算，不等同于当前单样本贪心验证。

### 7.1 当前 loss 不能简单叫“与论文完全一致的 PPO loss”

EA 只替换进入策略损失的优势，保留既有 loss 实现。当前 verl vanilla 路径还包含默认系数为 3 的负优势 dual clip，以及 ratio 的 log 值数值截断等实现细节。

当前 token-mean 按有效 response token 总数归一化。它与论文中按 response 汇总、固定长度缩放等选择的分母不同；长度随训练变化时，梯度尺度及与 KL 的相对权重也可能变化，不能未经核对就称作完全等价。

此外，top_p=0.99 会截断实际 rollout 采样分布，而 EA 使用当前 actor 完整词表的熵。当前组合没有另外校正树相关采样和截断采样带来的所有分布差异。

### 7.2 两种补采必须区分

`sync_refill_failed_groups=True` 表示 rollout 失败后补充失败组，不能替代 `algorithm.filter_groups.enable=True`。

同奖励组过滤开启后，逻辑是移除没有组内奖励差异的 prompt group，继续消耗后续 prompt 以补足训练组数；并不等于对同一题无限重试直到出现不同 reward。数据消耗和每 epoch 的实际更新步数会因此改变。

### 7.3 三组容易混淆的参数

| 参数 | 控制内容 | 不控制什么 |
| --- | --- | --- |
| `ROSE_TOP_K=20` | 每个位置用于 ROSE 语义评分的候选数 | 不限制 EA 全词表熵 |
| `ROSE_SEMANTIC_CHUNK_SIZE=256` | CPU 语义评分每块处理的位置数 | 不表示生成长度，也不是候选数 |
| `EA_ENTROPY_CHUNK_SIZE=64` | actor 设备上每块计算全词表熵的位置数 | 不截断词表，不改变 EA bonus 上限 |
| `ROSE_ALPHA=0` | ROSE 长度校准强度 | 不是 EA 系数 |
| `EA_COEF=0.1` | EA 熵偏移的缩放系数 | 不改变 ROSE 分支概率 |
| `ROSE_EPSILON=0.3` | 下一条轨迹从根部重采样的概率 | 不表示普通 PPO clip epsilon |

## 8. 哪些属于等价优化，哪些不是

| 选择 | 性质 | 需要如何解释 |
| --- | --- | --- |
| EA token 分块、FP32 稳定熵 | 数学等价计算，存在浮点误差 | 降低新增中间张量峰值，不减少词表 |
| EA no_grad、局部优势、不累加 bonus | 遵循 EA 的梯度/更新语义 | 不是为了提速随意切掉必要梯度 |
| TQ 元信息、mask/shift、分布式加权指标 | 数据/执行契约适配 | 正确传递和统计算法量 |
| ROSE 相似度向量化 | 对选定对角线约定等价 | 避免显式 K×K 矩阵 |
| ROSE top-K 部分熵替代全词表 H | 算法近似 | 可能改变分支排序 |
| ROSE 排除对角线 | 公式约定差异 | 不与论文印出公式直接画等号 |
| 静态 FP16 embedding 表 | 实现选择与数值近似 | 不随 actor 更新；并非 tokenizer 流程 |
| CPU 评分、NPU actor/rollout | 执行位置选择 | 主要影响性能，不天然改变数学公式 |
| 长度 alpha=0、过滤关闭、KL=0.01 | 实验配置差异 | 不是 NPU 强迫关闭的功能 |
| ROSE 优势上叠加 EA | 新算法组合 | 需要独立消融验证收益 |

## 9. 实验应该如何安排

### 9.1 先验证“加 EA 是否有效”

使用同一个脚本，固定模型、数据、reward、长度、KL、batch、过滤和验证预算，仅开关 EA：

```bash
# 在相同平台环境、相同模型/数据路径下运行，使用不同输出目录。
EA_ENABLE=False EXP_NAME=rose_control bash run_rose_ea.sh
EA_ENABLE=True EA_COEF=0.1 EA_KAPPA=2 EXP_NAME=rose_ea_trial bash run_rose_ea.sh
```

这回答“在当前 ROSE 工程基线上加 EA 有什么影响”，不是完整论文复现。

### 9.2 若要按 ROSE 论文复现，另设实验线

先逐项对齐 ROSE 的长度校准、动态过滤、损失聚合与缩放、KL、采样和验证设置，以及模型/数据/长度预算，然后在这个新基线上做 EA off/on。

不要在同一次实验中同时改 alpha、过滤、KL、batch 和 EA，再把所有变化归因于 EA。即便超参数对齐，模型和数据不同也不能称为原论文结果的直接复现。

### 9.3 需要记录哪些指标

- 任务正确率和明确采样协议下的 pass@1/pass@8。
- response 长度、截断率、生成 token 数、有效 prompt/组数。
- rollout 与 actor update 分项耗时、NPU 峰值显存和端到端时间。
- `actor/ea/entropy_token_mean`：当前 actor 的有效 response token 全词表熵。
- `actor/ea/bonus_ratio_mean`：非零基础优势 token 上 bonus/abs(A) 的均值。
- `actor/ea/cap_fraction`：达到 bonus 上限的非零优势 token 比例。
- `actor/ea/sign_flips`：应为 0；非零时先排查实现/数值问题。

EA 不新增 rollout 次数，但可能改变生成长度，进而改变后续 rollout 成本。比较时同时看更新步数、token 预算和真实训练时间，不能预先承诺更快或必然提高正确率。

保存每次 resolved config、环境变量覆盖、随机种子、代码版本及部署快照；只保存启动脚本不足以完整还原运行。设置种子也不保证异步调度和硬件算子的逐位确定性。

## 10. 当前完成度与文档导航

已完成 P0-P5 的工程实现及本地验证。历史验证记录为 **164 passed、1 skipped**，跳过项为 macOS 缺少 Linux `/proc` 的现有检查；详细命令见 WORKLOG。本文新增时不重新执行算法测试，不能把历史 CPU 测试描述为新跑出的 NPU 训练结果。

仍待服务器完成 P6 NPU smoke/显存/性能验证，以及 P7 算法效果实验。本地通过不代表 9B、多节点 HCCL 和真实训练已经验证成功。

- [工作入口与当前状态](ROSE_EA_README.md)
- [算法与适配设计](ROSE_EA_DESIGN.md)
- [详细实施方案](ROSE_EA_IMPLEMENTATION_PLAN.md)
- [实施与验证日志](ROSE_EA_WORKLOG.md)
- [ROSE+EA 运行脚本](run_rose_ea.sh)

最后用一句话区分：**ROSE 决定“从哪里采样、怎样按树分配信用”；EA 决定“在更新当前策略时，怎样根据当前熵调整这些信用的权重”；verl/NPU 适配负责把二者正确、高效地执行起来。**
