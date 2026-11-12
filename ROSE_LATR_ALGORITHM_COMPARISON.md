# ROSE、LATR 与当前融合算法：原理、实现及差异

## 1. 阅读前先明确结论

当前 `rose-plus-latr` 分支实现的是：

> ROSE with sampled lookahead pruning：ROSE 负责选择分支点，随机短程试跑与编辑距离负责筛选分支，最后继续使用 ROSE 的树结构优势估计。

它不是完整 LATR 的逐项复现，也不是将两篇论文的所有模块原封不动叠加。

可以用三句话理解：

- **ROSE：在哪里分支，以及怎样根据树给 token 分配优势。**
- **LATR：分支之后先向前看一段，剪掉很快与父路径重新趋同的分支。**
- **当前融合：先按 ROSE 找位置，再借鉴 LATR 做短程差异检查，接受后才生成剩余长响应。**

“差异足够大”不等于“答案正确”，也不等于“数学推理语义一定不同”。当前剪枝使用 token 编辑距离，不使用正确答案、verifier 或额外 reward model 预测分支质量。

### 1.1 本文对应的状态

- 日期：2026-11-12。
- 分支：`rose-plus-latr`。
- 首版 A 的代码、配置、独立脚本已实现；源码入口见第 9 节。
- 既有本地验证记录为 135 项 CPU/mock 测试通过，详细命令和限制见 [工作日志](ROSE_LATR_WORKLOG.md)。本次仅整理文档，没有重新运行这组测试。
- 尚未完成 Ascend NPU 实机 rollout、两节点训练、actor update 与效果/速度验收。
- 本文区分“论文算法”“当前实现”“脚本默认值”和“待验证效果”。不得将这些概念互相替代。

配套材料：

- [项目导航](ROSE_LATR_PROJECT.md)
- [详细实施方案](ROSE_LATR_IMPLEMENTATION_PLAN.md)
- [原始设计与取舍](ROSE_LATR_DESIGN.md)
- [运行指南](ROSE_LATR_RUN_GUIDE.md)

## 2. 原始 ROSE 算法

依据：[ROSE 论文](paper/ROSE.pdf) §3.1–3.3、§4.3。

### 2.1 普通 GRPO 的 rollout 是什么

对一个 prompt，独立采样 G 条完整 response。例如 G=8：

```text
prompt -> response 0
       -> response 1
       -> ...
       -> response 7
```

各响应之间没有“根据另一条响应中途分支”的算法依赖，可以并发提交给生成引擎。实际后端是否合并为单个请求、如何调度，是框架实现问题。

普通 GRPO 通常根据组内最终 reward 计算响应级优势，再将同一响应的优势用于其 token。

### 2.2 ROSE 如何生成 G 条轨迹

ROSE 不是一开始独立生成 G 条：

1. 从 prompt 根部生成第一条完整响应。
2. 对该响应的各位置计算语义熵。
3. 决定下一条响应的生成方式：
   - 以 epsilon 概率从根部独立生成。
   - 否则，从所有已有完整响应中选择高语义熵位置，保留此前缀，重新采样后缀。
4. 对新响应计算语义熵，并将它加入已有响应集合。
5. 重复，直到获得 G 条完整响应。

因此，如果第二条从根部生成了，第三条不是固定沿第一条或第二条分支，而是可以在所有已有完整响应中选点。新生成且保留的响应，也可以成为后续响应的父轨迹。

这里的“树”来自轨迹之间的实际继承关系，而不是仅仅把几条字符串摆成树形。

### 2.3 语义熵为何需要 embedding

普通策略熵表示“下一 token 的概率是否分散”，但不能区分候选 token 在语义上是否相似。ROSE 额外使用候选 token 的 embedding 衡量差异。

论文的基本表达是：

```text
H_t  = - sum_v p_t(v) * log p_t(v)
SD_t = - sum_{i,j in V_t} p_t(i) * p_t(j) * cosine(e_i, e_j)
SE_t = H_t * SD_t
```

其中 `V_t` 是当前位置的高概率候选 token 集合，论文使用 top-20。

注意这三个概念不同：

- top-20：同一个生成位置上的 20 个候选 token，不是整条 response 中概率最高的 20 个位置。
- 语义熵：ROSE 的分支评分，不等于训练日志里的策略熵。
- 候选 embedding：用于评分的输入 embedding 查表，不是把 prompt 改写成另一份 embedding 再传给 rollout。

本项目正常生成仍把 token IDs 交给 vLLM，模型内部照常执行 embedding 和前向计算。额外的 CPU embedding 表只参与 ROSE 选点，不替代模型的输入 embedding。

### 2.4 ROSE 的树结构优势估计

生成完整响应后，使用 verifier 得到最终 reward。对于每个树节点：

```text
V(node) = 所有经过该节点的最终响应的 reward 均值
```

相邻节点之间的整个 segment 使用：

```text
A(segment) = V(segment_end) - V(segment_start)
```

例如，根节点下共有 8 条响应，其中 4 条正确，则根节点 value 为 0.5。某条共享前缀后的子树有 2 条响应且都正确，则该子节点 value 为 1，该段优势为 `1 - 0.5 = 0.5`。

它不需要额外训练一个 critic，也不是用模型在线预测中间节点 reward；节点 value 是由最终叶子 reward 回算的。

### 2.5 长度校准 alpha

在正确响应中找到最短正确响应 `o_s`。对其他正确响应 `o_c`，找到二者树路径的分叉位置 `b_c`，对分叉后的 token 调整优势：

```text
q  = (len(o_s) - b_c) / (len(o_c) - b_c)
A' = A - abs(A) * (1 - q**alpha)
```

较长正确响应通常有 `0 < q < 1`。alpha 越大，`q**alpha` 越小，因此下调越强。

这不是简单地“优势全部乘一个长度系数”：正优势被减小；负优势还会变得更负。`alpha=0` 时该校准关闭。

### 2.6 ROSE 与 DAPO 的准确关系

ROSE 论文 §3.1 明确使用 dynamic sampling，过滤组内 reward 全相同的 prompt group。

但不能因此说“ROSE 的训练目标就是完整 DAPO”：论文 §3.3 写的是采用 Dr.GRPO 改进的优化目标并加入 KL penalty。动态组过滤只是可单独辨认的训练机制，不等于同时启用了 DAPO 的全部 loss、clip 和 KL 设置。

本项目的 `algorithm.filter_groups.enable` 控制外层组过滤。lookahead 开关不负责自动开启它；当前融合也没有为了组合算法而修改它。

### 2.7 原论文的部分实验设置

ROSE §4.3 给出的设置包括：G=8、batch size=512、学习率 `1e-6`、clip=0.2、KL 系数 0.001、最大生成长度 4096、默认 epsilon=0.5；alpha 从 `{0.5, 1, 2, 3}` 搜索。

验证使用 temperature=0.6、top-p=0.95、每题 8 条候选，报告 pass@8。这一段是验证配置，不能误写为训练采样要求。

这些数值用于理解原论文，不代表当前脚本已经逐项照搬。

## 3. 原始 LATR 算法

依据：[LATR 论文](paper/LATR.pdf) §3、Algorithm 1、Appendix B.3。未核对 LATR 官方仓库，因此本文不替官方代码补充论文没有说明的边界行为。

### 3.1 与 ROSE 的起点不同

LATR 维护的是正在生成中的部分序列树，而不是先拿到一条完整 response 再回到历史位置分支。

在搜索阶段，对各活动分支：

1. 用最高概率 token 扩展主路径。
2. 检查其他候选 token 是否满足概率条件。
3. 满足条件且树宽允许时，建立子分支。
4. 分支继续生成，达到 lookahead 检查时机后，与父分支比较短程差异。
5. 差异不足时，删除子分支及其后代。

树宽达到目标后，论文提出切换到标准随机续写。Algorithm 1 还规定，最终数量不足时 padding 到目标条数。

### 3.2 候选 token 的概率门控

设 `c*` 为最高概率 token，其他候选 `c` 要同时满足：

```text
p(c) > tau_abs
p(c*) - p(c) < tau_rel
```

论文参数是 `tau_abs=0.25`、`tau_rel=0.15`。

这不是按照 ROSE 语义熵挑选历史位置，也不是对任意随机生成的后缀做相同处理。候选必须有足够概率，并且与 top-1 足够接近。

### 3.3 Lookahead 检查什么

分支向前生成后，比较父子相应 token 段的归一化 Levenshtein 编辑距离：

```text
d = token 编辑距离 / 窗口长度
```

编辑距离统计 token ID 的插入、删除和替换，不比较 token embedding，也不核验数学结论。

论文主文以单窗口解释；附录 B.3 的实验使用 20、30、50 三个窗口，阈值 0.4，要求全部条件满足才保留。

论文按照分支年龄、最近 r token 描述检查。下面的融合实现明确使用“分支点之后的前 r 个新 token”，不声称与原在线算法在时间步/off-by-one 上逐项等价。

### 3.4 Hybrid rollout 与 eta 衰减

论文并非始终让所有最终样本都来自 LATR：

```text
k_latr = round(eta * k)
k_std  = k - k_latr
eta_i  = eta_0 * gamma**i
```

eta 随训练步下降，后期增加标准随机采样的占比。附录使用 eta_0=1，Countdown 的 gamma=0.985，Math 的 gamma=0.995。

它不同于 ROSE 的 epsilon：eta 是随训练变化的轨迹数分配比例；epsilon 是当前响应是否从根部重新采样的概率。二者不能直接视为同一参数。

### 3.5 LATR 改的主要是 rollout

LATR 可以与 GRPO 或 DAPO 等训练算法结合，本身不要求使用 ROSE 的树分段优势或 alpha 长度校准。

附录中训练采样为 temperature=1、top-p=1、top-k=-1；验证为 temperature=0.6、top-p=0.95、top-k=20。这里的 top-k=20 是验证采样截断，不是 ROSE 的候选概率返回数量。

“算法逐 token 扩树”也不等于“必须在 Python 中每个 token 发一次 vLLM RPC”。前者是算法状态描述；后者是一种可能很低效的外部实现方式，不能混为一谈。

## 4. 当前融合算法的完整流程

### 4.1 保留了哪些 ROSE 部分

- 第一条响应从根部完整生成。
- epsilon 根部重采样。
- 从所有已保留完整轨迹中按语义熵选择分支位置。
- 最终每个成功 group 保留 G 条完整响应。
- 真实父子关系、分支位置与 `finalize_tree`。
- `rose_tree` 分段优势和 alpha 长度校准能力。
- 原有 verifier、actor loss 和外层 DAPO 过滤逻辑。

“保留能力”不等于“脚本一定启用”：例如当前脚本默认 alpha=0，长度校准虽然存在但不产生作用。

### 4.2 一个最终轨迹槽位怎样生成

第一条响应直接完整生成。对其后的每个槽位：

```text
只抽一次 epsilon
  |
  +-- 根部采样 -> 完整生成 -> 保留
  |
  +-- 分支搜索
        -> 对全部已保留响应的合法位置按语义熵排序
        -> 取父轨迹 P 和分支位置 b
        -> 复制 P.response_ids[:b]
        -> 一次请求随机生成最多 50 个 probe token
        -> 与 P.response_ids[b:] 做多窗口比较
             |
             +-- 拒绝 -> 换试次/位置，仍然填当前槽位
             |
             +-- 接受 -> 复用 probe，必要时续写到结束 -> 保留
        -> 搜索上限/预算耗尽 -> 从根部完整采样 -> 保留
```

分支位置必须满足父轨迹在该位置之后至少剩余最大窗口长度，且 response 总预算也足够。非有限语义熵不能作为候选。

候选按分数从高到低排序，平分时按父轨迹编号和位置稳定排序。实际相同的 prefix 与 b 会去重，避免同一个共享前缀在多条轨迹中占据多个候选名额。

失败位置的试次数在当前槽位内受限；下一槽位会重新搜索，并非维护整个训练过程中永不再试的黑名单。

### 4.3 “一个 token 一个 token”实际怎么执行

当前不是外层反复调用 `max_tokens=1`。

普通接受路径的请求分为：

1. 一次 `probe` 请求，`max_new_tokens=50`。
2. probe 通过且尚未终止时，再发一次 `completion` 请求，生成剩余 token。

vLLM 在一次请求内部仍然自回归地产生 token；Python 控制层只在段与段之间介入。拒绝后可能再发新的 probe 请求。

这避开了逐 token 外部 RPC，但也意味着当前不会在第 20 个 token 刚产生时立即取消一个已注定不通过的 50-token 请求：返回整段后才统一检查。

### 4.4 多窗口的具体判定

令 Q 为本次 probe，新 token 从分支位置 b 开始：

```text
d20 = EditDistance(Q[:20], P[b:b+20]) / 20
d30 = EditDistance(Q[:30], P[b:b+30]) / 30
d50 = EditDistance(Q[:50], P[b:b+50]) / 50

接受条件：d20 >= 0.4 且 d30 >= 0.4 且 d50 >= 0.4
```

不计入 prompt、共享 prefix 或 padding。目前只与选中的父轨迹比较，不与全部兄弟轨迹逐一比较，因此不保证 G 条响应全局两两不同。

例如距离分别为 0.50、0.43、0.36，仍然拒绝，因为 50-token 窗口未通过。

### 4.5 接受后的 token 拼接

```text
P = 继承的父响应前缀
Q = 已接受的 probe
C = 接受之后的新续写，若已终止则为空

最终 response = P + Q + C
最终 logprobs = P.logprobs + Q.logprobs + C.logprobs
最终语义熵   = P.entropy  + Q.entropy  + C.entropy
```

三个数组必须逐 token 对齐，且最终 response 总长不超过原有 response_length。

Q 不会被丢掉重生成。续写请求输入为“原 prompt + P + Q”，但在训练样本中 Q 依然属于 response，不会因为作为后续请求输入而被挪到训练 prompt。

树中分支位置仍然是 b，不会因为 probe 长度为 50，就改为 b+50 或额外创造一个树节点。执行分段不等于 credit assignment 分段。

### 4.6 拒绝后做什么

拒绝的 probe：

- 不提交到 reward/verifier。
- 不作为 TQ 的最终训练样本。
- 不加入最终树，不参与节点 value 或优势计算。
- 不计算其 ROSE 语义熵，但仍产生了生成、top-k 返回等实际成本。
- 保留必要的成本统计；开启 trace 时可额外保存事件。

只有最终保留的完整响应可以作为后续轨迹的父节点。probe 尚未被接受时不会拥有已提交的子树，因此当前没有实现 LATR 那种“在线删除一个分支及其所有后代”的状态机。

### 4.7 本地工程规则：预算、短终止与异常

以下是当前融合设计，不是直接摘自论文：

| 情况 | 当前处理 |
|---|---|
| 每位置重试 | 每槽位同一 pivot 最多 2 次 |
| 不同位置数量 | 每槽位最多尝试 4 个不同 prefix 的 pivot |
| group probe 预算 | 默认 700 个实际返回的新 token，接受与拒绝的 probe 都计入 |
| 提交前预留 | 必须能预留完整最大窗口 50；不够则回退，不缩短检查窗口 |
| 无合法点/搜索耗尽 | 从根部重新完整采样，填满当前槽位 |
| probe 不足 50 就自然 stop | 完整响应非重复时保留，标记 short_terminal；不称为完整多窗口通过 |
| probe 恰好在 50 处自然 stop | 先完成窗口检查，通过后无需再续写 |
| abort、协议错误、版本不一致 | group 失败，不伪装成普通相似度拒绝 |
| 请求超时 | 默认 600 秒；失败并记录，不能保证后端已取消 |

short_terminal 的去重对象是所有已保留完整 response，与它们比较完整 token IDs。它是明确的例外通道，不能把其数量混进完整窗口通过率而不加区分。

700 是 probe token 预算，不是整个 group 的总生成预算，也不是“最多浪费 700 token”：根部响应、接受后的 completion、外层 DAPO 补采均另计。异常请求尚未返回时，其真实 token 成本可能未知。

根部回退是重新采样，不是复制已有响应填充。即便如此，独立采样也可能偶然重复，所以当前不承诺 G 条完全不同。

## 5. 三者的核心差异总表

| 维度 | ROSE 原算法 | LATR 原算法 | 当前融合实现 |
|---|---|---|---|
| 选点时机 | 已有完整响应生成后 | 部分序列在线生成时 | 已有完整响应生成后 |
| 分支依据 | 高语义熵 | 候选概率绝对/相对门槛 | 高语义熵 |
| 搜索期主路径 | 已采样的完整路径 | top-1 扩展 | 已采样的完整路径 |
| 子分支首 token | 重新采样 | 指定满足门控的候选 | 随机采样，不强制不同首 token |
| 短程剪枝 | 无该 lookahead 模块 | 有，剪子分支及后代 | 有，只筛尚未加入树的 probe |
| 差异指标 | embedding 参与选点 | token 编辑距离参与剪枝 | 两者分工保留 |
| 父路径 lookahead | 不适用 | 随在线生成获得 | 直接读取已有父响应后缀 |
| 请求组织 | 完整后缀生成 | 在线树算法，不限定 RPC 接口 | probe + 接受后 completion |
| 未凑足轨迹 | 持续 rollout 至 G | Algorithm 1 要求 padding | 有界搜索后根部独立重采样 |
| 标准随机采样比例 | epsilon 根部重启 | eta 混合且随训练衰减 | 保留 epsilon；另有搜索回退 |
| 树的用途 | rollout 与分段优势 | 主要控制 rollout 探索 | rollout 与 ROSE 分段优势 |
| 长度校准 | alpha | 不要求 ROSE alpha | 保留能力，脚本默认 alpha=0 |
| 训练目标 | Dr.GRPO 风格目标 + KL | 可接 GRPO/DAPO | 沿用本项目既有 actor loss |

## 6. 哪些是工程适配，哪些实际改变了算法

### 6.1 主要是实现正确性保障

- 将生成 helper 拆成可限制新 token 数的 segment 接口。
- 保留后端原始 finish_reason，区别正常 stop、段长度上限和 abort。
- 严格校验 token、logprob、top-k payload 的长度和形状。
- 保证同一棵树的参数版本一致。
- 对最终 metadata、指标去重和历史保存进行适配。
- 在不改第三方引擎的情况下，通过同步 client 使用现有 vLLM-Ascend。

这些改动主要解决“正确执行已定义算法”，不代表新提出了候选搜索原则。

### 6.2 属于实质性算法取舍

- 用 ROSE 的离线高语义熵选点替代 LATR 在线概率门控。
- 使用随机 probe，而非 LATR 搜索期的贪心主路径与受控候选首 token。
- 使用已完整生成的父路径作为参考，不在线并行扩展父子树。
- 不实现 eta 衰减，保留 ROSE epsilon 和预算回退。
- 不足时根部重采样，而非 LATR padding。
- 添加短自然终止例外、有限候选数和 group probe 预算。
- 只比较父子，不增加全兄弟去重。

这些不是数学上等价的加速技巧，会改变轨迹分布。报告实验时应称为融合/变体，不应写成“无损复现完整 LATR”。

剪枝本身也会产生选择偏差：最终样本更倾向于与父轨迹不同。当前没有新增关于整套搜索/接受概率的无偏重要性修正；沿用原训练 loss 不等于证明了严格 on-policy 等价。

### 6.3 原有 ROSE 评分也需单独区分论文与代码

当前 scorer 使用有限 top-k payload 估计 H，没有计算全词表精确熵，也没有将返回的 top-k 概率重新归一化到和为 1。

当前默认 `exclude_diagonal=True`，在 pairwise 项中去除自相似对角项；论文展示的公式 (2) 按其书面求和形式未显式排除对角。这属于既有 ROSE 适配的公式细节，不是本轮引入 lookahead 时修改的。若要求严格论文对齐，需要单独审计并做对照，不能靠本文将它宣称为完全一致。

CPU scorer 使用预导出的归一化 embedding mmap，并非随每次 actor 更新自动同步。分块大小影响 CPU 内存与吞吐，不改变“每位置考察若干候选”的含义。它与 NPU 中模型持续更新的输入 embedding 是两条不同的数据通路。

## 7. 性能和显存：能说什么，不能说什么

### 7.1 为什么可能比普通 GRPO 慢

普通 GRPO 的同 prompt 多条响应可以同时启动；当前 ROSE/融合在同一个 group 内有串行依赖：先得到已有完整响应，才能据此选择下一条的分支点。

融合版本另外增加：

- probe 请求和调度开销。
- 被拒绝 probe 的实际生成成本。
- 接受后 completion 的再次请求及可能的 prefix prefill。
- CPU 编辑距离、ROSE 语义熵计算和元数据处理。

不同 prompt group 仍然可以并发，所以实际速度还受 batch size、worker 并发、序列长度、TP、引擎调度和 NPU 利用率影响。

### 7.2 为什么仍可能值得做

动机是：在生成很长的重复后缀之前，先用短段发现相似分支，把探索预算转向其他位置。

这只是一种潜在收益。如果拒绝率很高、prefill 很贵、根部回退很多，wall-clock 可能反而增加。需要比较同等训练成本下的验证表现，而不只是比较单步是否更快。

LATR 论文的 backtracking-free 复杂度结论和其配置下的耗时增幅，不能直接套到当前有重试、回退和独立请求的融合实现上。

### 7.3 共享前缀是否自动节省 NPU 显存

没有保证。

当前共享首先是算法上的 token 前缀继承：复制已有 token IDs，不是强制 vLLM 把两条响应绑定到同一份 KV 状态。每段是独立请求，当前不依赖会话级 KV continuation，也不保证请求路由到同一引擎或命中缓存。

即使开启 prefix caching，收益也受后端支持、路由和缓存生命周期影响。训练阶段保存的最终响应张量仍含完整前缀，不能直接按“只存新后缀”估算 actor 显存。

### 7.4 CPU 计算与 NPU 生成的分工

- NPU：模型前向、自回归生成，以及原框架的训练计算。
- CPU：控制流、候选 embedding 查表/语义熵、短窗口编辑距离等。

服务器同样有 CPU。把控制与小规模比较放在 CPU，不代表模型在 CPU 上生成；是否成为瓶颈要看实测耗时。当前未加入专用 NPU scorer/kernel，也未证明 CPU scorer 在目标配置中不构成瓶颈。

## 8. 当前脚本不等于论文配置

新的独立脚本为 [run_rose_latr.sh](run_rose_latr.sh)，基于 `run_rose_new.sh`，后者未因融合而修改。

当前新增 lookahead 默认值：

| 脚本变量 | 默认值 | 含义 |
|---|---|---|
| ROSE_LOOKAHEAD_ENABLE | true | 新脚本启用；通用 YAML 默认关闭 |
| ROSE_LOOKAHEAD_WINDOWS | [20,30,50] | 多窗口长度 |
| ROSE_LOOKAHEAD_THRESHOLD | 0.4 | 各窗口最低归一化距离 |
| ROSE_LOOKAHEAD_MAX_TRIALS | 2 | 单槽位单 pivot 试次数 |
| ROSE_LOOKAHEAD_MAX_PIVOTS | 4 | 单槽位候选 prefix 数 |
| ROSE_LOOKAHEAD_GROUP_BUDGET | 700 | group probe token 预算 |
| ROSE_LOOKAHEAD_REQUEST_TIMEOUT | 600 | 单请求超时秒数 |

保留的 baseline 包括 epsilon=0.3、alpha=0、ROSE_TOP_K=20、semantic chunk size=256、KL loss 系数 0.01；当前验证是 `n=1`、不采样的贪心验证，不是论文 pass@8。

脚本没有为了融合显式加入 `algorithm.filter_groups.enable=True`，也没有为了论文对齐而改 loss 聚合/尺度。实际是否启用过滤，应查看启动后的完整解析配置，不根据“使用了 ROSE”或脚本文件名猜测。

因此，公平验证融合效果的首个对照应是：同一个新脚本只切换 lookahead 开关，其他超参保持一致。严格复现论文配置是另一项工作，不应同时修改一批参数后将差异全归因于 lookahead。

## 9. 代码对应关系

| 文件 | 作用 |
|---|---|
| `verl/experimental/rose/lookahead.py` | token 编辑距离、多窗口判断、合法 pivot 排序/去重、probe 预算和 seed 派生 |
| `verl/experimental/rose/lookahead_rollout.py` | 每个 group 的槽位搜索、重试、接受/拒绝、续写、回退和统计 |
| `verl/experimental/rose/rose_agent_loop.py` | prompt 构造、segment 生成接口、停止原因及数据对齐检查 |
| `verl/experimental/rose/rose_agent_loop_tq.py` | lookahead 开关接入、最终叶子提交和验证 bypass |
| `verl/experimental/rose/semantic_entropy.py` | 既有 CPU 语义熵计算，本轮未改数学公式 |
| `verl/experimental/rose/tree.py` | 根据最终保留轨迹构建稳定树 metadata |
| `verl/experimental/rose/tree_advantage.py` | 节点 reward 均值、分段优势、alpha 校准 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 保留原始 stop/length/abort 元数据 |
| `verl/workers/config/rollout.py` | typed lookahead 配置及校验 |
| `verl/trainer/config/rollout/rollout.yaml` | 通用默认关闭的 lookahead 配置 |
| `verl/trainer/ppo/v1/trainer_base.py` | group 统计去重和最终 rollout 历史保存 |
| `run_rose_latr.sh` | Ascend 平台独立启动脚本 |

验证路径明确绕过训练 lookahead，继续走独立生成。开启 `val_before_train` 不会令验证集自动执行分支剪枝。

## 10. 如何判断融合是否有效

至少同时看：

1. **正确性**：token/logprob 对齐、版本一致、最终 G 条、拒绝样本不进入 reward/tree。
2. **搜索行为**：probe 尝试数、完整窗口通过率、短终止保留数、各类 root fallback 比例。
3. **成本**：总生成 token、拒绝 token、group wall time、prefill/cache 情况、CPU scorer 耗时。
4. **训练效果**：相同验证采样下的正确率/pass@k、长度、策略熵、KL，以及多 seed 稳定性。

当前观测入口包括 `[ROSE_LOOKAHEAD_PROGRESS]`、`[ROSE_LOOKAHEAD_GROUP]`、可选 worker trace，以及最终 rollout history。

注意 trainer 中按 tree_id 去重的聚合指标只覆盖最终进入 batch 的 group。DAPO 丢弃或失败的 group 不能凭这些指标自动计入整步总成本；要结合 group 终结日志和 trace 判断完整 workload。

最终应回答的是“同等计算预算下是否获得更好的策略”，而不只是“保留的轨迹更不相似”。目前没有 NPU 实验数据，不能提前认定更快、更省显存或正确率更高。

## 11. 后续更接近 LATR 的方向

当前首版稳定并完成对照后，可以分别研究：

- 在 ROSE pivot 上加入候选概率门控。
- 显式选择不同首 token，并正确处理其 logprob。
- 引入类似 eta 的训练阶段混合比例衰减。
- 加入兄弟轨迹比较或更强的语义去重。
- 将控制下沉为真正在线树，支持早期剪枝和后代清理。
- 优化请求 affinity、prefix cache 或真实 KV continuation。

这些方向不是当前已经实现的功能。尤其“概率门控 + 强制首 token”需要先核对 raw/processed logprobs、temperature/top-p 与训练 old logprob 的契约，不能只在现有随机 probe 前加一个阈值就宣称复现了 LATR。

---

一句话总结：**当前算法保留 ROSE 的离线选点和树优势，在每次分支时增加 LATR 风格的短程差异筛选；为了适配现有 VeRL/vLLM-Ascend 请求接口，采用随机分段生成、有界重试和根部回退，因此它是明确的融合变体，而不是完整 LATR。**
