
## PLA 方法差异与对 learned index 的作用路径

### 统一问题表述：ε-PLA 在 learned index 中的角色

在 ε-PLA learned index 中，排序键集合的“键→秩（位置）”映射可视作缩放后的 CDF；ε-PLA 用若干线段去近似该映射，并保证任意键的预测位置与真实位置的最大偏差不超过 ε。查询通常分两步：先定位覆盖查询键的线段并计算预测位置，再在 `[ŷ-ε, ŷ+ε]` 的“last-mile”区间里做精确搜索。  

PGM-index 与 FITing-Tree 的差异，核心就在“如何组织并索引这些线段”：FITing-Tree 用传统 B+-tree 去索引线段；PGM-index 则递归地对线段端点再做 ε-PLA，直到形成根，从而得到多层 learned routing。

### 三类 PLA 拟合算法的“可预期差异”

在同一误差界 ε 下，PLA 拟合算法主要在三方面不同：

第一，**线段条数（模型规模）**：OptimalPLA 旨在最小化线段数；Greedy 系通常更快但可能产生更多线段。 

第二，**构建时复杂度与内存**：经典最优算法通常通过在线维护凸包/可行域来保证最优性，单趟 O(n) 时间但需要 O(n) 辅助空间；典型 greedy 算法可将额外空间压到 O(1) 量级（以牺牲线段最优性为代价）。  

第三，**线段端点与斜率分布**：SwingFilter 使用“以首点为 pivot、逐步收紧上下斜率界”的策略；GreedyPLA 类似但使用不同 pivot 初始化策略（使用两条极值线的交点/中点），因此在相同 ε 下可能得到不同端点与斜率分布。 

第四，**平均误差：** SwingFilter 和GreedyPLA虽然产生了更多的线段数量，但是其平均误差（预测值与真实值）明显小于OptimalPLA;  这会影响 lookup latency的大小

## 统一对比基准的核心设计

### 必须统一的接口与指标

第一层是 **PLA-only（仅分段算法）**，输出“线段集合”；第二层是 **index-level（端到端索引）**，把该线段集合嵌入 PGM-index、FITing-Tree、LOFT 与 PGM-disk。

统一报告的指标集合（按场景分主次）：

- 结构规模：线段条数（总数、各层/各节点分布）、索引高度/层数、索引内存占用（bytes/key 或峰值 RSS）、磁盘场景还需磁盘空间占用。
- 构建成本：单线程构建时间、多线程构建时间与扩展性（尤其要记录“并行切块导致的额外线段数”）。
- 运行时性能：lookup 吞吐/延迟（p50/p95/p99），多线程吞吐扩展曲线；动态负载还要插入吞吐/尾延迟抖动；磁盘还要 IOPS/每次查询读页数的分布与 direct I/O/OS cache 的配置。
- 微架构/系统计数器：内存场景建议采集 cache misses、branch misses、instructions；这是解释 learned index 差异的关键证据链。

### 数据集与负载的基线选择

内存场景建议以 SOSD/“Benchmarking Learned Indexes”常用的数据形态为基线：排序数组上的 equality lookup（10M 次）、数据分布覆盖近似均匀、正态、对数正态、稀疏均匀、真实数据等；这些基准同时强调 warm cache vs cold cache、线程数变化对吞吐的影响。 

动态负载建议遵循 LOFT 的实验范式：YCSB（Zipfian）与真实数据集，区分 read-intensive 与 write-intensive，并显式设置后台线程/前台线程配比；LOFT 的公开 slide 里给出了“每 12 个 worker 配一个后台线程”等设置线索。  

磁盘场景建议直接复用 SIGMOD’24 的设定：使用 SOSD 的 Facebook/Amazon/OSM 数据集（论文中以 200M、8 字节 key + 8 字节 payload 指针为例），采用 direct I/O 绕过 OS page cache，YCSB-C（100% read）、100% insert、50/50 read-insert 三类，初始化 150M key、执行 10M 操作，并以吞吐、内存峰值、磁盘空间为主要 metrics。

## 内存场景实验清单：PGM-index 与 FITing-Tree

### 实验 IM-A：PLA-only 的“iso-ε”基准曲线（先做这个）

目标是建立“给定 ε 时，三种 PLA 输出规模与成本的差异”，这是解释后续一切端到端结果的必要先验。

做法：对每个数据集，扫一组 ε（如 2³…2¹³），对三种 PLA 分别测：线段数、构建时间、峰值内存、线段长度分布（均值/分位数）、斜率/截距分布（用于后续 cache 行为解释）

关键注意点：

并行构建时，如果你沿用 PGM-index 常见的“把 key 切成连续块并行分段，再拼接结果”的策略，它会破坏 OptimalPLA 的全局最优性；PLABench 明确讨论了这一点，并指出额外线段数与线程数存在上界关系，这在你做“并行构建”实验前必须先验证并记录。 

baselines: ALEX LIPP RMI 

long tail latency 

### 实验 IM-B：端到端 iso-ε（同 ε、同 last-mile 搜索策略）

目标是回答最直接的问题：同一个 ε 下，把 PLA 方法替换掉，PGM-index 与 FITing-Tree 的性能怎么变？差异来自哪里？

设置：

对每个数据集 D、每个 ε，构建 6 个索引变体：

PGM-index × {OptimalPLA, GreedyPLA, SwingFilter}；FITing-Tree × {OptimalPLA, GreedyPLA, SwingFilter}。其中 FITing-Tree 原生就是“线段 + B+-tree 索引线段”的设计；PGM-index 原生是递归 PLA 层。  

统一约束：

last-mile 必须固定为同一种实现（例如始终用二分 `lower_bound`），否则你会把“最后一步搜索算法选择”的差异误归因给 PLA。SOSD/Benchmarking Learned Indexes 强调 last-mile 往往是 cache miss 的主要来源之一，因此这个控制变量非常关键。  

测量：

1) 构建时间（单线程与多线程各一组）；2) 索引大小（bytes/key、段数、PGM 层数或 FIT 树高）；3) 单线程 lookup 延迟与吞吐；4) 性能计数器（cache/branch/instructions）。Benchmarking Learned Indexes 的回归分析指出 cache misses/branch misses/instructions 对 lookup latency 有显著解释力，建议你也用同类分解来解释“为什么更少 segment 未必更快/为什么更少层数更快”等现象。  

## 动态负载实验清单：LOFT

LOFT 的公开材料给出了足够明确的实验接口：代码仓库提供 `microbench`；slide 给出了 workload 类型（YCSB Zipfian）、硬件配置与其关注点（扩展性、适应性、插入比例变化下的稳定性）。  

你要测 PLA 的影响，建议把实验组织成“先验证 model 层行为，再做端到端吞吐/尾延迟”的两段式。

### 实验 DW-A：PLA-only 在 LOFT 训练/再训练路径中的代价

目标：确认在 LOFT 的实际训练输入规模与调用方式下，三种 PLA 的训练时间与输出规模差异。

挂载点：LOFT 仓库中 `piecewise_linear_model.hpp` 明确实现了 `OptimalPiecewiseLinearModel` 与分段构造逻辑；把 GreedyPLA 与 SwingFilter 以相同 API 接进去，确保训练输出的是同语义的线段（并与 LOFT 的 model 接口兼容）。

测量：**单次训练时间**、**输出线段数**、以及**训练期间的内存峰值**；再训练路径同样测一遍。slide 强调了“non-blocking retraining process”与“self-tuning retraining”，把**“再训练耗时/频率”**作为 PLA 影响的重要观测量。 

### 实验 DW-B：端到端 iso-ε（read/insert 比例扫参）

目标：回答“同一个 ε 下，换 PLA 会不会改变 LOFT 的吞吐/尾延迟抖动”。

设置：用 `microbench` 跑至少三种负载点： read-only / write-heavy / balanced。LOFT README 指出 microbench 参数包含 `read/insert`、`data_num` 等，并给出了示例命令行；slide 也强调动态负载“包含 insert、数据规模增长、分布变化”。 

测量：吞吐（ops/s）、p50/p95/p99 延迟、tail jitter（例如 p99 随时间的方差）、CAS 插入失败重试次数、以及“retraining stage 对前台延迟的影响”（例如在 retraining 发生窗口内单独统计）

## 磁盘场景实验清单：PGM-disk

### 实验 OD-A：磁盘 PLA-only（对齐“页级语义”的 iso-ε 与 iso-Rp）

目标：把误差界从“items”转换为“pages”来观察 PLA 的真实 I/O 影响。

依据：磁盘论文给出了 I/O 页数与 ε、每页 item 数 P 的关系式，并把 last-mile 范围映射为需要读取的页区间；该映射决定了磁盘场景里 ε 的工程意义。

做法：

对每个数据集与 ε，先用 PLA-only 生成线段并得到预测误差界，然后计算理论 Rp（或直接在实现里统计实际读取页数分布）。同时做 **iso-Rp**：固定期望读取页数（例如 1 页、2 页、4 页），反推每种 PLA 需要的 ε（或页对齐后的 ε′），观察它们为了达到同等 I/O 需要付出的“模型规模”。

### 实验 OD-B：端到端 iso-ε（固定 G1/G2/G3/G4 设置，仅换 PLA）

目标：回答“在磁盘优化策略固定时，PLA 选择会不会改变吞吐/内存/空间”。

固定外壳：

- 页获取策略（G1）：默认 all-at-once；但要在一个代表性点上再测 one-by-one 以检查交互。论文指出两者优劣与磁盘类型、线程数、dist、Rp 相关，并给出“多数情况下 all-at-once 更好，但高端 SSD + 更多线程 + 小 dist 时 one-by-one 更优”的经验。 

- 预测粒度（G2）：先固定为 item-level（，再做一次 page-level 对照，看 PLA 方法与粒度的交互是否显著。  

- 页对齐扩展误差界（G3）：固定开启或固定关闭

- 参数压缩（G4）：固定关闭再固定开启各做一轮

指标：吞吐、峰值内存、磁盘空间；并额外记录“每次查询读取页数分布”与“CPU 时间占比”（若能采集）。  

### 实验 OD-C：G1 页获取策略 × PLA 的交互矩阵

目标：验证“PLA 改变预测误差分布 → 改变 dist/Rp → 改变 one-by-one vs all-at-once 的最佳选择”这一交互链。

做法：在每个数据集上选 2–3 个代表性 ε（对应 Rp≈1、Rp≈2–4、Rp≈8–16），对三种 PLA 各跑 one-by-one 与 all-at-once 的吞吐/延迟；并在不同线程数下重复（例如 1/16/64）。论文图示显示策略偏好会随线程数与磁盘类型明显变化，因此这个矩阵实验能帮助你得出“PLA 方法的收益是否被策略选择所吞噬”的结论。

### 实验 OD-D：G2 预测粒度 × PLA 的交互

目标：检查“page-level 预测”是否会让不同 PLA 的差异缩小（因为输出尺度变粗），或扩大（因为不同 PLA 可能更难在 page-level 上保持低误差）。

依据：磁盘论文讨论了 page-level granularity 与 item-level granularity 在 I/O 页数上的差异，并指出经验上 page-level 不一定总更好。  

做法：对每个 PLA，各做 item-level 与 page-level 两种训练与查询，并在同等 Ri 或同等 Rp 下对齐，再比较模型规模与吞吐。建议同时输出：模型节点数、每层节点数、以及页边界浪费比例（启用/关闭 G3 时）。

### 实验 OD-E：G3 页对齐扩展误差界对不同 PLA 的影响强度

目标：回答“G3 对齐策略的收益是否依赖 PLA 类型”。

依据：G3 的直觉是：训练时扩大误差界以与页边界对齐，减少“读到页里但落在误差界之外”的浪费，从而可用更少模型达到相同 I/O 页数。不同 PLA 在“线段端点/误差分布”上不同，因此 G3 的收益可能不均匀。  

做法：对每个 PLA、每个数据集，固定目标 Rp，然后分别在“不开启 G3（原始 ε）”与“开启 G3（对齐 ε）”下调参到同 Rp，比较：模型数、内存、吞吐。这个实验能回答“是换 PLA 更重要，还是先做 SGACRU 对齐更重要”。  

探究一个自变量的变化情况

### 实验 OD-F：更新工作负载下的 PLA 影响（在 hybrid 框架内）

目标：在磁盘场景中评估 PLA 对 update-heavy 的影响时，尽量避免“要求 learned index 本体可更新”带来的额外设计变量。

依据：SIGMOD’24 提出用 hybrid indexes 来处理更新（G6），并指出这可以让底层 learned index 不必自身支持更新，从而绕开“为磁盘专门设计可更新 learned index”的难题，同时在写入与均衡负载下能显著优于 B+tree。  

做法：选用仓库中已有的 hybrid-PGM-disk 路线作为外壳，然后对比三种 PLA 在相同内存 footprint 下的吞吐与读取页数。

##