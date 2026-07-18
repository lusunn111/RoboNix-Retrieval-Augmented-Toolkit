就是写个idea文档，先把两篇文章（RT-cache和DB）讲清楚，然后简单构建一下我们的idea这样子

## RT-cache

![img](https://wcn2gtezvkxq.feishu.cn/space/api/box/stream/download/asynccode/?code=YjAzODhmYWJhMGU2N2U2Yjg5NTY1ZmIyZjBlMjdhNzRfV3NWOUJObTNGM0JQU2FKMUloWFM4TWFrVmFLQThJSDlfVG9rZW46SmdRMmJ6RTRab3FCeUZ4YXp6QmN2S2tObkRjXzE3Njc1MDA2ODg6MTc2NzUwNDI4OF9WNA)

![img](https://wcn2gtezvkxq.feishu.cn/space/api/box/stream/download/asynccode/?code=OGQ0MzBmZTQ4OTc3ODczYTg5NjgwYTk1YzZiZmRlNWFfc3lVRjRCQ1BYNnd0bm5icEVndnVRalNKRW02dnZieXdfVG9rZW46V0VBNWJ0S2Vib0swVm54QWV4YWNqZFRkbmUwXzE3Njc1MDA2ODg6MTc2NzUwNDI4OF9WNA)

### 基本思想

完全使用数据库代替模型，去执行动作。

RT-cache 的核心想法是：别让模型每一步都算动作了，**直接用数据库里存的“成功轨迹经验”来控制机器人**。运行时把当前相机画面编码成一个向量，去库里找“最像的历史画面”，然后把那条历史轨迹从这个时刻开始的后续连续 N 步动作片段（trajectory snippet）拿出来直接执行，这 N 步期间不再做额外推理调用。

### 数据怎么表示

论文把每个状态表示成**纯图像 embedding**（它也提到“可选文本”，但整体讨论强调 image-only 的局限）。这也是为什么它的实验任务被设计得比较“轻任务”：主要是让机械臂末端移动到目标物体附近，而不是完整 pick-and-place。 

具体编码方式是把两种视觉特征拼起来：DINOv2 提供 1024 维，SigLIP 提供 1152 维，拼成 2176 维，并对拼接向量做归一化。论文说这个拼接在经验上比单独用其中一个编码器更好，但**没给具体消融数字**说明提升幅度。 

### 数据库怎么建

他们用了两层数据库：**Qdrant 存向量**，负责相似度检索；**MongoDB 存轨迹原始数据**，负责把检索到的向量位置映射回“哪条 episode 的哪一步”，并取出对应的 timestamp、7D 动作、RGB 等信息。MongoDB 用 (episode id, step id) 作为索引来回查。论文没有给更细的 schema 字段表，但整体分工就是“向量库检索 + 轨迹库取动作片段”。

为了把 Open-X 各子数据集混在一起用，论文先把动作做了统一：控制频率统一到 10Hz；动作统一成 7D 的末端（EEF）笛卡尔表示；速度型动作按 0.1s 积分成位移；不符合形状的 episode 会被过滤掉。

### 怎么检索

论文承认全库直接检索会超时（>300s），所以它做了一个“先粗后细”的分层筛库：先在 **dataset 级别**做粗定位，再在候选子集里做精检索。 

具体做法是：对 Open-X 的每个子数据集，离线把该子集内所有 embedding 求平均得到一个 **centroid**；在线时把当前向量 e_t 和所有 centroid 做相似度比较，先选出最像的 top-m 个子数据集。 接着在这 m 个子数据集里，各自再“随机抽样（或聚类抽样）”出一个规模为 S（几千级）的**小子集**建 local index，在每个 local index 上跑更细的 kNN，把复杂度压到 O($$m\times S$$)。论文写了这个流程，但**没有把“抽样到底怎么抽”展开成可复现的实现细节**。 最后再对候选计算 cosine similarity 排序，取 top-K（例如 50）。

### 动作怎么执行（两种模式）

检索到近邻后，每个近邻都能在 MongoDB 里定位到轨迹上的一个时刻，并取出从该时刻开始的**未来 N 步动作序列**。 论文给了两种执行方式：一种是只用 top-1 近邻的 snippet 直接执行；另一种是把 top-K 的 snippet 按时间步做平均，等价于

$$a_{t:t+N}=\frac{1}{K}\sum_{j=1}^K a^{(j)}_{t:t+N}$$,

用“多邻居平均”来减噪。

snippet，一些连续的action box。

但是论文讨论了执行多少步比较好，给出的是3个action box.

### 实验任务与结果

他们的实验是一个简化桌面任务：Franka Panda 在三个相机视角（front/side/wrist）下，把末端移动到桌面目标物（cup/bowl/bottle）附近；成功标准是最后一步 EEF 进入目标物的 graspable region。

结果上，论文清楚地展示了一个现象：**zero-shot（库里没对应 domain 的数据）基本会失败**；而加少量 few-shot demo 后成功率会明显上升。 同时他们讨论了 snippet 长度 N 的权衡：N=1 适应性好但查库更频繁；N=3 往往最平衡；N=5 在不匹配时更容易出现 loop/overshoot。

### 创新点与局限

论文的主要贡献是：把“检索 + 轨迹片段回放”做成一条可用的机器人控制管线，并用 multi-stage retrieval 让大库检索在实践中能跑起来，同时在真实设置下报告了更快的平均操作时间与不差的成功率。 但它也承认 image-only retrieval 有上限，未来需要加入语言/高层提示等信号；另外环境一变化，回放 N 步可能会带来碰撞或 overshoot 风险，需要更频繁重检索或额外传感反馈。 

## Hierarchical Drafting

![img](https://wcn2gtezvkxq.feishu.cn/space/api/box/stream/download/asynccode/?code=YmIzOGM3Njc0MzAwMTE5Zjc3NjQzOTg5OGI1MDdhYjlfaDBwUEdMQzFhYmVEbk1QTGltTE5zbGZHVTVpWFFEV21fVG9rZW46V1NlSWJPMWxFb0dCZ2t4T25zVmNBY25lbmhiXzE3Njc1MDA2ODg6MTc2NzUwNDI4OF9WNA)

![img](https://wcn2gtezvkxq.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDMyNjFhZTkxZmE1MTc3MzIwZWM4ZDQyNzgxYjJlMTVfdGZ0a0JLYTdVNWpRaEp1WGN6WGFoazVEUjlZcTZWS3ZfVG9rZW46SFQ3ZmJBMFBnb29WZE14S25qYmNSNjllbmFkXzE3Njc1MDA2ODg6MTc2NzUwNDI4OF9WNA)

### 基本思想

这篇（HD / Hierarchy Drafting）的核心不是“再训练一个更小的 draft 模型”，而是走**纯算法 + 纯数据库**路线：用数据库先“草拟”一段候选 token，然后再让 target LLM **verify（验收）**，只接受与 target LLM 完全一致的前缀，所以整体是 **lossless** 的。它真正想解决的是：单一来源的 database drafting 在不同任务上会忽快忽慢，而把数据库做大又会被检索延迟拖死，于是作者借鉴“缓存/存储层级”的思路，用**时间局部性（temporal locality）**把 token 来源分层管理。 

HD 的主张很直白：**把 draft token 的来源分成三层库**，并按“局部性从高到低”依次查，先用最可能命中、最便宜的库；不够再用更大、更慢但覆盖更广的库补齐候选。

### 数据怎么表示

它的数据库本质是“前缀 → 后续 token 序列”的映射：用一段 prefix tokens 当 key，取紧跟其后的 token 片段当 value（draft token sequence）。

 论文里用超参记号表示：给定 previous token 长度 l，draft 序列长度 m，以及一次拿多少条候选组成 draft set（大小 N）。

### 数据库怎么建（三层库分别是什么）

HD 把 token 候选按局部性分成三类数据库：

- **Dc（context-dependent）**：强依赖当前这次生成过程的“局部重复”。来源包括：输入 prompt token、并行解码生成的 token、生成过程中丢弃/浪费的 token 等；Dc 在每个 forward step 都会更新，每次新 generation 会重新初始化，并用 **LRU** 维护。
- **Dm（model-dependent）**：模型跨很多生成过程都爱重复的“习惯片段”（不太依赖具体上下文）。构建方式是从模型生成文本里采样 top-k 高频 token 序列，形成 key/value。
- **Ds（statistics-dependent）**：来自大语料的通用短语（覆盖最大但局部性更弱）。为了在大语料上高效检索，他们用 **suffix array**（并沿用 REST 的实现思路）。

实现层面，论文在附录写得很明确：**Dc、Dm 就是 Python dictionary（lookup table）**；**Ds 用 REST 的 DraftRetriever。

### 怎么检索

它每一步都会按层级去“凑够”一个 draft set：先查 Dc，如果候选数 |\tilde{X}| 不够 N，就补查 Dm；还不够再补查 Ds，直到 $$|\tilde{X}|=N$$。这就是 Algorithm 1 的核心。

这里有两个很关键的工程细节：

1. **Dc/Dm 实际用 1-token key**：虽然超参里“给定的 previous token length”是 l，但在 Dc 和 Dm 里真正作为 dict key 的 prefix token length 设为 1，因为 prefix 变长会导致大量 miss（key 找不到）。
2. **Ds 的 miss 处理**：如果 Ds 用当前 previous token length 查不到，就会不断减小 previous token length 重试，直到找到或降到 0（跟随 REST 的实现）。

另外，论文也规定：对 Dc/Dm，同一个 key 下最多存的 value 数量不超过 draft set 大小 N。

### “动作/输出”怎么执行

HD 不是检索出 token 就直接输出，它会把 draft set $$\tilde{X}$$ 交给目标模型 M_p 做验证：目标模型一边验证候选序列，一边生成额外 tokens $$\hat{x} $$用来更新 Dc，然后推进当前位置 $$ n \leftarrow n+i$$ ，循环直到 EOS 或最大长度 T。这整个流程在 Algorithm 1 里写得很直。

论文还说明：HD 的 verification step 主要基于 LADE 的实现（n-gram verification），因为他们沿用了 LADE 的并行解码与验证分支设计。 

### 实验设置与结果（论文到底证明了什么）

实验在 Spec-Bench 的 6 类任务上做（多轮对话、翻译、摘要、QA、数学推理、RAG；每类 80 条，总 480 generations），模型覆盖 Vicuna 与 Llama-2-chat 的多个规模。

超参设置是： $$l=2, m=4, N=7, T=1024$$ 。Dc 包含 previous input tokens 和并行解码 token；Dm 来自 OASST 英文集上用同家族 7B 模型生成的 39,283 条文本，再取 100k 最常见 token 序列；Ds 沿用 REST，用 UltraChat 构建约 12GB 的库。

结果部分（主结论）是：HD 在不同模型/温度下整体速度最好；比如温度 0.0 时，HD 超过 1.5×，而其它方法没超过 1.4×；温度 1.0 时虽然整体加速会降一点，但 HD 仍是最快。

作者也专门解释了“为什么分层有用”：单库会导致 acceptance ratio 低、draft failure 高；多库组合能提升鲁棒性，但还要平衡 draft latency，否则像只用 Ds 这种大库会被延迟拖到**甚至比 AR 还慢**（他们在附录 Table 3 给了 Ds-only speedup 0.81× 的例子）。 

### 创新点与局限

- **把“draft token 的来源”当成缓存层级来设计**：Dc（最局部、最快）→ Dm → Ds（最泛、最慢），并且用层级访问把“覆盖”和“延迟”同时管住。 
- **把数据库 drafting 做得更稳**：作者用实验展示，单一来源会导致跨任务/跨模型表现起伏，而 HD 通过多来源分层组合，让加速在各任务上更一致。

局限/讨论方面，论文明确强调：再训练类方法（比如一些多头/分支）可能更快，但训练成本在真实服务场景（多模型、多 adapter、资源受限）不现实，所以他们定位 HD 是“免训练、lossless、工程友好”的折中路线。 

# Hierarchical SD for VLA Models

# 问题：

1. 现有的SD生成范式增加了VLA模型的推理成本
2. 单纯的SD难以利用VLA推理中的高度冗余性（冗余性指历史中相似的环境、动作等）
3. 对于推理过程中，已经成功的Case没有办法有效利用，即已经成功的记忆是存在的，但是无法被使用。
4. 之前的工作缺乏对任务文本的使用，仅仅使用图像作为索引，缺少任务语义。之前的工作也没有对上一步的动作进行编码索引，缺乏动作语义。

# 现象&分析：

1. SD中，添加了Drafter之后，会造成速度减慢（这个我们之前的论文测过），以及计算量的增加（测试一下Drafter增加了多少计算量），来佐证问题1。
2. VLA模型推理过程中存在高度的冗余性，可按如下分类：（1）相邻动作之间的高度相似（2）相似任务下的整体轨迹相似。（都需要测试和证明相似性）而SD无法利用这些冗余性，导致效率不高，作证问题2。
3. 记忆即曾经执行成功过的任务，只是单独存成了视频和动作记录，但之前没有数据库范式，并没有再次利用这些数据，用来作证问题3。
4. 观察RT-Cacho的论文，任务隐含于图像这个假设在一些数据集中是不成立的，在LIBRO中，相同场景任务不同，在同一场景下执行的动作也不同。作证问题4.

# 方法设计：

1. 构建 Hierarchical Retrieval-Augmented SD 框架(架构)
   1. 引入基于分层DB的记忆机制，在某些场景替代/辅助Drafter，直接利用P2提到的数据冗余性。
   2. 整体的Pipline 和 System设计，实现从纯模型推理到"检索-生成协同"的范式转变。
2. 设计多模态协同的状态表示
   1. 解决P4的歧义问题，构建视觉-文本-动作的联合特征表示。
   2. 引入视觉-文本协同：区分不同场景下的具体意图。
   3. 引入时序连贯性表示：对上一步动作进行编码，确保动作平滑准确。
   4. 允许动作连贯性上一定程度的不一致，避免动作连贯性的过度约束。
3. 构建检索-生成自适应策略
   1. 确定Drafter与DB的动态边界，设计门控机制，通过场景反馈与数据库反馈，自动判断是使用哪种策略。
4. 统计DB-Drafter协同范式
   1. 纯DB，直接复现多步
   2. 纯DM，完全按照SD范式来
   3. DB结果作为Verify
   4. DB结果作为SD范式Verify的补充
5. 构建Self-Evolving Memory闭环与对应的检索策略
   1. 对于成功的执行，应当并入数据库，同时针对检索策略构建可以实时更新的存入策略
   2. 构建合理的检索策略，在速度与准确率上做平衡。