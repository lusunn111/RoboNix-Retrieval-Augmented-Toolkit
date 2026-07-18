# Retrieval × Speculative Decoding：记忆–推理边界（Memory–Reasoning Boundary）

> 目标：把“检索（DB/RT-cache）”与“Speculative Decoding（SpecVLA）”统一成一个可解释的研究问题：**推理时哪些部分应该用记忆（memory）直接复用，哪些部分必须依赖模型推理（reasoning）**。

---

## 1. 一句话核心叙事（Paper-ready）

在 VLA 推理中存在大量可复用的时序模式与跨任务/跨 episode 冗余。我们把 DB 检索到的动作片段视为 **memory**，把 target VLA 模型的验证与回退视为 **reasoning**。通过把 DB 结果接入 speculative decoding 的 draft 通道，并统计 **accept length**，可以把“什么时候能相信记忆、什么时候必须推理”刻画成一个可量化的 **边界问题**。

---

## 2. 你现在的系统里已经具备的组件

- **Memory（DB）**：Qdrant + retrieval server，输入（image, instruction），输出 `rtcache_trajectory / averaged_trajectory`（动作序列）。
- **Reasoning（Target VLA）**：OpenVLA / SpecVLA 的 target 模型（保证准确率）。
- **Draft 来源**：
  - 传统 SD：小 draft 模型（SpecVLA 已有）
  - 本 idea：DB 检索动作作为“外部 draft”（无需额外训练）
- **可解释信号**：
  - `accept length`：target 逐 token/逐维度接受了多少 draft
  -（建议加）retrieval 相似度分数、top-k 分布熵、失败回退次数、DB latency 等

---

## 3. 检索 × SD 的三种“结合方式”（从容易到更强）

### A) DB as Draft（最贴合 speculative decoding）

- DB 返回候选动作（或动作片段），作为 draft tokens；
- target 模型逐 token 验证，得到 `accept length`；
- 不接受的部分回退到 target 自己生成。

你要的“机制可解释性”主要来自这里：accept length 本质就是“记忆被信任的长度”。

### B) DB as Branch（top-K 记忆分支）

- DB 返回 top-K 候选动作片段；
- 把 K 个候选当作多分支 draft，target 用 tree-attention / 并行验证选出最可接受分支；
- 研究点：**记忆的不确定性**（K 之间差异）如何影响边界。

### C) Hierarchical Drafting（DC→DB→Draft→Target）

把 LLM 的 Hierarchical SD 思想迁移到 VLA：

- **DC（短程 cache）**：当前 episode 内最近成功的片段/最近几步（最便宜）
- **DB（长程 memory）**：跨 episode 检索（覆盖更广但更慢）
- **Draft model**：覆盖最广（但计算最贵）
- **Target**：最终验证

研究点：边界不只是“用不用记忆”，而是“**不同层级记忆/草稿源的动态边界**”。

---

## 4. 你在 ideas.md 里写的 4 个问题 → 可做成 4 个可验证现象

### 现象 P1：SD 的额外成本（Draft 引入的推理开销）

要证明：SD 在 VLA 上并不总是加速，draft 的额外 forward/通信会拖慢。

**实验 E1（成本分解）**：
- 对比：AR / Spec / Relaxed-Spec
- 记录：每步 wall-clock、GPU time、每步调用次数（draft/target）、accept length 分布
- 结论：在哪些任务/阶段 accept 不够长 → SD 退化甚至变慢

### 现象 P2：推理过程高度冗余（相邻动作相似 + 跨 episode 轨迹相似）

要证明：存在“记忆可复用”的统计结构，而不是纯随机控制。

**实验 E2（冗余量化）**：
- 数据：LIBERO demo / rollout 轨迹
- 指标：`||a_t - a_{t-1}||` 分布、动作 token 重复率、state embedding 的近邻一致性
- 结论：冗余越强 → memory/draft 越有可能被接受（边界更偏向 memory）

### 现象 P3：成功经验没有被利用（memory 未进入推理环）

要证明：把成功 rollout 写回 DB，会带来 few-shot 式收益（无需训练）。

**实验 E3（Self-evolving memory）**：
- 设置：先用少量 demo 建库，跑一遍；再把成功 rollout 写回 DB，再跑一遍
- 指标：成功率提升、accept length 提升、retrieval latency 变化
- 结论：记忆“自进化”能把边界往 memory 方向推

### 现象 P4：image-only retrieval 在 LIBERO 存在语义歧义

要证明：同一场景不同指令/子任务，图像很像但动作/目标不同；只靠图像会检索错。

**实验 E4（歧义诊断 + 多模态表示消融）**：
- 设定：固定/相似图像帧，改变 instruction；比较检索结果差异
- 消融：image-only vs (image+text) vs (image+text+prev_action)
- 指标：检索命中率（task-id/轨迹片段一致性）、accept length、成功率
- 结论：多模态状态表示能显著提高“可被接受的记忆长度”

---

## 5. “边界问题”要怎么变成一个图和一个结论

建议最终把边界画成下面这种 **phase diagram**：

- 横轴：retrieval 相似度（top1 score、topK 熵、或 query–NN 距离）
- 纵轴：accept length（或 accept ratio）
- 颜色：成功率 / 每步耗时 / 回退次数

从图里抽出结论：

1) 存在一个相对稳定的分界：相似度高时 accept length 明显更长（memory 区域）  
2) 歧义/分布外时，accept length 迅速塌缩（reasoning 区域）  
3) 多模态表示、top-K 分支、以及写回记忆，会把边界往 memory 推，且不显著牺牲成功率  

---

## 6. 推荐的“最小闭环实验矩阵”（先做这个就够写故事）

### E0 基线（你已经有代码入口）

- AR：`openvla/experiments/robot/libero/run_libero_goal_AR.py`
- Spec/Relaxed：`openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py`
- 交替执行：`run_libero_goal_AR_DB.py`、`run_libero_goal_Spec_Relaxed_DB.py`

### E5 DB-as-draft（你要写的主线）

- 目标：DB 结果不直接执行，而是作为 SD draft → target 验证，记录 accept length
- 输出：accept length、回退次数、速度/成功率对比

**建议优先做的 2 个小消融**：
- DB 返回 top-1 vs top-K 平均（RT-cache 论文说平均能降噪）
- snippet 长度 N（1/2/3/5）：N 越大越“记忆化”，也越容易跑偏

---

## 7. 和代码对齐（你现在 repo 的落点）

- 你已经跑出来的 `.npz/.log` 多在 `exp/`
- LIBERO 的速度/成功率日志多在 `openvla/specdecoding/test-speed/`
- 如果需要从 `.npz` 重算/画图：用 `Lab/scripts/analyze_db_draft_acceptance.py`
- 如果需要汇总不同 run：用 `Lab/scripts/summarize_libero_runs.py`

---

## 8. 下一步我建议你先做的 3 件事（最省时间、最能出结论）

1) 先把 E1（成本分解）跑全：把“SD 不一定快”的现象写扎实  
2) 用 E4 证明 image-only 歧义：拿 LIBERO 的“同场景不同指令”做出清晰反例  
3) 在 E5 上用 accept length 做边界图：把“什么时候能用记忆”画成一张图  

---

## 9. 现阶段的经验结论（基于你已有的 DB_Tracking 结果）

### 9.1 `accept_threshold` 的速度–准确率权衡很强

你已经观察到：

- `accept_threshold=9` 时 mean accept length ≈ 0.84  
- `accept_threshold=15` 时 mean accept length ≈ 1.61  

这说明：**越放宽 token-space 接受条件，memory（DB draft）就越容易被“接受”**。

但从任务成功率角度，放宽也可能带来副作用：因为“被接受的 token”不再严格等于 target 的 argmax token，只是“在 token id 上足够接近”，对应到动作空间可能仍然会积累误差。

结论：`accept_threshold` 不能当成单纯的“加速 knob”，它是一个 **记忆–推理边界的控制旋钮**（边界往 memory 推的同时，可能牺牲稳定性）。

### 9.2 更推荐的结合方式：分离“DB 阈值”和“draft-model 阈值” + 门控

把 DB 看作 *noisy memory prior*，draft model 看作 *learned prior*，两者的误差结构不同：

- DB：可能很准，也可能因为歧义/检索错导致系统性偏差（尤其是 LIBERO 的同场景不同指令）
- draft model：更稳但更慢（你在 P1 里要证明的点）

因此一个更干净、也更容易写论文的系统形态是：

1) **DB-only candidate 验证（单候选）**：DB 只提供 *一个* draft 序列，target 只验证它（accept length 就是“信任记忆的长度”，解释性最强）  
2) **单独的 `db_accept_threshold`**：DB 的接受阈值要更保守；draft model 可以更激进（或者相反，取决于你要强调的 tradeoff）  
3) **门控（gating）**：当检索不确定（低相似度/高熵/历史 accept 很低）时，直接跳过 DB（或者降低 db_accept_threshold），回到 draft model 或 AR  

这样你就能把“边界”做成：**相似度/不确定性 → 触发门控 → accept length 分布变化 → 成功率/速度变化** 的闭环。
