# SpecVLA 推理策略完整文档

## 概述

本文档描述 SpecVLA 的完整推理策略，包括：
1. **综合指标计算** - 基于位移和曲率判断轨迹稳定性
2. **2:1 策略** - Retrieval 模式下的 DB/AR 切换
3. **Block-wise SD 验证** - 基于语义分块的并行验证

---

## 一、综合指标计算

### 1.1 两个基础指标

#### 位移指标 (Displacement)
```
定义: 滑动窗口内，最后一个点与前面所有点的欧式距离之和

计算公式:
  D = Σ ||p_last - p_i||  (i = 0, 1, ..., n-2)

物理含义:
  - 小 → 轨迹聚集，机器人在小范围移动（如精细操作）
  - 大 → 轨迹分散，机器人在大范围移动（如接近目标）
```

#### 曲率半径指标 (Curvature Radius)
```
定义: 滑动窗口内轨迹的拟合圆半径（最小二乘法）

计算方法:
  1. 将3D点投影到最佳拟合平面 (SVD)
  2. 在2D平面上拟合圆
  3. 返回圆的半径 R

物理含义:
  - 小 → 轨迹弯曲剧烈（如转向、精细调整）
  - 大 → 轨迹平直（如直线移动）
```

### 1.2 归一化

使用 **MinMax 归一化**，超出范围时裁剪到 [0, 1]：

```python
def normalize(value, min_val, max_val):
    if value <= min_val:
        return 0.0
    elif value >= max_val:
        return 1.0
    else:
        return (value - min_val) / (max_val - min_val)
```

#### 归一化参数（基于 libero_goal 统计）

| 指标 | 最小值 | 最大值 |
|------|--------|--------|
| 位移 (Displacement) | 0.000009 | 0.123381 |
| 曲率半径 (Radius) | 0.000001 | 0.014989 |

### 1.3 综合指标计算

```
综合指标 = α × 归一化曲率半径 + (1-α) × 归一化位移

其中 α = 0.5 (1:1 权重)
```

**综合指标阈值**: `0.143210`

### 1.4 决策逻辑

```
if 综合指标 > 0.143210:
    使用 Retrieval 策略 (轨迹稳定，检索可信)
    └── 2:1 策略: 2次 DB + 1次 Block SD 验证
else:
    使用 SD 策略 (原始 SpecVLA Speculative Decoding)
    
if 历史不足 (无法计算指标):
    使用 AR 策略 (最保守)
```

---

## 二、2:1 策略 + Block SD

### 2.1 策略说明

当使用 **Retrieval 策略** 时，采用 2:1 的 DB/Block SD 切换：

```
┌─────────────────────────────────────────────────────────────┐
│                 2:1 策略 (Retrieval 模式)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step N:   DB (直接使用检索的 action)                        │
│  Step N+1: DB (直接使用检索的 action)                        │
│  Step N+2: Block SD 验证 ← 用检索候选做 block 级验证纠偏     │
│  Step N+3: DB                                               │
│  Step N+4: DB                                               │
│  Step N+5: Block SD 验证                                    │
│  ...                                                        │
│                                                             │
│  Block SD 验证逻辑:                                          │
│    - 检索 Top-K 候选                                         │
│    - Tree decoding 并行验证                                  │
│    - 如果验证通过 → 使用检索结果                             │
│    - 如果验证失败 → fallback 到 AR                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 为什么在 2:1 的 AR 位置用 Block SD？

```
原来的 2:1 策略:
  - DB×2: 直接用检索 (快，但可能累积误差)
  - AR×1: 纯 AR 生成 (慢，但准确)

改进后的 2:1 + Block SD:
  - DB×2: 直接用检索 (快)
  - Block SD×1: 
    - 验证通过 → 用检索 (快！只需 2 次 forward)
    - 验证失败 → 用 AR (和原来一样准确)

好处:
  - 验证通过时: 比纯 AR 快 4 倍 (2 vs 8 forward)
  - 验证失败时: 和纯 AR 一样准确
  - 保持了 2:1 的纠偏机制
```

### 2.3 实现代码

```python
# 计数器
retrieval_consecutive_db_count = 0

if use_retrieval_strategy:
    if retrieval_consecutive_db_count < 2:
        # DB 模式: 直接使用检索的 action
        action = retrieved_action
        retrieval_consecutive_db_count += 1
    else:
        # Block SD 模式: 用 block 级验证 (替代原来的纯 AR)
        tokens, stats = model.block_sd_verify(
            input_ids, top_k_tokens, blocks, prob_threshold, ...
        )
        if stats['accepted']:
            action = tokens_to_action(tokens)  # 验证通过，用检索
        else:
            action = ar_generate()  # 验证失败，用 AR
        retrieval_consecutive_db_count = 0  # 重置计数器
```

---

## 三、Block-wise SD 验证

### 3.1 Block 定义

将 7 个 action tokens 按语义分成 3 个 block：

| Block | 名称 | Token 索引 | 含义 |
|-------|------|-----------|------|
| Block 0 | position | [0, 1, 2] | xyz 位置 |
| Block 1 | orientation | [3, 4, 5] | 旋转角度 |
| Block 2 | gripper | [6] | 夹爪状态 |

```python
BLOCKS = [
    {'name': 'position', 'indices': [0, 1, 2]},
    {'name': 'orientation', 'indices': [3, 4, 5]},
    {'name': 'gripper', 'indices': [6]},
]
```

### 3.2 完整执行流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           每个 Step 的执行流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │ 获取观测图像 │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 更新历史，计算综合指标                                            │       │
│  │ composite = α × norm_radius + (1-α) × norm_displacement         │       │
│  └─────────────────────────────────┬───────────────────────────────┘       │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────┐         ┌─────────────────┐         ┌─────────────┐       │
│  │ composite   │         │ composite <=    │         │ 历史不足    │       │
│  │ > threshold │         │ threshold       │         │ (nan)       │       │
│  └──────┬──────┘         └────────┬────────┘         └──────┬──────┘       │
│         │                         │                         │              │
│         ▼                         ▼                         ▼              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │ Retrieval 策略  │     │ SD 策略         │     │ AR 策略         │       │
│  │ (2:1)           │     │ (SpecVLA SD)    │     │ (最保守)        │       │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       │
│           │                       │                       │                │
│           ▼                       │                       │                │
│  ┌─────────────────────────┐      │                       │                │
│  │ 2:1 切换判断            │      │                       │                │
│  │ count < 2 ?             │      │                       │                │
│  └─────────┬───────────────┘      │                       │                │
│       ┌────┴────┐                 │                       │                │
│       ▼         ▼                 │                       │                │
│  ┌─────────┐ ┌──────────────┐     │                       │                │
│  │ DB 模式 │ │ Block SD     │     │                       │                │
│  │ (直接用 │ │ 验证模式     │     │                       │                │
│  │ 检索)   │ │ (见下方详细) │     │                       │                │
│  └────┬────┘ └──────┬───────┘     │                       │                │
│       │             │             │                       │                │
│       └──────┬──────┘             │                       │                │
│              ▼                    ▼                       ▼                │
│       ┌──────────────┐     ┌──────────────┐        ┌──────────────┐        │
│       │ 执行 action  │     │ 执行 action  │        │ 执行 action  │        │
│       └──────────────┘     └──────────────┘        └──────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Block SD 详细流程 (`block_sd_verify`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         block_sd_verify() 内部流程                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ 快速路径检查: prob_threshold < 0 ?                                    ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║ if prob_threshold < 0:                                                ║  │
│  ║     跳过 tree_verify，直接纯 AR 生成                                   ║  │
│  ║     return (ar_tokens, stats)                                         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ Step 1: Prefill (1 次 forward)                                        ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║ 输入: image tokens + prompt tokens                                    ║  │
│  ║ 输出:                                                                 ║  │
│  ║   - prefill_logits: 第一个 action token 的 logits                     ║  │
│  ║   - current_past_kv: KV cache (用于后续 AR fallback)                  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ Step 2: Tree Verify (1 次 forward，处理 K×7=35 tokens)                ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║ Tree Mask 结构 (K=5 条链，每条 7 tokens):                              ║  │
│  ║                                                                       ║  │
│  ║   候选0: [■ ■ ■ ■ ■ ■ ■]  ← 链内 causal attention                     ║  │
│  ║   候选1: [■ ■ ■ ■ ■ ■ ■]  ← 链间无 attention                          ║  │
│  ║   候选2: [■ ■ ■ ■ ■ ■ ■]                                              ║  │
│  ║   候选3: [■ ■ ■ ■ ■ ■ ■]                                              ║  │
│  ║   候选4: [■ ■ ■ ■ ■ ■ ■]                                              ║  │
│  ║                                                                       ║  │
│  ║ Position IDs: 同位置不同候选共享 position                              ║  │
│  ║   [ctx+0, ctx+1, ..., ctx+6] × K                                      ║  │
│  ║                                                                       ║  │
│  ║ 输出: all_logits [35, vocab_size]                                     ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ Step 3: 计算每个候选每个 Token 的概率                                  ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║ 对每个候选 k, 每个位置 i:                                              ║  │
│  ║   logits_i = prefill_logits if i==0 else all_logits[k*7 + i-1]       ║  │
│  ║   token_prob = softmax(logits_i)[candidate_token[k,i]]               ║  │
│  ║                                                                       ║  │
│  ║ 结果: all_token_probs = {k: [p0, p1, p2, p3, p4, p5, p6]}             ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ Step 4: 逐 Block 剪枝验证 (Token差值方法)                             ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║ alive_candidates = [0, 1, 2, 3, 4]  # 初始 5 个候选都存活             ║  │
│  ║                                                                       ║  │
│  ║ for block in [position, orientation, gripper]:                        ║  │
│  ║                                                                       ║  │
│  ║   # 计算每个候选在该 block 的 token 差值                               ║  │
│  ║   for k in alive_candidates:                                          ║  │
│  ║     diffs = [|candidate_token[i] - argmax_token[i]|                   ║  │
│  ║              for i in block.indices]                                  ║  │
│  ║     sum_diff = sum(diffs)                                             ║  │
│  ║     max_diff = max(diffs)                                             ║  │
│  ║                                                                       ║  │
│  ║   # 接受条件（两个都要满足）                                           ║  │
│  ║   passed = [k for k in alive                                          ║  │
│  ║             if sum_diff[k] < α AND max_diff[k] < μ]                   ║  │
│  ║                                                                       ║  │
│  ║   if len(passed) > 0:                                                 ║  │
│  ║     # 有候选通过 → 剪枝，只保留通过的                                  ║  │
│  ║     alive_candidates = passed                                         ║  │
│  ║     final_tokens.extend(best_candidate's_tokens_for_this_block)       ║  │
│  ║   else:                                                               ║  │
│  ║     # 全军覆没 → AR 生成这个 block                                    ║  │
│  ║     for token_idx in block.indices:                                   ║  │
│  ║       next_token = AR_generate()                                      ║  │
│  ║       final_tokens.append(next_token)                                 ║  │
│  ║     # 基于 AR 结果继续验证下一个 block                                 ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ Step 5: 结果统计                                                      ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║  已在 Step 4 中逐 block 生成 final_tokens                             ║  │
│  ║                                                                       ║  │
│  ║  ┌────────────────────────────────────────────────────────────────┐   ║  │
│  ║  │ 情况 A: 所有 Block 都有候选通过                                 │   ║  │
│  ║  ├────────────────────────────────────────────────────────────────┤   ║  │
│  ║  │ final_tokens = [block0_best, block1_best, block2_best]         │   ║  │
│  ║  │ stats['accepted'] = True                                       │   ║  │
│  ║  │ stats['mode'] = 'fully_verified'                               │   ║  │
│  ║  │                                                                │   ║  │
│  ║  │ Forward 次数: 2 + 6 = 8 (prefill + tree + KV更新×6)            │   ║  │
│  ║  │ 注: 即使全通过也需要更新 KV cache 以维护一致性                  │   ║  │
│  ║  └────────────────────────────────────────────────────────────────┘   ║  │
│  ║                                                                       ║  │
│  ║  ┌────────────────────────────────────────────────────────────────┐   ║  │
│  ║  │ 情况 B: 部分 Block 使用 AR (剪枝后继续)                        │   ║  │
│  ║  ├────────────────────────────────────────────────────────────────┤   ║  │
│  ║  │ 例如: Block 0 通过 → Block 1 全军覆没 → AR → Block 2 通过      │   ║  │
│  ║  │ final_tokens = [retrieval, retrieval, retrieval,               │   ║  │
│  ║  │                 AR, AR, AR,                                    │   ║  │
│  ║  │                 retrieval]                                     │   ║  │
│  ║  │                                                                │   ║  │
│  ║  │ stats['accepted'] = False                                      │   ║  │
│  ║  │ stats['mode'] = 'partial_AR'                                   │   ║  │
│  ║  │ stats['num_ar_blocks'] = 1                                     │   ║  │
│  ║  │                                                                │   ║  │
│  ║  │ Forward 次数: 取决于 AR 的 block 数量                          │   ║  │
│  ║  └────────────────────────────────────────────────────────────────┘   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、剪枝验证示例

**阈值设置**: α=10 (总差值), μ=5 (单个差值)

```
初始: alive = [0, 1, 2, 3, 4] (5个候选)
argmax_tokens = [32045, 32012, 31998, 32001, 32005, 32010, 32000]

Block 0 (position, indices=[0,1,2]):
  候选0: diffs=[2, 1, 0] → sum=3, max=2 → 3<10 AND 2<5 ✓
  候选1: diffs=[15, 3, 2] → sum=20, max=15 → 20>10 ✗ 剪掉
  候选2: diffs=[1, 0, 3] → sum=4, max=3 → 4<10 AND 3<5 ✓
  候选3: diffs=[8, 5, 2] → sum=15, max=8 → 15>10 ✗ 剪掉
  候选4: diffs=[0, 2, 1] → sum=3, max=2 → 3<10 AND 2<5 ✓
  
  → alive = [0, 2, 4] (3个存活)
  → final_tokens[0:3] = 候选0的position (总差值最小)

Block 1 (orientation, indices=[3,4,5]):
  候选0: diffs=[1, 2, 1] → sum=4, max=2 → 4<10 AND 2<5 ✓
  候选2: diffs=[3, 6, 2] → sum=11, max=6 → 6>5 ✗ 剪掉
  候选4: diffs=[2, 3, 7] → sum=12, max=7 → 7>5 ✗ 剪掉
  
  → alive = [0] (1个存活)
  → final_tokens[3:6] = 候选0的orientation

Block 2 (gripper, indices=[6]):
  候选0: diffs=[0] → sum=0, max=0 → 0<10 AND 0<5 ✓
  
  → alive = [0]
  → final_tokens[6] = 候选0的gripper

结果: 全部验证通过！final_tokens = 候选0的完整action
```

**如果 Block 1 全军覆没**：
```
Block 1 (orientation): 所有存活候选的 sum>α 或 max>μ
  → AR 生成 orientation [3,4,5]
  → 基于 AR 结果继续验证 Block 2
```

---

## 五、配置参数

### Block 差值验证阈值
```bash
BLOCK_SUM_THRESHOLD=10    # α: Block 内 token 差值之和的阈值
BLOCK_MAX_THRESHOLD=5     # μ: Block 内单个 token 差值的阈值
```

**接受条件**：`sum(diff) < α` AND `max(diff) < μ`

**阈值选择建议**：
- α (总差值): 根据 block 大小调整，position 有 3 个 token，可设 α=10
- μ (单个差值): 限制单个 token 的最大偏差，可设 μ=5

---

## 六、其他配置参数

### Shell 脚本参数 (`run_libero_block_sd.sh`)

```bash
# ========================================
# 综合指标参数
# ========================================
DISPLACEMENT_RANGE_MIN=0.000009    # 位移归一化下限
DISPLACEMENT_RANGE_MAX=0.123381    # 位移归一化上限
RADIUS_RANGE_MIN=0.000001          # 曲率半径归一化下限
RADIUS_RANGE_MAX=0.014989          # 曲率半径归一化上限
COMPOSITE_THRESHOLD=0.143210       # 综合指标阈值

# ========================================
# Block SD 参数
# ========================================
TOP_K=5                            # 检索候选数量
PROB_THRESHOLD=0.1                 # Block 验证概率阈值
                                   # 设为 -1 跳过 tree_verify，直接 AR
USE_AVG_PROB="True"                # 使用几何平均概率

# Block 特定阈值 (可选)
PROB_THRESHOLD_POSITION=""         # position block 阈值
PROB_THRESHOLD_ORIENTATION=""      # orientation block 阈值  
PROB_THRESHOLD_GRIPPER=""          # gripper block 阈值
```

### 阈值选择建议

| 阈值 | 效果 |
|------|------|
| `PROB_THRESHOLD > 0.3` | 大部分验证失败，fallback 到 AR |
| `PROB_THRESHOLD = 0.05~0.2` | 平衡验证通过率和质量 |
| `PROB_THRESHOLD < 0.01` | 几乎都通过，可能接受低质量候选 |
| `PROB_THRESHOLD = -1` | 跳过 tree_verify，纯 AR 基线 |

---

## 七、复杂度对比

| 方法 | Forward 次数 | 说明 |
|------|-------------|------|
| 纯 AR | 1 + 7 = 8 | prefill + AR×7 |
| Block SD (通过) | **2** | prefill + tree_verify |
| Block SD (失败) | 2 + 6 = 8 | prefill + tree_verify + AR×6 |
| SpecVLA SD | 2~8 | EA Layer draft + token 级验证 |

**关键洞察**：
- 验证通过率高 → Block SD 更快 (2 vs 8)
- 验证通过率低 → Block SD 和 AR 差不多 (8 vs 8)
- 阈值选择很重要！

---

## 八、TODO

- [ ] 实现部分接受策略（通过的 block 用 retrieval，失败的 block 用 AR）
- [ ] 将综合指标判断集成到 Block SD 流程
- [ ] Profile 分析最优阈值
- [ ] 性能 benchmark 对比
