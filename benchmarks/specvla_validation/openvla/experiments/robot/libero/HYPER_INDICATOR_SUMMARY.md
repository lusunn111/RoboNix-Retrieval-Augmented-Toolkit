# Hyper Indicator 实验 - 完成总结

## 📋 任务完成清单

### ✅ 已完成的工作

#### 1. 创建计算类模块（calc_r.py）
- ✅ **CurvatureCalculator** - 曲率半径计算（最小二乘圆法）
- ✅ **TrajectoryMetricsCalculator** - 轨迹位移指标计算
- ✅ **CompositeMetricsCalculator** - 综合指标计算

#### 2. 创建实验脚本
- ✅ **run_libero_goal_hyper_Indicator.py** - Python主实验脚本
- ✅ **run_libero_goal_hyper_Indicator.sh** - Shell启动脚本

#### 3. 创建文档
- ✅ **README_calc_r.md** - 计算类使用说明
- ✅ **README_hyper_Indicator.md** - 实验使用说明
- ✅ **HYPER_INDICATOR_SUMMARY.md** - 完成总结（本文档）

---

## 🎯 核心功能实现

### 1. 从环境获取位置信息 ✅

```python
# 从LIBERO环境的observation中获取机器人末端位置（准确的绝对位置）
eef_position = obs["robot0_eef_pos"]  # shape: (3,) - [x, y, z]

# 更新综合指标计算器
position_for_metrics = np.concatenate([eef_position, np.zeros(4)])
metrics_calc.update_history(position_for_metrics)
```

**关键点：**
- ✅ 从环境获取，不是从生成的action获取
- ✅ 使用 `obs["robot0_eef_pos"]` 获取准确的xyz坐标
- ✅ 每个时间步都更新计算器

### 2. 基于综合指标的动态决策 ✅

```python
# 计算综合指标
composite_metric = metrics_calc.compute_composite_metric(alpha=cfg.alpha)

# 决策逻辑
if composite_metric > cfg.composite_threshold:
    use_db = True   # 高于阈值 → 使用检索
else:
    use_db = False  # 低于阈值 → 使用AR
```

**参数设置（按要求）：**
- ✅ 检索与AR的分界阈值：**0.4**
- ✅ 混合指标的alpha：**0.5**（两个指标平均）

### 3. 无verify模式的特殊处理 ✅

```python
# 连续两次检索后，强制执行一次AR
if use_db:
    retrieval_counter += 1
    if retrieval_counter > 2:
        use_db = False
        retrieval_counter = 0
        decision_reason = "forced_AR_after_2_retrievals"
else:
    retrieval_counter = 0
```

**关键点：**
- ✅ 连续跑两次检索
- ✅ 加入一次AR执行
- ✅ 记录决策原因

### 4. 参考正确的代码结构 ✅

主体参考 `run_libero_goal_Retrieval_Verify.py`：
- ✅ 环境初始化
- ✅ Episode循环结构
- ✅ 检索API调用
- ✅ Action生成逻辑
- ✅ 数据记录格式

借鉴 `run_libero_goal_new_stage.py`：
- ✅ 从环境获取位置
- ✅ 动态决策逻辑
- ✅ 计算器使用方式

### 5. Shell脚本环境检查 ✅

参考 `run_libero_goal_Retrieval_Verify.sh`：
- ✅ 数据库恢复到base状态
- ✅ 服务可用性检查（Qdrant、Retrieval API、Embedding）
- ✅ Conda环境激活
- ✅ CUDA设备设置
- ✅ 详细的日志输出

---

## 📊 三个计算类详解

### 1. CurvatureCalculator（曲率半径）

**输入：** 滑动窗口内的3D点
**输出：** 曲率半径R（单位：米）

**计算方法：**
1. SVD分解找最佳拟合平面
2. 将3D点投影到2D平面
3. 最小二乘法拟合圆
4. 返回半径R

**物理意义：**
- R大 → 轨迹平直
- R小 → 轨迹弯曲剧烈

### 2. TrajectoryMetricsCalculator（位移指标）

**输入：** 滑动窗口内的点 [p1, p2, ..., p5]
**输出：** 位移指标值

**计算公式：**
```
位移指标 = ||p5-p1|| + ||p5-p2|| + ||p5-p3|| + ||p5-p4||
```

**物理意义：**
- 值大 → 当前点离历史点远，移动幅度大
- 值小 → 当前点离历史点近，移动幅度小

### 3. CompositeMetricsCalculator（综合指标）

**输入：** 机器人末端位置
**输出：** 综合指标 [0, 1]

**计算流程：**
1. 调用前两个计算器获取原始值
2. 按范围归一化到[0, 1]
   - 曲率半径：[0.000001, 0.014615] → [0, 1]
   - 位移指标：[0.000009, 0.120187] → [0, 1]
3. 加权求和：`alpha * 曲率 + (1-alpha) * 位移`

**归一化规则：**
- value ≤ min → 0.0
- value ≥ max → 1.0
- min < value < max → 线性归一化

---

## 🔧 参数配置

### 默认参数（已按要求设置）

| 参数 | 值 | 说明 |
|-----|---|------|
| `composite_threshold` | **0.4** | 检索与AR的分界阈值 |
| `alpha` | **0.5** | 两个指标平均 |
| `window_size` | 5 | 滑动窗口大小 |
| `displacement_range` | [0.000009, 0.120187] | 位移归一化范围 |
| `radius_range` | [0.000001, 0.014615] | 曲率半径归一化范围 |
| `accept_threshold` | 9 | AR验证接受阈值 |
| `num_trials_per_task` | 10 | 每个任务的试验次数 |

### 可调参数说明

**composite_threshold（综合指标阈值）**
- 增大 → 更多AR（更难超过阈值）
- 减小 → 更多检索（更容易超过阈值）
- 默认0.4：平衡策略

**alpha（曲率半径权重）**
- 0.7 → 更重视曲率（适合需要平滑运动的任务）
- 0.5 → 平衡（推荐）
- 0.3 → 更重视位移（适合大幅度移动的任务）

---

## 📁 文件结构

```
openvla/experiments/robot/libero/
├── calc_r.py                           # 计算类模块
├── run_libero_goal_hyper_Indicator.py  # Python实验脚本
├── run_libero_goal_hyper_Indicator.sh  # Shell启动脚本
├── README_calc_r.md                    # 计算类文档
├── README_hyper_Indicator.md           # 实验使用文档
└── HYPER_INDICATOR_SUMMARY.md          # 完成总结（本文档）
```

---

## 🚀 快速开始

### 1. 基础使用

```bash
# 直接运行（推荐）
bash openvla/experiments/robot/libero/run_libero_goal_hyper_Indicator.sh
```

### 2. 自定义参数

```bash
# 修改shell脚本中的参数
COMPOSITE_THRESHOLD=0.4
ALPHA=0.5
NUM_TRIALS=10

# 或直接运行Python脚本
python openvla/experiments/robot/libero/run_libero_goal_hyper_Indicator.py \
    --composite_threshold 0.4 \
    --alpha 0.5 \
    --num_trials_per_task 10
```

---

## 📈 输出数据

### 日志目录
```
./experiments/logs/HYPER_INDICATOR-libero_goal-alpha0.5-thresh0.4-YYYYMMDD_HHMMSS/
├── log.txt                  # 主日志
├── retrieval_data.json      # 详细步骤数据
├── observations_data.json   # Observations数据
└── rollout_*.mp4           # 视频文件
```

### retrieval_data.json 格式

每个步骤包含：
```json
{
  "step": 0,
  "mode": "DB/AR_verify/AR_only",
  "decision_reason": "composite=0.6234",
  "composite_metric": 0.6234,
  "raw_radius": 0.012,
  "raw_displacement": 0.100,
  "norm_radius": 0.8211,
  "norm_displacement": 0.8320,
  "retrieval_success": true,
  "accept_length": 7,
  "retrieval_time": 0.045,
  "tokenization_time": 0.002,
  "generation_time": 0.123,
  "total_time": 0.170
}
```

---

## ✅ 验证清单

### 代码正确性
- ✅ Python语法检查通过
- ✅ Linter检查无错误
- ✅ Shell脚本添加执行权限
- ✅ 计算类测试通过

### 功能实现
- ✅ 从环境获取位置（obs["robot0_eef_pos"]）
- ✅ 综合指标计算（alpha=0.5）
- ✅ 动态决策（threshold=0.4）
- ✅ 无verify模式（连续2次检索+1次AR）
- ✅ 参考正确的代码结构

### 文档完整性
- ✅ 计算类使用说明
- ✅ 实验使用说明
- ✅ 参数调整建议
- ✅ 调试技巧
- ✅ 常见问题解答

---

## 🎓 关键技术点

### 1. 位置数据来源

**正确做法 ✅**
```python
eef_position = obs["robot0_eef_pos"]  # 从环境获取
```

**错误做法 ❌**
```python
position = action[:3]  # 从生成的action获取（不准确）
```

### 2. 综合指标归一化

```python
def normalize_value(value, value_range):
    min_val, max_val = value_range
    
    if value <= min_val:
        return 0.0
    elif value >= max_val:
        return 1.0
    else:
        return (value - min_val) / (max_val - min_val)
```

### 3. 决策逻辑

```python
# 基础决策
use_db = composite_metric > composite_threshold

# 无verify模式：连续2次检索后强制AR
if use_db:
    retrieval_counter += 1
    if retrieval_counter > 2:
        use_db = False
        retrieval_counter = 0
```

---

## 🔍 与其他实验的对比

| 特性 | Retrieval_Verify | new_stage | **hyper_Indicator** |
|-----|-----------------|-----------|---------------------|
| 决策依据 | 固定比例（N:M） | 仅曲率半径 | **综合指标（曲率+位移）** |
| 位置数据 | - | 环境获取 | **环境获取** |
| 归一化 | 无 | 无 | **有（两个指标）** |
| 权重调节 | 无 | 无 | **可调（alpha）** |
| 无verify处理 | 固定比例 | 无 | **连续2次检索+1次AR** |
| 灵活性 | 低 | 中 | **高** |

---

## 📝 使用建议

### 1. 首次测试
```bash
# 使用默认参数，每个任务只跑1次
# 修改shell脚本：NUM_TRIALS=1
bash run_libero_goal_hyper_Indicator.sh
```

### 2. 完整实验
```bash
# 使用默认参数，每个任务跑10次
bash run_libero_goal_hyper_Indicator.sh
```

### 3. 参数调优
```bash
# 测试不同alpha值
for alpha in 0.3 0.5 0.7; do
    python run_libero_goal_hyper_Indicator.py \
        --alpha $alpha \
        --run_id_note "alpha${alpha/./}"
done

# 测试不同阈值
for thresh in 0.3 0.4 0.5; do
    python run_libero_goal_hyper_Indicator.py \
        --composite_threshold $thresh \
        --run_id_note "thresh${thresh/./}"
done
```

---

## 🎯 总结

### 已完成的核心要求

1. ✅ **位置数据获取**：从环境observation中获取准确的机器人末端位置
2. ✅ **动态决策**：高于阈值0.4检索，低于阈值0.4 AR
3. ✅ **综合指标**：alpha=0.5，两个指标平均
4. ✅ **无verify处理**：连续两次检索后强制一次AR
5. ✅ **代码结构**：参考了正确的代码模板
6. ✅ **环境检查**：Shell脚本包含完整的服务检查

### 创建的文件

1. **calc_r.py** - 三个计算类（已测试）
2. **run_libero_goal_hyper_Indicator.py** - Python实验脚本
3. **run_libero_goal_hyper_Indicator.sh** - Shell启动脚本
4. **README_calc_r.md** - 计算类文档
5. **README_hyper_Indicator.md** - 实验文档
6. **HYPER_INDICATOR_SUMMARY.md** - 本总结

### 下一步建议

1. **测试运行**：先用1个trial测试是否正常运行
2. **参数调优**：尝试不同的alpha和threshold组合
3. **结果分析**：对比不同配置下的成功率和模式分布
4. **论文实验**：确定最佳参数后进行完整的10 trials实验

---

## 📞 如有问题

- 查看：`README_hyper_Indicator.md` - 详细使用说明
- 查看：`README_calc_r.md` - 计算类API文档
- 调试：检查 `log.txt` 和 `retrieval_data.json`
- 测试：运行 `python calc_r.py` 测试计算类

祝实验顺利！ 🎉
