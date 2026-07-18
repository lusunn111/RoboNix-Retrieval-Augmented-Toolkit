# 轨迹指标计算模块使用说明

本模块提供了三个用于轨迹分析的计算类，位于 `calc_r.py` 文件中。

## 三个计算类

### 1. CurvatureCalculator - 曲率半径计算器

使用最小二乘法拟合圆来计算轨迹的曲率半径。

**主要功能：**
- 基于滑动窗口计算局部曲率半径
- 根据曲率半径阈值决定使用检索还是AR生成

**使用示例：**
```python
from calc_r import CurvatureCalculator

# 创建计算器（窗口大小5，阈值0.006米）
calc = CurvatureCalculator(window_size=5, curvature_threshold=0.006)

# 每个时间步更新action
action = np.array([x, y, z, rx, ry, rz, gripper])  # shape (7,)
calc.update_history(action)

# 获取当前曲率半径
radius = calc.get_current_radius()

# 判断是否使用检索
use_retrieval = calc.should_use_retrieval()  # True=检索, False=AR

# 获取详细信息
info = calc.get_decision_info()
# info = {'radius': 0.012, 'threshold': 0.006, 'history_length': 10, 
#         'use_retrieval': True, 'decision': 'Retrieval'}

# 新episode开始时清空历史
calc.clear_history()
```

---

### 2. TrajectoryMetricsCalculator - 轨迹位移指标计算器

计算滑动窗口内，窗口最后一个点与前面所有点的欧式距离之和。

**主要功能：**
- 基于滑动窗口计算位移指标
- 反映轨迹的分散程度

**使用示例：**
```python
from calc_r import TrajectoryMetricsCalculator

# 创建计算器（窗口大小5）
calc = TrajectoryMetricsCalculator(window_size=5)

# 每个时间步更新action
action = np.array([x, y, z, rx, ry, rz, gripper])  # shape (7,) 或 (3,)
calc.update_history(action)

# 获取当前位移指标
metric = calc.get_current_metric()

# 获取详细信息
info = calc.get_metric_info()
# info = {'displacement_metric': 0.100, 'history_length': 10, 'window_size': 5}

# 新episode开始时清空历史
calc.clear_history()
```

---

### 3. CompositeMetricsCalculator - 综合指标计算器

结合曲率半径指标和位移指标，进行归一化后加权求和。

**主要功能：**
- 同时调用前面两个计算器
- 自动归一化（低于下限为0，高于上限为1）
- 按权重计算综合指标：`alpha * 曲率半径指标 + (1-alpha) * 位移指标`

**归一化范围：**
- 位移指标：[0.000009, 0.120187]
- 曲率半径指标：[0.000001, 0.014615]

**使用示例：**
```python
from calc_r import CompositeMetricsCalculator

# 创建计算器（窗口大小5，默认归一化范围）
calc = CompositeMetricsCalculator(
    window_size=5,
    displacement_range=(0.000009, 0.120187),  # 可选，使用默认值
    radius_range=(0.000001, 0.014615)          # 可选，使用默认值
)

# 每个时间步更新action
action = np.array([x, y, z, rx, ry, rz, gripper])  # shape (7,)
calc.update_history(action)

# 计算综合指标（alpha=0.5表示两个指标权重相同）
composite = calc.compute_composite_metric(alpha=0.5)

# 获取所有指标的详细信息
metrics = calc.get_current_metrics(alpha=0.7)  # alpha=0.7表示更重视曲率半径
# metrics = {
#     'raw': {
#         'radius': 0.012,           # 原始曲率半径
#         'displacement': 0.100      # 原始位移指标
#     },
#     'normalized': {
#         'radius': 0.8211,          # 归一化后的曲率半径
#         'displacement': 0.8320     # 归一化后的位移指标
#     },
#     'composite': 0.8265,           # 综合指标
#     'alpha': 0.7,                  # 使用的权重
#     'history_length': 10           # 历史长度
# }

# 新episode开始时清空历史
calc.clear_history()
```

---

## 典型使用场景

### 场景1：在线决策（使用检索还是AR）

```python
from calc_r import CurvatureCalculator

calc = CurvatureCalculator(window_size=5, curvature_threshold=0.006)

for step in range(max_steps):
    # ... 获取observation ...
    
    # 根据曲率半径决定策略
    if calc.should_use_retrieval():
        action = retrieval_policy(obs)
    else:
        action = ar_policy(obs)
    
    # 更新历史
    calc.update_history(action)
    
    # ... 执行action ...
```

### 场景2：综合评估轨迹质量

```python
from calc_r import CompositeMetricsCalculator

calc = CompositeMetricsCalculator(window_size=5)
alpha = 0.6  # 稍微更重视曲率半径

for step in range(max_steps):
    # ... 获取action ...
    calc.update_history(action)
    
    # 计算综合指标
    score = calc.compute_composite_metric(alpha=alpha)
    
    if not np.isnan(score):
        print(f"Step {step}: Quality Score = {score:.4f}")
        
        # 根据分数做决策
        if score > 0.8:
            print("轨迹质量高")
        elif score > 0.5:
            print("轨迹质量中等")
        else:
            print("轨迹质量低")
```

### 场景3：离线分析

```python
from calc_r import CompositeMetricsCalculator
import numpy as np

# 假设已有完整轨迹数据
trajectory = np.load('trajectory.npy')  # shape (T, 7)

calc = CompositeMetricsCalculator(window_size=5)
scores = []

for action in trajectory:
    calc.update_history(action)
    score = calc.compute_composite_metric(alpha=0.5)
    scores.append(score)

# 分析整个轨迹
scores = np.array(scores)
valid_scores = scores[~np.isnan(scores)]

print(f"平均质量分数: {np.mean(valid_scores):.4f}")
print(f"最高质量分数: {np.max(valid_scores):.4f}")
print(f"最低质量分数: {np.min(valid_scores):.4f}")
```

---

## 参数调整建议

### window_size（滑动窗口大小）
- **小窗口 (3-5)**：对局部变化敏感，适合精细控制
- **大窗口 (7-10)**：更平滑，适合全局轨迹评估
- 默认推荐：**5**

### alpha（综合指标权重）
- **alpha > 0.5**：更重视曲率半径（轨迹弯曲程度）
- **alpha < 0.5**：更重视位移指标（轨迹分散程度）
- **alpha = 0.5**：两者权重相同
- 根据任务特点选择：
  - 需要平滑轨迹的任务：alpha = 0.6-0.7
  - 需要快速移动的任务：alpha = 0.3-0.4
  - 平衡考虑：alpha = 0.5

### curvature_threshold（曲率半径阈值）
- 用于 `CurvatureCalculator` 决定使用检索还是AR
- 默认：**0.006米**
- 调整建议：
  - 增大阈值：更多使用AR（适合需要精细控制的任务）
  - 减小阈值：更多使用检索（适合需要稳定性的任务）

---

## 注意事项

1. **历史记录不足**：前几个时间步可能返回 `np.nan`，因为窗口内点数不够
2. **计算失败**：如果点分布过于特殊（如完全共线），可能返回 `np.nan`
3. **episode边界**：每个新episode开始时记得调用 `clear_history()`
4. **坐标系统**：确保action的前3维是xyz位置坐标（单位：米）

---

## 测试

运行测试代码：
```bash
python calc_r.py
```

测试包括：
1. 直线、圆形等不同轨迹的曲率半径计算
2. 位移指标计算
3. 综合指标计算（不同alpha值）
