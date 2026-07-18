# Top-5 检索结果歧义性计算方法

## 1. 问题背景

在每一步检索时，我们从数据库中获取 Top-5 相似的样本，每个样本包含一个 7 维的 action 向量：

$$\mathbf{a} = [x, y, z, r_x, r_y, r_z, g]$$

其中：
- $(x, y, z)$：末端执行器的位移增量
- $(r_x, r_y, r_z)$：末端执行器的旋转增量（轴角表示）
- $g$：夹爪动作（开/关）

## 2. 几何中心（重心）计算

给定 Top-5 检索得到的 5 个 action 向量：

$$\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3, \mathbf{a}_4, \mathbf{a}_5 \in \mathbb{R}^7$$

**几何中心（Centroid）** 定义为所有向量的逐维平均：

$$\mathbf{c} = \frac{1}{5} \sum_{i=1}^{5} \mathbf{a}_i = \left[ \frac{1}{5}\sum_{i=1}^{5} x_i, \frac{1}{5}\sum_{i=1}^{5} y_i, \ldots, \frac{1}{5}\sum_{i=1}^{5} g_i \right]$$

### 示例

假设 Top-5 检索结果为：

| | $x$ | $y$ | $z$ | $r_x$ | $r_y$ | $r_z$ | $g$ |
|---|---|---|---|---|---|---|---|
| $\mathbf{a}_1$ | 0.25 | 0.00 | -0.01 | -0.03 | 0.04 | 0.01 | -1 |
| $\mathbf{a}_2$ | 0.28 | 0.00 | 0.00 | -0.04 | 0.04 | 0.01 | -1 |
| $\mathbf{a}_3$ | 0.29 | 0.00 | 0.00 | -0.04 | 0.04 | 0.01 | -1 |
| $\mathbf{a}_4$ | 0.38 | -0.01 | 0.00 | -0.04 | 0.07 | 0.01 | -1 |
| $\mathbf{a}_5$ | 0.30 | 0.02 | 0.03 | -0.05 | 0.05 | 0.02 | -1 |

几何中心：

$$\mathbf{c} = \left[ \frac{0.25+0.28+0.29+0.38+0.30}{5}, \frac{0+0+0-0.01+0.02}{5}, \ldots \right] = [0.30, 0.002, 0.004, -0.04, 0.048, 0.012, -1]$$

## 3. 歧义性（Ambiguity）计算

**歧义性** 定义为：所有 action 向量到几何中心的 **欧氏距离的平均值**。

### 3.1 单个向量到中心的距离

$$d_i = \|\mathbf{a}_i - \mathbf{c}\|_2 = \sqrt{\sum_{j=1}^{7} (a_{i,j} - c_j)^2}$$

### 3.2 歧义性定义

$$\text{Ambiguity} = \frac{1}{5} \sum_{i=1}^{5} d_i = \frac{1}{5} \sum_{i=1}^{5} \|\mathbf{a}_i - \mathbf{c}\|_2$$

## 4. 物理意义

| 歧义性值 | 含义 |
|---------|------|
| **小** | Top-5 检索结果非常一致，action 向量聚集在一起，检索结果可信度高 |
| **大** | Top-5 检索结果分散，action 向量差异较大，存在歧义，检索结果不确定性高 |

## 5. 代码实现

```python
import numpy as np

def compute_ambiguity(actions):
    """
    计算 Top-k actions 的歧义性
    
    Args:
        actions: List of action arrays, each shape (7,)
        
    Returns:
        ambiguity: 平均距离（歧义性）
    """
    if len(actions) == 0:
        return np.nan
    
    actions_array = np.array(actions)  # shape: (k, 7)
    
    # Step 1: 计算几何中心
    centroid = np.mean(actions_array, axis=0)  # shape: (7,)
    
    # Step 2: 计算每个 action 到中心的欧氏距离
    distances = [np.linalg.norm(a - centroid) for a in actions_array]
    
    # Step 3: 返回平均距离作为歧义性
    return np.mean(distances)
```

## 6. 可视化说明

生成的图表 `*_ambiguity.png` 展示了每一步的歧义性变化：

- **横轴**：时间步（Step）
- **纵轴**：歧义性（Ambiguity）
- **红色虚线**：平均歧义性

歧义性较高的时刻可能对应：
1. 场景存在多种合理的动作选择
2. 当前观测与数据库中的样本匹配度不高
3. 任务的关键转折点（如抓取、放置等）
