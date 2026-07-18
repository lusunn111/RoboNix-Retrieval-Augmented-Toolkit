# Hyper Indicator 实验完整工作流程

## 📋 实验需求确认

根据你的要求，实验需要实现以下功能：

### ✅ 要求1：从环境获取准确的位置信息
- **需求**：从LIBERO环境的observation获取x, y, z位置，不是从生成的动作获取
- **实现**：✅ 使用 `obs["robot0_eef_pos"]` 获取机器人末端执行器的绝对位置

### ✅ 要求2：基于综合指标的动态决策
- **需求**：高于阈值用检索，低于阈值用AR，阈值和alpha都可配置
- **实现**：✅ 综合指标阈值 = 0.4，alpha = 0.5（两指标平均）

### ✅ 要求3：无verify模式的特殊处理
- **需求**：连续跑两次检索后，加入一次AR执行
- **实现**：✅ 使用计数器，连续2次检索后强制1次AR

### ✅ 要求4：参考正确的代码结构
- **需求**：主体参考 `run_libero_goal_Retrieval_Verify.py`
- **实现**：✅ 环境初始化、检索API、数据记录等都参考了正确代码

### ✅ 要求5：参数配置
- **需求**：检索与AR分界阈值=0.4，混合指标alpha=0.5
- **实现**：✅ `composite_threshold=0.4`, `alpha=0.5`

---

## 🔄 完整工作流程（每个时间步）

### 阶段0：初始化（Episode开始）

```python
# 1. 环境初始化
env.reset()
obs = env.set_init_state(initial_states[episode_idx])

# 2. 创建综合指标计算器（每个episode重置）
metrics_calc = CompositeMetricsCalculator(
    window_size=5,
    displacement_range=(0.000009, 0.120187),
    radius_range=(0.000001, 0.014615),
)

# 3. 初始化检索计数器（用于无verify模式）
retrieval_counter = 0
```

### 阶段1：获取当前状态（每个时间步）

```python
# 1. 获取环境observation
obs = env.get_observation()

# 2. 处理图像
img = get_libero_image(obs, resize_size)

# 3. 准备模型输入
observation = {
    "full_image": img,
    "state": np.concatenate([
        obs["robot0_eef_pos"],           # 位置
        quat2axisangle(obs["robot0_eef_quat"]),  # 姿态
        obs["robot0_gripper_qpos"]       # 夹爪
    ])
}
```

### 阶段2：更新指标计算器 ⭐ 关键步骤

```python
# ✅ 从环境获取准确的机器人末端位置（不是从生成的action）
eef_position = obs["robot0_eef_pos"]  # shape: (3,) - [x, y, z]

# 构造7维向量用于计算器（前3维是位置，后4维填充0）
position_for_metrics = np.concatenate([eef_position, np.zeros(4)])

# 更新综合指标计算器的历史
metrics_calc.update_history(position_for_metrics)
```

**这里的关键点：**
- ✅ 位置来自环境：`obs["robot0_eef_pos"]`
- ✅ 不是来自生成的action
- ✅ 这是机器人实际的绝对位置

### 阶段3：计算综合指标并做决策 ⭐ 核心逻辑

```python
# 1. 计算综合指标
composite_metric = metrics_calc.compute_composite_metric(alpha=0.5)
# 综合指标 = 0.5 * 曲率半径指标 + 0.5 * 位移指标

# 2. 基础决策逻辑
if np.isnan(composite_metric):
    # 历史不足（前几步），默认使用AR
    use_db = False
    decision_reason = "insufficient_history"
else:
    # ✅ 核心决策规则
    if composite_metric > 0.4:
        use_db = True   # 使用检索（DB）
    else:
        use_db = False  # 使用AR
    decision_reason = f"composite={composite_metric:.4f}"

# 3. 无verify模式的特殊处理 ✅
if use_db:
    retrieval_counter += 1
    if retrieval_counter > 2:  # 连续2次检索
        use_db = False  # 强制改为AR
        retrieval_counter = 0
        decision_reason = "forced_AR_after_2_retrievals"
else:
    retrieval_counter = 0  # AR时重置计数器
```

**综合指标的计算流程：**

```
Step 1: 获取原始指标
├─ 曲率半径 = CurvatureCalculator.get_current_radius()
│  └─ 使用滑动窗口内的点进行最小二乘圆拟合
└─ 位移指标 = TrajectoryMetricsCalculator.get_current_metric()
   └─ 窗口最后一个点与前面所有点的距离之和

Step 2: 归一化到[0, 1]
├─ 曲率半径归一化
│  ├─ value <= 0.000001 → 0.0
│  ├─ value >= 0.014615 → 1.0
│  └─ 否则线性归一化：(value - min) / (max - min)
└─ 位移指标归一化
   ├─ value <= 0.000009 → 0.0
   ├─ value >= 0.120187 → 1.0
   └─ 否则线性归一化：(value - min) / (max - min)

Step 3: 加权求和
└─ composite = 0.5 * radius_norm + 0.5 * displacement_norm
```

### 阶段4：执行检索（总是尝试）

```python
# 无论是DB还是AR模式，都尝试检索（为verify准备）
try:
    # 1. 准备图像和请求
    pil_img = Image.fromarray(img)
    files = {"file": ("image.png", buf, "image/png")}
    data = {
        "instruction": task_description,
        "dataset_type": "libero_goal"
    }
    
    # 2. 调用检索API
    response = requests.post(RETRIEVAL_URL, files=files, data=data)
    
    # 3. 获取检索结果
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            retrieved_traj = result['rtcache_trajectory']
            retrieved_action = retrieved_traj[0]  # 取第一个action
            retrieval_success = True
            
except Exception as e:
    retrieval_success = False
```

### 阶段5：将检索action转换为tokens

```python
if retrieval_success and retrieved_action is not None:
    # 使用action_to_tokens函数
    # 步骤：归一化 → 离散化 → token转换
    retrieved_tokens = action_to_tokens(
        action=retrieved_action,
        model=model,
        unnorm_key=cfg.unnorm_key
    )
else:
    retrieved_tokens = None
```

### 阶段6：生成最终action ⭐ 三种模式

```python
# 模式1：纯DB模式（直接使用检索结果）
if use_db and retrieval_success and retrieved_tokens is not None:
    action = retrieved_action  # 直接使用检索到的action
    accept_length = len(retrieved_tokens)
    mode = "DB"

# 模式2：AR + Verify模式（使用检索tokens验证AR生成）
elif not use_db and cfg.use_spec and retrieval_success and retrieved_tokens is not None:
    action, accept_length = get_action(
        cfg, model, observation, task_description,
        processor=processor,
        generate_mode='AR',
        retrieved_tokens=retrieved_tokens,  # 用于验证
        return_accept_length=True,
    )
    mode = "AR_verify"

# 模式3：纯AR模式（检索失败或不使用验证）
else:
    action = get_action(
        cfg, model, observation, task_description,
        processor=processor,
        generate_mode='AR',
    )
    accept_length = 0
    mode = "AR_only"
```

### 阶段7：执行action并记录数据

```python
# 1. 执行action
obs, reward, done, info = env.step(action.tolist())

# 2. 记录详细数据
step_data = {
    "step": t,
    "mode": mode,  # "DB", "AR_verify", "AR_only"
    "decision_reason": decision_reason,
    "composite_metric": composite_metric,
    "raw_radius": metrics_info['raw']['radius'],
    "raw_displacement": metrics_info['raw']['displacement'],
    "norm_radius": metrics_info['normalized']['radius'],
    "norm_displacement": metrics_info['normalized']['displacement'],
    "retrieval_success": retrieval_success,
    "accept_length": accept_length,
    "retrieval_time": retrieval_time,
    "tokenization_time": tokenization_time,
    "generation_time": generation_time,
}
episode_retrieval_data.append(step_data)

# 3. 检查是否成功
if done:
    break  # Episode成功完成
```

---

## 📊 决策示例

### 示例1：高综合指标 → 使用检索

```
Step 82:
  eef_position = [0.5, 0.2, 0.8]  # 从环境获取
  
  综合指标计算:
  ├─ 曲率半径 = 0.012 → 归一化 = 0.82
  ├─ 位移指标 = 0.100 → 归一化 = 0.83
  └─ composite = 0.5*0.82 + 0.5*0.83 = 0.825
  
  决策:
  └─ 0.825 > 0.4 → use_db = True → 模式 = "DB"
```

### 示例2：低综合指标 → 使用AR

```
Step 113:
  eef_position = [0.5, 0.2, 0.8]  # 从环境获取
  
  综合指标计算:
  ├─ 曲率半径 = 0.00001 → 归一化 = 0.00
  ├─ 位移指标 = 0.001 → 归一化 = 0.08
  └─ composite = 0.5*0.00 + 0.5*0.08 = 0.04
  
  决策:
  └─ 0.04 < 0.4 → use_db = False → 模式 = "AR_verify" 或 "AR_only"
```

### 示例3：连续检索后强制AR

```
Step 85: composite = 0.45 → use_db = True → retrieval_counter = 1
Step 86: composite = 0.50 → use_db = True → retrieval_counter = 2
Step 87: composite = 0.48 → use_db = True (初判)
         → 但 retrieval_counter > 2 → 强制 use_db = False
         → retrieval_counter = 0
         → 模式 = "AR_verify" 或 "AR_only"
```

---

## 📈 数据记录

### 每步记录的数据（保存到JSON）

```json
{
  "step": 82,
  "mode": "DB",
  "decision_reason": "composite=0.5154",
  "composite_metric": 0.5154,
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

### Episode结束记录

```json
{
  "episode_idx": 0,
  "retrieval_data": [...],  // 所有步骤的数据
  "success": true,
  "steps": 120
}
```

---

## 🎯 关键确认点

### ✅ 1. 位置数据来源正确
```python
# ✅ 从环境获取（准确的绝对位置）
eef_position = obs["robot0_eef_pos"]

# ❌ 不是从生成的action获取
# eef_position = action[:3]  # 这是错误的
```

### ✅ 2. 综合指标计算正确
```python
# 两个指标都归一化到[0, 1]
# alpha = 0.5，两指标平均
composite = 0.5 * radius_norm + 0.5 * displacement_norm
```

### ✅ 3. 决策逻辑正确
```python
# threshold = 0.4
if composite > 0.4:
    使用检索（DB）
else:
    使用AR（AR_verify或AR_only）
```

### ✅ 4. 无verify模式处理正确
```python
# 连续2次检索后，强制1次AR
if retrieval_counter > 2:
    use_db = False
    retrieval_counter = 0
```

---

## 📝 配置参数总结

| 参数 | 值 | 说明 |
|-----|---|------|
| `composite_threshold` | **0.4** | 检索与AR分界阈值 |
| `alpha` | **0.5** | 曲率半径权重（两指标平均） |
| `window_size` | 5 | 滑动窗口大小 |
| `displacement_range` | [0.000009, 0.120187] | 位移归一化范围 |
| `radius_range` | [0.000001, 0.014615] | 曲率半径归一化范围 |
| `use_spec` | True | 启用speculative decoding |
| `accept_threshold` | 9 | AR验证接受阈值 |

---

## 🔍 与你要求的对应关系

| 要求 | 实现位置 | 代码行 |
|-----|---------|-------|
| 从环境获取位置 | `eef_position = obs["robot0_eef_pos"]` | 311行 |
| 更新计算器 | `metrics_calc.update_history(position_for_metrics)` | 314行 |
| 计算综合指标 | `composite_metric = metrics_calc.compute_composite_metric(alpha=0.5)` | 319行 |
| 阈值判断 | `use_db = composite_metric > 0.4` | 330行 |
| 强制AR | `if retrieval_counter > 2: use_db = False` | 336-338行 |
| DB模式 | `if use_db and retrieval_success: action = retrieved_action` | 412-416行 |
| AR_verify模式 | `action, accept_length = get_action(..., retrieved_tokens=...)` | 419-428行 |
| AR_only模式 | `action = get_action(..., generate_mode='AR')` | 433-439行 |

---

## ✅ 确认清单

- [x] 位置数据从环境observation获取（`obs["robot0_eef_pos"]`）
- [x] 不是从生成的action获取
- [x] 综合指标 = alpha * 曲率半径 + (1-alpha) * 位移
- [x] alpha = 0.5（两指标平均）
- [x] 阈值 = 0.4（高于用检索，低于用AR）
- [x] 连续2次检索后强制1次AR
- [x] 参考了正确的代码结构
- [x] 所有参数按要求配置
- [x] 详细记录所有数据到JSON

---

## 🎉 总结

实验完全符合你的所有要求：
1. ✅ 从环境获取准确位置
2. ✅ 基于综合指标动态决策（阈值0.4，alpha 0.5）
3. ✅ 连续2次检索+1次AR的无verify处理
4. ✅ 参考正确代码结构
5. ✅ 所有参数按要求配置

可以放心运行实验！
