# Hyper Indicator 实验说明

## 概述

`run_libero_goal_hyper_Indicator.py` 和 `run_libero_goal_hyper_Indicator.sh` 实现了基于综合指标（曲率半径 + 位移）的动态策略切换实验。

## 核心特性

### 1. 综合指标计算

使用 `CompositeMetricsCalculator` 类计算综合指标：

```
综合指标 = alpha * 曲率半径指标 + (1-alpha) * 位移指标
```

**指标归一化范围：**
- 曲率半径指标：[0.000001, 0.014615]
- 位移指标：[0.000009, 0.120187]

**位置数据来源：**
- 从 LIBERO 环境的 observation 中获取：`obs["robot0_eef_pos"]`
- 这是**机器人末端执行器的真实位置**（x, y, z），不是生成的action

### 2. 动态决策规则

```python
if composite_metric > composite_threshold:
    使用检索（纯DB）
else:
    使用AR生成（可能带verify）
```

**无verify模式的特殊规则：**
- 连续跑两次检索后，强制执行一次AR
- 避免过度依赖检索

### 3. 默认参数设置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `composite_threshold` | 0.4 | 综合指标阈值 |
| `alpha` | 0.5 | 曲率半径指标权重（两指标平均） |
| `window_size` | 5 | 滑动窗口大小 |
| `accept_threshold` | 9 | AR验证的接受阈值 |

## 使用方法

### 方法1：使用Shell脚本（推荐）

```bash
# 直接运行
bash openvla/experiments/robot/libero/run_libero_goal_hyper_Indicator.sh
```

Shell脚本会自动：
1. 恢复Qdrant数据库到base状态
2. 检查所有必需服务（Qdrant、Retrieval API、Embedding服务）
3. 激活conda环境
4. 设置CUDA设备
5. 运行Python实验脚本

### 方法2：直接运行Python脚本

```bash
# 激活环境
conda activate specvla

# 设置CUDA
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

# 运行
python openvla/experiments/robot/libero/run_libero_goal_hyper_Indicator.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/checkpoint \
    --spec_checkpoint /path/to/spec/checkpoint \
    --task_suite_name libero_goal \
    --center_crop True \
    --window_size 5 \
    --composite_threshold 0.4 \
    --alpha 0.5 \
    --num_trials_per_task 10
```

## 前置条件

运行实验前，确保以下服务已启动：

1. **Qdrant 数据库** (localhost:6333)
   ```bash
   # 检查
   curl -s "http://localhost:6333/collections"
   ```

2. **Retrieval API** (http://127.0.0.1:5002)
   ```bash
   # 启动
   cd /path/to/rtcache
   bash scripts/retrieval/start_libero_goal_retrieval.sh --skip-restore
   ```

3. **Embedding 服务** (http://127.0.0.1:9020)
   ```bash
   # 检查
   curl -s "http://127.0.0.1:9020"
   ```

## 输出文件

实验完成后，会在 `./experiments/logs/` 下生成一个时间戳目录，包含：

### 1. `log.txt`
主日志文件，包含：
- 实验配置信息
- 每个episode的执行情况
- 成功率统计
- 各模式使用次数统计

### 2. `retrieval_data.json`
详细的步骤级数据，每个步骤包含：
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

### 3. `observations_data.json`
每个episode的observations数据和结果

### 4. 视频文件
每个episode的rollout视频：
- 成功：`rollout_<episode>_success.mp4`
- 失败：`rollout_<episode>_failure.mp4`

## 调参建议

### composite_threshold（综合指标阈值）

**含义：** 决定使用检索还是AR的分界线

- **增大阈值（如0.6）**
  - 更多使用AR（因为更难超过阈值）
  - 适合：轨迹变化剧烈、需要精细控制的任务
  
- **减小阈值（如0.2）**
  - 更多使用检索（因为更容易超过阈值）
  - 适合：轨迹平滑、可重复性高的任务

- **默认0.4**
  - 平衡策略

### alpha（曲率半径权重）

**含义：** 两个指标的权重分配

```
综合指标 = alpha * 曲率半径指标 + (1-alpha) * 位移指标
```

- **alpha = 0.7**（更重视曲率）
  - 关注轨迹的弯曲程度
  - 适合：需要平滑运动的任务（如画圆）
  
- **alpha = 0.3**（更重视位移）
  - 关注移动的幅度
  - 适合：需要大幅度移动的任务（如搬运）
  
- **alpha = 0.5**（平衡）
  - 两个指标同等重要
  - **推荐默认值**

### window_size（滑动窗口大小）

**含义：** 计算指标时考虑的历史步数

- **小窗口（3-5）**
  - 对局部变化敏感
  - 反应快，适合快速变化的任务
  
- **大窗口（7-10）**
  - 更平滑，不易受噪声影响
  - 适合需要稳定决策的任务
  
- **默认5**
  - 平衡灵敏度和稳定性

## 实验对比

与其他实验的对比：

| 实验 | 决策依据 | 特点 |
|-----|---------|------|
| `run_libero_goal_Retrieval_Verify.py` | 固定比例（如N:M） | 简单，不考虑轨迹特征 |
| `run_libero_goal_new_stage.py` | 仅曲率半径 | 只考虑轨迹弯曲程度 |
| **`run_libero_goal_hyper_Indicator.py`** | **综合指标（曲率+位移）** | **同时考虑弯曲和移动幅度** |

## 典型使用场景

### 场景1：快速测试（1 trial）

```bash
# 修改shell脚本中的NUM_TRIALS
NUM_TRIALS=1

bash run_libero_goal_hyper_Indicator.sh
```

### 场景2：完整实验（10 trials）

```bash
# 默认配置
NUM_TRIALS=10

bash run_libero_goal_hyper_Indicator.sh
```

### 场景3：不同alpha值对比

```bash
# alpha=0.3 (更重视位移)
python run_libero_goal_hyper_Indicator.py --alpha 0.3 --run_id_note "alpha03"

# alpha=0.5 (平衡)
python run_libero_goal_hyper_Indicator.py --alpha 0.5 --run_id_note "alpha05"

# alpha=0.7 (更重视曲率)
python run_libero_goal_hyper_Indicator.py --alpha 0.7 --run_id_note "alpha07"
```

### 场景4：不同阈值对比

```bash
# 阈值0.3 (更多检索)
python run_libero_goal_hyper_Indicator.py --composite_threshold 0.3 --run_id_note "thresh03"

# 阈值0.5 (更多AR)
python run_libero_goal_hyper_Indicator.py --composite_threshold 0.5 --run_id_note "thresh05"
```

## 调试技巧

### 1. 查看实时决策信息

运行时会打印：
```
Step 0: mode=DB, composite=0.6234, accept_len=7
Step 1: mode=AR_verify, composite=0.3456, accept_len=5
Step 2: mode=AR_only, composite=nan, accept_len=0
```

### 2. 分析JSON数据

```python
import json
import numpy as np

# 读取数据
with open('experiments/logs/.../retrieval_data.json', 'r') as f:
    data = json.load(f)

# 统计各模式
modes = [d['mode'] for d in data]
print(f"DB: {modes.count('DB')}")
print(f"AR_verify: {modes.count('AR_verify')}")
print(f"AR_only: {modes.count('AR_only')}")

# 分析综合指标分布
composites = [d['composite_metric'] for d in data if d['composite_metric'] is not None]
print(f"Composite metric: mean={np.mean(composites):.4f}, std={np.std(composites):.4f}")
```

### 3. 检查服务状态

```bash
# 快速检查脚本
bash -c '
echo "Qdrant: $(curl -s http://localhost:6333/collections > /dev/null && echo OK || echo FAIL)"
echo "Retrieval: $(curl -s http://127.0.0.1:5002 > /dev/null && echo OK || echo FAIL)"
echo "Embedding: $(curl -s http://127.0.0.1:9020 > /dev/null && echo OK || echo FAIL)"
'
```

## 常见问题

### Q1: 为什么前几步总是使用AR？
**A:** 因为历史记录不足（window_size=5），前几步无法计算有效的综合指标，会返回nan，默认使用AR。

### Q2: 如何增加检索的使用频率？
**A:** 降低 `composite_threshold`，例如从0.4改为0.2。

### Q3: 无verify模式是什么意思？
**A:** 指不使用speculative decoding验证，直接执行检索到的action或纯AR生成。为避免过度依赖检索，连续两次检索后强制一次AR。

### Q4: 如何理解综合指标的值？
**A:** 
- 综合指标范围：[0, 1]
- 值越大：轨迹越"复杂"（曲率大或位移大）→ 适合检索
- 值越小：轨迹越"简单"（曲率小且位移小）→ 适合AR生成

## 技术细节

### 位置数据获取

```python
# 从环境observation获取末端执行器位置
eef_position = obs["robot0_eef_pos"]  # shape: (3,) - [x, y, z]

# 构造7维向量用于计算器（后4维填充0）
position_for_metrics = np.concatenate([eef_position, np.zeros(4)])

# 更新综合指标计算器
metrics_calc.update_history(position_for_metrics)
```

### 决策逻辑

```python
# 计算综合指标
composite_metric = metrics_calc.compute_composite_metric(alpha=0.5)

# 决策
if np.isnan(composite_metric):
    use_db = False  # 历史不足，默认AR
elif composite_metric > composite_threshold:
    use_db = True   # 使用检索
else:
    use_db = False  # 使用AR

# 无verify模式：连续两次检索后强制AR
if use_db:
    retrieval_counter += 1
    if retrieval_counter > 2:
        use_db = False
        retrieval_counter = 0
```

## 更多信息

- 计算类文档：`README_calc_r.md`
- 计算类代码：`calc_r.py`
- 原始实验：`run_libero_goal_Retrieval_Verify.py`
- 曲率实验：`run_libero_goal_new_stage.py`
