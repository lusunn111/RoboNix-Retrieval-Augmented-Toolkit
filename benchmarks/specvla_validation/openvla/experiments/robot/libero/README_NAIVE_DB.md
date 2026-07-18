# LIBERO 纯数据库检索实验指南

## 📦 已创建文件

1. **run_libero_naive_DB.py** - 纯DB检索实验脚本（不加载模型）
2. **run_libero_naive_DB.sh** - 启动脚本（可配置数据集）

## 🎯 功能特点

### 纯DB模式
- ✅ **不加载模型**: 节省GPU显存，启动更快
- ✅ **纯检索**: 只使用数据库检索，不进行模型推理
- ✅ **Action队列**: 每次检索取前2个action，减少API调用
- ✅ **完整统计**: 记录检索时间、成功率等指标

### 日志输出
- 保存位置: `/path/to/SpecVLA/openvla/specdecoding/test-speed/{task_suite}_naive_DB/`
- 日志格式: 
  - `.txt` - 详细运行日志
  - `_naive_DB.json` - 时间统计JSON数据

## 🚀 使用方法

### 方式1: 修改脚本变量（推荐）

编辑 `run_libero_naive_DB.sh`:

```bash
# ============================================
# Configuration - Modify these variables
# ============================================
TASK_SUITE="libero_goal"  # 修改这里选择数据集
NUM_TRIALS=50             # 每个任务的试验次数
RUN_ID_NOTE=""            # 可选的运行标记
```

数据集选项：
- `libero_goal` - LIBERO-Goal (10 tasks)
- `libero_spatial` - LIBERO-Spatial (10 tasks)
- `libero_object` - LIBERO-Object (9 tasks)
- `libero_10` - LIBERO-10 (10 tasks)

然后运行：
```bash
cd /path/to/SpecVLA/openvla/experiments/robot/libero
./run_libero_naive_DB.sh
```

### 方式2: 直接运行Python脚本

```bash
cd /path/to/SpecVLA

conda activate specvla
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=/path/to/SpecVLA/robosuite.log

python openvla/experiments/robot/libero/run_libero_naive_DB.py \
  --task_suite_name libero_goal \
  --num_trials_per_task 50 \
  --center_crop True
```

## 📊 实验示例

### 实验1: LIBERO-Goal 基线测试
```bash
# 编辑 run_libero_naive_DB.sh
TASK_SUITE="libero_goal"
NUM_TRIALS=50

# 运行
./run_libero_naive_DB.sh
```

### 实验2: LIBERO-Spatial 测试
```bash
# 编辑 run_libero_naive_DB.sh
TASK_SUITE="libero_spatial"
NUM_TRIALS=50

# 运行
./run_libero_naive_DB.sh
```

### 实验3: LIBERO-Object 测试
```bash
# 编辑 run_libero_naive_DB.sh
TASK_SUITE="libero_object"
NUM_TRIALS=50

# 运行
./run_libero_naive_DB.sh
```

### 实验4: LIBERO-10 测试
```bash
# 编辑 run_libero_naive_DB.sh
TASK_SUITE="libero_10"
NUM_TRIALS=50

# 运行
./run_libero_naive_DB.sh
```

## 📈 输出统计信息

### 控制台输出
```
[Timing] task 0: DB retrieval: 145 calls, avg 0.052341s | Total steps: 145 (50 episodes)
[Timing] task 1: DB retrieval: 132 calls, avg 0.048923s | Total steps: 132 (50 episodes)
...
================================================================================
Overall Statistics:
  DB Retrieval: 1450 calls, average time: 0.050123s
  Total steps: 1450
  Total episodes: 500
  Success rate: 375/500 (75.0%)
================================================================================

检索统计信息:
  检索次数: 1450
  检索平均每步时间: 0.050123s
  检索总时间: 72.678450s
================================================================================
```

### JSON文件内容
```json
{
  "db_times": [0.052, 0.048, 0.051, ...],
  "task_timing_stats": [
    {
      "task_id": 0,
      "num_db_calls": 145,
      "avg_db_time": 0.052341,
      "total_steps": 145,
      "num_episodes": 50
    },
    ...
  ]
}
```

## ⚙️ 环境配置

### GPU设置
- 使用 **GPU 1**: `CUDA_VISIBLE_DEVICES=1`
- EGL渲染: `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=1`

### Conda环境
- 环境名: `specvla`
- 自动激活: 脚本会自动激活conda环境

### 依赖检查
确保以下服务正在运行：
```bash
# 1. Qdrant数据库
curl http://localhost:6333/collections

# 2. 检索服务（端口5002）
curl http://127.0.0.1:5002/health
```

## 🔍 与其他模式对比

| 模式 | 脚本 | 特点 | 用途 |
|------|------|------|------|
| **纯DB** | run_libero_naive_DB.py | 只用检索，不加载模型 | 测试DB基线性能 |
| **AR+DB交替** | run_libero_goal_AR_DB.py | DB N步 + 模型M步交替 | 测试混合策略 |
| **Spec+DB** | run_libero_goal_Spec_Relaxed_DB.py | Speculative Decoding + DB | 测试推测解码 |

## 📝 实验记录建议

```markdown
### LIBERO-Goal 纯DB实验
- **时间**: 2026-01-05
- **数据集**: libero_goal
- **模式**: 纯DB检索（不加载模型）
- **试验次数**: 50/task
- **结果**:
  - 成功率: 75.0% (375/500)
  - 平均检索时间: 0.050s
  - 总检索次数: 1450
  - 总时间: 72.68s

### LIBERO-Spatial 纯DB实验
- ...
```

## 🐛 故障排除

### 问题1: ImportError
确保在SpecVLA目录下运行，且PYTHONPATH正确设置

### 问题2: 检索服务连接失败
```bash
# 检查服务状态
curl http://127.0.0.1:5002/health

# 如果服务未启动，运行：
cd /path/to/rtcache/scripts/retrieval
./start_libero_goal_retrieval.sh
```

### 问题3: Qdrant连接失败
```bash
# 检查Qdrant状态
curl http://localhost:6333/collections

# 确保collections已恢复
```

### 问题4: GPU显存不足
纯DB模式不加载模型，应该不会有显存问题。如果还有问题，检查其他程序是否占用GPU。

## 🎯 下一步

完成纯DB实验后，可以：
1. 比较不同数据集的DB检索性能
2. 分析哪些任务DB效果好，哪些效果差
3. 为混合策略（AR+DB）提供baseline数据
4. 测试添加运行时记忆的效果

---

**准备好了！现在可以运行纯DB实验了！** 🚀
