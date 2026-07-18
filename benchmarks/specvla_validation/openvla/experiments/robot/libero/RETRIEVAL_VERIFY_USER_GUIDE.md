# Retrieval Verification - 完整实验指南

## 概述
检索验证实验用于评估检索到的action与模型生成action的匹配程度（accept_length）。

## 实验配置
- **Accept Threshold**: 9
- **数据库**: 默认使用base backup
- **任务套件**: libero_goal, libero_object, libero_spatial, libero_10

## 前置条件

### 1. 服务启动
确保以下服务正在运行：

```bash
# Retrieval API (端口 5002)
curl http://127.0.0.1:5002/health

# Embedding Server (端口 9020)
curl http://127.0.0.1:9020/health

# Qdrant (端口 6333)
curl http://localhost:6333
```

### 2. 环境激活
```bash
conda activate specvla
```

### 3. 环境变量
```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

## 运行方式

### 方式1：运行所有任务套件（推荐）

使用总控制脚本一次性运行所有4个任务套件：

```bash
cd /path/to/SpecVLA/openvla/experiments/robot/libero
./run_all_suites_Retrieval_Verify.sh
```

**功能**：
- 自动运行 goal → object → spatial → long (10) 所有4个套件
- 每个套件运行前自动恢复数据库到base状态
- 为每个套件创建独立的日志目录
- 最后生成汇总报告

**输出位置**：
```
TGT_DIR/
├── retrieval_verify_goal_2025_XX_XX-XX_XX_XX/
│   ├── console_output.txt
│   └── EVAL-libero_goal-*.txt
├── retrieval_verify_object_2025_XX_XX-XX_XX_XX/
├── retrieval_verify_spatial_2025_XX_XX-XX_XX_XX/
├── retrieval_verify_10_2025_XX_XX-XX_XX_XX/
└── retrieval_verify_summary/
    └── summary_2025_XX_XX-XX_XX_XX.txt  # 汇总报告
```

### 方式2：运行单个任务套件

如果只需要运行特定的任务套件：

```bash
# 运行 goal
./run_libero_goal_Retrieval_Verify.sh

# 运行 object
python run_libero_object_Retrieval_Verify.py

# 运行 spatial
python run_libero_spatial_Retrieval_Verify.py

# 运行 long (libero_10)
python run_libero_10_Retrieval_Verify.py
```

## 统计指标

每个任务套件会输出以下统计：

### 1. 任务级别统计
每个任务会显示：
- 检索尝试次数
- 成功检索率
- 平均检索时间
- 平均生成时间
- **Accept Length 统计**:
  - Mean (平均值)
  - Median (中位数)
  - Std (标准差)
  - Min/Max (最小/最大值)

### 2. 整体统计
所有任务结束后显示：
- 总episodes数量
- 总成功数
- **Success Rate (SR)** - 成功率
- **整体Accept Length统计** (所有任务汇总)

### 3. 汇总报告格式

`summary_*.txt` 包含所有套件的对比：

```
Suite          | SR      | Accept Mean | Episodes | Duration
--------------------------------------------------------------------
libero_goal    | 85.0%   | 5.32       | 50       | 1234s
libero_object  | 78.5%   | 4.89       | 50       | 1156s
libero_spatial | 82.3%   | 5.01       | 50       | 1289s
libero_10      | 76.8%   | 4.67       | 100      | 2345s
```

## 输出文件

### 1. 控制台日志 (.txt)
包含完整的运行信息：
- 每个episode的步数和成功状态
- 每个任务的详细统计
- 整体统计汇总

### 2. 检索验证数据 (.json)
结构化数据，包含每一步的：
```json
{
  "task_id": 0,
  "episodes": [
    {
      "episode_idx": 0,
      "success": true,
      "steps": [
        {
          "episode": 0,
          "step": 0,
          "retrieval_success": true,
          "retrieval_time": 0.023,
          "tokenization_time": 0.001,
          "generation_time": 0.145,
          "accept_length": 5,
          "has_retrieved_tokens": true
        }
      ]
    }
  ]
}
```

## 数据库管理

### 自动恢复
总控制脚本会在每个套件运行前自动恢复数据库：
- 从 `backup_base` 恢复
- 确保每个套件使用相同的初始数据库状态

### 手动恢复
如果需要手动恢复数据库：
```bash
conda activate rt-mzh
python /path/to/rtcache/scripts/retrieval/restore_qdrant.py \
    --backup-dir /path/to/rtcache/scripts/retrieval/qdrant_backups/backup_base
conda activate specvla
```

## 关键参数说明

### Accept Threshold
- 默认值: 9
- 含义: 用于speculative decoding的阈值
- 位置: 在各个 `run_libero_*_Retrieval_Verify.py` 中可修改

### 每任务试验次数
- goal/object/spatial: 默认50次 (num_trials_per_task)
- libero_10: 默认10次 (因为有10个任务，总共100次)

### 检索URL
- Retrieval API: http://127.0.0.1:5002/pipeline
- Embedding Server: http://127.0.0.1:9020/predict

## 故障排除

### 服务未响应
确保所有服务都在运行：
```bash
# 检查Retrieval API
curl http://127.0.0.1:5002/health

# 检查Embedding Server
curl http://127.0.0.1:9020/health

# 检查Qdrant
curl http://localhost:6333
```

### Conda环境错误
确保使用正确的环境名：
```bash
conda activate specvla  # 不是 openvla
```

### MUJOCO错误
确保设置了环境变量：
```bash
export MUJOCO_EGL_DEVICE_ID=0
```

### 数据库恢复失败
检查backup路径是否存在：
```bash
ls /path/to/rtcache/scripts/retrieval/qdrant_backups/backup_base
```

## 实验流程总结

1. **前置准备**
   - 启动所有服务 (Retrieval, Embedding, Qdrant)
   - 激活conda环境
   - 设置环境变量

2. **运行实验**
   - 使用总控制脚本运行所有套件
   - 或单独运行特定套件

3. **查看结果**
   - 检查各套件的日志文件
   - 查看汇总报告
   - 分析accept_length统计

4. **数据分析**
   - 比较不同套件的SR和accept_length
   - 分析检索质量与成功率的关系
   - 优化检索策略

## 下一步

运行完成后，可以：
1. 分析 `.json` 数据文件进行深度统计
2. 绘制accept_length分布图
3. 对比不同任务套件的表现
4. 调整检索策略或阈值参数
