# DB检索接受长度追踪使用指南

## 功能说明

这个功能用于追踪从DB（数据库）检索得到的action slice的接受长度。当使用DB检索模式时，系统会：
1. 从检索API获取action slice（包含多个actions的序列）
2. 追踪实际使用了多少个actions（接受长度）
3. 保存检索到的action slice和对应的接受长度
4. 生成可视化图表

## 前置条件

### 1. 启动Qdrant数据库

```bash
cd /path/to/rtcache
./start_db.sh
```

### 2. 启动Embedding服务器

确保OpenVLA embedding服务器在运行（默认端口9020）。

### 3. 启动检索服务器

```bash
cd /path/to/rtcache/scripts/retrieval
./start_libero_goal_retrieval.sh
```

或者手动启动：

```bash
python retrieval_libero_goal.py \
    --host 0.0.0.0 \
    --port 5002 \
    --embedding-url http://127.0.0.1:9020/predict \
    --qdrant-host localhost \
    --qdrant-port 6333 \
    --log-level INFO
```

### 4. 验证服务

健康检查：
```bash
curl http://localhost:5002/health
```

## 运行方法

### 方法1: 使用提供的脚本（推荐）

```bash
cd /path/to/SpecVLA
bash openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed_DB_Tracking.sh
```

### 方法2: 手动运行完整命令

```bash
# 进入SpecVLA目录
cd /path/to/SpecVLA

# 激活conda环境
conda activate specvla

# 设置环境变量
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$PWD/robosuite.log
export CUDA_VISIBLE_DEVICES=0  # 根据你的GPU情况修改
export MUJOCO_EGL_DEVICE_ID=0   # 根据你的GPU情况修改

# 运行脚本
python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint $PWD/backbone_models/openvla-7b-finetuned-libero-goal \
    --spec_checkpoint $PWD/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190 \
    --task_suite_name libero_goal \
    --center_crop True \
    --use_spec True \
    --parallel_draft False \
    --accept_threshold 9 \
    --use_db_retrieval True \
    --track_accept_length True \
    --retrieval_url "http://127.0.0.1:5002/pipeline" \
    --num_trials_per_task 10 \
    --run_id_note "DB_Tracking" \
    --use_wandb False
```

### 方法3: 一行命令（适合快速测试）

```bash
cd /path/to/SpecVLA && \
conda activate specvla && \
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO && \
export MUJOCO_GL=egl && \
export ROBOSUITE_LOG_FILE=$PWD/robosuite.log && \
export CUDA_VISIBLE_DEVICES=0 && \
export MUJOCO_EGL_DEVICE_ID=0 && \
python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint $PWD/backbone_models/openvla-7b-finetuned-libero-goal \
    --spec_checkpoint $PWD/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190 \
    --task_suite_name libero_goal \
    --center_crop True \
    --use_spec True \
    --use_db_retrieval True \
    --track_accept_length True \
    --retrieval_url "http://127.0.0.1:5002/pipeline" \
    --num_trials_per_task 10
```

## 重要参数说明

- `--use_db_retrieval True`: 启用DB检索模式（必须）
- `--track_accept_length True`: 启用接受长度追踪（必须）
- `--retrieval_url`: DB检索API的URL（默认: http://127.0.0.1:5002/pipeline）
- `--num_trials_per_task`: 每个任务的试验次数（默认: 10）
- `--accept_threshold`: 接受阈值（仅在使用SpecVLA时有效，DB模式不使用）
- `--CUDA_VISIBLE_DEVICES`: 指定使用的GPU（根据实际情况修改）
- `--MUJOCO_EGL_DEVICE_ID`: 指定MUJOCO使用的GPU（通常与CUDA_VISIBLE_DEVICES相同）

## 输出文件

运行完成后，会在 `SpecVLA/exp` 目录下生成：

1. **数据文件**: `accept_length_data_<run_id>.npz`
   - `accept_lengths`: 每个task每个episode每个step的接受长度列表
   - `retrieved_slices`: 对应的检索到的action slice
   - `task_names`: 任务名称列表
   - `task_suite_name`: 任务套件名称
   - `use_db_retrieval`: 是否使用DB检索

2. **可视化图表**:
   - `accept_length_analysis_<run_id>.png`: 接受长度分析图表
   - `retrieved_slice_analysis_<run_id>.png`: 检索slice分析图表

## 注意事项

1. **确保检索服务器运行**: 在运行脚本之前，确保检索服务器（端口5002）和embedding服务器（端口9020）都在运行
2. **GPU设置**: 根据你的实际GPU情况修改 `CUDA_VISIBLE_DEVICES` 和 `MUJOCO_EGL_DEVICE_ID`
3. **路径设置**: 确保 `SPECVLA_ROOT` 路径正确
4. **conda环境**: 确保 `specvla` conda环境已安装所有依赖

## 故障排除

### 问题1: 连接检索服务器失败
- 检查检索服务器是否运行: `curl http://localhost:5002/health`
- 检查端口是否正确: 默认是5002

### 问题2: 找不到模块
- 检查 `PYTHONPATH` 是否正确设置
- 确保在 `SpecVLA` 根目录下运行

### 问题3: GPU相关错误
- 检查 `CUDA_VISIBLE_DEVICES` 和 `MUJOCO_EGL_DEVICE_ID` 是否设置正确
- 确保GPU可用: `nvidia-smi`

