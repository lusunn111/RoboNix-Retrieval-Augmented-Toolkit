#!/bin/bash
# 运行轨迹记录实验 - 记录每个轨迹的成功/失败状态
#
# 使用方法: bash run_record_guiji_suc_fail.sh
#
# 功能：
# 1. 恢复数据库到base状态
# 2. 运行实验进行对比：
#    - run_libero_naive_retrieval_Exp_online_Memory.py (纯DB检索) [默认跳过，取消注释可开启]
#    - run_libero_goal_Retrieval_Verify.py (模型+检索验证) [默认运行]
# 3. 每个任务运行5次，保存observations.npy（包含轨迹成功/失败状态）
#
# ==================== 如何开启第一个实验 ====================
# 找到下方 "实验1: 纯DB检索" 部分，取消以下代码块的注释即可：
#   python openvla/experiments/robot/libero/run_libero_naive_retrieval_Exp_online_Memory.py \
#       --task_suite_name "$TASK_SUITE" \
#       ...
# ===========================================================
#
# 输出文件格式：
#   observations_data[task_id][episode_idx] = {
#       'observations': list,      # 每步的观测数据
#       'success': bool,           # 轨迹是否成功
#       'task_description': str,   # 任务描述
#       'num_steps': int           # 步数
#   }
#
# 前置条件：
# 1. 确保 DB retrieval 服务器正在运行 (http://127.0.0.1:5002/pipeline)
# 2. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 3. 确保 Qdrant 数据库正在运行 (localhost:6333)

set -e  # 遇到错误立即退出

# =============================================================================
# 数据库恢复设置（使用base备份）
# =============================================================================
RTCACHE_ROOT="/path/to/rtcache"
BACKUP_BASE_DIR="$RTCACHE_ROOT/scripts/retrieval/qdrant_backups/backup_base"
BACKUP_LATEST_LINK="$RTCACHE_ROOT/scripts/retrieval/qdrant_backups/latest"
RESTORE_SCRIPT="$RTCACHE_ROOT/scripts/retrieval/restore_qdrant.py"
QDRANT_HOST="localhost"
QDRANT_PORT=6333

# 设置工作目录
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 检查目录是否存在
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "错误: SpecVLA 根目录不存在: $SPECVLA_ROOT"
    exit 1
fi

# =============================================================================
# 恢复数据库到base状态
# =============================================================================
echo "=========================================="
echo "正在恢复Qdrant数据库到base状态..."
echo "=========================================="

# 检查base备份是否存在
if [ ! -d "$BACKUP_BASE_DIR" ]; then
    echo "错误: base备份目录不存在: $BACKUP_BASE_DIR"
    exit 1
fi

# 更新latest软链接指向backup_base
echo "更新latest软链接指向backup_base..."
rm -f "$BACKUP_LATEST_LINK"
ln -s "$BACKUP_BASE_DIR" "$BACKUP_LATEST_LINK"
echo "latest -> $(readlink -f $BACKUP_LATEST_LINK)"

# 检查恢复脚本是否存在
if [ ! -f "$RESTORE_SCRIPT" ]; then
    echo "错误: 数据库恢复脚本不存在: $RESTORE_SCRIPT"
    exit 1
fi

# 激活conda环境（用于运行restore脚本）
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate rt-mzh || {
        echo "错误: 无法激活conda环境 'rt-mzh'"
        exit 1
    }
    echo "Conda环境已激活: rt-mzh"
else
    echo "警告: 未找到conda命令，跳过环境激活"
fi

# 运行恢复脚本
echo "执行数据库恢复..."
python3 "$RESTORE_SCRIPT" \
    --backup-dir "$BACKUP_LATEST_LINK" \
    --qdrant-host "$QDRANT_HOST" \
    --qdrant-port "$QDRANT_PORT" \
    --force || {
    echo "错误: 数据库恢复失败"
    exit 1
}

echo "=========================================="
echo "数据库恢复完成！"
echo "=========================================="
echo ""

# =============================================================================
# 检查服务可用性
# =============================================================================
echo "=========================================="
echo "检查服务可用性..."
echo "=========================================="

# 检查Qdrant
echo -n "检查Qdrant服务 (${QDRANT_HOST}:${QDRANT_PORT})... "
if curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "错误: Qdrant服务未运行！请先启动Qdrant服务。"
    exit 1
fi

# 检查Retrieval API
echo -n "检查Retrieval API (http://127.0.0.1:5002)... "
if curl -s "http://127.0.0.1:5002" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "错误: Retrieval API未运行！请先启动检索服务。"
    echo "提示: 使用 $RTCACHE_ROOT/scripts/retrieval/start_libero_goal_retrieval.sh --skip-restore"
    exit 1
fi

# 检查Embedding服务
echo -n "检查Embedding服务 (http://127.0.0.1:9020)... "
if curl -s "http://127.0.0.1:9020" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "警告: Embedding服务可能未运行（端口9020），但将继续执行..."
fi

echo "=========================================="
echo ""

# =============================================================================
# 实验参数设置
# =============================================================================
TASK_SUITE="libero_goal"
NUM_TRIALS=5                          # 每个任务运行5次
DB_STATE_NAME="base"                  # 数据库状态名称
RUN_ID_NOTE="record_guiji_suc_fail"   # 运行标记

# 模型参数（用于Retrieval Verify实验）
MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT="/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"

# 日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/record_guiji_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "实验配置"
echo "=========================================="
echo "任务套件: $TASK_SUITE"
echo "每任务试验次数: $NUM_TRIALS"
echo "数据库状态: $DB_STATE_NAME"
echo "运行标记: $RUN_ID_NOTE"
echo "日志目录: $LOG_DIR"
echo ""
echo "将运行以下实验："
echo "  1. 纯DB检索 (run_libero_naive_retrieval_Exp_online_Memory.py) [已跳过]"
echo "  2. 纯SD模型 (run_libero_goal_Retrieval_Verify.py, 不使用DB检索)"
echo "=========================================="
echo ""

# =============================================================================
# 实验1: 纯DB检索 [默认跳过，取消注释可开启]
# =============================================================================
# 如需运行第一个实验，取消以下代码块的注释：
# echo "=========================================="
# echo "[实验1/2] 开始运行纯DB检索实验..."
# echo "=========================================="
# echo ""
# 
# python openvla/experiments/robot/libero/run_libero_naive_retrieval_Exp_online_Memory.py \
#     --task_suite_name "$TASK_SUITE" \
#     --num_trials_per_task "$NUM_TRIALS" \
#     --db_state_name "$DB_STATE_NAME" \
#     --run_id_note "${RUN_ID_NOTE}_pure_db" \
#     2>&1 | tee "${LOG_DIR}/exp1_pure_db_${TIMESTAMP}.log"
# 
# echo ""
# echo "=========================================="
# echo "[实验1/2] 纯DB检索实验完成！"
# echo "=========================================="
# echo ""

# =============================================================================
# 实验2: 纯SD模型（不使用DB检索）
# =============================================================================
echo "=========================================="
echo "[实验] 开始运行纯SD模型实验（无DB检索）..."
echo "=========================================="
echo ""

# 激活SpecVLA环境
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla || {
        echo "错误: 无法激活conda环境 'specvla'"
        exit 1
    }
    echo "Conda环境已激活: specvla"
else
    echo "警告: 未找到conda命令"
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo ""

# 运行纯SD模型实验（DB:Model = 0:1，纯模型生成）
python openvla/experiments/robot/libero/run_libero_goal_Retrieval_Verify.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop True \
    --accept_threshold 9 \
    --db_model_ratio "0 1" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "${RUN_ID_NOTE}_pure_sd" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "日志目录: ${LOG_DIR}"
echo ""
echo "输出文件目录:"
echo "  实验1 (纯DB): ./openvla/specdecoding/test-speed/libero_naive_retrieval_Exp_online_Memory/"
echo "  实验2 (纯SD): ./openvla/specdecoding/test-speed/libero_goal_Retrieval_Verify/"
echo ""
echo ".npy 数据结构:"
echo "  observations_data[task_id][episode_idx] = {"
echo "      'observations': list,      # 每步的观测数据"
echo "      'success': bool,           # 轨迹是否成功"
echo "      'task_description': str,   # 任务描述"
echo "      'num_steps': int           # 步数"
echo "  }"
echo "=========================================="
