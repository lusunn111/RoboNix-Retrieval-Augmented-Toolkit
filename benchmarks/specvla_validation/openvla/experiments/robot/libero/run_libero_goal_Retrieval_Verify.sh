#!/bin/bash
# 运行检索验证实验
# 每次get_action时检索action并验证接受长度
#
# 使用方法: bash run_libero_goal_Retrieval_Verify.sh
#
# 前置条件：
# 1. 确保 DB retrieval 服务器正在运行 (http://127.0.0.1:5002/pipeline)
# 2. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 3. 确保 Qdrant 数据库正在运行 (localhost:6333)
# 4. 数据库已恢复base备份

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
MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT="/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
TASK_SUITE="libero_goal"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9
DB_MODEL_RATIO="1:0"  # N:M - N steps use DB, M steps use Model AR
NUM_TRIALS=1
RUN_ID_NOTE="Retrieval_Verify"

# 日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/retrieval_verify_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "实验配置"
echo "=========================================="
echo "模型: $MODEL_FAMILY"
echo "预训练检查点: $PRETRAINED_CHECKPOINT"
echo "Spec检查点: $SPEC_CHECKPOINT"
echo "任务套件: $TASK_SUITE"
echo "Center crop: $CENTER_CROP"
echo "Accept threshold: $ACCEPT_THRESHOLD"
echo "DB:Model Ratio: $DB_MODEL_RATIO"
echo "每任务试验次数: $NUM_TRIALS"
echo "运行标记: $RUN_ID_NOTE"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "开始运行检索验证实验..."
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

# 运行Python脚本
python openvla/experiments/robot/libero/run_libero_goal_Retrieval_Verify.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --db_model_ratio "$DB_MODEL_RATIO" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "检查日志目录: $LOG_DIR"
echo "检查结果目录: ./specdecoding/test-speed/libero_goal_Retrieval_Verify/"
echo "=========================================="
