#!/bin/bash
# 运行基于曲率半径的检索/AR切换实验
#
# 使用方法:
#   bash run_libero_goal_new_stage.sh
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
NUM_TRIALS=10

# 曲率半径参数
WINDOW_SIZE=5
# 要测试的曲率阈值列表
CURVATURE_THRESHOLDS=(0.01 0.06)

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

# =============================================================================
# 循环运行多个曲率阈值的实验
# =============================================================================
for CURVATURE_THRESHOLD in "${CURVATURE_THRESHOLDS[@]}"; do
    # 将阈值中的小数点替换为下划线用于文件名
    THRESHOLD_STR=$(echo "$CURVATURE_THRESHOLD" | sed 's/\./_/')
    RUN_ID_NOTE="new_stage_curvature_${THRESHOLD_STR}"
    
    # 日志目录
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_DIR="./experiments/logs/new_stage_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "================================================================================"
    echo "开始运行实验 - 曲率阈值: ${CURVATURE_THRESHOLD}m"
    echo "================================================================================"
    echo "实验配置:"
    echo "  模型: $MODEL_FAMILY"
    echo "  预训练检查点: $PRETRAINED_CHECKPOINT"
    echo "  Spec检查点: $SPEC_CHECKPOINT"
    echo "  任务套件: $TASK_SUITE"
    echo "  Center crop: $CENTER_CROP"
    echo "  Accept threshold: $ACCEPT_THRESHOLD"
    echo "  曲率窗口: $WINDOW_SIZE"
    echo "  曲率阈值: $CURVATURE_THRESHOLD"
    echo "  每任务试验次数: $NUM_TRIALS"
    echo "  运行标记: $RUN_ID_NOTE"
    echo "  日志目录: $LOG_DIR"
    echo "  CUDA设备: $CUDA_VISIBLE_DEVICES"
    echo "================================================================================"
    echo ""

    # 运行Python脚本
    python openvla/experiments/robot/libero/run_libero_goal_new_stage.py \
        --model_family "$MODEL_FAMILY" \
        --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
        --spec_checkpoint "$SPEC_CHECKPOINT" \
        --task_suite_name "$TASK_SUITE" \
        --center_crop "$CENTER_CROP" \
        --accept_threshold "$ACCEPT_THRESHOLD" \
        --curvature_window_size "$WINDOW_SIZE" \
        --curvature_threshold "$CURVATURE_THRESHOLD" \
        --num_trials_per_task "$NUM_TRIALS" \
        --run_id_note "$RUN_ID_NOTE" \
        --use_spec True \
        --parallel_draft False \
        --use_wandb False

    echo ""
    echo "================================================================================"
    echo "实验完成！曲率阈值: ${CURVATURE_THRESHOLD}m"
    echo "================================================================================"
    echo "检查日志目录: $LOG_DIR"
    echo "检查结果目录: ./specdecoding/test-speed/libero_goal_new_stage/"
    echo "================================================================================"
    echo ""
    
    # 在实验之间等待5秒
    if [ "$CURVATURE_THRESHOLD" != "${CURVATURE_THRESHOLDS[-1]}" ]; then
        echo "等待5秒后开始下一个实验..."
        sleep 5
    fi
done

echo ""
echo "================================================================================"
echo "所有实验完成！"
echo "================================================================================"
echo "测试的曲率阈值: ${CURVATURE_THRESHOLDS[@]}"
echo "结果目录: ./specdecoding/test-speed/libero_goal_new_stage/"
echo "================================================================================"
