#!/bin/bash
# 运行基于综合指标的SD + Retrieval实验（Mix视角）- LIBERO-Long (10任务)
# 
# 决策逻辑：
# - 综合指标 > threshold：使用Retrieval策略
#   - Retrieval策略：2次verify(DB) + 1次noverify(AR)
#   - verify(DB)：直接使用检索的action
#   - noverify(AR)：使用AR生成（不验证）
# - 综合指标 <= threshold：使用SD (Speculative Decoding)
#
# 归一化参数（minmax归一化）- LIBERO-Long：
# - 位移指标：[0.000098, 0.116458]
# - 曲率指标：[0.000010, 0.014151]
# - alpha = 0.5（1:1构造综合指标）
#
# 测试阈值：0.4, 0.3, 0.25, 0.20
#
# 使用方法: 
#   bash run_libero_retrieval_verify_sd_retrieval_mix_long.sh [threshold]
#   例如: bash run_libero_retrieval_verify_sd_retrieval_mix_long.sh 0.25
#   不带参数则运行所有4个阈值
#
# 前置条件：
# 1. 确保 Mix视角检索服务器正在运行 (http://127.0.0.1:5003/pipeline)
# 2. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 3. 确保 Qdrant 数据库正在运行 (localhost:6333)

set -e  # 遇到错误立即退出

# =============================================================================
# 设置工作目录
# =============================================================================
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 检查目录是否存在
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "错误: SpecVLA 根目录不存在: $SPECVLA_ROOT"
    exit 1
fi

# =============================================================================
# 检查服务可用性
# =============================================================================
echo "=========================================="
echo "检查服务可用性..."
echo "=========================================="

# 检查Qdrant
echo -n "检查Qdrant服务 (localhost:6333)... "
if curl -s "http://localhost:6333/collections" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "错误: Qdrant服务未运行！请先启动Qdrant服务。"
    exit 1
fi

# 检查Mix视角Retrieval API (端口5003)
echo -n "检查Mix视角Retrieval API (http://127.0.0.1:5003)... "
if curl -s "http://127.0.0.1:5003" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "错误: Mix视角Retrieval API未运行！请先启动检索服务（端口5003）。"
    exit 1
fi

# 检查Embedding服务 (尝试9020和9021端口)
echo -n "检查Embedding服务... "
if curl -s "http://127.0.0.1:9020" > /dev/null 2>&1; then
    echo "✓ (端口9020)"
elif curl -s "http://127.0.0.1:9021" > /dev/null 2>&1; then
    echo "✓ (端口9021)"
else
    echo "✗"
    echo "警告: Embedding服务可能未运行（端口9020/9021），但将继续执行..."
fi

echo "=========================================="
echo ""

# =============================================================================
# 实验参数设置 - LIBERO-Long (10任务)
# =============================================================================
MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT="/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-10"
SPEC_CHECKPOINT="/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_10_debug_ckpt/state_190"
TASK_SUITE="libero_10"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9

# 综合指标参数（LIBERO-Long 归一化范围）
WINDOW_SIZE=5
ALPHA=0.5  # 1:1权重
DISPLACEMENT_RANGE_MIN=0.000098
DISPLACEMENT_RANGE_MAX=0.116458
RADIUS_RANGE_MIN=0.000010
RADIUS_RANGE_MAX=0.014151

NUM_TRIALS=10

# 定义要测试的阈值列表
THRESHOLDS=(0.4 0.3 0.25 0.20)

# 如果提供了参数，只运行指定的阈值
if [ $# -eq 1 ]; then
    THRESHOLDS=($1)
    echo "只运行指定阈值: $1"
fi

# =============================================================================
# 运行函数
# =============================================================================
run_experiment() {
    local COMPOSITE_THRESHOLD=$1
    local RUN_ID_NOTE="SD_Retrieval_Mix_Long_thresh${COMPOSITE_THRESHOLD}"
    
    # 日志目录
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_DIR="./experiments/logs/sd_retrieval_mix_long_thresh${COMPOSITE_THRESHOLD}_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "######################################################################"
    echo "# 运行阈值: $COMPOSITE_THRESHOLD"
    echo "######################################################################"
    echo ""
    echo "=========================================="
    echo "实验配置 - LIBERO-Long"
    echo "=========================================="
    echo "模型: $MODEL_FAMILY"
    echo "预训练检查点: $PRETRAINED_CHECKPOINT"
    echo "Spec检查点: $SPEC_CHECKPOINT"
    echo "任务套件: $TASK_SUITE"
    echo "Center crop: $CENTER_CROP"
    echo "Accept threshold: $ACCEPT_THRESHOLD"
    echo "----------------------------------------"
    echo "综合指标参数 (LIBERO-Long):"
    echo "  窗口大小: $WINDOW_SIZE"
    echo "  综合指标阈值: $COMPOSITE_THRESHOLD"
    echo "  Alpha (曲率指标权重): $ALPHA (1:1权重)"
    echo "  位移归一化范围: [$DISPLACEMENT_RANGE_MIN, $DISPLACEMENT_RANGE_MAX]"
    echo "  曲率归一化范围: [$RADIUS_RANGE_MIN, $RADIUS_RANGE_MAX]"
    echo "----------------------------------------"
    echo "决策逻辑:"
    echo "  综合指标 > $COMPOSITE_THRESHOLD: 使用Retrieval策略"
    echo "    - Retrieval策略: 2次verify(DB) + 1次noverify(AR) = 2:1"
    echo "  综合指标 <= $COMPOSITE_THRESHOLD: 使用SD (Speculative Decoding)"
    echo "  历史不足时: 使用AR模式（避免SD状态问题）"
    echo "----------------------------------------"
    echo "Mix视角检索服务: http://127.0.0.1:5003/pipeline"
    echo "每任务试验次数: $NUM_TRIALS"
    echo "运行标记: $RUN_ID_NOTE"
    echo "日志目录: $LOG_DIR"
    echo "=========================================="
    echo ""

    # 运行Python脚本
    python openvla/experiments/robot/libero/run_libero_retrieval_verify_sd_retrieval_mix.py \
        --model_family "$MODEL_FAMILY" \
        --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
        --spec_checkpoint "$SPEC_CHECKPOINT" \
        --task_suite_name "$TASK_SUITE" \
        --center_crop "$CENTER_CROP" \
        --accept_threshold "$ACCEPT_THRESHOLD" \
        --window_size "$WINDOW_SIZE" \
        --composite_threshold "$COMPOSITE_THRESHOLD" \
        --alpha "$ALPHA" \
        --displacement_range_min "$DISPLACEMENT_RANGE_MIN" \
        --displacement_range_max "$DISPLACEMENT_RANGE_MAX" \
        --radius_range_min "$RADIUS_RANGE_MIN" \
        --radius_range_max "$RADIUS_RANGE_MAX" \
        --num_trials_per_task "$NUM_TRIALS" \
        --run_id_note "$RUN_ID_NOTE" \
        --use_spec True \
        --parallel_draft False \
        --use_wandb False

    echo ""
    echo "=========================================="
    echo "阈值 $COMPOSITE_THRESHOLD 实验完成！"
    echo "=========================================="
}

# =============================================================================
# 主程序
# =============================================================================
echo "=========================================="
echo "开始运行SD + Retrieval实验（Mix视角）- LIBERO-Long"
echo "=========================================="
echo "将测试以下阈值: ${THRESHOLDS[*]}"
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

# 循环运行所有阈值
for THRESH in "${THRESHOLDS[@]}"; do
    run_experiment "$THRESH"
done

echo ""
echo "######################################################################"
echo "# 所有实验完成！"
echo "######################################################################"
echo ""
echo "测试的阈值: ${THRESHOLDS[*]}"
echo "检查日志目录: ./experiments/logs/"
echo "检查结果目录: ./specdecoding/test-speed/libero_sd_retrieval_mix/"
echo "######################################################################"
