#!/bin/bash
# Block-wise Speculative Decoding - Spatial 阈值实验 (GPU 1)
# 
# 测试阈值: 0.30, 0.35, 0.40
#
# 使用方法: bash run_libero_block_sd_spatial_cuda1.sh
#
# 前置条件：
# 1. 确保 embedding 服务器正在运行 (http://127.0.0.1:9021/predict)
# 2. 确保 Qdrant 数据库正在运行 (localhost:6333)

# 注意: 不使用 set -e，允许单个实验失败后继续执行后续实验
# 失败的实验会被记录，最后汇总显示

# 记录失败的实验
FAILED_EXPERIMENTS=()

# =============================================================================
# 设置工作目录
# =============================================================================
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

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

# 检查Mix Embedding服务 (端口9021)
echo -n "检查Mix Embedding服务 (http://127.0.0.1:9021)... "
if curl -s "http://127.0.0.1:9021" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "警告: Mix Embedding服务可能未运行（端口9021），但将继续执行..."
fi

echo "=========================================="
echo ""

# =============================================================================
# 通用实验参数
# =============================================================================
MODEL_FAMILY="openvla"
CENTER_CROP="True"
NUM_TRIALS=10

# Block SD 参数
TOP_K=5
PROB_THRESHOLD=0.1
USE_AVG_PROB="True"
ACCEPT_THRESHOLD=9

# Block 差值验证阈值 - 纯 AR 模式
BLOCK_SUM_THRESHOLD=-1
BLOCK_MAX_THRESHOLD=-1

# Spatial 固定参数
TASK_SUITE="libero_spatial"
PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-spatial"
SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_spatial_debug_ckpt/state_190"
DISPLACEMENT_MIN=0.000027
DISPLACEMENT_MAX=0.128629
RADIUS_MIN=0.000019
RADIUS_MAX=0.015654

# =============================================================================
# 激活环境和设置GPU
# =============================================================================
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

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo ""

# =============================================================================
# 定义运行函数
# =============================================================================
run_experiment() {
    local COMPOSITE_THRESHOLD=$1
    local RUN_ID_NOTE=$2

    echo ""
    echo "=========================================="
    echo "运行 Spatial 实验: 阈值=${COMPOSITE_THRESHOLD}"
    echo "=========================================="
    echo "预训练检查点: $PRETRAINED_CHECKPOINT"
    echo "Spec检查点: $SPEC_CHECKPOINT"
    echo "----------------------------------------"
    echo "综合指标参数:"
    echo "  位移范围: [$DISPLACEMENT_MIN, $DISPLACEMENT_MAX]"
    echo "  曲率范围: [$RADIUS_MIN, $RADIUS_MAX]"
    echo "  综合阈值: $COMPOSITE_THRESHOLD"
    echo "----------------------------------------"
    echo "每任务试验次数: $NUM_TRIALS"
    echo "=========================================="
    echo ""

    python openvla/experiments/robot/libero/run_libero_block_sd.py \
        --model_family $MODEL_FAMILY \
        --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
        --spec_checkpoint $SPEC_CHECKPOINT \
        --task_suite_name $TASK_SUITE \
        --center_crop $CENTER_CROP \
        --top_k $TOP_K \
        --prob_threshold $PROB_THRESHOLD \
        --use_avg_prob $USE_AVG_PROB \
        --accept_threshold $ACCEPT_THRESHOLD \
        --block_sum_threshold $BLOCK_SUM_THRESHOLD \
        --block_max_threshold $BLOCK_MAX_THRESHOLD \
        --num_trials_per_task $NUM_TRIALS \
        --run_id_note "$RUN_ID_NOTE" \
        --use_spec True \
        --parallel_draft False \
        --use_wandb False \
        --composite_threshold $COMPOSITE_THRESHOLD \
        --displacement_range_min $DISPLACEMENT_MIN \
        --displacement_range_max $DISPLACEMENT_MAX \
        --radius_range_min $RADIUS_MIN \
        --radius_range_max $RADIUS_MAX

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  阈值=${COMPOSITE_THRESHOLD} 实验失败! (exit code: $exit_code)"
        echo "=========================================="
        echo ""
        FAILED_EXPERIMENTS+=("thresh_${COMPOSITE_THRESHOLD}")
        return $exit_code
    fi

    echo ""
    echo "=========================================="
    echo "✓ 阈值=${COMPOSITE_THRESHOLD} 实验完成！"
    echo "=========================================="
    echo ""
}

# =============================================================================
# 运行实验 (阈值: 0.30, 0.35, 0.40)
# =============================================================================
echo "=========================================="
echo "开始运行 Spatial 阈值实验 (GPU 1)"
echo "=========================================="
echo "测试阈值: 0.30, 0.35, 0.40"
echo "=========================================="
echo ""

# [1/3] 阈值 0.30
echo "============================================================"
echo "[1/3] 阈值 = 0.30"
echo "============================================================"
run_experiment 0.30 "Spatial_thresh_0.30"

# [2/3] 阈值 0.35
echo "============================================================"
echo "[2/3] 阈值 = 0.35"
echo "============================================================"
run_experiment 0.35 "Spatial_thresh_0.35"

# [3/3] 阈值 0.40
echo "============================================================"
echo "[3/3] 阈值 = 0.40"
echo "============================================================"
run_experiment 0.40 "Spatial_thresh_0.40"

echo ""
echo "=========================================="
echo "GPU 1 所有实验运行结束！"
echo "=========================================="

# 汇总结果
if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo "✓ 全部 3 个实验成功完成！"
else
    echo "⚠️  有 ${#FAILED_EXPERIMENTS[@]} 个实验失败:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "   - $exp"
    done
fi

echo "=========================================="
echo "检查结果目录: ./openvla/specdecoding/test-speed/libero_block_sd/"
echo "=========================================="
