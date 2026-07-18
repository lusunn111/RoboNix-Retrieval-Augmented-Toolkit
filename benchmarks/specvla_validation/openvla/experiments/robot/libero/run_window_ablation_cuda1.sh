#!/bin/bash
# Window Size 消融实验 - CUDA 1
# 
# Spatial: w=11, 13, 15, 17, 19
# Long (libero_10): w=11, 13, 15
#
# 使用方法: bash run_window_ablation_cuda1.sh

FAILED_TASKS=()

SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 检查服务
echo "检查服务..."
curl -s "http://localhost:6333/collections" > /dev/null 2>&1 || { echo "Qdrant未运行"; exit 1; }
curl -s "http://127.0.0.1:9021" > /dev/null 2>&1 || echo "警告: Mix Embedding服务可能未运行"

# 激活环境
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla
fi

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
echo "CUDA_VISIBLE_DEVICES=1"

# 通用参数
MODEL_FAMILY="openvla"
CENTER_CROP="True"
NUM_TRIALS=10
TOP_K=5
PROB_THRESHOLD=0.1
USE_AVG_PROB="True"
ACCEPT_THRESHOLD=9
BLOCK_SUM_THRESHOLD=45
BLOCK_MAX_THRESHOLD=25
ABLATION_MODE=3

# 运行函数
run_experiment() {
    local TASK_SUITE=$1
    local PRETRAINED_CHECKPOINT=$2
    local SPEC_CHECKPOINT=$3
    local DISPLACEMENT_MIN=$4
    local DISPLACEMENT_MAX=$5
    local RADIUS_MIN=$6
    local RADIUS_MAX=$7
    local COMPOSITE_THRESHOLD=$8
    local WINDOW_SIZE=$9

    echo ""
    echo "=========================================="
    echo "运行: $TASK_SUITE - Window $WINDOW_SIZE"
    echo "=========================================="

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
        --run_id_note "Window${WINDOW_SIZE}_Main" \
        --use_spec True \
        --parallel_draft False \
        --use_wandb False \
        --ablation_mode $ABLATION_MODE \
        --window_size $WINDOW_SIZE \
        --composite_threshold $COMPOSITE_THRESHOLD \
        --displacement_range_min $DISPLACEMENT_MIN \
        --displacement_range_max $DISPLACEMENT_MAX \
        --radius_range_min $RADIUS_MIN \
        --radius_range_max $RADIUS_MAX

    if [ $? -ne 0 ]; then
        FAILED_TASKS+=("${TASK_SUITE}_W${WINDOW_SIZE}")
    else
        echo "✓ $TASK_SUITE - Window $WINDOW_SIZE 完成"
    fi
}

echo "=========================================="
echo "开始 CUDA 1 实验"
echo "=========================================="

# ============================================
# Spatial: w=11, 13, 15, 17, 19
# ============================================
echo ""
echo "============ Spatial 环境 ============"
for W in 11 13 15 17 19; do
    run_experiment \
        "libero_spatial" \
        "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-spatial" \
        "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_spatial_debug_ckpt/state_190" \
        0.000027 0.128629 0.000019 0.015654 0.217119 \
        $W
done

# ============================================
# Long (libero_10): w=11, 13, 15
# ============================================
echo ""
echo "============ Long (libero_10) 环境 ============"
for W in 11 13 15; do
    run_experiment \
        "libero_10" \
        "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-10" \
        "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_10_debug_ckpt/state_190" \
        0.000008 0.102298 0.000001 0.012479 0.4 \
        $W
done

echo ""
echo "=========================================="
echo "CUDA 1 实验完成！"
echo "=========================================="

if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✓ 全部成功"
else
    echo "⚠️ 失败: ${FAILED_TASKS[*]}"
fi
