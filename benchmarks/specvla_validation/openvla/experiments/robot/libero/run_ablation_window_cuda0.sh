#!/bin/bash
# 主实验 - 滑动窗口大小消融 (w=6,7)
# 
# 在 CUDA 0 上运行，Goal 环境，每个任务 10 次
#
# 使用方法: bash run_ablation_window_cuda0.sh

# 记录失败的任务
FAILED_TASKS=()

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
BLOCK_SUM_THRESHOLD=45
BLOCK_MAX_THRESHOLD=25

# 消融模式 3: 主实验 (阈值分割 + Verify-Skip + Seq-Wise)
ABLATION_MODE=3

# =============================================================================
# 激活环境和设置GPU - CUDA 0
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

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo ""

# =============================================================================
# Goal 环境参数
# =============================================================================
TASK_SUITE="libero_goal"
PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
DISPLACEMENT_MIN=0.000009
DISPLACEMENT_MAX=0.139051
RADIUS_MIN=0.000001
RADIUS_MAX=0.016873
COMPOSITE_THRESHOLD=0.143210

# =============================================================================
# 定义运行函数
# =============================================================================
run_window_experiment() {
    local WINDOW_SIZE=$1

    echo ""
    echo "=========================================="
    echo "运行主实验 - Window Size = $WINDOW_SIZE"
    echo "=========================================="
    echo "任务套件: $TASK_SUITE"
    echo "消融模式: $ABLATION_MODE (主实验)"
    echo "滑动窗口大小: $WINDOW_SIZE"
    echo "----------------------------------------"
    echo "综合指标参数:"
    echo "  位移范围: [$DISPLACEMENT_MIN, $DISPLACEMENT_MAX]"
    echo "  曲率范围: [$RADIUS_MIN, $RADIUS_MAX]"
    echo "  综合阈值: $COMPOSITE_THRESHOLD"
    echo "----------------------------------------"
    echo "Block SD 参数:"
    echo "  Block验证阈值: sum=${BLOCK_SUM_THRESHOLD}, max=${BLOCK_MAX_THRESHOLD}"
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

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  Window Size = $WINDOW_SIZE 实验失败! (exit code: $exit_code)"
        echo "=========================================="
        echo ""
        FAILED_TASKS+=("Window$WINDOW_SIZE")
        return $exit_code
    fi

    echo ""
    echo "=========================================="
    echo "✓ Window Size = $WINDOW_SIZE 实验完成！"
    echo "=========================================="
    echo ""
}

# =============================================================================
# 运行实验 - w=6,7 (CUDA 0)
# =============================================================================
echo "=========================================="
echo "开始运行主实验 - 滑动窗口消融 (w=6,7)"
echo "GPU: CUDA 0"
echo "=========================================="
echo ""

# [1/2] Window Size = 6
echo "============================================================"
echo "[1/2] Window Size = 6"
echo "============================================================"
run_window_experiment 6

# [2/2] Window Size = 7
echo "============================================================"
echo "[2/2] Window Size = 7"
echo "============================================================"
run_window_experiment 7

echo ""
echo "=========================================="
echo "滑动窗口消融实验 (w=6,7) 运行结束！"
echo "=========================================="

# 汇总结果
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✓ 全部实验成功完成！"
else
    echo "⚠️  有 ${#FAILED_TASKS[@]} 个实验失败:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "   - $task"
    done
fi

echo "=========================================="
echo "检查结果目录: ./specdecoding/test-speed/libero_block_sd/"
echo "=========================================="
