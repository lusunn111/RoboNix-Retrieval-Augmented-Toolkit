#!/bin/bash
# 消融实验脚本 - Block-wise Speculative Decoding
# 
# 三个消融实验:
#   实验 1: 只有阈值分割 (阈值之下 SD，阈值之上 每步AR)
#   实验 2: + Verify-Skip (阈值之下 SD，阈值之上 2:1, 2步noverify + 1步AR)
#   实验 3: + Seq-Wise (阈值之下 SD，阈值之上 2:1, 2步noverify + 1步BlockSD) [主实验]
#
# 只运行 Goal 环境，每个任务 10 次
#
# 使用方法: 
#   bash run_libero_block_sd.sh           # 运行所有三个消融实验
#   bash run_libero_block_sd.sh 1         # 只运行消融实验 1
#   bash run_libero_block_sd.sh 2         # 只运行消融实验 2
#   bash run_libero_block_sd.sh 3         # 只运行消融实验 3
#
# 前置条件：
# 1. 确保 embedding 服务器正在运行 (http://127.0.0.1:9021/predict)
# 2. 确保 Qdrant 数据库正在运行 (localhost:6333)

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
TOP_K=5                      # 检索候选数量
PROB_THRESHOLD=0.1           # (已弃用)
USE_AVG_PROB="True"          # (已弃用)
ACCEPT_THRESHOLD=9           # (已弃用)

# Block 差值验证阈值 - 用于消融实验 3 (主实验)
BLOCK_SUM_THRESHOLD=45       # α: Block 内 token 差值之和的阈值
BLOCK_MAX_THRESHOLD=25       # μ: Block 内单个 token 差值的阈值

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
run_ablation_experiment() {
    local ABLATION_MODE=$1
    local ABLATION_NOTE=$2

    echo ""
    echo "=========================================="
    echo "运行消融实验 $ABLATION_MODE: $ABLATION_NOTE"
    echo "=========================================="
    echo "任务套件: $TASK_SUITE"
    echo "预训练检查点: $PRETRAINED_CHECKPOINT"
    echo "Spec检查点: $SPEC_CHECKPOINT"
    echo "----------------------------------------"
    echo "消融模式: $ABLATION_MODE"
    if [ "$ABLATION_MODE" -eq 1 ]; then
        echo "  策略: 阈值之下 SD，阈值之上 每步AR"
    elif [ "$ABLATION_MODE" -eq 2 ]; then
        echo "  策略: 阈值之下 SD，阈值之上 2:1 (2步noverify + 1步AR)"
    elif [ "$ABLATION_MODE" -eq 3 ]; then
        echo "  策略: 阈值之下 SD，阈值之上 2:1 (2步noverify + 1步BlockSD)"
        echo "  Block验证阈值: sum=${BLOCK_SUM_THRESHOLD}, max=${BLOCK_MAX_THRESHOLD}"
    fi
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
        --run_id_note "$ABLATION_NOTE" \
        --use_spec True \
        --parallel_draft False \
        --use_wandb False \
        --ablation_mode $ABLATION_MODE \
        --composite_threshold $COMPOSITE_THRESHOLD \
        --displacement_range_min $DISPLACEMENT_MIN \
        --displacement_range_max $DISPLACEMENT_MAX \
        --radius_range_min $RADIUS_MIN \
        --radius_range_max $RADIUS_MAX

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  消融实验 $ABLATION_MODE 失败! (exit code: $exit_code)"
        echo "=========================================="
        echo ""
        FAILED_TASKS+=("Ablation$ABLATION_MODE")
        return $exit_code
    fi

    echo ""
    echo "=========================================="
    echo "✓ 消融实验 $ABLATION_MODE 完成！"
    echo "=========================================="
    echo ""
}

# =============================================================================
# 运行消融实验
# =============================================================================
echo "=========================================="
echo "开始运行消融实验 (Goal 环境)"
echo "=========================================="
echo ""

# 检查是否指定了特定的消融实验
if [ -n "$1" ]; then
    # 只运行指定的消融实验
    case "$1" in
        1)
            echo "只运行消融实验 1: 只有阈值分割"
            run_ablation_experiment 1 "Ablation1_ThresholdOnly"
            ;;
        2)
            echo "只运行消融实验 2: + Verify-Skip"
            run_ablation_experiment 2 "Ablation2_VerifySkip"
            ;;
        3)
            echo "只运行消融实验 3: + Seq-Wise (主实验)"
            run_ablation_experiment 3 "Ablation3_SeqWise"
            ;;
        *)
            echo "错误: 未知的消融实验编号 '$1'"
            echo "用法: bash run_libero_block_sd.sh [1|2|3]"
            exit 1
            ;;
    esac
else
    # 运行消融实验 2 和 3
    echo "运行消融实验 2 和 3..."
    echo ""

    # [1/2] 消融实验 2: + Verify-Skip
    echo "============================================================"
    echo "[1/2] 消融实验 2: + Verify-Skip"
    echo "============================================================"
    run_ablation_experiment 2 "Ablation2_VerifySkip"

    # [2/2] 消融实验 3: + Seq-Wise (主实验)
    echo "============================================================"
    echo "[2/2] 消融实验 3: + Seq-Wise (主实验)"
    echo "============================================================"
    run_ablation_experiment 3 "Ablation3_SeqWise"
fi

echo ""
echo "=========================================="
echo "消融实验运行结束！"
echo "=========================================="

# 汇总结果
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✓ 全部消融实验成功完成！"
else
    echo "⚠️  有 ${#FAILED_TASKS[@]} 个消融实验失败:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "   - $task"
    done
fi

echo "=========================================="
echo "检查结果目录: ./specdecoding/test-speed/libero_block_sd/"
echo "=========================================="
