#!/bin/bash
# 运行带有详细计时功能的 SpecVLA 测试脚本
# 用法: bash run_timing_test.sh

set -e  # 遇到错误立即退出

# =============================================================================
# 环境配置
# =============================================================================

# 设置工作目录
SPECVLA_ROOT="/path/to/SpecVLA"
cd "$SPECVLA_ROOT"

echo "=========================================="
echo "运行 SpecVLA 详细计时测试"
echo "=========================================="

# 检查目录是否存在
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "错误: SpecVLA 根目录不存在: $SPECVLA_ROOT"
    exit 1
fi

# 激活 conda 环境（如果需要）
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    # 根据你的环境名称修改（可能是 specvla 或其他）
    conda activate specvla 2>/dev/null || conda activate base
    echo "Conda 环境已激活: $(conda info --envs | grep '*' | awk '{print $1}')"
else
    echo "警告: 未找到 conda 命令"
fi

# 设置 Python 路径（重要！）
export PYTHONPATH="${SPECVLA_ROOT}:${SPECVLA_ROOT}/openvla:${PYTHONPATH}"
export PYTHONPATH="${SPECVLA_ROOT}/LIBERO:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# 设置 CUDA 和 MUJOCO 设备（根据你的 GPU 情况修改）
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MUJOCO_EGL_DEVICE_ID: $MUJOCO_EGL_DEVICE_ID"

echo ""
echo "=========================================="
echo "环境配置完成"
echo "=========================================="
echo ""

# =============================================================================
# 模型路径配置
# =============================================================================

echo "=========================================="
echo "配置模型路径"
echo "=========================================="

# 根据你的实际路径修改这些变量
BACKBONE_MODEL="${SPECVLA_ROOT}/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="${SPECVLA_ROOT}/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"

# 检查模型路径是否存在
if [ ! -d "$BACKBONE_MODEL" ]; then
    echo "警告: Backbone 模型路径不存在: $BACKBONE_MODEL"
    echo "请修改脚本中的 BACKBONE_MODEL 变量"
    exit 1
fi

if [ ! -d "$SPEC_CHECKPOINT" ]; then
    echo "警告: Spec checkpoint 路径不存在: $SPEC_CHECKPOINT"
    echo "请修改脚本中的 SPEC_CHECKPOINT 变量"
    exit 1
fi

echo "Backbone 模型: $BACKBONE_MODEL"
echo "Spec 模型: $SPEC_CHECKPOINT"
echo ""

# =============================================================================
# 实验参数配置
# =============================================================================

echo "=========================================="
echo "实验参数配置"
echo "=========================================="

# 任务参数
TASK_SUITE="libero_goal"          # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
NUM_TRIALS=2                       # 每个任务的测试次数 (快速测试用2，完整测试用10)
CENTER_CROP="True"                 # 训练时有数据增强则设为 True
ACCEPT_THRESHOLD=9                 # Speculative decoding 接受阈值

# 运行标记（用于区分不同实验）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID_NOTE="Detailed_Timing_${TIMESTAMP}"

echo "任务套件: $TASK_SUITE"
echo "测试次数/任务: $NUM_TRIALS"
echo "Center crop: $CENTER_CROP"
echo "Accept threshold: $ACCEPT_THRESHOLD"
echo "运行标记: $RUN_ID_NOTE"
echo ""

# =============================================================================
# 运行实验
# =============================================================================

echo "=========================================="
echo "开始运行实验"
echo "=========================================="
echo ""

# 运行脚本
python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint "$BACKBONE_MODEL" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop "$CENTER_CROP" \
    --use_spec True \
    --parallel_draft False \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "结果文件位置："
echo "  - 日志文件: TGT_DIR/EVAL-${TASK_SUITE}-*.txt"
echo "  - 详细计时: TGT_DIR/EVAL-${TASK_SUITE}-*_detailed_timing.json"
echo "  - 原始数据: TGT_DIR/EVAL-${TASK_SUITE}-*${TASK_SUITE}.json"
echo ""
echo "查看详细计时统计："
echo "  tail -100 TGT_DIR/EVAL-${TASK_SUITE}-*.txt"
echo ""
echo "查看 JSON 数据："
echo "  cat TGT_DIR/EVAL-${TASK_SUITE}-*_detailed_timing.json | python -m json.tool"
echo ""
echo "查看最后的计时摘要："
echo "  grep -A 30 'Detailed Timing Breakdown' TGT_DIR/EVAL-${TASK_SUITE}-*.txt"
