#!/bin/bash
# 运行脚本：使用DB检索和SD/AR交替执行机制
# 使用方法: bash run_libero_goal_Spec_Relaxed_Alternating.sh
#
# 前置条件：
# 1. 确保 DB retrieval 服务器正在运行 (http://127.0.0.1:5002/pipeline)
# 2. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 3. 确保 Qdrant 数据库正在运行 (localhost:6333)

set -e  # 遇到错误立即退出

# 设置工作目录（根据实际情况修改）
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 检查目录是否存在
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "错误: SpecVLA 根目录不存在: $SPECVLA_ROOT"
    exit 1
fi

# 激活conda环境
echo "激活 conda 环境: specvla"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate specvla

# 检查 conda 环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "specvla" ]; then
    echo "警告: conda 环境可能未正确激活，当前环境: $CONDA_DEFAULT_ENV"
fi

# 设置环境变量
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
export CUDA_VISIBLE_DEVICES=1  # 根据你的GPU情况修改
export MUJOCO_EGL_DEVICE_ID=1   # 根据你的GPU情况修改

# 检查必要的文件是否存在
PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"

if [ ! -d "$PRETRAINED_CHECKPOINT" ]; then
    echo "警告: 预训练模型检查点不存在: $PRETRAINED_CHECKPOINT"
    echo "请检查路径是否正确"
fi

if [ ! -d "$SPEC_CHECKPOINT" ]; then
    echo "警告: SpecVLA 检查点不存在: $SPEC_CHECKPOINT"
    echo "请检查路径是否正确"
fi

# 检查 DB retrieval 服务器是否运行
echo "检查 DB retrieval 服务器..."
if ! curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
    echo "警告: DB retrieval 服务器可能未运行 (http://127.0.0.1:5002)"
    echo "请确保服务器正在运行，否则程序可能会失败"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ DB retrieval 服务器正在运行"
fi

# 打印配置信息
echo "=========================================="
echo "运行配置:"
echo "  SpecVLA 根目录: $SPECVLA_ROOT"
echo "  预训练模型: $PRETRAINED_CHECKPOINT"
echo "  SpecVLA 检查点: $SPEC_CHECKPOINT"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  DB Retrieval: 启用"
echo "  SD/AR Alternating: 启用"
echo "=========================================="
echo ""

# 运行脚本
echo "开始运行评估..."
python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name libero_goal \
    --center_crop True \
    --use_spec True \
    --parallel_draft False \
    --accept_threshold 9 \
    --num_trials_per_task 10 \
    --run_id_note "Spec_Relaxed_Alternating" \
    --use_wandb False

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 运行完成！"
    echo "=========================================="
    echo "结果文件保存在: $SPECVLA_ROOT/openvla/specdecoding/test-speed/libero_goal_Spec_Relaxed"
else
    echo ""
    echo "=========================================="
    echo "✗ 运行失败！"
    echo "=========================================="
    echo "请检查错误信息并重试"
    exit 1
fi

