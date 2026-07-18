#!/bin/bash
# 主实验 - 滑动窗口大小 w=17
# 
# 在 CUDA 1 上运行，Goal 环境，每个任务 10 次
#
# 使用方法: bash run_window15_cuda1.sh

SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 激活环境
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla
fi

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "运行 Window Size = 17 实验"

python openvla/experiments/robot/libero/run_libero_block_sd.py \
    --model_family openvla \
    --pretrained_checkpoint "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-goal" \
    --spec_checkpoint "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190" \
    --task_suite_name libero_goal \
    --center_crop True \
    --top_k 5 \
    --prob_threshold 0.1 \
    --use_avg_prob True \
    --accept_threshold 9 \
    --block_sum_threshold 45 \
    --block_max_threshold 25 \
    --num_trials_per_task 10 \
    --run_id_note "Window17_Main" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False \
    --ablation_mode 3 \
    --window_size 17 \
    --composite_threshold 0.143210 \
    --displacement_range_min 0.000009 \
    --displacement_range_max 0.139051 \
    --radius_range_min 0.000001 \
    --radius_range_max 0.016873

echo "Window Size = 17 实验完成！"
