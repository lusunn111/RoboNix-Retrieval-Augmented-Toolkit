#!/bin/bash
# Run LIBERO Pure DB Retrieval Evaluation
# Usage: Modify TASK_SUITE variable to choose dataset (goal, spatial, object, 10)

set -e  # Exit on error

# ============================================
# Configuration - Modify these variables
# ============================================
TASK_SUITE="libero_10"  # Options: libero_goal, libero_spatial, libero_object, libero_10
NUM_TRIALS=10             # Number of trials per task
RUN_ID_NOTE=""            # Optional note for run ID (leave empty for none)

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "LIBERO Pure DB Retrieval Evaluation"
echo "=========================================="
echo "Task Suite: $TASK_SUITE"
echo "Number of Trials: $NUM_TRIALS"
echo "=========================================="

# Set working directory
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# Check if directory exists
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "[ERROR] SpecVLA directory does not exist: $SPECVLA_ROOT"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: specvla"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate specvla

# Check if conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "specvla" ]; then
    echo "[WARNING] Conda environment may not be activated correctly. Current: $CONDA_DEFAULT_ENV"
fi

# Set environment variables for LIBERO
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

echo "[INFO] Working directory: $SPECVLA_ROOT"
echo "[INFO] Conda environment: $CONDA_DEFAULT_ENV"
echo "[INFO] CUDA_VISIBLE_DEVICES=1"
echo "[INFO] MUJOCO_EGL_DEVICE_ID=1"
echo "[INFO] PYTHONPATH=$PYTHONPATH"
echo "=========================================="

# Check DB retrieval server
echo "Checking DB retrieval server..."
if ! curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
    echo "[WARNING] DB retrieval server may not be running (http://127.0.0.1:5002)"
    echo "Please ensure the server is running, or the program may fail"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ DB retrieval server is running"
fi
echo "=========================================="

# ============================================
# Run Evaluation
# ============================================
echo "Starting evaluation..."

if [ -z "$RUN_ID_NOTE" ]; then
    # Run without note
    python openvla/experiments/robot/libero/run_libero_naive_DB.py \
        --task_suite_name "$TASK_SUITE" \
        --num_trials_per_task "$NUM_TRIALS" \
        --center_crop True
else
    # Run with note
    python openvla/experiments/robot/libero/run_libero_naive_DB.py \
        --task_suite_name "$TASK_SUITE" \
        --num_trials_per_task "$NUM_TRIALS" \
        --center_crop True \
        --run_id_note "$RUN_ID_NOTE"
fi

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Evaluation completed!"
    echo "=========================================="
    echo "Check logs in: $SPECVLA_ROOT/openvla/specdecoding/test-speed/${TASK_SUITE}_naive_DB/"
else
    echo ""
    echo "=========================================="
    echo "✗ Evaluation failed!"
    echo "=========================================="
    echo "Please check error messages and retry"
    exit 1
fi
