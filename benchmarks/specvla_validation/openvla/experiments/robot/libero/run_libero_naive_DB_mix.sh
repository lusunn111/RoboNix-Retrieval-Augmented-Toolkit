#!/bin/bash
# Run LIBERO Pure DB Retrieval Evaluation (Mix View: Third-Person + Wrist)
# Tests ALL four task suites: libero_goal, libero_spatial, libero_object, libero_10

# ============================================
# Configuration - Modify these variables
# ============================================
NUM_TRIALS=10             # Number of trials per task
RUN_ID_NOTE=""            # Optional note for run ID (leave empty for none)

# All four task suites to test
# TASK_SUITES=("libero_goal" "libero_spatial" "libero_object" "libero_10")
# TASK_SUITES=("libero_goal")
TASK_SUITES=("libero_spatial" "libero_object" "libero_10")

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "LIBERO Pure DB Retrieval - Mix View"
echo "(Third-Person + Wrist Camera)"
echo "=========================================="
echo "Task Suites: ${TASK_SUITES[*]}"
echo "Number of Trials per task: $NUM_TRIALS"
echo "Retrieval Server: http://127.0.0.1:5003"
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

# Check Mix DB retrieval server (port 5003)
echo "Checking Mix DB retrieval server (port 5003)..."
if ! curl -s --connect-timeout 2 http://127.0.0.1:5003/health > /dev/null 2>&1; then
    echo "[WARNING] Mix DB retrieval server may not be running (http://127.0.0.1:5003)"
    echo "Please ensure the mix retrieval server is running:"
    echo "  cd /path/to/rtcache/scripts/retrieval"
    echo "  ./start_libero_goal_retrieval_mix.sh --skip-restore"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Mix DB retrieval server is running (port 5003)"
fi

# Check Mix embedding server (port 9021)
echo "Checking Mix embedding server (port 9021)..."
if ! curl -s --connect-timeout 2 http://127.0.0.1:9021/health > /dev/null 2>&1; then
    echo "[WARNING] Mix embedding server may not be running (http://127.0.0.1:9021)"
    echo "Please ensure the mix embedding server is running"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Mix embedding server is running (port 9021)"
fi
echo "=========================================="

# ============================================
# Run Evaluation for ALL task suites
# ============================================
TOTAL_SUITES=${#TASK_SUITES[@]}
CURRENT_SUITE=0
FAILED_SUITES=()
SUCCESSFUL_SUITES=()

for TASK_SUITE in "${TASK_SUITES[@]}"; do
    CURRENT_SUITE=$((CURRENT_SUITE + 1))
    
    echo ""
    echo "=========================================="
    echo "[$CURRENT_SUITE/$TOTAL_SUITES] Starting: $TASK_SUITE"
    echo "=========================================="
    
    if [ -z "$RUN_ID_NOTE" ]; then
        python openvla/experiments/robot/libero/run_libero_naive_DB_mix.py \
            --task_suite_name "$TASK_SUITE" \
            --num_trials_per_task "$NUM_TRIALS" \
            --center_crop True
    else
        python openvla/experiments/robot/libero/run_libero_naive_DB_mix.py \
            --task_suite_name "$TASK_SUITE" \
            --num_trials_per_task "$NUM_TRIALS" \
            --center_crop True \
            --run_id_note "$RUN_ID_NOTE"
    fi
    
    if [ $? -eq 0 ]; then
        echo "[$CURRENT_SUITE/$TOTAL_SUITES] $TASK_SUITE completed successfully!"
        SUCCESSFUL_SUITES+=("$TASK_SUITE")
    else
        echo "[$CURRENT_SUITE/$TOTAL_SUITES] $TASK_SUITE FAILED!"
        FAILED_SUITES+=("$TASK_SUITE")
    fi
done

# ============================================
# Final Summary
# ============================================
echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETED (Mix View)"
echo "=========================================="
echo "Successful: ${#SUCCESSFUL_SUITES[@]}/$TOTAL_SUITES"
for suite in "${SUCCESSFUL_SUITES[@]}"; do
    echo "  [OK] $suite"
done

if [ ${#FAILED_SUITES[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED_SUITES[@]}/$TOTAL_SUITES"
    for suite in "${FAILED_SUITES[@]}"; do
        echo "  [FAIL] $suite"
    done
fi

echo ""
echo "Log directories:"
for suite in "${TASK_SUITES[@]}"; do
    echo "  - $SPECVLA_ROOT/openvla/specdecoding/test-speed/${suite}_naive_DB_mix/"
done
echo "=========================================="

# Exit with error if any suite failed
if [ ${#FAILED_SUITES[@]} -gt 0 ]; then
    exit 1
fi
