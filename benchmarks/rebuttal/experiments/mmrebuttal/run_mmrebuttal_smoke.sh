#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/MMRebuttal"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/zhihao/mmrebuttal_outputs/small_formal/smoke}"

cd "$PROJECT_ROOT"

SUITE="${SUITE:-libero_goal}" \
TASK_IDS="${TASK_IDS:-0}" \
TRIALS="${TRIALS:-1}" \
GPU="${GPU:-0}" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SAVE_VIDEOS="${SAVE_VIDEOS:-False}" \
bash experiments/mmrebuttal/run_mmrebuttal_suite.sh

OUTPUT_ROOT="$OUTPUT_ROOT" bash experiments/mmrebuttal/run_mmrebuttal_analysis.sh

echo "Smoke finished: $OUTPUT_ROOT"
