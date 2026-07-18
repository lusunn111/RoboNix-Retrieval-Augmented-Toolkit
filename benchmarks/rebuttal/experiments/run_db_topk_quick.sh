#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${ROOT_DIR:-/path/to/MMRebuttal}

OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/outputs/experiments/db_topk_quick"}
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
fi
OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")

TASK=${TASK:-0}
TRIALS=${TRIALS:-2}
GPU=${GPU:-cuda:0}
PORT_BASE=${PORT_BASE:-8503}
RTCACHE_TOP_K=${RTCACHE_TOP_K:-3}
RTCACHE_RERANK_MIN_ACCEPT_LEN=${RTCACHE_RERANK_MIN_ACCEPT_LEN:-8}
RTCACHE_NOVERIFY_POLICY=${RTCACHE_NOVERIFY_POLICY:-composite_2to1}
RTCACHE_NOVERIFY_MAX_CONSECUTIVE=${RTCACHE_NOVERIFY_MAX_CONSECUTIVE:-2}
SAVE_VIDEOS=${SAVE_VIDEOS:-0}

mkdir -p "$OUTPUT_ROOT"

suites=(libero_goal libero_spatial libero_object libero_10)
for i in "${!suites[@]}"; do
  suite="${suites[$i]}"
  port=$((PORT_BASE + i))
  run_task_label=${TASK//[^0-9A-Za-z_-]/_}
  run_name="db_topk_${suite}_task_${run_task_label}_trials_${TRIALS}"
  echo "[$(date '+%F %T')] suite=$suite task=$TASK trials=$TRIALS port=$port output=$OUTPUT_ROOT"
  SUITE="$suite" \
    TASK="$TASK" \
    TRIALS="$TRIALS" \
    GPU="$GPU" \
    PORT="$port" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RUN_NAME="$run_name" \
    RTCACHE_TOP_K="$RTCACHE_TOP_K" \
    RTCACHE_RERANK_MIN_ACCEPT_LEN="$RTCACHE_RERANK_MIN_ACCEPT_LEN" \
    RTCACHE_NOVERIFY_POLICY="$RTCACHE_NOVERIFY_POLICY" \
    RTCACHE_NOVERIFY_MAX_CONSECUTIVE="$RTCACHE_NOVERIFY_MAX_CONSECUTIVE" \
    SAVE_VIDEOS="$SAVE_VIDEOS" \
    "$SCRIPT_DIR/03_run_flash_db_draft_pytorch.sh"
done

(cd "$ROOT_DIR" && uv run python experiments/analyze_full_30.py \
  --root "$OUTPUT_ROOT" \
  --out "$OUTPUT_ROOT/summary")

echo "DB top-k quick run complete:"
echo "  output: $OUTPUT_ROOT"
echo "  summary: $OUTPUT_ROOT/summary"
