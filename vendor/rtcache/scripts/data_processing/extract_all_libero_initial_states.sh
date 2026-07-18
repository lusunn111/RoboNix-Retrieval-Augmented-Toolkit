#!/usr/bin/env bash
set -euo pipefail

# Extract initial states for all four LIBERO subsets.
# Output goes to: scripts/data_processing/data/libero_initial_states/<subset>/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT_DIR/scripts/data_processing/extract_libero_initial_states.py"

BASE_DATASET_PATH="/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"

for subset in goal 10 object spatial; do
  echo "[extract] subset=$subset"
  python3 "$SCRIPT" \
    --dataset_type "$subset" \
    --base_dataset_path "$BASE_DATASET_PATH" \
    --max_episodes 5 \
    --save_images \
    --image_format jpg \
    --overwrite \
    --log_level INFO
  echo
done

echo "All done."