#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/libero_common.sh"

CONFIG_NAME=${CONFIG_NAME:-pi0_libero}
PI0_TORCH_CKPT=${PI0_TORCH_CKPT:-"$ROOT_DIR/models/openpi/openpi-assets/checkpoints/pi0_libero_torch"}
DATASET_ROOT=${DATASET_ROOT:-"$ROOT_DIR/dataset/flash_episodes"}
INDEX_DIR=${INDEX_DIR:-"$ROOT_DIR/database/flash_index_pytorch"}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
VECTOR_DTYPE=${VECTOR_DTYPE:-float16}
EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-32}
FORCE_INDEX=${FORCE_INDEX:-0}

if [[ ! -f "$PI0_TORCH_CKPT/model.safetensors" ]]; then
  echo "Missing PyTorch checkpoint: $PI0_TORCH_CKPT/model.safetensors" >&2
  echo "Run experiments/00_prepare_pi0_torch.sh first." >&2
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Missing FLASH episode dataset: $DATASET_ROOT" >&2
  exit 1
fi

if [[ -f "$INDEX_DIR/manifest.json" && "$FORCE_INDEX" != "1" ]]; then
  echo "Using existing PyTorch retrieval index: $INDEX_DIR"
  exit 0
fi

ensure_transformers_replace "$FLASH_DIR"

args=(
  --dataset-root "$DATASET_ROOT"
  --output-dir "$INDEX_DIR"
  --flash-root "$FLASH_DIR"
  --embedder-backend pytorch
  --pytorch-checkpoint-dir "$PI0_TORCH_CKPT"
  --config-name "$CONFIG_NAME"
  --device "$PYTORCH_DEVICE"
  --sample-stride "$SAMPLE_STRIDE"
  --embed-batch-size "$EMBED_BATCH_SIZE"
  --vector-dtype "$VECTOR_DTYPE"
  --overwrite
)

if [[ -n "${MAX_RECORDS:-}" ]]; then
  args+=(--max-records "$MAX_RECORDS")
fi

if [[ -n "${SUITE_FILTER:-}" ]]; then
  args+=(--suite "$SUITE_FILTER")
fi

if [[ -n "${TASK_FILTER:-}" ]]; then
  args+=(--task "$TASK_FILTER")
fi

cd "$FLASH_DIR"
uv run python ../database/scripts/build_flash_index.py "${args[@]}"

uv run python - "$INDEX_DIR/manifest.json" <<'PY'
import json
import sys
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
if manifest.get("embedder_backend") != "pytorch":
    raise SystemExit(f"unexpected embedder_backend={manifest.get('embedder_backend')!r}")
print(json.dumps({
    "index": manifest["output_dir"],
    "records": manifest["record_count"],
    "vector_dim": manifest["vector_dim"],
    "embedder_backend": manifest["embedder_backend"],
}, indent=2))
PY
