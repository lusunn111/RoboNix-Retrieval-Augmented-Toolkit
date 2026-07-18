#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/libero_common.sh"

CONFIG_NAME=${CONFIG_NAME:-pi0_libero}
PRECISION=${PRECISION:-bfloat16}
FORCE=${FORCE:-0}

JAX_CHECKPOINT_DIR=${JAX_CHECKPOINT_DIR:-"$ROOT_DIR/models/openpi/openpi-assets/checkpoints/pi0_libero/pi0_libero"}
PI0_TORCH_CKPT=${PI0_TORCH_CKPT:-"$ROOT_DIR/models/openpi/openpi-assets/checkpoints/pi0_libero_torch"}

if [[ ! -d "$JAX_CHECKPOINT_DIR/params" ]]; then
  echo "Missing JAX params directory: $JAX_CHECKPOINT_DIR/params" >&2
  exit 1
fi

if [[ "$FORCE" == "1" || ! -f "$PI0_TORCH_CKPT/model.safetensors" ]]; then
  mkdir -p "$PI0_TORCH_CKPT"
  ensure_transformers_replace "$ROOT_DIR/openpi"
  cd "$ROOT_DIR/openpi"
  uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir "$JAX_CHECKPOINT_DIR" \
    --config_name "$CONFIG_NAME" \
    --output_path "$PI0_TORCH_CKPT" \
    --precision "$PRECISION"
else
  echo "Using existing PyTorch checkpoint: $PI0_TORCH_CKPT/model.safetensors"
fi

if [[ ! -d "$PI0_TORCH_CKPT/assets/physical-intelligence/libero" ]]; then
  ASSETS_SOURCE=""
  if [[ -d "$JAX_CHECKPOINT_DIR/../assets" ]]; then
    ASSETS_SOURCE=$(cd "$JAX_CHECKPOINT_DIR/.." && pwd)/assets
  elif [[ -d "$JAX_CHECKPOINT_DIR/assets" ]]; then
    ASSETS_SOURCE="$JAX_CHECKPOINT_DIR/assets"
  fi

  if [[ -z "$ASSETS_SOURCE" ]]; then
    echo "Could not find LIBERO assets next to JAX checkpoint." >&2
    exit 1
  fi

  rm -rf "$PI0_TORCH_CKPT/assets"
  mkdir -p "$PI0_TORCH_CKPT"
  cp -a "$ASSETS_SOURCE" "$PI0_TORCH_CKPT/assets"
fi

test -f "$PI0_TORCH_CKPT/model.safetensors"
test -d "$PI0_TORCH_CKPT/assets/physical-intelligence/libero"

echo "Prepared PyTorch pi0 checkpoint:"
echo "  $PI0_TORCH_CKPT"
