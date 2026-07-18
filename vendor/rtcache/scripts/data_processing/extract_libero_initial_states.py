#!/usr/bin/env python3
"""Extract initial states from modified LIBERO RLDS TFDS snapshots.

For each episode in a given LIBERO subset (goal/10/object/spatial), this script
extracts the first step (initial state) and writes a JSONL file.
Optionally saves the main camera image and wrist image.

Outputs (by default):
  scripts/data_processing/data/libero_initial_states/<dataset_type>/
    initial_states.jsonl
    images/<episode_idx>_main.jpg
    images/<episode_idx>_wrist.jpg

"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# TensorFlow / TFDS only for reading the dataset
import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image
from tqdm import tqdm


# Disable GPU for TensorFlow (data loading only)
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


DATASET_CONFIGS = {
    "goal": {
        "folder_name": "libero_goal_no_noops",
    },
    "10": {
        "folder_name": "libero_10_no_noops",
    },
    "object": {
        "folder_name": "libero_object_no_noops",
    },
    "spatial": {
        "folder_name": "libero_spatial_no_noops",
    },
}


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "numpy"):
        raw = value.numpy()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return raw
    return str(value)


def _ensure_uint8_image(array: Any) -> np.ndarray:
    """Ensure HxWxC uint8 array for PIL."""
    if isinstance(array, tf.Tensor):
        array = array.numpy()
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _save_image(img_array: Any, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(_ensure_uint8_image(img_array))
    # Keep consistent extension behavior
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(path, format="JPEG", quality=quality)
    elif path.suffix.lower() == ".png":
        img.save(path, format="PNG")
    else:
        # Default to JPEG
        img.save(path.with_suffix(".jpg"), format="JPEG", quality=quality)


def _extract_first_step(episode: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (first_step_dict, episode_metadata_dict) as numpy-friendly dicts."""
    steps_ds = episode["steps"]
    # steps_ds is a tf.data.Dataset of dicts
    first_step = next(iter(steps_ds))

    # episode_metadata may be absent in some variants, but present here
    metadata = episode.get("episode_metadata", {})

    return first_step, metadata


def _to_list(x: Any) -> Any:
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract initial states from LIBERO TFDS snapshots")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        required=True,
        help="Which LIBERO subset to extract: goal/10/object/spatial",
    )
    parser.add_argument(
        "--base_dataset_path",
        type=str,
        default="/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551",
        help="Base snapshot directory (contains libero_*_no_noops folders)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Full path to a specific dataset folder (overrides base_dataset_path+dataset_type)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: scripts/data_processing/data/libero_initial_states/<dataset_type>)",
    )
    parser.add_argument("--max_episodes", type=int, default=-1, help="Max episodes to export (-1 for all)")
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save main and wrist images alongside JSONL",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        choices=["jpg", "png"],
        default="jpg",
        help="Image format when --save_images is enabled",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSONL/images",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Resolve dataset path
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        folder = DATASET_CONFIGS[args.dataset_type]["folder_name"]
        dataset_path = Path(args.base_dataset_path) / folder / "1.0.0"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Default output dir inside scripts/data_processing
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "data" / "libero_initial_states" / args.dataset_type

    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / "initial_states.jsonl"
    images_dir = output_dir / "images"

    if out_jsonl.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {out_jsonl}. Use --overwrite or choose --output_dir."
        )

    if args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()

    if args.overwrite and images_dir.exists() and args.save_images:
        # Only remove images if explicitly saving images; avoid deleting other artifacts.
        for p in images_dir.glob("*.*"):
            try:
                p.unlink()
            except Exception:
                pass

    logger.info(f"Loading dataset from: {dataset_path}")

    builder = tfds.builder_from_directory(builder_dir=str(dataset_path))
    ds = builder.as_dataset(split="train", shuffle_files=False)

    total_episodes = builder.info.splits["train"].num_examples
    logger.info(f"Total episodes: {total_episodes}")

    exported = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for episode_idx, episode in enumerate(tqdm(ds, desc=f"Extracting {args.dataset_type} initial states")):
            if args.max_episodes > 0 and episode_idx >= args.max_episodes:
                break

            try:
                first_step, metadata = _extract_first_step(episode)

                # Extract fields
                language_instruction = _decode_text(first_step.get("language_instruction"))

                obs = first_step.get("observation", {})
                main_img = obs.get("image")
                wrist_img = obs.get("wrist_image")
                state = _to_list(obs.get("state"))
                joint_state = _to_list(obs.get("joint_state"))

                # Save images (optional)
                main_path = ""
                wrist_path = ""
                if args.save_images:
                    ext = args.image_format
                    main_path = str((images_dir / f"{episode_idx:06d}_main.{ext}").relative_to(output_dir))
                    wrist_path = str((images_dir / f"{episode_idx:06d}_wrist.{ext}").relative_to(output_dir))

                    if main_img is not None:
                        _save_image(main_img, output_dir / main_path)
                    if wrist_img is not None:
                        _save_image(wrist_img, output_dir / wrist_path)

                # episode_metadata.file_path is a Text feature
                file_path = ""
                if isinstance(metadata, dict) and "file_path" in metadata:
                    file_path = _decode_text(metadata.get("file_path"))

                record = {
                    "dataset_type": args.dataset_type,
                    "dataset_path": str(dataset_path),
                    "episode_idx": int(episode_idx),
                    "language_instruction": language_instruction,
                    "initial_observation": {
                        "state": state,
                        "joint_state": joint_state,
                        "main_image": main_path,
                        "wrist_image": wrist_path,
                    },
                    "episode_metadata": {
                        "file_path": file_path,
                    },
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

            except StopIteration:
                # Empty episode dataset (unexpected)
                logger.warning(f"Episode {episode_idx} has no steps; skipping")
                continue
            except Exception as e:
                logger.error(f"Failed on episode {episode_idx}: {e}")
                continue

    logger.info(f"Done. Exported {exported} episodes")
    logger.info(f"Wrote: {out_jsonl}")
    if args.save_images:
        logger.info(f"Images: {images_dir}")


if __name__ == "__main__":
    # Avoid TF grabbing GPUs via env as well
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
