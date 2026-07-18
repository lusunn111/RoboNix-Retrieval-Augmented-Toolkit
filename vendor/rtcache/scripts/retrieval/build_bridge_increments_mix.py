#!/usr/bin/env python3
"""
Build incremental Qdrant backups by augmenting MIX base collections with BridgeV2 episodes.

This script will:
1) Restore Qdrant from mix_base backup
2) Add N episodes per existing MIX collection from BridgeV2 TFRecords (using dual-view embeddings)
3) Backup to mix_base+N (without touching the latest symlink by default)
4) Repeat for N in [10, 20, 40, 80] by incremental additions

Key differences from single-view version:
- Uses mix embedding server (port 9021) with 4352-dim embeddings
- Extracts both image_0 (third-person) and image_1 (wrist) from BridgeV2
- Filters for mix collections: libero_*_mix_task_*

Usage:
  python build_bridge_increments_mix.py \
    --bridge-dir /path/to/rtcache/bridgev2/1.0.0 \
    --backup-root /path/to/rtcache/scripts/retrieval/qdrant_backups \
    --targets 10,20,40,80
"""

import argparse
import base64
import logging
import os
import time
import shutil
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import tensorflow as tf
import torch
from tqdm import tqdm
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Disable GPU for TensorFlow (data loading only)
tf.config.set_visible_devices([], "GPU")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.rt_cache_config import get_config


@dataclass
class BuildConfig:
    bridge_dir: str
    backup_root: str
    qdrant_host: str
    qdrant_port: int
    embedding_url: str  # Mix embedding server URL
    openvla_mix_dim: int  # 4352 for mix embeddings
    batch_size: int = 50
    update_latest: bool = False


def list_tfrecords(bridge_dir: str) -> List[str]:
    files = [
        os.path.join(bridge_dir, f)
        for f in os.listdir(bridge_dir)
        if f.startswith("bridge_dataset-") and "tfrecord" in f
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {bridge_dir}")
    return files


class EpisodeStream:
    def __init__(self, bridge_dir: str):
        self.files = list_tfrecords(bridge_dir)
        self.dataset = tf.data.TFRecordDataset(self.files, num_parallel_reads=1)
        self.iterator = iter(self.dataset)
        self.episode_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, bytes]:
        raw_record = next(self.iterator)
        episode_idx = self.episode_idx
        self.episode_idx += 1
        return episode_idx, raw_record.numpy()


def _find_image_keys(feature_keys: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Find image keys for third-person and wrist views.
    
    Returns:
        Tuple of (third_person_key, wrist_key)
    """
    keys = list(feature_keys)
    
    # Priority list for third-person view (usually image_0 or image)
    third_person_key = None
    for key in ["steps/observation/image_0", "steps/observation/image"]:
        if key in keys:
            third_person_key = key
            break
    
    # Priority list for wrist view (usually image_1)
    wrist_key = None
    for key in ["steps/observation/image_1", "steps/observation/wrist_image"]:
        if key in keys:
            wrist_key = key
            break
    
    # If no specific wrist key found, try to find any other image key
    if wrist_key is None and third_person_key is not None:
        for key in sorted(keys):
            if "steps/observation/image" in key and key != third_person_key:
                wrist_key = key
                break
    
    return third_person_key, wrist_key


def parse_bridge_episode_mix(raw_bytes: bytes) -> Tuple[List[bytes], List[bytes], np.ndarray, str]:
    """
    Parse a BridgeV2 episode from TFRecord Example for mix view.
    
    Returns:
        Tuple of (third_person_images, wrist_images, actions, instruction)
    """
    example = tf.train.Example()
    example.ParseFromString(raw_bytes)
    features = example.features.feature

    third_person_key, wrist_key = _find_image_keys(features.keys())
    
    if not third_person_key:
        raise ValueError("No third-person image key found in episode")

    third_person_bytes_list = list(features[third_person_key].bytes_list.value)
    num_steps = len(third_person_bytes_list)
    
    # Get wrist images if available, otherwise use third-person images as fallback
    if wrist_key and wrist_key in features:
        wrist_bytes_list = list(features[wrist_key].bytes_list.value)
        # Ensure same length
        if len(wrist_bytes_list) != num_steps:
            logging.warning(f"Wrist images count ({len(wrist_bytes_list)}) != third-person count ({num_steps}), using third-person as fallback")
            wrist_bytes_list = third_person_bytes_list
    else:
        # No wrist view available, use third-person as both views
        logging.debug("No wrist view found, using third-person view for both")
        wrist_bytes_list = third_person_bytes_list

    instruction = ""
    for key in [
        "steps/language_instruction",
        "steps/observation/natural_language_instruction",
        "steps/observation/language_instruction",
        "steps/observation/instruction",
    ]:
        if key in features:
            value = features[key].bytes_list.value
            if value:
                instruction = value[0].decode("utf-8", errors="ignore")
            break

    actions = None
    if "steps/action" in features:
        action_feature = features["steps/action"]
        if action_feature.float_list.value:
            flat = np.array(action_feature.float_list.value, dtype=np.float32)
            if num_steps > 0 and len(flat) % num_steps == 0:
                actions = flat.reshape(num_steps, -1)
            else:
                actions = flat.reshape(1, -1)
        elif action_feature.bytes_list.value:
            decoded = []
            for b in action_feature.bytes_list.value:
                try:
                    t = tf.io.parse_tensor(b, out_type=tf.float32).numpy()
                    decoded.append(t)
                except Exception:
                    continue
            if decoded:
                actions = np.stack(decoded, axis=0)

    if actions is None:
        actions = np.zeros((num_steps, 7), dtype=np.float32)

    return third_person_bytes_list, wrist_bytes_list, actions, instruction


def generate_mix_embedding(
    third_person_image: Image.Image, 
    wrist_image: Image.Image, 
    instruction: str, 
    url: str
) -> Optional[List[float]]:
    """
    Generate mix embedding via mix embedding server.
    
    Args:
        third_person_image: Third-person view PIL Image
        wrist_image: Wrist view PIL Image
        instruction: Optional text instruction
        url: Mix embedding server URL
        
    Returns:
        Mix embedding vector (4352 dims) or None if failed
    """
    try:
        # Prepare third-person image
        buf_third = BytesIO()
        third_person_image.save(buf_third, format="PNG")
        buf_third.seek(0)
        
        # Prepare wrist image
        buf_wrist = BytesIO()
        wrist_image.save(buf_wrist, format="PNG")
        buf_wrist.seek(0)

        files = {
            "third_person_image": ("third_person.png", buf_third, "image/png"),
            "wrist_image": ("wrist.png", buf_wrist, "image/png")
        }
        data = {
            "instruction": instruction or "",
            "return_individual": "false"
        }

        response = requests.post(url, files=files, data=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "mix_features" not in result:
            return None

        b64_string = result["mix_features"]
        binary_data = base64.b64decode(b64_string)
        buffer = BytesIO(binary_data)
        tensor = torch.load(buffer, map_location="cpu")
        return tensor.squeeze(0).tolist()
    except Exception as e:
        logging.error(f"Mix embedding generation failed: {e}")
        return None


def upsert_episode_mix(
    client: QdrantClient,
    collection_name: str,
    episode_idx: int,
    raw_bytes: bytes,
    embedding_url: str,
    batch_size: int,
    openvla_mix_dim: int,
) -> int:
    """
    Parse and upsert a single episode with mix embeddings.
    """
    third_person_bytes_list, wrist_bytes_list, actions, instruction = parse_bridge_episode_mix(raw_bytes)
    num_steps = len(third_person_bytes_list)
    if num_steps == 0:
        return 0

    if actions.shape[0] != num_steps:
        if actions.shape[0] == 1:
            actions = np.repeat(actions, num_steps, axis=0)
        else:
            min_steps = min(num_steps, actions.shape[0])
            third_person_bytes_list = third_person_bytes_list[:min_steps]
            wrist_bytes_list = wrist_bytes_list[:min_steps]
            actions = actions[:min_steps]
            num_steps = min_steps

    points: List[PointStruct] = []
    inserted = 0

    for step_idx in range(num_steps):
        try:
            third_person_image = Image.open(BytesIO(third_person_bytes_list[step_idx])).convert("RGB")
            wrist_image = Image.open(BytesIO(wrist_bytes_list[step_idx])).convert("RGB")
        except Exception:
            continue

        embedding = generate_mix_embedding(third_person_image, wrist_image, instruction, embedding_url)
        if embedding is None:
            continue
        if len(embedding) != openvla_mix_dim:
            logging.warning(
                f"Embedding dim mismatch: got {len(embedding)}, expected {openvla_mix_dim}"
            )

        next_actions = []
        for offset in range(1, 4):
            next_idx = step_idx + offset
            if next_idx < num_steps:
                next_actions.append(actions[next_idx].tolist())
            else:
                next_actions.append([0.0] * actions.shape[1])

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "dataset_name": "bridge_dataset",
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "current_action": actions[step_idx].tolist(),
                "next_actions": next_actions,
                "language_instruction": instruction or "",
            },
        )
        points.append(point)

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            inserted += len(points)
            points.clear()

    if points:
        client.upsert(collection_name=collection_name, points=points)
        inserted += len(points)

    return inserted


def restore_from_backup(client: QdrantClient, backup_dir: Path) -> None:
    """Restore all collections from a backup directory."""
    snapshot_files = list(backup_dir.glob("*.snapshot"))
    if not snapshot_files:
        raise FileNotFoundError(f"No snapshot files in {backup_dir}")

    for snapshot_file in snapshot_files:
        collection_name = snapshot_file.stem
        try:
            client.get_collection(collection_name)
            client.delete_collection(collection_name)
            time.sleep(0.2)
        except Exception:
            pass

        snapshot_data = snapshot_file.read_bytes()
        qdrant_url = f"http://{client._client._host}:{client._client._port}"
        upload_url = f"{qdrant_url}/collections/{collection_name}/snapshots/upload"
        response = requests.post(
            upload_url, files={"snapshot": snapshot_data}, timeout=300
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to restore {collection_name}: HTTP {response.status_code}"
            )


def _filter_mix_collections(collection_names: List[str]) -> List[str]:
    """Filter for mix collections only."""
    prefixes = [
        "libero_goal_mix_task_",
        "libero_spatial_mix_task_",
        "libero_object_mix_task_",
        "libero_10_mix_task_",
    ]
    return [name for name in collection_names if any(name.startswith(p) for p in prefixes)]


def backup_collections(
    client: QdrantClient,
    backup_root: Path,
    note: str,
    update_latest: bool,
) -> Path:
    """Backup all mix collections to a subdirectory."""
    backup_subdir = backup_root / f"mix_backup_{note}"
    if backup_subdir.exists():
        shutil.rmtree(backup_subdir)
    backup_subdir.mkdir(parents=True, exist_ok=True)

    collections = client.get_collections().collections
    collection_names = _filter_mix_collections([c.name for c in collections])

    for collection_name in collection_names:
        snapshot_info = client.create_snapshot(collection_name=collection_name)
        snapshot_name = snapshot_info.name

        qdrant_url = f"http://{client._client._host}:{client._client._port}"
        download_url = f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}"
        response = requests.get(download_url, timeout=300)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download snapshot for {collection_name}: HTTP {response.status_code}"
            )

        (backup_subdir / f"{collection_name}.snapshot").write_bytes(response.content)
        client.delete_snapshot(collection_name=collection_name, snapshot_name=snapshot_name)

    if update_latest:
        latest_link = backup_root / "latest_mix"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(backup_subdir.name)

    return backup_subdir


def get_collection_names(client: QdrantClient) -> List[str]:
    """Get list of mix collection names."""
    collections = client.get_collections().collections
    names = _filter_mix_collections([c.name for c in collections])
    names.sort()
    return names


def build_increments(config: BuildConfig, targets: List[int]) -> None:
    """Build incremental backups with mix embeddings."""
    logging.info("Connecting to Qdrant...")
    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port, timeout=60.0)

    backup_root = Path(config.backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)
    base_backup = backup_root / "mix_base"

    logging.info(f"Restoring mix base from {base_backup}")
    restore_from_backup(client, base_backup)

    collection_names = get_collection_names(client)
    logging.info(f"Found {len(collection_names)} mix collections")
    
    if not collection_names:
        logging.error("No mix collections found! Make sure mix_base backup exists with mix collections.")
        return

    stream = EpisodeStream(config.bridge_dir)
    prev_target = 0

    for target in targets:
        delta = target - prev_target
        if delta <= 0:
            continue

        logging.info(f"Adding {delta} episodes per mix collection (target {target})")
        remaining = {name: delta for name in collection_names}
        round_robin = list(collection_names)
        rr_idx = 0
        total_needed = len(collection_names) * delta
        with tqdm(total=total_needed, desc=f"Insert mix_base+{target}", unit="episode") as pbar:
            while any(v > 0 for v in remaining.values()):
                for _ in range(len(round_robin)):
                    collection_name = round_robin[rr_idx % len(round_robin)]
                    rr_idx += 1
                    if remaining[collection_name] > 0:
                        break
                else:
                    break

                episode_idx, raw_bytes = next(stream)
                inserted = upsert_episode_mix(
                    client,
                    collection_name,
                    episode_idx,
                    raw_bytes,
                    config.embedding_url,
                    config.batch_size,
                    config.openvla_mix_dim,
                )
                if inserted > 0:
                    remaining[collection_name] -= 1
                    pbar.update(1)

        backup_note = f"base+{target}"
        logging.info(f"Backing up to mix_{backup_note}")
        with tqdm(total=len(collection_names), desc=f"Backup mix_{backup_note}", unit="collection") as pbar:
            backup_subdir = backup_root / f"mix_diff_backup_{backup_note}"
            if backup_subdir.exists():
                shutil.rmtree(backup_subdir)
            backup_subdir.mkdir(parents=True, exist_ok=True)

            for collection_name in collection_names:
                snapshot_info = client.create_snapshot(collection_name=collection_name)
                snapshot_name = snapshot_info.name

                qdrant_url = f"http://{client._client._host}:{client._client._port}"
                download_url = f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}"
                response = requests.get(download_url, timeout=300)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download snapshot for {collection_name}: HTTP {response.status_code}"
                    )

                (backup_subdir / f"{collection_name}.snapshot").write_bytes(response.content)
                client.delete_snapshot(collection_name=collection_name, snapshot_name=snapshot_name)
                pbar.update(1)

            if config.update_latest:
                latest_link = backup_root / "latest_mix"
                if latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(backup_subdir.name)
        prev_target = target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build incremental Qdrant backups with BridgeV2 episodes (Mix View)"
    )
    parser.add_argument(
        "--bridge-dir",
        type=str,
        default="/path/to/rtcache/bridgev2/1.0.0",
        help="BridgeV2 dataset directory",
    )
    parser.add_argument(
        "--backup-root",
        type=str,
        default="/path/to/rtcache/scripts/retrieval/qdrant_backups",
        help="Qdrant backup root directory",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="10,20,30,40,50,60,70,80",
        help="Comma-separated targets for episodes per collection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for Qdrant upserts",
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Update latest_mix symlink to the last backup",
    )
    parser.add_argument(
        "--embedding-url",
        type=str,
        default=None,
        help="Mix embedding server URL (default: http://127.0.0.1:9021/predict)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load centralized config defaults
    rt_config = get_config()

    # Use mix embedding server URL (port 9021)
    embedding_url = args.embedding_url
    if embedding_url is None:
        # Default to mix embedding server
        embedding_url = "http://127.0.0.1:9021/predict"

    config = BuildConfig(
        bridge_dir=args.bridge_dir,
        backup_root=args.backup_root,
        qdrant_host=rt_config.database.qdrant_host,
        qdrant_port=rt_config.database.qdrant_port,
        embedding_url=embedding_url,
        openvla_mix_dim=4352,  # Mix embedding dimension: (1024 + 1152) * 2 views
        batch_size=args.batch_size,
        update_latest=args.update_latest,
    )

    targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    targets = sorted(set(targets))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("Starting incremental build (Mix View)")
    logging.info(f"Embedding server: {config.embedding_url}")
    logging.info(f"Embedding dimension: {config.openvla_mix_dim}")
    logging.info(f"Targets: {targets}")
    build_increments(config, targets)
    logging.info("All mix backups completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
