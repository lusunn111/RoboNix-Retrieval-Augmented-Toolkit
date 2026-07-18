"""
Utilities for inserting successful LIBERO rollouts into the RT-Cache Qdrant DB.

This mirrors the schema used by `rtcache/scripts/data_processing/process_libero_goal.py`:
- One Qdrant collection per task: `{prefix}{task_id}` (e.g., `libero_goal_task_93`)
- Each point stores an OpenVLA image embedding (2176-d) and payload:
  `dataset_name`, `episode_idx`, `step_idx`, `current_action`, `next_actions`, `language_instruction`
"""

from __future__ import annotations

import base64
import hashlib
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import requests
import torch
from PIL import Image


_DATASET_TYPE_TO_COLLECTION_PREFIX = {
    "goal": "libero_goal_task_",
    "10": "libero_10_task_",
    "object": "libero_object_task_",
    "spatial": "libero_spatial_task_",
    "90": "libero_90_task_",
}


def normalize_dataset_type(name: str) -> Optional[str]:
    if not name:
        return None
    normalized = name.lower().strip()
    if normalized.startswith("libero_"):
        normalized = normalized.replace("libero_", "", 1)
    if normalized.endswith("_no_noops"):
        normalized = normalized[: -len("_no_noops")]
    return normalized if normalized in _DATASET_TYPE_TO_COLLECTION_PREFIX else None


@dataclass(frozen=True)
class OnlineRolloutInsertConfig:
    embedding_server_url: str = "http://127.0.0.1:9020/predict"
    qdrant_url: str = "http://127.0.0.1:6333"
    openvla_image_dim: int = 2176
    request_timeout_s: float = 30.0
    upsert_wait: bool = True
    upsert_batch_size: int = 16


class LiberoOnlineRolloutInserter:
    def __init__(self, config: OnlineRolloutInsertConfig):
        self.config = config
        self._session = requests.Session()

    @staticmethod
    def compute_task_id(instruction: str) -> int:
        normalized = (instruction or "").lower().strip()
        if not normalized:
            raise ValueError("instruction must be a non-empty string")
        instruction_hash = int(hashlib.md5(normalized.encode("utf-8")).hexdigest(), 16)
        return instruction_hash % 1001

    def get_collection_name(self, dataset_type: str, instruction: str) -> str:
        normalized_dataset_type = normalize_dataset_type(dataset_type)
        if normalized_dataset_type is None:
            raise ValueError(
                f"Unrecognized dataset_type={dataset_type!r}. Expected one of: {sorted(_DATASET_TYPE_TO_COLLECTION_PREFIX)}"
            )
        prefix = _DATASET_TYPE_TO_COLLECTION_PREFIX[normalized_dataset_type]
        task_id = self.compute_task_id(instruction)
        return f"{prefix}{task_id}"

    def ensure_collection_exists(self, collection_name: str) -> None:
        base = self.config.qdrant_url.rstrip("/")
        url = f"{base}/collections/{collection_name}"
        resp = self._session.get(url, timeout=self.config.request_timeout_s)
        if resp.status_code == 404:
            create_body = {
                "vectors": {
                    "size": self.config.openvla_image_dim,
                    "distance": "Cosine",
                }
            }
            create_resp = self._session.put(url, json=create_body, timeout=self.config.request_timeout_s)
            create_resp.raise_for_status()
            return
        resp.raise_for_status()

    def generate_embedding(
        self, image: Union[np.ndarray, Image.Image], instruction: str = ""
    ) -> Optional[List[float]]:
        try:
            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")

            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            buf.seek(0)

            files = {"file": ("image.png", buf, "image/png")}
            data = {"instruction": instruction or "", "option": "image"}

            resp = self._session.post(
                self.config.embedding_server_url,
                files=files,
                data=data,
                timeout=self.config.request_timeout_s,
            )
            resp.raise_for_status()
            result = resp.json()
            b64_string = result.get("image_features")
            if not b64_string:
                return None

            binary_data = base64.b64decode(b64_string)
            tensor = torch.load(BytesIO(binary_data), map_location="cpu")
            return tensor.squeeze(0).tolist()
        except Exception:
            return None

    def _upsert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> None:
        if not points:
            return
        base = self.config.qdrant_url.rstrip("/")
        params = {"wait": "true"} if self.config.upsert_wait else None
        url = f"{base}/collections/{collection_name}/points"
        resp = self._session.put(
            url,
            params=params,
            json={"points": points},
            timeout=self.config.request_timeout_s,
        )
        resp.raise_for_status()

    def insert_trajectory(
        self,
        images: Sequence[Union[np.ndarray, Image.Image]],
        actions: Sequence[Union[np.ndarray, Sequence[float]]],
        instruction: str,
        dataset_type: str,
        *,
        dataset_name: str,
        episode_idx: int,
        stride: int = 1,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        if len(images) != len(actions):
            raise ValueError(f"images/actions length mismatch: {len(images)} != {len(actions)}")
        if stride <= 0:
            raise ValueError("stride must be >= 1")

        total_steps = len(actions)
        if total_steps == 0:
            return {
                "success": True,
                "inserted_points": 0,
                "failed_embeddings": 0,
                "collection_name": None,
                "task_id": None,
            }

        limit_steps = total_steps
        if max_steps is not None and max_steps >= 0:
            limit_steps = min(limit_steps, max_steps)

        task_id = self.compute_task_id(instruction)
        collection_name = self.get_collection_name(dataset_type, instruction)
        self.ensure_collection_exists(collection_name)

        points: List[Dict[str, Any]] = []
        inserted_points = 0
        failed_embeddings = 0

        for step_idx in range(0, limit_steps, stride):
            embedding = self.generate_embedding(images[step_idx], instruction=instruction)
            if embedding is None:
                failed_embeddings += 1
                continue

            current_action = np.asarray(actions[step_idx], dtype=np.float32).reshape(-1).tolist()
            current_action = (current_action + [0.0] * 7)[:7]

            next_actions: List[List[float]] = []
            for offset in range(1, 4):
                next_idx = step_idx + offset
                if next_idx < total_steps:
                    next_action = np.asarray(actions[next_idx], dtype=np.float32).reshape(-1).tolist()
                    next_action = (next_action + [0.0] * 7)[:7]
                else:
                    next_action = [0.0] * 7
                next_actions.append(next_action)

            points.append(
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "dataset_name": dataset_name,
                        "episode_idx": episode_idx,
                        "step_idx": step_idx,
                        "current_action": current_action,
                        "next_actions": next_actions,
                        "language_instruction": instruction or "",
                    },
                }
            )

            if len(points) >= self.config.upsert_batch_size:
                self._upsert_points(collection_name, points)
                inserted_points += len(points)
                points.clear()

        if points:
            self._upsert_points(collection_name, points)
            inserted_points += len(points)
            points.clear()

        return {
            "success": True,
            "task_id": task_id,
            "collection_name": collection_name,
            "inserted_points": inserted_points,
            "failed_embeddings": failed_embeddings,
            "total_steps": total_steps,
            "stride": stride,
        }

