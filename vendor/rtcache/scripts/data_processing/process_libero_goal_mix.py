#!/usr/bin/env python3
"""
Process LIBERO Datasets for RT-Cache System (Mix View: Third-Person + Wrist)

This script processes LIBERO datasets using BOTH third-person and wrist camera views.
For each task, it creates a separate Qdrant collection and stores:
- Image embeddings: Concatenated (DINOv2 + SigLIP) from both views = 4352 dims
- Metadata: dataset_name, episode_idx, step_idx, current_action, next_actions (next 3 steps)

Embedding structure per point:
- Third-person DINOv2: 1024 dims
- Third-person SigLIP: 1152 dims  
- Wrist DINOv2: 1024 dims
- Wrist SigLIP: 1152 dims
- Total: 4352 dims

Supported datasets:
- libero_goal: LIBERO-Goal (10 tasks)
- libero_10: LIBERO-10 (10 tasks)
- libero_object: LIBERO-Object (10 tasks)
- libero_spatial: LIBERO-Spatial (10 tasks)

Usage:
    # Process single dataset
    python process_libero_goal_mix.py --dataset_type goal
    
    # Process all datasets
    python process_libero_goal_mix.py --process_all
    
    # Clear database and process
    python process_libero_goal_mix.py --dataset_type goal --clear_db

Author: RT-Cache Team
Date: 2024
"""

import os
import sys
import argparse
import logging
import time
import uuid
import base64
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from io import BytesIO
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import requests

# Disable GPU for TensorFlow (only used for data loading, not inference)
tf.config.set_visible_devices([], 'GPU')

# Ensure PyTorch also doesn't initialize CUDA unnecessarily
os.environ['CUDA_VISIBLE_DEVICES'] = '' if 'CUDA_VISIBLE_DEVICES' not in os.environ else os.environ['CUDA_VISIBLE_DEVICES']

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.rt_cache_config import get_config


def setup_logging(level="INFO"):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Dataset configuration mapping
DATASET_CONFIGS = {
    "goal": {
        "folder_name": "libero_goal_no_noops",
        "dataset_name": "libero_goal_no_noops",
        "task_suite_name": "libero_goal",
        "collection_prefix": "libero_goal_mix"  # Note: mix suffix
    },
    "10": {
        "folder_name": "libero_10_no_noops",
        "dataset_name": "libero_10_no_noops",
        "task_suite_name": "libero_10",
        "collection_prefix": "libero_10_mix"
    },
    "object": {
        "folder_name": "libero_object_no_noops",
        "dataset_name": "libero_object_no_noops",
        "task_suite_name": "libero_object",
        "collection_prefix": "libero_object_mix"
    },
    "spatial": {
        "folder_name": "libero_spatial_no_noops",
        "dataset_name": "libero_spatial_no_noops",
        "task_suite_name": "libero_spatial",
        "collection_prefix": "libero_spatial_mix"
    }
}


@dataclass
class ProcessingConfig:
    """Configuration for LIBERO dataset processing (Mix view)"""
    
    # Dataset type: goal, 10, object, or spatial
    dataset_type: str = "goal"
    
    # Base dataset directory
    base_dataset_path: str = "/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
    
    # These will be set automatically based on dataset_type
    dataset_path: str = ""
    dataset_name: str = ""
    task_suite_name: str = ""
    collection_prefix: str = ""
    
    # Server URLs - use mix embedding server
    embedding_server_url: str = "http://127.0.0.1:9021/predict"
    
    # Database settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Processing parameters
    batch_size: int = 50
    max_episodes: int = -1  # -1 means all episodes
    min_episode_length: int = 5
    
    # Whether to clear existing collections before processing
    clear_db: bool = False
    
    # Embedding dimensions: (DINOv2 + SigLIP) * 2 views
    openvla_mix_dim: int = 4352
    
    # Backup settings
    backup_dir: str = "/path/to/rtcache/scripts/retrieval/qdrant_backups"
    backup_name: str = "mix_base"
    
    def __post_init__(self):
        """Initialize dataset-specific configuration based on dataset_type"""
        if self.dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}. Must be one of {list(DATASET_CONFIGS.keys())}")
        
        config = DATASET_CONFIGS[self.dataset_type]
        self.dataset_name = config["dataset_name"]
        self.task_suite_name = config["task_suite_name"]
        self.collection_prefix = config["collection_prefix"]
        
        # Construct full dataset path
        if not self.dataset_path:
            self.dataset_path = f"{self.base_dataset_path}/{config['folder_name']}/1.0.0"


class LiberoMixDatasetProcessor:
    """
    Processor for LIBERO datasets using Mix view (third-person + wrist).
    
    Creates separate Qdrant collections for each task and stores:
    - Mix image embeddings (4352 dims)
    - Action sequences (current + next 3 steps)
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the processor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self._init_storage()
        
        # Initialize task mapping
        self.task_id_to_collection = {}
        self.instruction_to_task_id = {}
        
        # Statistics
        self.stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "skipped_episodes": 0,
            "failed_embeddings": 0,
            "task_distribution": defaultdict(int)
        }
        
        # Track which instructions have been seen
        self.seen_instructions = set()
        
    def _init_storage(self):
        """Initialize Qdrant connection"""
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            timeout=60.0
        )
        self.logger.info(f"Connected to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}")
    
    def clear_mix_collections(self):
        """Clear all existing mix collections for the current dataset type"""
        self.logger.info(f"Clearing existing mix collections with prefix: {self.config.collection_prefix}")
        
        try:
            # Get all collections
            collections = self.qdrant_client.get_collections().collections
            
            # Find and delete mix collections
            deleted_count = 0
            for col in collections:
                if col.name.startswith(self.config.collection_prefix + "_task_"):
                    self.logger.info(f"Deleting collection: {col.name}")
                    self.qdrant_client.delete_collection(col.name)
                    deleted_count += 1
            
            self.logger.info(f"Deleted {deleted_count} existing collections")
            
        except Exception as e:
            self.logger.error(f"Error clearing collections: {e}")
    
    def clear_all_mix_collections(self):
        """Clear ALL mix collections (all dataset types)"""
        self.logger.info("Clearing ALL mix collections...")
        
        try:
            collections = self.qdrant_client.get_collections().collections
            
            mix_prefixes = [cfg["collection_prefix"] for cfg in DATASET_CONFIGS.values()]
            
            deleted_count = 0
            for col in collections:
                if any(col.name.startswith(prefix + "_task_") for prefix in mix_prefixes):
                    self.logger.info(f"Deleting collection: {col.name}")
                    self.qdrant_client.delete_collection(col.name)
                    deleted_count += 1
            
            self.logger.info(f"Deleted {deleted_count} mix collections total")
            
        except Exception as e:
            self.logger.error(f"Error clearing collections: {e}")
            
    def _get_or_create_collection(self, task_id: int) -> str:
        """
        Get or create collection name for a task_id.
        
        Args:
            task_id: Task ID
            
        Returns:
            Collection name
        """
        if task_id in self.task_id_to_collection:
            return self.task_id_to_collection[task_id]
        
        collection_name = f"{self.config.collection_prefix}_task_{task_id}"
        
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.openvla_mix_dim,  # 4352 dims
                    distance="Cosine"
                )
            )
            self.logger.info(f"Created mix collection: {collection_name} (dim={self.config.openvla_mix_dim})")
        
        self.task_id_to_collection[task_id] = collection_name
        return collection_name
    
    def _match_task_id(self, language_instruction: str) -> Optional[int]:
        """
        Match language instruction to task_id using hash-based mapping.
        
        Args:
            language_instruction: The language instruction from the dataset
            
        Returns:
            task_id based on MD5 hash
        """
        if not language_instruction:
            return None
            
        instruction_lower = language_instruction.lower().strip()
        
        # Check cache
        if instruction_lower in self.instruction_to_task_id:
            return self.instruction_to_task_id[instruction_lower]
        
        # Use MD5 hash for consistent assignment
        instruction_hash = int(hashlib.md5(instruction_lower.encode('utf-8')).hexdigest(), 16)
        task_id = instruction_hash % 1001
        
        self.instruction_to_task_id[instruction_lower] = task_id
        
        # Log once per unique instruction
        if instruction_lower not in self.seen_instructions:
            self.seen_instructions.add(instruction_lower)
            self.logger.info(f"Hash-based task_id={task_id} for: '{language_instruction}'")
        
        return task_id
        
    def process_dataset(self):
        """Process the entire LIBERO dataset with mix view"""
        self.logger.info(f"Loading dataset from: {self.config.dataset_path}")
        
        # Clear existing collections if requested
        if self.config.clear_db:
            self.clear_mix_collections()
        
        # Load dataset
        builder = tfds.builder_from_directory(builder_dir=self.config.dataset_path)
        ds = builder.as_dataset(split='train', shuffle_files=False)
        
        # Get total number of episodes
        total_episodes = builder.info.splits['train'].num_examples
        self.logger.info(f"Total episodes in dataset: {total_episodes}")
        
        # Process episodes
        batch_buffers = defaultdict(lambda: {'points': []})
        
        episode_count = 0
        for episode_idx, episode in enumerate(tqdm(ds, desc="Processing episodes")):
            if self.config.max_episodes > 0 and episode_idx >= self.config.max_episodes:
                break
                
            try:
                task_id = self._process_episode(episode, episode_idx, batch_buffers)
                
                if task_id is not None:
                    episode_count += 1
                    self.stats['task_distribution'][task_id] += 1
                    
                    # Flush batch if needed
                    if len(batch_buffers[task_id]['points']) >= self.config.batch_size:
                        self._flush_batch(task_id, batch_buffers[task_id])
                            
            except Exception as e:
                self.logger.error(f"Error processing episode {episode_idx}: {e}")
                self.stats['skipped_episodes'] += 1
                continue
                
        # Final flush for all tasks
        for task_id, buffer in batch_buffers.items():
            if buffer['points']:
                self._flush_batch(task_id, buffer)
                
        # Print statistics
        self._print_statistics()
        
    def _process_episode(self, episode: Dict, episode_idx: int, 
                        batch_buffers: Dict) -> Optional[int]:
        """
        Process a single episode with mix view.
        
        Args:
            episode: Episode data
            episode_idx: Episode index
            batch_buffers: Batch buffers for each task
            
        Returns:
            task_id if successful, None otherwise
        """
        # Convert steps to list
        steps_dataset = episode['steps']
        steps_list = list(steps_dataset.as_numpy_iterator())
        
        if len(steps_list) < self.config.min_episode_length:
            self.stats['skipped_episodes'] += 1
            return None
            
        self.stats['total_episodes'] += 1
        
        # Extract language instruction from first step
        first_step = steps_list[0]
        language_instruction = None
        
        if 'language_instruction' in first_step:
            lang_data = first_step['language_instruction']
            if isinstance(lang_data, bytes):
                language_instruction = lang_data.decode('utf-8')
            elif isinstance(lang_data, str):
                language_instruction = lang_data
            elif hasattr(lang_data, 'numpy'):
                language_instruction = lang_data.numpy().decode('utf-8')
                
        # Match to task_id
        task_id = self._match_task_id(language_instruction)
        
        if task_id is None:
            task_id = episode_idx % 1001
            
        # Get or create collection
        collection_name = self._get_or_create_collection(task_id)
        
        # Process each step
        total_steps = len(steps_list)
        for step_idx, step in enumerate(steps_list):
            self.stats['total_steps'] += 1
            
            # Extract action
            action = step['action']
            if isinstance(action, tf.Tensor):
                action = action.numpy()
            action = np.array(action, dtype=np.float32)
            
            # Get next 3 actions
            next_actions = []
            for offset in range(1, 4):
                next_idx = step_idx + offset
                if next_idx < total_steps:
                    next_action = steps_list[next_idx]['action']
                    if isinstance(next_action, tf.Tensor):
                        next_action = next_action.numpy()
                    next_actions.append(next_action.tolist())
                else:
                    next_actions.append([0.0] * 7)
                    
            # Extract BOTH images (third-person and wrist)
            third_person_data = step['observation']['image']
            wrist_data = step['observation']['wrist_image']
            
            if isinstance(third_person_data, tf.Tensor):
                third_person_data = third_person_data.numpy()
            if isinstance(wrist_data, tf.Tensor):
                wrist_data = wrist_data.numpy()
                
            third_person_image = Image.fromarray(third_person_data)
            wrist_image = Image.fromarray(wrist_data)
            
            # Generate MIX embedding
            embedding = self._generate_mix_embedding(
                third_person_image, wrist_image, language_instruction
            )
            
            if embedding is None:
                self.stats['failed_embeddings'] += 1
                continue
                
            # Create Qdrant point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'dataset_name': self.config.dataset_name,
                    'episode_idx': episode_idx,
                    'step_idx': step_idx,
                    'current_action': action.tolist(),
                    'next_actions': next_actions,
                    'language_instruction': language_instruction or ""
                }
            )
            
            batch_buffers[task_id]['points'].append(point)
            
        return task_id
        
    def _generate_mix_embedding(
        self, 
        third_person_image: Image.Image, 
        wrist_image: Image.Image,
        instruction: Optional[str]
    ) -> Optional[List[float]]:
        """
        Generate mix embedding via mix embedding server.
        
        Args:
            third_person_image: Third-person view PIL Image
            wrist_image: Wrist view PIL Image
            instruction: Optional text instruction
            
        Returns:
            Mix embedding vector (4352 dims) or None if failed
        """
        try:
            # Prepare third-person image
            buf_third = BytesIO()
            third_person_image.save(buf_third, format='PNG')
            buf_third.seek(0)
            
            # Prepare wrist image
            buf_wrist = BytesIO()
            wrist_image.save(buf_wrist, format='PNG')
            buf_wrist.seek(0)
            
            files = {
                "third_person_image": ("third_person.png", buf_third, "image/png"),
                "wrist_image": ("wrist.png", buf_wrist, "image/png")
            }
            data = {
                "instruction": instruction if instruction else "",
                "return_individual": "false"
            }
            
            # Send request
            response = requests.post(
                self.config.embedding_server_url,
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            
            # Decode embedding
            result = response.json()
            
            if "mix_features" in result:
                b64_string = result["mix_features"]
                binary_data = base64.b64decode(b64_string)
                buffer = BytesIO(binary_data)
                tensor = torch.load(buffer, map_location="cpu")
                return tensor.squeeze(0).tolist()
            else:
                self.logger.warning("No mix_features in embedding response")
                return None
                
        except Exception as e:
            self.logger.error(f"Mix embedding generation failed: {e}")
            return None
            
    def _flush_batch(self, task_id: int, buffer: Dict):
        """
        Flush batch buffer to Qdrant.
        
        Args:
            task_id: Task ID
            buffer: Batch buffer containing points
        """
        if not buffer['points']:
            return
            
        collection_name = self._get_or_create_collection(task_id)
        
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=buffer['points']
            )
            self.logger.debug(f"Inserted {len(buffer['points'])} points to {collection_name}")
            buffer['points'].clear()
        except Exception as e:
            self.logger.error(f"Failed to insert batch to {collection_name}: {e}")
            
    def _print_statistics(self):
        """Print processing statistics"""
        self.logger.info("=" * 60)
        self.logger.info("Processing Statistics (Mix View):")
        self.logger.info(f"  Total episodes: {self.stats['total_episodes']}")
        self.logger.info(f"  Total steps: {self.stats['total_steps']}")
        self.logger.info(f"  Skipped episodes: {self.stats['skipped_episodes']}")
        self.logger.info(f"  Failed embeddings: {self.stats['failed_embeddings']}")
        self.logger.info(f"  Embedding dimension: {self.config.openvla_mix_dim}")
        self.logger.info("  Task distribution:")
        for task_id, count in sorted(self.stats['task_distribution'].items()):
            self.logger.info(f"    Task {task_id}: {count} episodes")
        self.logger.info("=" * 60)


def backup_mix_collections(config: ProcessingConfig, note: str = "mix_base"):
    """
    Backup all mix collections to the specified directory.
    
    Args:
        config: Processing configuration
        note: Backup folder name
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting backup of mix collections to {config.backup_dir}/{note}")
    
    try:
        client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            timeout=60.0
        )
        
        # Get all mix collections
        collections = client.get_collections().collections
        mix_prefixes = [cfg["collection_prefix"] for cfg in DATASET_CONFIGS.values()]
        
        mix_collections = [
            col.name for col in collections
            if any(col.name.startswith(prefix + "_task_") for prefix in mix_prefixes)
        ]
        
        if not mix_collections:
            logger.warning("No mix collections found to backup!")
            return
        
        logger.info(f"Found {len(mix_collections)} mix collections to backup")
        
        # Create backup directory
        backup_subdir = Path(config.backup_dir) / note
        if backup_subdir.exists():
            logger.info(f"Removing existing backup: {backup_subdir}")
            shutil.rmtree(backup_subdir)
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # Backup each collection
        success_count = 0
        for collection_name in tqdm(mix_collections, desc="Backing up collections"):
            try:
                # Create snapshot
                snapshot_info = client.create_snapshot(collection_name=collection_name)
                snapshot_name = snapshot_info.name
                
                # Download snapshot via REST API
                qdrant_url = f"http://{config.qdrant_host}:{config.qdrant_port}"
                download_url = f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}"
                
                response = requests.get(download_url, timeout=300)
                if response.status_code != 200:
                    logger.error(f"Failed to download snapshot for {collection_name}")
                    continue
                
                # Save to backup directory
                backup_file = backup_subdir / f"{collection_name}.snapshot"
                backup_file.write_bytes(response.content)
                
                # Delete snapshot from server
                client.delete_snapshot(collection_name=collection_name, snapshot_name=snapshot_name)
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to backup {collection_name}: {e}")
        
        # Create latest symlink
        latest_link = Path(config.backup_dir) / "latest_mix"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(note)
        
        logger.info(f"Backup completed: {success_count}/{len(mix_collections)} collections")
        logger.info(f"Backup location: {backup_subdir}")
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process LIBERO datasets for RT-Cache (Mix View: Third-Person + Wrist)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["goal", "10", "object", "spatial"],
        default="goal",
        help="LIBERO dataset type to process"
    )
    parser.add_argument(
        "--process_all",
        action="store_true",
        help="Process all 4 dataset types (goal, 10, object, spatial)"
    )
    parser.add_argument(
        "--base_dataset_path",
        type=str,
        default="/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551",
        help="Base path to LIBERO datasets directory"
    )
    parser.add_argument(
        "--embedding_server_url",
        type=str,
        default="http://127.0.0.1:9021/predict",
        help="URL of mix embedding server"
    )
    parser.add_argument(
        "--qdrant_host",
        type=str,
        default="localhost",
        help="Qdrant host"
    )
    parser.add_argument(
        "--qdrant_port",
        type=int,
        default=6333,
        help="Qdrant port"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for insertion"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="Maximum episodes to process (-1 for all)"
    )
    parser.add_argument(
        "--clear_db",
        action="store_true",
        help="Clear existing mix collections before processing"
    )
    parser.add_argument(
        "--clear_all",
        action="store_true",
        help="Clear ALL mix collections (all dataset types) before processing"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup collections after processing"
    )
    parser.add_argument(
        "--backup_name",
        type=str,
        default="mix_base",
        help="Backup folder name"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Determine which datasets to process
    if args.process_all:
        dataset_types = ["goal", "10", "object", "spatial"]
    else:
        dataset_types = [args.dataset_type]
    
    logger.info("=" * 60)
    logger.info("LIBERO Mix View Dataset Processing")
    logger.info("=" * 60)
    logger.info(f"Datasets to process: {dataset_types}")
    logger.info(f"Embedding server: {args.embedding_server_url}")
    logger.info(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    logger.info(f"Clear DB: {args.clear_db or args.clear_all}")
    logger.info(f"Backup after: {args.backup}")
    logger.info("=" * 60)
    
    # Clear all mix collections if requested
    if args.clear_all:
        temp_config = ProcessingConfig(
            dataset_type="goal",
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port
        )
        processor = LiberoMixDatasetProcessor(temp_config)
        processor.clear_all_mix_collections()
    
    # Process each dataset type
    for dt in dataset_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dt.upper()}")
        logger.info(f"{'='*60}")
        
        # Create configuration
        config = ProcessingConfig(
            dataset_type=dt,
            base_dataset_path=args.base_dataset_path,
            embedding_server_url=args.embedding_server_url,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            batch_size=args.batch_size,
            max_episodes=args.max_episodes,
            clear_db=args.clear_db and not args.clear_all,  # Don't clear again if clear_all was used
            backup_dir="/path/to/rtcache/scripts/retrieval/qdrant_backups",
            backup_name=args.backup_name
        )
        
        # Override with config file if available
        try:
            rt_config = get_config()
            config.qdrant_host = rt_config.database.qdrant_host
            config.qdrant_port = rt_config.database.qdrant_port
        except Exception as e:
            logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        # Apply command line overrides again
        config.qdrant_host = args.qdrant_host
        config.qdrant_port = args.qdrant_port
        
        logger.info(f"  Dataset name: {config.dataset_name}")
        logger.info(f"  Dataset path: {config.dataset_path}")
        logger.info(f"  Collection prefix: {config.collection_prefix}")
        logger.info(f"  Embedding dimension: {config.openvla_mix_dim}")
        
        # Create processor and run
        processor = LiberoMixDatasetProcessor(config)
        processor.process_dataset()
        
        logger.info(f"Completed processing: {dt.upper()}")
    
    # Backup after all processing
    if args.backup:
        logger.info("\n" + "=" * 60)
        logger.info("Creating backup...")
        logger.info("=" * 60)
        
        backup_config = ProcessingConfig(
            dataset_type="goal",
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            backup_dir="/path/to/rtcache/scripts/retrieval/qdrant_backups",
            backup_name=args.backup_name
        )
        backup_mix_collections(backup_config, args.backup_name)
    
    logger.info("\n" + "=" * 60)
    logger.info("All dataset processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
