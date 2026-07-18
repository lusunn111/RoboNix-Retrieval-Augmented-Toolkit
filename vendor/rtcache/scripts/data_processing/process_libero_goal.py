#!/usr/bin/env python3
"""
Process LIBERO Datasets for RT-Cache System

This script processes LIBERO datasets (goal/10/object/spatial).
For each task, it creates a separate Qdrant collection and stores:
- Image embeddings (OpenVLA features)
- Metadata: dataset_name, episode_idx, step_idx, current_action, next_actions (next 3 steps)

Supported datasets:
- libero_goal: LIBERO-Goal (10 tasks)
- libero_10: LIBERO-10 (10 tasks)
- libero_object: LIBERO-Object (10 tasks)
- libero_spatial: LIBERO-Spatial (10 tasks)

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

# Try to import LIBERO benchmark
try:
    from libero.libero import benchmark
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("Warning: LIBERO benchmark not available. Task matching will use language_instruction only.")


# Dataset configuration mapping
DATASET_CONFIGS = {
    "goal": {
        "folder_name": "libero_goal_no_noops",
        "dataset_name": "libero_goal_no_noops",
        "task_suite_name": "libero_goal",
        "collection_prefix": "libero_goal"
    },
    "10": {
        "folder_name": "libero_10_no_noops",
        "dataset_name": "libero_10_no_noops",
        "task_suite_name": "libero_10",
        "collection_prefix": "libero_10"
    },
    "object": {
        "folder_name": "libero_object_no_noops",
        "dataset_name": "libero_object_no_noops",
        "task_suite_name": "libero_object",
        "collection_prefix": "libero_object"
    },
    "spatial": {
        "folder_name": "libero_spatial_no_noops",
        "dataset_name": "libero_spatial_no_noops",
        "task_suite_name": "libero_spatial",
        "collection_prefix": "libero_spatial"
    }
}


def setup_logging(level="INFO"):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@dataclass
class ProcessingConfig:
    """Configuration for LIBERO dataset processing"""
    
    # Dataset type: goal, 10, object, or spatial
    dataset_type: str = "goal"
    
    # Base dataset directory
    base_dataset_path: str = "/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
    
    # These will be set automatically based on dataset_type
    dataset_path: str = ""
    dataset_name: str = ""
    task_suite_name: str = ""
    collection_prefix: str = ""
    
    # Server URLs
    embedding_server_url: str = "http://127.0.0.1:9020/predict"
    
    # Database settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Processing parameters
    batch_size: int = 50
    max_episodes: int = -1  # -1 means all episodes
    min_episode_length: int = 5

    # Whether to rely on LIBERO benchmark task IDs (disabled per user request)
    use_benchmark: bool = False
    
    # Embedding dimensions
    openvla_image_dim: int = 2176
    
    def __post_init__(self):
        """Initialize dataset-specific configuration based on dataset_type"""
        if self.dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}. Must be one of {list(DATASET_CONFIGS.keys())}")
        
        config = DATASET_CONFIGS[self.dataset_type]
        self.dataset_name = config["dataset_name"]
        self.task_suite_name = config["task_suite_name"]
        self.collection_prefix = config["collection_prefix"]
        
        # Construct full dataset path
        if not self.dataset_path:  # Only set if not already specified
            self.dataset_path = f"{self.base_dataset_path}/{config['folder_name']}/1.0.0"


class LiberoDatasetProcessor:
    """
    Processor for LIBERO datasets (goal/10/object/spatial).
    
    Creates separate Qdrant collections for each task and stores:
    - Image embeddings
    - Action sequences (current + next 3 steps)
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the processor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self._init_storage()
        
        # Initialize task mapping
        self._init_task_mapping()
        
        # Statistics
        self.stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "skipped_episodes": 0,
            "failed_embeddings": 0,
            "task_distribution": defaultdict(int)
        }
        
        # Track which instructions have been seen (to reduce log noise)
        self.seen_instructions = set()
        
    def _init_storage(self):
        """Initialize Qdrant connection"""
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            timeout=60.0
        )
        self.logger.info(f"Connected to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}")
        
    def _init_task_mapping(self):
        """Initialize task mapping from LIBERO benchmark"""
        self.task_descriptions = {}
        self.task_id_to_collection = {}
        self.instruction_to_task_id = {}  # Cache for hash-based assignment
        if self.config.use_benchmark and LIBERO_AVAILABLE:
            try:
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict[self.config.task_suite_name]()
                num_tasks = task_suite.n_tasks
                
                self.logger.info(f"Found {num_tasks} tasks in {self.config.task_suite_name}")
                
                for task_id in range(num_tasks):
                    task = task_suite.get_task(task_id)
                    task_description = task.language
                    self.task_descriptions[task_description.lower().strip()] = task_id
                    
                    # Create collection name: {prefix}_task_0, {prefix}_task_1, etc.
                    collection_name = f"{self.config.collection_prefix}_task_{task_id}"
                    self.task_id_to_collection[task_id] = collection_name
                    
                    # Create collection if it doesn't exist
                    if not self.qdrant_client.collection_exists(collection_name):
                        self.qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=self.config.openvla_image_dim,
                                distance="Cosine"
                            )
                        )
                        self.logger.info(f"Created collection: {collection_name}")
                    else:
                        self.logger.info(f"Collection {collection_name} already exists")
                        
                self.logger.info(f"Initialized {len(self.task_descriptions)} task mappings")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize LIBERO benchmark: {e}")
                self.logger.warning("Will use language_instruction matching only")
        else:
            self.logger.info("Using hash-based task IDs (benchmark disabled). Collections created on-the-fly.")
            
    def _match_task_id(self, language_instruction: str) -> Optional[int]:
        """
        Match language instruction to task_id.
        
        Args:
            language_instruction: The language instruction from the dataset
            
        Returns:
            task_id if matched, None otherwise
        """
        if not language_instruction:
            return None
            
        instruction_lower = language_instruction.lower().strip()
        
        # Direct match
        if instruction_lower in self.task_descriptions:
            return self.task_descriptions[instruction_lower]
        
        # Fuzzy match: check if any task description is contained in the instruction
        for task_desc, task_id in self.task_descriptions.items():
            if task_desc in instruction_lower or instruction_lower in task_desc:
                return task_id
                
        return None
        
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
        
        # Create collection name on-the-fly
        collection_name = f"{self.config.collection_prefix}_task_{task_id}"
        
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.openvla_image_dim,
                    distance="Cosine"
                )
            )
            self.logger.info(f"Created collection on-the-fly: {collection_name}")
        
        self.task_id_to_collection[task_id] = collection_name
        return collection_name
        
    def process_dataset(self):
        """Process the entire LIBERO-Goal dataset"""
        self.logger.info(f"Loading dataset from: {self.config.dataset_path}")
        
        # Load dataset
        builder = tfds.builder_from_directory(builder_dir=self.config.dataset_path)
        ds = builder.as_dataset(split='train', shuffle_files=False)
        
        # Get total number of episodes
        total_episodes = builder.info.splits['train'].num_examples
        self.logger.info(f"Total episodes in dataset: {total_episodes}")
        
        # Process episodes
        batch_buffers = defaultdict(lambda: {
            'points': []
        })
        
        episode_count = 0
        for episode_idx, episode in enumerate(tqdm(ds, desc="Processing episodes")):
            if self.config.max_episodes > 0 and episode_idx >= self.config.max_episodes:
                break
                
            try:
                # Process episode
                task_id = self._process_episode(
                    episode, episode_idx, batch_buffers
                )
                
                if task_id is not None:
                    episode_count += 1
                    self.stats['task_distribution'][task_id] += 1
                    
                    # Flush batch if needed (check only the current task's buffer)
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
        Process a single episode.
        
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
            # If no match, use hash-based assignment to ensure same instruction always gets same task_id
            # This is a fallback for when LIBERO benchmark is not available
            if language_instruction:
                # Check if we've already assigned this instruction
                if language_instruction in self.instruction_to_task_id:
                    task_id = self.instruction_to_task_id[language_instruction]
                else:
                    # Use hash of instruction to consistently assign task_id (0-1000)
                    import hashlib
                    instruction_hash = int(hashlib.md5(language_instruction.encode('utf-8')).hexdigest(), 16)
                    task_id = instruction_hash % 1001  # Use 0-1000 as task_id range
                    self.instruction_to_task_id[language_instruction] = task_id
                    
                    # Only log once per unique instruction to reduce log noise
                    if language_instruction not in self.seen_instructions:
                        self.seen_instructions.add(language_instruction)
                        self.logger.info(f"Using hash-based task_id={task_id} for instruction: '{language_instruction}' (LIBERO benchmark not available)")
            else:
                # Fallback to episode_idx if no instruction (keep a wider range to reduce collisions)
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
            
            # Get next 3 actions (if available)
            next_actions = []
            for offset in range(1, 4):  # next 1, 2, 3 steps
                next_idx = step_idx + offset
                if next_idx < total_steps:
                    next_action = steps_list[next_idx]['action']
                    if isinstance(next_action, tf.Tensor):
                        next_action = next_action.numpy()
                    next_actions.append(next_action.tolist())
                else:
                    # Pad with zeros if not available
                    next_actions.append([0.0] * 7)
                    
            # Extract image
            image_data = step['observation']['image']
            if isinstance(image_data, tf.Tensor):
                image_data = image_data.numpy()
            image = Image.fromarray(image_data)
            
            # Generate embedding
            embedding = self._generate_embedding(image, language_instruction)
            
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
                    'next_actions': next_actions,  # List of 3 actions
                    'language_instruction': language_instruction or ""
                }
            )
            
            batch_buffers[task_id]['points'].append(point)
            
        return task_id
        
    def _generate_embedding(self, image: Image.Image, instruction: Optional[str]) -> Optional[List[float]]:
        """
        Generate image embedding via embedding server.
        
        Args:
            image: PIL Image
            instruction: Optional text instruction
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Prepare request
            buf = BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            
            files = {"file": ("image.png", buf, "image/png")}
            data = {
                "instruction": instruction if instruction else "",
                "option": "image"  # Get image embeddings only
            }
            
            # Send request
            response = requests.post(
                self.config.embedding_server_url,
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            
            # Decode embedding
            result = response.json()
            
            if "image_features" in result:
                b64_string = result["image_features"]
                binary_data = base64.b64decode(b64_string)
                buffer = BytesIO(binary_data)
                tensor = torch.load(buffer, map_location="cpu")
                return tensor.squeeze(0).tolist()
            else:
                self.logger.warning("No image_features in embedding response")
                return None
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
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
        self.logger.info("Processing Statistics:")
        self.logger.info(f"  Total episodes: {self.stats['total_episodes']}")
        self.logger.info(f"  Total steps: {self.stats['total_steps']}")
        self.logger.info(f"  Skipped episodes: {self.stats['skipped_episodes']}")
        self.logger.info(f"  Failed embeddings: {self.stats['failed_embeddings']}")
        self.logger.info("  Task distribution:")
        for task_id, count in sorted(self.stats['task_distribution'].items()):
            self.logger.info(f"    Task {task_id}: {count} episodes")
        self.logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process LIBERO datasets for RT-Cache"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["goal", "10", "object", "spatial"],
        default="goal",
        help="LIBERO dataset type to process"
    )
    parser.add_argument(
        "--base_dataset_path",
        type=str,
        default="/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551",
        help="Base path to LIBERO datasets directory"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Full path to specific dataset (overrides base_dataset_path and dataset_type)"
    )
    parser.add_argument(
        "--embedding_server_url",
        type=str,
        default="http://127.0.0.1:9020/predict",
        help="URL of embedding server"
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
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--use_benchmark",
        action="store_true",
        help="Use LIBERO benchmark task IDs (default off; hash-based IDs when disabled)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = ProcessingConfig(
        dataset_type=args.dataset_type,
        base_dataset_path=args.base_dataset_path,
        dataset_path=args.dataset_path,
        embedding_server_url=args.embedding_server_url,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        use_benchmark=args.use_benchmark
    )
    
    # Override with config file if available
    try:
        rt_config = get_config()
        config.embedding_server_url = rt_config.server.embedding_url
        config.qdrant_host = rt_config.database.qdrant_host
        config.qdrant_port = rt_config.database.qdrant_port
        config.openvla_image_dim = rt_config.retrieval.openvla_dim
    except Exception as e:
        logger.warning(f"Could not load config file: {e}. Using defaults.")
    
    logger.info(f"Processing LIBERO dataset: {config.dataset_type.upper()}")
    logger.info(f"  Dataset name: {config.dataset_name}")
    logger.info(f"  Dataset path: {config.dataset_path}")
    logger.info(f"  Task suite: {config.task_suite_name}")
    logger.info(f"  Collection prefix: {config.collection_prefix}")
    logger.info(f"  Embedding server: {config.embedding_server_url}")
    logger.info(f"  Qdrant: {config.qdrant_host}:{config.qdrant_port}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Use benchmark task IDs: {config.use_benchmark}")
    
    # Create processor and run
    processor = LiberoDatasetProcessor(config)
    processor.process_dataset()
    
    logger.info("Dataset processing complete!")


if __name__ == "__main__":
    main()

