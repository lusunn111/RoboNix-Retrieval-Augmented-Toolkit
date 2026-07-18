#!/usr/bin/env python3
"""
RT-Cache Retrieval Server for LIBERO (Mix View: Third-Person + Wrist)

This retrieval server is designed for LIBERO datasets stored in Qdrant
with mix view embeddings (third-person + wrist camera concatenated).

Collection naming: libero_{goal|10|object|spatial}_mix_task_{id}
Embedding dimension: 4352 (DINOv2 + SigLIP from both views)

Features:
- Accepts two images (third-person and wrist view)
- Uses mix embedding server for query embedding generation
- Returns action trajectories from similar samples

Author: RT-Cache Team
Date: 2024
"""

import os
import sys
import argparse
import logging
import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import requests
from PIL import Image
from flask import Flask, request, jsonify

from qdrant_client import QdrantClient
from qdrant_client.http import models


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


# Dataset configuration mapping (mix version)
DATASET_CONFIGS = {
    "goal": {
        "collection_prefix": "libero_goal_mix_task_"
    },
    "10": {
        "collection_prefix": "libero_10_mix_task_"
    },
    "object": {
        "collection_prefix": "libero_object_mix_task_"
    },
    "spatial": {
        "collection_prefix": "libero_spatial_mix_task_"
    },
}


def normalize_dataset_type(name: str) -> Optional[str]:
    """Normalize dataset type name"""
    if not name:
        return None
    n = name.lower().strip()
    if n.startswith("libero_"):
        n = n.replace("libero_", "", 1)
    if n.endswith("_no_noops"):
        n = n[:-9]
    if n.endswith("_mix"):
        n = n[:-4]
    return n if n in DATASET_CONFIGS else None


###############################################################################
# Configuration
###############################################################################
class RetrievalConfig:
    """Configuration for mix retrieval server"""
    
    # Server settings
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5003  # Different port from single-view server (5002)
    
    # Mix Embedding server
    EMBEDDING_URL = "http://127.0.0.1:9021/predict"
    
    # Qdrant settings
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    
    # Dataset types to serve
    DATASET_TYPES = ["goal"]
    
    # Retrieval parameters
    TOP_K = 10
    NUM_ACTIONS = 3
    SIMILARITY_THRESHOLD = 0.5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    LOG_LEVEL = "INFO"


###############################################################################
# Task Matching
###############################################################################
class TaskMatcher:
    """Match language instructions to task IDs using hash-based mapping"""
    
    def __init__(self):
        self.instruction_cache = {}
    
    def match(self, instruction: str) -> Optional[int]:
        """
        Match instruction to task_id using hash-based mapping.
        
        Args:
            instruction: Language instruction
            
        Returns:
            task_id based on MD5 hash
        """
        if not instruction:
            return None
        
        instruction_lower = instruction.lower().strip()
        
        if instruction_lower in self.instruction_cache:
            return self.instruction_cache[instruction_lower]
        
        instruction_hash = int(hashlib.md5(instruction_lower.encode('utf-8')).hexdigest(), 16)
        task_id = instruction_hash % 1001
        
        self.instruction_cache[instruction_lower] = task_id
        
        logging.info(f"Hash-based mapping: '{instruction}' -> task_id={task_id}")
        
        return task_id


###############################################################################
# Payload Cache
###############################################################################
class PayloadCache:
    """In-memory cache for all collection payloads"""
    
    def __init__(self, qdrant_client: QdrantClient, collection_names: List[str]):
        self.qdrant_client = qdrant_client
        self.collection_names = collection_names
        
        self.cache = {}
        self.stats = {
            "total_points": 0,
            "collections": {}
        }
        
        self._load_all_payloads(collection_names)
    
    def _load_all_payloads(self, collection_names: List[str]):
        """Load all payloads from specified collections into memory"""
        logging.info("Loading payloads into memory for specified collections...")
        for collection_name in collection_names:
            self._load_collection(collection_name)
        logging.info(f"Payload cache loaded: {self.stats['total_points']} total points across {len(self.cache)} collections")

    def _load_collection(self, collection_name: str):
        """Load a single collection's payloads into memory"""
        try:
            if collection_name in self.cache:
                return
            if not self.qdrant_client.collection_exists(collection_name):
                logging.warning(f"Collection {collection_name} does not exist, skipping load")
                return
            
            collection_info = self.qdrant_client.get_collection(collection_name)
            point_count = getattr(collection_info, 'points_count', None)
            if point_count is not None:
                logging.info(f"Loading {point_count} points from {collection_name}...")
            else:
                logging.info(f"Loading points from {collection_name}...")
            
            points_dict = {}
            offset = None
            batch_size = 100
            while True:
                records, offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                if not records:
                    break
                for record in records:
                    points_dict[str(record.id)] = record.payload
                if offset is None:
                    break
            
            self.cache[collection_name] = points_dict
            self.stats["collections"][collection_name] = len(points_dict)
            self.stats["total_points"] += len(points_dict)
            logging.info(f"Loaded {len(points_dict)} points from {collection_name}")
        except Exception as e:
            logging.error(f"Error loading payloads from {collection_name}: {e}")

    def ensure_collection_loaded(self, collection_name: str):
        """Ensure a collection is loaded into cache"""
        if collection_name not in self.cache:
            self._load_collection(collection_name)
    
    def get_payload(self, collection_name: str, point_id: str) -> Optional[Dict]:
        """Get payload for a point"""
        if collection_name not in self.cache:
            return None
        return self.cache[collection_name].get(point_id)
    
    def get_all_payloads(self, collection_name: str) -> Dict[str, Dict]:
        """Get all payloads for a collection"""
        return self.cache.get(collection_name, {})


###############################################################################
# Mix Embedding Generator
###############################################################################
class MixEmbeddingGenerator:
    """Generate mix embeddings via remote server"""
    
    def __init__(self, embedding_url: str):
        self.embedding_url = embedding_url
    
    def generate(
        self, 
        third_person_image: Image.Image, 
        wrist_image: Image.Image,
        instruction: str = ""
    ) -> Optional[torch.Tensor]:
        """
        Generate mix embedding from two views.
        
        Args:
            third_person_image: Third-person view image
            wrist_image: Wrist view image
            instruction: Optional text instruction
            
        Returns:
            Mix embedding tensor (4352 dims)
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
                "instruction": instruction,
                "return_individual": "false"
            }
            
            # Send request
            response = requests.post(
                self.embedding_url,
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
                return tensor.squeeze(0)
            else:
                logging.error("No mix_features in embedding response")
                return None
                
        except Exception as e:
            logging.error(f"Mix embedding generation failed: {e}")
            return None


###############################################################################
# Retrieval Engine
###############################################################################
class MixRetrievalEngine:
    """Main retrieval engine for mix view embeddings"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            timeout=60.0
        )
        
        # Initialize task matcher
        self.task_matcher = TaskMatcher()
        
        # Initialize mix embedding generator
        self.embedding_generator = MixEmbeddingGenerator(config.EMBEDDING_URL)
        
        # Normalize dataset types
        normalized_types = []
        for dt in config.DATASET_TYPES:
            ndt = normalize_dataset_type(dt)
            if ndt:
                normalized_types.append(ndt)
            else:
                logging.warning(f"Unknown dataset type '{dt}', skipping")
        self.dataset_types = normalized_types or ["goal"]
        
        # Discover and preload collections
        all_collections: List[str] = []
        for dt in self.dataset_types:
            prefix = DATASET_CONFIGS[dt]["collection_prefix"]
            found = self._discover_collections(prefix)
            all_collections.extend(found)
        
        # Initialize payload cache
        self.payload_cache = PayloadCache(self.qdrant_client, all_collections)
        
        logging.info(f"Mix retrieval engine initialized for datasets: {self.dataset_types}")

    def _discover_collections(self, prefix: str) -> List[str]:
        """Discover existing Qdrant collections with given prefix"""
        names: List[str] = []
        try:
            resp = self.qdrant_client.get_collections()
            coll_list = getattr(resp, 'collections', [])
            for c in coll_list:
                cname = getattr(c, 'name', None) or (c.get('name') if isinstance(c, dict) else None)
                if cname and cname.startswith(prefix):
                    names.append(cname)
        except Exception as e:
            logging.warning(f"Could not list collections from Qdrant: {e}")
        logging.info(f"Discovered {len(names)} collections with prefix '{prefix}'")
        return names

    def _search_points(self, collection_name: str, query_vector: List[float], limit: int = 10):
        """Version-agnostic Qdrant search"""
        if hasattr(self.qdrant_client, "search"):
            try:
                return self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logging.warning(f"Legacy search failed, will try query_points: {e}")

        try:
            result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            if hasattr(result, "points"):
                return result.points
            return result
        except Exception as e:
            logging.error(f"query_points fallback failed: {e}")
            raise
    
    def retrieve(
        self,
        third_person_image: Image.Image,
        wrist_image: Image.Image,
        instruction: str,
        dataset_type: Optional[str] = None,
        top_k: int = None
    ) -> Dict:
        """
        Retrieve similar samples and return action trajectory.
        
        Args:
            third_person_image: Third-person view image
            wrist_image: Wrist view image
            instruction: Language instruction
            dataset_type: Optional dataset type override
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with retrieval results
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        # Resolve dataset_type
        resolved_dt = normalize_dataset_type(dataset_type) if dataset_type else None
        if not resolved_dt:
            resolved_dt = self.dataset_types[0]
        if resolved_dt not in DATASET_CONFIGS:
            return {
                "success": False,
                "error": f"Unsupported dataset_type: {dataset_type}",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        prefix = DATASET_CONFIGS[resolved_dt]["collection_prefix"]
        
        # Match instruction to task
        if not instruction:
            return {
                "success": False,
                "error": "Instruction must be provided",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        task_id = self.task_matcher.match(instruction)
        
        if task_id is None:
            return {
                "success": False,
                "error": "Failed to map instruction to task_id",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        collection_name = f"{prefix}{task_id}"
        
        # Ensure collection is loaded
        if collection_name not in self.payload_cache.cache:
            self.payload_cache.ensure_collection_loaded(collection_name)
            if collection_name not in self.payload_cache.cache:
                return {
                    "success": False,
                    "error": f"Collection {collection_name} not found",
                    "rtcache_trajectory": None,
                    "averaged_trajectory": None
                }
        
        # Generate mix embedding
        embedding = self.embedding_generator.generate(
            third_person_image, wrist_image, instruction
        )
        
        if embedding is None:
            return {
                "success": False,
                "error": "Failed to generate mix embedding",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        # Search in Qdrant
        try:
            search_results = self._search_points(
                collection_name=collection_name,
                query_vector=embedding.tolist(),
                limit=top_k,
            )
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return {
                "success": False,
                "error": f"Search failed: {e}",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        if not search_results:
            return {
                "success": False,
                "error": "No results found",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        # Get payloads from cache
        results = []
        for result in search_results:
            point_id = str(result.id)
            
            payload = self.payload_cache.get_payload(collection_name, point_id)
            
            if payload is None:
                payload = getattr(result, "payload", None)
                if payload:
                    try:
                        self.payload_cache.cache.setdefault(collection_name, {})[point_id] = payload
                    except Exception:
                        pass
            
            if payload is None and hasattr(self.qdrant_client, "retrieve"):
                try:
                    retrieved = self.qdrant_client.retrieve(
                        collection_name=collection_name,
                        ids=[result.id],
                        with_payload=True,
                        with_vectors=False,
                    )
                    if retrieved:
                        payload = retrieved[0].payload
                        try:
                            self.payload_cache.cache.setdefault(collection_name, {})[point_id] = payload
                        except Exception:
                            pass
                except Exception:
                    payload = None
            
            if payload:
                results.append({
                    "id": point_id,
                    "score": result.score,
                    "payload": payload,
                })
        
        if not results:
            return {
                "success": False,
                "error": "No payloads found in cache",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        # Extract actions
        top_result = results[0]
        current_action = top_result["payload"].get("current_action", [])
        next_actions = top_result["payload"].get("next_actions", [])
        
        trajectory = [current_action] + next_actions
        
        # Compute averaged trajectory
        all_trajectories = []
        for r in results[:min(1, len(results))]:
            traj = [r["payload"].get("current_action", [])] + r["payload"].get("next_actions", [])
            all_trajectories.append(traj)
        
        if all_trajectories:
            try:
                all_traj_array = np.array(all_trajectories)
                averaged_traj = np.mean(all_traj_array, axis=0)
                averaged_trajectory = averaged_traj.tolist()
            except:
                averaged_trajectory = trajectory
        else:
            averaged_trajectory = trajectory
        
        return {
            "success": True,
            "task_id": task_id,
            "collection_name": collection_name,
            "top_score": results[0]["score"],
            "num_results": len(results),
            "rtcache_trajectory": trajectory,
            "averaged_trajectory": averaged_trajectory,
            "metadata": {
                "episode_idx": top_result["payload"].get("episode_idx"),
                "step_idx": top_result["payload"].get("step_idx"),
                "dataset_name": top_result["payload"].get("dataset_name"),
                "language_instruction": top_result["payload"].get("language_instruction")
            }
        }


###############################################################################
# Flask Server
###############################################################################
def create_app(config: RetrievalConfig) -> Flask:
    """Create Flask application"""
    
    app = Flask(__name__)
    
    # Initialize retrieval engine
    engine = MixRetrievalEngine(config)
    
    @app.route("/pipeline", methods=["POST"])
    def pipeline():
        """
        Main retrieval endpoint for mix view.
        
        Expects:
        - third_person_image: Third-person view image file
        - wrist_image: Wrist view image file
        - instruction: Language instruction (required)
        - dataset_type: Optional dataset type (goal, 10, object, spatial)
        """
        try:
            # Get third-person image
            if 'third_person_image' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No third_person_image provided"
                }), 400
            
            # Get wrist image
            if 'wrist_image' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No wrist_image provided"
                }), 400
            
            third_person_file = request.files['third_person_image']
            wrist_file = request.files['wrist_image']
            
            third_person_image = Image.open(third_person_file.stream).convert('RGB')
            wrist_image = Image.open(wrist_file.stream).convert('RGB')
            
            # Get instruction
            instruction = request.form.get('instruction', '')
            dataset_type = request.form.get('dataset_type', '')
            
            if not instruction:
                return jsonify({
                    "success": False,
                    "error": "Instruction must be provided"
                }), 400
            
            # Retrieve
            result = engine.retrieve(
                third_person_image, wrist_image, instruction, 
                dataset_type=dataset_type
            )
            
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error in pipeline: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e),
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }), 500
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "view_type": "mix",
            "embedding_dim": 4352,
            "collections": len(engine.payload_cache.cache),
            "total_points": engine.payload_cache.stats["total_points"]
        })
    
    @app.route("/stats", methods=["GET"])
    def stats():
        """Statistics endpoint"""
        return jsonify(engine.payload_cache.stats)
    
    return app


###############################################################################
# Main
###############################################################################
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RT-Cache Mix Retrieval Server for LIBERO (Third-Person + Wrist View)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=RetrievalConfig.SERVER_HOST,
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=RetrievalConfig.SERVER_PORT,
        help="Server port (default: 5003)"
    )
    parser.add_argument(
        "--embedding-url",
        type=str,
        default=RetrievalConfig.EMBEDDING_URL,
        help="Mix embedding server URL"
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=RetrievalConfig.QDRANT_HOST,
        help="Qdrant host"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=RetrievalConfig.QDRANT_PORT,
        help="Qdrant port"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=RetrievalConfig.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        default="goal",
        help="Comma-separated dataset types to serve (goal,object,spatial,10). Use 'all' to load all."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Update config
    config = RetrievalConfig()
    config.SERVER_HOST = args.host
    config.SERVER_PORT = args.port
    config.EMBEDDING_URL = args.embedding_url
    config.QDRANT_HOST = args.qdrant_host
    config.QDRANT_PORT = args.qdrant_port
    
    if args.dataset_types.lower() == "all":
        config.DATASET_TYPES = list(DATASET_CONFIGS.keys())
    else:
        config.DATASET_TYPES = [dt.strip() for dt in args.dataset_types.split(',') if dt.strip()]
    
    logging.info("=" * 60)
    logging.info("RT-Cache Mix Retrieval Server")
    logging.info("=" * 60)
    logging.info(f"Server: {config.SERVER_HOST}:{config.SERVER_PORT}")
    logging.info(f"Mix Embedding URL: {config.EMBEDDING_URL}")
    logging.info(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    logging.info(f"Datasets: {config.DATASET_TYPES}")
    logging.info(f"Device: {config.DEVICE}")
    logging.info(f"Embedding dim: 4352 (mix view)")
    logging.info("=" * 60)
    
    # Create and run app
    app = create_app(config)
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=False,
        threaded=True
    )


if __name__ == "__main__":
    main()
