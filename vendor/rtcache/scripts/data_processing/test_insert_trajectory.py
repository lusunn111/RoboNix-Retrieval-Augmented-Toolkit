#!/usr/bin/env python3
import os
import sys
import time
import uuid
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from io import BytesIO
import requests
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from pathlib import Path
import base64
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import get_config

def main():
    config = get_config()
    
    # 1. Connect to Databases
    print("Connecting to databases...")
    mongo_client = MongoClient(config.database.mongo_url)
    mongo_db = mongo_client[config.database.mongo_db_name]
    mongo_collection = mongo_db["trajectories"]
    
    qdrant_client = QdrantClient(
        host=config.database.qdrant_host,
        port=config.database.qdrant_port
    )
    
    # 2. Load Dataset
    print("Loading dataset...")
    # Use the path from the user's notebook
    db_dir = '/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_goal_no_noops/1.0.0'
    builder = tfds.builder_from_directory(db_dir)
    ds = builder.as_dataset(split='train[:1]') # Take 1 episode
    
    dataset_name = "libero_goal_no_noops"
    
    # 3. Process Episode
    for episode_idx, episode in enumerate(ds):
        print(f"Processing episode {episode_idx}...")
        # Convert to numpy iterator to get actual data
        steps = list(episode['steps'].as_numpy_iterator())
        total_steps = len(steps)
        
        # --- Extract Text (Fix 1) ---
        # Text is in the step directly, not in observation
        first_step = steps[0]
        if 'language_instruction' in first_step:
            episode_text = first_step['language_instruction'].decode('utf-8')
        else:
            episode_text = ""
        print(f"  Instruction: {episode_text}")
        
        batch_points_image = []
        batch_points_clip = []
        mongo_docs = []
        
        for step_idx, step in enumerate(steps):
            # --- Extract Action (Fix 2) ---
            # Action is a numpy array (7,)
            raw_action = step['action'] 
            
            # Check if it's all zeros (just to see)
            if np.all(raw_action == 0):
                print(f"  Warning: Action at step {step_idx} is all zeros.")
            
            # Normalize action
            normalized_action = raw_action.copy()
            normalized_action[:3] = np.clip(normalized_action[:3], -0.1, 0.1)
            normalized_action[3:6] = np.clip(normalized_action[3:6], -0.5, 0.5)
            normalized_action[6] = 1.0 if normalized_action[6] > 0 else 0.0
            
            # --- Extract Image ---
            image_data = step['observation']['image']
            image = Image.fromarray(image_data)
            
            # Save Image
            doc_id = f"{dataset_name}_test_{episode_idx}_{step_idx}"
            image_dir = Path(config.paths.image_storage_path)
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{doc_id}.png"
            image.save(image_path)
            
            # --- Generate Embeddings ---
            buf = BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            files = {"file": ("image.png", buf, "image/png")}
            data = {"instruction": episode_text, "option": "image"} 
            
            try:
                resp = requests.post(config.server.embedding_url, files=files, data=data, timeout=30)
                resp.raise_for_status()
                emb_result = resp.json()
            except Exception as e:
                print(f"  Error getting embedding for step {step_idx}: {e}")
                continue

            # --- Prepare DB Records ---
            
            # MongoDB Document
            mongo_doc = {
                'id': doc_id,
                'dataset_name': dataset_name,
                'episode_idx': episode_idx,
                'step_idx': step_idx,
                'total_steps': total_steps,
                'raw_action': raw_action.tolist(),
                'normalized_action': normalized_action.tolist(),
                'text': episode_text,
                'image_path': str(image_path)
            }
            mongo_docs.append(mongo_doc)
            
            # Qdrant Points
            if "image_features" in emb_result:
                b64 = emb_result["image_features"]
                tensor = torch.load(BytesIO(base64.b64decode(b64)), map_location="cpu")
                vector = tensor.squeeze(0).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        'logical_id': doc_id,
                        'dataset_name': dataset_name,
                        'episode_idx': episode_idx,
                        'step_idx': step_idx,
                        'text': episode_text
                    }
                )
                batch_points_image.append(point)

            if "clip_image_features" in emb_result:
                b64 = emb_result["clip_image_features"]
                tensor = torch.load(BytesIO(base64.b64decode(b64)), map_location="cpu")
                vector = tensor.squeeze(0).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        'logical_id': doc_id,
                        'dataset_name': dataset_name,
                        'episode_idx': episode_idx,
                        'step_idx': step_idx,
                        'text': episode_text
                    }
                )
                batch_points_clip.append(point)
                
            if step_idx % 10 == 0:
                print(f"  Processed step {step_idx}/{total_steps}")

        # --- Insert Batch ---
        if mongo_docs:
            print(f"Inserting {len(mongo_docs)} documents to MongoDB...")
            mongo_collection.insert_many(mongo_docs)
            
        if batch_points_image:
            print(f"Inserting {len(batch_points_image)} points to Qdrant (image_collection)...")
            qdrant_client.upsert(
                collection_name="image_collection",
                points=batch_points_image
            )
            
        if batch_points_clip:
            print(f"Inserting {len(batch_points_clip)} points to Qdrant (clip_image_collection)...")
            qdrant_client.upsert(
                collection_name="clip_image_collection",
                points=batch_points_clip
            )
            
    print("Done! You can now check the database for records with id starting with 'libero_goal_no_noops_test_'")

if __name__ == "__main__":
    main()
