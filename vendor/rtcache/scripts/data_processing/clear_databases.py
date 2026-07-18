#!/usr/bin/env python3
"""
Script to clear all data from MongoDB and Qdrant databases.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "config"))

from pymongo import MongoClient
from qdrant_client import QdrantClient
from config.rt_cache_config import get_config

def clear_databases():
    config = get_config()
    
    print("WARNING: This will delete ALL data from the configured databases.")
    print(f"MongoDB: {config.database.mongo_db_name}")
    print(f"Qdrant Host: {config.database.qdrant_host}:{config.database.qdrant_port}")
    
    confirm = input("Are you sure you want to continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return

    # Clear MongoDB
    try:
        print(f"Connecting to MongoDB at {config.database.mongo_url}...")
        mongo_client = MongoClient(config.database.mongo_url)
        db_name = config.database.mongo_db_name
        
        if db_name in mongo_client.list_database_names():
            mongo_client.drop_database(db_name)
            print(f"Dropped MongoDB database: {db_name}")
        else:
            print(f"MongoDB database {db_name} does not exist.")
            
    except Exception as e:
        print(f"Error clearing MongoDB: {e}")

    # Clear Qdrant
    try:
        print(f"Connecting to Qdrant at {config.database.qdrant_host}:{config.database.qdrant_port}...")
        qdrant_client = QdrantClient(
            host=config.database.qdrant_host, 
            port=config.database.qdrant_port
        )
        
        # List of collections to delete
        # Note: These names should match what's used in process_datasets.py
        collections = [
            "image_collection", 
            "text_collection", 
            "clip_image_collection", 
            "clip_text_collection"
        ]
        
        # Also check for dataset-specific collections if any
        # (The current process_datasets.py seems to use fixed names, but let's be safe)
        response = qdrant_client.get_collections()
        all_collections = [c.name for c in response.collections]
        
        for col_name in all_collections:
            qdrant_client.delete_collection(col_name)
            print(f"Deleted Qdrant collection: {col_name}")
            
    except Exception as e:
        print(f"Error clearing Qdrant: {e}")

    print("Database clearing complete.")

if __name__ == "__main__":
    clear_databases()
