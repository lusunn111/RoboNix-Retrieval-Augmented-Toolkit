#!/usr/bin/env python3
"""
Backup Qdrant collections for LIBERO experiments

This script creates snapshots of all LIBERO collections
(goal, spatial, object, 10) and saves them to a backup directory.

Usage:
    python backup_qdrant.py [--backup-dir <path>] [--qdrant-host <host>] [--qdrant-port <port>]
"""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_libero_collections(client: QdrantClient) -> List[str]:
    """Get all libero_*_task_* collection names (goal, spatial, object, 10)"""
    all_collections = client.get_collections().collections
    
    # Match all LIBERO dataset types
    libero_prefixes = [
        "libero_goal_task_",
        "libero_spatial_task_",
        "libero_object_task_",
        "libero_10_task_"
    ]
    
    libero_collections = [
        col.name for col in all_collections 
        if any(col.name.startswith(prefix) for prefix in libero_prefixes)
    ]
    
    # Count by type
    counts = {}
    for prefix in libero_prefixes:
        count = sum(1 for col in libero_collections if col.startswith(prefix))
        if count > 0:
            dataset_name = prefix.replace("libero_", "").replace("_task_", "")
            counts[dataset_name] = count
    
    logging.info(f"Found {len(libero_collections)} LIBERO collections: {counts}")
    return libero_collections


def backup_collection(client: QdrantClient, collection_name: str, backup_dir: Path) -> bool:
    """
    Create snapshot for a single collection and save to backup directory
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection to backup
        backup_dir: Directory to save the snapshot
        
    Returns:
        True if backup successful, False otherwise
    """
    try:
        logging.info(f"Creating snapshot for {collection_name}...")
        
        # Create snapshot using Qdrant API
        snapshot_info = client.create_snapshot(collection_name=collection_name)
        snapshot_name = snapshot_info.name
        
        logging.info(f"Snapshot created: {snapshot_name}")
        
        # Download snapshot using REST API
        import requests
        
        # Get Qdrant URL
        qdrant_url = f"http://{client._client._host}:{client._client._port}"
        download_url = f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}"
        
        logging.info(f"Downloading snapshot from {download_url}...")
        response = requests.get(download_url, timeout=300)
        
        if response.status_code != 200:
            logging.error(f"Failed to download snapshot: HTTP {response.status_code}")
            return False
        
        snapshot_data = response.content
        
        # Save to backup directory
        backup_file = backup_dir / f"{collection_name}.snapshot"
        backup_file.write_bytes(snapshot_data)
        
        logging.info(f"Backup saved to {backup_file} ({len(snapshot_data)} bytes)")
        
        # Delete the snapshot from Qdrant server to save space
        client.delete_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name
        )
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to backup {collection_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Backup Qdrant collections for LIBERO (goal, spatial, object, 10)"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./qdrant_backups",
        help="Directory to save backups (default: ./qdrant_backups)"
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)"
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note to add to backup directory name (e.g., 'base+5')"
    )
    
    args = parser.parse_args()
    
    # Create backup directory
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Use fixed naming based on note (no timestamp)
    # This allows overwriting previous backups with same note
    if args.note:
        backup_subdir = backup_dir / f"backup_{args.note}"
    else:
        # If no note provided, use timestamp to avoid overwriting
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / f"backup_{timestamp}"
    
    # Remove existing backup with same name if it exists
    if backup_subdir.exists():
        logging.info(f"Removing existing backup: {backup_subdir}")
        shutil.rmtree(backup_subdir)
    
    backup_subdir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("Qdrant Backup Script")
    logging.info("=" * 60)
    logging.info(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    logging.info(f"Backup directory: {backup_subdir}")
    if args.note:
        logging.info(f"Note: {args.note}")
    logging.info("=" * 60)
    
    # Connect to Qdrant
    try:
        client = QdrantClient(
            host=args.qdrant_host,
            port=args.qdrant_port,
            timeout=60.0
        )
        logging.info("Connected to Qdrant")
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant: {e}")
        return 1
    
    # Get all LIBERO collections
    collections = get_libero_collections(client)
    
    if not collections:
        logging.warning("No LIBERO collections found!")
        return 0
    
    # Backup each collection
    success_count = 0
    failed_collections = []
    
    for collection_name in collections:
        if backup_collection(client, collection_name, backup_subdir):
            success_count += 1
        else:
            failed_collections.append(collection_name)
    
    # Summary
    logging.info("=" * 60)
    logging.info(f"Backup completed: {success_count}/{len(collections)} collections")
    
    if failed_collections:
        logging.warning(f"Failed collections: {', '.join(failed_collections)}")
    
    # Create a "latest" symlink for easy access
    latest_link = backup_dir / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(backup_subdir.name)
    
    logging.info(f"Latest backup: {latest_link} -> {backup_subdir.name}")
    logging.info("=" * 60)
    
    return 0 if success_count == len(collections) else 1


if __name__ == "__main__":
    exit(main())
