#!/usr/bin/env python3
"""
Restore Qdrant collections from backup

This script restores LIBERO collections (goal, spatial, object, 10) from snapshot backups.
It will delete existing collections and restore from the backup.

Usage:
    python restore_qdrant.py [--backup-dir <path>] [--qdrant-host <host>] [--qdrant-port <port>]
"""

import argparse
import logging
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


def get_backup_files(backup_dir: Path) -> List[Path]:
    """Get all .snapshot files from backup directory"""
    snapshot_files = list(backup_dir.glob("*.snapshot"))
    logging.info(f"Found {len(snapshot_files)} snapshot files in {backup_dir}")
    return snapshot_files


def restore_collection(client: QdrantClient, snapshot_file: Path) -> bool:
    """
    Restore a single collection from snapshot file
    
    Args:
        client: Qdrant client
        snapshot_file: Path to snapshot file
        
    Returns:
        True if restore successful, False otherwise
    """
    # Extract collection name from filename
    collection_name = snapshot_file.stem  # Remove .snapshot extension
    
    try:
        logging.info(f"Restoring {collection_name}...")
        
        # Check if collection exists and delete it
        try:
            client.get_collection(collection_name)
            logging.info(f"Deleting existing collection {collection_name}...")
            client.delete_collection(collection_name)
            time.sleep(0.5)  # Give Qdrant time to clean up
        except Exception:
            logging.info(f"Collection {collection_name} does not exist, creating new...")
        
        # Read snapshot data
        snapshot_data = snapshot_file.read_bytes()
        logging.info(f"Snapshot size: {len(snapshot_data)} bytes")
        
        # Upload snapshot to Qdrant
        # Note: We need to use the REST API for this
        # The Python client doesn't have a direct upload method
        import requests
        
        # Get Qdrant URL from client
        qdrant_url = f"http://{client._client._host}:{client._client._port}"
        
        # Upload snapshot
        upload_url = f"{qdrant_url}/collections/{collection_name}/snapshots/upload"
        response = requests.post(
            upload_url,
            files={"snapshot": snapshot_data},
            timeout=300  # 5 minutes timeout for large snapshots
        )
        
        if response.status_code == 200:
            logging.info(f"Successfully restored {collection_name}")
            return True
        else:
            logging.error(f"Failed to restore {collection_name}: HTTP {response.status_code}")
            logging.error(f"Response: {response.text}")
            return False
        
    except Exception as e:
        logging.error(f"Failed to restore {collection_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Restore Qdrant collections from backup"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./qdrant_backups/latest",
        help="Directory containing backup files (default: ./qdrant_backups/latest)"
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
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Resolve backup directory (handles symlinks)
    backup_dir = Path(args.backup_dir).resolve()
    
    if not backup_dir.exists():
        logging.error(f"Backup directory does not exist: {backup_dir}")
        return 1
    
    logging.info("=" * 60)
    logging.info("Qdrant Restore Script")
    logging.info("=" * 60)
    logging.info(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    logging.info(f"Backup directory: {backup_dir}")
    logging.info("=" * 60)
    
    # Get backup files
    snapshot_files = get_backup_files(backup_dir)
    
    if not snapshot_files:
        logging.error("No snapshot files found in backup directory!")
        return 1
    
    # Confirm restoration
    if not args.force:
        print(f"\nWARNING: This will DELETE and RESTORE {len(snapshot_files)} collections:")
        for f in snapshot_files:
            print(f"  - {f.stem}")
        
        response = input("\nContinue? (yes/no): ")
        if response.lower() != "yes":
            logging.info("Restore cancelled by user")
            return 0
    
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
    
    # Restore each collection
    success_count = 0
    failed_collections = []
    
    for snapshot_file in snapshot_files:
        if restore_collection(client, snapshot_file):
            success_count += 1
        else:
            failed_collections.append(snapshot_file.stem)
    
    # Summary
    logging.info("=" * 60)
    logging.info(f"Restore completed: {success_count}/{len(snapshot_files)} collections")
    
    if failed_collections:
        logging.warning(f"Failed collections: {', '.join(failed_collections)}")
    
    logging.info("=" * 60)
    
    return 0 if success_count == len(snapshot_files) else 1


if __name__ == "__main__":
    exit(main())
