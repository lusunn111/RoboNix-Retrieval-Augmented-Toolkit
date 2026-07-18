import os
import sys
import time
import uuid
import json
import base64
import requests
import torch
import argparse
from io import BytesIO
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, Distance

# Configuration
IMAGE_PATH = "/path/to/rtcache/test_fire.png"
EMBEDDING_SERVER_URL = "http://127.0.0.1:9020/predict"
RETRIEVAL_SERVER_URL = "http://127.0.0.1:5002/pipeline"
MONGO_URL = "mongodb://localhost:27017/"
MONGO_DB_NAME = "OpenVLACollection"
DATASET_NAME = "test"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
CONSECUTIVE_STEPS = 2

def get_embedding(image_path):
    print(f"Getting embedding for {image_path}...")
    with open(image_path, "rb") as f:
        resp = requests.post(EMBEDDING_SERVER_URL, files={"file": ("image.png", f, "image/png")}, 
                             data={"instruction": "test instruction", "option": "image"})
    if resp.status_code != 200: sys.exit(f"Error getting embedding: {resp.text}")
    
    result = resp.json()
    b64_str = result.get("clip_image_features") or result.get("image_features")
    if not b64_str: sys.exit("No image features found in response")
        
    tensor = torch.load(BytesIO(base64.b64decode(b64_str)), map_location="cpu")
    return tensor.squeeze().tolist()

def setup_db(embedding_dim):
    print("Setting up databases...")
    mongo_coll = MongoClient(MONGO_URL)[MONGO_DB_NAME][f"{DATASET_NAME}_collection"]
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    coll_name = f"image_collection_{DATASET_NAME}_clip"
    
    if not qdrant_client.collection_exists(coll_name):
        qdrant_client.create_collection(coll_name, vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE))
        
    return mongo_coll, qdrant_client, coll_name

def insert_data(mongo_coll, qdrant_client, qdrant_coll_name, embedding, image_path):
    print("Inserting data...")
    base_id = "test_1_100"
    
    # Insert consecutive steps into MongoDB
    for i in range(CONSECUTIVE_STEPS + 1):
        step_id = f"test_1_{100 + i}"
        doc = {
            "logical_id": step_id, "dataset": DATASET_NAME,
            "raw_action": [0.1 * (i+1)] * 7, "action": [0.1 * (i+1)] * 7,
            "observation": {"image": "dummy_path"}, "instruction": "test instruction"
        }
        mongo_coll.replace_one({"logical_id": step_id}, doc, upsert=True)

    # Insert base step into Qdrant
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, base_id))
    point = PointStruct(id=point_id, vector=embedding, payload={
        "logical_id": base_id, "dataset_name": DATASET_NAME,
        "text": "test instruction", "image_path": image_path
    })
    qdrant_client.upsert(collection_name=qdrant_coll_name, points=[point])
    print(f"Inserted data with logical_id: {base_id}")
    return base_id

def test_retrieval(image_path):
    print("Testing retrieval...")
    with open(image_path, "rb") as f:
        try:
            resp = requests.post(RETRIEVAL_SERVER_URL, files={"file": ("image.png", f, "image/png")},
                                 data={"instruction": "test instruction", "option": "both"})
        except requests.exceptions.ConnectionError:
            return print(f"Error: Could not connect to retrieval server at {RETRIEVAL_SERVER_URL}")
        
    if resp.status_code != 200: return print(f"Error calling retrieval server: {resp.text}")
    print("Retrieval Result:", json.dumps(resp.json(), indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["insert", "retrieve", "both"], default="both", help="Mode: insert, retrieve, or both")
    args = parser.parse_args()

    if not os.path.exists(IMAGE_PATH): sys.exit(f"Image not found: {IMAGE_PATH}")
    
    embedding = get_embedding(IMAGE_PATH)
    mongo_coll, qdrant_client, qdrant_coll_name = setup_db(len(embedding))
    
    if args.mode in ["insert", "both"]:
        insert_data(mongo_coll, qdrant_client, qdrant_coll_name, embedding, IMAGE_PATH)
        if args.mode == "both":
            print("\nWARNING: Restart retrieval server to load new MongoDB data, then press Enter...")
            input()
    
    if args.mode in ["retrieve", "both"]:
        test_retrieval(IMAGE_PATH)

if __name__ == "__main__":
    main()
