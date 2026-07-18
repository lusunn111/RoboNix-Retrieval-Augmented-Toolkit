import os
from pymongo import MongoClient
from qdrant_client import QdrantClient

# Config
MONGO_URL = "mongodb://localhost:27017/"
MONGO_DB_NAME = "OpenVLACollection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def check_status():
    print("Checking database status...")
    
    # Check MongoDB
    try:
        mongo_client = MongoClient(MONGO_URL)
        db = mongo_client[MONGO_DB_NAME]
        coll_name = "trajectories"
        count = db[coll_name].count_documents({})
        print(f"MongoDB - Database: {MONGO_DB_NAME}")
        print(f"MongoDB - Collection: {coll_name}")
        print(f"MongoDB - Total Documents: {count}")
    except Exception as e:
        print(f"MongoDB Error: {e}")

    # Check Qdrant
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = ["image_collection", "text_collection", "clip_image_collection", "clip_text_collection"]
        
        print("\nQdrant Collections:")
        for name in collections:
            try:
                info = qdrant_client.get_collection(name)
                print(f"  - {name}: {info.points_count} points")
            except Exception:
                print(f"  - {name}: Not found")
                
    except Exception as e:
        print(f"Qdrant Error: {e}")

if __name__ == "__main__":
    check_status()
