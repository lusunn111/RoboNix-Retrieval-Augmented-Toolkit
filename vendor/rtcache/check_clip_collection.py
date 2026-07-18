from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

try:
    count = client.count(collection_name="clip_image_collection")
    print(f"clip_image_collection count: {count.count}")
except Exception as e:
    print(f"Error checking clip_image_collection: {e}")

try:
    count = client.count(collection_name="image_collection")
    print(f"image_collection count: {count.count}")
except Exception as e:
    print(f"Error checking image_collection: {e}")
