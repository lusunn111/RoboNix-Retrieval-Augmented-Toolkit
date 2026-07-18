from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

print("Checking 'clip_image_collection' payloads...")
try:
    # Fetch a few points
    points, _ = client.scroll(
        collection_name="clip_image_collection",
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    
    if not points:
        print("No points found in collection!")
    else:
        for p in points:
            print(f"ID: {p.id}")
            print(f"Payload keys: {list(p.payload.keys())}")
            print(f"Dataset Name: {p.payload.get('dataset_name')}")
            print(f"Text: {p.payload.get('text')}")
            print("-" * 20)

except Exception as e:
    print(f"Error: {e}")
