#!/usr/bin/env python3
"""Quick end-to-end test: embed an image, insert to Qdrant, then retrieve via retrieval_server.

Prereq:
- embedding_server running on http://127.0.0.1:9020
- retrieval_server running on http://127.0.0.1:5002
- Qdrant running on localhost:6333, collection OpenVLACollection (2176-d openvla, 512-d clip)
- Install deps: pip install requests qdrant-client pillow

Usage:
1) Save your test image as ./test_fire.png (or change IMAGE_PATH).
2) Run: python test.py
"""

import base64
import io
import zipfile
import pickle
import torch
import requests
from PIL import Image  # noqa: F401  # only used to ensure pillow is installed
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

EMB_URL = "http://127.0.0.1:9020/predict"
RETR_URL = "http://127.0.0.1:5002/retrieve"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "OpenVLACollection"
IMAGE_PATH = "./test_fire.png"
TOP_K = 3


def decode_feat(b64_str):
	"""Decode server-returned base64(zip(...)) tensor. Try pickle then torch.load."""
	raw = base64.b64decode(b64_str)
	with zipfile.ZipFile(io.BytesIO(raw)) as zf:
		name = zf.namelist()[0]
		with zf.open(name) as f:
			buf = f.read()
	try:
		return pickle.loads(buf)
	except Exception:
		return torch.load(io.BytesIO(buf), map_location="cpu")


def main():
	# 1) Call embedding_server with image
	files = {
		"image": open(IMAGE_PATH, "rb"),
		"instruction": (None, "locate fire station"),
		"option": (None, "both"),  # ask server to return image+text branches
	}
	resp = requests.post(EMB_URL, files=files)
	resp.raise_for_status()
	data = resp.json()

	print("keys returned:", list(data.keys()))

	# Prefer image_features (OpenVLA vision) and clip_image_features; fall back to text if absent.
	if data.get("image_features"):
		openvla_vec = decode_feat(data["image_features"])
	elif data.get("llm_features"):
		openvla_vec = decode_feat(data["llm_features"])
	else:
		raise RuntimeError("No OpenVLA embedding returned (image_features/llm_features missing)")

	if data.get("clip_image_features"):
		clip_img_vec = decode_feat(data["clip_image_features"])
	elif data.get("clip_text_features"):
		clip_img_vec = decode_feat(data["clip_text_features"])
	else:
		raise RuntimeError("No CLIP embedding returned (clip_image_features/clip_text_features missing)")

	# 2) Insert into Qdrant
	cli = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
	try:
		cli.get_collection(COLLECTION)
	except Exception:
		cli.recreate_collection(
			COLLECTION,
			vectors_config={
				"openvla": qm.VectorParams(size=len(openvla_vec), distance=qm.Distance.COSINE),
				"clip": qm.VectorParams(size=len(clip_img_vec), distance=qm.Distance.COSINE),
			},
		)

	point_id = "fire_test_1"
	cli.upsert(
		collection_name=COLLECTION,
		points=[
			qm.PointStruct(
				id=point_id,
				vector={"openvla": openvla_vec, "clip": clip_img_vec},
				payload={"note": "fire network test image"},
			)
		],
	)

	# 3) Query via retrieval_server
	payload = {"query": "find fire station", "top_k": TOP_K}
	r = requests.post(RETR_URL, json=payload, timeout=10)
	r.raise_for_status()
	print("Retrieve response:", r.json())


if __name__ == "__main__":
	main()
