"""
create_vector_db.py
Builds a local, embedded Qdrant collection from the child chunks
(chunks.json) and their embeddings (embeddings.npy). Each point's payload
stores the parent_id, so retrieve.py can hand back full parent context
for whichever child chunk matched the query.

Runs Qdrant in local/embedded mode (on-disk, no server or Docker needed).

Usage:
    python create_vector_db.py
"""

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA_DIR = Path(__file__).parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
QDRANT_DIR = Path(__file__).parent / "qdrant_db"

COLLECTION_NAME = "psychotherapy_docs"
VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5 output dimension


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)

    assert len(chunks) == len(embeddings), (
        f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings. "
        "Re-run create_embeddings.py."
    )

    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DIR))

    # start clean each time this script runs, so re-runs don't duplicate data
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=c["chunk_id"],
            vector=embeddings[i].tolist(),
            payload={
                "parent_id": c["parent_id"],
                "source": c["source"],
                "page": c["page"],
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]

    BATCH = 500
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + BATCH])
        print(f"  upserted {min(i + BATCH, len(points))}/{len(points)} child chunks")

    count = client.count(COLLECTION_NAME).count
    print(f"\nQdrant collection '{COLLECTION_NAME}' built at {QDRANT_DIR} with {count} points")


if __name__ == "__main__":
    main()