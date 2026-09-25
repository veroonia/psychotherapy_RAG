"""
create_embeddings.py
Loads chunks.json, embeds every chunk with BAAI/bge-m3 (local,
sentence-transformers, no API key needed), and saves the embeddings
as a .npy array aligned with chunks.json by index.

Usage:
    python create_embeddings.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

#MODEL_NAME = "BAAI/bge-m3"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks to embed")

    print(f"Loading embedding model: {MODEL_NAME} (first run downloads it, ~2GB) ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding ...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )

    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()