"""
retrieve.py
Query the local Qdrant collection built by create_vector_db.py.

Parent-child retrieval: the query is matched against small, precise child
chunks (what was embedded), but each result also carries its parent
chunk's full text as richer context -- the "small-to-big" retrieval
pattern. Use `matched_text` to show/highlight exactly what matched, and
`context` (the parent text) as what you actually feed to the LLM.

Usage (as a script, for quick testing):
    python retrieve.py "how does psychodrama address trauma?"

Usage (as a module, e.g. from your backend):
    from retrieve import retrieve
    results = retrieve("your query", top_k=5)
"""

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
PARENTS_FILE = DATA_DIR / "parents.json"
QDRANT_DIR = Path(__file__).parent / "qdrant_db"
COLLECTION_NAME = "psychotherapy_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None
_client = None
_parents = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_DIR))
    return _client


def _get_parents() -> dict:
    global _parents
    if _parents is None:
        with open(PARENTS_FILE, "r", encoding="utf-8") as f:
            plist = json.load(f)
        _parents = {p["parent_id"]: p for p in plist}
    return _parents


def retrieve(query: str, top_k: int = 5, dedupe_parents: bool = True) -> list[dict]:
    """Return the top_k most relevant child-chunk hits, each enriched
    with its parent chunk's full text as context.

    If dedupe_parents is True, only the best-scoring child per parent is
    kept, so you don't get several near-duplicate hits from the same
    section (common when a query matches more than one child inside it).
    """
    model = _get_model()
    client = _get_client()
    parents = _get_parents()

    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()

    search_limit = top_k * 4 if dedupe_parents else top_k
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=search_limit,
    ).points

    results = []
    seen_parents = set()
    for hit in hits:
        payload = hit.payload
        parent_id = payload["parent_id"]
        if dedupe_parents and parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        parent = parents.get(parent_id)
        results.append({
            "score": hit.score,
            "source": payload["source"],
            "page": payload["page"],
            "matched_text": payload["text"],
            "parent_id": parent_id,
            "context": parent["text"] if parent else payload["text"],
            "heading": parent["heading"] if parent else None,
        })
        if len(results) >= top_k:
            break

    return results


def main():
    query = " ".join(sys.argv[1:]) or "What is psychodrama?"
    print(f"Query: {query}\n")

    hits = retrieve(query, top_k=5)
    for i, hit in enumerate(hits, start=1):
        heading = f" [{hit['heading']}]" if hit["heading"] else ""
        print(f"[{i}] {hit['source']} p.{hit['page']}{heading} - score {hit['score']:.3f}")
        print(f"    matched: {hit['matched_text'][:150]}...")
        print(f"    context: {hit['context'][:150]}...\n")


if __name__ == "__main__":
    main()