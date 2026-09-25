"""
generate.py
Takes a user question, retrieves relevant chunks (via retrieve.py), and
calls an LLM through OpenRouter to produce an answer grounded in those
chunks.

Uses the "openrouter/free" route, which auto-selects a working free
model on OpenRouter's side, so you don't have to track which specific
free model is currently available.

Setup:
    1. Put OPENROUTER_API_KEY=sk-or-v1-... in a .env file in this folder.
    2. pip install -r requirements.txt

Usage (as a script, for quick testing):
    python generate.py "how does psychodrama address trauma?"

Usage (as a module, e.g. from your backend):
    from generate import answer_question
    result = answer_question("your question")
    print(result["answer"])
"""

import os
import sys

import requests
from dotenv import load_dotenv

from retrieve import retrieve

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "inclusionai/ling-3.0-flash-sante:free"

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant answering questions about psychodrama, "
    "trauma, and psychotherapy, based ONLY on the reference material provided "
    "to you in each message. Rules:\n"
    "- Answer strictly from the provided context. If the context doesn't "
    "contain the answer, say so plainly instead of guessing.\n"
    "- Cite the source document and page number for claims you make, "
    "e.g. (Trauma and Recovery, p.12).\n"
    "- You are not a therapist and cannot provide diagnosis, treatment, or "
    "crisis support. If a question asks for personal clinical advice, "
    "answer only with what the source material says in general, and note "
    "that a licensed professional should be consulted for individual care.\n"
    "- Be clear and concise."
)


def build_context_block(hits: list[dict]) -> str:
    """Turn retrieved hits into a numbered context block for the prompt."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[{i}] Source: {hit['source']} (p.{hit['page']})\n{hit['context']}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, top_k: int = 5) -> dict:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Add it to a .env file in this folder."
        )

    hits = retrieve(question, top_k=top_k)
    context_block = build_context_block(hits)

    user_message = (
        f"Reference material:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    answer = data["choices"][0]["message"]["content"]
    return {
        "answer": answer,
        "sources": hits,
        "model_used": data.get("model", MODEL),
    }


def main():
    question = " ".join(sys.argv[1:]) or "What is psychodrama?"
    print(f"Question: {question}\n")

    result = answer_question(question)
    print(f"Answer ({result['model_used']}):\n{result['answer']}\n")

    print("Sources used:")
    for hit in result["sources"]:
        print(f"  - {hit['source']} p.{hit['page']} (score {hit['score']:.3f})")


if __name__ == "__main__":
    main()