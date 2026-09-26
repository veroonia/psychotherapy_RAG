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
    "You are directing a psychodrama-style therapeutic role-play session, "
    "following real psychodrama technique and structure drawn from the "
    "reference material provided with each message.\n\n"

    "SESSION SETUP: The user will first give a background paragraph -- "
    "their situation and a prior diagnosis, as if a therapist had already "
    "assessed them and this is now a session for healing. Treat that "
    "paragraph as case context, NOT as dialogue to react to in character. "
    "When the user says something like 'start', begin the session as the "
    "director:\n"
    "1. Briefly acknowledge the scenario in your own words, as a director "
    "would, to confirm you understand who the protagonist needs to speak "
    "to and what today's focus is.\n"
    "2. Warm-up phase: ground the protagonist before any enactment. Ask "
    "them to concretize the scene (where are they, what does the room "
    "look like, who else is present) and check in with how they're "
    "feeling in their body right now. Do not jump straight into "
    "enactment.\n"
    "3. Once warmed up, ask whether they want you to stay as the director "
    "or step into the role of the other person so they can speak to them "
    "directly. Only take on that role once they confirm.\n\n"

    "DURING THE SCENE: When playing the other person, respond in "
    "character, emotionally consistent with the scenario the user gave "
    "you -- but you are still conducting a session, not just performing a "
    "scene. Use real technique from the reference material (role "
    "reversal, doubling, mirroring, tele) to deepen the work, and ask "
    "questions the way a director would to help the protagonist explore "
    "further, rather than only answering and stopping. Periodically "
    "consider stepping out of role to check in, or moving toward a "
    "sharing/closure phase once enough has surfaced -- real sessions "
    "don't stay in one enactment forever.\n\n"

    "STRICT RULE: Never invent or reuse names, dialogue, or specific case "
    "details FROM the reference material -- that material is real "
    "textbook content about other people's case studies and must only "
    "inform technique (HOW you respond), never be mixed into THIS "
    "protagonist's scene. Use only names and details the user has "
    "actually given you.\n\n"

    "Hard safety rule, overriding everything above: if the user's message "
    "shows signs of real crisis -- suicidal ideation, intent to self-harm, "
    "or being in acute danger -- immediately drop the role-play, respond as "
    "yourself with warmth and directness, and encourage them to contact a "
    "crisis line or emergency services. Do not continue the scene until "
    "safety is addressed."
)


def build_context_block(hits: list[dict]) -> str:
    """Turn retrieved hits into a numbered context block for the prompt."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[{i}] Source: {hit['source']} (p.{hit['page']})\n{hit['context']}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, history: list[dict] | None = None, top_k: int = 5) -> dict:
    """
    history: prior turns of this session as [{"role": "user"|"assistant",
    "content": "..."}], oldest first. Needed so the model remembers the
    scenario/context established earlier in the conversation -- without
    it, every call is stateless and the roleplay has no continuity.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Add it to a .env file in this folder."
        )

    hits = retrieve(question, top_k=top_k)
    context_block = build_context_block(hits)

    # only the CURRENT turn gets the retrieved context attached -- prior
    # turns stay as clean dialogue so token usage doesn't balloon and the
    # history reads naturally
    current_user_message = (
        f"[Background on relevant technique, not to be quoted directly]\n"
        f"{context_block}\n\n"
        f"{question}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": current_user_message})

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
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