"""
app.py
Minimal backend for the AI Chat frontend. Serves index.html/style.css
from ../frontend and exposes a /chat endpoint that calls OpenRouter.

This is step one: a working chatbot with no RAG yet. Once this works,
we'll swap the plain system prompt for retrieved context from the RAG
pipeline (rag/retrieve.py).

Setup:
    1. Put OPENROUTER_API_KEY=sk-or-v1-... in a .env file in this folder.
    2. pip install -r requirements.txt
    3. python app.py
    4. Open http://localhost:5000 in your browser.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "inclusionai/ling-3.0-flash-sante:free"

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

SYSTEM_PROMPT = (
    "You are a helpful, warm AI assistant for a psychotherapy-focused chat app. "
    "Be clear and concise. You are not a therapist and cannot provide diagnosis "
    "or crisis support -- if asked for personal clinical advice, answer generally "
    "and note that a licensed professional should be consulted for individual care."
)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY not set on the server."}), 500

    data = request.get_json(force=True, silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
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
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"OpenRouter request failed: {e}"}), 502
    except (KeyError, IndexError):
        return jsonify({"error": "Unexpected response format from OpenRouter."}), 502

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)