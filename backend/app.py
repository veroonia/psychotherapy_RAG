"""
app.py
Backend for the AI Chat frontend, now grounded in the RAG pipeline.
Serves index.html/style.css from ../frontend and exposes a /chat
endpoint that retrieves relevant chunks from the vector DB and asks
OpenRouter to answer using them (see rag/generate.py).

Setup:
    1. Put OPENROUTER_API_KEY=sk-or-v1-... in a .env file in this folder.
    2. pip install -r requirements.txt
    3. python app.py
    4. Open http://localhost:5000 in your browser.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

# make rag/ importable so we can reuse its retrieve + generate logic
RAG_DIR = Path(__file__).parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))
from generate import answer_question  # noqa: E402

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not os.environ.get("OPENROUTER_API_KEY"):
        return jsonify({"error": "OPENROUTER_API_KEY not set on the server."}), 500

    data = request.get_json(force=True, silent=True) or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        result = answer_question(user_message, history=history)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to generate answer: {e}"}), 502

    return jsonify({
        "reply": result["answer"],
        "sources": [
            {"source": h["source"], "page": h["page"]}
            for h in result["sources"]
        ],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)