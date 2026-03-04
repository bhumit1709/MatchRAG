"""
server.py
---------
Flask API server for the Cricket Match RAG chatbot.

Exposes:
  GET  /            → health-check / info
  POST /api/ask     → run the full RAG pipeline and return the answer
  GET  /api/status  → check whether the ChromaDB index has been built

Run:
  python server.py
  python server.py --rebuild   # force re-index on startup
  python server.py --port 8000 # custom port (default: 5000)
"""

import argparse
import time
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import EMBED_MODEL, LLM_MODEL, DATA_FILE, API_PORT, API_HOST
from rag.load_match import load_match
from rag.flatten_data import flatten_deliveries
from rag.embedding_pipeline import generate_embeddings, check_model_available
from rag.vector_store import build_index, collection_exists
from rag.rag_graph import ask

app = Flask(__name__)
CORS(app)  # Allow requests from the React dev server (localhost:5173)


# ---------------------------------------------------------------------------
# Ingest helper (mirrors chat.py)
# ---------------------------------------------------------------------------

def run_ingest(filepath: str, force_rebuild: bool = False) -> None:
    """Build the ChromaDB index if it doesn't exist, or force-rebuild it."""
    if collection_exists() and not force_rebuild:
        print(f"✓ Existing ChromaDB index found — skipping ingest.")
        return

    print(f"\nBuilding index from '{filepath}'...")
    t0 = time.time()

    if not check_model_available(EMBED_MODEL):
        print(f"⚠  Ollama model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}")
        sys.exit(1)

    data = load_match(filepath)
    docs = flatten_deliveries(data)
    print(f"  Flattened {len(docs)} deliveries.")
    embeddings = generate_embeddings(docs, verbose=True)
    build_index(docs, embeddings, reset=force_rebuild)
    print(f"✓ Index ready in {time.time() - t0:.1f}s\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return jsonify({
        "service": "Cricket Match RAG API",
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "indexed": collection_exists(),
        "endpoints": {
            "POST /api/ask": "Ask a question about the match",
            "GET /api/status": "Check index status",
        },
    })


@app.route("/api/status")
def status():
    return jsonify({
        "indexed": collection_exists(),
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
    })


@app.route("/api/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    if not collection_exists():
        return jsonify({"error": "ChromaDB index not built yet. Restart the server."}), 503

    t0 = time.time()
    try:
        answer = ask(question)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "question": question,
        "answer": answer,
        "elapsed": round(time.time() - t0, 2),
    })


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cricket RAG Flask API")
    parser.add_argument("--file", default=DATA_FILE, help="Path to match JSON")
    parser.add_argument("--rebuild", action="store_true", help="Force re-index on startup")
    parser.add_argument("--port", type=int, default=API_PORT, help="Port to listen on")
    parser.add_argument("--host", default=API_HOST, help="Host to bind to")
    args = parser.parse_args()

    run_ingest(args.file, force_rebuild=args.rebuild)

    print(f"\n🏏 Cricket RAG API running on http://{args.host}:{args.port}")
    print(f"   Embed: {EMBED_MODEL}  |  LLM: {LLM_MODEL}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
