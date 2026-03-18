"""
server.py
---------
Flask API server for the Cricket Match RAG chatbot.

Exposes:
  GET  /                  → health-check / info
  POST /api/ask           → run RAG pipeline; streams answer via SSE
  GET  /api/status        → check whether the ChromaDB index has been built
  POST /api/session/clear → clear chat history for a session

Run:
  python server.py
  python server.py --rebuild   # force re-index on startup
  python server.py --port 8000 # custom port (default: 5001)
"""

import argparse
import json
import time
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from config import DATA_FILE, API_HOST, API_PORT
from rag.ingest import run_ingest
from rag.providers import runtime_summary
from rag.rag_graph import ask_stream
from rag.session_store import get_history, add_turn, clear_session
from rag.vector_store import collection_exists

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    runtime = runtime_summary()
    return jsonify({
        "service":    "Cricket Match RAG API",
        "embed_model": runtime["embed_model"],
        "llm_model":   runtime["llm_model"],
        "llm_runtime": runtime["llm_runtime"],
        "indexed":     collection_exists(),
        "endpoints": {
            "POST /api/ask":           "Ask a question (SSE streaming)",
            "GET  /api/status":        "Check index status",
            "POST /api/session/clear": "Clear session history",
        },
    })


@app.route("/api/status")
def status():
    runtime = runtime_summary()
    return jsonify({
        "indexed":     collection_exists(),
        "embed_model": runtime["embed_model"],
        "llm_model":   runtime["llm_model"],
        "llm_runtime": runtime["llm_runtime"],
    })


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    Stream the RAG answer back to the client via Server-Sent Events (SSE).

    Request body:
      {
        "question":   "...",          # required
        "session_id": "uuid-string"   # optional — enables session memory
      }

    SSE event format:
      data: {"type": "token",  "content": "..."}   ← one per token
      data: {"type": "done",   "elapsed": 1.23}     ← final event
      data: {"type": "error",  "message": "..."}    ← on failure
    """
    body       = request.get_json(silent=True) or {}
    question   = (body.get("question") or "").strip()
    session_id = (body.get("session_id") or "").strip() or None

    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    if not collection_exists():
        return jsonify({"error": "ChromaDB index not built yet. Restart the server."}), 503

    # Load prior history for this session (empty list if no session_id)
    chat_history = get_history(session_id) if session_id else []

    def generate():
        t0          = time.time()
        full_answer = []

        try:
            for item in ask_stream(question, chat_history=chat_history):
                # First yield is always a metadata dict (pipeline inspector)
                if isinstance(item, dict):
                    payload = json.dumps({"type": "meta", **item})
                    yield f"data: {payload}\n\n"
                    continue

                # Subsequent yields are token strings
                full_answer.append(item)
                payload = json.dumps({"type": "token", "content": item})
                yield f"data: {payload}\n\n"

            answer  = "".join(full_answer)
            elapsed = round(time.time() - t0, 2)

            # Persist the turn into session history
            if session_id:
                add_turn(session_id, question, answer)

            payload = json.dumps({"type": "done", "elapsed": elapsed})
            yield f"data: {payload}\n\n"

        except Exception as exc:
            payload = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {payload}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        },
    )


@app.route("/api/session/clear", methods=["POST"])
def clear_session_route():
    """Clear the chat history for a given session_id."""
    body       = request.get_json(silent=True) or {}
    session_id = (body.get("session_id") or "").strip()

    if not session_id:
        return jsonify({"error": "Missing 'session_id'."}), 400

    clear_session(session_id)
    return jsonify({"status": "cleared", "session_id": session_id})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cricket RAG Flask API")
    parser.add_argument("--file",    default=DATA_FILE, help="Path to match JSON")
    parser.add_argument("--rebuild", action="store_true", help="Force re-index on startup")
    parser.add_argument("--port",    type=int, default=API_PORT, help="Port to listen on")
    parser.add_argument("--host",    default=API_HOST, help="Host to bind to")
    args = parser.parse_args()

    ingest_result = run_ingest(args.file, force_rebuild=args.rebuild, verbose=False)
    if ingest_result["skipped"]:
        print("✓ Existing ChromaDB index found — skipping ingest.")
    else:
        print(
            f"✓ Index ready with {ingest_result['records']} deliveries "
            f"in {ingest_result['elapsed']:.1f}s"
        )

    runtime = runtime_summary()
    print(f"\n🏏 Cricket RAG API running on http://{args.host}:{args.port}")
    print(f"   Runtime: {runtime['llm_runtime']} | Embed: {runtime['embed_model']} | LLM: {runtime['llm_model']}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
