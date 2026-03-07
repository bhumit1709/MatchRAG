"""
config.py
---------
Central configuration for the MatchRAG project.

All tuneable constants are defined here so they can be changed
in one place and are easily overridden via environment variables.

Usage:
    from config import EMBED_MODEL, LLM_MODEL, CHROMA_PATH, DATA_FILE
"""

import os

# ── Ollama models ────────────────────────────────────────────────────────────

# Embedding model — must be available locally via Ollama
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

# LLM model used for answer generation
LLM_MODEL: str = os.getenv("LLM_MODEL", "mistral")

# ── ChromaDB ─────────────────────────────────────────────────────────────────

# Directory where ChromaDB persists the vector index (relative to project root)
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "chroma_db")

# ChromaDB collection name
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "cricket_commentary")

# ── RAG pipeline ─────────────────────────────────────────────────────────────

# Number of top-K deliveries to retrieve initially from ChromaDB
INITIAL_TOP_K: int = int(os.getenv("INITIAL_TOP_K", "30"))

# Number of top-K deliveries to retain after reranking
TOP_K: int = int(os.getenv("TOP_K", "10"))

# Local reranker model (FlashRank)
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2")

# ── Data ─────────────────────────────────────────────────────────────────────

# Default match JSON file path (relative to project root)
DATA_FILE: str = os.getenv("DATA_FILE", "data/IndVsWI.json")

# ── Session memory ────────────────────────────────────────────────────────────

# Max number of Q&A turns to retain per session (safety-net FIFO cap)
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))

# Cosine similarity threshold for smart pruning (0.0–1.0)
# Older turns below this similarity to the current question are dropped
HISTORY_RELEVANCE_THRESHOLD: float = float(os.getenv("HISTORY_RELEVANCE_THRESHOLD", "0.6"))

# ── API server ───────────────────────────────────────────────────────────────

API_PORT: int = int(os.getenv("API_PORT", "5001"))
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
