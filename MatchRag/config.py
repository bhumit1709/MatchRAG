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

# Number of top-K deliveries to retrieve per question
TOP_K: int = int(os.getenv("TOP_K", "15"))

# ── Data ─────────────────────────────────────────────────────────────────────

# Default match JSON file path (relative to project root)
DATA_FILE: str = os.getenv("DATA_FILE", "data/IndVsWI.json")

# ── API server ───────────────────────────────────────────────────────────────

API_PORT: int = int(os.getenv("API_PORT", "5001"))
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
