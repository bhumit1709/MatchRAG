"""Central configuration for the MatchRAG project."""

from pathlib import Path
import os


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_path_env(name: str) -> str | None:
    raw = os.getenv(name, "").strip()
    return raw or None


def _default_embed_model_path() -> str | None:
    candidate = Path("models/bge-small-en-v1.5")
    return str(candidate) if candidate.exists() else None


# ── Local generation runtime (llama.cpp) ─────────────────────────────────────

LLM_MODEL_PATH: str = os.getenv("LLM_MODEL_PATH", "models/llama-chat.gguf")
LLM_MODEL: str = Path(LLM_MODEL_PATH).name
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_N_CTX: int = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_BATCH: int = int(os.getenv("LLM_N_BATCH", "256"))
LLM_N_THREADS: int = int(os.getenv("LLM_N_THREADS", str(os.cpu_count() or 4)))
LLM_N_GPU_LAYERS: int = int(os.getenv("LLM_N_GPU_LAYERS", "0"))

# ── Local embeddings ──────────────────────────────────────────────────────────

EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBED_MODEL_PATH: str | None = _get_path_env("EMBED_MODEL_PATH") or _default_embed_model_path()
EMBED_MODEL: str = EMBED_MODEL_PATH or EMBED_MODEL_NAME
EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")
EMBED_CACHE_DIR: str | None = _get_path_env("EMBED_CACHE_DIR")

# ── ChromaDB ─────────────────────────────────────────────────────────────────

CHROMA_PATH: str = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "cricket_commentary")

# ── Retrieval / RAG pipeline ─────────────────────────────────────────────────

RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "20"))
INITIAL_TOP_K: int = int(os.getenv("INITIAL_TOP_K", str(RETRIEVER_TOP_K)))
TOP_K: int = int(os.getenv("TOP_K", "8"))
MULTI_QUERY_COUNT: int = int(os.getenv("MULTI_QUERY_COUNT", "3"))
ENABLE_MULTI_QUERY: bool = _get_bool("ENABLE_MULTI_QUERY", True)
ENABLE_CONTEXT_COMPRESSION: bool = _get_bool("ENABLE_CONTEXT_COMPRESSION", True)
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2")

# ── Data ─────────────────────────────────────────────────────────────────────

DATA_FILE: str = os.getenv("DATA_FILE", "data/IndVsNZ.json")

# ── Session memory ───────────────────────────────────────────────────────────

MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))
HISTORY_RELEVANCE_THRESHOLD: float = float(
    os.getenv("HISTORY_RELEVANCE_THRESHOLD", "0.6")
)

# ── API server ───────────────────────────────────────────────────────────────

API_PORT: int = int(os.getenv("API_PORT", "5001"))
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
