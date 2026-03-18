"""Shared LangChain model providers for local generation and embeddings."""

from functools import lru_cache
from pathlib import Path

from langchain_community.chat_models import ChatLlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBED_CACHE_DIR,
    EMBED_DEVICE,
    EMBED_MODEL,
    EMBED_MODEL_PATH,
    LLM_MAX_TOKENS,
    LLM_MODEL_PATH,
    LLM_N_BATCH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
)


def _require_local_llm_path() -> Path:
    path = Path(LLM_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Local llama.cpp model file was not found. "
            f"Set LLM_MODEL_PATH to a valid GGUF file. Current value: {LLM_MODEL_PATH}"
        )
    return path


def _resolve_embed_model_source() -> str:
    return EMBED_MODEL_PATH or EMBED_MODEL


@lru_cache(maxsize=1)
def get_chat_model() -> ChatLlamaCpp:
    """Return the shared local chat model."""
    model_path = _require_local_llm_path()
    return ChatLlamaCpp(
        model_path=str(model_path),
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        n_ctx=LLM_N_CTX,
        n_batch=LLM_N_BATCH,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        verbose=False,
        streaming=True,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared local embedding model."""
    model_source = _resolve_embed_model_source()
    return HuggingFaceEmbeddings(
        model_name=model_source,
        cache_folder=EMBED_CACHE_DIR,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_local_models_ready() -> None:
    """Validate that the configured local runtime is usable."""
    _require_local_llm_path()
    if EMBED_MODEL_PATH and not Path(EMBED_MODEL_PATH).exists():
        raise FileNotFoundError(
            "Local embedding model path was not found. "
            f"Set EMBED_MODEL_PATH to a valid local directory. Current value: {EMBED_MODEL_PATH}"
        )


def runtime_summary() -> dict[str, str]:
    """Return a small summary payload for status endpoints and banners."""
    return {
        "llm_runtime": "llama.cpp",
        "llm_model": Path(LLM_MODEL_PATH).name,
        "embed_model": _resolve_embed_model_source(),
    }
