"""Embedding helpers backed by LangChain local embedding models."""

import sys
from pathlib import Path

from config import EMBED_MODEL, EMBED_MODEL_PATH
from rag.providers import get_embeddings


def embed_text(text: str) -> list[float]:
    """Generate a single embedding vector for a text string."""
    return get_embeddings().embed_query(text)


def generate_embeddings(
    documents: list[dict],
    batch_size: int = 50,
    verbose: bool = True,
) -> list[list[float]]:
    """Generate embeddings for a list of delivery documents."""
    texts = [doc["text"] for doc in documents]
    embeddings: list[list[float]] = []
    total = len(texts)

    if verbose:
        print(f"Generating embeddings for {total} deliveries using '{EMBED_MODEL}'...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        embeddings.extend(get_embeddings().embed_documents(batch))
        if verbose:
            print(f"  Embedded {end}/{total} deliveries...", end="\r")

    if verbose:
        print(f"\nDone. Generated {len(embeddings)} embedding vectors.")

    return embeddings


def check_model_available(model: str = EMBED_MODEL) -> bool:
    """Check whether the configured local embedding model can be resolved."""
    if EMBED_MODEL_PATH:
        return Path(EMBED_MODEL_PATH).exists()
    return bool(model)


if __name__ == "__main__":
    from rag.flatten_data import flatten_deliveries
    from rag.load_match import load_match

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/IndVsNZ.json"
    data = load_match(filepath)
    docs = flatten_deliveries(data)
    embeddings = generate_embeddings(docs)

    assert len(embeddings) == len(docs), "Embedding count mismatch!"
    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
