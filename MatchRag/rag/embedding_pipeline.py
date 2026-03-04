"""
embedding_pipeline.py
---------------------
Step 5 of the RAG pipeline.

Generates vector embeddings for each delivery document using
Ollama's nomic-embed-text model.  Runs fully locally — no cloud APIs.
"""

import sys
from typing import Optional
import ollama

from config import EMBED_MODEL


def embed_text(text: str) -> list[float]:
    """
    Generate a single embedding vector for a text string.

    Args:
        text: The string to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    response = ollama.embed(model=EMBED_MODEL, input=text)
    # ollama.embed returns {"embeddings": [[...vector...]]}
    return response["embeddings"][0]


def generate_embeddings(
    documents: list[dict],
    batch_size: int = 50,
    verbose: bool = True,
) -> list[list[float]]:
    """
    Generate embeddings for a list of delivery documents.

    Each document must have a 'text' field (built by flatten_data.py).
    Processes documents in batches to manage memory and show progress.

    Args:
        documents: List of flat delivery dicts each containing a 'text' key.
        batch_size: Number of documents per batch call.
        verbose: Whether to print progress to stdout.

    Returns:
        List of embedding vectors in the same order as `documents`.
    """
    texts = [doc["text"] for doc in documents]
    embeddings = []
    total = len(texts)

    if verbose:
        print(f"Generating embeddings for {total} deliveries using '{EMBED_MODEL}'...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]

        # Ollama supports batch embeds in one call
        response = ollama.embed(model=EMBED_MODEL, input=batch)
        batch_embeddings = response["embeddings"]
        embeddings.extend(batch_embeddings)

        if verbose:
            print(f"  Embedded {end}/{total} deliveries...", end="\r")

    if verbose:
        print(f"\nDone. Generated {len(embeddings)} embedding vectors.")

    return embeddings


def check_model_available(model: str = EMBED_MODEL) -> bool:
    """
    Check if the given Ollama model is available locally.

    Args:
        model: Model name to check.

    Returns:
        True if model is available, False otherwise.
    """
    try:
        models = ollama.list()
        names = [m.model for m in models.models]
        return any(model in name for name in names)
    except Exception:
        return False


if __name__ == "__main__":
    from rag.load_match import load_match
    from rag.flatten_data import flatten_deliveries

    if not check_model_available(EMBED_MODEL):
        print(f"ERROR: Ollama model '{EMBED_MODEL}' is not available.")
        print(f"  Run: ollama pull {EMBED_MODEL}")
        sys.exit(1)

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/IndVsWI.json"
    data = load_match(filepath)
    docs = flatten_deliveries(data)
    embeddings = generate_embeddings(docs)

    # Quick sanity check
    assert len(embeddings) == len(docs), "Embedding count mismatch!"
    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
