"""
vector_store.py
---------------
Step 6 of the RAG pipeline.

Manages a persistent ChromaDB collection for cricket commentary embeddings.
Provides functions to build the index and run semantic searches.
"""

import sys
import chromadb
from chromadb.config import Settings
import ollama

from rag.embedding_pipeline import EMBED_MODEL
from config import CHROMA_PATH, COLLECTION_NAME


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client stored at CHROMA_PATH."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(client: chromadb.PersistentClient = None):
    """
    Get or create the cricket commentary ChromaDB collection.

    Args:
        client: Optional existing ChromaDB client; one is created if not provided.

    Returns:
        ChromaDB Collection object.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Use cosine distance for semantic similarity
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    documents: list[dict],
    embeddings: list[list[float]],
    reset: bool = False,
) -> None:
    """
    Upsert all delivery documents and their embeddings into ChromaDB.

    Args:
        documents: Flat delivery dicts (from flatten_data.py).
        embeddings: Embedding vectors in the same order as documents.
        reset: If True, delete the existing collection before inserting.
    """
    client = get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = get_collection(client)

    # ChromaDB requires string IDs, plain text content, and a metadata dict
    ids = [doc["id"] for doc in documents]

    # Metadata: only scalar types (str, int, float, bool) are supported
    metadatas = [
        {
            "match":   doc["match"],
            "innings": doc["innings"],
            "over":    doc["over"],
            "ball":    doc["ball"],
            "batter":  doc["batter"],
            "bowler":  doc["bowler"],
            "event":   doc["event"],
            "venue":   doc["venue"],
            "season":  doc["season"],
            "batting_team": doc["batting_team"],
            "player_out":   doc.get("player_out", ""),
            "wicket_kind":  doc.get("wicket_kind", ""),
            "wicket_fielder": doc.get("wicket_fielder", ""),
            "runs_total":   doc["runs_total"],
        }
        for doc in documents
    ]

    # The document text stored in Chroma (used for display, not search)
    texts = [doc["text"] for doc in documents]

    print(f"Upserting {len(ids)} records into ChromaDB collection '{COLLECTION_NAME}'...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Index built. Total records: {collection.count()}")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def query(
    question: str,
    n_results: int = 6,
    where: dict = None,
) -> list[dict]:
    """
    Embed the question and search ChromaDB for the most relevant deliveries.

    Args:
        question: Natural language question from the user.
        n_results: Number of top results to return.
        where: Optional ChromaDB metadata filter (e.g., {"event": "wicket"}).

    Returns:
        List of result dicts, each with 'text', 'metadata', 'distance'.
    """
    # Embed the query using the same model used for indexing
    q_embedding = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]

    collection = get_collection()
    kwargs = dict(
        query_embeddings=[q_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Unpack ChromaDB's batch result format
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def collection_exists() -> bool:
    """Return True if the ChromaDB collection has been built and has records."""
    try:
        col = get_collection()
        return col.count() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not collection_exists():
        print("No index found. Building from scratch...")
        from rag.load_match import load_match
        from rag.flatten_data import flatten_deliveries
        from rag.embedding_pipeline import generate_embeddings

        filepath = sys.argv[1] if len(sys.argv) > 1 else "data/IndVsWI.json"
        data = load_match(filepath)
        docs = flatten_deliveries(data)
        embeddings = generate_embeddings(docs)
        build_index(docs, embeddings)

    print("\nRunning sample queries:")
    queries = [
        "Who dismissed Shimron Hetmyer?",
        "What happened in the final over?",
        "Biggest six hit in the match",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        results = query(q, n_results=2)
        for r in results:
            m = r["metadata"]
            print(f"  [{m['innings']} inn | Over {m['over']}.{m['ball']}] "
                  f"{m['batter']} vs {m['bowler']} | Event: {m['event']}")
            print(f"  {r['text'][:120]}...")
