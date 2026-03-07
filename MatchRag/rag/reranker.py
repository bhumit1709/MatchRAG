"""
reranker.py
-----------
Uses FlashRank to rerank retrieved documents for better precision.
"""

from flashrank import Ranker, RerankRequest
from config import RERANK_MODEL, TOP_K

# FlashRank Ranker is a singleton to avoid reloading the model
_ranker = None

def get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=RERANK_MODEL)
    return _ranker

def rerank_documents(query: str, documents: list[dict], top_n: int = TOP_K) -> list[dict]:
    """
    Rerank a list of retrieved documents based on the query.

    Args:
        query: The user query string
        documents: List of retrieved dicts containing 'text', 'metadata', 'distance'.
        top_n: Number of documents to return after reranking.

    Returns:
        Sorted list of top_n most relevant document dicts.
    """
    if not documents:
        return []

    ranker = get_ranker()

    passages = []
    for i, doc in enumerate(documents):
        passages.append({
            "id": i,
            "text": doc["text"],
            "meta": doc.get("metadata", {})
        })

    rerank_request = RerankRequest(query=query, passages=passages)
    
    results = ranker.rerank(rerank_request)

    reranked_docs = []
    for r in results[:top_n]:
        original_idx = r["id"]
        original_doc = documents[original_idx]
        reranked_docs.append({
            "text": original_doc["text"],
            "metadata": original_doc["metadata"],
            "distance": original_doc["distance"],
            "score": r["score"]
        })

    return reranked_docs
