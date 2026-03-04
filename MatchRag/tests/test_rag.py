"""
tests/test_rag.py
-----------------
Smoke test / manual integration test for the full RAG pipeline.

Requires the ChromaDB index to already be built (run chat.py or server.py first).

Usage:
    python tests/test_rag.py
    # or via pytest (skipped automatically if index is missing):
    pytest tests/test_rag.py -v
"""

from rag.rag_graph import ask
from rag.vector_store import collection_exists

question = "How did India win the match?"
print(f"Question: {question}")
print("-" * 60)
answer = ask(question)
print(answer)
