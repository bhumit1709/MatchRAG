"""
rag — Cricket Match RAG pipeline package.

Public API:
    ask(question)            → str   Run the full RAG pipeline for a question
    load_match(filepath)     → dict  Load a CricSheet-format JSON file
    flatten_deliveries(data) → list  Flatten match JSON to delivery records
    build_index(docs)                Build / refresh the ChromaDB index
    collection_exists()      → bool  Check if the index has been built
"""

from rag.rag_graph import ask
from rag.load_match import load_match
from rag.flatten_data import flatten_deliveries
from rag.vector_store import build_index, collection_exists

__all__ = [
    "ask",
    "load_match",
    "flatten_deliveries",
    "build_index",
    "collection_exists",
]
