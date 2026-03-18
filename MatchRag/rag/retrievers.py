"""Retriever helpers built on top of LangChain documents and Chroma."""

from collections import OrderedDict

from config import ENABLE_CONTEXT_COMPRESSION, ENABLE_MULTI_QUERY, INITIAL_TOP_K, TOP_K
from rag.chains import generate_query_variants
from rag.documents import document_id, serialize_document
from rag.reranker import rerank_documents
from rag.vector_store import get_vector_store


def _merge_scored_results(scored_results: list[tuple]) -> list[dict]:
    merged: OrderedDict[str, dict] = OrderedDict()

    for document, distance in scored_results:
        doc_id = document_id(document) or f"anon-{len(merged)}"
        payload = serialize_document(document, distance=distance)
        current = merged.get(doc_id)
        if current is None or payload["distance"] < current["distance"]:
            merged[doc_id] = payload

    return list(merged.values())


def retrieve_documents(
    question: str,
    where: dict | None = None,
    enable_multi_query: bool | None = None,
    enable_context_compression: bool | None = None,
) -> tuple[list[str], list[dict], list[dict], dict | None]:
    """Retrieve candidate documents using semantic search, multi-query expansion, and compression."""
    vector_store = get_vector_store()
    if enable_multi_query is None:
        enable_multi_query = ENABLE_MULTI_QUERY
    if enable_context_compression is None:
        enable_context_compression = ENABLE_CONTEXT_COMPRESSION

    query_variants = [question]
    trace = None

    scored_results = vector_store.similarity_search_with_score(
        query=question,
        k=INITIAL_TOP_K,
        filter=where,
    )
    initial_docs = _merge_scored_results(scored_results)
    merged_docs = list(initial_docs)

    if enable_multi_query:
        query_variants, trace = generate_query_variants(question)
        all_results = list(scored_results)
        for variant in query_variants[1:]:
            all_results.extend(
                vector_store.similarity_search_with_score(
                    query=variant,
                    k=INITIAL_TOP_K,
                    filter=where,
                )
            )
        merged_docs = _merge_scored_results(all_results)

    if enable_context_compression:
        final_docs = rerank_documents(question, merged_docs, top_n=TOP_K)
    else:
        final_docs = merged_docs[:TOP_K]

    return query_variants, initial_docs, final_docs, trace
