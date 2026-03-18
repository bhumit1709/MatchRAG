from rag.retrievers import retrieve_documents


class FakeDocument:
    def __init__(self, doc_id: str, text: str, metadata: dict):
        self.page_content = text
        self.metadata = {"id": doc_id, **metadata}


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query: str, k: int, filter=None):
        self.calls.append((query, k, filter))
        if query == "Who dismissed Abhishek Sharma?":
            return [
                (FakeDocument("d1", "Ravindra got Abhishek Sharma.", {"event": "wicket"}), 0.11),
                (FakeDocument("d2", "Another wicket chance.", {"event": "wicket"}), 0.44),
            ]
        if query == "Abhishek Sharma dismissal":
            return [
                (FakeDocument("d1", "Ravindra got Abhishek Sharma.", {"event": "wicket"}), 0.09),
                (FakeDocument("d3", "Abhishek Sharma edged behind.", {"event": "wicket"}), 0.20),
            ]
        return []


def test_retrieve_documents_merges_multi_query_results(monkeypatch):
    fake_store = FakeVectorStore()

    monkeypatch.setattr("rag.retrievers.get_vector_store", lambda: fake_store)
    monkeypatch.setattr(
        "rag.retrievers.generate_query_variants",
        lambda question: ([question, "Abhishek Sharma dismissal"], {"node": "multi_query"}),
    )
    monkeypatch.setattr(
        "rag.retrievers.rerank_documents",
        lambda question, documents, top_n: documents[:top_n],
    )

    query_variants, initial_docs, final_docs, trace = retrieve_documents("Who dismissed Abhishek Sharma?")

    assert query_variants == ["Who dismissed Abhishek Sharma?", "Abhishek Sharma dismissal"]
    assert len(initial_docs) == 2
    assert len(final_docs) == 3
    assert final_docs[0]["metadata"]["id"] == "d1"
    assert trace == {"node": "multi_query"}


def test_retrieve_documents_without_multi_query(monkeypatch):
    fake_store = FakeVectorStore()

    monkeypatch.setattr("rag.retrievers.get_vector_store", lambda: fake_store)
    monkeypatch.setattr("rag.retrievers.ENABLE_MULTI_QUERY", False)
    monkeypatch.setattr("rag.retrievers.ENABLE_CONTEXT_COMPRESSION", False)

    query_variants, initial_docs, final_docs, trace = retrieve_documents("Who dismissed Abhishek Sharma?")

    assert query_variants == ["Who dismissed Abhishek Sharma?"]
    assert initial_docs == final_docs
    assert trace is None
