"""
rag_graph.py
------------
Steps 7 & 8 of the RAG pipeline.

Implements a LangGraph StateGraph with nodes:
  rewrite_question → extract_filters → compute_aggregate_stats → retrieve → rerank_docs → build_context → generate_answer

All LLM inference runs locally through Ollama (mistral or llama3).
Supports session memory (chat history) and streaming token output.
"""

from typing import Generator
from langgraph.graph import StateGraph, END

from rag.state import RAGState
from rag.graph_nodes import (
    rewrite_question,
    extract_filters,
    compute_aggregate_stats,
    retrieve,
    rerank_docs,
    build_context,
    generate_answer,
    generate_answer_stream
)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assemble and compile the LangGraph RAG workflow.

    Pipeline:  rewrite_question → extract_filters → compute_aggregate_stats → retrieve → rerank_docs → build_context → generate_answer
    """
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_question",        rewrite_question)
    graph.add_node("extract_filters",         extract_filters)
    graph.add_node("compute_aggregate_stats", compute_aggregate_stats)
    graph.add_node("retrieve",                retrieve)
    graph.add_node("rerank_docs",             rerank_docs)
    graph.add_node("build_context",           build_context)
    graph.add_node("generate_answer",         generate_answer)

    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question",        "extract_filters")
    graph.add_edge("extract_filters",         "compute_aggregate_stats")
    graph.add_edge("compute_aggregate_stats", "retrieve")
    graph.add_edge("retrieve",                "rerank_docs")
    graph.add_edge("rerank_docs",             "build_context")
    graph.add_edge("build_context",           "generate_answer")
    graph.add_edge("generate_answer",         END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------

_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _initial_state(question: str, chat_history: list[dict]) -> RAGState:
    return {
        "question":           question,
        "rewritten_question": "",
        "chat_history":       chat_history or [],
        "retrieval_filters":  None,
        "player_stats":       None,
        "aggregate_stats":    None,
        "initial_docs":       [],
        "retrieved_docs":     [],
        "context":            "",
        "answer":             "",
        "group_by":           "player",
        "metric":             "count",
        "is_stat_question":   False,
    }


def ask(question: str, chat_history: list[dict] = None) -> str:
    """
    Run the full RAG pipeline for a single question and return the answer.
    Optionally accepts prior chat_history for follow-up support.
    """
    state = _initial_state(question, chat_history or [])

    # Run pipeline manually so we can stream generation
    state = rewrite_question(state)
    state = extract_filters(state)
    state = compute_aggregate_stats(state)
    state = retrieve(state)
    state = rerank_docs(state)
    state = build_context(state)
    state = generate_answer(state)
    return state["answer"]


def ask_stream(
    question: str,
    chat_history: list[dict] = None,
) -> Generator[str | dict, None, None]:
    """
    Streaming variant of ask().

    Yields:
      1. FIRST: a dict with pipeline metadata (for the inspector)
         {"rewritten_question": ..., "num_docs": ..., "top_docs": [...]}
      2. THEN: token strings from the LLM generation

    Callers should check `isinstance(item, dict)` for the metadata event.
    """
    state = _initial_state(question, chat_history or [])
    state = rewrite_question(state)
    state = extract_filters(state)
    state = compute_aggregate_stats(state)
    state = retrieve(state)
    state = rerank_docs(state)
    state = build_context(state)

    # Build inspector metadata from the state AFTER retrieval
    docs = state["retrieved_docs"]
    initial_docs = state["initial_docs"]
    
    def _format_docs(doc_list):
        formatted = []
        for doc in doc_list:   # all retrieved docs for the inspector
            m = doc["metadata"]
            formatted.append({
                "innings": m.get("innings", "?"),
                "over":    f"{m.get('over', '?')}.{m.get('ball', '?')}",
                "batter":  m.get("batter", "?"),
                "bowler":  m.get("bowler", "?"),
                "event":   m.get("event", "?"),
                "runs":    m.get("runs_total", "?"),
                "distance": float(round(doc.get("distance", 0), 4)),
                "score": float(round(doc.get("score", 0), 4)) if "score" in doc else None,
            })
        return formatted

    top_docs = _format_docs(docs)
    initial_top_docs = _format_docs(initial_docs)

    meta = {
        "rewritten_question": state["rewritten_question"],
        "was_rewritten":      state["rewritten_question"] != question,
        "retrieval_filters":  state.get("retrieval_filters"),
        "aggregate_stats":    state.get("aggregate_stats"),
        "group_by":           state.get("group_by", "player"),
        "metric":             state.get("metric", "count"),
        "num_docs":           len(docs),
        "initial_num_docs":   len(initial_docs),
        "top_docs":           top_docs,
        "initial_top_docs":   initial_top_docs,
        "history_turns":      len(chat_history or []) // 2,
    }

    yield meta   # <-- first yield is always the metadata dict
    yield from generate_answer_stream(state)  # then token strings


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rag.vector_store import collection_exists

    if not collection_exists():
        print("ChromaDB index not found. Run chat.py to auto-build the index first.")
        raise SystemExit(1)

    test_questions = [
        ("Who dismissed Shimron Hetmyer?", []),
        ("What over was that?",            [
            {"role": "user",      "content": "Who dismissed Shimron Hetmyer?"},
            {"role": "assistant", "content": "Hetmyer was dismissed by JJ Bumrah in Over 12.3 (caught)."},
        ]),
    ]
    for q, history in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"A: {ask(q, chat_history=history)}")
