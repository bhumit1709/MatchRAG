"""LangGraph workflow assembly for the MatchRAG pipeline."""

from typing import Generator
from time import perf_counter
from langgraph.graph import StateGraph, END

from rag.state import RAGState
from rag.graph_nodes import (
    rewrite_question,
    plan_retrieval,
    compute_aggregate_stats,
    retrieve,
    build_context,
    generate_answer,
    generate_answer_stream,
)


# ---------------------------------------------------------------------------
# Node timing wrapper
# ---------------------------------------------------------------------------

def _timed(node_name: str):
    """
    Wrap a graph node function so its wall-clock time is recorded in RAGState.

    The wrapped function is **looked up by name on this module at call time**
    (via `getattr`), not captured at graph-build time. This means
    `monkeypatch.setattr("rag.rag_graph.<node_name>", mock)` in tests
    is respected even after the compiled graph has been created.
    """
    import rag.rag_graph as _this_module  # late import avoids circular ref

    def wrapper(state: RAGState) -> RAGState:
        fn = getattr(_this_module, node_name)  # dynamic — picks up monkeypatches
        start = perf_counter()
        result = fn(state)
        elapsed_ms = round((perf_counter() - start) * 1000, 1)
        # Accumulate timings: prior timings live in result (nodes return full state)
        timings = dict(result.get("stage_timings_ms", state.get("stage_timings_ms", {})))
        timings[node_name] = elapsed_ms
        return {**result, "stage_timings_ms": timings}

    return wrapper


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_full_graph():
    """
    6-node graph used by ask().

    Pipeline: rewrite_question → plan_retrieval → compute_aggregate_stats
              → retrieve → build_context → generate_answer
    """
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_question",       _timed("rewrite_question"))
    graph.add_node("plan_retrieval",         _timed("plan_retrieval"))
    graph.add_node("compute_aggregate_stats", _timed("compute_aggregate_stats"))
    graph.add_node("retrieve",               _timed("retrieve"))
    graph.add_node("build_context",          _timed("build_context"))
    graph.add_node("generate_answer",        _timed("generate_answer"))

    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question",        "plan_retrieval")
    graph.add_edge("plan_retrieval",          "compute_aggregate_stats")
    graph.add_edge("compute_aggregate_stats", "retrieve")
    graph.add_edge("retrieve",                "build_context")
    graph.add_edge("build_context",           "generate_answer")
    graph.add_edge("generate_answer",         END)

    return graph.compile()


def _build_pre_answer_graph():
    """
    5-node graph used by ask_stream().

    Runs everything up to and including build_context, then returns control
    so token streaming can happen outside the graph.

    Why not use the full graph for streaming?
    LangGraph's .stream() emits completed-node state snapshots, not token-level
    chunks from within a node. The token stream from generate_answer_stream()
    must therefore be driven separately via SSE.
    """
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_question",       _timed("rewrite_question"))
    graph.add_node("plan_retrieval",         _timed("plan_retrieval"))
    graph.add_node("compute_aggregate_stats", _timed("compute_aggregate_stats"))
    graph.add_node("retrieve",               _timed("retrieve"))
    graph.add_node("build_context",          _timed("build_context"))

    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question",        "plan_retrieval")
    graph.add_edge("plan_retrieval",          "compute_aggregate_stats")
    graph.add_edge("compute_aggregate_stats", "retrieve")
    graph.add_edge("retrieve",                "build_context")
    graph.add_edge("build_context",           END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Compiled graph singletons (lazy initialisation)
# ---------------------------------------------------------------------------

_full_graph = None
_pre_answer_graph = None


def _get_full_graph():
    global _full_graph
    if _full_graph is None:
        _full_graph = _build_full_graph()
    return _full_graph


def _get_pre_answer_graph():
    global _pre_answer_graph
    if _pre_answer_graph is None:
        _pre_answer_graph = _build_pre_answer_graph()
    return _pre_answer_graph


# Backward-compatible alias — kept so any code that calls build_graph()
# directly (e.g. scripts or notebooks) continues to work.
def build_graph():
    """Return a compiled 6-node full-pipeline graph. Kept for compatibility."""
    return _build_full_graph()


# ---------------------------------------------------------------------------
# Initial state factory
# ---------------------------------------------------------------------------

def _initial_state(question: str, chat_history: list[dict]) -> RAGState:
    return {
        "question":           question,
        "rewritten_question": "",
        "query_variants":     [],
        "chat_history":       chat_history or [],
        "retrieval_plan":     None,
        "retrieval_filters":  None,
        "player_stats":       None,
        "aggregate_stats":    None,
        "aggregate_rows":     None,
        "answer_strategy":    "semantic",
        "initial_docs":       [],
        "retrieved_docs":     [],
        "context":            "",
        "answer":             "",
        "group_by":           "player",
        "metric":             "count",
        "is_stat_question":   False,
        "is_sequential":      False,
        "sort_direction":     "asc",
        "limit":              None,
        "stage_timings_ms":   {},
        "llm_traces":         [],
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def ask(question: str, chat_history: list[dict] = None) -> str:
    """
    Run the full RAG pipeline for a single question and return the answer.

    Uses the compiled 6-node LangGraph so the framework drives state passing,
    edge traversal, and future features (checkpointing, parallelism, tracing).
    """
    state = _initial_state(question, chat_history or [])
    result = _get_full_graph().invoke(state)
    return result["answer"]


def ask_stream(
    question: str,
    chat_history: list[dict] = None,
) -> Generator[str | dict, None, None]:
    """
    Streaming variant of ask().

    Uses the compiled 5-node pre-answer graph to drive nodes 1–5, then streams
    LLM tokens outside the graph via generate_answer_stream().

    Yields:
      1. FIRST: a dict with pipeline metadata (for the Pipeline Inspector)
         {\"rewritten_question\": ..., \"num_docs\": ..., \"top_docs\": [...], ...}
      2. THEN: raw token strings from the LLM answer step

    Callers should check `isinstance(item, dict)` for the metadata event.
    """
    state = _initial_state(question, chat_history or [])
    state = _get_pre_answer_graph().invoke(state)

    def _format_docs(doc_list: list[dict]) -> list[dict]:
        formatted = []
        if not doc_list:
            return formatted
        for doc in doc_list:
            m = doc["metadata"]
            d = {
                "innings": m.get("innings", "?"),
                "over":    f"{m.get('over', '?')}.{m.get('ball', '?')}",
                "batter":  m.get("batter", "?"),
                "bowler":  m.get("bowler", "?"),
                "event":   m.get("event", "?"),
                "runs":    m.get("runs_total", "?"),
            }
            if "distance" in doc:
                d["distance"] = float(round(doc["distance"], 4))
            if "score" in doc:
                d["score"] = float(round(doc["score"], 4))
            formatted.append(d)
        return formatted

    top_docs = _format_docs(state["retrieved_docs"])
    initial_top_docs = _format_docs(state["initial_docs"])

    # Time the stream setup (prompt formatting + first-token latency init)
    answer_start = perf_counter()
    token_stream, trace = generate_answer_stream(state)
    answer_setup_ms = round((perf_counter() - answer_start) * 1000, 1)

    stage_timings = {
        **state.get("stage_timings_ms", {}),
        "generate_answer_setup": answer_setup_ms,
    }
    llm_traces = state.get("llm_traces", []) + [trace]

    meta = {
        "rewritten_question": state["rewritten_question"],
        "was_rewritten":      state["rewritten_question"] != question,
        "query_variants":     state.get("query_variants", []),
        "retrieval_filters":  state.get("retrieval_filters"),
        "aggregate_stats":    state.get("aggregate_stats"),
        "answer_strategy":    state.get("answer_strategy", "semantic"),
        "group_by":           state.get("group_by", "player"),
        "metric":             state.get("metric", "count"),
        "stage_timings_ms":   stage_timings,
        "num_docs":           len(state["retrieved_docs"]),
        "initial_num_docs":   len(state["initial_docs"]),
        "top_docs":           top_docs,
        "initial_top_docs":   initial_top_docs,
        "history_turns":      len(chat_history or []) // 2,
        "llm_traces":         llm_traces,
    }

    yield meta           # <-- first yield is always the metadata dict
    yield from token_stream  # then token strings


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rag.vector_store import collection_exists

    if not collection_exists():
        print("ChromaDB index not found. Run chat.py to auto-build the index first.")
        raise SystemExit(1)

    test_questions = [
        ("Who dismissed Abhishek Sharma?", []),
        ("What over was that?",            [
            {"role": "user",      "content": "Who dismissed Abhishek Sharma?"},
            {"role": "assistant", "content": "Abhishek Sharma was dismissed by R Ravindra in Over 7.1 (caught by TL Seifert)."},
        ]),
    ]
    for q, history in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"A: {ask(q, chat_history=history)}")
