"""LangGraph workflow assembly for the MatchRAG pipeline."""

from typing import Generator
from time import perf_counter
from langgraph.graph import StateGraph, END

from rag.state import RAGState
from rag.graph_nodes import (
    rewrite_question,
    classify_question,
    generate_answer,
    generate_answer_stream,
)
from rag.question_handlers.match_summary import handle_match_summary
from rag.question_handlers.player_performance import handle_player_performance
from rag.question_handlers.over_summary import handle_over_summary
from rag.question_handlers.comparison import handle_comparison
from rag.question_handlers.general import handle_general


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
# Routing function for conditional edges
# ---------------------------------------------------------------------------

def _route_by_question_type(state: RAGState) -> str:
    """Return the question type for LangGraph conditional routing."""
    return state.get("question_type", "general")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_full_graph():
    """
    Conditional routing graph used by ask().

    Pipeline:
      rewrite_question → classify_question ──┬── handle_match_summary ──┐
                                             ├── handle_player_performance ──┤
                                             ├── handle_over_summary ──┤
                                             ├── handle_comparison ──├── generate_answer → END
                                             └── handle_general ──┘
    """
    graph = StateGraph(RAGState)

    # Common entry nodes
    graph.add_node("rewrite_question",           _timed("rewrite_question"))
    graph.add_node("classify_question",          _timed("classify_question"))

    # Type-specific handler nodes
    graph.add_node("handle_match_summary",       _timed("handle_match_summary"))
    graph.add_node("handle_player_performance",  _timed("handle_player_performance"))
    graph.add_node("handle_over_summary",        _timed("handle_over_summary"))
    graph.add_node("handle_comparison",          _timed("handle_comparison"))
    graph.add_node("handle_general",             _timed("handle_general"))

    # Common exit
    graph.add_node("generate_answer",            _timed("generate_answer"))

    # Entry
    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question", "classify_question")

    # Conditional routing based on question type
    graph.add_conditional_edges(
        "classify_question",
        _route_by_question_type,
        {
            "match_summary":      "handle_match_summary",
            "player_performance": "handle_player_performance",
            "over_summary":       "handle_over_summary",
            "comparison":         "handle_comparison",
            "general":            "handle_general",
        },
    )

    # All handler branches converge to answer generation
    for handler in [
        "handle_match_summary",
        "handle_player_performance",
        "handle_over_summary",
        "handle_comparison",
        "handle_general",
    ]:
        graph.add_edge(handler, "generate_answer")

    graph.add_edge("generate_answer", END)

    return graph.compile()


def _build_pre_answer_graph():
    """
    Conditional routing graph used by ask_stream().

    Runs everything up to and including the type-specific handlers, then
    returns control so token streaming can happen outside the graph.

    Why not use the full graph for streaming?
    LangGraph's .stream() emits completed-node state snapshots, not token-level
    chunks from within a node. The token stream from generate_answer_stream()
    must therefore be driven separately via SSE.
    """
    graph = StateGraph(RAGState)

    # Common entry nodes
    graph.add_node("rewrite_question",           _timed("rewrite_question"))
    graph.add_node("classify_question",          _timed("classify_question"))

    # Type-specific handler nodes
    graph.add_node("handle_match_summary",       _timed("handle_match_summary"))
    graph.add_node("handle_player_performance",  _timed("handle_player_performance"))
    graph.add_node("handle_over_summary",        _timed("handle_over_summary"))
    graph.add_node("handle_comparison",          _timed("handle_comparison"))
    graph.add_node("handle_general",             _timed("handle_general"))

    # Entry
    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question", "classify_question")

    # Conditional routing
    graph.add_conditional_edges(
        "classify_question",
        _route_by_question_type,
        {
            "match_summary":      "handle_match_summary",
            "player_performance": "handle_player_performance",
            "over_summary":       "handle_over_summary",
            "comparison":         "handle_comparison",
            "general":            "handle_general",
        },
    )

    # All handler branches go to END (no generate_answer for streaming)
    for handler in [
        "handle_match_summary",
        "handle_player_performance",
        "handle_over_summary",
        "handle_comparison",
        "handle_general",
    ]:
        graph.add_edge(handler, END)

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
    """Return a compiled full-pipeline graph. Kept for compatibility."""
    return _build_full_graph()


# ---------------------------------------------------------------------------
# Initial state factory
# ---------------------------------------------------------------------------

def _initial_state(question: str, chat_history: list[dict]) -> RAGState:
    return {
        "question":           question,
        "rewritten_question": "",
        "question_type":      "general",
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

    Uses the compiled conditional-routing LangGraph so the framework drives
    state passing, edge traversal, and routing to type-specific handlers.
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

    Uses the compiled pre-answer graph to drive nodes (rewrite, classify,
    handler), then streams LLM tokens outside the graph via
    generate_answer_stream().

    Yields:
      1. FIRST: a dict with pipeline metadata (for the Pipeline Inspector)
         {\"rewritten_question\": ..., \"question_type\": ..., \"num_docs\": ..., ...}
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
        "question_type":      state.get("question_type", "general"),
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
