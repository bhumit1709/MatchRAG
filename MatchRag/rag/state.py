from typing import TypedDict

class RAGState(TypedDict):
    """Shared state that flows through all nodes of the graph."""
    question:           str
    rewritten_question: str
    chat_history:       list[dict]   # [{role, content}, ...]
    retrieval_filters:  dict | None  # ChromaDB where-clause from entity extraction
    player_stats:       dict | None  # Deterministic stats for the extracted player
    aggregate_stats:    str | None   # Deterministic aggregate stats (leaderboard/count) for stat questions
    group_by:           str          # Event grouping field (player, over, innings)
    metric:             str          # Stat metric to calculate (count, runs_total)
    is_stat_question:   bool         # Whether the question is an aggregate calculation
    initial_docs:       list[dict]   # Before reranking
    retrieved_docs:     list[dict]   # After reranking
    context:            str
    answer:             str
    llm_traces:         list[dict]   # [{"node": "name", "prompt": "...", "response": "..."}, ...]
