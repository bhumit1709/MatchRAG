from typing import TypedDict

from rag.schemas import RetrievalPlan, QuestionType


class RAGState(TypedDict):
    """Shared state that flows through all nodes of the graph."""
    question:           str
    rewritten_question: str
    question_type:      QuestionType # match_summary, player_performance, over_summary, comparison, general
    query_variants:     list[str]
    chat_history:       list[dict]   # [{role, content}, ...]
    retrieval_plan:     RetrievalPlan | None
    retrieval_filters:  dict | None  # ChromaDB where-clause from entity extraction
    player_stats:       list[dict] | None  # Deterministic stats for the extracted player(s)
    aggregate_stats:    str | None   # Deterministic aggregate stats (leaderboard/count) for stat questions
    aggregate_rows:     list[dict] | None  # Raw deterministic leaderboard rows for direct-answer fast paths
    answer_strategy:    str          # semantic, aggregate, sequential, or hybrid
    group_by:           str          # Event grouping field (player, over, innings)
    metric:             str          # Stat metric to calculate (count, runs_total)
    is_stat_question:   bool         # Whether the question is an aggregate calculation
    is_sequential:      bool         # Whether to bypass semantic search for exact chronological order
    sort_direction:     str          # "asc" or "desc" for sequential queries
    limit:              int | None   # Number of items to fetch for sequential queries (e.g. "first" -> 1)
    initial_docs:       list[dict]   # Before reranking
    retrieved_docs:     list[dict]   # After reranking
    stage_timings_ms:   dict[str, float]  # Per-stage timing breakdown in milliseconds
    context:            str
    answer:             str
    llm_traces:         list[dict]   # [{"node": "name", "prompt": "...", "response": "..."}, ...]
