"""Handler for general question type.

Wraps the existing plan_retrieval → compute_aggregate_stats → retrieve → build_context
pipeline logic. Zero behavioral change from the pre-refactor linear pipeline.
"""

from __future__ import annotations

from rag.graph_nodes import (
    plan_retrieval,
    compute_aggregate_stats,
    retrieve,
    build_context,
)
from rag.state import RAGState


def handle_general(state: RAGState) -> RAGState:
    """Handle general questions using the full existing pipeline logic.

    Runs: plan_retrieval → compute_aggregate_stats → retrieve → build_context
    This preserves all existing behavior: stat leaderboards, ordered events,
    hybrid leader-support, multi-query expansion, reranking, phase stats, etc.
    """
    state = plan_retrieval(state)
    state = compute_aggregate_stats(state)
    state = retrieve(state)
    state = build_context(state)
    return state
