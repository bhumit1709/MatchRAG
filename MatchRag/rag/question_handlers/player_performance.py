"""Handler for player_performance question type.

Produces player stats and semantic retrieval filtered to a specific player.
"""

from __future__ import annotations

from rag.retrievers import retrieve_documents
from rag.state import RAGState
from rag.question_handlers.utils import question_mentions_players, build_player_filter, format_delivery_header
from rag.state import RAGState
from rag.vector_store import get_known_players, get_player_stats


def _format_player_stats_block(stats: dict) -> str:
    """Format player stats into the system stat block."""
    batting = stats["batting"]
    bowling = stats["bowling"]

    lines = ["=== EXACT STATS FOR REQUESTED PLAYER ==="]
    lines.append(f"Player: {stats['name']}")

    sr = round((batting["runs"] / batting["balls"]) * 100, 2) if batting["balls"] else 0.0
    lines.append(
        f"  Batting: {batting['runs']} runs off {batting['balls']} balls "
        f"({batting['fours']} fours, {batting['sixes']} sixes) | SR: {sr}"
    )
    if batting["dismissal"]:
        lines.append(f"  Dismissal: {batting['dismissal']}")
    else:
        lines.append("  Dismissal: not out")

    if bowling["overs"] != "0.0":
        bowl_overs = bowling["overs"]
        # Parse overs string for economy calculation
        parts = bowl_overs.split(".")
        total_balls = int(parts[0]) * 6 + (int(parts[1]) if len(parts) > 1 else 0)
        econ = round((bowling["runs"] / (total_balls / 6)), 2) if total_balls else 0.0
        lines.append(
            f"  Bowling: {bowling['wickets']} wickets for {bowling['runs']} runs "
            f"in {bowl_overs} overs | Econ: {econ}"
        )
    else:
        lines.append("  Bowling: did not bowl")

    lines.append("=========================================")
    return "\n".join(lines)


# _build_player_filter removed in favor of utils function


def handle_player_performance(state: RAGState) -> RAGState:
    """Handle player_performance questions with player stats and filtered retrieval."""
    question = state["rewritten_question"] or state["question"]
    known_players = get_known_players()
    mentioned_players = question_mentions_players(question, known_players)

    if not mentioned_players:
        # Fallback: no player detected, use semantic retrieval without filter
        query_variants, initial_docs, retrieved_docs, trace = retrieve_documents(
            question,
            where=None,
            enable_multi_query=True,
            enable_context_compression=True,
        )
        llm_traces = state.get("llm_traces", [])
        if trace is not None:
            llm_traces = llm_traces + [trace]

        return {
            **state,
            "query_variants": query_variants,
            "initial_docs": initial_docs,
            "retrieved_docs": retrieved_docs,
            "context": "No relevant match data found.",
            "aggregate_stats": None,
            "player_stats": None,
            "answer_strategy": "semantic",
            "llm_traces": llm_traces,
        }

    # Primary player (first mentioned)
    player_name = mentioned_players[0]

    # ── Stats: exact batting + bowling stats ──────────────────────────────
    player_stats_data = get_player_stats(player_name)
    aggregate_stats = ""
    player_stats_list = []
    if player_stats_data:
        aggregate_stats = _format_player_stats_block(player_stats_data)
        player_stats_list = [player_stats_data]

    # ── Retrieval: semantic search filtered to this player ────────────────
    player_filter = build_player_filter(player_name)
    query_variants, initial_docs, retrieved_docs, trace = retrieve_documents(
        question,
        where=player_filter,
        enable_multi_query=True,
        enable_context_compression=True,
    )
    llm_traces = state.get("llm_traces", [])
    if trace is not None:
        llm_traces = llm_traces + [trace]

    # ── Context: player stats + key delivery highlights ───────────────────
    context_lines = []
    if aggregate_stats:
        context_lines.append(aggregate_stats)
        context_lines.append("")

    if retrieved_docs:
        context_lines.append("=== Key Deliveries ===")
        for index, doc in enumerate(retrieved_docs, start=1):
            meta = doc["metadata"]
            header = format_delivery_header(meta, index)
            commentary = (
                meta.get("commentary")
                or doc["text"].split("Commentary:")[-1].strip()
            )
            context_lines.append(header)
            if commentary:
                context_lines.append(f"    Commentary: {commentary}")
            context_lines.append("")

    context = "\n".join(line for line in context_lines if line is not None).strip()
    if not context:
        context = "No relevant match data found."

    return {
        **state,
        "retrieval_plan": None,
        "retrieval_filters": player_filter,
        "player_stats": player_stats_list or None,
        "aggregate_stats": aggregate_stats or None,
        "query_variants": query_variants,
        "initial_docs": initial_docs,
        "retrieved_docs": retrieved_docs,
        "context": context,
        "answer_strategy": "semantic",
        "llm_traces": llm_traces,
    }
