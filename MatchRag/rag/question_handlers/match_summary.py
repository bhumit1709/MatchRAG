"""Handler for match_summary question type.

Produces a full match scorecard and key moments for narrative match summaries.
"""

from __future__ import annotations

from rag.state import RAGState
from rag.vector_store import (
    get_event_leaderboard,
    get_sequential_deliveries,
)
from rag.question_handlers.utils import format_delivery_header


def _compute_innings_stats(innings: int) -> dict | None:
    """Compute scorecard-level stats for one innings."""
    # Total runs
    runs_board = get_event_leaderboard(
        where_filter={"innings": {"$eq": innings}},
        event_type=None,
        group_by="innings",
        metric="runs_total",
    )
    total_runs = runs_board[0]["count"] if runs_board else 0

    # Total wickets
    wicket_board = get_event_leaderboard(
        where_filter={"$and": [{"innings": {"$eq": innings}}, {"event": {"$eq": "wicket"}}]},
        event_type="wicket",
        group_by="innings",
        metric="count",
    )
    total_wickets = wicket_board[0]["count"] if wicket_board else 0

    # Total balls
    all_deliveries = get_sequential_deliveries(
        where={"innings": {"$eq": innings}},
        sort_direction="asc",
        limit=None,
    )
    total_balls = len(all_deliveries)
    overs_str = f"{total_balls // 6}.{total_balls % 6}" if total_balls else "0.0"
    run_rate = round((total_runs / total_balls) * 6, 2) if total_balls else 0.0

    # Top batters by runs
    batter_board = get_event_leaderboard(
        where_filter={"innings": {"$eq": innings}},
        event_type=None,
        group_by="player",
        metric="runs_total",
    )
    top_batters = batter_board[:5] if batter_board else []

    # Top bowlers by wickets
    bowler_board = get_event_leaderboard(
        where_filter={"$and": [{"innings": {"$eq": innings}}, {"event": {"$eq": "wicket"}}]},
        event_type="wicket",
        group_by="player",
        metric="count",
    )
    top_bowlers = bowler_board[:3] if bowler_board else []

    # Count key events
    sixes_board = get_event_leaderboard(
        where_filter={"$and": [{"innings": {"$eq": innings}}, {"event": {"$eq": "six"}}]},
        event_type="six",
        group_by="innings",
        metric="count",
    )
    total_sixes = sixes_board[0]["count"] if sixes_board else 0

    fours_board = get_event_leaderboard(
        where_filter={"$and": [{"innings": {"$eq": innings}}, {"event": {"$eq": "four"}}]},
        event_type="four",
        group_by="innings",
        metric="count",
    )
    total_fours = fours_board[0]["count"] if fours_board else 0

    return {
        "innings": innings,
        "runs": total_runs,
        "wickets": total_wickets,
        "overs": overs_str,
        "run_rate": run_rate,
        "top_batters": top_batters,
        "top_bowlers": top_bowlers,
        "sixes": total_sixes,
        "fours": total_fours,
    }


def _format_innings_block(stats: dict) -> str:
    """Format one innings into a scorecard text block."""
    lines = [
        f"Innings {stats['innings']}: {stats['runs']}/{stats['wickets']} "
        f"in {stats['overs']} overs (RR: {stats['run_rate']})"
    ]

    if stats["top_batters"]:
        batters = ", ".join(
            f"{b['player']} ({b['count']} runs)" for b in stats["top_batters"]
        )
        lines.append(f"  Top batters: {batters}")

    if stats["top_bowlers"]:
        bowlers = ", ".join(
            f"{b['player']} ({b['count']} wickets)" for b in stats["top_bowlers"]
        )
        lines.append(f"  Top bowlers: {bowlers}")

    lines.append(
        f"  Key events: {stats['wickets']} wickets, "
        f"{stats['sixes']} sixes, {stats['fours']} fours"
    )
    return "\n".join(lines)


def _format_match_scorecard(inn1: dict, inn2: dict | None) -> str:
    """Format the full match scorecard stat block."""
    lines = ["=== MATCH SCORECARD ==="]
    lines.append(_format_innings_block(inn1))

    if inn2:
        lines.append("")
        lines.append(_format_innings_block(inn2))

    lines.append("===========================")
    return "\n".join(lines)


def handle_match_summary(state: RAGState) -> RAGState:
    """Handle match_summary questions with full scorecard and key moments."""
    question = state["rewritten_question"] or state["question"]

    # ── Stats: compute scorecard for both innings ─────────────────────────
    inn1_stats = _compute_innings_stats(1)
    inn2_stats = _compute_innings_stats(2)

    aggregate_stats = ""
    if inn1_stats:
        aggregate_stats = _format_match_scorecard(inn1_stats, inn2_stats)

    # ── Retrieval: key moments (wickets + sixes) across both innings ──────
    key_moments = get_sequential_deliveries(
        where={"$or": [{"event": {"$eq": "wicket"}}, {"event": {"$eq": "six"}}]},
        sort_direction="asc",
        limit=30,
    )

    # ── Context: scorecard + key highlights ───────────────────────────────
    lines = []
    if aggregate_stats:
        lines.append(aggregate_stats)
        lines.append("")

    if key_moments:
        lines.append("=== Key Match Moments ===")
        for index, doc in enumerate(key_moments, start=1):
            meta = doc["metadata"]
            header = format_delivery_header(meta, index)
            commentary = (
                meta.get("commentary")
                or doc["text"].split("Commentary:")[-1].strip()
            )
            lines.append(header)
            if commentary:
                lines.append(f"    Commentary: {commentary}")
            lines.append("")

    context = "\n".join(line for line in lines if line is not None).strip()
    if not context:
        context = "No relevant match data found."

    return {
        **state,
        "aggregate_stats": aggregate_stats or None,
        "retrieved_docs": key_moments,
        "initial_docs": key_moments,
        "query_variants": [question],
        "context": context,
        "answer_strategy": "hybrid",
    }
