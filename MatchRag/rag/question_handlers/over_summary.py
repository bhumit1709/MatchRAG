"""Handler for over_summary question type.

Handles both specific over queries and phase (powerplay/middle/death) queries.
"""

from __future__ import annotations

import re

from rag.graph_nodes import _PHASE_KEYWORDS
from rag.state import RAGState
from rag.vector_store import (
    format_phase_stats_block,
    get_match_metadata,
    get_phase_stats,
    get_sequential_deliveries,
)


def _detect_phase(question_lower: str) -> str | None:
    """Detect a T20 match phase from the question text."""
    for keyword in sorted(_PHASE_KEYWORDS, key=len, reverse=True):
        if keyword in question_lower:
            return _PHASE_KEYWORDS[keyword]
    return None


def _detect_over(question_lower: str) -> tuple[int | None, int | None]:
    """Detect a specific over number and optional innings from the question.

    Returns (over, innings) — either may be None.
    """
    if "last over" in question_lower or "final over" in question_lower:
        match_meta = get_match_metadata()
        if match_meta:
            return match_meta["max_over"], match_meta["max_innings"]
        return None, None

    over_match = re.search(r"\bover\s+(\d{1,2})\b", question_lower)
    if over_match:
        return int(over_match.group(1)), None

    return None, None


def _format_over_stats(deliveries: list[dict], over: int, innings: int | None) -> str:
    """Format stats for a specific over from fetched deliveries."""
    if not deliveries:
        return ""

    total_runs = sum(d["metadata"].get("runs_total", 0) for d in deliveries)
    total_wickets = sum(1 for d in deliveries if d["metadata"].get("event") == "wicket")
    total_balls = len(deliveries)

    batters = sorted(set(d["metadata"].get("batter", "?") for d in deliveries))
    bowlers = sorted(set(d["metadata"].get("bowler", "?") for d in deliveries))
    events = [str(d["metadata"].get("event", "?")).upper() for d in deliveries]

    inn_label = f"Innings {innings}, " if innings else ""
    balls_note = ""
    if total_balls < 6:
        balls_note = f" | {total_balls} balls (innings ended)"

    lines = [
        f"=== OVER STATS ===",
        f"{inn_label}Over {over}: {total_runs} runs | {total_wickets} wickets{balls_note}",
        f"  Deliveries: {', '.join(events)}",
        f"  Batter(s): {', '.join(batters)}",
        f"  Bowler: {', '.join(bowlers)}",
        f"==================",
    ]
    return "\n".join(lines)


def handle_over_summary(state: RAGState) -> RAGState:
    """Handle over_summary questions with sequential fetch or phase stats."""
    question = state["rewritten_question"] or state["question"]
    q_lower = question.lower().strip()

    # ── Phase path ────────────────────────────────────────────────────────
    phase = _detect_phase(q_lower)
    if phase is not None:
        # Detect optional innings filter
        innings = None
        inn_match = re.search(r"\binnings\s*(\d)\b", q_lower)
        if inn_match:
            innings = int(inn_match.group(1))

        phase_stats = get_phase_stats(phase=phase, innings=innings)
        aggregate_stats = format_phase_stats_block(phase_stats) if phase_stats else None

        # Retrieve key deliveries from this phase for context
        where_parts: list[dict] = [{"phase": {"$eq": phase}}]
        if innings is not None:
            where_parts.append({"innings": {"$eq": innings}})
        where = {"$and": where_parts} if len(where_parts) > 1 else where_parts[0]

        deliveries = get_sequential_deliveries(
            where=where,
            sort_direction="asc",
            limit=30,
        )

        # Build context
        context_lines = []
        if aggregate_stats:
            context_lines.append(aggregate_stats)
            context_lines.append("")

        if deliveries:
            context_lines.append("=== Phase Deliveries ===")
            for index, doc in enumerate(deliveries, start=1):
                meta = doc["metadata"]
                header = (
                    f"[{index}] Inn {meta.get('innings', '?')} | "
                    f"{meta.get('over', '?')}.{meta.get('ball', '?')} | "
                    f"Batter: {meta.get('batter', '?')} | "
                    f"Bowler: {meta.get('bowler', '?')} | "
                    f"Event: {str(meta.get('event', '?')).upper()}"
                )
                if meta.get("event") != "wicket":
                    header += f" | Runs: {meta.get('runs_total', '?')}"
                if meta.get("player_out"):
                    header += (
                        f" | OUT: {meta['player_out']} ({meta.get('wicket_kind', '')})"
                    )
                context_lines.append(header)

        context = "\n".join(line for line in context_lines if line is not None).strip()

        return {
            **state,
            "aggregate_stats": aggregate_stats,
            "retrieved_docs": deliveries,
            "initial_docs": deliveries,
            "query_variants": [question],
            "context": context or "No relevant match data found.",
            "answer_strategy": "sequential",
            "is_sequential": True,
        }

    # ── Specific over path ────────────────────────────────────────────────
    over, innings = _detect_over(q_lower)
    if over is None:
        # Fallback: couldn't parse which over — use general path behavior
        return {
            **state,
            "context": "No relevant match data found.",
            "answer_strategy": "sequential",
        }

    # Build filter for the specific over
    where_parts = [{"over": {"$eq": over}}]
    if innings is not None:
        where_parts.append({"innings": {"$eq": innings}})
    where = {"$and": where_parts} if len(where_parts) > 1 else where_parts[0]

    deliveries = get_sequential_deliveries(
        where=where,
        sort_direction="asc",
        limit=20,
    )

    # Compute over stats from deliveries
    actual_innings = innings
    if not actual_innings and deliveries:
        actual_innings = deliveries[0]["metadata"].get("innings")
    aggregate_stats = _format_over_stats(deliveries, over, actual_innings) if deliveries else None

    # Build context
    context_lines = []
    if aggregate_stats:
        context_lines.append(aggregate_stats)
        context_lines.append("")

    if deliveries:
        context_lines.append("=== Complete Chronological Delivery Sequence ===")
        if actual_innings is not None:
            context_lines.append(
                f"This is the complete set of recorded deliveries for "
                f"innings {actual_innings}, over {over}."
            )
        else:
            context_lines.append(
                f"This is the complete set of recorded deliveries for over {over}."
            )
        if len(deliveries) < 6:
            last_meta = deliveries[-1]["metadata"]
            context_lines.append(
                f"Only {len(deliveries)} deliveries are present because the innings ended at "
                f"{last_meta.get('over', '?')}.{last_meta.get('ball', '?')}."
            )
        context_lines.append("")

        for index, doc in enumerate(deliveries, start=1):
            meta = doc["metadata"]
            header = (
                f"[{index}] Inn {meta.get('innings', '?')} | "
                f"{meta.get('over', '?')}.{meta.get('ball', '?')} | "
                f"Batter: {meta.get('batter', '?')} | "
                f"Bowler: {meta.get('bowler', '?')} | "
                f"Event: {str(meta.get('event', '?')).upper()}"
            )
            if meta.get("event") != "wicket":
                header += f" | Runs: {meta.get('runs_total', '?')}"
            if meta.get("player_out"):
                header += (
                    f" | OUT: {meta['player_out']} ({meta.get('wicket_kind', '')})"
                )
            commentary = (
                meta.get("commentary")
                or doc["text"].split("Commentary:")[-1].strip()
            )
            context_lines.append(header)
            if commentary:
                context_lines.append(f"    Commentary: {commentary}")
            context_lines.append("")

    context = "\n".join(line for line in context_lines if line is not None).strip()

    return {
        **state,
        "retrieval_filters": where,
        "aggregate_stats": aggregate_stats,
        "retrieved_docs": deliveries,
        "initial_docs": deliveries,
        "query_variants": [question],
        "context": context or "No relevant match data found.",
        "answer_strategy": "sequential",
        "is_sequential": True,
    }
