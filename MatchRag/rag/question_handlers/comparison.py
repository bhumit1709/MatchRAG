"""Handler for comparison question type.

Produces side-by-side player stats and dual retrieval for comparative analysis.
"""

from __future__ import annotations

from rag.retrievers import retrieve_documents
from rag.state import RAGState
from rag.vector_store import get_known_players, get_player_stats
from rag.question_handlers.utils import question_mentions_players, build_player_filter, format_delivery_header


def _format_comparison_block(stats_a: dict, stats_b: dict) -> str:
    """Format two players' stats into a side-by-side comparison block."""
    bat_a = stats_a["batting"]
    bat_b = stats_b["batting"]
    bowl_a = stats_a["bowling"]
    bowl_b = stats_b["bowling"]

    name_a = stats_a["name"]
    name_b = stats_b["name"]

    # Compute derived stats
    sr_a = round((bat_a["runs"] / bat_a["balls"]) * 100, 2) if bat_a["balls"] else 0.0
    sr_b = round((bat_b["runs"] / bat_b["balls"]) * 100, 2) if bat_b["balls"] else 0.0

    # Column widths
    col_w = max(len(name_a), len(name_b), 20)

    lines = ["=== PLAYER COMPARISON ==="]
    lines.append(f"{'':20s}{name_a:<{col_w}s}{name_b}")
    lines.append(f"{'Batting Runs:':20s}{bat_a['runs']:<{col_w}}{bat_b['runs']}")
    lines.append(f"{'Balls Faced:':20s}{bat_a['balls']:<{col_w}}{bat_b['balls']}")
    lines.append(f"{'Strike Rate:':20s}{sr_a:<{col_w}}{sr_b}")
    lines.append(f"{'Fours:':20s}{bat_a['fours']:<{col_w}}{bat_b['fours']}")
    lines.append(f"{'Sixes:':20s}{bat_a['sixes']:<{col_w}}{bat_b['sixes']}")

    dism_a = bat_a["dismissal"] or "not out"
    dism_b = bat_b["dismissal"] or "not out"
    lines.append(f"{'Dismissal:':20s}{dism_a:<{col_w}s}{dism_b}")

    # Bowling stats
    def _bowl_str(bowling: dict) -> tuple[str, str, str]:
        if bowling["overs"] == "0.0":
            return "—", "—", "—"
        parts = bowling["overs"].split(".")
        total_balls = int(parts[0]) * 6 + (int(parts[1]) if len(parts) > 1 else 0)
        econ = round((bowling["runs"] / (total_balls / 6)), 2) if total_balls else 0.0
        return str(bowling["wickets"]), str(bowling["runs"]), str(econ)

    w_a, r_a, e_a = _bowl_str(bowl_a)
    w_b, r_b, e_b = _bowl_str(bowl_b)
    lines.append(f"{'Bowling Wickets:':20s}{w_a:<{col_w}s}{w_b}")
    lines.append(f"{'Bowling Runs:':20s}{r_a:<{col_w}s}{r_b}")
    lines.append(f"{'Bowling Overs:':20s}{bowl_a['overs']:<{col_w}s}{bowl_b['overs']}")
    lines.append(f"{'Bowling Economy:':20s}{e_a:<{col_w}s}{e_b}")

    lines.append("=========================")
    return "\n".join(lines)


# _build_player_filter removed in favor of utils function


def _format_player_docs(docs: list[dict], player_name: str) -> list[str]:
    """Format delivery documents for one player."""
    lines = [f"=== KEY DELIVERIES: {player_name} ==="]
    for index, doc in enumerate(docs[:8], start=1):
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
    return lines


def handle_comparison(state: RAGState) -> RAGState:
    """Handle comparison questions with dual stats and dual retrieval."""
    question = state["rewritten_question"] or state["question"]
    known_players = get_known_players()
    mentioned_players = question_mentions_players(question, known_players)

    player_a = mentioned_players[0]
    player_b = mentioned_players[1]

    # ── Stats: exact stats for both players ───────────────────────────────
    stats_a = get_player_stats(player_a)
    stats_b = get_player_stats(player_b)

    aggregate_stats = ""
    player_stats_list = []
    if stats_a and stats_b:
        aggregate_stats = _format_comparison_block(stats_a, stats_b)
        player_stats_list = [stats_a, stats_b]
    elif stats_a:
        player_stats_list = [stats_a]
    elif stats_b:
        player_stats_list = [stats_b]

    # ── Retrieval: separate retrieval for each player ─────────────────────
    filter_a = build_player_filter(player_a)
    filter_b = build_player_filter(player_b)

    _, initial_a, docs_a, trace_a = retrieve_documents(
        f"{player_a} key deliveries",
        where=filter_a,
        enable_multi_query=False,
        enable_context_compression=True,
    )
    _, initial_b, docs_b, trace_b = retrieve_documents(
        f"{player_b} key deliveries",
        where=filter_b,
        enable_multi_query=False,
        enable_context_compression=True,
    )

    llm_traces = state.get("llm_traces", [])
    if trace_a is not None:
        llm_traces = llm_traces + [trace_a]
    if trace_b is not None:
        llm_traces = llm_traces + [trace_b]

    # ── Context: comparison block + per-player deliveries ─────────────────
    context_lines = []
    if aggregate_stats:
        context_lines.append(aggregate_stats)
        context_lines.append("")

    if docs_a:
        context_lines.extend(_format_player_docs(docs_a, player_a))
    if docs_b:
        context_lines.extend(_format_player_docs(docs_b, player_b))

    context = "\n".join(line for line in context_lines if line is not None).strip()
    if not context:
        context = "No relevant match data found."

    all_docs = docs_a + docs_b
    all_initial = initial_a + initial_b

    return {
        **state,
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": player_stats_list or None,
        "aggregate_stats": aggregate_stats or None,
        "query_variants": [f"{player_a} key deliveries", f"{player_b} key deliveries"],
        "initial_docs": all_initial,
        "retrieved_docs": all_docs,
        "context": context,
        "answer_strategy": "semantic",
        "llm_traces": llm_traces,
    }
