"""LangGraph node implementations for the MatchRAG workflow."""

from __future__ import annotations

import difflib
import re
from typing import Generator

from rag.chains import invoke_answer_chain, rewrite_followup_question, stream_answer_chain
from rag.chains import build_retrieval_plan
from rag.schemas import RetrievalPlan
from rag.retrievers import retrieve_documents
from rag.state import RAGState
from rag.vector_store import (
    get_event_leaderboard,
    get_known_players,
    get_match_metadata,
    get_player_stats,
    get_sequential_deliveries,
)


_FOLLOWUP_SIGNALS = {
    "he hit",
    "he scored",
    "he score",
    "he took",
    "he bowled",
    "he got",
    "he made",
    "he played",
    "he faced",
    "she hit",
    "she scored",
    "they scored",
    "they won",
    "they made",
    "the bowler",
    "the batter",
    "the player",
    "the fielder",
    "the wicket",
    "that over",
    "that delivery",
    "that wicket",
    "that shot",
    "that ball",
    "this over",
    "this delivery",
    "this wicket",
    "same over",
    "same player",
    "what about him",
    "what about her",
    "what about them",
}

_FOLLOWUP_WORD_SIGNALS = {
    "that",
    "this",
    "those",
    "these",
    "him",
    "his",
    "her",
    "them",
    "their",
}

_STAT_SIGNAL_WORDS = {
    "how many",
    "most",
    "highest",
    "fewest",
    "total",
    "count",
    "runs",
}


def _needs_rewrite(question: str, has_history: bool) -> bool:
    if not has_history:
        return False

    q_lower = question.lower().strip()
    if any(signal in q_lower for signal in _FOLLOWUP_SIGNALS):
        return True

    words = [word.strip("?!.,;:") for word in q_lower.split()]
    return len(words) <= 7 and any(signal in words for signal in _FOLLOWUP_WORD_SIGNALS)


def rewrite_question(state: RAGState) -> RAGState:
    """Rewrite follow-up questions into standalone retrieval queries."""
    question = state["question"]
    history = state["chat_history"]

    if not _needs_rewrite(question, has_history=bool(history)):
        return {**state, "rewritten_question": question}

    rewritten, trace = rewrite_followup_question(question, history)
    llm_traces = state.get("llm_traces", []) + [trace]
    return {**state, "rewritten_question": rewritten, "llm_traces": llm_traces}


def _resolve_players(candidates: list[str], known_players: list[str]) -> list[str]:
    resolved: list[str] = []
    for candidate in candidates:
        if candidate in known_players:
            resolved.append(candidate)
            continue

        match = difflib.get_close_matches(candidate, known_players, n=1, cutoff=0.6)
        if match:
            resolved.append(match[0])
            continue

        last_name = candidate.split()[-1].lower()
        for known in known_players:
            if last_name and last_name in known.lower():
                resolved.append(known)
                break

    deduped: list[str] = []
    for player in resolved:
        if player not in deduped:
            deduped.append(player)
    return deduped


def _normalize_plan(plan: RetrievalPlan, question: str, known_players: list[str]) -> RetrievalPlan:
    plan = plan.model_copy(deep=True)
    plan.normalized_question = plan.normalized_question.strip() or question
    plan.players = _resolve_players(plan.players, known_players)

    if plan.over == "last":
        match_meta = get_match_metadata()
        if match_meta:
            plan.over = match_meta["max_over"]
            if plan.innings is None:
                plan.innings = match_meta["max_innings"]
        else:
            plan.over = None

    if plan.limit is not None and plan.limit <= 0:
        plan.limit = None

    # Over-scoped narrative questions are best answered from chronological
    # deliveries rather than compressed semantic highlights.
    if plan.over is not None and not plan.is_stat_question:
        plan.is_sequential = True
        if not plan.sort_direction:
            plan.sort_direction = "asc"

    return plan


def _build_where_filter(plan: RetrievalPlan) -> dict | None:
    clauses = []

    if plan.players:
        player_clauses = []
        for name in plan.players:
            player_clauses.extend(
                [
                    {"batter": {"$eq": name}},
                    {"bowler": {"$eq": name}},
                    {"player_out": {"$eq": name}},
                ]
            )
        clauses.append({"$or": player_clauses})

    if plan.event:
        clauses.append({"event": {"$eq": plan.event}})

    if plan.over is not None:
        clauses.append({"over": {"$eq": plan.over}})

    if plan.innings is not None:
        clauses.append({"innings": {"$eq": plan.innings}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _build_fast_path_plan(question: str) -> RetrievalPlan | None:
    question_lower = question.lower().strip()

    if any(signal in question_lower for signal in _STAT_SIGNAL_WORDS):
        return None

    if "last over" in question_lower or "final over" in question_lower:
        return RetrievalPlan(
            normalized_question=question,
            over="last",
            is_sequential=True,
        )

    over_match = re.search(r"\bover\s+(\d{1,2})\b", question_lower)
    if over_match:
        return RetrievalPlan(
            normalized_question=question,
            over=int(over_match.group(1)),
            is_sequential=True,
        )

    return None


def plan_retrieval(state: RAGState) -> RAGState:
    """Produce a structured retrieval plan and metadata filter."""
    question = state["rewritten_question"] or state["question"]
    known_players = get_known_players()
    fast_path_plan = _build_fast_path_plan(question)

    if fast_path_plan is not None:
        plan = _normalize_plan(fast_path_plan, question, known_players)
        llm_traces = state.get("llm_traces", [])
    elif known_players:
        plan, trace = build_retrieval_plan(question, known_players)
        plan = _normalize_plan(plan, question, known_players)
        llm_traces = state.get("llm_traces", []) + [trace]
    else:
        plan = RetrievalPlan(normalized_question=question)
        llm_traces = state.get("llm_traces", [])

    player_stats = [stats for player in plan.players if (stats := get_player_stats(player))]

    return {
        **state,
        "retrieval_plan": plan,
        "retrieval_filters": _build_where_filter(plan),
        "player_stats": player_stats or None,
        "aggregate_stats": None,
        "group_by": plan.group_by,
        "metric": plan.metric,
        "is_stat_question": plan.is_stat_question,
        "is_sequential": plan.is_sequential,
        "sort_direction": plan.sort_direction,
        "limit": plan.limit,
        "llm_traces": llm_traces,
    }


def compute_aggregate_stats(state: RAGState) -> RAGState:
    """Calculate exact stats for aggregate-style questions."""
    plan = state.get("retrieval_plan")
    if not plan or not plan.is_stat_question:
        return state

    where = state.get("retrieval_filters")
    leaderboard = get_event_leaderboard(
        where_filter=where,
        event_type=plan.event,
        group_by=plan.group_by,
        metric=plan.metric,
    )
    if not leaderboard:
        return state

    lines = ["=== SYSTEM CALCULATED EXACT STATS ==="]
    if plan.event:
        lines.append(f"Stat leaderboard for event '{plan.event.upper()}':")
    else:
        lines.append("Stat leaderboard:")

    for index, row in enumerate(leaderboard[:10], start=1):
        label = "items"
        if plan.metric == "runs_total":
            label = "runs"
        elif plan.metric == "impact":
            label = "impact score"
        elif plan.event == "six":
            label = "sixes"
        elif plan.event == "four":
            label = "fours"
        elif plan.event == "wicket":
            label = "wickets"
        elif plan.event:
            label = f"{plan.event}s"

        prefix = ""
        if plan.group_by == "over":
            parts = str(row["player"]).split("_")
            if len(parts) == 2:
                lines.append(f"{index}. Innings {parts[0]} Over {parts[1]} — {row['count']} {label}")
                continue
            prefix = "Over "
        elif plan.group_by == "innings":
            prefix = "Innings "

        if plan.metric == "impact":
            stats = get_player_stats(row["player"])
            if stats:
                runs = stats["batting"]["runs"]
                wickets = stats["bowling"]["wickets"]
                lines.append(
                    f"{index}. {prefix}{row['player']} — Runs: {runs} | "
                    f"Wickets: {wickets} | Impact Score: {row['count']}"
                )
                continue

        lines.append(f"{index}. {prefix}{row['player']} — {row['count']} {label}")

    lines.append("=====================================\n")
    return {**state, "aggregate_stats": "\n".join(lines)}


def retrieve(state: RAGState) -> RAGState:
    """Retrieve documents after retrieval planning and filter translation."""
    plan = state.get("retrieval_plan") or RetrievalPlan(normalized_question=state["question"])
    query_text = plan.normalized_question or state["rewritten_question"] or state["question"]
    where = state.get("retrieval_filters")

    if plan.is_sequential:
        limit = plan.limit or 20
        results = get_sequential_deliveries(
            where=where,
            sort_direction=plan.sort_direction,
            limit=limit,
        )
        return {
            **state,
            "query_variants": [query_text],
            "initial_docs": results,
            "retrieved_docs": results,
        }

    query_variants, initial_docs, retrieved_docs, trace = retrieve_documents(query_text, where=where)
    llm_traces = state.get("llm_traces", [])
    if trace is not None:
        llm_traces = llm_traces + [trace]

    return {
        **state,
        "query_variants": query_variants,
        "initial_docs": initial_docs,
        "retrieved_docs": retrieved_docs,
        "llm_traces": llm_traces,
    }


def build_context(state: RAGState) -> RAGState:
    """Format the retrieved deliveries and exact stats into the answer context."""
    docs = state["retrieved_docs"]
    if not docs:
        return {**state, "context": "No relevant match data found."}

    lines = []
    aggregate_stats = state.get("aggregate_stats")
    player_stats = state.get("player_stats")
    plan = state.get("retrieval_plan")

    def _format_player_stats(stats: dict) -> list[str]:
        batting = stats["batting"]
        bowling = stats["bowling"]
        output = [f"Player: {stats['name']}"]
        output.append(
            f" Batting: {batting['runs']} runs off {batting['balls']} balls "
            f"({batting['fours']} fours, {batting['sixes']} sixes)"
        )
        if batting["dismissal"]:
            output.append(f" Dismissal: {batting['dismissal']}")
        if bowling["overs"] != "0.0":
            output.append(
                f" Bowling: {bowling['wickets']} wickets for {bowling['runs']} runs "
                f"in {bowling['overs']} overs"
            )
        return output

    if player_stats:
        lines.append(aggregate_stats.strip() if aggregate_stats else "")
        lines.append("=== EXACT STATS FOR REQUESTED PLAYERS (ENTIRE MATCH) ===")
        for stats in player_stats:
            lines.extend(_format_player_stats(stats))
            lines.append("")
        lines.append("========================================================\n")
    elif aggregate_stats:
        lines.append(aggregate_stats)

    if state.get("is_sequential") and plan and plan.over is not None:
        lines.append("=== Complete Chronological Delivery Sequence ===")
        if plan.innings is not None:
            lines.append(
                f"This is the complete set of recorded deliveries for innings {plan.innings}, over {plan.over}."
            )
        else:
            lines.append(f"This is the complete set of recorded deliveries for over {plan.over}.")
        if len(docs) < 6:
            last_meta = docs[-1]["metadata"]
            lines.append(
                f"Only {len(docs)} deliveries are present because the innings ended at "
                f"{last_meta.get('over', '?')}.{last_meta.get('ball', '?')}."
            )
        lines.append("")
    else:
        lines.append("=== Relevant Match Highlight Deliveries ===")

    for index, doc in enumerate(docs, start=1):
        meta = doc["metadata"]
        header = (
            f"[{index}] Inn {meta.get('innings', '?')} | {meta.get('over', '?')}.{meta.get('ball', '?')} | "
            f"Batter: {meta.get('batter', '?')} | Bowler: {meta.get('bowler', '?')} | "
            f"Event: {str(meta.get('event', '?')).upper()}"
        )
        if meta.get("event") != "wicket":
            header += f" | Runs: {meta.get('runs_total', '?')}"
        if meta.get("player_out"):
            header += f" | OUT: {meta['player_out']} ({meta.get('wicket_kind', '')})"

        commentary = meta.get("commentary") or doc["text"].split("Commentary:")[-1].strip()
        lines.append(header)
        if commentary:
            lines.append(f"    Commentary: {commentary}")
        lines.append("")

    context = "\n".join(line for line in lines if line is not None)
    context = re.sub(r"\n{3,}", "\n\n", context).strip()
    return {**state, "context": context}


def generate_answer(state: RAGState) -> RAGState:
    """Run the final answer-generation chain."""
    answer, trace = invoke_answer_chain(
        question=state["question"],
        chat_history=state["chat_history"],
        context=state["context"],
        aggregate_stats=state.get("aggregate_stats"),
    )
    llm_traces = state.get("llm_traces", []) + [trace]
    return {**state, "answer": answer, "llm_traces": llm_traces}


def generate_answer_stream(state: RAGState) -> tuple[Generator[str, None, None], dict]:
    """Stream the final answer-generation chain and return the prompt trace."""
    stream, trace = stream_answer_chain(
        question=state["question"],
        chat_history=state["chat_history"],
        context=state["context"],
        aggregate_stats=state.get("aggregate_stats"),
    )
    return iter(stream), trace
