"""LangGraph node implementations for the MatchRAG workflow."""

from __future__ import annotations

import difflib
import re
from typing import Generator

from rag.chains import invoke_answer_chain, rewrite_followup_question, stream_answer_chain
from rag.chains import build_retrieval_plan
from rag.schemas import AnswerStrategy, RetrievalPlan
from rag.retrievers import retrieve_documents
from rag.state import RAGState
from rag.vector_store import (
    format_phase_stats_block,
    get_event_leaderboard,
    get_known_players,
    get_match_metadata,
    get_phase_stats,
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

_STRONG_PRONOUN_SIGNALS = {
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

_STAT_EVENT_KEYWORDS = {
    "sixes": "six",
    "six": "six",
    "fours": "four",
    "four": "four",
    "wickets": "wicket",
    "wicket": "wicket",
    "dots": "dot",
    "dot balls": "dot",
    "dot ball": "dot",
}

_ORDERED_EVENT_TERMS = {
    "wicket": "wicket",
    "wickets": "wicket",
    "six": "six",
    "sixes": "six",
    "four": "four",
    "fours": "four",
    "dot": "dot",
    "dot ball": "dot",
    "dot balls": "dot",
}

_SUMMARY_SIGNALS = {
    "performance",
    "perform",
    "how did",
    "how was",
    "summary",
    "won",
}

# Maps phase surface forms (lowercase) → canonical phase value stored in ChromaDB.
_PHASE_KEYWORDS: dict[str, str] = {
    "powerplay":    "powerplay",
    "power play":   "powerplay",
    "power-play":   "powerplay",
    "pp":           "powerplay",
    "first 6":      "powerplay",
    "first six":    "powerplay",
    "middle overs": "middle",
    "middle phase": "middle",
    "middle over":  "middle",
    "death overs":  "death",
    "death over":   "death",
    "death phase":  "death",
    "last 5":       "death",
    "last five":    "death",
    "final overs":  "death",
}

# For phase questions: which strategy to use based on question intent.
# Totals → aggregate | Performance/best → hybrid | Narrative → sequential
_PHASE_AGGREGATE_SIGNALS = {"how many", "total", "runs scored", "runs did", "wickets fell", "wickets taken"}
_PHASE_HYBRID_SIGNALS    = {"best", "worst", "who", "impact", "effective", "good", "bad", "performance"}


def _strategy_uses_aggregate(strategy: AnswerStrategy) -> bool:
    return strategy in {"aggregate", "hybrid"}


def _strategy_uses_sequential(strategy: AnswerStrategy) -> bool:
    return strategy == "sequential"


def _strategy_uses_semantic(strategy: AnswerStrategy) -> bool:
    return strategy in {"semantic", "hybrid"}


def _needs_rewrite(question: str, has_history: bool) -> bool:
    if not has_history:
        return False

    q_lower = question.lower().strip()
    if any(signal in q_lower for signal in _FOLLOWUP_SIGNALS):
        return True

    words = [word.strip("?!.,;:") for word in q_lower.split()]
    if any(signal in words for signal in _STRONG_PRONOUN_SIGNALS):
        return True
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
    if plan.over is not None and not _strategy_uses_aggregate(plan.answer_strategy):
        plan.answer_strategy = "sequential"
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

    if plan.phase:
        clauses.append({"phase": {"$eq": plan.phase}})

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


def _combine_filters(*filters: dict | None) -> dict | None:
    clauses: list[dict] = []
    for current in filters:
        if not current:
            continue
        if set(current.keys()) == {"$and"} and isinstance(current.get("$and"), list):
            clauses.extend(current["$and"])
        else:
            clauses.append(current)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _extract_ordered_event(question_lower: str) -> tuple[str, str] | None:
    if "first" in question_lower:
        direction = "asc"
    elif "last" in question_lower:
        direction = "desc"
    else:
        return None

    for term, event in _ORDERED_EVENT_TERMS.items():
        if term in question_lower:
            return event, direction

    return None


def _is_summary_question(question_lower: str) -> bool:
    # Use a word-boundary check for "over" so "overall" is not falsely excluded.
    if re.search(r"\bover\b", question_lower) or "first " in question_lower or "last " in question_lower:
        return False
    if any(signal in question_lower for signal in _STAT_SIGNAL_WORDS):
        return False
    return any(signal in question_lower for signal in _SUMMARY_SIGNALS)


def _build_fast_path_plan(question: str) -> RetrievalPlan | None:
    question_lower = question.lower().strip()

    # ── Phase detection ───────────────────────────────────────────────────────
    # Check for match-phase keywords before any other routing.
    # Multi-word phrases must be checked before single-word ones (e.g. "death
    # overs" before "death") so longer matches take precedence.
    detected_phase: str | None = None
    for keyword in sorted(_PHASE_KEYWORDS, key=len, reverse=True):
        if keyword in question_lower:
            detected_phase = _PHASE_KEYWORDS[keyword]
            break

    if detected_phase is not None:
        # Determine strategy from intent signals in the question
        if any(signal in question_lower for signal in _PHASE_AGGREGATE_SIGNALS):
            strategy: AnswerStrategy = "aggregate"
        elif any(signal in question_lower for signal in _PHASE_HYBRID_SIGNALS):
            strategy = "hybrid"
        elif "what happened" in question_lower or "over by over" in question_lower:
            strategy = "sequential"
        else:
            strategy = "aggregate"  # default: caller usually wants a number

        return RetrievalPlan(
            normalized_question=question,
            phase=detected_phase,
            answer_strategy=strategy,
        )

    ordered_event = _extract_ordered_event(question_lower)
    if ordered_event is not None:
        event, direction = ordered_event
        return RetrievalPlan(
            normalized_question=question,
            event=event,
            answer_strategy="sequential",
            sort_direction=direction,
            limit=1,
        )

    for keyword, event in _STAT_EVENT_KEYWORDS.items():
        if keyword not in question_lower:
            continue

        if any(phrase in question_lower for phrase in {"most", "highest", "fewest"}):
            return RetrievalPlan(
                normalized_question=question,
                event=event,
                answer_strategy="hybrid",
                group_by="player",
                metric="count",
            )

        if question_lower.startswith("who hit the most ") or question_lower.startswith("who took the most "):
            return RetrievalPlan(
                normalized_question=question,
                event=event,
                answer_strategy="hybrid",
                group_by="player",
                metric="count",
            )

    if "which over" in question_lower and any(phrase in question_lower for phrase in {"most runs", "highest runs"}):
        return RetrievalPlan(
            normalized_question=question,
            answer_strategy="hybrid",
            group_by="over",
            metric="runs_total",
        )

    if _is_summary_question(question_lower):
        return RetrievalPlan(
            normalized_question=question,
            answer_strategy="semantic",
        )

    if any(signal in question_lower for signal in _STAT_SIGNAL_WORDS):
        return None

    if "last over" in question_lower or "final over" in question_lower:
        return RetrievalPlan(
            normalized_question=question,
            answer_strategy="sequential",
            over="last",
            is_sequential=True,
        )

    over_match = re.search(r"\bover\s+(\d{1,2})\b", question_lower)
    if over_match:
        return RetrievalPlan(
            normalized_question=question,
            answer_strategy="sequential",
            over=int(over_match.group(1)),
            is_sequential=True,
        )

    return None


def _question_mentions_players(question: str, known_players: list[str]) -> list[str]:
    question_lower = question.lower()
    matches: list[str] = []

    for player in known_players:
        player_lower = player.lower()
        if player_lower in question_lower:
            matches.append(player)
            continue

        last_name = player_lower.split()[-1]
        if len(last_name) >= 4 and re.search(rf"\b{re.escape(last_name)}\b", question_lower):
            matches.append(player)

    deduped: list[str] = []
    for player in matches:
        if player not in deduped:
            deduped.append(player)
    return deduped


def _build_player_summary_plan(question: str, explicit_players: list[str]) -> RetrievalPlan:
    return RetrievalPlan(
        normalized_question=question,
        players=explicit_players,
        answer_strategy="semantic",
    )


def _leader_support_filter(plan: RetrievalPlan, row: dict, base_where: dict | None) -> dict | None:
    if plan.group_by == "player":
        leader = row["player"]
        if plan.metric == "impact":
            actor_filter = {
                "$or": [
                    {"batter": {"$eq": leader}},
                    {"bowler": {"$eq": leader}},
                ]
            }
        elif plan.event in {"wicket", "dot"}:
            actor_filter = {"bowler": {"$eq": leader}}
        elif plan.event in {"six", "four", "single", "run"} or plan.event is None:
            actor_filter = {"batter": {"$eq": leader}}
        else:
            actor_filter = {"$or": [{"batter": {"$eq": leader}}, {"bowler": {"$eq": leader}}]}
        return _combine_filters(base_where, actor_filter)

    if plan.group_by == "over":
        innings, _, over = str(row["player"]).partition("_")
        if innings and over:
            return _combine_filters(
                base_where,
                {"innings": {"$eq": int(innings)}},
                {"over": {"$eq": int(over)}},
            )
        return base_where

    if plan.group_by == "innings":
        try:
            return _combine_filters(base_where, {"innings": {"$eq": int(row["player"])}})
        except (TypeError, ValueError):
            return base_where

    if plan.group_by == "wicket_kind":
        return _combine_filters(base_where, {"wicket_kind": {"$eq": row["player"]}})

    return base_where


def _leader_support_query(plan: RetrievalPlan, row: dict, question: str) -> str:
    if plan.group_by == "player":
        label = _metric_label(plan)
        return f"{row['player']} {label}"
    if plan.group_by == "over":
        innings, _, over = str(row["player"]).partition("_")
        if innings and over:
            return f"Innings {innings} over {over} {question}"
    if plan.group_by == "innings":
        return f"Innings {row['player']} {question}"
    if plan.group_by == "wicket_kind":
        return f"{row['player']} {question}"
    return question


def plan_retrieval(state: RAGState) -> RAGState:
    """Produce a structured retrieval plan and metadata filter."""
    question = state["rewritten_question"] or state["question"]
    fast_path_plan = _build_fast_path_plan(question)

    if fast_path_plan is not None:
        # For semantic fast-path plans, enrich with explicit players if present.
        # This covers "How was SV Samson's performance?" hitting the summary fast-path
        # ─ the returned plan has no players without this step.
        if fast_path_plan.answer_strategy == "semantic":
            known_players = get_known_players()
            explicit_players = _question_mentions_players(question, known_players)
            if explicit_players:
                fast_path_plan = _build_player_summary_plan(question, explicit_players)
            plan = _normalize_plan(fast_path_plan, question, known_players if explicit_players else [])
        else:
            plan = _normalize_plan(fast_path_plan, question, [])
        llm_traces = state.get("llm_traces", [])
    else:
        known_players = get_known_players()
        explicit_players = _question_mentions_players(question, known_players)
        question_lower = question.lower().strip()

        if explicit_players and _is_summary_question(question_lower):
            plan = _normalize_plan(
                _build_player_summary_plan(question, explicit_players),
                question,
                known_players,
            )
            llm_traces = state.get("llm_traces", [])
        elif known_players:
            plan, trace = build_retrieval_plan(question, known_players)
            plan = _normalize_plan(plan, question, known_players)
            # If the LLM planner routed a player-summary question to aggregate/hybrid,
            # override it back to semantic so the answer is player-scoped narrative.
            if explicit_players and _is_summary_question(question_lower) and plan.answer_strategy in {"aggregate", "hybrid"}:
                plan = _normalize_plan(
                    _build_player_summary_plan(question, explicit_players),
                    question,
                    known_players,
                )
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
        "answer_strategy": plan.answer_strategy,
        "is_stat_question": plan.is_stat_question,
        "is_sequential": plan.is_sequential,
        "sort_direction": plan.sort_direction,
        "limit": plan.limit,
        "llm_traces": llm_traces,
    }


def compute_aggregate_stats(state: RAGState) -> RAGState:
    """Calculate exact stats for aggregate-style questions."""
    plan = state.get("retrieval_plan")
    if not plan or not _strategy_uses_aggregate(plan.answer_strategy):
        return state

    # ── Phase stats path ─────────────────────────────────────────────────────
    # When the question targets a match phase, use get_phase_stats() which
    # produces richer phase-scoped aggregates (run rate, top batters/bowlers).
    if plan.phase:
        stats = get_phase_stats(
            phase=plan.phase,
            innings=plan.innings,
            event_type=plan.event,
        )
        if stats:
            return {
                **state,
                "aggregate_stats": format_phase_stats_block(stats),
                "aggregate_rows": stats.get("top_batters", []),
            }
        return state

    # ── Event leaderboard path (existing) ────────────────────────────────────
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
    return {**state, "aggregate_stats": "\n".join(lines), "aggregate_rows": leaderboard}


def _metric_label(plan: RetrievalPlan) -> str:
    if plan.metric == "runs_total":
        return "runs"
    if plan.metric == "impact":
        return "impact score"
    if plan.event == "six":
        return "sixes"
    if plan.event == "four":
        return "fours"
    if plan.event == "wicket":
        return "wickets"
    if plan.event:
        return f"{plan.event}s"
    return "items"


def retrieve(state: RAGState) -> RAGState:
    """Retrieve documents after retrieval planning and filter translation."""
    plan = state.get("retrieval_plan") or RetrievalPlan(normalized_question=state["question"])
    query_text = plan.normalized_question or state["rewritten_question"] or state["question"]
    where = state.get("retrieval_filters")
    aggregate_rows = state.get("aggregate_rows") or []

    if not _strategy_uses_semantic(plan.answer_strategy) and _strategy_uses_aggregate(plan.answer_strategy):
        return {
            **state,
            "query_variants": [query_text],
            "initial_docs": [],
            "retrieved_docs": [],
        }

    if _strategy_uses_sequential(plan.answer_strategy):
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

    if plan.answer_strategy == "hybrid" and aggregate_rows:
        leader = aggregate_rows[0]
        support_where = _leader_support_filter(plan, leader, where)
        support_query = _leader_support_query(plan, leader, query_text)

        if plan.group_by == "over":
            results = get_sequential_deliveries(
                where=support_where,
                sort_direction="asc",
                limit=20,
            )
            return {
                **state,
                "query_variants": [support_query],
                "initial_docs": results,
                "retrieved_docs": results,
            }

        query_variants, initial_docs, retrieved_docs, trace = retrieve_documents(
            support_query,
            where=support_where,
            enable_multi_query=False,
            enable_context_compression=False,
        )
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

    query_variants, initial_docs, retrieved_docs, trace = retrieve_documents(
        query_text,
        where=where,
        enable_multi_query=_strategy_uses_semantic(plan.answer_strategy),
        enable_context_compression=_strategy_uses_semantic(plan.answer_strategy),
    )
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

    if not docs:
        context = "\n".join(line for line in lines if line is not None).strip()
        if context:
            return {**state, "context": context}
        return {**state, "context": "No relevant match data found."}

    if plan and plan.answer_strategy == "sequential" and plan.over is not None:
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
