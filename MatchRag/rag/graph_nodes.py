import json
import re
from typing import Generator

from config import TOP_K, INITIAL_TOP_K
from rag.state import RAGState
from rag.prompts import SYSTEM_PROMPT
from rag.llm_services import call_rewrite_llm, call_extract_llm, call_chat_llm, call_chat_llm_stream
from rag.vector_store import query as vector_query, get_known_players, get_player_stats, get_match_metadata, get_event_leaderboard
from rag.reranker import rerank_documents

# ---------------------------------------------------------------------------
# Heuristic: detect follow-up questions that need rewriting
# ---------------------------------------------------------------------------

_FOLLOWUP_SIGNALS = {
    # Pronoun + verb combos (reliable follow-up indicators regardless of length)
    "he hit", "he scored", "he score", "he took", "he bowled", "he got",
    "he made", "he played", "he faced",
    "she hit", "she scored",
    "they scored", "they won", "they made",
    # Specific reference markers
    "the bowler", "the batter", "the player", "the fielder", "the wicket",
    "that over", "that delivery", "that wicket", "that shot", "that ball",
    "this over", "this delivery", "this wicket",
    "same over", "same player",
    "what about him", "what about her", "what about them",
}

_FOLLOWUP_WORD_SIGNALS = {
    # Single-word signals that only fire when question is short (≤ 7 words)
    "that", "this", "those", "these", "him", "his", "her", "them", "their",
}

def _needs_rewrite(question: str, has_history: bool) -> bool:
    """Return True if the question is a follow-up that needs context to resolve."""
    if not has_history:
        return False
    q_lower = question.lower().strip()

    # Multi-word signals are reliable regardless of question length
    if any(signal in q_lower for signal in _FOLLOWUP_SIGNALS):
        return True

    # Single-word signals only apply for short, context-dependent questions
    words = [w.strip("?!.,;:") for w in q_lower.split()]
    if len(words) <= 7 and any(signal in words for signal in _FOLLOWUP_WORD_SIGNALS):
        return True

    return False


# ---------------------------------------------------------------------------
# Node 0 — Rewrite question (only for follow-ups)
# ---------------------------------------------------------------------------

def rewrite_question(state: RAGState) -> RAGState:
    """
    Use the LLM to resolve pronouns/references in follow-up questions.
    Skips the LLM call entirely if the question appears standalone.
    """
    question = state["question"]
    history  = state["chat_history"]

    if not _needs_rewrite(question, has_history=bool(history)):
        return {**state, "rewritten_question": question}

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
    )
    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Latest question: {question}"
    )

    rewritten = call_rewrite_llm(prompt).strip().strip('"')

    # Strip any parenthetical asides the LLM might add despite instructions
    rewritten = re.sub(r'\s*\(.*?\)\s*$', '', rewritten).strip()

    # If LLM returned empty or multi-line, fall back to original question
    if not rewritten or '\n' in rewritten:
        rewritten = question

    trace = {
        "node": "rewrite_question",
        "prompt": prompt,
        "response": rewritten,
    }
    llm_traces = state.get("llm_traces", []) + [trace]

    return {**state, "rewritten_question": rewritten, "llm_traces": llm_traces}


# ---------------------------------------------------------------------------
# Node 1 — Extract filters (LLM-based entity extraction)
# ---------------------------------------------------------------------------

def _build_where_filter(players: list[str], event: str | None, over: int | None, innings: int | None) -> dict | None:
    """Convert extracted entities into a ChromaDB where-clause."""
    clauses = []

    if players:
        # Search across batter, bowler, and player_out for each player
        player_clauses = []
        for name in players:
            player_clauses.extend([
                {"batter":     {"$eq": name}},
                {"bowler":     {"$eq": name}},
                {"player_out": {"$eq": name}},
            ])
        if len(player_clauses) == 1:
            clauses.append(player_clauses[0])
        else:
            clauses.append({"$or": player_clauses})

    if event:
        clauses.append({"event": {"$eq": event}})

    if over is not None:
        clauses.append({"over": {"$eq": over}})
        
    if innings is not None:
        clauses.append({"innings": {"$eq": innings}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def extract_filters(state: RAGState) -> RAGState:
    """
    Use the LLM to extract player names and event type from the question,
    then build a ChromaDB metadata where-filter for precise retrieval.
    Falls back to None (unfiltered) if extraction fails or finds nothing.
    """
    question = state["rewritten_question"] or state["question"]
    known    = get_known_players()

    if not known:
        return {**state, "retrieval_filters": None}

    prompt = (
        f"Known players: {', '.join(known)}\n"
        f"Question: {question}"
    )

    try:
        raw = call_extract_llm(prompt).strip()
        
        # Robustly extract just the FIRST complete JSON object using brace counting
        start = raw.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == '{':
                    depth += 1
                elif raw[i] == '}':
                    depth -= 1
                if depth == 0:
                    raw = raw[start:i+1]
                    break
            
        extracted = json.loads(raw)
        
        # Fuzzy match players if the LLM didn't return exact KNOWN player name
        llm_players = extracted.get("players", [])
        players = []
        for p in llm_players:
            if p in known:
                players.append(p)
            else:
                last_name = p.split()[-1] if ' ' in p else p
                for kp in known:
                    if last_name.lower() in kp.lower():
                        if kp not in players:
                            players.append(kp)
                            
        event   = extracted.get("event") or None
        
        over_val = extracted.get("over")
        innings_val = extracted.get("innings")
        
        if over_val == "last" or over_val == "final":
            match_meta = get_match_metadata()
            if match_meta:
                over_val = match_meta["max_over"]
                if not innings_val:
                    innings_val = match_meta["max_innings"]
            else:
                over_val = None
        elif isinstance(over_val, str):
            try:
                over_val = int(over_val)
            except ValueError:
                over_val = None

        if isinstance(innings_val, str):
            try:
                innings_val = int(innings_val)
            except ValueError:
                innings_val = None

        where   = _build_where_filter(players, event, over_val, innings_val)
        
        stats = []
        if players:
            for p in players:
                p_stat = get_player_stats(p)
                if p_stat:
                    stats.append(p_stat)
        if not stats:
            stats = None
        
        is_stat = bool(extracted.get("is_stat_question", False))
        group_by = extracted.get("group_by", "player")
        metric   = extracted.get("metric", "count")
        
        is_seq = bool(extracted.get("is_sequential", False))
        sort_dir = extracted.get("sort_direction", "asc")
        limit_val = extracted.get("limit")
        if isinstance(limit_val, str):
            try:
                limit_val = int(limit_val)
            except ValueError:
                limit_val = None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Extraction failed. Raw output was: {raw if 'raw' in locals() else 'unbound'}")
        where = None
        stats = None
        is_stat = False
        group_by = "player"
        metric = "count"
        is_seq = False
        sort_dir = "asc"
        limit_val = None

    trace = {
        "node": "extract_filters",
        "prompt": prompt,
        "response": raw if 'raw' in locals() else "FAILED",
    }
    llm_traces = state.get("llm_traces", []) + [trace]

    return {
        **state,
        "retrieval_filters": where,
        "player_stats": stats,
        "aggregate_stats": None,
        "is_stat_question": is_stat,
        "group_by": group_by,
        "metric": metric,
        "is_sequential": is_seq,
        "sort_direction": sort_dir,
        "limit": limit_val,
        "llm_traces": llm_traces
    }


# ---------------------------------------------------------------------------
# Node 2 — Compute Aggregate Stats
# ---------------------------------------------------------------------------

def compute_aggregate_stats(state: RAGState) -> RAGState:
    """
    If the question is a stat question, bypass vector search and deterministically
    compute the aggregate leaderboard from ChromaDB metadata.
    """
    if not state.get("is_stat_question"):
        return state

    where = state.get("retrieval_filters")
    
    # Extract event type from the filter manually
    event_type = None
    if where and isinstance(where, dict):
        if "event" in where and isinstance(where["event"], dict):
            event_type = where["event"].get("$eq")
        elif "$and" in where:
            for clause in where["$and"]:
                if "event" in clause and isinstance(clause["event"], dict):
                    event_type = clause["event"].get("$eq")
                    
    leaderboard = get_event_leaderboard(
        where_filter=where, 
        event_type=event_type,
        group_by=state.get("group_by", "player"),
        metric=state.get("metric", "count")
    )
    
    if not leaderboard:
        return state
        
    # Format the stats
    lines = ["=== SYSTEM CALCULATED EXACT STATS ==="]
    if event_type:
        lines.append(f"Stat leaderboard for event '{event_type.upper()}':")
    else:
        lines.append("Stat leaderboard:")
        
    for i, row in enumerate(leaderboard[:10], start=1):  # top 10 is enough
        if state.get("metric") == "runs_total":
            label = "runs"
        elif state.get("metric") == "impact":
            label = "impact score"
        elif event_type:
            if event_type.lower() == "six":
                label = "sixes"
            elif event_type.endswith("s"):
                label = event_type
            else:
                label = f"{event_type}s"
        else:
            label = "deliveries involved in"
            
        prefix = ""
        if state.get("group_by") == "over":
            parts = str(row['player']).split('_')
            if len(parts) == 2:
                lines.append(f"{i}. Innings {parts[0]} Over {parts[1]} — {row['count']} {label}")
                continue
            else:
                prefix = "Over "
        elif state.get("group_by") == "innings":
            prefix = "Innings "
            
        # For impact metrics, pull their exact match stats directly so the LLM has them without hallucinating
        if state.get("metric") == "impact":
            p_stat = get_player_stats(row['player'])
            if p_stat:
                runs = p_stat['batting']['runs']
                wkts = p_stat['bowling']['wickets']
                lines.append(f"{i}. {prefix}{row['player']} — Runs: {runs} | Wickets: {wkts} | Impact Score: {row['count']}")
                continue

        lines.append(f"{i}. {prefix}{row['player']} — {row['count']} {label}")
        
    lines.append("=====================================\n")
    # Inject the top contenders into the retrieval filters ONLY if no specific player was requested.
    # We use the top 3 to avoid confirmation bias (so the LLM sees highlights from multiple top players).
    if not state.get("player_stats") and leaderboard:
        group_by = state.get("group_by", "player")
        player_clauses = []
        
        for top_player_dict in leaderboard[:3]:
            top_player = top_player_dict["player"]
            if group_by == "over":
                parts = str(top_player).split('_')
                if len(parts) == 2:
                    try:
                        innings_val = int(parts[0])
                        over_val = int(parts[1])
                        player_clauses.append({"$and": [{"innings": {"$eq": innings_val}}, {"over": {"$eq": over_val}}]})
                    except ValueError:
                        player_clauses.append({"over": {"$eq": top_player}})
                else:
                    try:
                        player_clauses.append({"over": {"$eq": int(float(top_player))}})
                    except ValueError:
                        player_clauses.append({"over": {"$eq": top_player}})
            elif group_by == "innings":
                try:
                    player_clauses.append({"innings": {"$eq": int(float(top_player))}})
                except ValueError:
                    player_clauses.append({"innings": {"$eq": top_player}})
            elif group_by == "wicket_kind":
                player_clauses.append({"wicket_kind": {"$eq": top_player}})
            else:
                filter_key = "batter"
                if event_type in ("wicket", "dot"):
                    filter_key = "bowler"
                player_clauses.append({filter_key: {"$eq": top_player}})
        
        if len(player_clauses) == 1:
            player_filter = player_clauses[0]
        elif len(player_clauses) > 1:
            player_filter = {"$or": player_clauses}
        else:
            player_filter = None
            
        if player_filter:
            if where is None:
                where = player_filter
            elif "$and" in where:
                where["$and"].append(player_filter)
            else:
                where = {"$and": [where, player_filter]}
    
    return {**state, "aggregate_stats": "\n".join(lines), "retrieval_filters": where}


# ---------------------------------------------------------------------------
# Node 3 — Retrieve
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    """
    Query the ChromaDB vector store using the (possibly rewritten) question,
    with optional metadata filters from extract_filters.
    If the question is sequential, fetch exact matches sorted chronologically.
    """
    query_text = state["rewritten_question"] or state["question"]
    where      = state.get("retrieval_filters")
    
    if state.get("is_sequential"):
        from rag.vector_store import get_sequential_deliveries
        limit = state.get("limit") or 20
        results = get_sequential_deliveries(where=where, sort_direction=state.get("sort_direction", "asc"), limit=limit)
    else:
        results    = vector_query(query_text, n_results=INITIAL_TOP_K, where=where)
        
    return {**state, "retrieved_docs": results, "initial_docs": results}


# ---------------------------------------------------------------------------
# Node 4 — Rerank Docs
# ---------------------------------------------------------------------------

def rerank_docs(state: RAGState) -> RAGState:
    """Rerank retrieved documents using FlashRank."""
    if state.get("is_sequential"):
        return state
        
    docs = state["retrieved_docs"]
    query_text = state["rewritten_question"] or state["question"]
    reranked = rerank_documents(query_text, docs, top_n=TOP_K)
    return {**state, "retrieved_docs": reranked}


# ---------------------------------------------------------------------------
# Node 5 — Build context
# ---------------------------------------------------------------------------

def build_context(state: RAGState) -> RAGState:
    """
    Format the retrieved delivery documents into a context block.
    """
    docs = state["retrieved_docs"]
    if not docs:
        return {**state, "context": "No relevant match data found."}

    lines = []
    
    # Inject exact deterministic stats if available
    agg_stats = state.get("aggregate_stats")
    p_stats = state.get("player_stats")
    
    def _format_player_stats(p_stats):
        lines = [f"Player: {p_stats['name']}"]
        b = p_stats['batting']
        lines.append(f" Batting: {b['runs']} runs off {b['balls']} balls ({b['fours']} fours, {b['sixes']} sixes)")
        if b['dismissal']:
            lines.append(f" Dismissal: {b['dismissal']}")
        w = p_stats['bowling']
        if w['overs'] != "0.0":
            lines.append(f" Bowling: {w['wickets']} wickets for {w['runs']} runs in {w['overs']} overs")
        return lines

    if p_stats: # Prioritize the specific requested players' exact stats
        new_agg_stats_lines = []
        
        if agg_stats:
            new_agg_stats_lines.append(agg_stats.strip())
            new_agg_stats_lines.append("")
            
        new_agg_stats_lines.append("=== EXACT STATS FOR REQUESTED PLAYERS (ENTIRE MATCH) ===")
        for p_stat in p_stats:
            new_agg_stats_lines.extend(_format_player_stats(p_stat))
            new_agg_stats_lines.append("")
            
        if new_agg_stats_lines[-1] == "":
            new_agg_stats_lines.pop()
        new_agg_stats_lines.append("========================================================\n")
        lines.extend(new_agg_stats_lines)
        state["aggregate_stats"] = "\n".join(new_agg_stats_lines)
    elif agg_stats: 
        new_agg_stats_lines = [agg_stats]
        # INJECT INDIVIDUAL STATS FOR TOP PLAYERS TO AVOID HALLUCINATION
        leaderboard_match = re.search(r'Stat leaderboard.*:\n(.*?)======', agg_stats, re.DOTALL)
        if leaderboard_match:
            new_agg_stats_lines.append("=== EXACT STATS FOR TOP CONTENDERS ===")
            top_players = []
            for row in leaderboard_match.group(1).strip().split('\n')[:3]:
                # Extract player name from something like "1. SV Samson — 51 runs"
                parts = row.split(' — ')[0].split('. ')
                if len(parts) > 1:
                    top_players.append(parts[1].strip())
            
            for p_name in top_players:
                stats = get_player_stats(p_name)
                if stats:
                    new_agg_stats_lines.extend(_format_player_stats(stats))
                    new_agg_stats_lines.append("")
            
            if new_agg_stats_lines[-1] == "":
                new_agg_stats_lines.pop()
            new_agg_stats_lines.append("======================================\n")
        lines.extend(new_agg_stats_lines)
        state["aggregate_stats"] = "\n".join(new_agg_stats_lines)

    lines.append("=== Relevant Match Highlight Deliveries ===")

    for i, doc in enumerate(docs, start=1):
        m    = doc["metadata"]
        inn  = m.get("innings", "?")
        over = m.get("over", "?")
        ball = m.get("ball", "?")
        batter  = m.get("batter", "?")
        bowler  = m.get("bowler", "?")
        event   = m.get("event", "?")
        runs    = m.get("runs_total", "?")

        header = (
            f"[{i}] Inn {inn} | {over}.{ball} | "
            f"Batter: {batter} | Bowler: {bowler} | Event: {event.upper()}"
        )
        if event != 'wicket':
             header += f" | Runs: {runs}"
             
        if m.get("player_out"):
            header += f" | OUT: {m['player_out']} ({m.get('wicket_kind', '')})"

        lines.append(header)
        # Prefer raw commentary metadata field, fall back to text-extracted
        raw_commentary = m.get("commentary") or doc['text'].split('Commentary:')[-1].strip()
        if raw_commentary and raw_commentary != doc['text']:
            lines.append(f"    Commentary: {raw_commentary}")
        lines.append("")

    return {**state, "context": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Node 6 — Generate answer (with streaming)
# ---------------------------------------------------------------------------

def build_messages(state: RAGState) -> list[dict]:
    """Build the full message list for the LLM, including history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject prior conversation turns
    if state["chat_history"]:
        messages.extend(state["chat_history"])

    # Append the current context + question as the new user turn
    user_content = f"{state['context']}\n\nQuestion: {state['question']}"
    messages.append({"role": "user", "content": user_content})

    # Optimization 5: Assistant Framing Prefill
    messages.append({"role": "assistant", "content": "Based on the match data provided:"})
    return messages


def generate_answer(state: RAGState) -> RAGState:
    """
    Non-streaming answer generation (used by the LangGraph .invoke() path
    and the CLI chat.py).
    """
    answer = call_chat_llm(build_messages(state)).strip()
    return {**state, "answer": answer}


def generate_answer_stream(state: RAGState) -> Generator[str, None, str]:
    """
    Streaming answer generation — yields tokens as they arrive from Ollama.
    Returns the full assembled answer string when exhausted.

    Usage:
        gen = generate_answer_stream(state)
        for token in gen:
            send_to_client(token)
    """
    full_answer = []
    for token in call_chat_llm_stream(build_messages(state)):
        full_answer.append(token)
        yield token

    return "".join(full_answer)
