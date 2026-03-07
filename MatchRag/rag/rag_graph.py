"""
rag_graph.py
------------
Steps 7 & 8 of the RAG pipeline.

Implements a LangGraph StateGraph with four nodes:
  rewrite_question → retrieve → build_context → generate_answer

All LLM inference runs locally through Ollama (mistral or llama3).
Supports session memory (chat history) and streaming token output.
"""

import json
import re
from typing import TypedDict, Generator
from langgraph.graph import StateGraph, END
import ollama
from rag.vector_store import query as vector_query, get_known_players, get_player_stats, get_match_metadata, get_event_leaderboard
from config import LLM_MODEL, TOP_K, INITIAL_TOP_K
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
# State schema
# ---------------------------------------------------------------------------

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
    initial_docs:       list[dict]   # Before reranking
    retrieved_docs:     list[dict]   # After reranking
    context:            str
    answer:             str


# ---------------------------------------------------------------------------
# Node 0 — Rewrite question (only for follow-ups)
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """\
You are a question rewriter for a cricket match Q&A system.

TASK: Replace pronouns and vague references in the user's latest question \
with the specific names, overs, or events from the conversation history.

RULES:
1. ONLY replace pronouns (he, she, they, him, his, that, this, etc.) with \
the concrete entity from the conversation.
2. Keep the question short and specific — one sentence maximum.
3. Preserve the context (match, series, topic) established in the conversation.
4. Do NOT add assumptions, caveats, or explanations.
5. Do NOT change the intent of the question.
6. Output ONLY the rewritten question — nothing else.

EXAMPLE:
History: "Who dismissed Hetmyer?" → "Hetmyer was dismissed by JJ Bumrah"
Latest: "How many sixes did he hit?"
Output: How many sixes did Shimron Hetmyer hit?"""


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

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _REWRITE_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    rewritten = response["message"]["content"].strip().strip('"')

    # Strip any parenthetical asides the LLM might add despite instructions
    import re
    rewritten = re.sub(r'\s*\(.*?\)\s*$', '', rewritten).strip()

    # If LLM returned empty or multi-line, fall back to original question
    if not rewritten or '\n' in rewritten:
        rewritten = question

    return {**state, "rewritten_question": rewritten}


# ---------------------------------------------------------------------------
# Node 1 — Extract filters (LLM-based entity extraction)
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are an entity extractor for a cricket match Q&A system.

Given the question and the list of known player names, extract:
- "players": list of exact player names from the 'Known players' list below that match the players mentioned in the question. You MUST use the exact spelling from the known list (e.g. if question says 'Shimron Hetmyer' and known list has 'SO Hetmyer', output 'SO Hetmyer').
- "event": one of ["wicket", "six", "four", "dot", "single", "run"] if the question targets a specific event. ONLY use "run" if they ask about running between wickets (1s, 2s, 3s). If they ask for "total runs", set this to null!
- "over": integer if a specific over is mentioned (e.g. 11 for "12th over"), or the string "last" ONLY IF they explicitly say "last over" or "final over". Else null.
- "innings": integer if a specific innings is mentioned (1 or 2), else null.
- "is_stat_question": boolean True ONLY if the user asks for an aggregate calculation across the match like "most", "highest", "total count", "leaderboard", or "who scored the most". False if they ask about a specific event (e.g. "Who dismissed X?", "What happened in the 5th over?").
- "group_by": one of ["player", "over", "innings", "wicket_kind"]. Default is "player". If the question asks "Which over...", use "over". If "Which team...", use "innings".
- "metric": one of ["count", "runs_total"]. Use "count" to count events (e.g., most sixes, most wickets). Use "runs_total" to sum up runs (e.g., most runs, highest run scorer).

RULES:
1. Extract any player name mentioned in the question and map it to the closest name in the Known players list.
2. If no player is mentioned, return empty list for players.
3. Return ONLY valid JSON. No explanation.

EXAMPLE 1:
Known players: RG Sharma, SO Hetmyer, JJ Bumrah
Question: "Who hit the most sixes?"
Output: {"players": [], "event": "six", "over": null, "innings": null, "is_stat_question": true, "group_by": "player", "metric": "count"}

EXAMPLE 2:
Question: "Show all wickets taken by Bumrah."
Output: {"players": ["JJ Bumrah"], "event": "wicket", "over": null, "innings": null, "is_stat_question": true, "group_by": "player", "metric": "count"}

EXAMPLE 3:
Question: "Which over had the most runs?"
Output: {"players": [], "event": null, "over": null, "innings": null, "is_stat_question": true, "group_by": "over", "metric": "runs_total"}

EXAMPLE 4:
Known players: SO Hetmyer
Question: "Who dismissed Shimron Hetmyer?"
Output: {"players": ["SO Hetmyer"], "event": "wicket", "over": null, "innings": null, "is_stat_question": false, "group_by": "player", "metric": "count"}"""


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
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response["message"]["content"].strip()
        
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
        stats   = get_player_stats(players[0]) if players else None
        
        is_stat = bool(extracted.get("is_stat_question", False))
        group_by = extracted.get("group_by", "player")
        metric   = extracted.get("metric", "count")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Extraction failed. Raw output was: {raw if 'raw' in locals() else 'unbound'}")
        where = None
        stats = None
        is_stat = False
        group_by = "player"
        metric = "count"

    return {
        **state,
        "retrieval_filters": where,
        "player_stats": stats,
        "aggregate_stats": None,
        "is_stat_question": is_stat,
        "group_by": group_by,
        "metric": metric
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
        label = "occurrences"
        if state.get("metric") == "runs_total":
            label = "runs"
            
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
            
        lines.append(f"{i}. {prefix}{row['player']} — {row['count']} {label}")
        
    lines.append("=====================================\n")
    # Inject the winner into the retrieval filters ONLY if no specific player was requested.
    # If the user asked "How many wickets did Bumrah take?", we don't want to filter by the
    # batter he bowled to the most!
    if not state.get("player_stats"):
        top_player = leaderboard[0]["player"]
        
        group_by = state.get("group_by", "player")
        if group_by == "over":
            parts = str(top_player).split('_')
            if len(parts) == 2:
                try:
                    innings_val = int(parts[0])
                    over_val = int(parts[1])
                    player_filter = {"$and": [{"innings": {"$eq": innings_val}}, {"over": {"$eq": over_val}}]}
                except ValueError:
                    player_filter = {"over": {"$eq": top_player}}
            else:
                try:
                    player_filter = {"over": {"$eq": int(float(top_player))}}
                except ValueError:
                    player_filter = {"over": {"$eq": top_player}}
        elif group_by == "innings":
            try:
                player_filter = {"innings": {"$eq": int(float(top_player))}}
            except ValueError:
                player_filter = {"innings": {"$eq": top_player}}
        elif group_by == "wicket_kind":
            player_filter = {"wicket_kind": {"$eq": top_player}}
        else:
            filter_key = "batter"
            if event_type in ("wicket", "dot"):
                filter_key = "bowler"
            player_filter = {filter_key: {"$eq": top_player}}
        
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
    """
    query_text = state["rewritten_question"] or state["question"]
    where      = state.get("retrieval_filters")
    results    = vector_query(query_text, n_results=INITIAL_TOP_K, where=where)
    return {**state, "retrieved_docs": results, "initial_docs": results}


# ---------------------------------------------------------------------------
# Node 3 — Rerank Docs
# ---------------------------------------------------------------------------

def rerank_docs(state: RAGState) -> RAGState:
    """Rerank retrieved documents using FlashRank."""
    docs = state["retrieved_docs"]
    query_text = state["rewritten_question"] or state["question"]
    reranked = rerank_documents(query_text, docs, top_n=TOP_K)
    return {**state, "retrieved_docs": reranked}


# ---------------------------------------------------------------------------
# Node 4 — Build context
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
    
    if agg_stats:
        lines.append(agg_stats)
    elif p_stats:
        lines.append("=== SYSTEM CALCULATED EXACT STATS ===")
        lines.append(f"Player: {p_stats['name']}")
        b = p_stats['batting']
        lines.append(f"Batting: {b['runs']} runs off {b['balls']} balls ({b['fours']} fours, {b['sixes']} sixes)")
        if b['dismissal']:
            lines.append(f"Dismissal: {b['dismissal']}")
        w = p_stats['bowling']
        if w['overs'] != "0.0":
            lines.append(f"Bowling: {w['wickets']} wickets for {w['runs']} runs in {w['overs']} overs")
        lines.append("=====================================\n")

    lines.append("=== Relevant Match Highlight Deliveries ===")

    # Inject Match-Level Metadata
    m = docs[0]["metadata"]
    lines.append(f"Match: {m.get('match', 'Unknown')}")
    lines.append(f"Venue: {m.get('venue', 'Unknown')}")
    lines.append(f"Season: {m.get('season', 'Unknown')}")
    lines.append("")

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
            f"[{i}] Innings {inn} | Over {over}.{ball} | "
            f"{batter} vs {bowler} | {event.upper()} | Runs: {runs}"
        )
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
# Node 3 — Generate answer (with streaming)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a cricket match reporter. You will be provided with a block of "exact match stats" calculated by the system, followed by specific highlight deliveries.

Your task is to answer the user's question using ONLY this provided information.

Guidelines:
1. When asked about aggregate numbers (match totals, balls faced, boundaries hit), ALWAYS use the numbers provided in the "=== SYSTEM CALCULATED EXACT STATS ===" block. Do not recalculate them.
2. When asked about specific events (how someone got out, how they hit a boundary), read the "=== Relevant Match Highlight Deliveries ===" section.
3. You MUST use the "Commentary" text from the highlights to provide narrative descriptions. Even for simple questions like "Who dismissed X?", you MUST describe HOW the dismissal happened or HOW the shot was played using the commentary. Paraphrase the commentary to tell an engaging story, but DO NOT invent any adjectives or actions that aren't in the raw text.
4. Always cite the exact over and ball number as it appears in the context (e.g., "In Over 12.3" or "In Over 19.1"). DO NOT alter or increment the over numbers, even if it is the last over of the match.
5. If the provided information does not contain the answer, explicitly state that you don't have enough data."""


def _build_messages(state: RAGState) -> list[dict]:
    """Build the full message list for the LLM, including history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject prior conversation turns
    if state["chat_history"]:
        messages.extend(state["chat_history"])

    # Append the current context + question as the new user turn
    user_content = f"{state['context']}\n\nQuestion: {state['question']}"
    messages.append({"role": "user", "content": user_content})
    return messages


def generate_answer(state: RAGState) -> RAGState:
    """
    Non-streaming answer generation (used by the LangGraph .invoke() path
    and the CLI chat.py).
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=_build_messages(state),
    )
    answer = response["message"]["content"].strip()
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
    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=_build_messages(state),
        stream=True,
    ):
        token = chunk["message"]["content"]
        full_answer.append(token)
        yield token

    return "".join(full_answer)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assemble and compile the LangGraph RAG workflow.

    Pipeline:  rewrite_question → retrieve → rerank_docs → build_context → generate_answer
    """
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("extract_filters",         extract_filters)
    graph.add_node("compute_aggregate_stats", compute_aggregate_stats)
    graph.add_node("retrieve",                retrieve)
    graph.add_node("rerank_docs",      rerank_docs)
    graph.add_node("build_context",    build_context)
    graph.add_node("generate_answer",  generate_answer)

    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question",        "extract_filters")
    graph.add_edge("extract_filters",         "compute_aggregate_stats")
    graph.add_edge("compute_aggregate_stats", "retrieve")
    graph.add_edge("retrieve",                "rerank_docs")
    graph.add_edge("rerank_docs",      "build_context")
    graph.add_edge("build_context",    "generate_answer")
    graph.add_edge("generate_answer",  END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------

_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _initial_state(question: str, chat_history: list[dict]) -> RAGState:
    return {
        "question":           question,
        "rewritten_question": "",
        "chat_history":       chat_history or [],
        "retrieval_filters":  None,
        "player_stats":       None,
        "aggregate_stats":    None,
        "initial_docs":       [],
        "retrieved_docs":     [],
        "context":            "",
        "answer":             "",
        "group_by":           "player",
        "metric":             "count",
    }


def ask(question: str, chat_history: list[dict] = None) -> str:
    """
    Run the full RAG pipeline for a single question and return the answer.
    Optionally accepts prior chat_history for follow-up support.
    """
    state = _initial_state(question, chat_history or [])

    # Run pipeline manually so we can stream generation
    state = rewrite_question(state)
    state = extract_filters(state)
    state = compute_aggregate_stats(state)
    state = retrieve(state)
    state = rerank_docs(state)
    state = build_context(state)
    state = generate_answer(state)
    return state["answer"]


def ask_stream(
    question: str,
    chat_history: list[dict] = None,
) -> Generator[str | dict, None, None]:
    """
    Streaming variant of ask().

    Yields:
      1. FIRST: a dict with pipeline metadata (for the inspector)
         {"rewritten_question": ..., "num_docs": ..., "top_docs": [...]}
      2. THEN: token strings from the LLM generation

    Callers should check `isinstance(item, dict)` for the metadata event.
    """
    state = _initial_state(question, chat_history or [])
    state = rewrite_question(state)
    state = extract_filters(state)
    state = compute_aggregate_stats(state)
    state = retrieve(state)
    state = rerank_docs(state)
    state = build_context(state)

    # Build inspector metadata from the state AFTER retrieval
    docs = state["retrieved_docs"]
    initial_docs = state["initial_docs"]
    
    def _format_docs(doc_list):
        formatted = []
        for doc in doc_list:   # all retrieved docs for the inspector
            m = doc["metadata"]
            formatted.append({
                "innings": m.get("innings", "?"),
                "over":    f"{m.get('over', '?')}.{m.get('ball', '?')}",
                "batter":  m.get("batter", "?"),
                "bowler":  m.get("bowler", "?"),
                "event":   m.get("event", "?"),
                "runs":    m.get("runs_total", "?"),
                "distance": float(round(doc.get("distance", 0), 4)),
                "score": float(round(doc.get("score", 0), 4)) if "score" in doc else None,
            })
        return formatted

    top_docs = _format_docs(docs)
    initial_top_docs = _format_docs(initial_docs)

    meta = {
        "rewritten_question": state["rewritten_question"],
        "was_rewritten":      state["rewritten_question"] != question,
        "retrieval_filters":  state.get("retrieval_filters"),
        "aggregate_stats":    state.get("aggregate_stats"),
        "group_by":           state.get("group_by", "player"),
        "metric":             state.get("metric", "count"),
        "num_docs":           len(docs),
        "initial_num_docs":   len(initial_docs),
        "top_docs":           top_docs,
        "initial_top_docs":   initial_top_docs,
        "history_turns":      len(chat_history or []) // 2,
    }

    yield meta   # <-- first yield is always the metadata dict
    yield from generate_answer_stream(state)  # then token strings


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rag.vector_store import collection_exists

    if not collection_exists():
        print("ChromaDB index not found. Run chat.py to auto-build the index first.")
        raise SystemExit(1)

    test_questions = [
        ("Who dismissed Shimron Hetmyer?", []),
        ("What over was that?",            [
            {"role": "user",      "content": "Who dismissed Shimron Hetmyer?"},
            {"role": "assistant", "content": "Hetmyer was dismissed by JJ Bumrah in Over 12.3 (caught)."},
        ]),
    ]
    for q, history in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"A: {ask(q, chat_history=history)}")
