"""
rag_graph.py
------------
Steps 7 & 8 of the RAG pipeline.

Implements a LangGraph StateGraph with four nodes:
  rewrite_question → retrieve → build_context → generate_answer

All LLM inference runs locally through Ollama (mistral or llama3).
Supports session memory (chat history) and streaming token output.
"""

from typing import TypedDict, Generator
from langgraph.graph import StateGraph, END
import ollama
from rag.vector_store import query as vector_query
from config import LLM_MODEL, TOP_K

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
    retrieved_docs:     list[dict]
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
# Node 1 — Retrieve
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    """
    Query the ChromaDB vector store using the (possibly rewritten) question.
    """
    query_text = state["rewritten_question"] or state["question"]
    results = vector_query(query_text, n_results=TOP_K)
    return {**state, "retrieved_docs": results}


# ---------------------------------------------------------------------------
# Node 2 — Build context
# ---------------------------------------------------------------------------

def build_context(state: RAGState) -> RAGState:
    """
    Format the retrieved delivery documents into a context block.
    """
    docs = state["retrieved_docs"]
    if not docs:
        return {**state, "context": "No relevant match data found."}

    lines = ["=== Relevant Match Deliveries ===\n"]

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

SYSTEM_PROMPT = """You are an expert cricket analyst answering questions about a specific cricket match. You have access ONLY to the ball-by-ball delivery data and commentary provided below.

RULES:
1. NEVER invent, guess, or hallucinate player names, scores, or events that are NOT present in the provided context.
2. Always cite the over and ball number when referring to a specific delivery (e.g., "Over 12.3").
3. For wickets — mention: who bowled, who got out, the type of dismissal, and any fielder involved.
4. For boundaries — mention: the batter, bowler, and the shot played if commentary describes it.
5. Use the Commentary field to enrich your answer with narrative detail (e.g., shot description, crowd reaction, match context). Quote or paraphrase it naturally — do NOT just dump raw text.
6. Write in a vivid, engaging cricket-commentary style. Be specific and descriptive, not just a data dump.
7. If the context is insufficient, say what you DO know and note what is missing — but still use all commentary available.
8. Do NOT use general cricket knowledge to fill gaps not present in the context."""


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

    Pipeline:  rewrite_question → retrieve → build_context → generate_answer
    """
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("retrieve",         retrieve)
    graph.add_node("build_context",    build_context)
    graph.add_node("generate_answer",  generate_answer)

    graph.set_entry_point("rewrite_question")
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("retrieve",         "build_context")
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
        "retrieved_docs":     [],
        "context":            "",
        "answer":             "",
    }


def ask(question: str, chat_history: list[dict] = None) -> str:
    """
    Run the full RAG pipeline for a single question and return the answer.
    Optionally accepts prior chat_history for follow-up support.
    """
    state = _initial_state(question, chat_history or [])

    # Run rewrite + retrieve + build_context manually so we can stream generation
    state = rewrite_question(state)
    state = retrieve(state)
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
    state = retrieve(state)
    state = build_context(state)

    # Build inspector metadata from the state AFTER retrieval
    docs = state["retrieved_docs"]
    top_docs = []
    for doc in docs:   # all retrieved docs for the inspector
        m = doc["metadata"]
        top_docs.append({
            "innings": m.get("innings", "?"),
            "over":    f"{m.get('over', '?')}.{m.get('ball', '?')}",
            "batter":  m.get("batter", "?"),
            "bowler":  m.get("bowler", "?"),
            "event":   m.get("event", "?"),
            "runs":    m.get("runs_total", "?"),
            "distance": round(doc.get("distance", 0), 4),
        })

    meta = {
        "rewritten_question": state["rewritten_question"],
        "was_rewritten":      state["rewritten_question"] != question,
        "num_docs":           len(docs),
        "top_docs":           top_docs,
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
