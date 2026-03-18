"""In-memory session store with embedding-based history pruning."""

import threading
from collections import defaultdict

from config import HISTORY_RELEVANCE_THRESHOLD, MAX_HISTORY_TURNS
from rag.providers import get_embeddings


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# session_id → list of {"role": "user"|"assistant", "content": str}
_sessions: dict[str, list[dict]] = defaultdict(list)

# session_id → list of raw question strings (for similarity comparison)
_questions: dict[str, list[str]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_history(session_id: str) -> list[dict]:
    """Return the chat history for a session (already pruned at add time)."""
    with _lock:
        return list(_sessions[session_id])


def add_turn(session_id: str, question: str, answer: str) -> None:
    """
    Append a completed Q&A turn to the session.
    Runs smart pruning to keep only relevant prior turns.
    """
    with _lock:
        _questions[session_id].append(question)
        _sessions[session_id].append({"role": "user",      "content": question})
        _sessions[session_id].append({"role": "assistant", "content": answer})

        # Prune after adding so the store stays lean
        _sessions[session_id] = _prune(
            _sessions[session_id],
            _questions[session_id],
        )


def clear_session(session_id: str) -> None:
    """Remove all history for a session."""
    with _lock:
        _sessions.pop(session_id, None)
        _questions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Smart pruning
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _prune(history: list[dict], questions: list[str]) -> list[dict]:
    """
    Smart topic-aware pruning:
    - Always keep the last MAX_HISTORY_TURNS turns (safety net)
    - Among older turns, keep only those whose question is semantically
      similar (cosine >= HISTORY_RELEVANCE_THRESHOLD) to the latest question

    history  : flat list of {role, content} messages (user+assistant pairs)
    questions: parallel list of raw question strings per turn
    """
    if len(questions) <= 1:
        return history

    # Each turn = 2 messages (user + assistant)
    total_turns = len(questions)
    keep_last   = min(MAX_HISTORY_TURNS, total_turns)

    # Turns we always keep (most recent N)
    always_keep_start = total_turns - keep_last

    current_q   = questions[-1]
    older_questions = questions[:always_keep_start]

    if not older_questions:
        # All turns fall within the always-keep window
        return history

    try:
        all_texts = [current_q] + older_questions
        embeddings = get_embeddings().embed_documents(all_texts)
        current_emb  = embeddings[0]
        older_embs   = embeddings[1:]
    except Exception:
        # If embedding fails, fall back to simple FIFO cap
        return history[-(keep_last * 2):]

    # Decide which older turns to keep
    keep_older_indices = []
    for i, (q, emb) in enumerate(zip(older_questions, older_embs)):
        sim = _cosine_similarity(current_emb, emb)
        if sim >= HISTORY_RELEVANCE_THRESHOLD:
            keep_older_indices.append(i)

    # Reconstruct message list from kept turn indices
    pruned: list[dict] = []
    for turn_idx in keep_older_indices:
        msg_start = turn_idx * 2
        pruned.extend(history[msg_start: msg_start + 2])

    # Append the always-keep recent turns
    pruned.extend(history[always_keep_start * 2:])

    return pruned
