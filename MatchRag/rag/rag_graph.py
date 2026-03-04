"""
rag_graph.py
------------
Steps 7 & 8 of the RAG pipeline.

Implements a LangGraph StateGraph with three nodes:
  retrieve → build_context → generate_answer

All LLM inference runs locally through Ollama (mistral or llama3).
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
import ollama
from rag.vector_store import query as vector_query
from config import LLM_MODEL, TOP_K


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    """Shared state that flows through all nodes of the graph."""
    question:    str
    retrieved_docs: list[dict]
    context:     str
    answer:      str


# ---------------------------------------------------------------------------
# Node 1 — Retrieve
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    """
    Query the ChromaDB vector store for deliveries relevant to the question.
    Stores the top-K results in state["retrieved_docs"].
    """
    results = vector_query(state["question"], n_results=TOP_K)
    return {**state, "retrieved_docs": results}


# ---------------------------------------------------------------------------
# Node 2 — Build context
# ---------------------------------------------------------------------------

def build_context(state: RAGState) -> RAGState:
    """
    Format the retrieved delivery documents into a context block
    that will be injected into the LLM prompt.
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
        m = doc["metadata"]
        inn  = m.get("innings", "?")
        over = m.get("over", "?")
        ball = m.get("ball", "?")
        batter  = m.get("batter", "?")
        bowler  = m.get("bowler", "?")
        event   = m.get("event", "?")
        runs    = m.get("runs_total", "?")

        # Build a compact header for each delivery
        header = (
            f"[{i}] Innings {inn} | Over {over}.{ball} | "
            f"{batter} vs {bowler} | {event.upper()} | Runs: {runs}"
        )

        # Optionally add wicket info
        if m.get("player_out"):
            header += f" | OUT: {m['player_out']} ({m.get('wicket_kind', '')})"

        lines.append(header)
        lines.append(f"    Commentary: {doc['text'].split('Commentary:')[-1].strip()}")
        lines.append("")

    return {**state, "context": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Node 3 — Generate answer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert cricket analyst answering questions about a specific cricket match. You have access ONLY to the ball-by-ball delivery data provided below.

STRICT RULES — you MUST follow these:
1. NEVER invent, guess, or hallucinate player names, scores, or events that are NOT explicitly present in the provided context.
2. If the context does not contain enough information to fully answer the question, clearly state: "Based on the available data, I can only confirm..." and share what IS in the context.
3. Always cite the over and ball number when referring to a specific delivery (e.g., "Over 12.3").
4. For wickets, mention who bowled, who got out, and the type of dismissal — only if present in the data.
5. For boundaries (four/six), mention the batter and bowler — only if present in the data.
6. Do NOT use your general cricket knowledge to fill gaps. Stick strictly to the retrieved deliveries.
7. If you are unsure, say so. Do NOT make up facts."""


def generate_answer(state: RAGState) -> RAGState:
    """
    Send the context and question to Ollama and return the LLM's answer.
    Uses the chat completions API with a system prompt for consistent behaviour.
    """
    prompt = f"{state['context']}\n\nQuestion: {state['question']}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    answer = response["message"]["content"].strip()
    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assemble and compile the LangGraph RAG workflow.

    Pipeline:  retrieve → build_context → generate_answer → END
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",        retrieve)
    graph.add_node("build_context",   build_context)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",        "build_context")
    graph.add_edge("build_context",   "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

_compiled_graph = None


def ask(question: str) -> str:
    """
    Run the full RAG pipeline for a single question and return the answer.

    Args:
        question: Natural language question about the match.

    Returns:
        The LLM-generated answer string.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: RAGState = {
        "question":      question,
        "retrieved_docs": [],
        "context":       "",
        "answer":        "",
    }

    final_state = _compiled_graph.invoke(initial_state)
    return final_state["answer"]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rag.vector_store import collection_exists

    if not collection_exists():
        print("ChromaDB index not found. Run chat.py to auto-build the index first.")
        raise SystemExit(1)

    test_questions = [
        "Who dismissed Shimron Hetmyer?",
        "What happened in the final over?",
        "How many sixes were hit in the match?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"A: {ask(q)}")
