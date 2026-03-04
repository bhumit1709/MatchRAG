"""
chat.py
-------
Steps 9 & 10 of the RAG pipeline.

Entry-point CLI chatbot. On first run it automatically builds the
ChromaDB index. Subsequent runs reuse the existing index for fast startup.

Usage:
  python chat.py                   # Normal mode (reuses existing index)
  python chat.py --rebuild         # Force re-index before starting
  python chat.py --file other.json # Use a different match file
"""

import sys
import argparse
import time

from config import EMBED_MODEL, LLM_MODEL, DATA_FILE
from rag.load_match import load_match
from rag.flatten_data import flatten_deliveries
from rag.embedding_pipeline import generate_embeddings, check_model_available
from rag.vector_store import build_index, collection_exists
from rag.rag_graph import ask


# ---------------------------------------------------------------------------
# ANSI colour helpers (degrades gracefully if terminal doesn't support it)
# ---------------------------------------------------------------------------

BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def _print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗
║   🏏  Cricket Match RAG — Local AI Commentary Bot   ║
╚══════════════════════════════════════════════════════╝{RESET}
{DIM}  Embeddings : {EMBED_MODEL}
  LLM        : {LLM_MODEL}
  Vector DB  : ChromaDB (local){RESET}

Type your question and press Enter. Type {YELLOW}quit{RESET} or {YELLOW}exit{RESET} to leave.
""")


# ---------------------------------------------------------------------------
# Ingest pipeline helper
# ---------------------------------------------------------------------------

def run_ingest(filepath: str, force_rebuild: bool = False) -> None:
    """
    Load, flatten, embed, and index the match JSON into ChromaDB.

    Args:
        filepath: Path to the match JSON file.
        force_rebuild: If True, delete existing index and rebuild from scratch.
    """
    if collection_exists() and not force_rebuild:
        print(f"{GREEN}✓ Existing index found — skipping ingest.{RESET}")
        return

    print(f"\n{BOLD}Building index from '{filepath}'...{RESET}")
    t0 = time.time()

    # Pre-flight: verify Ollama models are available
    if not check_model_available(EMBED_MODEL):
        print(f"\n{YELLOW}⚠ Ollama model '{EMBED_MODEL}' not found.{RESET}")
        print(f"  Run: ollama pull {EMBED_MODEL}")
        sys.exit(1)

    # Step 1-4: Load and flatten
    print("  Loading match JSON...")
    data = load_match(filepath)
    docs = flatten_deliveries(data)
    print(f"  Flattened {len(docs)} deliveries.")

    # Step 5: Embed
    embeddings = generate_embeddings(docs, verbose=True)

    # Step 6: Store
    build_index(docs, embeddings, reset=force_rebuild)

    elapsed = time.time() - t0
    print(f"{GREEN}✓ Index ready in {elapsed:.1f}s{RESET}\n")


# ---------------------------------------------------------------------------
# Example questions shown on startup
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Who dismissed Shimron Hetmyer?",
    "What happened in the last over?",
    "Who hit the most sixes?",
    "Show all wickets taken by Bumrah.",
    "How did India win the match?",
]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Local RAG chatbot for cricket match commentary."
    )
    parser.add_argument(
        "--file", default=DATA_FILE,
        help=f"Path to the match JSON file (default: {DATA_FILE})"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force re-building the ChromaDB index even if one exists."
    )
    args = parser.parse_args()

    # Auto-ingest on startup
    run_ingest(args.file, force_rebuild=args.rebuild)

    _print_banner()

    print(f"{DIM}Example questions:{RESET}")
    for q in EXAMPLE_QUESTIONS:
        print(f"  {DIM}→ {q}{RESET}")
    print()

    # ---- REPL loop ----
    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye! 🏏{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q", ":q"}:
            print(f"{DIM}Goodbye! 🏏{RESET}")
            break

        print(f"\n{DIM}Thinking...{RESET}")
        t0 = time.time()

        try:
            answer = ask(user_input)
        except Exception as e:
            print(f"{YELLOW}⚠ Error: {e}{RESET}")
            continue

        elapsed = time.time() - t0
        print(f"\n{BOLD}{GREEN}Bot:{RESET} {answer}")
        print(f"{DIM}  [{elapsed:.1f}s]{RESET}\n")


if __name__ == "__main__":
    main()
