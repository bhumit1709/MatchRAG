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

import argparse
import time

from config import DATA_FILE
from rag.ingest import run_ingest
from rag.providers import runtime_summary
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
║   🏏  Cricket Match RAG — LangChain Learning Bot    ║
╚══════════════════════════════════════════════════════╝{RESET}
{DIM}  Runtime    : {runtime_summary()['llm_runtime']}
  LLM        : {runtime_summary()['llm_model']}
  Embedding  : {runtime_summary()['embed_model']}
  Vector DB  : ChromaDB (local){RESET}

Type your question and press Enter. Type {YELLOW}quit{RESET} or {YELLOW}exit{RESET} to leave.
""")


# ---------------------------------------------------------------------------
# Example questions shown on startup
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Who dismissed Abhishek Sharma?",
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
    ingest_result = run_ingest(args.file, force_rebuild=args.rebuild, verbose=False)
    if ingest_result["skipped"]:
        print(f"{GREEN}✓ Existing index found — skipping ingest.{RESET}")
    else:
        print(
            f"{GREEN}✓ Index ready with {ingest_result['records']} deliveries "
            f"in {ingest_result['elapsed']:.1f}s{RESET}\n"
        )

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
