# MatchRAG — Agent Configuration

## Architecture Reference

> **Before making any changes to this project, read [`docs/architecture.md`](../docs/architecture.md) for the full application architecture.**
> It contains the complete directory structure, module reference, data flow, dependencies, and known limitations.

## Project Summary

MatchRAG is a local RAG chatbot for cricket match analysis. Key facts:

- **Language:** Python 3.10+ (backend), React 19 + Vite 7 (frontend)
- **LLM/Embeddings:** Ollama (mistral + nomic-embed-text) — fully local
- **Vector DB:** ChromaDB (persistent, cosine similarity)
- **Pipeline:** LangGraph StateGraph → `retrieve → build_context → generate_answer`
- **API:** Flask REST server on port 5001 (`server.py`)
- **CLI:** Interactive REPL chatbot (`chat.py`)
- **Frontend:** React chat UI on port 5173 (`web/`)
- **Data:** CricSheet JSON format, flattened to delivery-level records

## Key File Locations

| Area | Path | Purpose |
|------|------|---------|
| Config | `config.py` | All tuneable constants (models, paths, ports) |
| RAG Core | `rag/` | Pipeline package (load → flatten → embed → store → query → answer) |
| API Server | `server.py` | Flask REST API |
| CLI | `chat.py` | Terminal chatbot |
| Frontend | `web/src/` | React components + CSS design system |
| Scripts | `scripts/` | Commentary scraping from ESPN Cricinfo |
| Data | `data/` | Match JSON files |
| Tests | `tests/` | pytest unit tests |
| Docs | `docs/architecture.md` | **Full architecture reference** |
| Backlog | `BACKLOG.md` | Production readiness items |

## Dev Commands

```bash
make install   # Setup .venv
make server    # Flask API (port 5001)
make chat      # CLI chatbot
make test      # pytest
make lint      # ruff
cd web && npm run dev  # React UI (port 5173)
```

## Conventions

- All Python config is centralized in `config.py`, overridable via env vars.
- Delivery IDs follow format: `inn{X}_ov{Y}_b{Z}`.
- The `rag/` package exposes a public API via `__init__.py`: `ask()`, `load_match()`, `flatten_deliveries()`, `build_index()`, `collection_exists()`.
- ChromaDB collection name: `cricket_commentary`.
- System prompt in `rag_graph.py` enforces strict factual answers — no hallucination.
