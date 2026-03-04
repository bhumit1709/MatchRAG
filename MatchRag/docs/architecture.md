# MatchRAG — RAG Pipeline Architecture

## Overview

MatchRAG is a local Retrieval-Augmented Generation (RAG) chatbot for cricket match analysis. It runs entirely on your machine using **Ollama** for embeddings and LLM inference, **ChromaDB** as the vector store, and **LangGraph** to orchestrate the pipeline.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MatchRAG Pipeline                            │
│                                                                      │
│  data/IndVsWI.json                                                   │
│         │                                                            │
│         ▼                                                            │
│  [1] load_match.py ──── load & validate CricSheet JSON              │
│         │                                                            │
│         ▼                                                            │
│  [2-4] flatten_data.py ─ flatten innings → delivers → flat dicts    │
│         │                detect events (wicket/six/four/dot/…)      │
│         │                build rich natural-language text field      │
│         │                                                            │
│         ▼                                                            │
│  [5] embedding_pipeline.py ── embed text via Ollama (nomic-embed)   │
│         │                                                            │
│         ▼                                                            │
│  [6] vector_store.py ──── upsert into ChromaDB (cosine similarity)  │
│                                                                      │
│         ┌──── Query time ────────────────────────────────────────┐  │
│         │                                                         │  │
│  User question                                                   │  │
│         │                                                         │  │
│         ▼                                                         │  │
│  [7] vector_store.query() ── embed question → top-K retrieval    │  │
│         │                                                         │  │
│         ▼                                                         │  │
│  [8] rag_graph.py ─── LangGraph StateGraph:                      │  │
│         │               retrieve → build_context → generate      │  │
│         │                                                         │  │
│         ▼                                                         │  │
│       Answer (via Ollama mistral)                                 │  │
│         └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Module Reference

| Module | Pipeline Step | Responsibility |
|---|---|---|
| `rag/load_match.py` | 1 | Load & validate CricSheet JSON |
| `rag/flatten_data.py` | 2–4 | Flatten, detect events, build embedding text |
| `rag/embedding_pipeline.py` | 5 | Batch embed docs via Ollama |
| `rag/vector_store.py` | 6–7 | Build ChromaDB index, run semantic search |
| `rag/rag_graph.py` | 8 | LangGraph graph: retrieve → context → answer |
| `config.py` | — | Centralised configuration (models, paths, ports) |
| `server.py` | — | Flask REST API (`POST /api/ask`) |
| `chat.py` | — | Interactive CLI REPL |
| `scripts/scrape_commentary.py` | — | Data collection utility |

## Configuration

All settings are in `config.py` and can be overridden via environment variables. See `.env.example` for the full list.

## Quick Start

```bash
# 1. Set up environment
make install

# 2. Pull Ollama models
ollama pull nomic-embed-text
ollama pull mistral

# 3. Start the API server (builds index on first run)
make server

# 4. Or use the CLI chatbot
make chat
```

## Testing

```bash
make test   # runs pytest tests/
```
