# MatchRAG — Agent Context

> Quick-reference for AI agents and contributors working on this codebase.

---

## Project Identity

**MatchRAG** is a fully-local RAG chatbot for cricket match commentary. It uses LangChain, LangGraph, llama.cpp, ChromaDB, and FlashRank — everything runs on-device with no external API calls at query time.

**Primary match data:** T20 World Cup Final — India vs New Zealand (`data/IndVsNZ.json`)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| LLM | llama.cpp via `ChatLlamaCpp` (local GGUF model) |
| Embeddings | `BAAI/bge-small-en-v1.5` via `HuggingFaceEmbeddings` |
| Vector Store | ChromaDB (persistent, cosine-space HNSW) |
| Reranker | FlashRank (`ms-marco-TinyBERT-L-2-v2`) |
| Orchestration | LangGraph `StateGraph` |
| Chains | LangChain `ChatPromptTemplate`, `PydanticOutputParser` |
| API | Flask + flask-cors (SSE streaming) |
| Frontend | React 18 + Vite |

---

## Repository Structure

```text
MatchRag/
├── server.py               # Flask API (entry point for web UI)
├── chat.py                 # CLI REPL (entry point for terminal)
├── config.py               # All configuration (env vars + .env)
├── rag/                    # Core RAG pipeline package
│   ├── rag_graph.py        # LangGraph assembly + ask()/ask_stream()
│   ├── graph_nodes.py      # 6 pipeline node implementations
│   ├── chains.py           # LangChain chain functions
│   ├── prompts.py          # All prompt templates
│   ├── schemas.py          # Pydantic models (RetrievalPlan, LLMTrace)
│   ├── state.py            # RAGState TypedDict
│   ├── providers.py        # Model singletons (LLM + embeddings)
│   ├── llm_services.py     # Low-level LLM call wrappers
│   ├── retrievers.py       # Multi-query + rerank retrieval
│   ├── reranker.py         # FlashRank reranking
│   ├── vector_store.py     # ChromaDB + deterministic stat helpers
│   ├── documents.py        # Record ↔ Document conversion
│   ├── flatten_data.py     # CricSheet JSON → flat records
│   ├── load_match.py       # Match JSON loader
│   ├── embedding_pipeline.py # Batch embedding helpers
│   ├── ingest.py           # Startup data ingestion
│   └── session_store.py    # In-memory session store + smart pruning
├── web/src/                # React frontend
│   ├── App.jsx             # Chat UI with SSE streaming
│   └── components/         # ChatMessage, ExampleChips, ThinkingIndicator, PipelineInspector
├── scripts/                # Data acquisition
│   ├── scrape_commentary.py    # ESPN Cricinfo scraper
│   └── append_commentary.py    # Commentary merger
├── tests/                  # pytest test suite
├── data/                   # Match JSON files
├── models/                 # Local GGUF model files
└── chroma_db/              # Persistent vector store
```

---

## Pipeline Flow

```text
question → rewrite_question → plan_retrieval → compute_aggregate_stats → retrieve → build_context → generate_answer
```

### Answer Strategies

| Strategy | When Used | Retrieval Behavior |
|----------|-----------|-------------------|
| `semantic` | Narrative/descriptive questions | Multi-query expansion + FlashRank reranking |
| `aggregate` | "How many X?" — pure stat questions | No retrieval; deterministic stats only |
| `sequential` | "What happened in over 15?" | Exact metadata fetch, chronological order |
| `hybrid` | "Who hit the most sixes?" | Deterministic leaderboard + supporting docs for leader |

---

## Key Conventions

### Code Style
- **Line length:** 100 (ruff)
- **Target Python:** 3.10+
- **Formatting:** Follow existing patterns (no black/ruff-format enforced yet)
- **Imports:** Group stdlib → third-party → local; relative imports within `rag/`

### Architecture Rules
1. **Deterministic stats are never LLM-generated.** Player stats, leaderboards, and aggregates are computed from ChromaDB metadata in plain Python.
2. **LLM is used only for:** question rewriting, retrieval planning, multi-query expansion, and final answer generation.
3. **All LLM calls return traces.** Every chain function returns `(result, trace_dict)` to support the Pipeline Inspector.
4. **Fast-path first.** `plan_retrieval` tries deterministic pattern matching before falling back to LLM-based planning.
5. **State immutability pattern.** Graph nodes return `{**state, ...updates}` rather than mutating state in-place.

### Prompt Guidelines
- Prompts live in `rag/prompts.py` as module-level constants.
- System messages define strict behavioral rules for the LLM.
- The answer prompt enforces: over.ball citations, minimum 2 sentences, no invention, one analysis sentence.

### Data Contract
- Match data must be CricSheet JSON format with `info` and `innings` keys.
- Each delivery gets a unique ID: `inn{N}_ov{N}_b{N}`.
- Events are classified as: `wicket | six | four | dot | single | run`.
- Metadata fields stored in ChromaDB: `id, match, venue, season, event_name, innings, batting_team, over, ball, batter, bowler, non_striker, event, player_out, wicket_kind, wicket_fielder, runs_batter, runs_extras, runs_total, commentary`.

---

## Development Commands

```bash
make install    # Create .venv and install dependencies
make server     # Start Flask API on port 5001
make chat       # Start CLI chatbot
make test       # Run pytest suite
make lint       # Run ruff linting
make rebuild    # Force ChromaDB re-index
make clean      # Remove caches and build artifacts

cd web && npm install    # Install frontend dependencies
cd web && npm run dev    # Start Vite dev server (port 5173)
```

---

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_MODEL_PATH` | `models/llama-chat.gguf` | Path to local GGUF model |
| `LLM_N_GPU_LAYERS` | `-1` on Apple Silicon | Set to 0 for CPU-only |
| `EMBED_MODEL_PATH` | Auto-detect `models/bge-small-en-v1.5` | Local embedding model directory |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HuggingFace model ID (fallback) |
| `DATA_FILE` | `data/IndVsNZ.json` | Default match file |
| `CHROMA_PATH` | `chroma_db` | ChromaDB persistence directory |
| `TOP_K` | `8` | Final docs after reranking |
| `RETRIEVER_TOP_K` | `20` | Initial retrieval candidate count |
| `MULTI_QUERY_COUNT` | `3` | Number of query expansion variants |
| `MAX_HISTORY_TURNS` | `5` | Session history window |
| `API_PORT` | `5001` | Flask server port |

---

## Known Limitations

See `BACKLOG.md` for the full list. Key gaps:

- **Single-match scope** — No cross-match querying
- **No phase analytics** — Powerplay/death overs not modeled
- **No partnership tracking** — Partnership spans not derived
- **No batter-vs-bowler matchups** — Pair-wise aggregation unsupported
- **No score progression** — Running totals not computed
- **No visual analytics** — No chart/graph generation
- **In-memory sessions** — Lost on server restart
- **No auth/rate limiting** — Dev-only Flask server
- **Unpinned dependencies** — No lockfile

---

## Testing

Tests live in `tests/` and are split into:
- `test_pipeline.py` — Ingestion and data shape
- `test_graph.py` — Graph behavior
- `test_chains.py` — Prompt chain shape
- `test_retrievers.py` — Retrieval behavior

Run: `python3 -m pytest tests -q`

---

## Reference Documentation

- [Architecture deep-dive](docs/architecture.md) — Full system diagram, node-by-node explanations, and module reference
- [Backlog](BACKLOG.md) — Production readiness roadmap with priority matrix
- [README](README.md) — Setup and quickstart guide
