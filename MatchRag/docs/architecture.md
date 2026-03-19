# MatchRAG Architecture

> Fully local RAG chatbot for cricket match commentary — LangChain · LangGraph · llama.cpp · ChromaDB

---

## Overview

MatchRAG is a **Retrieval-Augmented Generation** system that answers natural-language questions about cricket matches using ball-by-ball commentary data from CricSheet JSON files. Every component runs locally:

| Layer | Technology | Role |
|-------|-----------|------|
| LLM | `llama.cpp` via `ChatLlamaCpp` | Generation (GGUF model, Metal on Apple Silicon) |
| Embeddings | `BAAI/bge-small-en-v1.5` via `HuggingFaceEmbeddings` | Semantic search vectors |
| Vector DB | `ChromaDB` (persistent, cosine HNSW) | Document storage and retrieval |
| Reranker | `FlashRank` (`ms-marco-TinyBERT-L-2-v2`) | Context compression / precision boost |
| Orchestration | `LangGraph` (`StateGraph`) | Pipeline workflow DAG |
| Chains | `LangChain` (`ChatPromptTemplate`, `PydanticOutputParser`) | Prompt construction and structured output |
| API Server | `Flask` + `flask-cors` | REST API with SSE streaming |
| Frontend | `React` + `Vite` | Chat UI with Pipeline Inspector |

The system intentionally separates **deterministic cricket/stat logic** (exact aggregations, leaderboards, player stats) from **LLM-driven language tasks** (question rewriting, retrieval planning, answer generation).

---

## System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite)                    │
│   App.jsx → ChatMessage · ExampleChips · ThinkingIndicator      │
│             PipelineInspector (LLM traces, doc tables, timings) │
└────────────────────────────┬────────────────────────────────────┘
                             │ SSE (text/event-stream)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask API Server (server.py)                  │
│   POST /api/ask → ask_stream() → SSE {meta, tokens, done}      │
│   GET  /api/status · POST /api/session/clear                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              LangGraph RAG Pipeline (rag_graph.py)              │
│                                                                 │
│   rewrite_question ──► plan_retrieval ──► compute_aggregate_stats│
│         │                    │                     │             │
│         ▼                    ▼                     ▼             │
│      retrieve ──────► build_context ──────► generate_answer      │
│                                                                 │
│   State: RAGState (TypedDict with 19 fields)                    │
└──────────┬─────────────┬────────────────────────┬───────────────┘
           │             │                        │
    ┌──────▼──────┐ ┌────▼─────┐          ┌──────▼──────┐
    │  ChromaDB   │ │ LLM      │          │ FlashRank   │
    │ vector_store│ │ llama.cpp│          │  reranker   │
    └─────────────┘ └──────────┘          └─────────────┘
```

---

## Data Ingestion Pipeline

```text
CricSheet JSON ──► load_match() ──► flatten_deliveries() ──► records_to_documents() ──► build_index()
                   (load_match.py)   (flatten_data.py)       (documents.py)             (vector_store.py)
```

### Step-by-step

1. **`load_match(filepath)`** — Loads and validates a CricSheet JSON file, checks for `info` and `innings` keys.
2. **`extract_metadata(data)`** — Pulls match-level metadata (teams, venue, season, winner, event name).
3. **`flatten_deliveries(data)`** — Walks the nested `innings → overs → deliveries` structure and produces flat dicts with:
   - Match context: match name, venue, season, event
   - Delivery position: innings, over, ball
   - Players: batter, bowler, non-striker
   - Runs: batter, extras, total
   - Event classification: `wicket | six | four | dot | single | run` (rule-based priority)
   - Wicket details: player out, kind, fielder
   - Raw commentary text (HTML stripped)
   - Rich natural-language `text` field for embedding
   - Unique `id`: `inn{N}_ov{N}_b{N}`
4. **`records_to_documents(records)`** — Converts flat dicts into LangChain `Document` objects with 19 metadata fields.
5. **`build_index(documents)`** — Upserts documents into ChromaDB with deterministic IDs. Writes `index_metadata.json` to track data file hash and embedding model for stale-index detection.

### Index Staleness Detection

The system writes `chroma_db/index_metadata.json` containing the collection name, embed model path, data file path, and SHA-256 hash of the source JSON. On startup, `index_matches_runtime()` compares this against the current config and auto-rebuilds if anything changed.

---

## Query Pipeline (LangGraph Nodes)

The pipeline is a linear `StateGraph` with 6 nodes. Each node receives and returns a `RAGState` TypedDict.

### Node 1: `rewrite_question`

**Purpose:** Resolve pronouns and vague references in follow-up questions.

- **Skip condition:** Question is standalone (no history, or no follow-up signals detected).
- **Follow-up detection:** Heuristic checks for pronoun signals (`he`, `him`, `his`, `that`, `this`), phrase signals (`that over`, `the batter`, `same player`), and short questions with demonstratives.
- **LLM call:** Uses `REWRITE_PROMPT` to rewrite using conversation history (last 6 messages).

### Node 2: `plan_retrieval`

**Purpose:** Produce a structured `RetrievalPlan` that routes the question to the right retrieval strategy.

**Fast-path (no LLM):** Deterministic pattern matching for:
- Ordered events: "first wicket", "last six" → `sequential` strategy with limit=1
- Stat leaderboards: "most sixes", "highest fours" → `hybrid` strategy
- Over-specific queries: "over 15", "last over", "final over" → `sequential` strategy
- Summary questions: "how did X perform" → `semantic` strategy
- Player-specific summaries → `semantic` with player filter

**LLM path:** When fast-path doesn't match, invokes `build_retrieval_plan()`:
- Sends question + known player list + JSON schema instructions
- Parses response with `PydanticOutputParser(RetrievalPlan)`
- Handles malformed JSON with `_extract_json_object()` fallback

**Post-processing (`_normalize_plan`):**
- Fuzzy-resolves player names against known roster (via `difflib.get_close_matches`, last-name fallback)
- Resolves `over: "last"` to actual max over from match metadata
- Assigns `sequential` strategy when a specific over is targeted

**Output:** `RetrievalPlan` model with fields: `normalized_question`, `players[]`, `event`, `over`, `innings`, `answer_strategy`, `is_stat_question`, `group_by`, `metric`, `is_sequential`, `sort_direction`, `limit`.

### Node 3: `compute_aggregate_stats`

**Purpose:** Calculate exact deterministic stats for aggregate/hybrid questions.

- **Skip condition:** Strategy is not `aggregate` or `hybrid`.
- Calls `get_event_leaderboard()` which queries ChromaDB metadata directly (no vector search).
- Groups by `player | over | innings | wicket_kind`.
- Supports metrics: `count`, `runs_total`, `impact` (composite batting+bowling score with strike-rate/economy bonuses).
- Formats a text block: `=== SYSTEM CALCULATED EXACT STATS ===` with numbered leaderboard.

### Node 4: `retrieve`

**Purpose:** Fetch supporting documents based on the retrieval plan.

**Four execution paths:**

| Strategy | Behavior |
|----------|----------|
| `aggregate` (pure) | Skip retrieval entirely — stats block suffices |
| `sequential` | Exact metadata fetch via `get_sequential_deliveries()`, sorted chronologically |
| `hybrid` | Retrieve support docs for the leaderboard leader (filter by top player/over) |
| `semantic` | Full multi-query expansion + FlashRank reranking pipeline |

**Semantic retrieval pipeline** (`retrieve_documents()`):
1. Base `similarity_search_with_score` against ChromaDB (k=20 by default)
2. Optional multi-query expansion: LLM generates 3 alternate queries, results merged by best distance
3. Optional FlashRank reranking: top-8 documents selected by cross-encoder relevance score

### Node 5: `build_context`

**Purpose:** Format retrieved data into the final context string for the LLM.

Concatenates:
- Player stats block (batting/bowling summary) when player-specific
- Aggregate stats leaderboard when applicable
- Document deliveries as structured lines: `[N] Inn X | Over.Ball | Batter | Bowler | Event | Runs | Commentary`
- Special handling for sequential/over queries: "Complete Chronological Delivery Sequence" header with explanation when innings ended early (<6 deliveries)

### Node 6: `generate_answer`

**Purpose:** Produce the final natural-language answer.

- Uses `ANSWER_PROMPT` with system rules (cite over.ball, minimum 2 sentences, no invention, include analysis)
- Supports both synchronous (`invoke_answer_chain`) and streaming (`stream_answer_chain`) modes
- The streaming variant returns a generator + prompt trace, enabling SSE token-by-token delivery

---

## Answer Strategy Routing

```text
                    ┌─── "Who hit the most sixes?" ──► hybrid (stats + support docs)
                    │
User Question ──────┼─── "What happened in over 15?" ──► sequential (chronological fetch)
                    │
                    ├─── "How many wickets fell?" ──► aggregate (stats only)
                    │
                    └─── "Who dismissed Abhishek?" ──► semantic (commentary retrieval)
```

---

## Session Memory

**Module:** `session_store.py`

- In-memory thread-safe store keyed by `session_id` (UUID from frontend).
- Each turn stores user question + assistant answer.
- **Smart pruning** on every turn:
  1. Always keep the last `MAX_HISTORY_TURNS` (default: 5) turns.
  2. For older turns, embed all questions and compute cosine similarity to the current question.
  3. Keep older turns only if similarity ≥ `HISTORY_RELEVANCE_THRESHOLD` (default: 0.6).
- This ensures topically relevant context is preserved while unrelated old turns are dropped.

---

## LLM Tracing

Every LLM call produces an `LLMTrace` (node name, full prompt, raw response) that is:
1. Accumulated in `RAGState.llm_traces[]` as the pipeline runs.
2. Included in the SSE `meta` event payload.
3. Rendered in the frontend's `PipelineInspector` as expandable trace boxes.

---

## Frontend Architecture

**Stack:** React 18 + Vite (development server on port 5173)

### Components

| Component | Purpose |
|-----------|---------|
| `App.jsx` | Main container; manages messages state, SSE streaming, session lifecycle |
| `ChatMessage.jsx` | Renders user/bot message bubbles with elapsed-time badge |
| `ExampleChips.jsx` | Welcome screen with clickable starter questions |
| `ThinkingIndicator.jsx` | Animated bouncing-dots loading indicator |
| `PipelineInspector.jsx` | Collapsible panel showing: rewritten query, history turns, retrieval filters, timing breakdown, aggregate stats, initial/reranked doc tables, LLM call traces |

### SSE Protocol

The `/api/ask` endpoint streams events:

```text
data: {"type": "meta", "rewritten_question": "...", "num_docs": N, "top_docs": [...], ...}
data: {"type": "token", "content": "partial text"}
data: {"type": "token", "content": "more text"}
data: {"type": "done", "elapsed": 2.34, "stage_timings_ms": {...}}
```

---

## Configuration

All config via `config.py` with environment variable overrides and optional `.env` file support.

### Key Configuration Groups

| Group | Variables | Defaults |
|-------|-----------|----------|
| LLM | `LLM_MODEL_PATH`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_N_CTX`, `LLM_N_GPU_LAYERS` | `models/llama-chat.gguf`, 0, 512, 4096, auto-Metal |
| Embeddings | `EMBED_MODEL_NAME`, `EMBED_MODEL_PATH`, `EMBED_DEVICE` | `BAAI/bge-small-en-v1.5`, auto-detect, cpu |
| ChromaDB | `CHROMA_PATH`, `COLLECTION_NAME` | `chroma_db`, `cricket_commentary` |
| Retrieval | `RETRIEVER_TOP_K`, `TOP_K`, `MULTI_QUERY_COUNT`, `RERANK_MODEL` | 20, 8, 3, `ms-marco-TinyBERT-L-2-v2` |
| Session | `MAX_HISTORY_TURNS`, `HISTORY_RELEVANCE_THRESHOLD` | 5, 0.6 |
| API | `API_PORT`, `API_HOST` | 5001, 0.0.0.0 |

---

## Data Scripts

| Script | Purpose |
|--------|---------|
| `scripts/scrape_commentary.py` | Fetches ball-by-ball commentary from ESPN Cricinfo mobile API, resolves athlete names, outputs structured JSON |
| `scripts/append_commentary.py` | Merges scraped commentary text back into CricSheet JSON files by matching over/delivery order |

---

## Module Reference

```text
MatchRag/
├── server.py                  # Flask API server (SSE streaming, session management)
├── chat.py                    # CLI REPL chatbot
├── config.py                  # Centralized configuration with env var overrides
├── rag/
│   ├── __init__.py            # Package public API
│   ├── rag_graph.py           # LangGraph assembly, ask() and ask_stream() entry points
│   ├── graph_nodes.py         # Node implementations (rewrite, plan, stats, retrieve, context, answer)
│   ├── chains.py              # LangChain chains (rewrite, retrieval plan, multi-query, answer)
│   ├── prompts.py             # ChatPromptTemplate definitions
│   ├── schemas.py             # Pydantic models (RetrievalPlan, LLMTrace, AnswerStrategy)
│   ├── state.py               # RAGState TypedDict (19-field pipeline state)
│   ├── providers.py           # Model factories (ChatLlamaCpp, HuggingFaceEmbeddings)
│   ├── llm_services.py        # Low-level LLM wrappers (call/stream)
│   ├── retrievers.py          # Multi-query expansion + reranking composition
│   ├── reranker.py            # FlashRank cross-encoder reranking
│   ├── vector_store.py        # ChromaDB operations + deterministic stat helpers
│   ├── documents.py           # Record ↔ LangChain Document conversion
│   ├── flatten_data.py        # CricSheet JSON → flat delivery records
│   ├── load_match.py          # JSON file loading and validation
│   ├── embedding_pipeline.py  # Batch embedding generation helpers
│   ├── ingest.py              # Startup ingestion orchestrator
│   └── session_store.py       # In-memory session store with smart pruning
├── web/
│   └── src/
│       ├── App.jsx            # Main React app (SSE client, state management)
│       ├── main.jsx           # React entry point
│       ├── index.css          # Global styles (dark theme, glassmorphism)
│       ├── App.css            # Additional styles
│       └── components/
│           ├── ChatMessage.jsx
│           ├── ExampleChips.jsx
│           ├── ThinkingIndicator.jsx
│           └── PipelineInspector.jsx
├── scripts/
│   ├── scrape_commentary.py   # ESPN Cricinfo commentary scraper
│   └── append_commentary.py   # Commentary merger for CricSheet files
├── data/
│   ├── IndVsNZ.json           # Primary match data (T20 WC Final)
│   ├── IndVsWI.json           # Secondary match data
│   └── recently_added_30_male_json/   # Additional match JSONs
├── tests/
│   ├── test_pipeline.py
│   ├── test_graph.py
│   ├── test_chains.py
│   └── test_retrievers.py
├── docs/
│   └── architecture.md        # This file
├── models/                    # Local GGUF model files
├── chroma_db/                 # Persistent ChromaDB storage
├── requirements.txt
├── pyproject.toml
├── Makefile
├── BACKLOG.md
└── README.md
```

---

## Design Principles

1. **Fully local runtime** — No external API calls at query time. All models run on-device.
2. **Deterministic stats over LLM** — Exact cricket computations (leaderboards, player stats, aggregates) are calculated from structured metadata, not hallucinated by the LLM.
3. **LangChain for abstraction** — Use LangChain where it adds meaningful value (prompts, models, documents, vector stores), keep domain logic in plain Python.
4. **Inspectability** — Full prompt/response traces, per-stage timings, retrieval metadata, and document tables visible through the Pipeline Inspector.
5. **Strategy-based routing** — Questions are classified into `semantic | aggregate | sequential | hybrid` strategies, each with optimized retrieval paths.
6. **Learning reference** — Clean architecture that demonstrates real RAG patterns rather than shortcuts.
