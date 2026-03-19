# MatchRAG — Code Optimisation Backlog

> Improvement areas identified from full code analysis. Organised by section with actionable tasks.

---

## 1. LangChain / LangGraph (Score: 8/10)

### 1.1 Use the compiled LangGraph for execution ✅
**Files:** `rag/rag_graph.py`, `rag/graph_nodes.py`
- [x] Two compiled graphs: `_full_graph` (6 nodes, for `ask()`) and `_pre_answer_graph` (5 nodes, for `ask_stream()`)
- [x] `_timed(node_name)` wrapper uses dynamic `getattr` lookup at call time — monkeypatching works in tests
- [x] `ask()` calls `_get_full_graph().invoke(state)` — LangGraph drives all state passing
- [x] `ask_stream()` calls `_get_pre_answer_graph().invoke(state)` for nodes 1–5, then streams tokens outside the graph
- [x] `build_graph()` kept as backward-compatible alias
- [x] Also fixed 2 pre-existing bugs discovered during verification (see below):
  - `_is_summary_question`: `"over" in q` falsely matched `"overall"` — fixed with `re.search(r"\bover\b")`
  - Fast-path semantic plans now enriched with player detection so `retrieval_plan.players` is populated
- [x] **All 34 tests pass** (`test_graph.py` + `test_pipeline.py`)

### 1.2 Add conditional edges for strategy routing
**Files:** `rag/rag_graph.py`, `rag/graph_nodes.py`
- [ ] Use `add_conditional_edges` to route between strategy-specific retrieval nodes based on `answer_strategy`
- [ ] Makes the routing logic explicit in the graph topology instead of hidden inside if/else branches in `retrieve()`

### 1.3 Remove vestigial `llm_services.py`
**Files:** `rag/llm_services.py`
- [ ] `llm_services.py` duplicates what `chains.py` + `providers.py` already provide
- [ ] It is never imported by the main pipeline — remove or consolidate

### 1.4 Use LangChain callbacks for tracing
**Files:** `rag/chains.py`
- [ ] Replace manual `_trace()` wrappers with LangChain `CallbackHandler` for prompt/response tracing
- [ ] Benefits: automatic capture, nesting support, future integration with LangSmith

---

## 2. Python Quality (Score: 7.5/10)

### 2.1 Replace bare `except Exception` blocks
**Files:** `rag/vector_store.py` (6+ occurrences), `rag/session_store.py`
- [ ] Add `logging.exception()` or `logging.warning()` inside every catch
- [ ] Consider raising domain-specific exceptions instead of returning `None` silently
- [ ] Critical in `get_player_stats()`, `get_event_leaderboard()`, `get_sequential_deliveries()`, `get_known_players()`

### 2.2 Eliminate global mutable state
**Files:** `rag/vector_store.py`, `rag/reranker.py`, `rag/rag_graph.py`
- [ ] Wrap `_vector_store`, `_known_players_cache`, `_match_metadata_cache`, `_ranker`, `_compiled_graph` in a dependency-injection pattern or service container
- [ ] Makes testing significantly easier (inject mocks without monkeypatching globals)

### 2.3 Adopt the `logging` module
**Files:** `server.py`, `chat.py`, `rag/ingest.py`, `rag/embedding_pipeline.py`
- [ ] Replace all `print()` calls with `logging.info()` / `logging.debug()`
- [ ] Add structured JSON logging for production use
- [ ] Configure log levels via environment variable

### 2.4 Introduce a config dataclass
**Files:** `config.py`
- [ ] Replace ~25 module-level variables with a `@dataclass` config object
- [ ] Add validation (e.g., `LLM_N_CTX > 0`, `TOP_K <= RETRIEVER_TOP_K`)
- [ ] Makes config testable and injectable

### 2.5 Reduce state copying overhead
**Files:** `rag/graph_nodes.py`
- [ ] The `{**state, ...}` pattern on every node creates transient copies of large lists (`retrieved_docs`, `initial_docs`, `llm_traces`)
- [ ] Consider using a mutable state class with explicit update methods, or passing references

### 2.6 Pin all dependencies
**Files:** `requirements.txt`, `pyproject.toml`
- [ ] Pin exact versions (`chromadb==0.5.23`, not `chromadb>=0.5`)
- [ ] Add a lockfile via `pip-compile` or `uv lock`

---

## 3. Architecture & Flow (Score: 8.5/10)

### 3.1 Add error recovery / retry for LLM calls
**Files:** `rag/chains.py`
- [ ] `build_retrieval_plan()` has one JSON fallback (`_extract_json_object`) but no retry
- [ ] Add exponential backoff or a degradation path (fall back to pure semantic when planning fails)
- [ ] Consider `tenacity` library for structured retries

### 3.2 Add hierarchical chunking
**Files:** `rag/flatten_data.py`, `rag/documents.py`
- [ ] Currently each delivery is one document — no over-level or innings-level summaries
- [ ] Create parent documents for over summaries and innings summaries
- [ ] These would improve recall for broad questions ("How did the powerplay go?")

### 3.3 Build an evaluation framework
**Files:** New: `eval/` directory
- [ ] Create a golden Q&A dataset (50–100 questions with ground-truth answers)
- [ ] Add RAGAS or custom metrics (faithfulness, relevance, answer correctness)
- [ ] Run evals as part of CI to catch answer quality regressions

### 3.4 Design for multi-match from the start
**Files:** `rag/vector_store.py`, `config.py`
- [ ] Namespace ChromaDB collections per match (e.g., `cricket_ind_vs_nz_2026`)
- [ ] Add match selection to the retrieval plan
- [ ] Prevents a large refactor when multi-match support is needed

---

## 4. Code Organisation (Score: 8/10)

### 4.1 Split `graph_nodes.py`
**Files:** `rag/graph_nodes.py` (763 lines)
- [ ] Extract fast-path planning into `rag/fast_path.py`
- [ ] Extract filter building (`_build_where_filter`, `_combine_filters`) into `rag/filters.py`
- [ ] Extract context formatting (`build_context`, `_format_player_stats`) into `rag/context_builder.py`
- [ ] Keep only node function signatures and delegation in `graph_nodes.py`

### 4.2 Move inline styles to CSS
**Files:** `web/src/components/PipelineInspector.jsx`
- [ ] Replace `style={{...}}` objects with CSS classes in `index.css`
- [ ] Improves consistency and maintainability

---

## 5. Testing (Score: 4/10)

### 5.1 Make tests runnable without models
- [ ] Mock `get_chat_model()` and `get_embeddings()` in all unit tests
- [ ] Add `pytest.mark.skipif` for tests requiring the ChromaDB index

### 5.2 Add API integration tests
**Files:** New: `tests/test_api.py`
- [ ] Use Flask test client to test `/api/ask`, `/api/status`, `/api/session/clear`
- [ ] Mock the RAG pipeline to isolate API logic

### 5.3 Add golden Q&A assertions
**Files:** New: `tests/test_answers.py`
- [ ] Test 10–20 representative questions against expected answer patterns
- [ ] Assert that specific over.ball references and player names appear in responses

### 5.4 Add CI pipeline
**Files:** New: `.github/workflows/ci.yml`
- [ ] Lint (ruff) → Type check (mypy) → Unit tests → Coverage report
- [ ] Target ≥80% coverage on `rag/` package

---

## 6. Production Readiness (Score: 4/10)

### 6.1 Add authentication and rate limiting
**Files:** `server.py`
- [ ] Add `X-API-Key` header authentication
- [ ] Add `flask-limiter` rate limiting (e.g., 10 req/min per IP)

### 6.2 Switch to production WSGI server
**Files:** `server.py`, new: `Dockerfile`
- [ ] Use Gunicorn with 2–4 workers
- [ ] Add `Dockerfile` + `docker-compose.yml`

### 6.3 Restrict CORS origins
**Files:** `server.py`
- [ ] Replace blanket `CORS(app)` with explicit origin allowlist

### 6.4 Add request timeouts
**Files:** `server.py`, `rag/chains.py`
- [ ] Set LLM call timeouts (e.g., 60s max) to prevent hung threads
- [ ] Return graceful timeout errors to the client

### 6.5 Persist sessions
**Files:** `rag/session_store.py`
- [ ] Replace in-memory `defaultdict` with SQLite or Redis
- [ ] Sessions currently lost on every server restart

---

## Priority Matrix

| # | Item | Effort | Impact | Priority |
|---|------|--------|--------|----------|
| 2.1 | Fix silent exception swallowing | Low | 🔴 Critical | **P0** |
| 1.1 | Use compiled LangGraph | Medium | 🔴 Critical | **P0** |
| 2.3 | Adopt `logging` module | Low | 🟠 High | **P1** |
| 4.1 | Split `graph_nodes.py` | Medium | 🟠 High | **P1** |
| 1.2 | Conditional edges for routing | Medium | 🟠 High | **P1** |
| 3.3 | Evaluation framework | High | 🔴 Critical | **P1** |
| 5.1 | Mock-based runnable tests | Medium | 🟠 High | **P1** |
| 3.1 | LLM retry / error recovery | Low | 🟠 High | **P1** |
| 1.3 | Remove `llm_services.py` | Low | 🟡 Medium | **P2** |
| 2.2 | Eliminate global mutable state | High | 🟡 Medium | **P2** |
| 2.4 | Config dataclass | Medium | 🟡 Medium | **P2** |
| 2.6 | Pin dependencies | Low | 🟡 Medium | **P2** |
| 3.2 | Hierarchical chunking | High | 🟠 High | **P2** |
| 3.4 | Multi-match namespacing | Medium | 🟡 Medium | **P2** |
| 5.2 | API integration tests | Medium | 🟡 Medium | **P2** |
| 5.4 | CI pipeline | Medium | 🟡 Medium | **P2** |
| 6.1–6.5 | Production hardening (all) | High | 🟡 Medium | **P3** |
