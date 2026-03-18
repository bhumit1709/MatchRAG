# 🏏 MatchRAG — Production Readiness Backlog

> **Goal:** Transform the current POC into a robust, deployable, multi-match RAG system.
> Created: 2026-03-05

---

## 1. 🔁 Conversation Memory & Multi-Turn Chat

**Problem:** Every question is stateless — follow-ups like *"What else did he do?"* fail because no chat history reaches the LLM.

**Action Items:**
- [ ] Maintain a per-session message history (last N turns) in the backend
- [ ] Pass conversation history to the LLM alongside the retrieved context
- [ ] Add a `/api/session` endpoint (or accept a `session_id` in `/api/ask`) to track sessions
- [ ] Add a "New Chat" button in the React UI to reset the conversation
- [ ] Cap history window (e.g., last 10 turns) to stay within token limits

**Impact:** Unlocks natural follow-up questions, pronoun resolution, and comparative queries — the #1 feature users expect from a "chatbot".

---

## 2. 📂 Multi-Match Support & Dynamic Ingestion

**Problem:** The system is hardwired to a single match file (`IndVsNZ.json`). Loading a second match silently overwrites the index.

**Action Items:**
- [ ] Namespace ChromaDB collections per match (e.g., `cricket_ind_vs_wi_2024`)
- [ ] Add a `/api/matches` endpoint to list available/indexed matches
- [ ] Add a `/api/ingest` endpoint to upload and index new match JSON files at runtime
- [ ] Support querying across multiple matches simultaneously (cross-match search)
- [ ] Add a match selector dropdown in the React UI
- [ ] Store match metadata in a lightweight SQLite/JSON manifest

**Impact:** Turns the project from a one-trick demo into a reusable tool for any cricket match.

---

## 3. 🔒 API Security, Rate Limiting & Production Server

**Problem:** The Flask server has no auth, no rate limiting, no timeouts, and uses the built-in dev server.

**Action Items:**
- [ ] Add API key authentication (header-based, e.g., `X-API-Key`)
- [ ] Add rate limiting per IP/key (e.g., via `flask-limiter`, 10 req/min)
- [ ] Set request timeouts for Ollama calls (e.g., 60s max) to prevent hung threads
- [ ] Switch to a production WSGI server (Gunicorn with 2-4 workers)
- [ ] Add structured JSON logging (replace all `print()` with `logging` module)
- [ ] Add CORS origin allowlisting instead of blanket `CORS(app)`
- [ ] Add a `Dockerfile` + `docker-compose.yml` for reproducible deployment

**Impact:** Prevents abuse, improves reliability, and makes the service deployable beyond localhost.

---

## 4. ✅ Comprehensive Testing & CI Pipeline

**Problem:** `test_rag.py` is a bare script (not a real test), there are zero API/integration tests, and no CI.

**Action Items:**
- [ ] Fix `test_rag.py` — add `pytest.mark.skipif` when index is missing, add assertions
- [ ] Add unit tests for `vector_store.query()` with mocked embeddings
- [ ] Add API integration tests for `/api/ask`, `/api/status` (use Flask test client)
- [ ] Mock Ollama calls in tests so the suite runs without a GPU/model
- [ ] Add a GitHub Actions CI workflow (lint → test → type-check on every PR)
- [ ] Add `mypy` or `pyright` for static type checking
- [ ] Target ≥80% code coverage on the `rag/` package

**Impact:** Catches regressions early, enables confident refactoring, and signals project maturity.

---

## 5. 📌 Dependency Pinning & Reproducible Environments

**Problem:** `requirements.txt` has zero version pins — installs may break at any time due to upstream changes.

**Action Items:**
- [ ] Pin all direct dependencies with exact versions (e.g., `chromadb==0.5.23`)
- [ ] Generate a lockfile (`pip-compile` via `pip-tools`, or use `uv`)
- [ ] Add a `python-version` badge/check (enforce ≥3.10)
- [ ] Pin Node.js / npm versions for the frontend (`engines` in `package.json`)
- [ ] Add `pre-commit` hooks for linting (ruff) and formatting (black/ruff-format)
- [ ] Document the exact Ollama model versions tested against

**Impact:** Guarantees that `git clone && make install` works identically on any machine, today and 6 months from now.

---

## Priority Matrix

| # | Item | Effort | Impact | Priority |
|---|------|--------|--------|----------|
| 1 | Conversation Memory | Medium | 🔴 Critical | P0 |
| 2 | Multi-Match Support | High | 🔴 Critical | P0 |
| 3 | API Security & Prod Server | Medium | 🟠 High | P1 |
| 4 | Testing & CI | Medium | 🟠 High | P1 |
| 5 | Dependency Pinning | Low | 🟡 Medium | P2 |
