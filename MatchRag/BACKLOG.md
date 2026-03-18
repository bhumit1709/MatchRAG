# 🏏 MatchRAG — Production Readiness Backlog

> **Goal:** Transform the current POC into a robust, deployable, multi-match RAG system.
> Created: 2026-03-05

---

## Unsupported Question Families For The India vs New Zealand Final

This section lists the question types the chatbot does not currently support reliably for the indexed India vs New Zealand final match.

### 1. Phase-Based Exact Analytics

**Unsupported questions:**
- "How many runs did India score in the powerplay?"
- "Who was best in the death overs?"
- "How many wickets fell in the middle overs?"

**Why not supported:**
- The retrieval plan and metadata filters do not model match phases such as powerplay, middle overs, or death overs.

**Tasks:**
- [ ] Add phase fields to the retrieval plan schema.
- [ ] Implement deterministic filters for powerplay, middle overs, and death overs.
- [ ] Add exact aggregate helpers for runs, wickets, strike rate, and economy by phase.
- [ ] Add tests for phase-based questions.

### 2. Partnership Analytics

**Unsupported questions:**
- "What was the highest partnership?"
- "Who had the best partnership for India?"
- "How many runs did Samson and Pandya add together?"

**Why not supported:**
- Partnership spans and partnership runs are not derived anywhere in the current pipeline.

**Tasks:**
- [ ] Compute partnerships from wicket boundaries and scoring sequences.
- [ ] Add partnership aggregate helpers and retrieval routes.
- [ ] Add answer templates for partnership summary questions.
- [ ] Add tests for highest partnership and named-pair partnership queries.

### 3. Batter-vs-Bowler Matchup Stats

**Unsupported questions:**
- "How many runs did Samson score off Henry?"
- "How did Bumrah bowl to Seifert?"
- "Which bowler troubled Samson the most?"

**Why not supported:**
- The planner does not support exact aggregation over batter-bowler pairs.

**Tasks:**
- [ ] Add matchup intent detection to the retrieval planner.
- [ ] Implement exact aggregations over batter and bowler combinations.
- [ ] Add answer templates for matchup questions.
- [ ] Add tests for batter-vs-bowler stats.

### 4. Exact Score Progression And Scorecards

**Unsupported questions:**
- "What was India's score after 10 overs?"
- "What was New Zealand's score at 15.3?"
- "Give me the batting scorecard."
- "Show the bowling figures."

**Why not supported:**
- The current context builder does not compute running score progression or full scorecard outputs.

**Tasks:**
- [ ] Build running score progression by ball and by over.
- [ ] Add helpers for score-at-over and score-at-ball questions.
- [ ] Generate batting and bowling scorecards from structured data.
- [ ] Add tests for score snapshots and scorecards.

### 5. Extras And Wicket-Mode Analytics

**Unsupported questions:**
- "How many wides were bowled?"
- "How many leg-byes were there?"
- "What was the most common wicket type?"

**Why not supported:**
- Extras and wicket-kind analysis are not exposed as user-facing exact aggregations.

**Tasks:**
- [ ] Extend the event taxonomy to include wides, no-balls, byes, leg-byes, and wicket kinds.
- [ ] Add exact aggregation helpers for extras and dismissal types.
- [ ] Add planner support for these analytics questions.
- [ ] Add tests for extras and wicket-kind queries.

### 6. Review And Umpiring Analytics

**Unsupported questions:**
- "How many DRS reviews were taken?"
- "Which reviews were successful?"
- "How many umpire's calls were there?"

**Why not supported:**
- Review events only exist implicitly in commentary text and are not extracted into structured statistics.

**Tasks:**
- [ ] Extract review events from commentary into structured metadata.
- [ ] Add exact counters for reviews, outcomes, and umpire's call decisions.
- [ ] Add answer templates for review-analysis questions.
- [ ] Add tests for DRS-related questions.

### 7. Cross-Match Or Tournament Questions

**Unsupported questions:**
- "How did India perform across the tournament?"
- "Compare this final to the West Indies match."
- "Which match was Samson's best?"

**Why not supported:**
- The active app is scoped to a single indexed match and does not support cross-match retrieval in one query.

**Tasks:**
- [ ] Add multi-match indexing with per-match namespaces.
- [ ] Add match selection and cross-match retrieval support.
- [ ] Add planner support for cross-match comparison questions.
- [ ] Add tests for multi-match querying.

### 8. Hypothetical Or Speculative Questions

**Unsupported questions:**
- "What if Samson got out early?"
- "Why did New Zealand choke?"
- "Would India still have won without Bumrah?"

**Why not supported:**
- The assistant is intentionally grounded in supplied match data and should not invent counterfactual analysis.

**Tasks:**
- [ ] Add explicit unsupported-intent detection for speculative questions.
- [ ] Return a clear refusal or fallback message for counterfactual requests.
- [ ] Add tests that verify the bot declines speculative questions safely.

### 9. Visual Analytics Requests

**Unsupported questions:**
- "Show a worm chart."
- "Plot a wagon wheel for Samson."
- "Give me a run-rate graph."

**Why not supported:**
- The app does not generate visual analytics or chart-ready structured output.

**Tasks:**
- [ ] Add chart-oriented data builders for run rate, score progression, and shot summaries.
- [ ] Decide whether visuals should be rendered in the frontend or returned as structured API payloads.
- [ ] Add tests for chart data generation.

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
