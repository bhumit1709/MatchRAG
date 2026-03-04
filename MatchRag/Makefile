.PHONY: help install server chat test lint rebuild clean

PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest

# ── Default target ──────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  MatchRAG — available commands"
	@echo "  ──────────────────────────────────────────────"
	@echo "  make install   Install Python dependencies into .venv"
	@echo "  make server    Start the Flask API server (port 5001)"
	@echo "  make chat      Start the interactive CLI chatbot"
	@echo "  make test      Run test suite with pytest"
	@echo "  make lint      Lint Python code with ruff"
	@echo "  make rebuild   Force-rebuild the ChromaDB index"
	@echo "  make clean     Remove __pycache__, chroma_db, dist"
	@echo ""

# ── Setup ───────────────────────────────────────────────────────────────────
install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── Run ─────────────────────────────────────────────────────────────────────
server:
	$(PYTHON) server.py --port 5001

chat:
	$(PYTHON) chat.py

rebuild:
	$(PYTHON) server.py --rebuild --port 5001

# ── Quality ─────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v

lint:
	$(PYTHON) -m ruff check . --fix

# ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	rm -rf chroma_db dist build *.egg-info .pytest_cache .coverage
