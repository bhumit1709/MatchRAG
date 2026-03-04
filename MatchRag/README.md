# 🏏 Cricket Match RAG Pipeline

A fully local Retrieval Augmented Generation (RAG) system for querying ball-by-ball cricket commentary in natural language.

**Stack:** Python · LangGraph · ChromaDB · Ollama

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | [python.org](https://python.org) |
| Ollama | [ollama.com](https://ollama.com) |

---

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Pull required Ollama models (only needed once)
ollama pull nomic-embed-text
ollama pull mistral

# 3. Install React frontend dependencies
cd web && npm install && cd ..
```

---

## Usage

### 🌐 Web UI (React)

Start the Flask API backend and the React dev server in two separate terminals:

**Terminal 1 — Flask API:**
```bash
python server.py
```

**Terminal 2 — React frontend:**
```bash
cd web && npm run dev
```

Then open **http://localhost:5173** in your browser.

- On first run, the API automatically builds the ChromaDB index.
- The React UI shows live model/index status in the header.
- Click example question chips or type your own question.

---

### 💻 CLI Chatbot

```bash
python chat.py
```


On first run, the pipeline **automatically**:
1. Loads `IndVsWI.json`
2. Flattens all 249 deliveries into flat records with event detection
3. Generates embeddings via `nomic-embed-text`
4. Stores vectors in a local ChromaDB database (`./chroma_db/`)

Subsequent runs reuse the index (fast startup — under 2 seconds).

### Force re-index

```bash
python chat.py --rebuild
```

### Use a different match file

```bash
python chat.py --file path/to/other_match.json
```

---

## Example Questions

```
Who dismissed Shimron Hetmyer?
What happened in the final over?
Who hit the most sixes?
Show all wickets taken by Bumrah.
How many dot balls were bowled in the first innings?
What was Samson's contribution to India's win?
Which over had the most runs scored?
```

---

## File Structure

```
MatchRag/
├── IndVsWI.json            # Source match data (CricSheet format + commentary)
├── load_match.py           # Step 1: JSON loader & metadata extractor
├── flatten_data.py         # Steps 2-4: Flatten, event detection, embedding text
├── embedding_pipeline.py   # Step 5: Ollama nomic-embed-text embeddings
├── vector_store.py         # Step 6: ChromaDB persistence & retrieval
├── rag_graph.py            # Steps 7-8: LangGraph retrieve→context→answer graph
├── chat.py                 # Steps 9-10: CLI chatbot entry point
├── requirements.txt
└── chroma_db/              # Auto-created persistent vector store
```

---

## Pipeline Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  retrieve   │  ← ChromaDB cosine search (nomic-embed-text)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  build_context   │  ← Format top-6 deliveries into prompt context
└──────┬───────────┘
       │
       ▼
┌─────────────────────┐
│  generate_answer    │  ← Ollama mistral LLM response
└──────┬──────────────┘
       │
       ▼
   Answer ✅
```

---

## Running Individual Modules

```bash
python load_match.py          # Verify JSON loads correctly
python flatten_data.py        # Preview flattened records + event breakdown
python embedding_pipeline.py  # Test embedding generation
python vector_store.py        # Build index + run sample queries
python rag_graph.py           # Run a quick 3-question LLM test
```
