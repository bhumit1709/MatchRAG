# MatchRAG

Local cricket-match RAG for learning modern `LangChain` + `LangGraph` patterns with small local models.

The app stays fully local at runtime:

- Generation: `llama.cpp` through LangChain `ChatLlamaCpp`
- Embeddings: local Hugging Face model through LangChain `HuggingFaceEmbeddings`
- Retrieval: `langchain_chroma.Chroma`
- Orchestration: `LangGraph`

## Why This Repo Exists

This project is meant to be a strong learning reference, not a shortcut demo:

- use LangChain wherever it meaningfully improves clarity
- keep deterministic cricket/stat logic in plain Python
- keep models local and inspectable
- preserve good architecture even in a learning project

## Setup

1. Install Python dependencies.

```bash
pip install -r requirements.txt
```

2. Download a local GGUF chat model and point `LLM_MODEL_PATH` to it.

```bash
export LLM_MODEL_PATH=/absolute/path/to/your-model.gguf
```

3. Optionally pin the embedding model to a local folder.

```bash
export EMBED_MODEL_PATH=/absolute/path/to/local-bge-model
```

If `EMBED_MODEL_PATH` is unset, the default `BAAI/bge-small-en-v1.5` model is downloaded once and then used locally from cache.

4. Install frontend dependencies.

```bash
cd web && npm install && cd ..
```

## Run

Backend:

```bash
python3 server.py
```

CLI:

```bash
python3 chat.py
```

Frontend:

```bash
cd web && npm run dev
```

## Pipeline

```text
load_match
  -> flatten_deliveries
  -> LangChain Documents
  -> Chroma index

question
  -> rewrite_question
  -> retrieval_plan
  -> metadata filters
  -> semantic retrieval
  -> multi-query expansion
  -> compression / rerank
  -> context builder
  -> answer generation
```

## Main Modules

- `rag/providers.py`: shared local LLM and embedding providers
- `rag/chains.py`: LangChain prompt chains
- `rag/retrievers.py`: semantic retrieval, multi-query expansion, compression
- `rag/graph_nodes.py`: LangGraph node logic
- `rag/vector_store.py`: Chroma integration plus deterministic cricket helpers
- `rag/ingest.py`: shared indexing bootstrap for CLI and server

## Example Questions

```text
Who dismissed Abhishek Sharma?
What happened in the final over?
Who hit the most sixes?
Show all wickets taken by Bumrah.
Which over had the most runs scored?
```

## Tests

The test suite is split into:

- ingestion/data-shape tests
- retrieval behavior tests
- graph behavior tests
- chain prompt-shape tests

Run with:

```bash
python3 -m pytest tests -q
```
