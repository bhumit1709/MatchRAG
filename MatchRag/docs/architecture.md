# MatchRAG Architecture

## Overview

MatchRAG is a fully local learning project for RAG architecture built with modern LangChain and LangGraph patterns.

- `ChatLlamaCpp` handles generation from a local GGUF model
- `HuggingFaceEmbeddings` handles local embeddings
- `Chroma` stores delivery vectors
- `LangGraph` orchestrates retrieval planning, retrieval, context building, and answer generation

The app intentionally separates deterministic cricket/stat logic from LLM-driven language tasks.

## Request Flow

```text
user question
  -> rewrite_question
  -> plan_retrieval
  -> compute_aggregate_stats
  -> retrieve
  -> build_context
  -> generate_answer
```

### 1. Rewrite

Only follow-up questions are rewritten. Standalone questions bypass this step.

### 2. Retrieval Planning

A LangChain prompt chain produces a typed `RetrievalPlan` with:

- normalized question
- players
- event
- innings
- over
- stat/sequential flags
- grouping and metric hints

Deterministic Python then converts that plan into Chroma metadata filters.

### 3. Retrieval

The retriever layer uses:

- base semantic search from Chroma
- optional multi-query expansion
- compression/rerank to reduce noisy candidates

Sequential requests bypass vector similarity and use exact metadata retrieval.

### 4. Context Building

Context combines:

- deterministic aggregate stats when needed
- exact player stats when requested
- retrieved commentary deliveries with over.ball citations

### 5. Answer Generation

The answer chain uses `ChatPromptTemplate` plus the local chat model and produces the final response or streamed tokens.

## Module Responsibilities

- `rag/providers.py`: local model factories and runtime validation
- `rag/documents.py`: flattened-record to LangChain `Document` conversion
- `rag/chains.py`: prompt chains and structured parsing
- `rag/retrievers.py`: retrieval composition and query expansion
- `rag/vector_store.py`: Chroma integration plus deterministic stats helpers
- `rag/graph_nodes.py`: LangGraph node orchestration
- `rag/rag_graph.py`: graph assembly and public ask/ask_stream entrypoints
- `rag/session_store.py`: memory pruning with local embeddings

## Design Principles

- keep all runtime models local
- use LangChain for model, prompt, document, and vector-store abstractions
- keep exact computations deterministic
- preserve inspectability through prompt/response traces
- favor clean learning architecture over one-off shortcuts
