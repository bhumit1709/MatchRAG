import ollama
from typing import Generator
from config import LLM_MODEL
from rag.prompts import REWRITE_PROMPT, EXTRACT_PROMPT

def call_rewrite_llm(prompt: str) -> str:
    """Call the LLM for rewriting a question."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return response["message"]["content"]

def call_extract_llm(prompt: str) -> str:
    """Call the LLM for entity extraction."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return response["message"]["content"]

def call_chat_llm(messages: list[dict]) -> str:
    """Call the LLM for main answer generation without streaming."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
    )
    return response["message"]["content"]

def call_chat_llm_stream(messages: list[dict]) -> Generator[str, None, None]:
    """Call the LLM for main answer generation with streaming."""
    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
    ):
        yield chunk["message"]["content"]
