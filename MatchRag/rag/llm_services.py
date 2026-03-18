"""Compatibility wrappers for LangChain-backed local model access."""

from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag.providers import get_chat_model


def _to_messages(messages: list[dict]):
    converted = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def call_chat_llm(messages: list[dict]) -> str:
    """Call the local chat model without streaming."""
    response = get_chat_model().invoke(_to_messages(messages))
    return response.content if isinstance(response.content, str) else str(response.content)


def call_chat_llm_stream(messages: list[dict]) -> Generator[str, None, None]:
    """Call the local chat model with streaming."""
    for chunk in get_chat_model().stream(_to_messages(messages)):
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield content
