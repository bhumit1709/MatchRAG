"""LangChain chains for rewriting, retrieval planning, query expansion, and answering."""

from __future__ import annotations

import json
import re
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from config import MULTI_QUERY_COUNT
from rag.prompts import ANSWER_PROMPT, MULTI_QUERY_PROMPT, RETRIEVAL_PLAN_PROMPT, REWRITE_PROMPT
from rag.providers import get_chat_model
from rag.schemas import LLMTrace, RetrievalPlan


_retrieval_plan_parser = PydanticOutputParser(pydantic_object=RetrievalPlan)


def _stringify_prompt(prompt_value) -> str:
    return prompt_value.to_string()


def _trace(node: str, prompt_value, response: str) -> dict:
    return LLMTrace(
        node=node,
        prompt=_stringify_prompt(prompt_value),
        response=response,
    ).model_dump()


def _extract_json_object(raw: str) -> str:
    cleaned = raw.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
    start = fenced.find("{")
    if start == -1:
        return fenced

    depth = 0
    for index in range(start, len(fenced)):
        char = fenced[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return fenced[start : index + 1]
    return fenced[start:]


def history_to_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert stored history dicts into LangChain message objects."""
    messages: list[BaseMessage] = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def rewrite_followup_question(question: str, history: list[dict]) -> tuple[str, dict]:
    """Rewrite a follow-up question into a standalone query."""
    history_text = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in history[-6:]
    )
    inputs = {"history": history_text, "question": question}
    prompt_value = REWRITE_PROMPT.invoke(inputs)
    chain = REWRITE_PROMPT | get_chat_model() | StrOutputParser()
    rewritten = chain.invoke(inputs).strip().strip('"')
    rewritten = re.sub(r"\s*\(.*?\)\s*$", "", rewritten).strip() or question
    return rewritten, _trace("rewrite_question", prompt_value, rewritten)


def build_retrieval_plan(question: str, known_players: list[str]) -> tuple[RetrievalPlan, dict]:
    """Create a structured retrieval plan for the question."""
    inputs = {
        "question": question,
        "known_players": ", ".join(known_players) if known_players else "None available",
        "format_instructions": _retrieval_plan_parser.get_format_instructions(),
    }
    prompt_value = RETRIEVAL_PLAN_PROMPT.invoke(inputs)
    chain = RETRIEVAL_PLAN_PROMPT | get_chat_model() | StrOutputParser()
    raw = chain.invoke(inputs).strip()

    try:
        plan = _retrieval_plan_parser.parse(raw)
    except Exception:
        plan = _retrieval_plan_parser.parse(_extract_json_object(raw))

    if not plan.normalized_question:
        plan.normalized_question = question

    return plan, _trace("plan_retrieval", prompt_value, raw)


def generate_query_variants(question: str, count: int = MULTI_QUERY_COUNT) -> tuple[list[str], dict]:
    """Generate alternate retrieval queries for better recall."""
    inputs = {"question": question, "count": count}
    prompt_value = MULTI_QUERY_PROMPT.invoke(inputs)
    chain = MULTI_QUERY_PROMPT | get_chat_model() | StrOutputParser()
    raw = chain.invoke(inputs).strip()

    variants = [question]
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*[-*\d.]+\s*", "", line).strip().strip('"')
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    return variants[: count + 1], _trace("generate_query_variants", prompt_value, "\n".join(variants))


def build_answer_prompt_value(
    question: str,
    chat_history: list[dict],
    context: str,
    aggregate_stats: str | None,
):
    """Build the final answer prompt value."""
    return ANSWER_PROMPT.invoke(
        {
            "chat_history": history_to_messages(chat_history),
            "aggregate_block": aggregate_stats or "No exact aggregate stats supplied.",
            "context": context,
            "question": question,
        }
    )


def invoke_answer_chain(
    question: str,
    chat_history: list[dict],
    context: str,
    aggregate_stats: str | None,
) -> tuple[str, dict]:
    """Run the final answer-generation chain."""
    prompt_value = build_answer_prompt_value(question, chat_history, context, aggregate_stats)
    response = get_chat_model().invoke(prompt_value.to_messages())
    content = response.content if isinstance(response.content, str) else json.dumps(response.content)
    return content.strip(), _trace("generate_answer", prompt_value, content)


def stream_answer_chain(
    question: str,
    chat_history: list[dict],
    context: str,
    aggregate_stats: str | None,
) -> tuple[Iterable[str], dict]:
    """Stream the final answer-generation chain."""
    prompt_value = build_answer_prompt_value(question, chat_history, context, aggregate_stats)
    trace = _trace("generate_answer", prompt_value, "<streamed to chat UI>")

    def _stream() -> Iterable[str]:
        for chunk in get_chat_model().stream(prompt_value.to_messages()):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield content

    return _stream(), trace
