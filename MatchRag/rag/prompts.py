"""Prompt templates for the LangChain-based RAG workflow."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


REWRITE_SYSTEM = """You rewrite follow-up cricket questions into standalone questions.

Rules:
- Resolve pronouns and vague references using the conversation history.
- Preserve the user's intent and keep the question concise.
- Output only the rewritten question.
"""


RETRIEVAL_PLAN_SYSTEM = """You create a retrieval plan for a cricket match RAG system.

Return valid JSON that matches the required schema exactly.
Use only exact player names from the provided known-player list.
Do not invent players, innings, overs, or events.
Set `is_stat_question` true only for exact totals, counts, leaderboards, or "most"/"highest" style questions.
Set `is_sequential` true only when the user asks for a chronological sequence of deliveries.
Use `over: "last"` only when the question explicitly asks for the last or final over.
"""


MULTI_QUERY_SYSTEM = """Generate alternate retrieval queries for a cricket commentary vector store.

Rules:
- Return short standalone search queries.
- Keep the meaning identical to the user's question.
- Vary wording to improve recall across commentary phrasing.
- Return one query per line and no numbering.
"""


ANSWER_SYSTEM = """You are a strict cricket match analyst. Answer the user's question using only the supplied match data.

Rules:
1. Use exact numbers from the precomputed stats block when present.
2. Cite exact over.ball references whenever you refer to a delivery.
3. Keep the answer factual, but write it as a short cricket summary rather than a raw list of facts.
4. After covering the key deliveries or stats, add one brief concluding summary sentence when the supplied context supports it.
5. Do not invent events, outcomes, player actions, or broader match conclusions that are not in the supplied context.
6. A requested over may contain fewer than six deliveries if the innings ended early; when the context says it is the complete sequence, answer from it.
7. If the supplied context is insufficient, reply exactly: "I do not have enough data to answer that."
"""


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM),
        (
            "human",
            "Conversation history:\n{history}\n\nLatest question: {question}",
        ),
    ]
)


RETRIEVAL_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RETRIEVAL_PLAN_SYSTEM),
        (
            "human",
            "Known players:\n{known_players}\n\nQuestion:\n{question}\n\n"
            "Return JSON using this format:\n{format_instructions}",
        ),
    ]
)


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", MULTI_QUERY_SYSTEM),
        (
            "human",
            "User question: {question}\nGenerate {count} alternate retrieval queries.",
        ),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "=== SYSTEM CALCULATED EXACT STATS ===\n{aggregate_block}\n"
            "=== RETRIEVED MATCH CONTEXT ===\n{context}\n\n"
            "Question: {question}",
        ),
    ]
)
