"""Utilities for converting flattened deliveries into LangChain documents."""

from langchain_core.documents import Document


METADATA_FIELDS = (
    "id",
    "match",
    "venue",
    "season",
    "event_name",
    "innings",
    "batting_team",
    "over",
    "ball",
    "batter",
    "bowler",
    "non_striker",
    "event",
    "player_out",
    "wicket_kind",
    "wicket_fielder",
    "runs_batter",
    "runs_extras",
    "runs_total",
    "commentary",
)


def record_to_document(record: dict) -> Document:
    """Convert a flattened delivery record into a LangChain document."""
    metadata = {field: record.get(field, "") for field in METADATA_FIELDS}
    return Document(page_content=record["text"], metadata=metadata)


def records_to_documents(records: list[dict]) -> list[Document]:
    """Convert all flattened records to LangChain documents."""
    return [record_to_document(record) for record in records]


def document_id(document: Document) -> str:
    """Return the stable delivery id for a document."""
    return str(document.metadata.get("id", ""))


def serialize_document(
    document: Document,
    *,
    distance: float | None = None,
    score: float | None = None,
) -> dict:
    """Convert a LangChain document into the dict shape used by the UI."""
    payload = {
        "text": document.page_content,
        "metadata": dict(document.metadata),
    }
    if distance is not None:
        payload["distance"] = float(distance)
    if score is not None:
        payload["score"] = float(score)
    return payload
