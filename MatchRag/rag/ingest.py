"""Shared ingestion/bootstrap helpers used by the CLI and API server."""

from time import perf_counter

from rag.documents import records_to_documents
from rag.flatten_data import flatten_deliveries
from rag.load_match import load_match
from rag.providers import ensure_local_models_ready
from rag.vector_store import build_index, collection_exists, index_matches_runtime


def run_ingest(filepath: str, force_rebuild: bool = False, verbose: bool = True) -> dict:
    """Load the match data and ensure the Chroma index exists."""
    existing_index = collection_exists()
    compatible_index = existing_index and index_matches_runtime(filepath)
    should_rebuild = force_rebuild or (existing_index and not compatible_index)

    if existing_index and compatible_index and not force_rebuild:
        return {"skipped": True, "records": 0, "elapsed": 0.0, "reason": "up_to_date"}

    ensure_local_models_ready()

    start = perf_counter()
    data = load_match(filepath)
    records = flatten_deliveries(data)
    documents = records_to_documents(records)
    build_index(documents, reset=should_rebuild, source_file=filepath)

    result = {
        "skipped": False,
        "records": len(records),
        "elapsed": perf_counter() - start,
        "reason": "forced" if force_rebuild else ("embedding_changed" if should_rebuild else "missing"),
    }

    if verbose:
        print(f"Indexed {result['records']} deliveries in {result['elapsed']:.1f}s.")

    return result
