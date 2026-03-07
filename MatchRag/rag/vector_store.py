"""
vector_store.py
---------------
Step 6 of the RAG pipeline.

Manages a persistent ChromaDB collection for cricket commentary embeddings.
Provides functions to build the index and run semantic searches.
"""

import sys
import chromadb
from chromadb.config import Settings
import ollama

from rag.embedding_pipeline import EMBED_MODEL
from config import CHROMA_PATH, COLLECTION_NAME


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client stored at CHROMA_PATH."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(client: chromadb.PersistentClient = None):
    """
    Get or create the cricket commentary ChromaDB collection.

    Args:
        client: Optional existing ChromaDB client; one is created if not provided.

    Returns:
        ChromaDB Collection object.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Use cosine distance for semantic similarity
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    documents: list[dict],
    embeddings: list[list[float]],
    reset: bool = False,
) -> None:
    """
    Upsert all delivery documents and their embeddings into ChromaDB.

    Args:
        documents: Flat delivery dicts (from flatten_data.py).
        embeddings: Embedding vectors in the same order as documents.
        reset: If True, delete the existing collection before inserting.
    """
    client = get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = get_collection(client)

    # ChromaDB requires string IDs, plain text content, and a metadata dict
    ids = [doc["id"] for doc in documents]

    # Metadata: only scalar types (str, int, float, bool) are supported
    metadatas = [
        {
            "match":   doc["match"],
            "innings": doc["innings"],
            "over":    doc["over"],
            "ball":    doc["ball"],
            "batter":  doc["batter"],
            "bowler":  doc["bowler"],
            "event":   doc["event"],
            "venue":   doc["venue"],
            "season":  doc["season"],
            "batting_team": doc["batting_team"],
            "player_out":   doc.get("player_out", ""),
            "wicket_kind":  doc.get("wicket_kind", ""),
            "wicket_fielder": doc.get("wicket_fielder", ""),
            "runs_total":   doc["runs_total"],
        }
        for doc in documents
    ]

    # The document text stored in Chroma (used for display, not search)
    texts = [doc["text"] for doc in documents]

    print(f"Upserting {len(ids)} records into ChromaDB collection '{COLLECTION_NAME}'...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Index built. Total records: {collection.count()}")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def query(
    question: str,
    n_results: int = 6,
    where: dict = None,
) -> list[dict]:
    """
    Embed the question and search ChromaDB for the most relevant deliveries.

    Args:
        question: Natural language question from the user.
        n_results: Number of top results to return.
        where: Optional ChromaDB metadata filter (e.g., {"event": "wicket"}).

    Returns:
        List of result dicts, each with 'text', 'metadata', 'distance'.
    """
    # Embed the query using the same model used for indexing
    q_embedding = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]

    collection = get_collection()
    kwargs = dict(
        query_embeddings=[q_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Unpack ChromaDB's batch result format
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def collection_exists() -> bool:
    """Return True if the ChromaDB collection has been built and has records."""
    try:
        col = get_collection()
        return col.count() > 0
    except Exception:
        return False


def get_player_stats(player_name: str) -> dict | None:
    """
    Retrieve all deliveries involving this player to calculate aggregate match stats.
    Returns a dict with batting, bowling, and dismissal stats, or None if player not found.
    """
    try:
        col = get_collection()
        results = col.get(
            where={"$or": [
                {"batter": {"$eq": player_name}},
                {"bowler": {"$eq": player_name}},
                {"player_out": {"$eq": player_name}}
            ]},
            include=["metadatas"]
        )
        metas = results.get("metadatas") or []
        if not metas:
            return None

        bat_runs = 0
        bat_balls = 0
        bat_fours = 0
        bat_sixes = 0
        dismissal = None

        bowl_balls = 0
        bowl_runs = 0
        bowl_wickets = 0

        for m in metas:
            # Batting stats
            if m.get("batter") == player_name:
                # Approximate 1 ball faced (wides shouldn't count, but we lack exact data without rebuilding DB)
                bat_balls += 1
                bat_runs += m.get("runs_total", 0)  # slightly overcounts if extras occur on same ball
                
                if m.get("event") == "four":
                    bat_fours += 1
                elif m.get("event") == "six":
                    bat_sixes += 1
            
            # Dismissal
            if m.get("player_out") == player_name:
                w_kind = m.get("wicket_kind", "")
                w_bowler = m.get("bowler", "")
                w_fielder = m.get("wicket_fielder", "")
                over = m.get("over", "?")
                ball = m.get("ball", "?")
                dismissal = f"b {w_bowler}"
                if w_kind:
                    dismissal = f"{w_kind} {dismissal}"
                if w_fielder:
                    dismissal += f" c {w_fielder}"
                dismissal += f" (Over {over}.{ball})"
            
            # Bowling stats
            if m.get("bowler") == player_name:
                bowl_balls += 1
                bowl_runs += m.get("runs_total", 0)
                if m.get("player_out") and m.get("wicket_kind") not in ("run out", "retired hurt", "obstructing the field"):
                    bowl_wickets += 1
                    
        overs = bowl_balls // 6
        rem_balls = bowl_balls % 6
        
        return {
            "name": player_name,
            "batting": {
                "runs": bat_runs,
                "balls": bat_balls,
                "fours": bat_fours,
                "sixes": bat_sixes,
                "dismissal": dismissal
            },
            "bowling": {
                "overs": f"{overs}.{rem_balls}",
                "runs": bowl_runs,
                "wickets": bowl_wickets
            }
        }
    except Exception:
        return None


def get_event_leaderboard(where_filter: dict | None, event_type: str | None, group_by: str = "player", metric: str = "count") -> list[dict] | None:
    """
    Query ChromaDB for all deliveries matching the filter, and aggregate by the group_by field.
    Returns a sorted leaderboard.
    """
    try:
        col = get_collection()
        
        kwargs = {"include": ["metadatas"]}
        if where_filter:
            kwargs["where"] = where_filter
            
        results = col.get(**kwargs)
        metas = results.get("metadatas") or []
        if not metas:
            return None
            
        counts = {}
        for m in metas:
            if group_by == "over":
                innings_val = m.get("innings", "?")
                over_val = m.get("over", "?")
                key = f"{innings_val}_{over_val}"
            elif group_by == "innings":
                key = str(m.get("innings", "Unknown"))
            elif group_by == "wicket_kind":
                key = m.get("wicket_kind", "Unknown")
                if not key:
                    continue
            else:
                gb_field = "batter"
                if event_type in ("wicket", "dot"):
                    gb_field = "bowler"
                key = m.get(gb_field, "Unknown")

            if key not in counts:
                counts[key] = 0
            
            if metric == "runs_total":
                counts[key] += m.get("runs_total", 0)
            else:
                counts[key] += 1
            
        leaderboard = [{"player": p, "count": c} for p, c in counts.items()]
        leaderboard.sort(key=lambda x: x["count"], reverse=True)
        return leaderboard
    except Exception:
        return None


# Cache so we don't hit ChromaDB on every request
_known_players_cache: list[str] | None = None
_match_metadata_cache: dict | None = None


def get_match_metadata() -> dict | None:
    """
    Return match metadata, specifically determining the exact final (max) over 
    bowled in the final (max) innings of the match.
    Result is cached.
    """
    global _match_metadata_cache
    if _match_metadata_cache is not None:
        return _match_metadata_cache

    try:
        col = get_collection()
        # Fetch all metadata to find max innings and max over.
        results = col.get(include=["metadatas"])
        metas = results.get("metadatas") or []
        
        if not metas:
            return None

        max_innings = max((m.get("innings", 1) for m in metas), default=1)
        max_over = max((m.get("over", 0) for m in metas if m.get("innings") == max_innings), default=0)
        
        _match_metadata_cache = {
            "max_innings": max_innings,
            "max_over": max_over
        }
    except Exception:
        _match_metadata_cache = None

    return _match_metadata_cache


def get_known_players() -> list[str]:
    """
    Return all distinct player names (batters + bowlers + players_out) from
    the ChromaDB collection metadata. Result is cached after the first call.
    """
    global _known_players_cache
    if _known_players_cache is not None:
        return _known_players_cache

    try:
        col = get_collection()
        # ChromaDB doesn't support SELECT DISTINCT, so we fetch all metadata
        # and deduplicate. Limit to a safe upper bound (1-day T20 = ~250 deliveries).
        results  = col.get(include=["metadatas"], limit=500)
        names: set[str] = set()
        for m in (results.get("metadatas") or []):
            for field in ("batter", "bowler", "player_out", "wicket_fielder"):
                val = m.get(field, "")
                if val:
                    names.add(val)
        _known_players_cache = sorted(names)
    except Exception:
        _known_players_cache = []

    return _known_players_cache


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not collection_exists():
        print("No index found. Building from scratch...")
        from rag.load_match import load_match
        from rag.flatten_data import flatten_deliveries
        from rag.embedding_pipeline import generate_embeddings

        filepath = sys.argv[1] if len(sys.argv) > 1 else "data/IndVsWI.json"
        data = load_match(filepath)
        docs = flatten_deliveries(data)
        embeddings = generate_embeddings(docs)
        build_index(docs, embeddings)

    print("\nRunning sample queries:")
    queries = [
        "Who dismissed Shimron Hetmyer?",
        "What happened in the final over?",
        "Biggest six hit in the match",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        results = query(q, n_results=2)
        for r in results:
            m = r["metadata"]
            print(f"  [{m['innings']} inn | Over {m['over']}.{m['ball']}] "
                  f"{m['batter']} vs {m['bowler']} | Event: {m['event']}")
            print(f"  {r['text'][:120]}...")
