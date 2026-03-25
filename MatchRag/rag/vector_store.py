"""LangChain Chroma vector store plus deterministic cricket-stat helpers."""

from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, EMBED_MODEL_PATH
from rag.documents import document_id, records_to_documents, serialize_document
from rag.providers import get_embeddings


_vector_store: Chroma | None = None
_known_players_cache: list[str] | None = None
_match_metadata_cache: dict | None = None
_INDEX_METADATA_FILE = "index_metadata.json"


def _clear_caches() -> None:
    global _known_players_cache, _match_metadata_cache
    _known_players_cache = None
    _match_metadata_cache = None


def _index_metadata_path() -> Path:
    return Path(CHROMA_PATH) / _INDEX_METADATA_FILE


def _data_source_metadata(data_file: str | None) -> dict[str, str]:
    if not data_file:
        return {}

    path = Path(data_file)
    metadata = {"data_file": str(path.resolve(strict=False))}
    try:
        metadata["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        pass
    return metadata


def _current_index_metadata(data_file: str | None = None) -> dict[str, str]:
    metadata = {
        "collection_name": COLLECTION_NAME,
        "embed_model": EMBED_MODEL_PATH or EMBED_MODEL,
    }
    metadata.update(_data_source_metadata(data_file))
    return metadata


def read_index_metadata() -> dict | None:
    """Return the persisted index metadata, if present."""
    path = _index_metadata_path()
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def index_matches_runtime(data_file: str | None = None) -> bool:
    """Return True when the current Chroma index matches the active embedding config."""
    return read_index_metadata() == _current_index_metadata(data_file)


def _write_index_metadata(data_file: str | None = None) -> None:
    path = _index_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_current_index_metadata(data_file), indent=2, sort_keys=True))


def get_vector_store() -> Chroma:
    """Return the shared persistent Chroma vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vector_store


def build_index(
    documents: list,
    embeddings: list[list[float]] | None = None,
    reset: bool = False,
    source_file: str | None = None,
) -> None:
    """Build or refresh the Chroma index from records or LangChain documents."""
    global _vector_store
    vector_store = get_vector_store()

    if reset:
        try:
            vector_store.delete_collection()
        except Exception:
            pass
        _vector_store = None
        vector_store = get_vector_store()

    if documents and isinstance(documents[0], dict):
        documents = records_to_documents(documents)

    ids = [document_id(document) for document in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    _write_index_metadata(source_file)
    _clear_caches()


def query(question: str, n_results: int = 6, where: dict | None = None) -> list[dict]:
    """Search the vector store and return UI-friendly payloads."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(
        query=question,
        k=n_results,
        filter=where,
    )
    return [serialize_document(document, distance=distance) for document, distance in results]


def get_sequential_deliveries(
    where: dict | None = None,
    sort_direction: str = "asc",
    limit: int | None = 20,
) -> list[dict]:
    """Return metadata-matched deliveries in chronological order."""
    try:
        results = get_vector_store().get(where=where, include=["documents", "metadatas"])
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        deliveries = [
            {"text": text, "metadata": meta, "distance": 0.0}
            for text, meta in zip(docs, metas)
        ]
        deliveries.sort(
            key=lambda item: (
                item["metadata"].get("innings", 0),
                item["metadata"].get("over", 0),
                item["metadata"].get("ball", 0),
            ),
            reverse=(sort_direction == "desc"),
        )

        if limit:
            deliveries = deliveries[:limit]
        if sort_direction == "desc":
            deliveries.reverse()
        return deliveries
    except Exception:
        return []


def collection_exists() -> bool:
    """Return True when the Chroma collection already contains records."""
    try:
        results = get_vector_store().get(limit=1, include=["metadatas"])
        return bool(results.get("ids"))
    except Exception:
        return False


def get_player_stats(player_name: str) -> dict | None:
    """Calculate deterministic batting and bowling stats for one player."""
    try:
        results = get_vector_store().get(
            where={
                "$or": [
                    {"batter": {"$eq": player_name}},
                    {"bowler": {"$eq": player_name}},
                    {"player_out": {"$eq": player_name}},
                ]
            },
            include=["metadatas"],
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

        for meta in metas:
            if meta.get("batter") == player_name:
                bat_balls += 1
                bat_runs += meta.get("runs_total", 0)
                if meta.get("event") == "four":
                    bat_fours += 1
                elif meta.get("event") == "six":
                    bat_sixes += 1

            if meta.get("player_out") == player_name:
                wicket_kind = meta.get("wicket_kind", "")
                wicket_bowler = meta.get("bowler", "")
                wicket_fielder = meta.get("wicket_fielder", "")
                over = meta.get("over", "?")
                ball = meta.get("ball", "?")
                dismissal = f"b {wicket_bowler}"
                if wicket_kind:
                    dismissal = f"{wicket_kind} {dismissal}"
                if wicket_fielder:
                    dismissal += f" c {wicket_fielder}"
                dismissal += f" (Over {over}.{ball})"

            if meta.get("bowler") == player_name:
                bowl_balls += 1
                bowl_runs += meta.get("runs_total", 0)
                if meta.get("player_out") and meta.get("wicket_kind") not in {
                    "run out",
                    "retired hurt",
                    "obstructing the field",
                }:
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
                "dismissal": dismissal,
            },
            "bowling": {
                "overs": f"{overs}.{rem_balls}",
                "runs": bowl_runs,
                "wickets": bowl_wickets,
            },
        }
    except Exception:
        return None


def get_event_leaderboard(
    where_filter: dict | None,
    event_type: str | None,
    group_by: str = "player",
    metric: str = "count",
) -> list[dict] | None:
    """Aggregate a deterministic leaderboard from Chroma metadata."""
    try:
        results = get_vector_store().get(where=where_filter, include=["metadatas"])
        metas = results.get("metadatas") or []
        if not metas:
            return None

        counts: dict[str, float] = {}
        impact_stats: dict[str, dict[str, float]] = {}

        for meta in metas:
            if group_by == "over":
                key = f"{meta.get('innings', '?')}_{meta.get('over', '?')}"
            elif group_by == "innings":
                key = str(meta.get("innings", "Unknown"))
            elif group_by == "wicket_kind":
                key = meta.get("wicket_kind", "Unknown")
                if not key:
                    continue
            else:
                field_name = "batter"
                if event_type in {"wicket", "dot"}:
                    field_name = "bowler"
                key = meta.get(field_name, "Unknown")

            counts.setdefault(key, 0)

            if metric == "runs_total":
                counts[key] += meta.get("runs_total", 0)
            elif metric == "impact":
                batter = meta.get("batter")
                bowler = meta.get("bowler")
                if batter:
                    impact_stats.setdefault(
                        batter,
                        {
                            "bat_runs": 0,
                            "bat_balls": 0,
                            "bat_boundaries": 0,
                            "bowl_wickets": 0,
                            "bowl_runs": 0,
                            "bowl_balls": 0,
                        },
                    )
                    impact_stats[batter]["bat_balls"] += 1
                    impact_stats[batter]["bat_runs"] += meta.get("runs_total", 0)
                    if meta.get("event") in {"four", "six"}:
                        impact_stats[batter]["bat_boundaries"] += 1
                if bowler:
                    impact_stats.setdefault(
                        bowler,
                        {
                            "bat_runs": 0,
                            "bat_balls": 0,
                            "bat_boundaries": 0,
                            "bowl_wickets": 0,
                            "bowl_runs": 0,
                            "bowl_balls": 0,
                        },
                    )
                    impact_stats[bowler]["bowl_balls"] += 1
                    impact_stats[bowler]["bowl_runs"] += meta.get("runs_total", 0)
                    if meta.get("player_out") and meta.get("wicket_kind") not in {
                        "run out",
                        "retired hurt",
                        "obstructing the field",
                    }:
                        impact_stats[bowler]["bowl_wickets"] += 1
            else:
                counts[key] += 1

        if metric == "impact":
            for player, stats in impact_stats.items():
                score = stats["bat_runs"] + stats["bat_boundaries"] + (stats["bowl_wickets"] * 25)
                if stats["bat_balls"] > 10:
                    strike_rate = (stats["bat_runs"] / stats["bat_balls"]) * 100
                    if strike_rate > 150:
                        score += 10
                    elif strike_rate < 120:
                        score -= 5
                if stats["bowl_balls"] > 12:
                    overs = stats["bowl_balls"] / 6
                    economy = stats["bowl_runs"] / overs
                    if economy <= 6.0:
                        score += 15
                    elif economy <= 8.0:
                        score += 5
                    elif economy > 10.0:
                        score -= 10
                counts[player] = round(score, 1)

        leaderboard = [{"player": player, "count": count} for player, count in counts.items()]
        leaderboard.sort(key=lambda row: row["count"], reverse=True)
        return leaderboard
    except Exception:
        return None


def get_phase_stats(
    phase: str,
    innings: int | None = None,
    event_type: str | None = None,
) -> dict | None:
    """
    Deterministic aggregate stats for a T20 match phase.

    Args:
        phase:      'powerplay', 'middle', or 'death'
        innings:    Optional innings filter (1 or 2). None = both innings.
        event_type: Optional event filter (e.g. 'wicket', 'six'). None = all events.

    Returns a dict with:
        phase, innings (or 'both'), runs, wickets, sixes, fours, dots, balls,
        run_rate, top_batters (list), top_bowlers (list)
    """
    try:
        where_parts: list[dict] = [{"phase": {"$eq": phase}}]
        if innings is not None:
            where_parts.append({"innings": {"$eq": innings}})
        if event_type is not None:
            where_parts.append({"event": {"$eq": event_type}})

        where = {"$and": where_parts} if len(where_parts) > 1 else where_parts[0]

        results = get_vector_store().get(where=where, include=["metadatas"])
        metas = results.get("metadatas") or []
        if not metas:
            return None

        runs = 0
        wickets = 0
        sixes = 0
        fours = 0
        dots = 0
        balls = 0
        batter_runs: dict[str, int] = {}
        bowler_wickets: dict[str, int] = {}
        bowler_runs: dict[str, int] = {}
        bowler_balls: dict[str, int] = {}
        non_wicket_kinds = {"run out", "retired hurt", "obstructing the field"}

        for meta in metas:
            runs += meta.get("runs_total", 0)
            balls += 1
            event = meta.get("event", "")

            if event == "wicket":
                wickets += 1
            elif event == "six":
                sixes += 1
            elif event == "four":
                fours += 1
            elif event == "dot":
                dots += 1

            batter = meta.get("batter", "")
            if batter:
                batter_runs[batter] = batter_runs.get(batter, 0) + meta.get("runs_total", 0)

            bowler = meta.get("bowler", "")
            if bowler:
                bowler_balls[bowler] = bowler_balls.get(bowler, 0) + 1
                bowler_runs[bowler] = bowler_runs.get(bowler, 0) + meta.get("runs_total", 0)
                if meta.get("player_out") and meta.get("wicket_kind") not in non_wicket_kinds:
                    bowler_wickets[bowler] = bowler_wickets.get(bowler, 0) + 1

        run_rate = round((runs / balls) * 6, 2) if balls else 0.0

        top_batters = sorted(
            [{"player": p, "runs": r} for p, r in batter_runs.items()],
            key=lambda x: x["runs"],
            reverse=True,
        )[:5]

        top_bowlers = sorted(
            [
                {
                    "player": p,
                    "wickets": bowler_wickets.get(p, 0),
                    "runs": bowler_runs.get(p, 0),
                    "balls": bowler_balls.get(p, 0),
                }
                for p in bowler_balls
            ],
            key=lambda x: (-x["wickets"], x["runs"]),
        )[:5]

        return {
            "phase": phase,
            "innings": innings if innings is not None else "both",
            "runs": runs,
            "wickets": wickets,
            "sixes": sixes,
            "fours": fours,
            "dots": dots,
            "balls": balls,
            "run_rate": run_rate,
            "top_batters": top_batters,
            "top_bowlers": top_bowlers,
        }
    except Exception:
        return None


def format_phase_stats_block(stats: dict) -> str:
    """
    Format a get_phase_stats() result into the === SYSTEM CALCULATED === block
    injected into the LLM prompt context.
    """
    phase_label = stats["phase"].replace("_", " ").title()
    innings_label = f"Innings {stats['innings']}" if stats["innings"] != "both" else "both innings"

    batters_str = ", ".join(
        f"{b['player']} ({b['runs']} runs)" for b in stats["top_batters"]
    ) or "—"
    bowlers_str = ", ".join(
        f"{b['player']} ({b['wickets']}w/{b['runs']}r in {b['balls']//6}.{b['balls']%6} ov)"
        for b in stats["top_bowlers"]
    ) or "—"

    return (
        f"=== SYSTEM CALCULATED EXACT STATS ===\n"
        f"Phase: {phase_label} | {innings_label}\n"
        f"Runs: {stats['runs']} | Wickets: {stats['wickets']} | "
        f"Run Rate: {stats['run_rate']} | Balls: {stats['balls']}\n"
        f"Sixes: {stats['sixes']} | Fours: {stats['fours']} | Dots: {stats['dots']}\n"
        f"Top batters: {batters_str}\n"
        f"Top bowlers: {bowlers_str}\n"
        f"=====================================\n"
    )


def get_match_metadata() -> dict | None:
    """Return cached match metadata derived from the indexed deliveries."""
    global _match_metadata_cache
    if _match_metadata_cache is not None:
        return _match_metadata_cache

    try:
        results = get_vector_store().get(include=["metadatas"])
        metas = results.get("metadatas") or []
        if not metas:
            return None

        max_innings = max((meta.get("innings", 1) for meta in metas), default=1)
        max_over = max(
            (meta.get("over", 0) for meta in metas if meta.get("innings") == max_innings),
            default=0,
        )
        _match_metadata_cache = {"max_innings": max_innings, "max_over": max_over}
    except Exception:
        _match_metadata_cache = None

    return _match_metadata_cache


def get_known_players() -> list[str]:
    """Return all distinct players from the indexed metadata."""
    global _known_players_cache
    if _known_players_cache is not None:
        return _known_players_cache

    try:
        results = get_vector_store().get(include=["metadatas"])
        names: set[str] = set()
        for meta in results.get("metadatas") or []:
            for field in ("batter", "bowler", "player_out", "wicket_fielder"):
                value = meta.get(field, "")
                if value:
                    names.add(value)
        _known_players_cache = sorted(names)
    except Exception:
        _known_players_cache = []

    return _known_players_cache


if __name__ == "__main__":
    if not collection_exists():
        print("No index found. Building from scratch...")
        from rag.flatten_data import flatten_deliveries
        from rag.load_match import load_match

        filepath = sys.argv[1] if len(sys.argv) > 1 else "data/IndVsNZ.json"
        data = load_match(filepath)
        docs = flatten_deliveries(data)
        build_index(docs, reset=True)

    print("\nRunning sample queries:")
    for question in [
        "Who dismissed Abhishek Sharma?",
        "What happened in the final over?",
        "Biggest six hit in the match",
    ]:
        print(f"\nQ: {question}")
        for result in query(question, n_results=2):
            meta = result["metadata"]
            print(
                f"  [{meta['innings']} inn | Over {meta['over']}.{meta['ball']}] "
                f"{meta['batter']} vs {meta['bowler']} | Event: {meta['event']}"
            )
            print(f"  {result['text'][:120]}...")
