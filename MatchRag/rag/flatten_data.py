"""
flatten_data.py
---------------
Steps 2, 3 & 4 of the RAG pipeline.

Flattens the nested innings → overs → deliveries structure into a list
of flat dicts. Applies rule-based event detection and builds a rich
natural-language 'text' field for embedding.
"""

import re
import sys
from rag.load_match import load_match, extract_metadata


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

_PHASE_BOUNDARIES = {
    "powerplay": range(0, 6),   # Overs 0-5  (1st-6th over in broadcast notation)
    "middle":    range(6, 15),  # Overs 6-14 (7th-15th over)
    "death":     range(15, 20), # Overs 15-19 (16th-20th over)
}


def classify_phase(over: int) -> str:
    """
    Classify a CricSheet over number (0-indexed) into a T20 match phase.

    Returns:
        'powerplay' for overs 0-5 (broadcast: 1-6)
        'middle'    for overs 6-14 (broadcast: 7-15)
        'death'     for overs 15-19 (broadcast: 16-20)
    """
    if over <= 5:
        return "powerplay"
    if over <= 14:
        return "middle"
    return "death"


# ---------------------------------------------------------------------------
# HTML tag stripping
# ---------------------------------------------------------------------------
def strip_html(text: str) -> str:
    """Remove HTML tags from commentary strings."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_event(delivery: dict) -> str:
    """
    Classify a delivery into one of:  wicket | six | four | dot | single | run

    Rules applied in priority order:
      1. wicket  → delivery has a 'wickets' list
      2. six     → runs.batter == 6
      3. four    → runs.batter == 4
      4. dot     → runs.total  == 0
      5. single  → runs.total  == 1
      6. run     → everything else (2s, 3s, wides+runs, etc.)
    """
    runs = delivery.get("runs", {})

    if delivery.get("wickets"):
        return "wicket"
    if runs.get("batter") == 6:
        return "six"
    if runs.get("batter") == 4:
        return "four"
    if runs.get("total", 0) == 0:
        return "dot"
    if runs.get("total") == 1:
        return "single"
    return "run"


# ---------------------------------------------------------------------------
# Embedding text builder
# ---------------------------------------------------------------------------

def build_text(record: dict) -> str:
    """
    Build a descriptive natural-language string suitable for embedding.

    Example:
      "Match: India vs New Zealand at Eden Gardens, Kolkata. Innings 1.
       Over 3.2. Abhishek Sharma facing MJ Santner. Event: four. Runs: 4.
       Commentary: drives through cover."
    """
    wicket_line = ""
    if record["event"] == "wicket":
        kind = record.get("wicket_kind", "")
        player = record.get("player_out", "")
        fielder = record.get("wicket_fielder", "")
        wicket_line = f"WICKET: {player} dismissed"
        if kind:
            wicket_line += f" ({kind})"
        if fielder:
            wicket_line += f", fielder: {fielder}"
        wicket_line += ". "

    return (
        f"Match: {record['match']} at {record['venue']}. "
        f"Innings {record['innings']} ({record['batting_team']}). "
        f"Phase: {record['phase']}. "
        f"Over {record['over']}.{record['ball']}. "
        f"{record['batter']} facing {record['bowler']}. "
        f"Event: {record['event']}. Runs scored: {record['runs_total']}. "
        f"{wicket_line}"
        f"Commentary: {record['commentary']}"
    ).strip()


# ---------------------------------------------------------------------------
# Main flattening function
# ---------------------------------------------------------------------------

def flatten_deliveries(data: dict) -> list[dict]:
    """
    Convert the nested match JSON into a flat list of delivery records.

    Each record contains all fields needed for embedding and metadata storage.

    Args:
        data: Parsed match JSON dict (as returned by load_match).

    Returns:
        List of flat delivery dicts with 'text' field ready for embedding.
    """
    meta = extract_metadata(data)
    records = []

    for innings_idx, innings in enumerate(data.get("innings", []), start=1):
        batting_team = innings.get("team", f"Team {innings_idx}")

        for over_data in innings.get("overs", []):
            # Over is 0-indexed in CricSheet format
            over_num = over_data.get("over", 0)

            for ball_idx, delivery in enumerate(over_data.get("deliveries", []), start=1):
                runs = delivery.get("runs", {})
                event = detect_event(delivery)
                commentary = strip_html(delivery.get("commentary", ""))

                # Wicket details
                wickets = delivery.get("wickets", [])
                wicket_info = wickets[0] if wickets else {}
                player_out = wicket_info.get("player_out", "")
                wicket_kind = wicket_info.get("kind", "")
                # Fielder can be a list in CricSheet
                fielders = wicket_info.get("fielders", [])
                wicket_fielder = fielders[0].get("name", "") if fielders else ""

                record = {
                    # Match context
                    "match": meta["match"],
                    "venue": meta["venue"],
                    "season": meta["season"],
                    "event_name": meta["event_name"],
                    # Innings context
                    "innings": innings_idx,
                    "batting_team": batting_team,
                    # Delivery position
                    "over": over_num,
                    "ball": ball_idx,
                    # Match phase (derived from over number)
                    "phase": classify_phase(over_num),
                    # Players
                    "batter": delivery.get("batter", ""),
                    "bowler": delivery.get("bowler", ""),
                    "non_striker": delivery.get("non_striker", ""),
                    # Runs
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    # Event
                    "event": event,
                    # Wicket
                    "player_out": player_out,
                    "wicket_kind": wicket_kind,
                    "wicket_fielder": wicket_fielder,
                    # Raw commentary
                    "commentary": commentary,
                }

                # Build the embedding text last (needs all fields populated)
                record["text"] = build_text(record)

                # Unique ID for ChromaDB upserts
                record["id"] = f"inn{innings_idx}_ov{over_num}_b{ball_idx}"

                records.append(record)

    return records


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/IndVsNZ.json"
    data = load_match(filepath)
    docs = flatten_deliveries(data)

    print(f"Flattened {len(docs)} deliveries.\n")

    # Show event breakdown
    from collections import Counter
    events = Counter(d["event"] for d in docs)
    print("Event breakdown:")
    for evt, count in sorted(events.items()):
        print(f"  {evt:8s}: {count}")

    # Sample records
    print("\n--- Sample: first delivery ---")
    print(json.dumps({k: v for k, v in docs[0].items() if k != "text"}, indent=2))
    print("\nEmbedding text:")
    print(docs[0]["text"])

    print("\n--- Sample: first wicket ---")
    wicket = next((d for d in docs if d["event"] == "wicket"), None)
    if wicket:
        print(json.dumps({k: v for k, v in wicket.items() if k != "text"}, indent=2))
        print("\nEmbedding text:")
        print(wicket["text"])
