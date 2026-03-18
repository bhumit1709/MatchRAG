"""
load_match.py
-------------
Step 1 of the RAG pipeline.

Loads a CricSheet-format JSON file and returns the parsed dict
along with a convenience metadata object for downstream modules.
"""

import json
import sys
from pathlib import Path


def load_match(filepath: str = "data/IndVsNZ.json") -> dict:
    """
    Load and validate a match JSON file.

    Args:
        filepath: Path to the CricSheet-format JSON match file.

    Returns:
        Parsed dict of the entire match JSON.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is missing required keys.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Match file not found: {filepath}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic validation
    required_keys = {"info", "innings"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Match JSON is missing required keys: {missing}")

    return data


def extract_metadata(data: dict) -> dict:
    """
    Pull top-level match metadata out for easy access across the pipeline.

    Args:
        data: Parsed match JSON dict.

    Returns:
        Flat metadata dict with match-level information.
    """
    info = data["info"]
    teams = info.get("teams", [])

    return {
        "match": " vs ".join(teams) if len(teams) == 2 else "Unknown Match",
        "venue": info.get("venue", "Unknown Venue"),
        "season": info.get("season", "Unknown Season"),
        "city": info.get("city", ""),
        "match_type": info.get("match_type", ""),
        "event_name": info.get("event", {}).get("name", ""),
        "match_number": info.get("event", {}).get("match_number", ""),
        "date": info.get("dates", [""])[0],
        "winner": info.get("outcome", {}).get("winner", ""),
        "player_of_match": info.get("player_of_match", []),
        "teams": teams,
    }


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/IndVsNZ.json"
    data = load_match(filepath)
    meta = extract_metadata(data)
    print("Match loaded successfully.")
    print(f"  Match  : {meta['match']}")
    print(f"  Venue  : {meta['venue']}")
    print(f"  Season : {meta['season']}")
    print(f"  Winner : {meta['winner']}")
    print(f"  Innings: {len(data['innings'])}")
