import re
import difflib

def format_delivery_header(meta: dict, index: int) -> str:
    """Format the standard header for a matched delivery."""
    header = (
        f"[{index}] Inn {meta.get('innings', '?')} | "
        f"{meta.get('over', '?')}.{meta.get('ball', '?')} | "
        f"Batter: {meta.get('batter', '?')} | "
        f"Bowler: {meta.get('bowler', '?')} | "
        f"Event: {str(meta.get('event', '?')).upper()}"
    )
    if meta.get("event") != "wicket":
        header += f" | Runs: {meta.get('runs_total', '?')}"
    if meta.get("player_out"):
        header += (
            f" | OUT: {meta['player_out']} ({meta.get('wicket_kind', '')})"
        )
    return header

def build_player_filter(player_name: str) -> dict:
    """Build a ChromaDB where-filter for a specific player."""
    return {
        "$or": [
            {"batter": {"$eq": player_name}},
            {"bowler": {"$eq": player_name}},
            {"player_out": {"$eq": player_name}},
        ]
    }

def question_mentions_players(question: str, known_players: list[str]) -> list[str]:
    """Identify players mentioned in the question."""
    question_lower = question.lower()
    matches: list[str] = []

    for player in known_players:
        player_lower = player.lower()
        if player_lower in question_lower:
            matches.append(player)
            continue

        last_name = player_lower.split()[-1]
        if len(last_name) >= 4 and re.search(rf"\b{re.escape(last_name)}\b", question_lower):
            matches.append(player)

    deduped: list[str] = []
    for player in matches:
        if player not in deduped:
            deduped.append(player)
    return deduped

# Maps phase surface forms (lowercase) to canonical phase value stored in ChromaDB.
PHASE_KEYWORDS: dict[str, str] = {
    "powerplay":    "powerplay",
    "power play":   "powerplay",
    "power-play":   "powerplay",
    "pp":           "powerplay",
    "first 6":      "powerplay",
    "first six":    "powerplay",
    "middle overs": "middle",
    "middle phase": "middle",
    "middle over":  "middle",
    "death overs":  "death",
    "death over":   "death",
    "death phase":  "death",
    "last 5":       "death",
    "last five":    "death",
    "final overs":  "death",
}
