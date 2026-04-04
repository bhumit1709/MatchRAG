import pytest
from unittest.mock import patch, MagicMock

from rag.graph_nodes import classify_question
from rag.question_handlers.utils import (
    question_mentions_players,
    build_player_filter,
    format_delivery_header,
)

# ── Tests for utils.py ───────────────────────────────────────────────────

def test_question_mentions_players():
    known_players = ["Virat Kohli", "Jasprit Bumrah", "Rohit Sharma", "Hardik Pandya"]
    
    # Exact full name match
    assert "Virat Kohli" in question_mentions_players("how did Virat Kohli play?", known_players)
    
    # Last name match (>= 4 chars)
    assert "Rohit Sharma" in question_mentions_players("how many runs for Sharma?", known_players)
    assert "Jasprit Bumrah" in question_mentions_players("bumrah stats", known_players)
    
    # No false positive for short names without exact match
    known_players_short = ["Sky Yadav", "KL Rahul", "Surya"]
    # If the last name is "Rahul" (5 chars), it should match
    assert "KL Rahul" in question_mentions_players("did rahul bat well?", known_players_short)

def test_build_player_filter():
    player = "Virat Kohli"
    f = build_player_filter(player)
    assert "$or" in f
    assert len(f["$or"]) == 3
    assert {"batter": {"$eq": player}} in f["$or"]
    assert {"bowler": {"$eq": player}} in f["$or"]
    assert {"player_out": {"$eq": player}} in f["$or"]

def test_format_delivery_header():
    meta = {
        "innings": 1,
        "over": 19,
        "ball": 5,
        "batter": "Virat Kohli",
        "bowler": "Trent Boult",
        "event": "run",
        "runs_total": 4
    }
    header = format_delivery_header(meta, 1)
    assert "[1] Inn 1" in header
    assert "19.5" in header
    assert "Batter: Virat Kohli" in header
    assert "Event: RUN" in header
    assert "Runs: 4" in header

    meta_wicket = {
        "innings": 2, "over": 0, "ball": 3, "batter": "Rohit Sharma", "bowler": "Tim Southee", "event": "wicket", "player_out": "Rohit Sharma", "wicket_kind": "lbw"
    }
    header_w = format_delivery_header(meta_wicket, 2)
    assert "Event: WICKET" in header_w
    assert "Runs: " not in header_w
    assert "OUT: Rohit Sharma (lbw)" in header_w


# ── Tests for classify_question ───────────────────────────────────────────

@pytest.fixture
def mock_get_known_players():
    with patch("rag.graph_nodes.get_known_players", return_value=["Virat Kohli", "Jasprit Bumrah"]) as mock:
        yield mock

def test_classify_match_summary(mock_get_known_players):
    state = {"question": "what happened in the match?", "rewritten_question": "what happened in the match?"}
    res = classify_question(state)
    assert res["question_type"] == "match_summary"

def test_classify_comparison(mock_get_known_players):
    state = {"question": "Kohli vs Bumrah", "rewritten_question": "Kohli vs. Bumrah"}
    res = classify_question(state)
    assert res["question_type"] == "comparison"

def test_classify_comparison_requires_two_players(mock_get_known_players):
    # Only 1 player mentioned with 'vs' keyword -> should NOT be comparison
    state = {"question": "Kohli vs", "rewritten_question": "Kohli vs anyone"}
    res = classify_question(state)
    assert res["question_type"] != "comparison"
    # Actually, the fallback for "players >= 1 and not stat_like" usually returns general or player_performance
    # Since only 1 player, classification falls down to the player check at the end of classify_question (wait, classify_question only returns match_summary, over_summary, comparison. General is default in graph logic? No, classify_question sets types. Wait, if no type is set, does it set something?)

def test_classify_over_summary(mock_get_known_players):
    state = {"question": "middle overs analysis", "rewritten_question": "middle overs analysis"}
    res = classify_question(state)
    assert res["question_type"] == "over_summary"

    state2 = {"question": "what happened in over 12", "rewritten_question": "what happened in over 12"}
    res2 = classify_question(state2)
    assert res2["question_type"] == "over_summary"
