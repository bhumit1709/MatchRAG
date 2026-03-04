"""
tests/test_pipeline.py
----------------------
Unit tests for the MatchRAG data pipeline.

Tests load_match, flatten_deliveries, detect_event, and build_text
using the included data/IndVsWI.json fixture.

Run:
    pytest tests/test_pipeline.py -v
"""

import pytest
from pathlib import Path

# ── Fixture path ─────────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent.parent / "data" / "IndVsWI.json"


# ── load_match ───────────────────────────────────────────────────────────────

class TestLoadMatch:
    def test_loads_successfully(self):
        from rag.load_match import load_match
        data = load_match(str(DATA_FILE))
        assert isinstance(data, dict)

    def test_has_required_keys(self):
        from rag.load_match import load_match
        data = load_match(str(DATA_FILE))
        assert "info" in data
        assert "innings" in data

    def test_has_innings_data(self):
        from rag.load_match import load_match
        data = load_match(str(DATA_FILE))
        assert len(data["innings"]) >= 1

    def test_raises_on_missing_file(self):
        from rag.load_match import load_match
        with pytest.raises(FileNotFoundError):
            load_match("nonexistent_file.json")

    def test_extract_metadata_returns_teams(self):
        from rag.load_match import load_match, extract_metadata
        data = load_match(str(DATA_FILE))
        meta = extract_metadata(data)
        assert "match" in meta
        assert "venue" in meta
        assert meta["match"] != ""


# ── flatten_deliveries ───────────────────────────────────────────────────────

class TestFlattenDeliveries:
    @pytest.fixture(scope="class")
    def docs(self):
        from rag.load_match import load_match
        from rag.flatten_data import flatten_deliveries
        data = load_match(str(DATA_FILE))
        return flatten_deliveries(data)

    def test_returns_non_empty_list(self, docs):
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_each_doc_has_required_fields(self, docs):
        required = {"id", "text", "match", "innings", "over", "ball",
                    "batter", "bowler", "event", "runs_total"}
        for doc in docs[:10]:  # spot-check first 10
            assert required.issubset(doc.keys()), f"Missing keys in: {doc.keys()}"

    def test_ids_are_unique(self, docs):
        ids = [d["id"] for d in docs]
        assert len(ids) == len(set(ids)), "Duplicate delivery IDs found"

    def test_text_field_is_non_empty(self, docs):
        for doc in docs[:10]:
            assert isinstance(doc["text"], str)
            assert len(doc["text"]) > 0

    def test_events_are_valid(self, docs):
        valid_events = {"wicket", "six", "four", "dot", "single", "run"}
        for doc in docs[:50]:
            assert doc["event"] in valid_events


# ── detect_event ─────────────────────────────────────────────────────────────

class TestDetectEvent:
    def _delivery(self, runs_batter=0, runs_total=0, has_wicket=False):
        d = {"runs": {"batter": runs_batter, "extras": 0, "total": runs_total}}
        if has_wicket:
            d["wickets"] = [{"player_out": "Batsman", "kind": "caught"}]
        return d

    def test_wicket(self):
        from rag.flatten_data import detect_event
        assert detect_event(self._delivery(has_wicket=True)) == "wicket"

    def test_six(self):
        from rag.flatten_data import detect_event
        assert detect_event(self._delivery(runs_batter=6, runs_total=6)) == "six"

    def test_four(self):
        from rag.flatten_data import detect_event
        assert detect_event(self._delivery(runs_batter=4, runs_total=4)) == "four"

    def test_dot(self):
        from rag.flatten_data import detect_event
        assert detect_event(self._delivery(runs_batter=0, runs_total=0)) == "dot"

    def test_single(self):
        from rag.flatten_data import detect_event
        assert detect_event(self._delivery(runs_batter=1, runs_total=1)) == "single"

    def test_run_two(self):
        from rag.flatten_data import detect_event
        assert detect_event(self._delivery(runs_batter=2, runs_total=2)) == "run"

    def test_wicket_takes_priority_over_runs(self):
        from rag.flatten_data import detect_event
        # Even if batter scored 6, wicket should win
        assert detect_event(self._delivery(runs_batter=6, runs_total=6, has_wicket=True)) == "wicket"
