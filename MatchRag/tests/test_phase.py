"""
tests/test_phase.py
-------------------
Tests for the phase analytics feature:
  - Phase classification (classify_phase)
  - Delivery flattening includes phase field
  - Embedding text includes phase
  - Fast-path plan phase detection
  - Where-filter phase clause
  - compute_aggregate_stats phase path
  - format_phase_stats_block formatter
"""

import pytest

# ---------------------------------------------------------------------------
# Layer 1: classify_phase
# ---------------------------------------------------------------------------

from rag.flatten_data import classify_phase, build_text


class TestClassifyPhase:
    """Over number → phase string."""

    def test_over_0_is_powerplay(self):
        assert classify_phase(0) == "powerplay"

    def test_over_5_is_powerplay(self):
        assert classify_phase(5) == "powerplay"

    def test_over_6_is_middle(self):
        assert classify_phase(6) == "middle"

    def test_over_14_is_middle(self):
        assert classify_phase(14) == "middle"

    def test_over_15_is_death(self):
        assert classify_phase(15) == "death"

    def test_over_19_is_death(self):
        assert classify_phase(19) == "death"

    def test_all_phases_are_valid_strings(self):
        valid = {"powerplay", "middle", "death"}
        for over in range(20):
            assert classify_phase(over) in valid


class TestFlattenIncludesPhase:
    """flatten_deliveries() inserts 'phase' into every record."""

    @pytest.fixture
    def sample_data(self):
        return {
            "info": {
                "teams": ["India", "New Zealand"],
                "venue": "Test Ground",
                "season": "2024",
                "event": {"name": "T20 WC"},
            },
            "innings": [
                {
                    "team": "India",
                    "overs": [
                        {
                            "over": 0,
                            "deliveries": [
                                {
                                    "batter": "Rohit Sharma",
                                    "bowler": "T Boult",
                                    "non_striker": "V Kohli",
                                    "runs": {"batter": 4, "extras": 0, "total": 4},
                                }
                            ],
                        },
                        {
                            "over": 10,
                            "deliveries": [
                                {
                                    "batter": "SK Samson",
                                    "bowler": "L Ngidi",
                                    "non_striker": "HH Pandya",
                                    "runs": {"batter": 1, "extras": 0, "total": 1},
                                }
                            ],
                        },
                        {
                            "over": 17,
                            "deliveries": [
                                {
                                    "batter": "HH Pandya",
                                    "bowler": "T Boult",
                                    "non_striker": "SK Samson",
                                    "runs": {"batter": 6, "extras": 0, "total": 6},
                                }
                            ],
                        },
                    ],
                }
            ],
        }

    def test_every_record_has_phase(self, sample_data):
        from rag.flatten_data import flatten_deliveries
        records = flatten_deliveries(sample_data)
        for rec in records:
            assert "phase" in rec, f"Record missing 'phase': {rec}"

    def test_over_0_record_is_powerplay(self, sample_data):
        from rag.flatten_data import flatten_deliveries
        records = flatten_deliveries(sample_data)
        over_0 = next(r for r in records if r["over"] == 0)
        assert over_0["phase"] == "powerplay"

    def test_over_10_record_is_middle(self, sample_data):
        from rag.flatten_data import flatten_deliveries
        records = flatten_deliveries(sample_data)
        over_10 = next(r for r in records if r["over"] == 10)
        assert over_10["phase"] == "middle"

    def test_over_17_record_is_death(self, sample_data):
        from rag.flatten_data import flatten_deliveries
        records = flatten_deliveries(sample_data)
        over_17 = next(r for r in records if r["over"] == 17)
        assert over_17["phase"] == "death"

    def test_phase_in_embedding_text(self, sample_data):
        from rag.flatten_data import flatten_deliveries
        records = flatten_deliveries(sample_data)
        for rec in records:
            phase_tag = f"Phase: {rec['phase']}"
            assert phase_tag in rec["text"], (
                f"embedding text missing '{phase_tag}': {rec['text'][:120]}"
            )


# ---------------------------------------------------------------------------
# Layer 2: documents.py — phase in METADATA_FIELDS
# ---------------------------------------------------------------------------

def test_phase_in_metadata_fields():
    from rag.documents import METADATA_FIELDS
    assert "phase" in METADATA_FIELDS


# ---------------------------------------------------------------------------
# Layer 3: schemas.py — RetrievalPlan accepts phase
# ---------------------------------------------------------------------------

class TestRetrievalPlanPhase:
    def test_phase_defaults_to_none(self):
        from rag.schemas import RetrievalPlan
        plan = RetrievalPlan(normalized_question="test")
        assert plan.phase is None

    def test_phase_powerplay(self):
        from rag.schemas import RetrievalPlan
        plan = RetrievalPlan(normalized_question="test", phase="powerplay")
        assert plan.phase == "powerplay"

    def test_phase_middle(self):
        from rag.schemas import RetrievalPlan
        plan = RetrievalPlan(normalized_question="test", phase="middle")
        assert plan.phase == "middle"

    def test_phase_death(self):
        from rag.schemas import RetrievalPlan
        plan = RetrievalPlan(normalized_question="test", phase="death")
        assert plan.phase == "death"


# ---------------------------------------------------------------------------
# Layer 5: fast-path phase detection
# ---------------------------------------------------------------------------

class TestFastPathPhaseDetection:
    """_build_fast_path_plan detects phase keywords and routes correctly."""

    def _build(self, question: str):
        from rag.graph_nodes import _build_fast_path_plan
        return _build_fast_path_plan(question)

    # ── powerplay ──
    def test_detects_powerplay(self):
        plan = self._build("How many runs in the powerplay?")
        assert plan is not None
        assert plan.phase == "powerplay"

    def test_detects_power_play_two_words(self):
        plan = self._build("What was the power play total?")
        assert plan is not None
        assert plan.phase == "powerplay"

    def test_powerplay_totals_routed_aggregate(self):
        plan = self._build("How many runs did India score in the powerplay?")
        assert plan.answer_strategy == "aggregate"

    def test_powerplay_performance_routed_hybrid(self):
        plan = self._build("Who was best in the powerplay?")
        assert plan.answer_strategy == "hybrid"

    def test_powerplay_narrative_routed_sequential(self):
        plan = self._build("What happened in the powerplay?")
        assert plan.answer_strategy == "sequential"

    # ── middle ──
    def test_detects_middle_overs(self):
        plan = self._build("How many wickets fell in the middle overs?")
        assert plan is not None
        assert plan.phase == "middle"

    def test_middle_overs_aggregate_strategy(self):
        plan = self._build("How many wickets fell in the middle overs?")
        assert plan.answer_strategy == "aggregate"

    # ── death ──
    def test_detects_death_overs(self):
        plan = self._build("Who bowled best in the death overs?")
        assert plan is not None
        assert plan.phase == "death"

    def test_death_overs_hybrid_strategy(self):
        plan = self._build("Who bowled best in the death overs?")
        assert plan.answer_strategy == "hybrid"

    def test_detects_final_overs(self):
        plan = self._build("What happened in the final overs?")
        assert plan is not None
        assert plan.phase == "death"

    # ── non-phase questions unaffected ──
    def test_non_phase_question_returns_none_or_other_plan(self):
        # "most sixes" question should NOT be routed as phase
        plan = self._build("Who hit the most sixes?")
        assert plan is None or plan.phase is None


# ---------------------------------------------------------------------------
# Layer 5: where-filter phase clause
# ---------------------------------------------------------------------------

class TestWhereFilterPhase:
    def _filter(self, **kwargs):
        from rag.schemas import RetrievalPlan
        from rag.graph_nodes import _build_where_filter
        plan = RetrievalPlan(normalized_question="test", **kwargs)
        return _build_where_filter(plan)

    def test_phase_only_filter(self):
        f = self._filter(phase="powerplay", answer_strategy="aggregate")
        assert f == {"phase": {"$eq": "powerplay"}}

    def test_phase_and_innings_filter(self):
        f = self._filter(phase="death", innings=2, answer_strategy="aggregate")
        assert f is not None
        # must include both clauses in $and
        assert "$and" in f
        clauses = f["$and"]
        assert {"phase": {"$eq": "death"}} in clauses
        assert {"innings": {"$eq": 2}} in clauses

    def test_no_phase_no_clause(self):
        f = self._filter(answer_strategy="semantic")
        assert f is None  # no filters at all


# ---------------------------------------------------------------------------
# Layer 4: format_phase_stats_block
# ---------------------------------------------------------------------------

class TestFormatPhaseStatsBlock:
    def test_formats_correctly(self):
        from rag.vector_store import format_phase_stats_block
        stats = {
            "phase": "powerplay",
            "innings": 1,
            "runs": 52,
            "wickets": 0,
            "sixes": 2,
            "fours": 5,
            "dots": 10,
            "balls": 36,
            "run_rate": 8.67,
            "top_batters": [{"player": "Rohit Sharma", "runs": 30}],
            "top_bowlers": [{"player": "T Boult", "wickets": 0, "runs": 20, "balls": 12}],
        }
        block = format_phase_stats_block(stats)
        assert "Phase: Powerplay" in block
        assert "Innings 1" in block
        assert "Runs: 52" in block
        assert "Wickets: 0" in block
        assert "Run Rate: 8.67" in block
        assert "Rohit Sharma" in block
        assert "T Boult" in block

    def test_both_innings_label(self):
        from rag.vector_store import format_phase_stats_block
        stats = {
            "phase": "death",
            "innings": "both",
            "runs": 80, "wickets": 4, "sixes": 6, "fours": 3,
            "dots": 5, "balls": 30, "run_rate": 16.0,
            "top_batters": [], "top_bowlers": [],
        }
        block = format_phase_stats_block(stats)
        assert "both innings" in block
        assert "Phase: Death" in block
