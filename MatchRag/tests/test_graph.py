from rag.schemas import RetrievalPlan


def test_rewrite_question_rewrites_long_pronoun_follow_up(monkeypatch):
    from rag.graph_nodes import rewrite_question

    monkeypatch.setattr(
        "rag.graph_nodes.rewrite_followup_question",
        lambda question, history: ("How was SV Samson's overall performance in the match?", {"node": "rewrite_question"}),
    )

    state = {
        "question": "How was his overall performance in the match?",
        "rewritten_question": "",
        "query_variants": [],
        "chat_history": [
            {"role": "user", "content": "Who hit the most sixes?"},
            {"role": "assistant", "content": "SV Samson hit the most sixes."},
        ],
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "semantic",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "stage_timings_ms": {},
        "llm_traces": [],
    }

    updated = rewrite_question(state)

    assert updated["rewritten_question"] == "How was SV Samson's overall performance in the match?"
    assert updated["llm_traces"][0]["node"] == "rewrite_question"


def test_plan_retrieval_builds_filters_and_player_stats(monkeypatch):
    from rag.graph_nodes import plan_retrieval

    monkeypatch.setattr("rag.graph_nodes.get_known_players", lambda: ["JJ Bumrah", "Abhishek Sharma"])
    monkeypatch.setattr(
        "rag.graph_nodes.build_retrieval_plan",
        lambda question, known: (
            RetrievalPlan(
                normalized_question=question,
                players=["Bumrah"],
                event="wicket",
                innings=2,
            ),
            {"node": "plan_retrieval"},
        ),
    )
    monkeypatch.setattr(
        "rag.graph_nodes.get_player_stats",
        lambda player: {"name": player, "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "dismissal": None}, "bowling": {"overs": "4.0", "runs": 12, "wickets": 2}},
    )

    state = {
        "question": "Who dismissed Abhishek Sharma?",
        "rewritten_question": "Who dismissed Abhishek Sharma?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "semantic",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = plan_retrieval(state)

    assert updated["retrieval_plan"].players == ["JJ Bumrah"]
    assert updated["answer_strategy"] == "semantic"
    assert updated["retrieval_filters"]["$and"][1] == {"event": {"$eq": "wicket"}}
    assert updated["player_stats"][0]["name"] == "JJ Bumrah"
    assert updated["llm_traces"][0]["node"] == "plan_retrieval"


def test_plan_retrieval_for_over_question_forces_sequential(monkeypatch):
    from rag.graph_nodes import plan_retrieval

    monkeypatch.setattr("rag.graph_nodes.get_known_players", lambda: ["SV Samson"])
    monkeypatch.setattr("rag.graph_nodes.build_retrieval_plan", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM planner should be bypassed")))
    monkeypatch.setattr("rag.graph_nodes.get_match_metadata", lambda: {"max_over": 19, "max_innings": 2})

    state = {
        "question": "What happened in the last over?",
        "rewritten_question": "What happened in the last over?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "semantic",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = plan_retrieval(state)

    assert updated["retrieval_plan"].over == 19
    assert updated["retrieval_plan"].innings == 2
    assert updated["answer_strategy"] == "sequential"
    assert updated["is_sequential"] is True
    assert updated["retrieval_filters"] == {"$and": [{"over": {"$eq": 19}}, {"innings": {"$eq": 2}}]}
    assert updated["llm_traces"] == []


def test_plan_retrieval_for_most_sixes_bypasses_llm_and_player_lookup(monkeypatch):
    from rag.graph_nodes import plan_retrieval

    monkeypatch.setattr(
        "rag.graph_nodes.get_known_players",
        lambda: (_ for _ in ()).throw(AssertionError("Known players should not be loaded for stat fast path")),
    )
    monkeypatch.setattr(
        "rag.graph_nodes.build_retrieval_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM planner should be bypassed")),
    )

    state = {
        "question": "Who hit the most sixes?",
        "rewritten_question": "Who hit the most sixes?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "semantic",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = plan_retrieval(state)

    assert updated["retrieval_plan"].event == "six"
    assert updated["retrieval_plan"].answer_strategy == "hybrid"
    assert updated["retrieval_plan"].is_stat_question is True
    assert updated["retrieval_plan"].players == []
    assert updated["retrieval_filters"] == {"event": {"$eq": "six"}}
    assert updated["llm_traces"] == []


def test_plan_retrieval_for_first_wicket_uses_ordered_sequential_path(monkeypatch):
    from rag.graph_nodes import plan_retrieval

    monkeypatch.setattr(
        "rag.graph_nodes.get_known_players",
        lambda: (_ for _ in ()).throw(AssertionError("Known players should not be loaded for ordered event fast path")),
    )
    monkeypatch.setattr(
        "rag.graph_nodes.build_retrieval_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM planner should be bypassed")),
    )

    state = {
        "question": "Who took the first wicket?",
        "rewritten_question": "Who took the first wicket?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "semantic",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = plan_retrieval(state)

    assert updated["retrieval_plan"].event == "wicket"
    assert updated["retrieval_plan"].answer_strategy == "sequential"
    assert updated["retrieval_plan"].limit == 1
    assert updated["sort_direction"] == "asc"
    assert updated["retrieval_filters"] == {"event": {"$eq": "wicket"}}


def test_plan_retrieval_for_player_summary_bypasses_llm(monkeypatch):
    from rag.graph_nodes import plan_retrieval

    monkeypatch.setattr("rag.graph_nodes.get_known_players", lambda: ["SV Samson", "JJ Bumrah"])
    monkeypatch.setattr(
        "rag.graph_nodes.build_retrieval_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM planner should be bypassed")),
    )
    monkeypatch.setattr(
        "rag.graph_nodes.get_player_stats",
        lambda player: {"name": player, "batting": {"runs": 50, "balls": 30, "fours": 4, "sixes": 8, "dismissal": None}, "bowling": {"overs": "0.0", "runs": 0, "wickets": 0}},
    )

    state = {
        "question": "How was SV Samson's overall performance in the match?",
        "rewritten_question": "How was SV Samson's overall performance in the match?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "semantic",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "stage_timings_ms": {},
        "llm_traces": [],
    }

    updated = plan_retrieval(state)

    assert updated["answer_strategy"] == "semantic"
    assert updated["retrieval_plan"].players == ["SV Samson"]
    assert updated["retrieval_filters"]["$or"][0] == {"batter": {"$eq": "SV Samson"}}
    assert updated["player_stats"][0]["name"] == "SV Samson"
    assert updated["llm_traces"] == []


def test_plan_retrieval_overrides_bad_planner_route_for_player_summary(monkeypatch):
    from rag.graph_nodes import plan_retrieval

    monkeypatch.setattr("rag.graph_nodes.get_known_players", lambda: ["SV Samson", "JJ Bumrah"])
    monkeypatch.setattr(
        "rag.graph_nodes.build_retrieval_plan",
        lambda question, known: (
            RetrievalPlan(
                normalized_question=question,
                players=["SV Samson"],
                answer_strategy="aggregate",
                group_by="player",
                metric="count",
            ),
            {"node": "plan_retrieval"},
        ),
    )
    monkeypatch.setattr(
        "rag.graph_nodes.get_player_stats",
        lambda player: {"name": player, "batting": {"runs": 89, "balls": 42, "fours": 5, "sixes": 8, "dismissal": None}, "bowling": {"overs": "0.0", "runs": 0, "wickets": 0}},
    )

    state = {
        "question": "How was SV Samson's overall performance?",
        "rewritten_question": "How was SV Samson's overall performance?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": None,
        "retrieval_filters": None,
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "semantic",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "stage_timings_ms": {},
        "llm_traces": [],
    }

    updated = plan_retrieval(state)

    assert updated["answer_strategy"] == "semantic"
    assert updated["retrieval_plan"].answer_strategy == "semantic"
    assert updated["retrieval_plan"].players == ["SV Samson"]
    assert updated["player_stats"][0]["batting"]["runs"] == 89


def test_generate_answer_does_not_short_circuit_for_aggregate_route_with_players():
    from rag.graph_nodes import generate_answer
    from rag import graph_nodes

    graph_nodes.invoke_answer_chain = lambda **kwargs: ("SV Samson played a superb innings of 89.", {"node": "generate_answer"})

    state = {
        "question": "How was SV Samson's overall performance?",
        "rewritten_question": "How was SV Samson's overall performance?",
        "query_variants": ["How was SV Samson's overall performance?"],
        "chat_history": [],
        "retrieval_plan": RetrievalPlan(
            normalized_question="How was SV Samson's overall performance?",
            players=["SV Samson"],
            answer_strategy="aggregate",
            group_by="player",
            metric="count",
        ),
        "retrieval_filters": {"$or": [{"batter": {"$eq": "SV Samson"}}]},
        "player_stats": None,
        "aggregate_stats": "=== SYSTEM CALCULATED EXACT STATS ===",
        "aggregate_rows": [{"player": "SV Samson", "count": 50}],
        "answer_strategy": "aggregate",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": True,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "context",
        "answer": "",
        "stage_timings_ms": {},
        "llm_traces": [],
    }

    updated = generate_answer(state)

    assert updated["answer"] == "SV Samson played a superb innings of 89."
    assert updated["llm_traces"][0]["node"] == "generate_answer"


def test_retrieve_stat_question_skips_semantic_retrieval(monkeypatch):
    from rag.graph_nodes import retrieve

    monkeypatch.setattr(
        "rag.graph_nodes.retrieve_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Semantic retrieval should be skipped")),
    )

    state = {
        "question": "Who hit the most sixes?",
        "rewritten_question": "Who hit the most sixes?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": RetrievalPlan(
            normalized_question="Who hit the most sixes?",
            event="six",
            answer_strategy="aggregate",
            is_stat_question=True,
        ),
        "retrieval_filters": {"event": {"$eq": "six"}},
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "aggregate",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": True,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = retrieve(state)

    assert updated["query_variants"] == ["Who hit the most sixes?"]
    assert updated["initial_docs"] == []
    assert updated["retrieved_docs"] == []


def test_retrieve_hybrid_question_keeps_semantic_retrieval(monkeypatch):
    from rag.graph_nodes import retrieve

    captured = {}

    def fake_retrieve_documents(question, where=None, enable_multi_query=True, enable_context_compression=True):
        captured["question"] = question
        captured["where"] = where
        captured["enable_multi_query"] = enable_multi_query
        captured["enable_context_compression"] = enable_context_compression
        return [question], [{"metadata": {"id": "d1"}}], [{"metadata": {"id": "d1"}}], None

    monkeypatch.setattr("rag.graph_nodes.retrieve_documents", fake_retrieve_documents)

    state = {
        "question": "How impactful was Bumrah in the death overs?",
        "rewritten_question": "How impactful was Bumrah in the death overs?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": RetrievalPlan(
            normalized_question="How impactful was Bumrah in the death overs?",
            players=["JJ Bumrah"],
            answer_strategy="hybrid",
            metric="impact",
            is_stat_question=True,
        ),
        "retrieval_filters": {"$or": [{"bowler": {"$eq": "JJ Bumrah"}}]},
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "hybrid",
        "group_by": "player",
        "metric": "impact",
        "is_stat_question": True,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = retrieve(state)

    assert captured["question"] == "How impactful was Bumrah in the death overs?"
    assert captured["enable_multi_query"] is True
    assert captured["enable_context_compression"] is True
    assert updated["retrieved_docs"] == [{"metadata": {"id": "d1"}}]


def test_retrieve_hybrid_leader_question_filters_support_to_leader(monkeypatch):
    from rag.graph_nodes import retrieve

    captured = {}

    def fake_retrieve_documents(question, where=None, enable_multi_query=True, enable_context_compression=True):
        captured["question"] = question
        captured["where"] = where
        captured["enable_multi_query"] = enable_multi_query
        captured["enable_context_compression"] = enable_context_compression
        return [question], [{"metadata": {"id": "d1"}}], [{"metadata": {"id": "d1"}}], None

    monkeypatch.setattr("rag.graph_nodes.retrieve_documents", fake_retrieve_documents)

    state = {
        "question": "Who hit the most sixes?",
        "rewritten_question": "Who hit the most sixes?",
        "query_variants": [],
        "chat_history": [],
        "retrieval_plan": RetrievalPlan(
            normalized_question="Who hit the most sixes?",
            event="six",
            answer_strategy="hybrid",
            group_by="player",
            metric="count",
            is_stat_question=True,
        ),
        "retrieval_filters": {"event": {"$eq": "six"}},
        "player_stats": None,
        "aggregate_stats": "=== SYSTEM CALCULATED EXACT STATS ===",
        "aggregate_rows": [{"player": "SV Samson", "count": 8}],
        "answer_strategy": "hybrid",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": True,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = retrieve(state)

    assert captured["question"] == "SV Samson sixes"
    assert captured["where"] == {"$and": [{"event": {"$eq": "six"}}, {"batter": {"$eq": "SV Samson"}}]}
    assert captured["enable_multi_query"] is False
    assert captured["enable_context_compression"] is False
    assert updated["retrieved_docs"] == [{"metadata": {"id": "d1"}}]


def test_build_context_uses_aggregate_stats_without_docs():
    from rag.graph_nodes import build_context

    state = {
        "question": "Who hit the most sixes?",
        "rewritten_question": "Who hit the most sixes?",
        "query_variants": ["Who hit the most sixes?"],
        "chat_history": [],
        "retrieval_plan": RetrievalPlan(
            normalized_question="Who hit the most sixes?",
            event="six",
            answer_strategy="aggregate",
            is_stat_question=True,
        ),
        "retrieval_filters": {"event": {"$eq": "six"}},
        "player_stats": None,
        "aggregate_stats": (
            "=== SYSTEM CALCULATED EXACT STATS ===\n"
            "Stat leaderboard for event 'SIX':\n"
            "1. SV Samson — 8 sixes\n"
            "2. TL Seifert — 5 sixes\n"
            "=====================================\n"
        ),
        "aggregate_rows": None,
        "answer_strategy": "aggregate",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": True,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = build_context(state)

    assert "SV Samson — 8 sixes" in updated["context"]
    assert "Relevant Match Highlight Deliveries" not in updated["context"]


def test_ask_stream_emits_metadata_then_tokens(monkeypatch):
    from rag import rag_graph

    monkeypatch.setattr("rag.rag_graph.rewrite_question", lambda state: {**state, "rewritten_question": state["question"]})
    monkeypatch.setattr(
        "rag.rag_graph.plan_retrieval",
        lambda state: {
            **state,
            "retrieval_plan": RetrievalPlan(normalized_question=state["question"]),
            "retrieval_filters": None,
            "answer_strategy": "semantic",
        },
    )
    monkeypatch.setattr("rag.rag_graph.compute_aggregate_stats", lambda state: state)
    monkeypatch.setattr(
        "rag.rag_graph.retrieve",
        lambda state: {
            **state,
            "query_variants": [state["question"]],
            "initial_docs": [{"metadata": {"innings": 1, "over": 2, "ball": 3, "batter": "A", "bowler": "B", "event": "six", "runs_total": 6}}],
            "retrieved_docs": [{"metadata": {"innings": 1, "over": 2, "ball": 3, "batter": "A", "bowler": "B", "event": "six", "runs_total": 6}}],
        },
    )
    monkeypatch.setattr("rag.rag_graph.build_context", lambda state: {**state, "context": "sample context"})
    monkeypatch.setattr(
        "rag.rag_graph.generate_answer_stream",
        lambda state: (iter(["Hello", " world"]), {"node": "generate_answer", "prompt": "p", "response": "<streamed>"}),
    )

    events = list(rag_graph.ask_stream("How did India win?"))

    assert isinstance(events[0], dict)
    assert events[0]["answer_strategy"] == "semantic"
    assert events[0]["query_variants"] == ["How did India win?"]
    assert events[1:] == ["Hello", " world"]


def test_build_context_marks_complete_short_final_over():
    from rag.graph_nodes import build_context

    state = {
        "question": "What happened in the last over?",
        "rewritten_question": "What happened in the last over?",
        "query_variants": ["What happened in the last over?"],
        "chat_history": [],
        "retrieval_plan": RetrievalPlan(
            normalized_question="What happened in the last over?",
            answer_strategy="sequential",
            over=19,
            innings=2,
            is_sequential=True,
        ),
        "retrieval_filters": {"$and": [{"over": {"$eq": 19}}, {"innings": {"$eq": 2}}]},
        "player_stats": None,
        "aggregate_stats": None,
        "aggregate_rows": None,
        "answer_strategy": "sequential",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": False,
        "is_sequential": True,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [
            {
                "text": "Commentary: scores are level",
                "metadata": {"innings": 2, "over": 19, "ball": 1, "batter": "SV Samson", "bowler": "R Shepherd", "event": "six", "runs_total": 6},
            },
            {
                "text": "Commentary: Samson has won this for India",
                "metadata": {"innings": 2, "over": 19, "ball": 2, "batter": "SV Samson", "bowler": "R Shepherd", "event": "four", "runs_total": 4},
            },
        ],
        "context": "",
        "answer": "",
        "llm_traces": [],
    }

    updated = build_context(state)

    assert "complete set of recorded deliveries for innings 2, over 19" in updated["context"]
    assert "Only 2 deliveries are present because the innings ended at 19.2." in updated["context"]


def test_generate_answer_uses_llm_for_aggregate_route(monkeypatch):
    from rag.graph_nodes import generate_answer
    from rag import graph_nodes

    graph_nodes.invoke_answer_chain = lambda **kwargs: (
        "SV Samson had the most sixes with 8. The commentary-backed context shows he kept putting pressure on the bowlers whenever he got width.",
        {"node": "generate_answer"},
    )

    state = {
        "question": "Who hit the most sixes?",
        "rewritten_question": "Who hit the most sixes?",
        "query_variants": ["Who hit the most sixes?"],
        "chat_history": [],
        "retrieval_plan": RetrievalPlan(
            normalized_question="Who hit the most sixes?",
            event="six",
            answer_strategy="aggregate",
            is_stat_question=True,
            group_by="player",
            metric="count",
        ),
        "retrieval_filters": {"event": {"$eq": "six"}},
        "player_stats": None,
        "aggregate_stats": "=== SYSTEM CALCULATED EXACT STATS ===",
        "aggregate_rows": [{"player": "SV Samson", "count": 8}],
        "answer_strategy": "aggregate",
        "group_by": "player",
        "metric": "count",
        "is_stat_question": True,
        "is_sequential": False,
        "sort_direction": "asc",
        "limit": None,
        "initial_docs": [],
        "retrieved_docs": [],
        "context": "=== SYSTEM CALCULATED EXACT STATS ===",
        "answer": "",
        "llm_traces": [],
    }

    updated = generate_answer(state)

    assert "SV Samson had the most sixes with 8." in updated["answer"]
    assert updated["llm_traces"][0]["node"] == "generate_answer"
