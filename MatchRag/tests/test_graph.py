from rag.schemas import RetrievalPlan


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
    assert updated["is_sequential"] is True
    assert updated["retrieval_filters"] == {"$and": [{"over": {"$eq": 19}}, {"innings": {"$eq": 2}}]}
    assert updated["llm_traces"] == []


def test_ask_stream_emits_metadata_then_tokens(monkeypatch):
    from rag import rag_graph

    monkeypatch.setattr("rag.rag_graph.rewrite_question", lambda state: {**state, "rewritten_question": state["question"]})
    monkeypatch.setattr(
        "rag.rag_graph.plan_retrieval",
        lambda state: {
            **state,
            "retrieval_plan": RetrievalPlan(normalized_question=state["question"]),
            "retrieval_filters": None,
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
            over=19,
            innings=2,
            is_sequential=True,
        ),
        "retrieval_filters": {"$and": [{"over": {"$eq": 19}}, {"innings": {"$eq": 2}}]},
        "player_stats": None,
        "aggregate_stats": None,
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
