from rag.chains import build_answer_prompt_value, history_to_messages
from rag.schemas import RetrievalPlan


def test_history_to_messages_preserves_roles():
    history = [
        {"role": "user", "content": "Who dismissed Abhishek Sharma?"},
        {"role": "assistant", "content": "R Ravindra dismissed him."},
    ]

    messages = history_to_messages(history)

    assert len(messages) == 2
    assert messages[0].type == "human"
    assert messages[1].type == "ai"


def test_build_answer_prompt_value_includes_stats_and_context():
    prompt_value = build_answer_prompt_value(
        question="Who hit the most sixes?",
        chat_history=[{"role": "user", "content": "How did India bat?"}],
        context="Over 2.3 Samson hit a six.",
        aggregate_stats="1. SV Samson — 3 sixes",
    )

    prompt_text = prompt_value.to_string()

    assert "Who hit the most sixes?" in prompt_text
    assert "SV Samson — 3 sixes" in prompt_text
    assert "Over 2.3 Samson hit a six." in prompt_text


def test_retrieval_plan_defaults_null_optional_fields():
    plan = RetrievalPlan.model_validate(
        {
            "normalized_question": "Who dismissed Abhishek Sharma?",
            "players": ["Abhishek Sharma"],
            "event": None,
            "over": None,
            "innings": None,
            "is_stat_question": False,
            "group_by": "player",
            "metric": None,
            "is_sequential": False,
            "sort_direction": None,
            "limit": None,
        }
    )

    assert plan.metric == "count"
    assert plan.sort_direction == "asc"
