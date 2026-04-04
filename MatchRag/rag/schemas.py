"""Typed schemas used across LangChain chains and LangGraph state."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


AnswerStrategy = Literal["semantic", "aggregate", "sequential", "hybrid"]
QuestionType = Literal["match_summary", "player_performance", "over_summary", "comparison", "general"]
Phase = Literal["powerplay", "middle", "death"]

class RetrievalPlan(BaseModel):
    """Structured routing plan produced before retrieval."""

    normalized_question: str = Field(
        default="",
        description="Standalone form of the user's question for retrieval and answer generation.",
    )
    players: list[str] = Field(
        default_factory=list,
        description="Exact player names when they can be resolved from the known-player list.",
    )
    event: str | None = Field(
        default=None,
        description="One of wicket, six, four, dot, single, or run when the question targets an event.",
    )
    phase: Phase | None = Field(
        default=None,
        description=(
            "T20 match phase when the question targets a specific phase: "
            "'powerplay' (overs 1-6), 'middle' (overs 7-15), or 'death' (overs 16-20)."
        ),
    )
    over: int | Literal["last"] | None = Field(
        default=None,
        description="Specific over number or 'last' if the user asks about the final over.",
    )
    innings: int | None = Field(default=None, description="Specific innings number when given.")
    answer_strategy: AnswerStrategy = Field(
        default="semantic",
        description=(
            "Primary execution path: semantic retrieval, deterministic aggregate stats, "
            "exact sequential fetch, or hybrid aggregate-plus-retrieval."
        ),
    )
    is_stat_question: bool = Field(
        default=False,
        description="True when the answer requires exact aggregate stats rather than only narrative retrieval.",
    )
    group_by: Literal["player", "over", "innings", "wicket_kind"] = Field(default="player")
    metric: Literal["count", "runs_total", "impact"] = Field(default="count")
    is_sequential: bool = Field(
        default=False,
        description="True when the user wants chronological events rather than semantic similarity.",
    )
    sort_direction: Literal["asc", "desc"] = Field(default="asc")
    limit: int | None = Field(default=None, description="Optional item limit for sequential retrieval.")

    @field_validator("group_by", mode="before")
    @classmethod
    def _default_group_by(cls, value: Any) -> Any:
        return "player" if value in (None, "") else value

    @field_validator("metric", mode="before")
    @classmethod
    def _default_metric(cls, value: Any) -> Any:
        return "count" if value in (None, "") else value

    @field_validator("sort_direction", mode="before")
    @classmethod
    def _default_sort_direction(cls, value: Any) -> Any:
        return "asc" if value in (None, "") else value

    @field_validator("answer_strategy", mode="before")
    @classmethod
    def _default_answer_strategy(cls, value: Any) -> Any:
        return "semantic" if value in (None, "") else value

    @model_validator(mode="after")
    def _synchronize_routing_flags(self) -> "RetrievalPlan":
        if self.answer_strategy == "aggregate":
            self.is_stat_question = True
            self.is_sequential = False
        elif self.answer_strategy == "sequential":
            self.is_stat_question = False
            self.is_sequential = True
        elif self.answer_strategy == "hybrid":
            self.is_stat_question = True
            self.is_sequential = False
        else:
            self.is_stat_question = False
            self.is_sequential = False

        return self


class LLMTrace(BaseModel):
    """Prompt/response trace shown in the pipeline inspector."""

    node: str
    prompt: str
    response: str
