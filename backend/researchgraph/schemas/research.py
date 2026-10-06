"""Planning-related models: the research plan, its subquestions and per-round tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from researchgraph.schemas.common import DomainModel

SourcePreference = Literal["academic", "web", "mixed"]


class SearchQuery(DomainModel):
    query: str = Field(min_length=3, max_length=300)
    source_preference: SourcePreference = "mixed"

    @field_validator("query")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        return " ".join(value.split())


class ResearchSubquestion(DomainModel):
    id: str = Field(pattern=r"^SQ\d+$")
    question: str = Field(min_length=5, max_length=500)
    rationale: str = ""
    information_requirements: list[str] = Field(default_factory=list)
    search_queries: list[SearchQuery] = Field(default_factory=list)
    preferred_sources: SourcePreference = "mixed"


class ResearchPlan(DomainModel):
    objective: str
    scope: str = ""
    subquestions: list[ResearchSubquestion] = Field(min_length=1, max_length=12)
    source_strategy: list[str] = Field(default_factory=list)
    stopping_criteria: list[str] = Field(default_factory=list)
    version: int = 1

    def subquestion(self, subquestion_id: str) -> ResearchSubquestion | None:
        return next((sq for sq in self.subquestions if sq.id == subquestion_id), None)

    @property
    def subquestion_ids(self) -> list[str]:
        return [sq.id for sq in self.subquestions]


TaskReason = Literal["initial", "insufficient_evidence", "critic_feedback"]


class ResearchTask(DomainModel):
    """One unit of parallel work: research a single subquestion in a single round."""

    task_id: str
    subquestion_id: str
    question: str
    queries: list[SearchQuery]
    iteration: int
    reason: TaskReason = "initial"
    focus: str | None = None
    tool_budget: int = Field(ge=0)
    preferred_sources: SourcePreference = "mixed"


class ResearchGap(DomainModel):
    """Missing evidence that justifies another research round."""

    subquestion_id: str
    description: str
    reason: Literal["insufficient_evidence", "critic_feedback"]
    suggested_queries: list[str] = Field(default_factory=list)


class PlanDecision(DomainModel):
    """The human reviewer's response to the plan-approval interrupt."""

    action: Literal["approve", "edit", "replan", "cancel"]
    plan: ResearchPlan | None = None
    approve: bool = Field(
        default=True, description="For 'edit': approve the edited plan immediately."
    )
    feedback: str | None = Field(default=None, max_length=2000)


class RunLimits(DomainModel):
    """Per-run budget, stored in state so routing functions stay pure and testable."""

    max_iterations: int = Field(ge=1)
    max_tool_calls: int = Field(ge=1)
    max_citation_repairs: int = Field(ge=0)
    min_evidence_per_subquestion: int = Field(ge=1)
    quality_threshold: float = Field(ge=0, le=1)
    credible_source_threshold: float = Field(ge=0, le=1)
    min_evidence_relevance: float = Field(default=0.4, ge=0, le=1)
