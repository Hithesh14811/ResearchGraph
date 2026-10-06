"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchgraph.core.text import sanitize_user_text
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.evidence import Contradiction, Evidence
from researchgraph.schemas.report import Citation, ReportMetrics, ResearchFinding
from researchgraph.schemas.research import ResearchPlan
from researchgraph.schemas.review import Critique, QualityAssessment
from researchgraph.schemas.sources import Source, SourceQuality

RunMode = Literal["demo", "live"]

MAX_PAGE_SIZE = 500


class CreateResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=10, max_length=4000, description="The research question.")
    auto_approve: bool = Field(
        default=False, description="Skip the human plan-approval checkpoint."
    )
    failure_scenarios: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Demo mode only: failures to simulate (or ['all']).",
    )
    mode: RunMode = Field(
        default="demo",
        description="With the live mode gate on: 'live' uses the real provider and needs a "
        "token from POST /auth/live in the X-Live-Token header. Ignored otherwise.",
    )

    @field_validator("question")
    @classmethod
    def _sanitize(cls, value: str) -> str:
        cleaned = sanitize_user_text(value, max_chars=4000)
        if len(cleaned) < 10:
            raise ValueError("Question is too short")
        return cleaned


class EditPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: ResearchPlan
    approve: bool = True


class ReplanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback: str = Field(min_length=3, max_length=2000)


class RunResponse(BaseModel):
    id: str
    question: str
    status: RunStatus
    current_node: str | None
    current_stage: str | None
    progress: float
    iteration: int
    auto_approve: bool
    mode: RunMode
    failure_scenarios: list[str]
    metrics: dict[str, Any]
    usage: dict[str, Any]
    error: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    awaiting_approval: bool


class RunListResponse(BaseModel):
    items: list[RunResponse]
    total: int


class PlanResponse(BaseModel):
    research_id: str
    status: RunStatus
    editable: bool
    plan: ResearchPlan | None


class SourceItem(BaseModel):
    source: Source
    quality: SourceQuality | None


class SourcesResponse(BaseModel):
    items: list[SourceItem]
    total: int


class EvidenceResponse(BaseModel):
    items: list[Evidence]
    total: int


class FindingsResponse(BaseModel):
    items: list[ResearchFinding]


class ReportResponse(BaseModel):
    research_id: str
    title: str
    markdown: str
    executive_summary: str
    bibliography: list[Citation]
    metrics: ReportMetrics
    review_notes: list[str]
    critique: Critique | None
    quality: QualityAssessment | None
    contradictions: list[Contradiction]
    generated_at: datetime


class EventItem(BaseModel):
    seq: int
    type: str
    node: str | None
    message: str
    data: dict[str, Any]
    timestamp: datetime


class EventsResponse(BaseModel):
    items: list[EventItem]


class GraphResponse(BaseModel):
    mermaid: str


class LiveModeInfo(BaseModel):
    llm_provider: str
    model: str
    search_provider: str


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    llm_provider: str
    model: str
    search_provider: str
    embedding_provider: str
    vector_store: str
    database: str
    tracing_enabled: bool
    active_runs: int
    live_mode: LiveModeInfo | None = Field(
        default=None,
        description="Present when a password-protected live lane runs next to the demo lane.",
    )


class LiveAuthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: str = Field(min_length=1, max_length=256)


class LiveAuthResponse(BaseModel):
    token: str
    expires_at: int = Field(description="Unix time at which the token stops working.")
