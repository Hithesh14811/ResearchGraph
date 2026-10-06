"""Synthesis and report models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from researchgraph.schemas.common import DomainModel, utc_now
from researchgraph.schemas.sources import SourceType

Confidence = Literal["high", "moderate", "low"]


class ResearchFinding(DomainModel):
    id: str
    subquestion_id: str
    statement: str
    evidence_ids: list[str]
    confidence: Confidence
    caveats: list[str] = Field(default_factory=list)
    contradiction_ids: list[str] = Field(default_factory=list)


class Synthesis(DomainModel):
    """The analysis step's output: structured findings over the evidence pool."""

    iteration: int
    findings: list[ResearchFinding]
    overall_assessment: str
    consensus_points: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    dropped_finding_count: int = 0


class ReportClaim(DomainModel):
    """A single citable sentence of the report. Citations are attached by evidence ID —
    the model never writes citation numbers or URLs itself."""

    id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReportSection(DomainModel):
    heading: str
    paragraphs: list[list[ReportClaim]]

    @property
    def claims(self) -> list[ReportClaim]:
        return [claim for paragraph in self.paragraphs for claim in paragraph]


class ReportDraft(DomainModel):
    title: str
    executive_summary: list[ReportClaim]
    sections: list[ReportSection]
    recommendations: list[ReportClaim]
    limitations: list[str] = Field(default_factory=list)
    removed_claims: list[ReportClaim] = Field(default_factory=list)

    def all_claims(self) -> list[ReportClaim]:
        claims = list(self.executive_summary)
        for section in self.sections:
            claims.extend(section.claims)
        claims.extend(self.recommendations)
        return claims


class Citation(DomainModel):
    number: int
    source_id: str
    title: str
    url: str
    authors: list[str] = Field(default_factory=list)
    published_date: date | None = None
    venue: str | None = None
    source_type: SourceType = SourceType.UNKNOWN
    quality: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReportMetrics(DomainModel):
    total_claims: int
    cited_claims: int
    citation_coverage: float
    unsupported_claim_rate: float
    removed_claims: int
    sources_cited: int
    mean_cited_source_quality: float
    evidence_items: int
    research_iterations: int


class FinalReport(DomainModel):
    research_id: str
    question: str
    title: str
    executive_summary_markdown: str
    markdown: str
    draft: ReportDraft
    bibliography: list[Citation]
    metrics: ReportMetrics
    review_notes: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
