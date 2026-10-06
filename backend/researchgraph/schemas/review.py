"""Review artifacts: evidence coverage, critique, quality gate and citation checks."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from researchgraph.schemas.common import DomainModel
from researchgraph.schemas.research import ResearchGap


class SubquestionCoverage(DomainModel):
    subquestion_id: str
    evidence_count: int
    credible_evidence_count: int
    source_count: int
    mean_source_quality: float
    sufficient: bool


class EvidenceAssessment(DomainModel):
    """Output of the evidence gate after each research round."""

    iteration: int
    coverage: list[SubquestionCoverage]
    sufficient: bool
    gaps: list[ResearchGap] = Field(default_factory=list)

    @property
    def insufficient_ids(self) -> list[str]:
        return [c.subquestion_id for c in self.coverage if not c.sufficient]


class IssueType(StrEnum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    WEAK_SOURCE = "weak_source"
    CITATION_MISMATCH = "citation_mismatch"
    LOGICAL_GAP = "logical_gap"
    UNREPRESENTED_CONTRADICTION = "unrepresented_contradiction"
    MISSING_COVERAGE = "missing_coverage"
    OVERCONFIDENCE = "overconfidence"
    OVERGENERALIZATION = "overgeneralization"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        return {"low": 0.25, "medium": 0.5, "high": 1.0, "critical": 2.0}[self.value]


class CritiqueIssue(DomainModel):
    id: str
    issue_type: IssueType
    severity: Severity
    description: str
    finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    subquestion_id: str | None = None
    suggested_action: str = ""
    suggested_queries: list[str] = Field(default_factory=list)
    origin: Literal["rule", "llm"] = "llm"


CritiqueVerdict = Literal["acceptable", "needs_revision", "major_problems"]


class Critique(DomainModel):
    iteration: int
    issues: list[CritiqueIssue]
    strengths: list[str] = Field(default_factory=list)
    summary: str = ""
    verdict: CritiqueVerdict

    def count(self, *severities: Severity) -> int:
        return sum(1 for issue in self.issues if issue.severity in severities)


class QualityAssessment(DomainModel):
    """The quality gate's decision record (deterministic and explainable)."""

    iteration: int
    score: float = Field(ge=0, le=1)
    threshold: float
    passed: bool
    components: dict[str, float]
    blocking_issues: list[str] = Field(default_factory=list)
    decision: Literal["proceed", "research_more", "proceed_with_limitations"]
    gaps: list[ResearchGap] = Field(default_factory=list)


CitationProblem = Literal[
    "missing_citation", "unknown_evidence", "low_relevance", "numeric_mismatch"
]


class CitationIssue(DomainModel):
    claim_id: str
    claim_text: str
    problem: CitationProblem
    detail: str


class CitationCheck(DomainModel):
    attempt: int
    total_claims: int
    cited_claims: int
    verified_claims: int
    coverage: float
    issues: list[CitationIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues
