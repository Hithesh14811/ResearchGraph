"""LLM-facing output schemas.

These are deliberately separate from the domain models: they contain only what the model
should decide (no IDs, URLs or timestamps — those are assigned deterministically in code),
they are tolerant of extra keys, and their field descriptions double as instructions in the
tool schema that structured output sends to the provider.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from researchgraph.schemas.evidence import EvidenceType
from researchgraph.schemas.review import IssueType, Severity

SourcePref = Literal["academic", "web", "mixed"]


class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")


# --- Planner --------------------------------------------------------------------------------
class PlannedSubquestion(LLMOutput):
    question: str = Field(description="A focused, independently answerable research subquestion.")
    rationale: str = Field(description="Why answering it matters for the overall objective.")
    information_requirements: list[str] = Field(
        description="Concrete facts or data needed, e.g. benchmark results, cost figures."
    )
    search_queries: list[str] = Field(
        min_length=1, max_length=5, description="2-3 precise search-engine queries."
    )
    preferred_sources: SourcePref = Field(
        description="'academic' for papers/benchmarks, 'web' for docs/industry practice, or 'mixed'."
    )


class PlannerOutput(LLMOutput):
    objective: str = Field(description="One-sentence statement of the research objective.")
    scope: str = Field(description="What is in and out of scope.")
    subquestions: list[PlannedSubquestion] = Field(min_length=2, max_length=10)
    source_strategy: list[str] = Field(description="Source types to prioritise, best first.")
    stopping_criteria: list[str] = Field(
        description="Observable conditions under which the evidence is sufficient."
    )


# --- Query generation -----------------------------------------------------------------------
class QueryPlan(LLMOutput):
    subquestion_id: str = Field(description="The subquestion ID, e.g. 'SQ2'.")
    queries: list[str] = Field(min_length=1, max_length=4)
    preferred_sources: SourcePref


class QueryGenerationOutput(LLMOutput):
    plans: list[QueryPlan]


# --- Evidence extraction --------------------------------------------------------------------
class ExtractedEvidence(LLMOutput):
    passage_id: str = Field(description="ID of the passage the quote is copied from, e.g. 'P3'.")
    claim: str = Field(description="One self-contained factual claim, stated precisely.")
    quote: str = Field(
        description="Exact verbatim text copied from that passage that supports the claim "
        "(one or two sentences, max ~350 characters). Do not paraphrase."
    )
    evidence_type: EvidenceType
    confidence: float = Field(ge=0, le=1, description="How strongly the quote supports the claim.")
    relevance: float = Field(
        ge=0, le=1, description="How relevant the claim is to the subquestion."
    )


class EvidenceExtractionOutput(LLMOutput):
    evidence: list[ExtractedEvidence] = Field(max_length=15)


# --- Contradiction detection -----------------------------------------------------------------
class ContradictionJudgement(LLMOutput):
    pair_id: str
    is_contradiction: bool = Field(
        description="True only if both claims cannot be simultaneously true under the same conditions."
    )
    explanation: str = Field(
        description="One sentence describing the disagreement or why it is not one."
    )
    severity: Literal["minor", "major"] = "minor"


class ContradictionOutput(LLMOutput):
    judgements: list[ContradictionJudgement]


# --- Synthesis ------------------------------------------------------------------------------
class SynthesizedFinding(LLMOutput):
    subquestion_id: str
    statement: str = Field(
        description="A finding stated no more strongly than the evidence allows."
    )
    evidence_ids: list[str] = Field(
        min_length=1, description="IDs of supporting evidence, e.g. 'E-1a2b'."
    )
    confidence: Literal["high", "moderate", "low"]
    caveats: list[str] = Field(default_factory=list)
    contradiction_ids: list[str] = Field(default_factory=list)


class SynthesisOutput(LLMOutput):
    findings: list[SynthesizedFinding]
    overall_assessment: str
    consensus_points: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# --- Critic ---------------------------------------------------------------------------------
class CritiqueIssueOutput(LLMOutput):
    issue_type: IssueType
    severity: Severity
    description: str
    finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    subquestion_id: str | None = None
    suggested_action: str = ""
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Searches that could resolve the issue, if research would help.",
    )


class CriticOutput(LLMOutput):
    issues: list[CritiqueIssueOutput]
    strengths: list[str] = Field(default_factory=list)
    summary: str
    verdict: Literal["acceptable", "needs_revision", "major_problems"]


# --- Report writing -------------------------------------------------------------------------
class DraftSentence(LLMOutput):
    text: str = Field(
        description="One sentence. Never include citation markers, numbers in brackets or URLs."
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="Evidence IDs that directly support this sentence."
    )


class DraftParagraph(LLMOutput):
    sentences: list[DraftSentence] = Field(min_length=1)


class DraftSection(LLMOutput):
    heading: str
    paragraphs: list[DraftParagraph] = Field(min_length=1)


class ReportDraftOutput(LLMOutput):
    title: str
    executive_summary: list[DraftSentence] = Field(min_length=1, max_length=8)
    sections: list[DraftSection] = Field(min_length=1)
    recommendations: list[DraftSentence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
