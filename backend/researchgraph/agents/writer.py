"""Report writer: verified findings → structured draft where every sentence carries evidence IDs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from researchgraph.agents.base import (
    UNTRUSTED_CONTENT_POLICY,
    call_structured,
    make_prompt,
    render_block,
)
from researchgraph.agents.synthesizer import evidence_rows
from researchgraph.core.text import strip_citation_markers, truncate
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.evidence import Contradiction, Evidence
from researchgraph.schemas.llm import DraftSentence, ReportDraftOutput
from researchgraph.schemas.report import ReportClaim, ReportDraft, ReportSection, Synthesis
from researchgraph.schemas.research import ResearchPlan
from researchgraph.schemas.review import Critique, QualityAssessment
from researchgraph.schemas.sources import SourceQuality

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SYSTEM = """You write the final technical report of an autonomous research system, for an expert audience.

Rules:
- Every sentence that states a fact must list the evidence IDs that support it in `evidence_ids`, using only IDs from <evidence>.
- Never write citation markers (like [1]) or URLs in the text — citations are generated automatically from evidence IDs.
- State nothing the findings and evidence do not support. Copy numbers exactly. Preserve caveats and uncertainty, and describe conflicting evidence where it exists.
- Structure: a specific title; an executive summary of 3-6 sentences; one section per subquestion (short topical heading), plus a comparative synthesis section when the question compares options; actionable recommendations grounded in evidence; limitations (include gaps from <quality_review>).
- Be concise and precise; avoid filler.
{policy}"""

HUMAN = """{request}

{findings}

{evidence}

{contradictions}"""

PROMPT = make_prompt(SYSTEM, HUMAN)


class _ClaimIds:
    def __init__(self) -> None:
        self.n = 0

    def claim(self, text: str, evidence_ids: list[str]) -> ReportClaim:
        self.n += 1
        return ReportClaim(
            id=f"CL{self.n}",
            text=truncate(strip_citation_markers(text), 1200),
            evidence_ids=list(dict.fromkeys(evidence_ids)),
        )


def to_draft(output: ReportDraftOutput) -> ReportDraft:
    ids = _ClaimIds()

    def convert(sentences: list[DraftSentence]) -> list[ReportClaim]:
        return [ids.claim(s.text, s.evidence_ids) for s in sentences if s.text.strip()]

    sections = []
    for section in output.sections:
        paragraphs = [p for p in (convert(par.sentences) for par in section.paragraphs) if p]
        if paragraphs:
            sections.append(
                ReportSection(heading=truncate(section.heading, 160), paragraphs=paragraphs)
            )
    return ReportDraft(
        title=truncate(output.title, 200),
        executive_summary=convert(output.executive_summary),
        sections=sections,
        recommendations=convert(output.recommendations),
        limitations=[truncate(item, 500) for item in output.limitations[:10]],
    )


def fallback_draft(
    *, question: str, plan: ResearchPlan, synthesis: Synthesis, quality: QualityAssessment | None
) -> ReportDraft:
    """Deterministic report assembled from findings when the writer model is unavailable."""
    ids = _ClaimIds()
    by_sq = {
        sq.id: [f for f in synthesis.findings if f.subquestion_id == sq.id]
        for sq in plan.subquestions
    }
    sections = [
        ReportSection(
            heading=sq.question,
            paragraphs=[[ids.claim(f.statement, f.evidence_ids) for f in by_sq[sq.id]]],
        )
        for sq in plan.subquestions
        if by_sq[sq.id]
    ]
    top = sorted(
        synthesis.findings, key=lambda f: {"high": 0, "moderate": 1, "low": 2}[f.confidence]
    )[:4]
    limitations = ["This report was assembled without the report-writing model; prose is minimal."]
    if quality is not None and not quality.passed:
        limitations.extend(quality.blocking_issues)
    return ReportDraft(
        title=f"Research report: {truncate(question, 120)}",
        executive_summary=[ids.claim(f.statement, f.evidence_ids) for f in top],
        sections=sections,
        recommendations=[],
        limitations=limitations,
    )


async def write_report(
    ctx: ResearchContext,
    *,
    question: str,
    plan: ResearchPlan,
    synthesis: Synthesis,
    evidence: Mapping[str, Evidence],
    quality_scores: Mapping[str, SourceQuality],
    contradictions: list[Contradiction],
    critique: Critique | None,
    quality: QualityAssessment | None,
) -> ReportDraft:
    # Evidence behind findings, plus both sides of every contradiction: the writer must be
    # able to cite a conflicting claim even when no finding rests on it.
    cited_ids = {e for f in synthesis.findings for e in f.evidence_ids}
    cited_ids |= {e for c in contradictions for e in c.evidence_ids}
    cited = [evidence[e] for e in sorted(cited_ids) if e in evidence]
    open_issues = [
        {"type": i.issue_type.value, "severity": i.severity.value, "description": i.description}
        for i in (critique.issues if critique else [])
        if i.severity.value in ("medium", "high", "critical")
    ]
    request = render_block(
        "report_request",
        {
            "question": question,
            "objective": plan.objective,
            "subquestions": [{"id": sq.id, "question": sq.question} for sq in plan.subquestions],
            "overall_assessment": synthesis.overall_assessment,
            "open_questions": synthesis.open_questions,
            "quality_review": {
                "score": quality.score if quality else None,
                "decision": quality.decision if quality else None,
                "blocking_issues": quality.blocking_issues if quality else [],
                "open_issues": open_issues[:12],
            },
        },
    )
    output = await call_structured(
        ctx,
        tier=ModelTier.STRONG,
        prompt=PROMPT,
        variables={
            "request": request,
            "findings": render_block(
                "findings", [f.model_dump(mode="json") for f in synthesis.findings]
            ),
            "evidence": render_block(
                "evidence", evidence_rows(cited, quality_scores, quote_chars=250)
            ),
            "contradictions": render_block(
                "contradictions", [c.model_dump(mode="json") for c in contradictions]
            ),
            "policy": UNTRUSTED_CONTENT_POLICY,
        },
        schema=ReportDraftOutput,
        operation="report_writer",
    )
    return to_draft(output)
