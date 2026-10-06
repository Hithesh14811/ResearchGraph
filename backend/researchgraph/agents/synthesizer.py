"""Synthesis (analysis): evidence pool → calibrated, evidence-linked findings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from researchgraph.agents.base import (
    UNTRUSTED_CONTENT_POLICY,
    call_structured,
    make_prompt,
    render_block,
)
from researchgraph.core.text import strip_citation_markers, truncate
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.evidence import Contradiction, Evidence
from researchgraph.schemas.llm import SynthesisOutput
from researchgraph.schemas.report import ResearchFinding, Synthesis
from researchgraph.schemas.research import ResearchPlan
from researchgraph.schemas.sources import SourceQuality

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SYSTEM = """You are the analysis component of an autonomous research system. Synthesise the evidence into findings.

Rules:
- Write 1-3 findings per subquestion. Each finding must cite the evidence IDs it rests on, using only IDs that appear in <evidence>.
- Calibrate confidence: 'high' only with multiple credible, consistent sources; 'moderate' with one strong or several weaker sources; 'low' otherwise.
- Prefer evidence from higher-quality sources (source_quality, 0-1). Flag in caveats when a finding depends on weak or single sources.
- Where evidence conflicts (see <contradictions>), state the disagreement in the finding and list the contradiction IDs.
- Never introduce facts, numbers or sources that are not in the evidence. Do not overgeneralise beyond the conditions the evidence describes.
{policy}"""

HUMAN = """{request}

{evidence}

{contradictions}"""

PROMPT = make_prompt(SYSTEM, HUMAN)
MAX_EVIDENCE_PER_SUBQUESTION = 10


def evidence_rank(item: Evidence, quality: Mapping[str, SourceQuality]) -> float:
    q = quality.get(item.source_id)
    return (q.overall if q else 0.3) * 0.5 + item.relevance * 0.3 + item.confidence * 0.2


def select_evidence(
    plan: ResearchPlan,
    evidence: Mapping[str, Evidence],
    quality: Mapping[str, SourceQuality],
    per_subquestion: int,
) -> list[Evidence]:
    selected: list[Evidence] = []
    for sq in plan.subquestions:
        items = sorted(
            (e for e in evidence.values() if e.subquestion_id == sq.id),
            key=lambda e: (evidence_rank(e, quality), e.id),
            reverse=True,
        )
        selected.extend(items[:per_subquestion])
    return selected


def evidence_rows(
    items: list[Evidence], quality: Mapping[str, SourceQuality], quote_chars: int = 300
) -> list[dict[str, Any]]:
    rows = []
    for e in items:
        q = quality.get(e.source_id)
        rows.append(
            {
                "id": e.id,
                "subquestion_id": e.subquestion_id,
                "claim": e.claim,
                "quote": truncate(e.supporting_text, quote_chars),
                "evidence_type": e.evidence_type.value,
                "source_title": truncate(e.source_title, 120),
                "source_type": q.source_type.value if q else "unknown",
                "source_quality": q.overall if q else None,
                "published": str(e.published_date or "unknown"),
            }
        )
    return rows


def validate_findings(
    output: SynthesisOutput,
    plan: ResearchPlan,
    evidence: Mapping[str, Evidence],
    contradictions: list[Contradiction],
    iteration: int,
) -> Synthesis:
    valid_sq = set(plan.subquestion_ids)
    valid_contradictions = {c.id for c in contradictions}
    findings: list[ResearchFinding] = []
    dropped = 0
    for item in output.findings:
        evidence_ids = list(dict.fromkeys(e for e in item.evidence_ids if e in evidence))
        if item.subquestion_id not in valid_sq or not evidence_ids:
            dropped += 1  # a finding without real evidence is never kept
            continue
        findings.append(
            ResearchFinding(
                id=f"F{len(findings) + 1}",
                subquestion_id=item.subquestion_id,
                statement=truncate(strip_citation_markers(item.statement), 800),
                evidence_ids=evidence_ids,
                confidence=item.confidence,
                caveats=[truncate(c, 300) for c in item.caveats[:4]],
                contradiction_ids=[c for c in item.contradiction_ids if c in valid_contradictions],
            )
        )
    return Synthesis(
        iteration=iteration,
        findings=findings,
        overall_assessment=truncate(output.overall_assessment, 2000),
        consensus_points=output.consensus_points[:8],
        disagreements=output.disagreements[:8],
        open_questions=output.open_questions[:8],
        dropped_finding_count=dropped,
    )


def fallback_synthesis(
    plan: ResearchPlan,
    evidence: Mapping[str, Evidence],
    quality: Mapping[str, SourceQuality],
    iteration: int,
) -> Synthesis:
    """Deterministic findings (top evidence per subquestion) when the model is unavailable."""
    findings: list[ResearchFinding] = []
    for sq in plan.subquestions:
        items = sorted(
            (e for e in evidence.values() if e.subquestion_id == sq.id),
            key=lambda e: (evidence_rank(e, quality), e.id),
            reverse=True,
        )[:2]
        for item in items:
            findings.append(
                ResearchFinding(
                    id=f"F{len(findings) + 1}",
                    subquestion_id=sq.id,
                    statement=item.claim,
                    evidence_ids=[item.id],
                    confidence="low",
                    caveats=[
                        "Automatically derived from a single evidence item (analysis model unavailable)."
                    ],
                )
            )
    return Synthesis(
        iteration=iteration,
        findings=findings,
        overall_assessment="Findings were assembled deterministically from the strongest evidence because the analysis model was unavailable.",
    )


async def synthesize(
    ctx: ResearchContext,
    *,
    question: str,
    plan: ResearchPlan,
    evidence: Mapping[str, Evidence],
    quality: Mapping[str, SourceQuality],
    contradictions: list[Contradiction],
    iteration: int,
) -> Synthesis:
    selected = select_evidence(plan, evidence, quality, MAX_EVIDENCE_PER_SUBQUESTION)
    request = render_block(
        "synthesis_request",
        {
            "question": question,
            "subquestions": [{"id": sq.id, "question": sq.question} for sq in plan.subquestions],
        },
    )
    output = await call_structured(
        ctx,
        tier=ModelTier.STRONG,
        prompt=PROMPT,
        variables={
            "request": request,
            "evidence": render_block("evidence", evidence_rows(selected, quality)),
            "contradictions": render_block(
                "contradictions",
                [
                    {
                        "id": c.id,
                        "evidence_ids": c.evidence_ids,
                        "description": c.description,
                        "severity": c.severity,
                    }
                    for c in contradictions
                ],
            ),
            "policy": UNTRUSTED_CONTENT_POLICY,
        },
        schema=SynthesisOutput,
        operation="synthesis",
    )
    return validate_findings(output, plan, evidence, contradictions, iteration)
