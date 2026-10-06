"""The quality gate: a deterministic, explainable decision on whether research is good enough.

The critic (an LLM) supplies judgement; this gate turns that judgement plus measurable
properties of the evidence into a score and a routing decision. Keeping the final decision
deterministic makes the loop testable and prevents a model from talking itself past
weak evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

from researchgraph.schemas.evidence import Contradiction, Evidence
from researchgraph.schemas.report import Synthesis
from researchgraph.schemas.research import ResearchGap, ResearchPlan, RunLimits
from researchgraph.schemas.review import Critique, IssueType, QualityAssessment, Severity
from researchgraph.schemas.sources import SourceQuality

WEIGHTS = {
    "coverage": 0.30,
    "support": 0.25,
    "source_quality": 0.20,
    "critic": 0.15,
    "contradictions": 0.10,
}
_GAP_ISSUES = {
    IssueType.MISSING_COVERAGE,
    IssueType.WEAK_SOURCE,
    IssueType.UNSUPPORTED_CLAIM,
    IssueType.UNREPRESENTED_CONTRADICTION,
}


def budget_remaining(
    *, iteration: int, tool_calls_used: int, limits: RunLimits, reserve_calls: int = 3
) -> bool:
    return (
        iteration < limits.max_iterations
        and tool_calls_used + reserve_calls <= limits.max_tool_calls
    )


def _supports(item: Evidence, quality: Mapping[str, SourceQuality], limits: RunLimits) -> bool:
    """Evidence counts as support only if its source is credible *and* it is relevant."""
    q = quality.get(item.source_id)
    return (
        q is not None
        and q.overall >= limits.credible_source_threshold
        and item.relevance >= limits.min_evidence_relevance
    )


def derive_gaps(plan: ResearchPlan, synthesis: Synthesis, critique: Critique) -> list[ResearchGap]:
    """Turn high-impact critique issues (and uncovered subquestions) into research targets."""
    finding_sq = {f.id: f.subquestion_id for f in synthesis.findings}
    gaps: dict[str, ResearchGap] = {}
    for issue in critique.issues:
        if issue.issue_type not in _GAP_ISSUES or issue.severity not in (
            Severity.HIGH,
            Severity.CRITICAL,
        ):
            continue
        sq_id = issue.subquestion_id or next(
            (finding_sq[f] for f in issue.finding_ids if f in finding_sq), None
        )
        if sq_id is None or plan.subquestion(sq_id) is None or sq_id in gaps:
            continue
        gaps[sq_id] = ResearchGap(
            subquestion_id=sq_id,
            description=issue.description,
            reason="critic_feedback",
            suggested_queries=issue.suggested_queries[:3],
        )
    covered = {f.subquestion_id for f in synthesis.findings}
    for sq in plan.subquestions:
        if sq.id not in covered and sq.id not in gaps:
            gaps[sq.id] = ResearchGap(
                subquestion_id=sq.id,
                description="No finding addresses this subquestion",
                reason="critic_feedback",
                suggested_queries=[q.query for q in sq.search_queries[:2]],
            )
    return list(gaps.values())


def assess_quality(
    *,
    plan: ResearchPlan,
    synthesis: Synthesis,
    critique: Critique,
    evidence: Mapping[str, Evidence],
    quality: Mapping[str, SourceQuality],
    contradictions: list[Contradiction],
    iteration: int,
    tool_calls_used: int,
    limits: RunLimits,
) -> QualityAssessment:
    findings = synthesis.findings

    covered = {
        f.subquestion_id
        for f in findings
        if any(
            eid in evidence and _supports(evidence[eid], quality, limits) for eid in f.evidence_ids
        )
    }
    coverage = len(covered) / len(plan.subquestions) if plan.subquestions else 0.0

    supported = [
        f
        for f in findings
        if any(
            eid in evidence and _supports(evidence[eid], quality, limits) for eid in f.evidence_ids
        )
    ]
    support = len(supported) / len(findings) if findings else 0.0

    cited_sources = {
        evidence[eid].source_id for f in findings for eid in f.evidence_ids if eid in evidence
    }
    source_scores = [quality[s].overall for s in cited_sources if s in quality]
    source_quality = sum(source_scores) / len(source_scores) if source_scores else 0.0

    penalty = sum(
        0.15 * issue.severity.weight
        if issue.severity in (Severity.HIGH, Severity.CRITICAL)
        else 0.05 * issue.severity.weight
        for issue in critique.issues
    )
    critic = max(0.0, 1.0 - penalty)

    major = [c for c in contradictions if c.severity == "major"]
    referenced = {cid for f in findings for cid in f.contradiction_ids}
    contradiction_score = (
        (sum(1 for c in major if c.id in referenced) / len(major)) if major else 1.0
    )

    components = {
        "coverage": round(coverage, 3),
        "support": round(support, 3),
        "source_quality": round(source_quality, 3),
        "critic": round(critic, 3),
        "contradictions": round(contradiction_score, 3),
    }
    score = round(sum(WEIGHTS[k] * v for k, v in components.items()), 3)

    blocking = []
    critical = critique.count(Severity.CRITICAL)
    if critical:
        blocking.append(f"{critical} critical issue(s) raised by the critic")
    uncovered = [sq.id for sq in plan.subquestions if sq.id not in covered]
    if uncovered:
        blocking.append(f"No credible finding answers {', '.join(uncovered)}")
    if not findings:
        blocking.append("No findings were produced")

    passed = score >= limits.quality_threshold and not blocking
    gaps: list[ResearchGap] = []
    if passed:
        decision = "proceed"
    elif budget_remaining(iteration=iteration, tool_calls_used=tool_calls_used, limits=limits):
        gaps = derive_gaps(plan, synthesis, critique)
        decision = "research_more" if gaps else "proceed_with_limitations"
    else:
        decision = "proceed_with_limitations"

    return QualityAssessment(
        iteration=iteration,
        score=score,
        threshold=limits.quality_threshold,
        passed=passed,
        components=components,
        blocking_issues=blocking,
        decision=decision,
        gaps=gaps,
    )
