"""Independent critic: reviews research artifacts, never rewrites them.

Two layers:
1. Rule-based checks that are cheap and certain (no evidence, only weak sources,
   single-source "high confidence", ignored contradictions, uncovered subquestions).
2. An LLM review for judgement calls (overgeneralisation, logical gaps, citation relevance,
   what further research would change the conclusion).
Both produce the same ``CritiqueIssue`` type and are merged.
"""

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
from researchgraph.core.text import truncate
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.evidence import Contradiction, Evidence
from researchgraph.schemas.llm import CriticOutput
from researchgraph.schemas.report import Synthesis
from researchgraph.schemas.research import ResearchPlan
from researchgraph.schemas.review import (
    Critique,
    CritiqueIssue,
    CritiqueVerdict,
    EvidenceAssessment,
    IssueType,
    Severity,
)
from researchgraph.schemas.sources import SourceQuality

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SYSTEM = """You are an independent reviewer of a research analysis. You did not write it, and you must not rewrite it — identify problems.

Check the findings against the evidence they cite:
1. Unsupported claims: statements that go beyond the cited evidence.
2. Weak sources: findings resting mainly on low-quality sources (source_quality below 0.45) or on a single source.
3. Overgeneralisation / overconfidence: conclusions stronger or broader than the evidence's conditions.
4. Contradictions: conflicting evidence (see <contradictions>) that the findings ignore.
5. Citation relevance: cited evidence that does not actually support the statement.
6. Missing coverage: subquestions or important aspects without adequate findings (see <coverage>).
7. What additional research would materially change the conclusion?

Severity: critical = the conclusion is likely wrong; high = materially weakens a conclusion; medium = should be qualified; low = minor.
Only raise issues you can tie to specific findings, evidence or subquestions. For issues that more research could fix, give specific suggested_queries.
verdict: 'acceptable' if no high/critical issues, 'needs_revision' if fixable issues exist, 'major_problems' if conclusions are unreliable.
{policy}"""

HUMAN = """{request}

{findings}

{evidence}

{contradictions}"""

PROMPT = make_prompt(SYSTEM, HUMAN)
_VERDICT_RANK: dict[str, int] = {"acceptable": 0, "needs_revision": 1, "major_problems": 2}


def rule_based_issues(
    *,
    plan: ResearchPlan,
    synthesis: Synthesis,
    evidence: Mapping[str, Evidence],
    quality: Mapping[str, SourceQuality],
    contradictions: list[Contradiction],
    credible_threshold: float,
) -> list[CritiqueIssue]:
    issues: list[CritiqueIssue] = []

    def add(**kwargs: object) -> None:
        issues.append(CritiqueIssue(id=f"R{len(issues) + 1}", origin="rule", **kwargs))

    for finding in synthesis.findings:
        cited = [evidence[e] for e in finding.evidence_ids if e in evidence]
        sources = {e.source_id for e in cited}
        scores = [quality[s].overall for s in sources if s in quality]
        if not cited:
            add(
                issue_type=IssueType.UNSUPPORTED_CLAIM,
                severity=Severity.HIGH,
                finding_ids=[finding.id],
                subquestion_id=finding.subquestion_id,
                description=f"{finding.id} cites no valid evidence.",
                suggested_action="Remove the finding or find supporting evidence.",
            )
            continue
        if scores and max(scores) < credible_threshold:
            sq = plan.subquestion(finding.subquestion_id)
            add(
                issue_type=IssueType.WEAK_SOURCE,
                severity=Severity.HIGH,
                finding_ids=[finding.id],
                subquestion_id=finding.subquestion_id,
                description=f"{finding.id} relies only on low-quality sources (best score {max(scores):.2f}).",
                suggested_action="Find primary research or official documentation that supports or refutes it.",
                suggested_queries=[f"{sq.question} peer-reviewed study"] if sq else [],
            )
        if finding.confidence == "high" and len(sources) < 2:
            add(
                issue_type=IssueType.OVERCONFIDENCE,
                severity=Severity.MEDIUM,
                finding_ids=[finding.id],
                subquestion_id=finding.subquestion_id,
                description=f"{finding.id} is marked high confidence but rests on a single source.",
                suggested_action="Lower the confidence or corroborate with an independent source.",
            )

    referenced = {cid for f in synthesis.findings for cid in f.contradiction_ids}
    for contradiction in contradictions:
        if contradiction.id not in referenced:
            add(
                issue_type=IssueType.UNREPRESENTED_CONTRADICTION,
                severity=Severity.HIGH if contradiction.severity == "major" else Severity.MEDIUM,
                evidence_ids=contradiction.evidence_ids,
                subquestion_id=contradiction.subquestion_id,
                description=f"Contradiction {contradiction.id} is not reflected in any finding: {contradiction.description}",
                suggested_action="Represent the disagreement and explain under which conditions each claim holds.",
            )

    covered = {f.subquestion_id for f in synthesis.findings}
    for sq in plan.subquestions:
        if sq.id not in covered:
            add(
                issue_type=IssueType.MISSING_COVERAGE,
                severity=Severity.HIGH,
                subquestion_id=sq.id,
                description=f"No finding answers {sq.id}: {sq.question}",
                suggested_action="Research this subquestion further.",
                suggested_queries=[q.query for q in sq.search_queries[:2]],
            )
    return issues


_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


def _merge(
    rule_issues: list[CritiqueIssue], llm_issues: list[CritiqueIssue]
) -> list[CritiqueIssue]:
    """De-duplicate issues that describe the same problem, keeping the *most severe* one
    and the union of suggested queries (a duplicate must never downgrade severity)."""
    merged: dict[tuple[object, ...], CritiqueIssue] = {}
    for issue in [*rule_issues, *llm_issues]:
        key = (issue.issue_type, tuple(sorted(issue.finding_ids)), issue.subquestion_id)
        existing = merged.get(key)
        if existing is None:
            merged[key] = issue
            continue
        stronger, weaker = (
            (issue, existing)
            if _SEVERITY_RANK[issue.severity] > _SEVERITY_RANK[existing.severity]
            else (existing, issue)
        )
        queries = list(dict.fromkeys([*stronger.suggested_queries, *weaker.suggested_queries]))[:4]
        merged[key] = stronger.model_copy(update={"suggested_queries": queries})
    return [issue.model_copy(update={"id": f"I{i + 1}"}) for i, issue in enumerate(merged.values())]


def _verdict(issues: list[CritiqueIssue], llm_verdict: str | None) -> CritiqueVerdict:
    derived = (
        "major_problems"
        if any(i.severity is Severity.CRITICAL for i in issues)
        else (
            "needs_revision" if any(i.severity is Severity.HIGH for i in issues) else "acceptable"
        )
    )
    if llm_verdict and _VERDICT_RANK.get(llm_verdict, 0) > _VERDICT_RANK[derived]:
        return llm_verdict  # type: ignore[return-value]
    return derived  # type: ignore[return-value]


async def review(
    ctx: ResearchContext | None,
    *,
    question: str,
    plan: ResearchPlan,
    synthesis: Synthesis,
    evidence: Mapping[str, Evidence],
    quality: Mapping[str, SourceQuality],
    contradictions: list[Contradiction],
    assessment: EvidenceAssessment | None,
    iteration: int,
    credible_threshold: float,
) -> tuple[Critique, Exception | None]:
    """Return the merged critique and, if the LLM review failed, the error (rules still apply)."""
    rules = rule_based_issues(
        plan=plan,
        synthesis=synthesis,
        evidence=evidence,
        quality=quality,
        contradictions=contradictions,
        credible_threshold=credible_threshold,
    )
    llm_issues: list[CritiqueIssue] = []
    strengths: list[str] = []
    summary = ""
    llm_verdict: str | None = None
    error: Exception | None = None

    if ctx is not None:
        cited_ids = {e for f in synthesis.findings for e in f.evidence_ids}
        cited = [evidence[e] for e in sorted(cited_ids) if e in evidence]
        request = render_block(
            "review_request",
            {
                "question": question,
                "subquestions": [
                    {"id": sq.id, "question": sq.question} for sq in plan.subquestions
                ],
                "coverage": [c.model_dump() for c in assessment.coverage] if assessment else [],
                "overall_assessment": synthesis.overall_assessment,
            },
        )
        try:
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
                        "evidence", evidence_rows(cited, quality, quote_chars=200)
                    ),
                    "contradictions": render_block(
                        "contradictions", [c.model_dump(mode="json") for c in contradictions]
                    ),
                    "policy": UNTRUSTED_CONTENT_POLICY,
                },
                schema=CriticOutput,
                operation="critic",
            )
            valid_findings = {f.id for f in synthesis.findings}
            for item in output.issues:
                llm_issues.append(
                    CritiqueIssue(
                        id="tmp",
                        issue_type=item.issue_type,
                        severity=item.severity,
                        description=truncate(item.description, 600),
                        finding_ids=[f for f in item.finding_ids if f in valid_findings],
                        evidence_ids=[e for e in item.evidence_ids if e in evidence],
                        subquestion_id=item.subquestion_id
                        if item.subquestion_id in plan.subquestion_ids
                        else None,
                        suggested_action=truncate(item.suggested_action, 300),
                        suggested_queries=[q[:300] for q in item.suggested_queries[:3]],
                        origin="llm",
                    )
                )
            strengths, summary, llm_verdict = output.strengths[:6], output.summary, output.verdict
        except Exception as exc:
            error = exc

    issues = _merge(rules, llm_issues)
    if not summary:
        summary = f"{len(issues)} issue(s) found by automated checks."
    return (
        Critique(
            iteration=iteration,
            issues=issues,
            strengths=strengths,
            summary=truncate(summary, 1500),
            verdict=_verdict(issues, llm_verdict),
        ),
        error,
    )
