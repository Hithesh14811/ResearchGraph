"""Deterministic evaluation metrics, computed from a run's final graph state and event log.

These metrics are *measurements of the pipeline's behaviour* (grounding, coverage, source
quality, cost, latency). They are fully reproducible and require no model. Answer quality
as judged by a model lives separately in ``judge.py``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from researchgraph.schemas.events import EventType, WorkflowEvent
from researchgraph.schemas.sources import QualityTier


class ExpectedCharacteristics(BaseModel):
    """What a good run on a dataset question should exhibit (from the dataset file)."""

    min_sources: int = 3
    min_evidence: int = 5
    min_subquestion_coverage: float = 0.8
    max_subquestion_coverage: float = 1.0
    min_citation_coverage: float = 1.0
    max_unsupported_claim_rate: float = 0.2
    must_cite_primary_research: bool = True
    expects_contradiction: bool = False
    expected_aspects: list[list[str]] = Field(
        default_factory=list,
        description="Each aspect is a list of synonyms; covered if any appears.",
    )
    expect_limitations: bool = False


class RunMetrics(BaseModel):
    status: str
    # grounding / citation
    citation_coverage: float
    unsupported_claim_rate: float
    first_pass_citation_precision: float | None
    quote_verification_rate: float | None
    # evidence & sources
    evidence_quality: float
    high_quality_source_share: float
    primary_research_cited: int
    evidence_relevance: float
    evidence_items: int
    sources_found: int
    sources_cited: int
    contradictions: int
    # completeness
    subquestion_coverage: float
    aspect_coverage: float | None
    research_completeness: float
    # process
    research_iterations: int
    quality_gate_score: float | None
    critic_issues: int
    recoverable_errors: int
    # cost & latency
    latency_seconds: float
    llm_calls: int
    total_tokens: int
    estimated_cost_usd: float | None
    tool_calls: int


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _event_data(events: Sequence[WorkflowEvent], node: str) -> list[dict[str, Any]]:
    return [e.data for e in events if e.node == node and e.type is EventType.PROGRESS]


def aspect_coverage(markdown: str, aspects: Sequence[Sequence[str]]) -> float | None:
    if not aspects:
        return None
    text = markdown.lower()
    hits = sum(
        1
        for synonyms in aspects
        if any(re.search(rf"\b{re.escape(s.lower())}", text) for s in synonyms)
    )
    return round(hits / len(aspects), 4)


def compute_run_metrics(
    state: Mapping[str, Any],
    events: Sequence[WorkflowEvent],
    *,
    usage: Mapping[str, Any],
    latency_seconds: float,
    expected: ExpectedCharacteristics | None = None,
) -> RunMetrics:
    report = state.get("final_report")
    evidence = state.get("evidence") or {}
    sources = state.get("sources") or {}
    quality = state.get("source_quality") or {}
    plan = state.get("plan")
    analysis = state.get("analysis")
    status = state.get("status")

    cited_sources = [c.source_id for c in report.bibliography] if report else []
    cited_quality = [quality[s] for s in cited_sources if s in quality]
    cited_evidence_ids = (
        {e for c in report.draft.all_claims() for e in c.evidence_ids} if report else set()
    )

    extraction = _event_data(events, "evidence_extraction")
    extracted = sum(int(d.get("evidence", 0)) for d in extraction)
    rejected = sum(int(d.get("rejected", 0)) for d in extraction)
    verification = _event_data(events, "citation_verification")
    first_pass = None
    if verification and verification[0].get("total"):
        first_pass = round(verification[0]["verified"] / verification[0]["total"], 4)

    covered = {f.subquestion_id for f in analysis.findings} if analysis else set()
    sq_coverage = len(covered) / len(plan.subquestions) if plan and plan.subquestions else 0.0
    aspects = aspect_coverage(
        report.markdown if report else "", expected.expected_aspects if expected else []
    )
    completeness = _mean([sq_coverage, aspects]) if aspects is not None else round(sq_coverage, 4)

    gate = state.get("quality")
    critique = state.get("critique")
    return RunMetrics(
        status=getattr(status, "value", str(status)),
        citation_coverage=report.metrics.citation_coverage if report else 0.0,
        unsupported_claim_rate=report.metrics.unsupported_claim_rate if report else 1.0,
        first_pass_citation_precision=first_pass,
        quote_verification_rate=round(extracted / (extracted + rejected), 4)
        if extracted + rejected
        else None,
        evidence_quality=_mean([q.overall for q in cited_quality]),
        high_quality_source_share=_mean(
            [1.0 if q.tier is QualityTier.HIGH else 0.0 for q in cited_quality]
        ),
        primary_research_cited=sum(
            1 for q in cited_quality if q.source_type.value == "primary_research"
        ),
        evidence_relevance=_mean(
            [evidence[e].relevance for e in cited_evidence_ids if e in evidence]
        ),
        evidence_items=len(evidence),
        sources_found=len(sources),
        sources_cited=len(cited_sources),
        contradictions=len(state.get("contradictions") or []),
        subquestion_coverage=round(sq_coverage, 4),
        aspect_coverage=aspects,
        research_completeness=completeness,
        research_iterations=int(state.get("iteration") or 0),
        quality_gate_score=gate.score if gate else None,
        critic_issues=len(critique.issues) if critique else 0,
        recoverable_errors=len(state.get("errors") or []),
        latency_seconds=round(latency_seconds, 3),
        llm_calls=int(usage.get("llm_calls", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
        estimated_cost_usd=usage.get("estimated_cost_usd"),
        tool_calls=int(state.get("tool_calls_used") or 0),
    )


def check_expectations(
    metrics: RunMetrics, expected: ExpectedCharacteristics, report_markdown: str
) -> list[CheckResult]:
    def check(name: str, passed: bool, detail: str) -> CheckResult:
        return CheckResult(name=name, passed=passed, detail=detail)

    checks = [
        check("completed", metrics.status == "completed", f"status={metrics.status}"),
        check(
            "min_sources",
            metrics.sources_found >= expected.min_sources,
            f"{metrics.sources_found} >= {expected.min_sources}",
        ),
        check(
            "min_evidence",
            metrics.evidence_items >= expected.min_evidence,
            f"{metrics.evidence_items} >= {expected.min_evidence}",
        ),
        check(
            "subquestion_coverage",
            metrics.subquestion_coverage >= expected.min_subquestion_coverage,
            f"{metrics.subquestion_coverage:.2f} >= {expected.min_subquestion_coverage:.2f}",
        ),
        check(
            "no_off_topic_findings",
            metrics.subquestion_coverage <= expected.max_subquestion_coverage,
            f"{metrics.subquestion_coverage:.2f} <= {expected.max_subquestion_coverage:.2f}",
        ),
        check(
            "citation_coverage",
            metrics.citation_coverage >= expected.min_citation_coverage,
            f"{metrics.citation_coverage:.2f} >= {expected.min_citation_coverage:.2f}",
        ),
        check(
            "unsupported_claim_rate",
            metrics.unsupported_claim_rate <= expected.max_unsupported_claim_rate,
            f"{metrics.unsupported_claim_rate:.3f} <= {expected.max_unsupported_claim_rate:.3f}",
        ),
    ]
    if expected.must_cite_primary_research:
        checks.append(
            check(
                "cites_primary_research",
                metrics.primary_research_cited > 0,
                f"{metrics.primary_research_cited} primary sources cited",
            )
        )
    if expected.expects_contradiction:
        checks.append(
            check(
                "detects_contradiction",
                metrics.contradictions > 0,
                f"{metrics.contradictions} contradictions",
            )
        )
    if metrics.aspect_coverage is not None:
        checks.append(
            check(
                "aspect_coverage",
                metrics.aspect_coverage >= 0.6,
                f"{metrics.aspect_coverage:.2f} >= 0.60",
            )
        )
    if expected.expect_limitations:
        has_limitations = "## Limitations" in report_markdown
        checks.append(
            check("states_limitations", has_limitations, "report has a Limitations section")
        )
    return checks
