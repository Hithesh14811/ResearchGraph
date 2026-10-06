"""Report writing, citation verification/repair and final review."""

from __future__ import annotations

from langgraph.runtime import Runtime

from researchgraph.agents.writer import fallback_draft, write_report
from researchgraph.core.errors import LLMCallError, StructuredOutputError
from researchgraph.core.text import truncate
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.instrumentation import emit, instrumented, workflow_error
from researchgraph.graph.state import ResearchState
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.events import EventType
from researchgraph.schemas.report import FinalReport, ReportMetrics
from researchgraph.services.citations import (
    build_bibliography,
    dedupe_claims,
    repair_citations,
    verify_citations,
)
from researchgraph.services.report_renderer import render_executive_summary, render_markdown


@instrumented("report_writer", start="Writing report...")
async def report_writer_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    plan, analysis, quality = state["plan"], state["analysis"], state.get("quality")
    errors = []
    try:
        draft = await write_report(
            runtime.context,
            question=state["question"],
            plan=plan,
            synthesis=analysis,
            evidence=state.get("evidence") or {},
            quality_scores=state.get("source_quality") or {},
            contradictions=state.get("contradictions") or [],
            critique=state.get("critique"),
            quality=quality,
        )
    except (LLMCallError, StructuredOutputError) as exc:
        errors.append(workflow_error("report_writer", exc))
        emit(EventType.WARNING, "Report writer unavailable; assembling the report from findings.")
        draft = fallback_draft(
            question=state["question"], plan=plan, synthesis=analysis, quality=quality
        )

    if quality is not None and quality.decision == "proceed_with_limitations":
        extra = [
            f"The quality gate was not passed (score {quality.score:.2f}, threshold {quality.threshold:.2f}) "
            "and the research budget was exhausted; conclusions are provisional."
        ]
        extra += [issue for issue in quality.blocking_issues if issue not in draft.limitations]
        draft = draft.model_copy(update={"limitations": [*draft.limitations, *extra]})
    emit(
        EventType.PROGRESS,
        f"Drafted report: {len(draft.sections)} sections, {len(draft.all_claims())} citable sentences.",
        sections=len(draft.sections),
        claims=len(draft.all_claims()),
    )
    return {"draft_report": draft, "citation_repairs": 0, "errors": errors}


@instrumented("citation_verification", start="Verifying citations...")
async def citation_verification_node(
    state: ResearchState, runtime: Runtime[ResearchContext]
) -> dict:
    check = verify_citations(
        state["draft_report"],
        state.get("evidence") or {},
        attempt=state.get("citation_repairs", 0),
        relevance_threshold=runtime.context.settings.citation_relevance_threshold,
    )
    problems = ", ".join(sorted({i.problem.replace("_", " ") for i in check.issues})) or "none"
    emit(
        EventType.PROGRESS,
        f"Verified citations: {check.verified_claims}/{check.total_claims} sentences supported; issues: {problems}.",
        verified=check.verified_claims,
        total=check.total_claims,
        issues=len(check.issues),
    )
    return {"citation_check": check}


@instrumented("citation_repair", start="Repairing citations...")
async def citation_repair_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    outcome = repair_citations(
        state["draft_report"],
        state["citation_check"],
        state.get("evidence") or {},
        relevance_threshold=runtime.context.settings.citation_relevance_threshold,
    )
    emit(
        EventType.PROGRESS,
        f"Citation repair: {outcome.reattributed} sentence(s) re-attributed, {outcome.removed} removed as unsupported.",
        reattributed=outcome.reattributed,
        removed=outcome.removed,
    )
    return {"draft_report": outcome.draft, "citation_repairs": state.get("citation_repairs", 0) + 1}


@instrumented("final_review", start="Final quality review...")
async def final_review_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    """Last line of defence: nothing unsupported is published, then render and measure."""
    s = runtime.context.settings
    evidence = state.get("evidence") or {}
    sources = state.get("sources") or {}
    quality_scores = state.get("source_quality") or {}
    draft, duplicates = dedupe_claims(state["draft_report"])
    notes: list[str] = [f"Removed {duplicates} duplicated sentence(s)."] if duplicates else []

    check = verify_citations(
        draft,
        evidence,
        attempt=state.get("citation_repairs", 0),
        relevance_threshold=s.citation_relevance_threshold,
    )
    if not check.passed:
        draft = repair_citations(
            draft, check, evidence, relevance_threshold=s.citation_relevance_threshold
        ).draft
        notes.append(
            f"{len(check.issues)} sentence(s) still failed verification after repair and were removed."
        )
        check = verify_citations(
            draft,
            evidence,
            attempt=check.attempt + 1,
            relevance_threshold=s.citation_relevance_threshold,
        )

    numbers, bibliography = build_bibliography(draft, evidence, sources, quality_scores)
    claims = draft.all_claims()
    cited_quality = [c.quality for c in bibliography if c.quality is not None]
    total = len(claims) + len(draft.removed_claims)
    metrics = ReportMetrics(
        total_claims=len(claims),
        cited_claims=check.cited_claims,
        citation_coverage=check.coverage,
        unsupported_claim_rate=round(len(draft.removed_claims) / total, 4) if total else 0.0,
        removed_claims=len(draft.removed_claims),
        sources_cited=len(bibliography),
        mean_cited_source_quality=round(sum(cited_quality) / len(cited_quality), 3)
        if cited_quality
        else 0.0,
        evidence_items=len(evidence),
        research_iterations=state.get("iteration", 0),
    )
    quality = state.get("quality")
    if quality is not None:
        notes.append(
            f"Quality gate {'passed' if quality.passed else 'not passed'} (score {quality.score:.2f})."
        )
    if not bibliography:
        notes.append("No sources could be cited; the report contains no verified claims.")

    markdown = render_markdown(
        question=state["question"],
        draft=draft,
        evidence=evidence,
        numbers=numbers,
        bibliography=bibliography,
        contradictions=state.get("contradictions") or [],
        quality=quality,
        metrics=metrics,
    )
    report = FinalReport(
        research_id=state["research_id"],
        question=state["question"],
        title=draft.title,
        executive_summary_markdown=render_executive_summary(draft, evidence, numbers),
        markdown=truncate(
            markdown, s.max_report_chars, suffix="\n\n_[Report truncated at size limit]_"
        ),
        draft=draft,
        bibliography=bibliography,
        metrics=metrics,
        review_notes=notes,
    )
    emit(
        EventType.PROGRESS,
        f"Research complete: {metrics.total_claims} verified sentences citing {metrics.sources_cited} sources.",
        citation_coverage=metrics.citation_coverage,
        sources_cited=metrics.sources_cited,
    )
    return {
        "final_report": report,
        "draft_report": draft,
        "citation_check": check,
        "status": RunStatus.COMPLETED,
    }
