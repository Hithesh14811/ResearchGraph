"""Per-round orchestration: query generation (before the fan-out) and source/evidence
evaluation (after the fan-in)."""

from __future__ import annotations

from datetime import date

from langgraph.runtime import Runtime

from researchgraph.agents.query_writer import QueryTarget, fallback_queries, write_queries
from researchgraph.core.errors import LLMCallError, StructuredOutputError
from researchgraph.core.text import normalize_for_match
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.instrumentation import emit, instrumented, workflow_error
from researchgraph.graph.state import ResearchState
from researchgraph.schemas.events import EventType
from researchgraph.schemas.research import ResearchTask
from researchgraph.schemas.sources import QualityTier
from researchgraph.services.coverage import assess_coverage
from researchgraph.services.quality_gate import budget_remaining
from researchgraph.services.source_quality import score_source


def _task_budget(ctx: ResearchContext, remaining: int, n_tasks: int) -> int:
    s = ctx.settings
    per_task_cap = s.max_queries_per_subquestion + 1 + s.max_sources_per_task
    return max(0, min(per_task_cap, remaining // max(1, n_tasks)))


@instrumented("query_generation", start="Generating search queries...")
async def query_generation_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    ctx = runtime.context
    plan, limits = state["plan"], state["limits"]
    iteration = state.get("iteration", 0) + 1
    gaps = state.get("missing_evidence") or []

    if iteration > 1 and gaps:
        targets = [
            QueryTarget(sq, gap)
            for gap in gaps
            if (sq := plan.subquestion(gap.subquestion_id)) is not None
        ]
        reason_by_sq = {gap.subquestion_id: gap.reason for gap in gaps}
    else:
        targets = [QueryTarget(sq) for sq in plan.subquestions]
        reason_by_sq = {}

    history = {normalize_for_match(q) for q in state.get("search_history", [])}
    errors = []
    try:
        generated = await write_queries(
            ctx, objective=plan.objective, targets=targets, history=history
        )
    except (LLMCallError, StructuredOutputError) as exc:
        errors.append(workflow_error("query_generation", exc))
        emit(EventType.WARNING, "Query writer unavailable; using planned queries.")
        generated = {}

    remaining = limits.max_tool_calls - state.get("tool_calls_used", 0)
    budget = _task_budget(ctx, remaining, len(targets))
    tasks: list[ResearchTask] = []
    if budget >= 2:
        for target in targets:
            sq = target.subquestion
            queries = generated.get(sq.id) or fallback_queries(
                target, history, ctx.settings.max_queries_per_subquestion
            )
            if not queries:
                continue
            tasks.append(
                ResearchTask(
                    task_id=f"T{iteration}-{sq.id}",
                    subquestion_id=sq.id,
                    question=sq.question,
                    queries=queries,
                    iteration=iteration,
                    reason=reason_by_sq.get(sq.id, "initial"),
                    focus=target.gap.description if target.gap else None,
                    tool_budget=budget,
                    preferred_sources=sq.preferred_sources,
                )
            )
    else:
        emit(EventType.WARNING, "Tool-call budget exhausted; no further searches will run.")

    label = "Initial research round" if iteration == 1 else f"Research round {iteration} (targeted)"
    emit(
        EventType.PROGRESS,
        f"{label}: dispatching {len(tasks)} parallel research workers.",
        iteration=iteration,
        workers=len(tasks),
        subquestions=[t.subquestion_id for t in tasks],
    )
    return {
        "iteration": iteration,
        "pending_tasks": tasks,
        "missing_evidence": [],
        "search_history": [q.query for t in tasks for q in t.queries],
        "errors": errors,
    }


@instrumented("source_evaluation", start="Evaluating sources and evidence...")
async def source_evaluation_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    s = runtime.context.settings
    plan, limits = state["plan"], state["limits"]
    sources = state.get("sources") or {}
    evidence = state.get("evidence") or {}

    by_source: dict[str, list] = {}
    for item in evidence.values():
        by_source.setdefault(item.source_id, []).append(item)
    sq_text = {
        sq.id: f"{sq.question} {' '.join(sq.information_requirements)}" for sq in plan.subquestions
    }
    today = date.today()
    quality = {
        sid: score_source(
            source,
            subquestion_texts=[sq_text[x] for x in source.subquestion_ids if x in sq_text]
            or [state["question"]],
            evidence=by_source.get(sid, []),
            today=today,
            half_life_years=s.recency_half_life_years,
            high_threshold=s.high_quality_source_threshold,
            credible_threshold=limits.credible_source_threshold,
        )
        for sid, source in sources.items()
    }
    tiers = {tier: sum(1 for q in quality.values() if q.tier is tier) for tier in QualityTier}
    emit(
        EventType.PROGRESS,
        f"Evaluated {len(quality)} sources: {tiers[QualityTier.HIGH]} high, {tiers[QualityTier.MEDIUM]} medium, "
        f"{tiers[QualityTier.LOW]} low quality.",
        sources=len(quality),
        high_quality=tiers[QualityTier.HIGH],
        evidence=len(evidence),
    )

    assessment = assess_coverage(
        plan, evidence, quality, iteration=state.get("iteration", 0), limits=limits
    )
    covered = sum(1 for c in assessment.coverage if c.sufficient)
    emit(
        EventType.PROGRESS,
        f"Evidence coverage: {covered}/{len(assessment.coverage)} subquestions sufficiently supported.",
    )

    missing = []
    if not assessment.sufficient:
        weak = ", ".join(assessment.insufficient_ids)
        if budget_remaining(
            iteration=state.get("iteration", 0),
            tool_calls_used=state.get("tool_calls_used", 0),
            limits=limits,
        ):
            missing = assessment.gaps
            emit(
                EventType.PROGRESS,
                f"Insufficient evidence for {weak} — performing additional research.",
                gaps=assessment.insufficient_ids,
            )
        else:
            emit(
                EventType.WARNING,
                f"Evidence for {weak} is still thin but the research budget is exhausted; continuing with limitations.",
            )
    return {
        "source_quality": quality,
        "evidence_assessment": assessment,
        "missing_evidence": missing,
    }
