"""Analysis and review: contradiction detection, synthesis, critic and the quality gate."""

from __future__ import annotations

from collections import Counter

from langgraph.runtime import Runtime

from researchgraph.agents.contradiction_judge import judge_contradictions
from researchgraph.agents.critic import review
from researchgraph.agents.synthesizer import fallback_synthesis, synthesize
from researchgraph.core.errors import LLMCallError, StructuredOutputError
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.instrumentation import emit, instrumented, workflow_error
from researchgraph.graph.state import ResearchState
from researchgraph.schemas.events import EventType
from researchgraph.schemas.review import Severity
from researchgraph.services.contradictions import find_candidate_pairs
from researchgraph.services.quality_gate import assess_quality

MAX_CONTRADICTION_PAIRS = 12


@instrumented("contradiction_detection", start="Checking for conflicting evidence...")
async def contradiction_detection_node(
    state: ResearchState, runtime: Runtime[ResearchContext]
) -> dict:
    evidence = state.get("evidence") or {}
    pairs = find_candidate_pairs(list(evidence.values()), max_pairs=MAX_CONTRADICTION_PAIRS)
    if not pairs:
        emit(EventType.PROGRESS, "No conflicting-evidence candidates found.")
        return {"contradictions": []}
    errors = []
    try:
        contradictions = await judge_contradictions(
            runtime.context, pairs, state.get("sources") or {}
        )
    except (LLMCallError, StructuredOutputError) as exc:
        errors.append(workflow_error("contradiction_detection", exc))
        emit(
            EventType.WARNING, "Contradiction checker unavailable; conflicts were not adjudicated."
        )
        contradictions = []
    emit(
        EventType.PROGRESS,
        f"Checked {len(pairs)} candidate pairs; {len(contradictions)} contradiction(s) confirmed.",
        candidates=len(pairs),
        contradictions=len(contradictions),
    )
    return {"contradictions": contradictions, "errors": errors}


@instrumented("synthesis", start="Analysing evidence...")
async def synthesis_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    plan = state["plan"]
    evidence = state.get("evidence") or {}
    quality = state.get("source_quality") or {}
    errors = []
    try:
        analysis = await synthesize(
            runtime.context,
            question=state["question"],
            plan=plan,
            evidence=evidence,
            quality=quality,
            contradictions=state.get("contradictions") or [],
            iteration=state.get("iteration", 0),
        )
    except (LLMCallError, StructuredOutputError) as exc:
        errors.append(workflow_error("synthesis", exc))
        emit(
            EventType.WARNING,
            "Analysis model unavailable; building findings from the strongest evidence.",
        )
        analysis = fallback_synthesis(plan, evidence, quality, state.get("iteration", 0))
    covered = len({f.subquestion_id for f in analysis.findings})
    emit(
        EventType.PROGRESS,
        f"Synthesised {len(analysis.findings)} findings across {covered}/{len(plan.subquestions)} subquestions"
        + (
            f" ({analysis.dropped_finding_count} unsupported finding(s) dropped)"
            if analysis.dropped_finding_count
            else ""
        )
        + ".",
        findings=len(analysis.findings),
    )
    return {"analysis": analysis, "errors": errors}


@instrumented("critic", start="Critic reviewing findings...")
async def critic_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    critique, error = await review(
        runtime.context,
        question=state["question"],
        plan=state["plan"],
        synthesis=state["analysis"],
        evidence=state.get("evidence") or {},
        quality=state.get("source_quality") or {},
        contradictions=state.get("contradictions") or [],
        assessment=state.get("evidence_assessment"),
        iteration=state.get("iteration", 0),
        credible_threshold=state["limits"].credible_source_threshold,
    )
    errors = []
    if error is not None:
        errors.append(workflow_error("critic", error))
        emit(EventType.WARNING, "Critic model unavailable; applied rule-based review only.")
    by_type = Counter(issue.issue_type.value.replace("_", " ") for issue in critique.issues)
    summary = ", ".join(f"{n} {kind}" for kind, n in by_type.most_common(4)) or "no issues"
    serious = critique.count(Severity.HIGH, Severity.CRITICAL)
    emit(
        EventType.PROGRESS,
        f"Critic identified {len(critique.issues)} issue(s) ({serious} high/critical): {summary}. Verdict: {critique.verdict.replace('_', ' ')}.",
        issues=len(critique.issues),
        serious=serious,
        verdict=critique.verdict,
    )
    return {"critique": critique, "critique_history": [critique], "errors": errors}


@instrumented("quality_gate", start="Applying quality gate...")
async def quality_gate_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    assessment = assess_quality(
        plan=state["plan"],
        synthesis=state["analysis"],
        critique=state["critique"],
        evidence=state.get("evidence") or {},
        quality=state.get("source_quality") or {},
        contradictions=state.get("contradictions") or [],
        iteration=state.get("iteration", 0),
        tool_calls_used=state.get("tool_calls_used", 0),
        limits=state["limits"],
    )
    verdict = {
        "proceed": "passed — proceeding to report writing",
        "research_more": f"failed — targeted research on {', '.join(g.subquestion_id for g in assessment.gaps)}",
        "proceed_with_limitations": "not met, but budget exhausted — writing report with explicit limitations",
    }[assessment.decision]
    emit(
        EventType.PROGRESS,
        f"Quality score {assessment.score:.2f} (threshold {assessment.threshold:.2f}): {verdict}.",
        score=assessment.score,
        decision=assessment.decision,
        components=assessment.components,
    )
    missing = assessment.gaps if assessment.decision == "research_more" else []
    return {"quality": assessment, "quality_history": [assessment], "missing_evidence": missing}
