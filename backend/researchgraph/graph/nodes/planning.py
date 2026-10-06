"""Intake, planning and the human plan-review checkpoint."""

from __future__ import annotations

from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from researchgraph.agents.planner import create_plan, heuristic_plan
from researchgraph.core.errors import LLMCallError, StructuredOutputError
from researchgraph.core.text import sanitize_user_text
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.instrumentation import emit, instrumented, workflow_error
from researchgraph.graph.state import ResearchState
from researchgraph.schemas.common import RunStatus, WorkflowError
from researchgraph.schemas.events import EventType
from researchgraph.schemas.research import PlanDecision, ResearchPlan, RunLimits

MIN_QUESTION_WORDS = 4


def default_limits(ctx: ResearchContext) -> RunLimits:
    s = ctx.settings
    return RunLimits(
        max_iterations=s.max_research_iterations,
        max_tool_calls=s.max_tool_calls,
        max_citation_repairs=s.max_citation_repair_attempts,
        min_evidence_per_subquestion=s.min_evidence_per_subquestion,
        quality_threshold=s.quality_gate_threshold,
        credible_source_threshold=s.credible_source_threshold,
    )


def validate_question(question: str) -> list[str]:
    problems = []
    if len(question.split()) < MIN_QUESTION_WORDS:
        problems.append(f"Question is too short (need at least {MIN_QUESTION_WORDS} words).")
    if not any(ch.isalpha() for ch in question):
        problems.append("Question contains no words.")
    return problems


@instrumented("intake", start="Validating research question...")
async def intake_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    ctx = runtime.context
    question = sanitize_user_text(
        state.get("original_question", ""), max_chars=ctx.settings.max_question_chars
    )
    problems = validate_question(question)
    if problems:
        emit(EventType.ERROR, "Research question rejected: " + " ".join(problems))
        return {
            "question": question,
            "status": RunStatus.REJECTED,
            "errors": [
                WorkflowError(
                    node="intake",
                    error_type="invalid_question",
                    message=" ".join(problems),
                    recoverable=False,
                )
            ],
        }
    emit(EventType.PROGRESS, "Research question accepted.", characters=len(question))
    return {
        "question": question,
        "status": RunStatus.RUNNING,
        "limits": state.get("limits") or default_limits(ctx),
        "iteration": 0,
        "citation_repairs": 0,
    }


@instrumented("planner", start="Planning research...")
async def planner_node(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    ctx = runtime.context
    errors = []
    previous = state.get("plan")
    try:
        plan = await create_plan(ctx, state["question"], feedback=state.get("plan_feedback"))
    except (LLMCallError, StructuredOutputError) as exc:
        errors.append(workflow_error("planner", exc))
        emit(EventType.WARNING, "Planner model unavailable; using the deterministic fallback plan.")
        plan = heuristic_plan(
            state["question"],
            max_subquestions=ctx.settings.max_subquestions,
            max_queries=ctx.settings.max_queries_per_subquestion,
        )
    if previous is not None:
        plan = plan.model_copy(update={"version": previous.version + 1})
    emit(
        EventType.PROGRESS,
        f"Generated {len(plan.subquestions)} subquestions.",
        subquestions=len(plan.subquestions),
        queries=sum(len(sq.search_queries) for sq in plan.subquestions),
    )
    return {"plan": plan, "errors": errors}


@instrumented("plan_review", start="Reviewing research plan...")
async def plan_review_node(
    state: ResearchState, runtime: Runtime[ResearchContext]
) -> Command[Literal["query_generation", "planner", "plan_review", "__end__"]]:
    """Human-in-the-loop checkpoint.

    ``interrupt()`` persists the run in the checkpointer and returns control to the caller;
    the run resumes (possibly after a server restart) when ``Command(resume=decision)`` is
    sent by the API's approve / edit-plan / cancel endpoints.
    """
    plan = state["plan"]
    if state.get("auto_approve"):
        emit(EventType.PROGRESS, "Plan auto-approved.")
        return Command(goto="query_generation", update={"status": RunStatus.RUNNING})

    raw = interrupt({"type": "plan_review", "plan": plan.model_dump(mode="json")})
    try:
        decision = PlanDecision.model_validate(raw)
    except ValidationError as exc:
        return Command(goto="plan_review", update={"errors": [workflow_error("plan_review", exc)]})

    if decision.action == "cancel":
        emit(EventType.RUN_STATUS, "Research cancelled during plan review.")
        return Command(goto="__end__", update={"status": RunStatus.CANCELLED})
    if decision.action == "replan":
        emit(EventType.PROGRESS, "Regenerating plan with reviewer feedback.")
        feedback = [*state.get("plan_feedback", []), decision.feedback or "Revise the plan."]
        return Command(goto="planner", update={"plan_feedback": feedback})
    if decision.action == "edit" and decision.plan is not None:
        edited = ResearchPlan.model_validate(
            {**decision.plan.model_dump(), "version": plan.version + 1}
        )
        emit(EventType.PROGRESS, f"Plan edited by reviewer (version {edited.version}).")
        if not decision.approve:
            return Command(goto="plan_review", update={"plan": edited})
        emit(EventType.PROGRESS, "Edited plan approved.")
        return Command(
            goto="query_generation", update={"plan": edited, "status": RunStatus.RUNNING}
        )

    emit(EventType.PROGRESS, "Plan approved by reviewer.")
    return Command(goto="query_generation", update={"status": RunStatus.RUNNING})
