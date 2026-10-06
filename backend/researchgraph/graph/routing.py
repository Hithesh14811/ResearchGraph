"""Conditional-edge functions.

All routing is a pure function of graph state (limits are stored in state), so every branch
of the workflow can be unit-tested without running a model.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Send

from researchgraph.graph.state import ResearchState, WorkerInput, WorkerState
from researchgraph.schemas.common import RunStatus
from researchgraph.services.quality_gate import budget_remaining


def within_budget(state: ResearchState) -> bool:
    limits = state.get("limits")
    if limits is None:
        return False
    return budget_remaining(
        iteration=state.get("iteration", 0),
        tool_calls_used=state.get("tool_calls_used", 0),
        limits=limits,
    )


def route_after_intake(state: ResearchState) -> Literal["planner", "__end__"]:
    return "__end__" if state.get("status") is RunStatus.REJECTED else "planner"  # "__end__" == END


def dispatch_research(state: ResearchState) -> list[Send] | Literal["source_evaluation"]:
    """Fan out one research worker per task (map step). No tasks → skip straight to evaluation."""
    tasks = state.get("pending_tasks") or []
    if not tasks:
        return "source_evaluation"
    known = {sid: s for sid, s in (state.get("sources") or {}).items() if s.status == "processed"}
    return [
        Send(
            "research_worker",
            WorkerInput(
                research_id=state["research_id"],
                question=state["question"],
                task=task,
                known_sources=known,
            ),
        )
        for task in tasks
    ]


def route_after_evidence(
    state: ResearchState,
) -> Literal["query_generation", "contradiction_detection"]:
    """Evidence gate: insufficient evidence + remaining budget → another research round."""
    assessment = state.get("evidence_assessment")
    if (
        assessment is not None
        and not assessment.sufficient
        and state.get("missing_evidence")
        and within_budget(state)
    ):
        return "query_generation"
    return "contradiction_detection"


def route_after_quality_gate(state: ResearchState) -> Literal["query_generation", "report_writer"]:
    """Quality gate: critic-driven targeted research, or proceed to writing."""
    quality = state.get("quality")
    if (
        quality is not None
        and quality.decision == "research_more"
        and state.get("missing_evidence")
        and within_budget(state)
    ):
        return "query_generation"
    return "report_writer"


def route_after_citation_check(state: ResearchState) -> Literal["citation_repair", "final_review"]:
    check = state.get("citation_check")
    limits = state.get("limits")
    max_repairs = limits.max_citation_repairs if limits else 0
    if check is not None and not check.passed and state.get("citation_repairs", 0) < max_repairs:
        return "citation_repair"
    return "final_review"


def route_search_agent(state: WorkerState) -> Literal["execute_tools", "source_processing"]:
    return "source_processing" if state.get("search_done") else "execute_tools"
