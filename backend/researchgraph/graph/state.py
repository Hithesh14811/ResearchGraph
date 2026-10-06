"""Typed graph state and the reducers that make parallel updates merge safely.

Fields written by parallel research workers use reducers:
* ``sources`` / ``evidence`` / ``source_quality`` are dicts keyed by content-addressed IDs,
  so two workers that find the same URL or quote converge on one entry (merge is
  commutative and idempotent);
* counters (``tool_calls_used``) are summed; logs (``errors``, ``node_metrics``) appended.
Every other field has last-writer-wins semantics and is written by exactly one node.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from researchgraph.schemas.common import NodeMetric, RunStatus, WorkflowError
from researchgraph.schemas.evidence import Contradiction, Evidence, Passage
from researchgraph.schemas.report import FinalReport, ReportDraft, Synthesis
from researchgraph.schemas.research import ResearchGap, ResearchPlan, ResearchTask, RunLimits
from researchgraph.schemas.review import (
    CitationCheck,
    Critique,
    EvidenceAssessment,
    QualityAssessment,
)
from researchgraph.schemas.sources import Source, SourceQuality

_STATUS_RANK = {"failed": 0, "discovered": 1, "processed": 2}


def merge_sources(
    left: dict[str, Source] | None, right: dict[str, Source] | None
) -> dict[str, Source]:
    """Union by source ID; prefer the processed copy and union subquestion links."""
    merged = dict(left or {})
    for source_id, incoming in (right or {}).items():
        existing = merged.get(source_id)
        if existing is None:
            merged[source_id] = incoming
            continue
        preferred = (
            incoming if _STATUS_RANK[incoming.status] > _STATUS_RANK[existing.status] else existing
        )
        links = list(dict.fromkeys([*existing.subquestion_ids, *incoming.subquestion_ids]))
        merged[source_id] = preferred.model_copy(update={"subquestion_ids": links})
    return merged


def merge_by_id[V](left: dict[str, V] | None, right: dict[str, V] | None) -> dict[str, V]:
    return {**(left or {}), **(right or {})}


def append_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    return list(dict.fromkeys([*(left or []), *(right or [])]))


class ResearchState(TypedDict, total=False):
    # --- identity & input ---------------------------------------------------------------
    research_id: str
    original_question: str
    question: str
    auto_approve: bool
    limits: RunLimits
    status: RunStatus
    # --- planning (human-in-the-loop) --------------------------------------------------------
    plan: ResearchPlan
    plan_feedback: list[str]
    # --- research loop -------------------------------------------------------------------------
    iteration: int
    pending_tasks: list[ResearchTask]
    missing_evidence: list[ResearchGap]
    tool_calls_used: Annotated[int, operator.add]
    search_history: Annotated[list[str], append_unique]
    sources: Annotated[dict[str, Source], merge_sources]
    source_quality: Annotated[dict[str, SourceQuality], merge_by_id]
    evidence: Annotated[dict[str, Evidence], merge_by_id]
    evidence_assessment: EvidenceAssessment
    # --- analysis & review -----------------------------------------------------------------------
    contradictions: list[Contradiction]
    analysis: Synthesis
    critique: Critique
    critique_history: Annotated[list[Critique], operator.add]
    quality: QualityAssessment
    quality_history: Annotated[list[QualityAssessment], operator.add]
    # --- reporting ---------------------------------------------------------------------------------
    draft_report: ReportDraft
    citation_check: CitationCheck
    citation_repairs: int
    final_report: FinalReport
    # --- diagnostics ---------------------------------------------------------------------------------
    errors: Annotated[list[WorkflowError], operator.add]
    node_metrics: Annotated[list[NodeMetric], operator.add]


class WorkerInput(TypedDict):
    """Payload of each ``Send`` that fans a task out to a research worker."""

    research_id: str
    question: str
    task: ResearchTask
    known_sources: dict[str, Source]


class WorkerState(TypedDict, total=False):
    research_id: str
    question: str
    task: ResearchTask
    known_sources: dict[str, Source]
    messages: Annotated[list[AnyMessage], add_messages]
    search_rounds: int
    search_done: bool
    scope_source_ids: list[str]
    passages: list[Passage]
    sources: Annotated[dict[str, Source], merge_sources]
    evidence: Annotated[dict[str, Evidence], merge_by_id]
    tool_calls_used: Annotated[int, operator.add]
    search_history: Annotated[list[str], append_unique]
    errors: Annotated[list[WorkflowError], operator.add]
    node_metrics: Annotated[list[NodeMetric], operator.add]


class WorkerOutput(TypedDict, total=False):
    """Only these keys flow back to the parent graph (merged by the parent's reducers)."""

    sources: dict[str, Source]
    evidence: dict[str, Evidence]
    tool_calls_used: int
    search_history: list[str]
    errors: list[WorkflowError]
    node_metrics: list[NodeMetric]
