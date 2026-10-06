from langgraph.types import Send

from researchgraph.graph.routing import (
    dispatch_research,
    route_after_citation_check,
    route_after_evidence,
    route_after_intake,
    route_after_quality_gate,
    route_search_agent,
)
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.research import ResearchGap, ResearchTask, RunLimits, SearchQuery
from researchgraph.schemas.review import (
    CitationCheck,
    CitationIssue,
    EvidenceAssessment,
    QualityAssessment,
)
from researchgraph.schemas.sources import Source

LIMITS = RunLimits(
    max_iterations=3,
    max_tool_calls=60,
    max_citation_repairs=2,
    min_evidence_per_subquestion=2,
    quality_threshold=0.65,
    credible_source_threshold=0.45,
)
GAP = ResearchGap(subquestion_id="SQ1", description="thin", reason="insufficient_evidence")


def _assessment(sufficient: bool) -> EvidenceAssessment:
    return EvidenceAssessment(
        iteration=1, coverage=[], sufficient=sufficient, gaps=[] if sufficient else [GAP]
    )


def test_intake_routing() -> None:
    assert route_after_intake({"status": RunStatus.REJECTED}) == "__end__"
    assert route_after_intake({"status": RunStatus.RUNNING}) == "planner"


def test_dispatch_fans_out_one_send_per_task_with_known_sources() -> None:
    task = ResearchTask(
        task_id="T1-SQ1",
        subquestion_id="SQ1",
        question="q?",
        queries=[SearchQuery(query="abc")],
        iteration=1,
        tool_budget=8,
    )
    processed = Source(id="S-1", url="https://e.org/1", title="t", provider="p", status="processed")
    failed = Source(id="S-2", url="https://e.org/2", title="t", provider="p", status="failed")
    sends = dispatch_research(
        {
            "research_id": "r",
            "question": "Q",
            "pending_tasks": [
                task,
                task.model_copy(update={"task_id": "T1-SQ2", "subquestion_id": "SQ2"}),
            ],
            "sources": {"S-1": processed, "S-2": failed},
        }
    )
    assert (
        isinstance(sends, list)
        and len(sends) == 2
        and all(isinstance(s, Send) and s.node == "research_worker" for s in sends)
    )
    assert list(sends[0].arg["known_sources"]) == ["S-1"]
    assert dispatch_research({"pending_tasks": []}) == "source_evaluation"


def test_evidence_gate_loops_only_with_gaps_and_budget() -> None:
    base = {"limits": LIMITS, "iteration": 1, "tool_calls_used": 10}
    assert (
        route_after_evidence(
            {**base, "evidence_assessment": _assessment(False), "missing_evidence": [GAP]}
        )
        == "query_generation"
    )
    assert (
        route_after_evidence(
            {**base, "evidence_assessment": _assessment(True), "missing_evidence": []}
        )
        == "contradiction_detection"
    )
    exhausted = {
        **base,
        "iteration": 3,
        "evidence_assessment": _assessment(False),
        "missing_evidence": [GAP],
    }
    assert route_after_evidence(exhausted) == "contradiction_detection"
    no_tools = {
        **base,
        "tool_calls_used": 59,
        "evidence_assessment": _assessment(False),
        "missing_evidence": [GAP],
    }
    assert route_after_evidence(no_tools) == "contradiction_detection"


def _qa(decision: str) -> QualityAssessment:
    return QualityAssessment(
        iteration=1,
        score=0.5,
        threshold=0.65,
        passed=decision == "proceed",
        components={},
        decision=decision,
    )  # type: ignore[arg-type]


def test_quality_gate_routing() -> None:
    base = {"limits": LIMITS, "iteration": 1, "tool_calls_used": 10}
    assert (
        route_after_quality_gate(
            {**base, "quality": _qa("research_more"), "missing_evidence": [GAP]}
        )
        == "query_generation"
    )
    assert (
        route_after_quality_gate({**base, "quality": _qa("proceed"), "missing_evidence": []})
        == "report_writer"
    )
    assert (
        route_after_quality_gate(
            {**base, "quality": _qa("proceed_with_limitations"), "missing_evidence": []}
        )
        == "report_writer"
    )
    assert (
        route_after_quality_gate(
            {**base, "iteration": 3, "quality": _qa("research_more"), "missing_evidence": [GAP]}
        )
        == "report_writer"
    )


def test_citation_routing_is_bounded() -> None:
    issue = CitationIssue(claim_id="C1", claim_text="x", problem="missing_citation", detail="d")
    failing = CitationCheck(
        attempt=0, total_claims=1, cited_claims=0, verified_claims=0, coverage=0, issues=[issue]
    )
    passing = CitationCheck(
        attempt=0, total_claims=1, cited_claims=1, verified_claims=1, coverage=1
    )
    assert (
        route_after_citation_check(
            {"limits": LIMITS, "citation_check": failing, "citation_repairs": 0}
        )
        == "citation_repair"
    )
    assert (
        route_after_citation_check(
            {"limits": LIMITS, "citation_check": failing, "citation_repairs": 2}
        )
        == "final_review"
    )
    assert (
        route_after_citation_check(
            {"limits": LIMITS, "citation_check": passing, "citation_repairs": 0}
        )
        == "final_review"
    )


def test_worker_search_routing() -> None:
    assert route_search_agent({"search_done": True}) == "source_processing"
    assert route_search_agent({"search_done": False}) == "execute_tools"
