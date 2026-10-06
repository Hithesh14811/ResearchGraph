"""End-to-end graph runs with the deterministic mock model and synthetic corpus."""

from __future__ import annotations

from typing import Any

import pytest

from researchgraph.core.text import normalize_for_match
from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.events import EventType


def assert_provenance_invariants(state: dict[str, Any]) -> None:
    """The core guarantees of the citation system, checked on real workflow output."""
    report = state["final_report"]
    evidence = state["evidence"]
    sources = state["sources"]
    # 1. Every evidence item points at a stored, processed source with the same URL.
    for item in evidence.values():
        source = sources[item.source_id]
        assert source.status == "processed" and source.url == item.source_url
        assert item.quote_verified
    # 2. Every published sentence cites existing evidence (no unsupported claims survive).
    for claim in report.draft.all_claims():
        assert claim.evidence_ids, claim.text
        assert all(eid in evidence for eid in claim.evidence_ids)
    # 3. Bibliography entries are exactly the sources behind cited evidence; no invented URLs.
    cited_sources = {
        evidence[e].source_id for c in report.draft.all_claims() for e in c.evidence_ids
    }
    assert {c.source_id for c in report.bibliography} == cited_sources
    assert all(c.url == sources[c.source_id].url for c in report.bibliography)
    assert [c.number for c in report.bibliography] == list(range(1, len(report.bibliography) + 1))
    # 4. Findings only reference real evidence.
    for finding in state["analysis"].findings:
        assert finding.evidence_ids and all(e in evidence for e in finding.evidence_ids)
    assert report.metrics.citation_coverage == 1.0


@pytest.mark.parametrize("sample", SAMPLE_QUESTIONS, ids=lambda s: s.key)
async def test_full_workflow_produces_verified_report(run_graph: Any, sample: Any) -> None:
    state, events, usage = await run_graph(sample.question, thread_id=sample.key)
    assert state["status"] is RunStatus.COMPLETED
    assert len(state["plan"].subquestions) == 5
    assert state["iteration"] >= 1 and state["tool_calls_used"] <= state["limits"].max_tool_calls
    assert len(state["sources"]) >= 5 and len(state["evidence"]) >= 10
    assert state["source_quality"].keys() == state["sources"].keys()
    assert state["critique_history"] and state["quality_history"]
    assert_provenance_invariants(state)
    report = state["final_report"]
    assert (
        report.markdown.startswith("# ")
        and "## References" in report.markdown
        and "[1]" in report.markdown
    )
    assert usage.llm_calls >= 10 and usage.total_tokens > 0

    # Parallel research: one worker per subquestion ran in the first round.
    worker_starts = {
        e["data"].get("subquestion_id")
        for e in events
        if e["type"] == EventType.NODE_STARTED.value and e["node"] == "search_agent"
    }
    assert {"SQ1", "SQ2", "SQ3", "SQ4", "SQ5"} <= worker_starts
    # Events are safe telemetry: plain strings, no raw prompts.
    assert all(isinstance(e["message"], str) and "<passages>" not in e["message"] for e in events)


async def test_parallel_workers_merge_without_duplicates(run_graph: Any) -> None:
    state, _, _ = await run_graph(SAMPLE_QUESTIONS[0].question)
    urls = [s.url for s in state["sources"].values()]
    assert len(urls) == len(set(urls))  # content-addressed IDs collapse duplicates across workers
    quotes = [
        (e.subquestion_id, e.source_id, normalize_for_match(e.supporting_text))
        for e in state["evidence"].values()
    ]
    assert len(quotes) == len(set(quotes))


async def test_invalid_question_is_rejected_at_intake(run_graph: Any) -> None:
    state, _, _ = await run_graph("rag?")
    assert state["status"] is RunStatus.REJECTED
    assert "plan" not in state and state["errors"][0].error_type == "invalid_question"


async def test_custom_question_uses_fallback_planning(run_graph: Any) -> None:
    state, _, _ = await run_graph(
        "Evaluate approaches for reducing hallucinations in retrieval systems used for legal research."
    )
    assert state["status"] is RunStatus.COMPLETED
    assert len(state["plan"].subquestions) >= 3


async def test_deterministic_research_agent_mode(graph: Any, settings: Any, tmp_path: Any) -> None:
    from researchgraph.runtime.container import ServiceContainer
    from tests.conftest import make_settings

    container = ServiceContainer.from_settings(
        make_settings(tmp_path, research_agent_mode="deterministic")
    )
    try:
        config = {"configurable": {"thread_id": "det"}, "recursion_limit": 200}
        await graph.ainvoke(
            {
                "research_id": "rg_det",
                "original_question": SAMPLE_QUESTIONS[2].question,
                "auto_approve": True,
            },
            config,
            context=container.context_for_run(),
        )
        state = (await graph.aget_state(config)).values
        assert state["status"] is RunStatus.COMPLETED and state["evidence"]
    finally:
        await container.aclose()
