"""The graph recovers from each simulated failure instead of only working on the happy path."""

from __future__ import annotations

from typing import Any

from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.events import EventType

QUESTION = SAMPLE_QUESTIONS[0].question


def _warnings(events: list[dict[str, Any]]) -> list[str]:
    return [e["message"] for e in events if e["type"] == EventType.WARNING.value]


async def test_failed_sources_are_recorded_and_degrade_to_snippets(run_graph: Any) -> None:
    state, _, _ = await run_graph(QUESTION, failures=frozenset({"failed_source"}))
    assert state["status"] is RunStatus.COMPLETED
    fetch_errors = [e for e in state["errors"] if e.node == "source_processing"]
    assert fetch_errors and all(
        e.error_type == "fetch_failed" and e.recoverable for e in fetch_errors
    )
    degraded = [s for s in state["sources"].values() if s.content_origin == "search_snippet"]
    assert degraded and all(s.error for s in degraded)


async def test_llm_timeout_is_retried(run_graph: Any) -> None:
    state, events, usage = await run_graph(QUESTION, failures=frozenset({"llm_timeout"}))
    assert state["status"] is RunStatus.COMPLETED
    assert any("TimeoutError on attempt 1; retrying" in w for w in _warnings(events))
    assert usage.llm_errors == 1
    assert not [e for e in state["errors"] if e.node == "planner"]  # recovered, so no fallback plan


async def test_invalid_structured_output_is_repaired(run_graph: Any) -> None:
    state, events, _ = await run_graph(QUESTION, failures=frozenset({"invalid_output"}))
    assert state["status"] is RunStatus.COMPLETED
    assert any("failed schema validation" in w for w in _warnings(events))
    assert not [e for e in state["errors"] if e.node == "critic"]


async def test_insufficient_evidence_triggers_another_research_round(run_graph: Any) -> None:
    state, events, _ = await run_graph(QUESTION, failures=frozenset({"insufficient_evidence"}))
    assert state["status"] is RunStatus.COMPLETED
    assert state["iteration"] >= 2
    rounds = [e for e in events if e["node"] == "query_generation" and "targeted" in e["message"]]
    assert rounds
    assert all(c.sufficient for c in state["evidence_assessment"].coverage)


async def test_critic_rejection_routes_back_through_the_quality_gate(run_graph: Any) -> None:
    state, _, _ = await run_graph(QUESTION, failures=frozenset({"critic_rejection"}))
    assert state["status"] is RunStatus.COMPLETED
    first, *rest = state["quality_history"]
    assert first.decision == "research_more" and not first.passed
    assert any("critical" in b for b in first.blocking_issues)
    assert rest and state["iteration"] >= 2
    assert state["critique_history"][0].verdict == "major_problems"


async def test_all_failures_at_once_still_produce_a_verified_report(run_graph: Any) -> None:
    from tests.integration.test_graph_e2e import assert_provenance_invariants

    state, _, _ = await run_graph(
        QUESTION,
        failures=frozenset(
            {
                "failed_source",
                "llm_timeout",
                "invalid_output",
                "insufficient_evidence",
                "critic_rejection",
            }
        ),
    )
    assert state["status"] is RunStatus.COMPLETED
    assert_provenance_invariants(state)
