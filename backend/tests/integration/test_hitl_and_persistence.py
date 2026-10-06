"""Human-in-the-loop interrupts and durable checkpoints (SQLite) across simulated restarts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.types import Command

from researchgraph.database.checkpointer import open_checkpointer
from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.graph.builder import build_research_graph
from researchgraph.runtime.container import ServiceContainer
from researchgraph.schemas.common import RunStatus
from tests.conftest import make_settings

QUESTION = SAMPLE_QUESTIONS[1].question


async def _start(graph: Any, container: ServiceContainer, thread: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 200}
    await graph.ainvoke(
        {"research_id": thread, "original_question": QUESTION, "auto_approve": False},
        config,
        context=container.context_for_run(),
    )
    return config


async def test_plan_interrupt_survives_restart_and_resumes_on_approval(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    container = ServiceContainer.from_settings(settings)
    try:
        # Process 1: run until the human checkpoint, then "crash".
        async with open_checkpointer(settings) as checkpointer:
            graph = build_research_graph(checkpointer)
            config = await _start(graph, container, "restart-1")
            snapshot = await graph.aget_state(config)
            assert snapshot.next == ("plan_review",)
            assert snapshot.interrupts[0].value["type"] == "plan_review"
            assert not snapshot.values.get("evidence")  # nothing expensive ran before approval

        # Process 2: fresh checkpointer connection + fresh graph, resume from disk.
        async with open_checkpointer(settings) as checkpointer:
            graph = build_research_graph(checkpointer)
            await graph.ainvoke(
                Command(resume={"action": "approve"}), config, context=container.context_for_run()
            )
            state = (await graph.aget_state(config)).values
            assert state["status"] is RunStatus.COMPLETED and state["final_report"].bibliography
    finally:
        await container.aclose()


async def test_edit_plan_replaces_subquestions(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    container = ServiceContainer.from_settings(settings)
    try:
        async with open_checkpointer(settings) as checkpointer:
            graph = build_research_graph(checkpointer)
            config = await _start(graph, container, "edit-1")
            plan = (await graph.aget_state(config)).values["plan"]
            edited = plan.model_copy(update={"subquestions": plan.subquestions[:2]})
            # Edit without approving: the graph pauses again for review of the edited plan.
            await graph.ainvoke(
                Command(
                    resume={
                        "action": "edit",
                        "plan": edited.model_dump(mode="json"),
                        "approve": False,
                    }
                ),
                config,
                context=container.context_for_run(),
            )
            snapshot = await graph.aget_state(config)
            assert snapshot.interrupts and snapshot.values["plan"].version == plan.version + 1
            assert len(snapshot.values["plan"].subquestions) == 2
            await graph.ainvoke(
                Command(resume={"action": "approve"}), config, context=container.context_for_run()
            )
            state = (await graph.aget_state(config)).values
            assert state["status"] is RunStatus.COMPLETED
            assert {e.subquestion_id for e in state["evidence"].values()} <= {"SQ1", "SQ2"}
    finally:
        await container.aclose()


async def test_replan_with_feedback_and_cancel(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    container = ServiceContainer.from_settings(settings)
    try:
        async with open_checkpointer(settings) as checkpointer:
            graph = build_research_graph(checkpointer)
            config = await _start(graph, container, "replan-1")
            await graph.ainvoke(
                Command(resume={"action": "replan", "feedback": "Cover energy efficiency"}),
                config,
                context=container.context_for_run(),
            )
            snapshot = await graph.aget_state(config)
            plan = snapshot.values["plan"]
            assert (
                snapshot.interrupts
                and plan.version == 2
                and "energy efficiency" in plan.subquestions[-1].question
            )
            await graph.ainvoke(
                Command(resume={"action": "cancel"}), config, context=container.context_for_run()
            )
            final = await graph.aget_state(config)
            assert final.values["status"] is RunStatus.CANCELLED and not final.next
            assert not final.values.get("evidence")
    finally:
        await container.aclose()
