"""Runs survive a server restart: drain at shutdown, resume from the checkpoint on startup."""

from __future__ import annotations

import asyncio
from pathlib import Path

from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.runtime.bootstrap import open_services
from researchgraph.schemas.common import RunStatus
from tests.conftest import make_settings, wait_for_status


async def test_running_research_is_drained_and_resumed_after_restart(tmp_path: Path) -> None:
    # Slow mock model so the run is still executing when "the server" shuts down.
    settings = make_settings(tmp_path, mock_llm_latency_ms=60)

    async with open_services(settings, recover=False) as services:
        research_id = await services.manager.create_run(
            SAMPLE_QUESTIONS[0].question, auto_approve=True
        )
        for _ in range(400):
            run = await services.repository.get_run(research_id)
            if run is not None and run.current_stage == "research":
                break
            await asyncio.sleep(0.02)
        # Leaving the context triggers manager.shutdown(): cooperative drain via RunControl.

    async with open_services(settings, recover=False) as services:
        run = await services.repository.get_run(research_id)
        assert run is not None and run.status == RunStatus.INTERRUPTED.value
        snapshot = await services.graph.aget_state({"configurable": {"thread_id": research_id}})
        assert snapshot.next  # there is remaining work in the checkpoint

        await services.manager.recover()  # what the API does on startup
        run = await wait_for_status(services, research_id, RunStatus.COMPLETED, within=60)
        assert run.progress == 100
        events = await services.repository.list_events(research_id)
        messages = [e.message for e in events]
        assert any("paused at a checkpoint" in m for m in messages)
        assert any("Resuming from last checkpoint" in m for m in messages)
        assert [e.seq for e in events] == list(
            range(1, len(events) + 1)
        )  # gap-free across restarts
        report = await services.repository.get_report(research_id)
        assert report is not None and report.markdown.startswith("# ")


async def test_awaiting_approval_survives_restart(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    async with open_services(settings, recover=False) as services:
        research_id = await services.manager.create_run(SAMPLE_QUESTIONS[2].question)
        await wait_for_status(services, research_id, RunStatus.AWAITING_APPROVAL)

    async with open_services(settings, recover=True) as services:
        run = await services.repository.get_run(research_id)
        assert run is not None and run.status == RunStatus.AWAITING_APPROVAL.value
        await services.manager.approve(research_id)
        await wait_for_status(services, research_id, RunStatus.COMPLETED)
