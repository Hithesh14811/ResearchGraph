"""Shared fixtures. Every test runs offline: mock LLM, synthetic corpus, hashing embeddings."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from researchgraph.config.settings import Settings
from researchgraph.graph.builder import build_research_graph
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.serde import state_serializer
from researchgraph.llm.usage import UsageTracker
from researchgraph.runtime.bootstrap import AppServices, open_services
from researchgraph.runtime.container import ServiceContainer
from researchgraph.schemas.common import RunStatus


def pytest_asyncio_loop_factories(config: pytest.Config, item: pytest.Item) -> dict[str, Any]:
    """PostgreSQL tests need a selector loop on Windows (psycopg async cannot use Proactor)."""
    if sys.platform == "win32" and item.get_closest_marker("postgres") is not None:
        return {"selector": asyncio.SelectorEventLoop}
    return {"default": asyncio.new_event_loop}


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": "mock",
        "search_provider": "mock",
        "embedding_provider": "hashing",
        "vector_store": "memory",
        "database_url": f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}",
        "resume_runs_on_startup": False,
        "langchain_tracing_v2": False,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
async def container(settings: Settings) -> AsyncIterator[ServiceContainer]:
    services = ServiceContainer.from_settings(settings)
    yield services
    await services.aclose()


@pytest.fixture
def context(container: ServiceContainer) -> ResearchContext:
    return container.context_for_run()


@pytest.fixture
def graph() -> Any:
    return build_research_graph(InMemorySaver(serde=state_serializer()))


@pytest.fixture
def run_graph(graph: Any, container: ServiceContainer) -> Callable[..., Any]:
    """Run the graph to completion (or interrupt) and return (state values, events, usage)."""

    async def _run(
        question: str,
        *,
        auto_approve: bool = True,
        failures: frozenset[str] = frozenset(),
        thread_id: str = "t1",
        context: ResearchContext | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
        ctx = context or container.context_for_run(failures=failures)
        tracker = UsageTracker()
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [tracker],
            "recursion_limit": 200,
        }
        events: list[dict[str, Any]] = []
        payload = {
            "research_id": f"rg_{thread_id}",
            "original_question": question,
            "auto_approve": auto_approve,
        }
        async for _ns, mode, chunk in graph.astream(
            payload, config, context=ctx, stream_mode=["custom", "updates"], subgraphs=True
        ):
            if mode == "custom":
                events.append(chunk)
        snapshot = await graph.aget_state(config)
        return dict(snapshot.values), events, tracker.snapshot()

    return _run


@pytest.fixture
async def services(settings: Settings) -> AsyncIterator[AppServices]:
    async with open_services(settings, recover=False) as app_services:
        yield app_services


async def wait_for_status(
    services: AppServices, research_id: str, *statuses: RunStatus, within: float = 30.0
) -> Any:
    wanted = {s.value for s in statuses}
    deadline = asyncio.get_running_loop().time() + within
    while True:
        run = await services.repository.get_run(research_id)
        if run is not None and run.status in wanted and not services.manager.is_active(research_id):
            return run
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(
                f"Run {research_id} did not reach {wanted}; last status {run.status if run else None}"
            )
        await asyncio.sleep(0.05)
