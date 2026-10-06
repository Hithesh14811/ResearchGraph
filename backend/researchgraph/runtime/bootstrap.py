"""Process-level wiring shared by the API server, the demo CLI and the evaluation runner."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncEngine

from researchgraph.config.observability import configure_tracing
from researchgraph.config.settings import Settings
from researchgraph.database.checkpointer import open_checkpointer
from researchgraph.database.engine import create_engine, init_models, session_factory
from researchgraph.database.repository import ResearchRepository
from researchgraph.graph.builder import build_research_graph
from researchgraph.runtime.container import ServiceContainer
from researchgraph.runtime.events import EventBroker
from researchgraph.runtime.manager import RunManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppServices:
    settings: Settings
    engine: AsyncEngine
    repository: ResearchRepository
    graph: CompiledStateGraph[Any, Any, Any, Any]
    container: ServiceContainer
    broker: EventBroker
    manager: RunManager
    tracing_enabled: bool


@asynccontextmanager
async def open_services(
    settings: Settings, *, recover: bool | None = None
) -> AsyncIterator[AppServices]:
    tracing = configure_tracing(settings)
    engine = create_engine(settings.database_url)
    await init_models(engine)
    repository = ResearchRepository(session_factory(engine))
    container = ServiceContainer.from_settings(settings)
    try:
        async with open_checkpointer(settings) as checkpointer:
            graph = build_research_graph(checkpointer)
            broker = EventBroker()
            manager = RunManager(
                settings=settings,
                graph=graph,
                repository=repository,
                container=container,
                broker=broker,
            )
            services = AppServices(
                settings=settings,
                engine=engine,
                repository=repository,
                graph=graph,
                container=container,
                broker=broker,
                manager=manager,
                tracing_enabled=tracing,
            )
            if settings.resume_runs_on_startup if recover is None else recover:
                await manager.recover()
            try:
                yield services
            finally:
                await manager.shutdown()
    finally:
        await container.aclose()
        await engine.dispose()
