"""Follow a run's events: replay persisted history, then live events, without gaps."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from researchgraph.database.repository import ResearchRepository
from researchgraph.runtime.events import EventBroker
from researchgraph.runtime.manager import RunManager
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.events import EventType, WorkflowEvent

TERMINAL_STATUSES = frozenset(s.value for s in RunStatus if s.is_terminal)


def is_terminal_event(event: WorkflowEvent) -> bool:
    return event.type is EventType.RUN_STATUS and event.data.get("status") in TERMINAL_STATUSES


async def follow_run_events(
    *,
    repository: ResearchRepository,
    broker: EventBroker,
    manager: RunManager,
    research_id: str,
    after_seq: int = 0,
    keepalive_seconds: float = 15.0,
) -> AsyncIterator[WorkflowEvent | None]:
    """Yield events in sequence order; ``None`` marks an idle keep-alive tick.

    Stops after a terminal run-status event, or when the run is terminal and idle.
    """
    queue = broker.subscribe(research_id)  # subscribe before replaying so nothing is missed
    try:
        sent = after_seq
        for record in await repository.list_events(research_id, after_seq=after_seq, limit=10_000):
            event = WorkflowEvent(
                seq=record.seq,
                research_id=research_id,
                type=EventType(record.type),
                node=record.node,
                message=record.message,
                data=record.data or {},
                timestamp=record.created_at,
            )
            sent = record.seq
            yield event
            if is_terminal_event(event) and not manager.is_active(research_id):
                return
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
            except TimeoutError:
                yield None
                run = await repository.get_run(research_id)
                if run is None or (
                    run.status in TERMINAL_STATUSES and not manager.is_active(research_id)
                ):
                    return
                continue
            if event.seq <= sent:
                continue
            sent = event.seq
            yield event
            if is_terminal_event(event):
                return
    finally:
        broker.unsubscribe(research_id, queue)
