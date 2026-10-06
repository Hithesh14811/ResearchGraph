"""In-process pub/sub for live workflow events (fan-out to SSE subscribers).

Events are also persisted, so a client that connects late (or reconnects with
``Last-Event-ID``) replays history from the database and then follows the live queue.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from researchgraph.schemas.events import WorkflowEvent

logger = logging.getLogger(__name__)

SUBSCRIBER_QUEUE_SIZE = 2000


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[WorkflowEvent]]] = defaultdict(set)

    def subscribe(self, research_id: str) -> asyncio.Queue[WorkflowEvent]:
        queue: asyncio.Queue[WorkflowEvent] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers[research_id].add(queue)
        return queue

    def unsubscribe(self, research_id: str, queue: asyncio.Queue[WorkflowEvent]) -> None:
        subscribers = self._subscribers.get(research_id)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(research_id, None)

    def publish(self, event: WorkflowEvent) -> None:
        for queue in list(self._subscribers.get(event.research_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # slow consumer: it will resync from the DB on reconnect
                logger.warning(
                    "Dropping event %s for slow subscriber of %s", event.seq, event.research_id
                )
