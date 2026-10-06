"""Server-Sent Events stream of a run's workflow events (replay + live follow).

Events are sent as default (unnamed) SSE messages; the workflow event type lives in the JSON
payload. Named SSE events would collide with ``EventSource``'s built-in ``error``/``open``
events (a workflow ``error`` event would be indistinguishable from a dropped connection).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from researchgraph.runtime.bootstrap import AppServices
from researchgraph.runtime.streaming import follow_run_events


def format_sse(seq: int, payload: dict[str, Any]) -> str:
    return f"id: {seq}\ndata: {json.dumps(payload, default=str)}\n\n"


async def event_stream(
    request: Request, services: AppServices, research_id: str, last_seq: int
) -> AsyncIterator[str]:
    async for event in follow_run_events(
        repository=services.repository,
        broker=services.broker,
        manager=services.manager,
        research_id=research_id,
        after_seq=last_seq,
    ):
        if await request.is_disconnected():
            return
        if event is None:
            yield ": keep-alive\n\n"
            continue
        yield format_sse(event.seq, event.model_dump(mode="json"))
