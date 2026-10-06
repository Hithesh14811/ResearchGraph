"""Node instrumentation: lifecycle events, latency metrics, timeouts and graceful degradation.

Every node is wrapped by ``@instrumented``. The wrapper

* emits ``node_started`` / ``node_completed`` custom stream events (safe workflow telemetry,
  never model reasoning) that the API relays to the UI;
* records a ``NodeMetric`` into state for latency reporting;
* optionally enforces a timeout and converts unexpected failures into a recorded
  ``WorkflowError`` plus a degraded update, so one failing worker cannot sink a run;
* always re-raises LangGraph control-flow signals (interrupts, drains).
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from time import perf_counter
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from researchgraph.core.errors import ResearchGraphError
from researchgraph.schemas.common import NodeMetric, WorkflowError
from researchgraph.schemas.events import EventType

logger = logging.getLogger(__name__)

_current_node: ContextVar[str | None] = ContextVar("researchgraph_node", default=None)
_current_subquestion: ContextVar[str | None] = ContextVar("researchgraph_subquestion", default=None)

NodeFn = Callable[..., Awaitable[Any]]
Degrade = Callable[[Mapping[str, Any], BaseException], dict[str, Any]]


def emit(event_type: EventType, message: str, **data: Any) -> None:
    """Send a workflow event on LangGraph's ``custom`` stream (no-op outside a graph run)."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    subquestion = _current_subquestion.get()
    if subquestion is not None:
        data.setdefault("subquestion_id", subquestion)
    writer(
        {"type": event_type.value, "node": _current_node.get(), "message": message, "data": data}
    )


def workflow_error(
    node: str, exc: BaseException, *, subquestion_id: str | None = None
) -> WorkflowError:
    code = exc.code if isinstance(exc, ResearchGraphError) else type(exc).__name__
    recoverable = exc.recoverable if isinstance(exc, ResearchGraphError) else True
    return WorkflowError(
        node=node,
        error_type=code,
        message=str(exc)[:500],
        recoverable=recoverable,
        subquestion_id=subquestion_id,
    )


def _with_metric(result: Any, metric: NodeMetric) -> Any:
    if isinstance(result, Command):
        update = (
            dict(result.update or {})
            if isinstance(result.update, Mapping | type(None))
            else result.update
        )
        if isinstance(update, dict):
            update["node_metrics"] = [*update.get("node_metrics", []), metric]
            return dataclasses.replace(result, update=update)
        return result
    if result is None:
        return {"node_metrics": [metric]}
    if isinstance(result, dict):
        return {**result, "node_metrics": [*result.get("node_metrics", []), metric]}
    return result


def _iteration(state: Mapping[str, Any]) -> int:
    task = state.get("task")
    if task is not None and hasattr(task, "iteration"):
        return int(task.iteration)
    return int(state.get("iteration", 0) or 0)


def instrumented(
    name: str,
    *,
    start: str | None = None,
    timeout_s: float | None = None,
    degrade: Degrade | None = None,
) -> Callable[[NodeFn], NodeFn]:
    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
            token = _current_node.set(name)
            task = state.get("task")
            sq_token = _current_subquestion.set(getattr(task, "subquestion_id", None))
            started = perf_counter()
            ok = True
            try:
                emit(EventType.NODE_STARTED, start or f"Running {name.replace('_', ' ')}...")
                try:
                    if timeout_s is not None:
                        async with asyncio.timeout(timeout_s):
                            result = await fn(state, *args, **kwargs)
                    else:
                        result = await fn(state, *args, **kwargs)
                except GraphBubbleUp:
                    raise  # interrupts/drains are control flow, not failures
                except Exception as exc:
                    ok = False
                    if degrade is None:
                        emit(EventType.ERROR, f"{name} failed: {type(exc).__name__}: {exc}"[:300])
                        raise
                    logger.exception("Node %s failed; degrading gracefully", name)
                    emit(
                        EventType.WARNING,
                        f"{name} failed ({type(exc).__name__}); continuing with partial results",
                    )
                    result = degrade(state, exc)
                duration_ms = round((perf_counter() - started) * 1000, 1)
                emit(
                    EventType.NODE_COMPLETED,
                    f"{name.replace('_', ' ').capitalize()} finished",
                    duration_ms=duration_ms,
                    ok=ok,
                )
                metric = NodeMetric(
                    node=name, duration_ms=duration_ms, iteration=_iteration(state), ok=ok
                )
                return _with_metric(result, metric)
            finally:
                _current_node.reset(token)
                _current_subquestion.reset(sq_token)

        return wrapper

    return decorator
