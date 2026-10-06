"""Workflow events streamed to clients.

Events are *workflow telemetry*: node lifecycle, tool status, counts and summaries.
They never contain model reasoning or hidden chain-of-thought.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from researchgraph.schemas.common import utc_now


class EventType(StrEnum):
    RUN_STATUS = "run_status"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    PROGRESS = "progress"
    TOOL = "tool"
    WARNING = "warning"
    ERROR = "error"
    PLAN_READY = "plan_ready"
    METRICS = "metrics"


class WorkflowEvent(BaseModel):
    seq: int = 0
    research_id: str
    type: EventType
    node: str | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)
