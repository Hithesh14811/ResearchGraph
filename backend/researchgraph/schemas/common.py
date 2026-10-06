"""Shared primitives for all domain models."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_research_id() -> str:
    return f"rg_{uuid.uuid4().hex[:16]}"


def stable_id(prefix: str, *parts: str, length: int = 10) -> str:
    """Deterministic, content-addressed identifier.

    Used for sources (canonical URL) and evidence (source + quote) so that the same item
    discovered by two parallel workers collapses to one key when their state is merged.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


class DomainModel(BaseModel):
    """Base for state models.

    Frozen so graph state is never mutated in place — nodes must return new objects
    (``model_copy(update=...)``), which keeps LangGraph reducers and checkpoints honest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.REJECTED,
        }


class WorkflowError(DomainModel):
    """An error captured in graph state. The workflow records errors and degrades
    gracefully instead of crashing whenever a step has a sensible fallback."""

    node: str
    error_type: str
    message: str
    recoverable: bool = True
    subquestion_id: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class NodeMetric(DomainModel):
    """Latency record for one node execution (also streamed to the UI)."""

    node: str
    duration_ms: float
    iteration: int = 0
    ok: bool = True
