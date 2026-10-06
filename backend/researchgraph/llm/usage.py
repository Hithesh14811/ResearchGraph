"""Token usage and cost accounting via a LangChain callback handler.

The handler is attached to the graph run's config, so every chat-model call made anywhere
inside the graph — including inside parallel worker subgraphs — is counted.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field


class ModelUsage(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class UsageSnapshot(BaseModel):
    llm_calls: int = 0
    llm_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    by_model: dict[str, ModelUsage] = Field(default_factory=dict)

    def plus(self, other: UsageSnapshot) -> UsageSnapshot:
        """Combine two snapshots (used when a run resumes after a restart)."""
        by_model = {k: v.model_copy() for k, v in self.by_model.items()}
        for name, usage in other.by_model.items():
            current = by_model.setdefault(name, ModelUsage())
            current.calls += usage.calls
            current.input_tokens += usage.input_tokens
            current.output_tokens += usage.output_tokens
        costs = [c for c in (self.estimated_cost_usd, other.estimated_cost_usd) if c is not None]
        return UsageSnapshot(
            llm_calls=self.llm_calls + other.llm_calls,
            llm_errors=self.llm_errors + other.llm_errors,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=round(sum(costs), 6) if costs else None,
            by_model=by_model,
        )


class UsageTracker(BaseCallbackHandler):
    """Thread-safe aggregation of ``usage_metadata`` reported by chat models."""

    run_inline = True  # aggregate synchronously; no executor hop per callback

    def __init__(
        self,
        *,
        input_cost_per_mtok: float | None = None,
        output_cost_per_mtok: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._input_cost = input_cost_per_mtok
        self._output_cost = output_cost_per_mtok
        self._calls = 0
        self._errors = 0
        self._by_model: dict[str, ModelUsage] = {}

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                metadata = getattr(message, "response_metadata", None) or {}
                model = str(metadata.get("model_name") or metadata.get("model") or "unknown")
                with self._lock:
                    self._calls += 1
                    bucket = self._by_model.setdefault(model, ModelUsage())
                    bucket.calls += 1
                    bucket.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    bucket.output_tokens += int(usage.get("output_tokens", 0) or 0)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            by_model = {k: v.model_copy() for k, v in self._by_model.items()}
            calls, errors = self._calls, self._errors
        input_tokens = sum(u.input_tokens for u in by_model.values())
        output_tokens = sum(u.output_tokens for u in by_model.values())
        cost = None
        if self._input_cost is not None and self._output_cost is not None:
            cost = round(
                input_tokens / 1e6 * self._input_cost + output_tokens / 1e6 * self._output_cost, 6
            )
        return UsageSnapshot(
            llm_calls=calls,
            llm_errors=errors,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=cost,
            by_model=by_model,
        )
