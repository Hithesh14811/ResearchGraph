"""Bounded retries with exponential backoff and jitter.

LangGraph's node-level ``RetryPolicy`` retries a whole node; this helper retries a single
operation (one LLM call, one HTTP request) so a transient failure doesn't redo work that
already succeeded inside the node.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_TRANSIENT_NAME_MARKERS = (
    "RateLimit",
    "APIConnection",
    "APITimeout",
    "Overloaded",
    "InternalServer",
    "ServiceUnavailable",
    "Timeout",
)


@dataclass(frozen=True)
class BackoffPolicy:
    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Delay before retry number ``attempt`` (1-based)."""
        delay = min(self.max_delay, self.initial_delay * self.multiplier ** (attempt - 1))
        if self.jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay


def is_transient_error(exc: BaseException) -> bool:
    """Heuristic classification of errors worth retrying.

    Provider SDKs (OpenAI, Anthropic, ...) each define their own exception types; rather
    than importing every SDK we look at HTTP status codes and well-known class names.
    """
    if isinstance(
        exc, TimeoutError | asyncio.TimeoutError | httpx.TimeoutException | httpx.NetworkError
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS_CODES:
        return True
    name = type(exc).__name__
    return any(marker in name for marker in _TRANSIENT_NAME_MARKERS)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: BackoffPolicy,
    retry_on: Callable[[BaseException], bool] = is_transient_error,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``operation`` until it succeeds, a non-retryable error occurs, or attempts run out."""
    attempt = 1
    while True:
        try:
            return await operation()
        except Exception as exc:
            if attempt >= policy.max_attempts or not retry_on(exc):
                raise
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            logger.warning(
                "Transient failure (attempt %d/%d): %s: %s; retrying in %.2fs",
                attempt,
                policy.max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            await sleep(delay)
            attempt += 1
