"""Search provider abstraction."""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import Protocol

from researchgraph.schemas.sources import SearchResult


class SearchChannel(StrEnum):
    WEB = "web"
    ACADEMIC = "academic"


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]: ...


class RateLimiter:
    """Minimum spacing between calls to a provider (arXiv asks for ~3s between requests)."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._interval - (time.monotonic() - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()
