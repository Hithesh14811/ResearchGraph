"""Search orchestration: provider fan-out, de-duplication and request coalescing.

Identical queries (from the same or concurrent research workers) are served from an
in-process cache, and concurrent identical requests share a single in-flight future —
avoiding duplicate external calls and wasted tool budget.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

from researchgraph.core.errors import SearchProviderError
from researchgraph.core.text import normalize_for_match
from researchgraph.schemas.sources import SearchResult
from researchgraph.tools.search.base import SearchChannel, SearchProvider
from researchgraph.tools.url_safety import canonicalize_url

logger = logging.getLogger(__name__)

CacheKey = tuple[str, str, int]


class SearchService:
    def __init__(
        self,
        *,
        web: list[SearchProvider],
        academic: list[SearchProvider],
        cache_size: int = 512,
    ) -> None:
        if not web and not academic:
            raise ValueError("SearchService needs at least one provider")
        self._providers = {SearchChannel.WEB: web, SearchChannel.ACADEMIC: academic}
        self._cache: OrderedDict[CacheKey, list[SearchResult]] = OrderedDict()
        self._inflight: dict[CacheKey, asyncio.Future[list[SearchResult]]] = {}
        self._cache_size = cache_size
        self.cache_hits = 0

    def providers_for(self, channel: SearchChannel) -> list[SearchProvider]:
        return (
            self._providers[channel]
            or self._providers[
                SearchChannel.ACADEMIC if channel is SearchChannel.WEB else SearchChannel.WEB
            ]
        )

    async def search(
        self, query: str, *, channel: SearchChannel, max_results: int
    ) -> list[SearchResult]:
        key: CacheKey = (channel.value, normalize_for_match(query), max_results)
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return list(self._cache[key])
        if key in self._inflight:
            self.cache_hits += 1
            return list(await self._inflight[key])

        future: asyncio.Future[list[SearchResult]] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            results = await self._search_uncached(query, channel=channel, max_results=max_results)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            future.exception()  # mark as retrieved; concurrent waiters re-raise it
            raise
        finally:
            self._inflight.pop(key, None)
        future.set_result(results)
        self._cache[key] = results
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return list(results)

    async def _search_uncached(
        self, query: str, *, channel: SearchChannel, max_results: int
    ) -> list[SearchResult]:
        providers = self.providers_for(channel)
        outcomes = await asyncio.gather(
            *(p.search(query, max_results=max_results) for p in providers), return_exceptions=True
        )
        merged: dict[str, SearchResult] = {}
        failures = []
        for provider, outcome in zip(providers, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures.append(f"{provider.name}: {outcome}")
                logger.warning("Search provider %s failed: %s", provider.name, outcome)
                continue
            for result in outcome:
                key = canonicalize_url(result.url)
                if key not in merged or result.score > merged[key].score:
                    merged[key] = result
        if failures and not merged and len(failures) == len(providers):
            raise SearchProviderError("All search providers failed: " + "; ".join(failures))
        ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return ranked[: max_results * max(1, len(providers))]
