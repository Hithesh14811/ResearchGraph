"""Concrete search providers: Tavily (web), arXiv and Semantic Scholar (academic).

Each provider normalises its response into ``SearchResult`` objects. Provider outputs are
untrusted data: titles/snippets are truncated and URLs are validated before any fetch.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

import httpx

from researchgraph.core.errors import SearchProviderError
from researchgraph.core.retry import BackoffPolicy, retry_async
from researchgraph.core.text import keywords, normalize_whitespace, truncate
from researchgraph.schemas.sources import SearchResult, SourceType
from researchgraph.tools.metadata import parse_date
from researchgraph.tools.search.base import RateLimiter
from researchgraph.tools.url_safety import validate_url_syntax

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 1500
_BACKOFF = BackoffPolicy(max_attempts=3, initial_delay=1.0, max_delay=10.0)


def _clean(text: Any, limit: int) -> str:
    return truncate(normalize_whitespace(str(text or "")), limit)


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return validate_url_syntax(url)
    except ValueError:
        return None


class TavilySearchProvider:
    """General web search via the Tavily API (requires TAVILY_API_KEY)."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, client: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._client = client

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        async def call() -> httpx.Response:
            response = await self._client.post(
                self.endpoint,
                json={
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                },
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            return response

        try:
            payload = (await retry_async(call, policy=_BACKOFF)).json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchProviderError(f"Tavily search failed: {exc}") from exc

        results = []
        for item in payload.get("results", [])[:max_results]:
            url = _safe_url(item.get("url"))
            if url is None:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=_clean(item.get("title"), 300) or url,
                    snippet=_clean(item.get("content"), _SNIPPET_CHARS),
                    provider=self.name,
                    published_date=parse_date(item.get("published_date")),
                    score=float(item.get("score") or 0.0),
                )
            )
        return results


class ArxivSearchProvider:
    """Preprint search via the public arXiv API (no key required)."""

    name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"
    _ns: ClassVar[dict[str, str]] = {"a": "http://www.w3.org/2005/Atom"}

    def __init__(self, client: httpx.AsyncClient, *, min_interval_seconds: float = 3.0) -> None:
        self._client = client
        self._limiter = RateLimiter(min_interval_seconds)

    @staticmethod
    def build_query(query: str) -> str:
        terms = [t for t in keywords(query, 6).split() if len(t) > 2]
        return " AND ".join(f"all:{t}" for t in terms) or f"all:{query}"

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        params: dict[str, str | int] = {
            "search_query": self.build_query(query),
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }

        async def call() -> httpx.Response:
            await self._limiter.wait()
            response = await self._client.get(self.endpoint, params=params)
            response.raise_for_status()
            return response

        try:
            root = ET.fromstring((await retry_async(call, policy=_BACKOFF)).text)
        except (httpx.HTTPError, ET.ParseError) as exc:
            raise SearchProviderError(f"arXiv search failed: {exc}") from exc

        results = []
        for rank, entry in enumerate(root.findall("a:entry", self._ns)):
            abs_url = (entry.findtext("a:id", default="", namespaces=self._ns) or "").strip()
            pdf_url = next(
                (
                    link.get("href")
                    for link in entry.findall("a:link", self._ns)
                    if link.get("title") == "pdf"
                ),
                None,
            )
            url = _safe_url((pdf_url or abs_url).replace("http://", "https://"))
            if url is None:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=_clean(entry.findtext("a:title", namespaces=self._ns), 300),
                    snippet=_clean(
                        entry.findtext("a:summary", namespaces=self._ns), _SNIPPET_CHARS
                    ),
                    provider=self.name,
                    published_date=parse_date(entry.findtext("a:published", namespaces=self._ns)),
                    authors=[
                        _clean(a.findtext("a:name", namespaces=self._ns), 120)
                        for a in entry.findall("a:author", self._ns)
                    ][:12],
                    venue="arXiv preprint",
                    type_hint=SourceType.PRIMARY_RESEARCH,
                    score=1.0 - rank / max(1, max_results),
                )
            )
        return results


class SemanticScholarSearchProvider:
    """Paper search via the Semantic Scholar Graph API (key optional, raises rate limits)."""

    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,url,abstract,year,publicationDate,authors,venue,citationCount,externalIds,openAccessPdf"

    def __init__(self, client: httpx.AsyncClient, api_key: str | None = None) -> None:
        self._client = client
        self._api_key = api_key
        self._limiter = RateLimiter(0.2 if api_key else 1.1)

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        headers = {"x-api-key": self._api_key} if self._api_key else {}

        async def call() -> httpx.Response:
            await self._limiter.wait()
            response = await self._client.get(
                self.endpoint,
                params={"query": query, "limit": max_results, "fields": self.fields},
                headers=headers,
            )
            response.raise_for_status()
            return response

        try:
            payload = (await retry_async(call, policy=_BACKOFF)).json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchProviderError(f"Semantic Scholar search failed: {exc}") from exc

        results = []
        for rank, paper in enumerate(payload.get("data") or []):
            pdf = (paper.get("openAccessPdf") or {}).get("url")
            url = _safe_url(pdf or paper.get("url"))
            if url is None:
                continue
            ids = paper.get("externalIds") or {}
            results.append(
                SearchResult(
                    url=url,
                    title=_clean(paper.get("title"), 300),
                    snippet=_clean(paper.get("abstract"), _SNIPPET_CHARS),
                    provider=self.name,
                    published_date=parse_date(paper.get("publicationDate") or paper.get("year")),
                    authors=[_clean(a.get("name"), 120) for a in (paper.get("authors") or [])][:12],
                    venue=_clean(paper.get("venue"), 200) or None,
                    doi=ids.get("DOI"),
                    citation_count=paper.get("citationCount"),
                    type_hint=SourceType.PRIMARY_RESEARCH,
                    score=1.0 - rank / max(1, max_results),
                )
            )
        return results
