"""Safe HTTP retrieval of source documents.

The fetcher is a *pipeline* component, not an LLM tool: the model can only choose among
URLs that a search provider returned, and every URL (including each redirect hop) passes
SSRF validation. Responses are streamed with a hard byte cap and a content-type allowlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urljoin

import httpx

from researchgraph.core.errors import DocumentTooLargeError, FetchError, UnsupportedContentTypeError
from researchgraph.core.retry import BackoffPolicy, is_transient_error, retry_async
from researchgraph.tools.documents import detect_format
from researchgraph.tools.url_safety import validate_public_url

logger = logging.getLogger(__name__)

USER_AGENT = "ResearchGraph/0.1 (automated research assistant)"


@dataclass(frozen=True)
class FetchedContent:
    url: str
    final_url: str
    content_type: str
    body: bytes
    status_code: int = 200


class Fetcher(Protocol):
    async def fetch(self, url: str) -> FetchedContent: ...


class HttpFetcher:
    def __init__(
        self,
        *,
        max_bytes: int,
        timeout_seconds: float,
        max_redirects: int,
        client: httpx.AsyncClient | None = None,
        resolve_dns: bool = True,
        backoff: BackoffPolicy | None = None,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._resolve_dns = resolve_dns
        self._backoff = backoff or BackoffPolicy(max_attempts=2, initial_delay=1.0)
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/pdf,text/plain,text/markdown;q=0.9,*/*;q=0.1",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch(self, url: str) -> FetchedContent:
        return await retry_async(
            lambda: self._fetch_once(url), policy=self._backoff, retry_on=is_transient_error
        )

    async def _fetch_once(self, url: str) -> FetchedContent:
        current = url
        for _ in range(self._max_redirects + 1):
            await validate_public_url(current, resolve=self._resolve_dns)
            async with self._client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError(
                            "Redirect without location", url=url, status_code=response.status_code
                        )
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    error = FetchError(
                        f"HTTP {response.status_code} for {current}",
                        url=url,
                        status_code=response.status_code,
                    )
                    raise error
                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                if detect_format(content_type, current) is None:
                    raise UnsupportedContentTypeError(
                        f"Unsupported content type {content_type!r}",
                        url=url,
                        status_code=response.status_code,
                    )
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self._max_bytes:
                    raise DocumentTooLargeError(
                        f"Document exceeds {self._max_bytes} bytes", url=url
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_bytes:
                        raise DocumentTooLargeError(
                            f"Document exceeds {self._max_bytes} bytes", url=url
                        )
                return FetchedContent(
                    url=url,
                    final_url=current,
                    content_type=content_type,
                    body=bytes(body),
                    status_code=response.status_code,
                )
        raise FetchError(f"Too many redirects (>{self._max_redirects})", url=url)
