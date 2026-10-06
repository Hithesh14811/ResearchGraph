import httpx
import pytest

from researchgraph.core.errors import (
    DocumentTooLargeError,
    UnsafeURLError,
    UnsupportedContentTypeError,
)
from researchgraph.tools.calculator import CalculatorError, evaluate_expression
from researchgraph.tools.fetcher import HttpFetcher
from researchgraph.tools.url_safety import (
    canonicalize_url,
    validate_public_url,
    validate_url_syntax,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.org/x",
        "http://localhost/admin",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.5/",
        "http://[::1]/",
        "https://user:pass@example.org/",
        "https://example.org:8443/",
        "http://printer.local/",
    ],
)
def test_unsafe_urls_are_rejected(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_url_syntax(url)


async def test_dns_resolution_to_private_address_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public-looking hostname that resolves to an internal IP (DNS-based SSRF)."""
    import asyncio
    import socket

    async def fake_getaddrinfo(
        host: str, *args: object, **kwargs: object
    ) -> list[tuple[object, ...]]:
        ip = "10.1.2.3" if host == "internal.example.com" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    assert await validate_public_url("https://example.com/paper") == "https://example.com/paper"
    with pytest.raises(UnsafeURLError):
        await validate_public_url("https://internal.example.com/")


def test_canonicalize_url_removes_tracking_and_fragments() -> None:
    assert (
        canonicalize_url("HTTPS://www.Example.org/a/?utm_source=x&b=2&a=1#frag")
        == "https://example.org/a?a=1&b=2"
    )


def test_calculator_is_safe() -> None:
    assert evaluate_expression("(71.2 - 64.5) / 64.5 * 100") == pytest.approx(10.3876, rel=1e-3)
    assert evaluate_expression("sqrt(16) + 2^3") == 12
    for expression in (
        "__import__('os').system('x')",
        "a + 1",
        "(1).__class__",
        "9 ** 999",
        "1/0",
        "[1, 2]",
    ):
        with pytest.raises(CalculatorError):
            evaluate_expression(expression)


def _fetcher(handler: httpx.MockTransport, max_bytes: int = 1000) -> HttpFetcher:
    client = httpx.AsyncClient(transport=handler, follow_redirects=False)
    return HttpFetcher(
        max_bytes=max_bytes, timeout_seconds=5, max_redirects=2, client=client, resolve_dns=False
    )


async def test_fetcher_follows_safe_redirects_and_blocks_private_targets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.org/final"})
        if request.url.path == "/evil":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>ok</p>")

    fetcher = _fetcher(httpx.MockTransport(handler))
    result = await fetcher.fetch("https://example.org/start")
    assert result.final_url == "https://example.org/final" and result.body == b"<p>ok</p>"
    with pytest.raises(UnsafeURLError):
        await fetcher.fetch("https://example.org/evil")


async def test_fetcher_enforces_size_and_content_type_limits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/big":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 5000)
        return httpx.Response(200, headers={"content-type": "application/zip"}, content=b"PK")

    fetcher = _fetcher(httpx.MockTransport(handler))
    with pytest.raises(DocumentTooLargeError):
        await fetcher.fetch("https://example.org/big")
    with pytest.raises(UnsupportedContentTypeError):
        await fetcher.fetch("https://example.org/archive")
