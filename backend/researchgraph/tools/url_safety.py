"""URL validation and canonicalisation (SSRF protection).

Only public http(s) URLs on standard ports are fetchable. Hostnames are resolved and every
resolved address must be globally routable, which blocks loopback, private ranges,
link-local (cloud metadata endpoints) and similar targets. Redirect targets are re-validated
by the fetcher on every hop.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from researchgraph.core.errors import UnsafeURLError

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({None, 80, 443})
_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp")
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref_src")
MAX_URL_LENGTH = 2048


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _check_ip(address: str) -> None:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop IPv6 zone index
    if not ip.is_global or ip.is_multicast:
        raise UnsafeURLError(f"URL resolves to a non-public address ({ip})")


def validate_url_syntax(url: str) -> str:
    """Structural checks that need no network access. Returns the stripped URL."""
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise UnsafeURLError("URL is too long")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Scheme {parts.scheme!r} is not allowed")
    if parts.username or parts.password:
        raise UnsafeURLError("Credentials in URLs are not allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURLError("URL has no host")
    try:
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError("Invalid port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError(f"Port {port} is not allowed")
    if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES):
        raise UnsafeURLError(f"Host {host!r} is not allowed")
    if _parse_ip(host) is not None:
        _check_ip(host)  # literal IP addresses are checked without DNS
    return url


async def validate_public_url(url: str, *, resolve: bool = True) -> str:
    """Full validation including DNS resolution of the hostname."""
    url = validate_url_syntax(url)
    if not resolve:
        return url
    host = urlsplit(url).hostname or ""
    if _parse_ip(host) is not None:
        return url  # literal IP already checked
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Cannot resolve host {host!r}") from exc
    for info in infos:
        _check_ip(str(info[4][0]))
    return url


def canonicalize_url(url: str) -> str:
    """Stable form used for deduplication and content-addressed source IDs."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith(_TRACKING_PARAMS)
        )
    )
    return urlunsplit((scheme, f"{host}{port}", path, query, ""))


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host
