"""Password gate for live (real-provider) runs on a public deployment.

A correct password is exchanged for a short-lived token that the dashboard sends with every
request that would spend provider credits. Tokens are stateless: ``<expiry>.<HMAC(expiry)>``
keyed by a hash of the password, so changing ``LIVE_MODE_PASSWORD`` revokes every token.
Failed attempts are rate-limited per client and globally to slow down guessing.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict, deque

TOKEN_HEADER = "X-Live-Token"


def _signing_key(password: str) -> bytes:
    return hashlib.sha256(b"researchgraph/live-token\x00" + password.encode()).digest()


def _sign(password: str, payload: str) -> str:
    return hmac.new(_signing_key(password), payload.encode(), hashlib.sha256).hexdigest()


def password_matches(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode(), expected.encode())


def issue_token(password: str, *, ttl_seconds: float, now: float | None = None) -> tuple[str, int]:
    expires_at = int((time.time() if now is None else now) + ttl_seconds)
    payload = str(expires_at)
    return f"{payload}.{_sign(password, payload)}", expires_at


def verify_token(token: str | None, password: str, *, now: float | None = None) -> bool:
    if not token or token.count(".") != 1:
        return False
    payload, signature = token.split(".")
    if not payload.isdigit():
        return False
    if not hmac.compare_digest(signature, _sign(password, payload)):
        return False
    return int(payload) > (time.time() if now is None else now)


class AttemptLimiter:
    """Sliding-window cap on failed password attempts, per client and in total."""

    def __init__(
        self, *, per_client: int = 5, overall: int = 50, window_seconds: float = 600.0
    ) -> None:
        self.per_client = per_client
        self.overall = overall
        self.window = window_seconds
        self._failures: defaultdict[str, deque[float]] = defaultdict(deque)

    def _recent(self, key: str, now: float) -> deque[float]:
        attempts = self._failures[key]
        while attempts and attempts[0] <= now - self.window:
            attempts.popleft()
        return attempts

    def blocked(self, client: str, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        recent = len(self._recent(client, now))
        if recent == 0:
            del self._failures[client]  # keep memory bounded by active offenders
        return recent >= self.per_client or len(self._recent("*", now)) >= self.overall

    def record_failure(self, client: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._recent(client, now).append(now)
        self._recent("*", now).append(now)
