"""FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from researchgraph.runtime.bootstrap import AppServices

_bearer = HTTPBearer(auto_error=False)


def get_services(request: Request) -> AppServices:
    services: AppServices | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is starting up"
        )
    return services


def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Optional bearer-token auth: enforced only when API_TOKEN is configured.

    EventSource cannot set headers, so the SSE endpoint also accepts ``?token=``.
    """
    services = get_services(request)
    expected = services.settings.api_token
    if expected is None:
        return
    supplied = credentials.credentials if credentials else request.query_params.get("token", "")
    if not secrets.compare_digest(supplied.encode(), expected.get_secret_value().encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token"
        )
