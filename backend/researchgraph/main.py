"""FastAPI application factory and entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from researchgraph import __version__
from researchgraph.api.routes import public, router
from researchgraph.config.logging import configure_logging
from researchgraph.config.settings import Settings, get_settings
from researchgraph.core.errors import (
    ConfigurationError,
    InvalidRunStateError,
    ResearchGraphError,
    RunNotFoundError,
    TooManyRunsError,
)
from researchgraph.runtime.bootstrap import open_services

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 256 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_services(settings) as services:
            app.state.services = services
            logger.info(
                "ResearchGraph API ready (llm=%s, search=%s)",
                settings.llm_provider,
                settings.search_provider,
            )
            yield
            app.state.services = None

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=__version__,
        description="Autonomous, evidence-driven research agent built on LangGraph.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID"],
    )

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)

    @app.exception_handler(RunNotFoundError)
    async def _not_found(_: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": exc.message, "code": exc.code})

    @app.exception_handler(InvalidRunStateError)
    async def _conflict(_: Request, exc: InvalidRunStateError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": exc.message, "code": exc.code})

    @app.exception_handler(TooManyRunsError)
    async def _too_many(_: Request, exc: TooManyRunsError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": exc.message, "code": exc.code})

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)[:500]})

    @app.exception_handler(ConfigurationError)
    async def _config_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": exc.message, "code": exc.code})

    @app.exception_handler(ResearchGraphError)
    async def _domain_error(_: Request, exc: ResearchGraphError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": exc.message, "code": exc.code})

    app.include_router(public)
    app.include_router(router)
    return app


def run() -> None:
    """Console entry point: ``researchgraph-api``."""
    import uvicorn

    from researchgraph.core.aio import use_selector_event_loop_on_windows

    use_selector_event_loop_on_windows()
    settings = get_settings()
    uvicorn.run(
        "researchgraph.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
