"""Optional tracing: LangSmith and Langfuse, independently of each other.

LangSmith is enabled only when both ``LANGCHAIN_TRACING_V2=true`` *and* an API key are
configured. Otherwise it is explicitly disabled so a half-configured environment never
causes failed uploads or warnings at runtime.

Langfuse is enabled only when both ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are
set and the optional ``langfuse`` package is installed. It works as a LangChain callback
handler attached to each run's config, so it never touches the process environment and a
Langfuse problem can only cost traces, never a research run.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from researchgraph.config.settings import Settings

logger = logging.getLogger(__name__)

_TRACING_ENV_KEYS = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")


def configure_tracing(settings: Settings) -> bool:
    """Apply tracing configuration to the process environment. Returns True if enabled."""
    if settings.tracing_enabled and settings.langchain_api_key is not None:
        api_key = settings.langchain_api_key.get_secret_value()
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGSMITH_PROJECT"] = settings.langchain_project
        if settings.langchain_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            os.environ["LANGSMITH_ENDPOINT"] = settings.langchain_endpoint
        logger.info("LangSmith tracing enabled (project=%s)", settings.langchain_project)
        return True

    for key in _TRACING_ENV_KEYS:
        os.environ[key] = "false"
    if settings.langchain_tracing_v2:
        logger.warning(
            "LANGCHAIN_TRACING_V2 is set but LANGCHAIN_API_KEY is missing; tracing disabled"
        )
    return False


class LangfuseTracing:
    """Hands out one Langfuse callback handler per run and flushes buffered traces."""

    def __init__(self, client: Any, handler_cls: type[BaseCallbackHandler], public_key: str):
        self._client = client
        self._handler_cls = handler_cls
        self._public_key = public_key

    def callback(self, research_id: str) -> BaseCallbackHandler | None:
        """Handler whose trace id is derived from the run id, so every segment of a run
        (planning, post-approval research, resume after restart) lands in one trace."""
        try:
            trace_id = self._client.create_trace_id(seed=research_id)
            return self._handler_cls(  # type: ignore[call-arg]
                public_key=self._public_key, trace_context={"trace_id": trace_id}
            )
        except Exception:
            logger.warning("Langfuse handler unavailable for run %s", research_id, exc_info=True)
            return None

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.warning("Langfuse flush failed; recent traces may be lost", exc_info=True)


def configure_langfuse(settings: Settings) -> LangfuseTracing | None:
    """Create the Langfuse client if both keys are set. Returns None when disabled."""
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        if settings.langfuse_public_key is not None or settings.langfuse_secret_key is not None:
            logger.warning(
                "Only one of LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY is set; Langfuse disabled"
            )
        else:
            logger.info("Langfuse tracing disabled (LANGFUSE_PUBLIC_KEY/SECRET_KEY not set)")
        return None
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning(
            "Langfuse keys are set but the package is missing "
            "(pip install 'researchgraph[langfuse]'); Langfuse disabled"
        )
        return None

    public_key = settings.langfuse_public_key.get_secret_value()
    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_host,
        )
    except Exception:
        logger.warning("Langfuse client could not be created; Langfuse disabled", exc_info=True)
        return None
    logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
    return LangfuseTracing(client, CallbackHandler, public_key)
