"""Optional LangSmith tracing.

Tracing is enabled only when both ``LANGCHAIN_TRACING_V2=true`` *and* an API key are
configured. Otherwise it is explicitly disabled so a half-configured environment never
causes failed uploads or warnings at runtime.
"""

from __future__ import annotations

import logging
import os

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
