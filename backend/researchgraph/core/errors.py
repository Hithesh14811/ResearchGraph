"""Structured exception hierarchy. Every error raised by ResearchGraph code carries a
stable ``code`` so it can be recorded in graph state and surfaced through the API."""

from __future__ import annotations


class ResearchGraphError(Exception):
    code = "researchgraph_error"

    def __init__(self, message: str, *, recoverable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.recoverable = recoverable


class ConfigurationError(ResearchGraphError):
    code = "configuration_error"


class UnsafeURLError(ResearchGraphError, ValueError):
    code = "unsafe_url"


class FetchError(ResearchGraphError):
    code = "fetch_failed"

    def __init__(self, message: str, *, url: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class DocumentTooLargeError(FetchError):
    code = "document_too_large"


class UnsupportedContentTypeError(FetchError):
    code = "unsupported_content_type"


class DocumentExtractionError(ResearchGraphError):
    code = "document_extraction_failed"


class SearchProviderError(ResearchGraphError):
    code = "search_failed"


class EmbeddingError(ResearchGraphError):
    code = "embedding_failed"


class StructuredOutputError(ResearchGraphError):
    """The model failed to produce schema-valid output after all repair attempts."""

    code = "invalid_structured_output"

    def __init__(self, message: str, *, schema: str, attempts: int) -> None:
        super().__init__(message)
        self.schema = schema
        self.attempts = attempts


class LLMCallError(ResearchGraphError):
    """A model call failed after retries (timeouts, rate limits, provider outages)."""

    code = "llm_call_failed"


class TooManyRunsError(ResearchGraphError):
    code = "too_many_runs"


class RunNotFoundError(ResearchGraphError):
    code = "run_not_found"


class InvalidRunStateError(ResearchGraphError):
    code = "invalid_run_state"
