"""Source models: search results, retrieved sources and their quality scores."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from researchgraph.schemas.common import DomainModel, utc_now


class SourceType(StrEnum):
    PRIMARY_RESEARCH = "primary_research"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    INSTITUTIONAL = "institutional"
    REPUTABLE_TECHNICAL = "reputable_technical"
    NEWS = "news"
    BLOG = "blog"
    FORUM = "forum"
    UNKNOWN = "unknown"


class DocumentFormat(StrEnum):
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class SearchResult(DomainModel):
    """A raw hit from a search provider, before any fetching."""

    url: str
    title: str
    snippet: str = ""
    provider: str
    published_date: date | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    type_hint: SourceType | None = None
    score: float = 0.0


SourceStatus = Literal["discovered", "processed", "failed"]


class ContentSignals(DomainModel):
    """Cheap structural signals computed from document text; inputs to quality scoring."""

    word_count: int = 0
    reference_count: int = 0
    methodology_terms: int = 0
    numeric_density: float = 0.0
    has_abstract: bool = False


class Source(DomainModel):
    id: str
    url: str
    title: str
    provider: str
    authors: list[str] = Field(default_factory=list)
    published_date: date | None = None
    venue: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    document_format: DocumentFormat | None = None
    type_hint: SourceType | None = None
    snippet: str = ""
    content_chars: int = 0
    chunk_count: int = 0
    content_origin: Literal["full_text", "search_snippet", "none"] = "none"
    signals: ContentSignals | None = None
    subquestion_ids: list[str] = Field(default_factory=list)
    status: SourceStatus = "discovered"
    error: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)


class QualityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceQuality(DomainModel):
    """Structured, explainable quality score for one source."""

    source_id: str
    source_type: SourceType
    authority: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    primary_source: float = Field(ge=0, le=1)
    methodology: float = Field(ge=0, le=1)
    citation_signal: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    tier: QualityTier
    rationale: list[str] = Field(default_factory=list)
