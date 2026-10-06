"""Evidence models. Every evidence item carries full provenance back to a source chunk."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import Field

from researchgraph.schemas.common import DomainModel


class EvidenceType(StrEnum):
    EMPIRICAL_RESULT = "empirical_result"
    BENCHMARK_RESULT = "benchmark_result"
    METHODOLOGY = "methodology"
    EXPERT_OPINION = "expert_opinion"
    CASE_STUDY = "case_study"
    LIMITATION = "limitation"
    BACKGROUND = "background"


class Passage(DomainModel):
    """A retrieved chunk handed to the evidence extractor."""

    id: str
    source_id: str
    chunk_index: int
    text: str
    score: float = 0.0


class Evidence(DomainModel):
    id: str
    subquestion_id: str
    claim: str
    supporting_text: str = Field(description="Verbatim quote from the source chunk.")
    source_id: str
    source_url: str
    source_title: str
    publication: str | None = None
    published_date: date | None = None
    passage_id: str
    chunk_index: int
    confidence: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    evidence_type: EvidenceType
    quote_verified: bool = True
    iteration: int = 1


class Contradiction(DomainModel):
    id: str
    subquestion_id: str | None
    evidence_ids: list[str] = Field(min_length=2, max_length=2)
    description: str
    severity: Literal["minor", "major"] = "minor"
