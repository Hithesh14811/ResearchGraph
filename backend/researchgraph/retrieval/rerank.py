"""Hybrid reranking: blend vector similarity with lexical coverage, then enforce diversity.

Pure vector search over-rewards chunks that are topically similar but lack the specific
terms a subquestion asks about; pure lexical search misses paraphrases. Blending the two
and capping chunks per source gives the extractor focused, diverse passages.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from researchgraph.core.text import coverage, jaccard


@dataclass(frozen=True)
class ScoredChunk:
    document: Document
    vector_score: float
    lexical_score: float
    score: float

    @property
    def source_id(self) -> str:
        return str(self.document.metadata["source_id"])

    @property
    def chunk_index(self) -> int:
        return int(self.document.metadata.get("chunk_index", 0))


def rerank(
    query: str,
    candidates: list[tuple[Document, float]],
    *,
    k: int,
    max_per_source: int,
    vector_weight: float = 0.6,
    near_duplicate_threshold: float = 0.85,
) -> list[ScoredChunk]:
    scored = [
        ScoredChunk(
            document=doc,
            vector_score=max(0.0, min(1.0, vector_score)),
            lexical_score=(lexical := coverage(query, doc.page_content)),
            score=vector_weight * max(0.0, min(1.0, vector_score)) + (1 - vector_weight) * lexical,
        )
        for doc, vector_score in candidates
    ]
    scored.sort(key=lambda c: c.score, reverse=True)

    selected: list[ScoredChunk] = []
    per_source: dict[str, int] = {}
    for chunk in scored:
        if per_source.get(chunk.source_id, 0) >= max_per_source:
            continue
        if any(
            jaccard(chunk.document.page_content, s.document.page_content)
            >= near_duplicate_threshold
            for s in selected
        ):
            continue
        selected.append(chunk)
        per_source[chunk.source_id] = per_source.get(chunk.source_id, 0) + 1
        if len(selected) >= k:
            break
    return selected
