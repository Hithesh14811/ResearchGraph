"""Source-aware vector index used by the research workers.

Documents → cleaning → chunking (+ provenance metadata) → batched embeddings → vector
store. Retrieval is always scoped to one research run and, optionally, a set of sources,
then reranked for relevance and diversity.

Two interchangeable backends implement the same small protocol:

* ``InMemoryBackend`` — LangChain ``InMemoryVectorStore`` (local dev, demo, tests).
* ``PGVectorBackend`` — ``langchain_postgres.PGVectorStore`` on PostgreSQL + pgvector,
  with ``research_id``/``source_id`` as indexed columns for filtered search.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from researchgraph.config.settings import Settings
from researchgraph.core.errors import ConfigurationError
from researchgraph.retrieval.embeddings import BatchingEmbeddings
from researchgraph.retrieval.loader import SourceDocumentLoader, build_splitter, chunk_document
from researchgraph.retrieval.rerank import ScoredChunk, rerank
from researchgraph.schemas.sources import ContentSignals, Source
from researchgraph.tools.documents import ExtractedDocument

logger = logging.getLogger(__name__)


class VectorBackend(Protocol):
    async def add(self, documents: list[Document]) -> None: ...

    async def search(
        self, query: str, *, k: int, research_id: str, source_ids: list[str] | None
    ) -> list[tuple[Document, float]]: ...

    async def delete_research(self, research_id: str) -> None: ...


class InMemoryBackend:
    def __init__(self, embeddings: BatchingEmbeddings) -> None:
        self._store = InMemoryVectorStore(embedding=embeddings)
        self._ids_by_research: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        ids = [str(d.id) for d in documents]
        await self._store.aadd_documents(documents, ids=ids)
        async with self._lock:
            for doc in documents:
                self._ids_by_research.setdefault(doc.metadata["research_id"], set()).add(
                    str(doc.id)
                )

    async def search(
        self, query: str, *, k: int, research_id: str, source_ids: list[str] | None
    ) -> list[tuple[Document, float]]:
        allowed = set(source_ids) if source_ids is not None else None

        def _filter(doc: Document) -> bool:
            return doc.metadata.get("research_id") == research_id and (
                allowed is None or doc.metadata.get("source_id") in allowed
            )

        return await self._store.asimilarity_search_with_score(query, k=k, filter=_filter)

    async def delete_research(self, research_id: str) -> None:
        async with self._lock:
            ids = self._ids_by_research.pop(research_id, set())
        if ids:
            await self._store.adelete(list(ids))


class PGVectorBackend:
    """pgvector backend; tables are created idempotently on first use.

    The SQLAlchemy engine is created lazily on the caller's event loop and wrapped with
    ``PGEngine.from_engine`` (``from_connection_string`` would run every query on a private
    background-thread loop, adding a thread hop per call).
    """

    TABLE = "researchgraph_chunks"

    def __init__(self, database_url: str, embeddings: BatchingEmbeddings, dimensions: int) -> None:
        self._url = database_url
        self._embeddings = embeddings
        self._dimensions = dimensions
        self._store: Any = None
        self._engine: Any = None
        self._lock = asyncio.Lock()

    async def _get_store(self) -> Any:
        if self._store is not None:
            return self._store
        async with self._lock:
            if self._store is None:
                try:
                    from langchain_postgres import Column, PGEngine, PGVectorStore
                except ImportError as exc:  # pragma: no cover - depends on optional extra
                    raise ConfigurationError(
                        "VECTOR_STORE=pgvector requires 'pip install researchgraph[pgvector]'"
                    ) from exc
                from sqlalchemy.ext.asyncio import create_async_engine

                self._engine = create_async_engine(self._url, pool_pre_ping=True)
                engine = PGEngine.from_engine(self._engine)
                try:
                    await engine.ainit_vectorstore_table(
                        self.TABLE,
                        self._dimensions,
                        id_column=Column("langchain_id", "TEXT"),
                        metadata_columns=[
                            Column("research_id", "TEXT"),
                            Column("source_id", "TEXT"),
                            Column("chunk_index", "INTEGER"),
                        ],
                    )
                except Exception as exc:  # table already exists
                    if "already exists" not in str(exc).lower():
                        raise
                self._store = await PGVectorStore.create(
                    engine,
                    embedding_service=self._embeddings,
                    table_name=self.TABLE,
                    id_column="langchain_id",
                    metadata_columns=["research_id", "source_id", "chunk_index"],
                )
        return self._store

    async def add(self, documents: list[Document]) -> None:
        if documents:
            store = await self._get_store()
            await store.aadd_documents(documents, ids=[str(d.id) for d in documents])

    async def search(
        self, query: str, *, k: int, research_id: str, source_ids: list[str] | None
    ) -> list[tuple[Document, float]]:
        store = await self._get_store()
        flt: dict[str, Any] = {"research_id": {"$eq": research_id}}
        if source_ids is not None:
            flt["source_id"] = {"$in": list(source_ids)}
        results = await store.asimilarity_search_with_score(query, k=k, filter=flt)
        # PGVectorStore returns cosine *distance*; convert to similarity.
        return [(doc, 1.0 - float(distance)) for doc, distance in results]

    async def delete_research(self, research_id: str) -> None:
        return None  # chunks are retained in Postgres for auditability

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()


@dataclass(frozen=True)
class IngestResult:
    chunk_count: int
    content_chars: int
    signals: ContentSignals


class EvidenceIndex:
    def __init__(self, backend: VectorBackend, *, chunk_size: int, chunk_overlap: int) -> None:
        self.backend = backend
        self._splitter = build_splitter(chunk_size, chunk_overlap)

    async def ingest(
        self, *, research_id: str, source: Source, extracted: ExtractedDocument
    ) -> IngestResult:
        loader = SourceDocumentLoader(research_id=research_id, source=source, extracted=extracted)
        documents = [doc async for doc in loader.alazy_load()]
        chunks = [chunk for doc in documents for chunk in chunk_document(doc, self._splitter)]
        await self.backend.add(chunks)
        return IngestResult(
            chunk_count=len(chunks),
            content_chars=sum(len(d.page_content) for d in documents),
            signals=loader.signals or ContentSignals(),
        )

    async def retrieve(
        self,
        *,
        research_id: str,
        queries: list[str],
        source_ids: list[str] | None,
        k: int,
        max_per_source: int,
    ) -> list[ScoredChunk]:
        """Multi-query retrieval (union, best score per chunk) followed by hybrid reranking."""
        if source_ids is not None and not source_ids:
            return []
        per_query = max(k, 6)
        batches = await asyncio.gather(
            *(
                self.backend.search(q, k=per_query, research_id=research_id, source_ids=source_ids)
                for q in queries
                if q.strip()
            )
        )
        best: dict[str, tuple[Document, float]] = {}
        for batch in batches:
            for doc, score in batch:
                key = str(doc.metadata.get("chunk_id") or doc.id)
                if key not in best or score > best[key][1]:
                    best[key] = (doc, score)
        return rerank(" ".join(queries), list(best.values()), k=k, max_per_source=max_per_source)

    async def release(self, research_id: str) -> None:
        await self.backend.delete_research(research_id)

    async def aclose(self) -> None:
        close = getattr(self.backend, "aclose", None)
        if close is not None:
            await close()


def build_index(settings: Settings, embeddings: BatchingEmbeddings) -> EvidenceIndex:
    backend: VectorBackend
    if settings.resolved_vector_store == "pgvector":
        backend = PGVectorBackend(settings.database_url, embeddings, settings.embedding_dimensions)
    else:
        backend = InMemoryBackend(embeddings)
    return EvidenceIndex(
        backend, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
