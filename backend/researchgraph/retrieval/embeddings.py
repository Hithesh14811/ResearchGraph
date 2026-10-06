"""Embedding models.

* ``HashingEmbeddings`` — deterministic feature-hashing embeddings (unigrams + bigrams).
  No network, no model download: used in demo mode, tests and as an offline fallback.
  It captures lexical rather than semantic similarity, which is documented as a tradeoff.
* ``BatchingEmbeddings`` — wraps any LangChain ``Embeddings`` with explicit batching,
  bounded retries, a query cache and counters. Vector stores call it transparently.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections import OrderedDict

import numpy as np
from langchain_core.embeddings import Embeddings

from researchgraph.config.settings import Settings
from researchgraph.core.errors import ConfigurationError, EmbeddingError
from researchgraph.core.retry import BackoffPolicy, retry_async
from researchgraph.core.text import content_terms

logger = logging.getLogger(__name__)


class HashingEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        terms = content_terms(text)
        features = [(t, 1.0) for t in terms] + [
            (f"{a}_{b}", 0.5) for a, b in itertools.pairwise(terms)
        ]
        for feature, weight in features:
            digest = int.from_bytes(
                hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little"
            )
            sign = 1.0 if digest >> 63 else -1.0
            vector[digest % self.dimensions] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class BatchingEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, *, batch_size: int, query_cache_size: int = 1024) -> None:
        self.inner = inner
        self.batch_size = batch_size
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = query_cache_size
        self._backoff = BackoffPolicy(max_attempts=3, initial_delay=0.5, max_delay=8.0)
        self.documents_embedded = 0
        self.batches = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                vectors.extend(self.inner.embed_documents(batch))
            except Exception as exc:
                raise EmbeddingError(f"Embedding batch failed: {exc}") from exc
            self.batches += 1
        self.documents_embedded += len(texts)
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            async def embed(chunk: list[str] = batch) -> list[list[float]]:
                return await self.inner.aembed_documents(chunk)

            try:
                vectors.extend(await retry_async(embed, policy=self._backoff))
            except Exception as exc:
                raise EmbeddingError(f"Embedding batch failed after retries: {exc}") from exc
            self.batches += 1
        self.documents_embedded += len(texts)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        vector = self.inner.embed_query(text)
        self._remember(text, vector)
        return vector

    async def aembed_query(self, text: str) -> list[float]:
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        try:
            vector = await retry_async(lambda: self.inner.aembed_query(text), policy=self._backoff)
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed after retries: {exc}") from exc
        self._remember(text, vector)
        return vector

    def _remember(self, text: str, vector: list[float]) -> None:
        self._query_cache[text] = vector
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)


def build_embeddings(settings: Settings) -> BatchingEmbeddings:
    inner: Embeddings
    if settings.embedding_provider == "hashing":
        inner = HashingEmbeddings(settings.embedding_dimensions)
    elif settings.embedding_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=openai requires 'langchain-openai'"
            ) from exc
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        inner = OpenAIEmbeddings(  # type: ignore[call-arg]
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=api_key,
        )
    else:  # pragma: no cover - guarded by Settings' Literal type
        raise ConfigurationError(f"Unknown embedding provider {settings.embedding_provider!r}")
    return BatchingEmbeddings(inner, batch_size=settings.embedding_batch_size)
