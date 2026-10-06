"""Dependency wiring: long-lived services built once from settings, plus per-run contexts."""

from __future__ import annotations

import logging

import httpx

from researchgraph.config.settings import Settings
from researchgraph.core.errors import ConfigurationError
from researchgraph.core.faults import FaultInjector, NoFaults
from researchgraph.graph.context import ResearchContext
from researchgraph.llm.factory import ModelRegistry, build_model_registry
from researchgraph.retrieval.embeddings import build_embeddings
from researchgraph.retrieval.index import EvidenceIndex, build_index
from researchgraph.tools.fetcher import USER_AGENT, Fetcher, HttpFetcher
from researchgraph.tools.search.base import SearchChannel, SearchProvider
from researchgraph.tools.search.providers import (
    ArxivSearchProvider,
    SemanticScholarSearchProvider,
    TavilySearchProvider,
)
from researchgraph.tools.search.service import SearchService

logger = logging.getLogger(__name__)


def _build_search(settings: Settings, client: httpx.AsyncClient) -> SearchService:
    if settings.search_provider == "mock":
        from researchgraph.demo.corpus import CorpusSearchProvider

        return SearchService(
            web=[CorpusSearchProvider(SearchChannel.WEB)],
            academic=[CorpusSearchProvider(SearchChannel.ACADEMIC)],
        )
    tavily_key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
    s2_key = (
        settings.semantic_scholar_api_key.get_secret_value()
        if settings.semantic_scholar_api_key
        else None
    )
    web: list[SearchProvider] = []
    academic: list[SearchProvider] = []
    if settings.search_provider in ("tavily", "auto") and tavily_key:
        web.append(TavilySearchProvider(tavily_key, client))
    elif settings.search_provider == "tavily":
        raise ConfigurationError("SEARCH_PROVIDER=tavily requires TAVILY_API_KEY")
    if settings.search_provider in ("arxiv", "auto"):
        academic.append(ArxivSearchProvider(client))
    if settings.search_provider in ("semantic_scholar", "auto"):
        academic.append(SemanticScholarSearchProvider(client, s2_key))
    return SearchService(web=web, academic=academic)


class ServiceContainer:
    """Owns shared clients (HTTP, search, vector index, models) for the process lifetime."""

    def __init__(
        self,
        settings: Settings,
        *,
        search: SearchService,
        fetcher: Fetcher,
        index: EvidenceIndex,
        models: ModelRegistry | None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.search = search
        self.fetcher = fetcher
        self.index = index
        self._models = models
        self._http = http_client

    @classmethod
    def from_settings(cls, settings: Settings) -> ServiceContainer:
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.fetch_timeout_seconds),
            headers={"User-Agent": USER_AGENT},
        )
        search = _build_search(settings, http)
        fetcher: Fetcher
        if settings.search_provider == "mock":
            from researchgraph.demo.corpus import CorpusFetcher

            fetcher = CorpusFetcher()
        else:
            fetcher = HttpFetcher(
                max_bytes=settings.max_document_bytes,
                timeout_seconds=settings.fetch_timeout_seconds,
                max_redirects=settings.max_redirects,
            )
        index = build_index(settings, build_embeddings(settings))
        # Real provider clients are shared; mock models are built per run (bound to its fault injector).
        models = None if settings.is_mock_llm else build_model_registry(settings)
        logger.info(
            "Services ready: llm=%s/%s search=%s embeddings=%s vector_store=%s",
            settings.llm_provider,
            settings.model_name,
            settings.search_provider,
            settings.embedding_provider,
            settings.resolved_vector_store,
        )
        return cls(
            settings, search=search, fetcher=fetcher, index=index, models=models, http_client=http
        )

    def context_for_run(self, *, failures: frozenset[str] = frozenset()) -> ResearchContext:
        faults: FaultInjector = NoFaults()
        if failures:
            from researchgraph.demo.faults import ScenarioFaultInjector

            faults = ScenarioFaultInjector(failures)
        models = (
            self._models
            if self._models is not None
            else build_model_registry(self.settings, faults=faults)
        )
        return ResearchContext(
            settings=self.settings,
            models=models,
            search=self.search,
            fetcher=self.fetcher,
            index=self.index,
            faults=faults,
        )

    async def aclose(self) -> None:
        await self.index.aclose()
        if isinstance(self.fetcher, HttpFetcher):
            await self.fetcher.aclose()
        if self._http is not None:
            await self._http.aclose()
