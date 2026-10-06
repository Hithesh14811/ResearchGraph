"""Runtime dependencies injected into graph nodes via LangGraph's ``context_schema``.

The context is *not* part of graph state and is never checkpointed: it holds live clients
(models, search, fetcher, vector index). After a restart a fresh context is built and the
run resumes from its checkpointed state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from researchgraph.config.settings import Settings
from researchgraph.core.faults import FaultInjector, NoFaults
from researchgraph.llm.factory import ModelRegistry
from researchgraph.retrieval.index import EvidenceIndex
from researchgraph.tools.fetcher import Fetcher
from researchgraph.tools.search.service import SearchService


@dataclass(frozen=True)
class ResearchContext:
    settings: Settings
    models: ModelRegistry
    search: SearchService
    fetcher: Fetcher
    index: EvidenceIndex
    faults: FaultInjector = field(default_factory=NoFaults)
