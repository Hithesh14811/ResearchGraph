import asyncio

import httpx
import numpy as np
import pytest

from researchgraph.core.errors import SearchProviderError
from researchgraph.demo.corpus import CorpusFetcher, CorpusSearchProvider, load_corpus
from researchgraph.demo.faults import ScenarioFaultInjector, parse_scenarios
from researchgraph.retrieval.embeddings import BatchingEmbeddings, HashingEmbeddings
from researchgraph.retrieval.index import EvidenceIndex, InMemoryBackend
from researchgraph.schemas.sources import SearchResult, Source, SourceType
from researchgraph.tools.documents import extract_document
from researchgraph.tools.search.base import SearchChannel
from researchgraph.tools.search.providers import (
    ArxivSearchProvider,
    SemanticScholarSearchProvider,
    TavilySearchProvider,
)
from researchgraph.tools.search.service import SearchService


def test_hashing_embeddings_are_deterministic_normalised_and_lexical() -> None:
    emb = HashingEmbeddings(256)
    a, b, c = emb.embed_documents(
        ["retrieval accuracy benchmark", "benchmark of retrieval accuracy", "cooking pasta recipes"]
    )
    assert a == emb.embed_query("retrieval accuracy benchmark")
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.dot(a, b) > np.dot(a, c)


def test_batching_embeddings_split_requests() -> None:
    wrapped = BatchingEmbeddings(HashingEmbeddings(64), batch_size=3)
    vectors = wrapped.embed_documents([f"text {i}" for i in range(7)])
    assert len(vectors) == 7 and wrapped.batches == 3 and wrapped.documents_embedded == 7


async def test_index_ingests_and_retrieves_with_research_and_source_scoping() -> None:
    index = EvidenceIndex(
        InMemoryBackend(BatchingEmbeddings(HashingEmbeddings(256), batch_size=16)),
        chunk_size=300,
        chunk_overlap=30,
    )
    doc = extract_document(
        b"Retrieval improved accuracy by 9 points. " * 20 + b"GPUs are fast. " * 30,
        content_type="text/plain",
        url="https://e.org/a",
        max_chars=10_000,
        max_pdf_pages=1,
    )
    src_a = Source(id="S-a", url="https://e.org/a", title="A", provider="t")
    src_b = Source(id="S-b", url="https://e.org/b", title="B", provider="t")
    result = await index.ingest(research_id="r1", source=src_a, extracted=doc)
    await index.ingest(research_id="r2", source=src_b, extracted=doc)
    assert result.chunk_count > 1
    chunks = await index.retrieve(
        research_id="r1", queries=["retrieval accuracy"], source_ids=None, k=3, max_per_source=2
    )
    assert chunks and all(
        c.source_id == "S-a" for c in chunks
    )  # never leaks another run's documents
    assert len(chunks) <= 2  # diversity cap per source
    assert "Retrieval" in chunks[0].document.page_content
    assert (
        await index.retrieve(research_id="r1", queries=["x"], source_ids=[], k=3, max_per_source=2)
        == []
    )
    await index.release("r1")
    assert (
        await index.retrieve(
            research_id="r1", queries=["retrieval"], source_ids=None, k=3, max_per_source=2
        )
        == []
    )


class _FakeProvider:
    def __init__(
        self, name: str, results: list[SearchResult] | Exception, delay: float = 0.0
    ) -> None:
        self.name = name
        self.results = results
        self.calls = 0
        self.delay = delay

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if isinstance(self.results, Exception):
            raise self.results
        return self.results


def _result(url: str, score: float) -> SearchResult:
    return SearchResult(url=url, title="t", provider="fake", score=score)


async def test_search_service_caches_coalesces_and_tolerates_partial_failure() -> None:
    good = _FakeProvider(
        "good",
        [_result("https://a.org/x?utm_source=1", 0.5), _result("https://a.org/x", 0.9)],
        delay=0.05,
    )
    bad = _FakeProvider("bad", SearchProviderError("down"))
    service = SearchService(web=[good, bad], academic=[])
    first, second = await asyncio.gather(
        service.search("RAG accuracy", channel=SearchChannel.WEB, max_results=5),
        service.search("rag  ACCURACY", channel=SearchChannel.WEB, max_results=5),
    )
    assert good.calls == 1 and service.cache_hits == 1  # concurrent duplicate shared one call
    assert len(first) == 1 and first[0].score == 0.9 and first == second  # deduped by canonical URL
    await service.search(
        "RAG accuracy", channel=SearchChannel.ACADEMIC, max_results=5
    )  # falls back to web providers
    with pytest.raises(SearchProviderError):
        await SearchService(web=[bad], academic=[]).search(
            "q", channel=SearchChannel.WEB, max_results=3
        )


ARXIV_XML = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Retrieval Study</title><summary>We find gains.</summary>
<published>2024-01-02T00:00:00Z</published><author><name>A. Author</name></author>
<link title="pdf" href="http://arxiv.org/pdf/2401.00001v1"/></entry></feed>"""


async def test_provider_response_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "arxiv" in request.url.host:
            assert "all:retrieval" in request.url.params["search_query"]
            return httpx.Response(200, text=ARXIV_XML)
        if "semanticscholar" in request.url.host:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "title": "S2 paper",
                            "url": "https://www.semanticscholar.org/p/1",
                            "abstract": "abs",
                            "year": 2023,
                            "authors": [{"name": "B"}],
                            "citationCount": 42,
                            "openAccessPdf": {"url": "https://example.org/p.pdf"},
                            "externalIds": {"DOI": "10.1/x"},
                        }
                    ]
                },
            )
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Web",
                        "url": "https://docs.example.org/a",
                        "content": "c",
                        "score": 0.7,
                    },
                    {"title": "Bad", "url": "ftp://bad", "content": "c"},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    arxiv = await ArxivSearchProvider(client, min_interval_seconds=0).search(
        "retrieval accuracy gains", max_results=3
    )
    assert (
        arxiv[0].url == "https://arxiv.org/pdf/2401.00001v1"
        and arxiv[0].type_hint is SourceType.PRIMARY_RESEARCH
    )
    assert arxiv[0].authors == ["A. Author"] and str(arxiv[0].published_date) == "2024-01-02"
    s2 = await SemanticScholarSearchProvider(client).search("x", max_results=3)
    assert (
        s2[0].url == "https://example.org/p.pdf"
        and s2[0].citation_count == 42
        and s2[0].doi == "10.1/x"
    )
    tavily = await TavilySearchProvider("key", client).search("x", max_results=3)
    assert [r.url for r in tavily] == ["https://docs.example.org/a"]  # unsafe URL dropped
    assert (
        ArxivSearchProvider.build_query("What practical guidance sources")
        == "all:practical AND all:guidance AND all:sources"
    )


async def test_demo_corpus_is_labelled_synthetic_and_servable_in_every_format() -> None:
    docs = load_corpus()
    assert len(docs) >= 20 and all(d.title.startswith("[Synthetic]") for d in docs)
    assert all(".example" in d.url for d in docs)  # reserved documentation domains only
    assert {d.format for d in docs} == {"pdf", "html", "markdown"}
    results = await CorpusSearchProvider(SearchChannel.ACADEMIC).search(
        "fine-tuning knowledge injection", max_results=3
    )
    assert (
        results
        and "Fine-Tuning" in results[0].title
        and results[0].type_hint is SourceType.PRIMARY_RESEARCH
    )
    fetcher = CorpusFetcher()
    for doc in docs:
        fetched = await fetcher.fetch(doc.url)
        text = extract_document(
            fetched.body,
            content_type=fetched.content_type,
            url=doc.url,
            max_chars=50_000,
            max_pdf_pages=10,
        ).text
        assert "Synthetic demo document" in text


def test_fault_injector_scenarios_are_bounded() -> None:
    assert parse_scenarios("all") == parse_scenarios(
        [
            "failed_source",
            "llm_timeout",
            "invalid_output",
            "insufficient_evidence",
            "critic_rejection",
        ]
    )
    with pytest.raises(ValueError):
        parse_scenarios("meteor_strike")
    faults = ScenarioFaultInjector("failed_source,insufficient_evidence,llm_timeout")
    assert [faults.should_fail("failed_source", u) for u in ("a", "b", "c", "a")] == [
        True,
        True,
        False,
        True,
    ]
    assert faults.should_fail("insufficient_evidence", "SQ2:1") and faults.should_fail(
        "insufficient_evidence", "SQ2:1"
    )
    assert not faults.should_fail("insufficient_evidence", "SQ3:1") and not faults.should_fail(
        "insufficient_evidence", "SQ2:2"
    )
    assert faults.should_fail("llm_timeout", "PlannerOutput") and not faults.should_fail(
        "llm_timeout", "PlannerOutput"
    )
    assert not faults.should_fail("critic_rejection", "x")  # not enabled
