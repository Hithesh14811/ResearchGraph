"""Research-worker subgraph nodes: search (tool-using agent) → source processing →
passage retrieval → evidence extraction. One worker instance runs per subquestion per round,
in parallel with the others."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.runtime import Runtime

from researchgraph.agents.extractor import extract_evidence
from researchgraph.agents.researcher import build_research_messages
from researchgraph.core.errors import FetchError
from researchgraph.core.retry import retry_async
from researchgraph.core.text import coverage
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.instrumentation import emit, instrumented, workflow_error
from researchgraph.graph.state import WorkerState
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.common import WorkflowError, stable_id
from researchgraph.schemas.events import EventType
from researchgraph.schemas.evidence import Passage
from researchgraph.schemas.research import ResearchTask
from researchgraph.schemas.sources import DocumentFormat, SearchResult, Source
from researchgraph.services.source_quality import AUTHORITY_PRIOR, classify_source
from researchgraph.tools.documents import ExtractedDocument, extract_document
from researchgraph.tools.research_tools import RESEARCH_TOOLS, SEARCH_TOOL_NAMES
from researchgraph.tools.url_safety import canonicalize_url, validate_url_syntax

logger = logging.getLogger(__name__)

MIN_SNIPPET_CHARS_FOR_FALLBACK = 200
FETCH_CONCURRENCY = 4


def _label(state: Mapping[str, Any]) -> str:
    task = state.get("task")
    return f"[{task.subquestion_id}] " if task is not None else ""


def _sq(state: Mapping[str, Any]) -> str | None:
    task = state.get("task")
    return task.subquestion_id if task is not None else None


# --------------------------------------------------------------------------------------
# 1. Search agent (bounded tool loop)
# --------------------------------------------------------------------------------------
def _channel_tool(task: ResearchTask, preference: str) -> str:
    pref = preference if preference != "mixed" else task.preferred_sources
    return "web_search" if pref == "web" else "academic_search"


def planned_tool_calls(task: ResearchTask, limit: int) -> AIMessage:
    """Deterministic policy: execute the planned queries (alternating tools for 'mixed')."""
    calls = []
    for i, query in enumerate(task.queries[:limit]):
        name = _channel_tool(task, query.source_preference)
        if task.preferred_sources == "mixed" and query.source_preference == "mixed":
            name = "academic_search" if i % 2 == 0 else "web_search"
        calls.append(
            {
                "name": name,
                "args": {"query": query.query, "max_results": 5},
                "id": f"plan_{task.task_id}_{i}",
                "type": "tool_call",
            }
        )
    return AIMessage(content="", tool_calls=calls)


def _truncate_tool_calls(message: AIMessage, max_external: int) -> tuple[AIMessage, int]:
    """Enforce the tool budget by keeping only the first N external calls.

    Content blocks for dropped calls are removed too, so providers that echo tool_use
    blocks (e.g. Anthropic) still see a consistent conversation."""
    kept, external = [], 0
    for call in message.tool_calls:
        if call["name"] in SEARCH_TOOL_NAMES:
            if external >= max_external:
                continue
            external += 1
        kept.append(call)
    if len(kept) == len(message.tool_calls):
        return message, external
    kept_ids = {c["id"] for c in kept}
    content = message.content
    if isinstance(content, list):
        content = [
            b
            for b in content
            if not (
                isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") not in kept_ids
            )
        ]
    return AIMessage(
        content=content, tool_calls=kept, id=message.id, usage_metadata=message.usage_metadata
    ), external


def _degrade_search(state: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "search_done": True,
        "errors": [workflow_error("search_agent", exc, subquestion_id=_sq(state))],
    }


@instrumented(
    "search_agent", start="Searching for sources...", timeout_s=300, degrade=_degrade_search
)
async def search_agent_node(state: WorkerState, runtime: Runtime[ResearchContext]) -> dict:
    ctx, s = runtime.context, runtime.context.settings
    task = state["task"]
    rounds = state.get("search_rounds", 0)
    search_budget = max(1, task.tool_budget - s.max_sources_per_task)
    remaining = search_budget - state.get("tool_calls_used", 0)
    if rounds >= s.max_search_rounds_per_task or remaining <= 0:
        return {"search_done": True}

    history: list[BaseMessage] = list(state.get("messages") or [])
    new_messages: list[BaseMessage] = []
    if not history:
        new_messages = build_research_messages(task, state["question"])
        history = list(new_messages)

    errors: list[WorkflowError] = []
    if s.research_agent_mode == "deterministic":
        response = (
            planned_tool_calls(task, remaining)
            if rounds == 0
            else AIMessage(content="Planned searches complete.")
        )
    else:
        llm = ctx.models.get(ModelTier.FAST).bind_tools(RESEARCH_TOOLS)
        policy = ctx.models.policy
        try:
            raw = await retry_async(
                lambda: asyncio.wait_for(llm.ainvoke(history), timeout=policy.timeout_seconds),
                policy=policy.backoff,
            )
            response = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
        except Exception as exc:
            errors.append(workflow_error("search_agent", exc, subquestion_id=task.subquestion_id))
            emit(
                EventType.WARNING,
                f"{_label(state)}Research agent unavailable; executing planned searches.",
            )
            response = (
                planned_tool_calls(task, remaining)
                if rounds == 0
                else AIMessage(content="Stopping search.")
            )

    response, external_calls = _truncate_tool_calls(response, remaining)
    if response.tool_calls:
        emit(
            EventType.PROGRESS,
            f"{_label(state)}Research agent issued {len(response.tool_calls)} tool call(s).",
            subquestion_id=task.subquestion_id,
            calls=[c["name"] for c in response.tool_calls],
        )
    return {
        "messages": [*new_messages, response],
        "search_rounds": rounds + 1,
        "search_done": not response.tool_calls,
        "tool_calls_used": external_calls,
        "search_history": [
            str(c["args"].get("query", ""))
            for c in response.tool_calls
            if c["name"] in SEARCH_TOOL_NAMES
        ],
        "errors": errors,
    }


# --------------------------------------------------------------------------------------
# 2. Source processing: select → fetch → parse → clean → chunk → embed → index
# --------------------------------------------------------------------------------------
def collect_search_results(messages: Sequence[BaseMessage]) -> list[SearchResult]:
    results: dict[str, SearchResult] = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in SEARCH_TOOL_NAMES:
            continue
        for row in message.artifact or []:
            try:
                result = SearchResult.model_validate(row)
                validate_url_syntax(result.url)
            except ValueError:
                continue
            key = canonicalize_url(result.url)
            if key not in results or result.score > results[key].score:
                results[key] = result
    return list(results.values())


def source_from_result(result: SearchResult, subquestion_id: str) -> Source:
    return Source(
        id=stable_id("S", canonicalize_url(result.url)),
        url=result.url,
        title=result.title or result.url,
        provider=result.provider,
        authors=result.authors,
        published_date=result.published_date,
        venue=result.venue,
        doi=result.doi,
        citation_count=result.citation_count,
        type_hint=result.type_hint,
        snippet=result.snippet,
        subquestion_ids=[subquestion_id],
    )


def candidate_priority(source: Source, task: ResearchTask, provider_score: float) -> float:
    source_type, _ = classify_source(source)
    relevance = coverage(task.question, f"{source.title} {source.snippet}")
    return 0.35 * provider_score + 0.35 * AUTHORITY_PRIOR[source_type] + 0.30 * relevance


def select_diverse(ranked: list[Source], k: int) -> list[Source]:
    """Top-(k-1) by priority plus one *diversity slot* for the best-ranked source of a type
    not yet represented. Practitioner sources rarely outrank papers, but without them the
    contradiction check could never see where practice disagrees with research; quality
    scoring still down-weights them later."""
    if k <= 1 or len(ranked) <= k:
        return ranked[:k]
    chosen = ranked[: k - 1]
    types = {classify_source(s)[0] for s in chosen}
    slot = next((s for s in ranked[k - 1 :] if classify_source(s)[0] not in types), ranked[k - 1])
    return [*chosen, slot]


async def _process_source(
    ctx: ResearchContext, research_id: str, source: Source
) -> tuple[Source, WorkflowError | None]:
    s = ctx.settings
    try:
        if ctx.faults.should_fail("failed_source", source.url):
            raise FetchError(
                f"Simulated outage: {source.url} is unavailable (HTTP 503)",
                url=source.url,
                status_code=503,
            )
        fetched = await ctx.fetcher.fetch(source.url)
        extracted = extract_document(
            fetched.body,
            content_type=fetched.content_type,
            url=fetched.final_url,
            max_chars=s.max_document_chars,
            max_pdf_pages=s.max_pdf_pages,
        )
        enriched = source.model_copy(
            update={
                "title": source.title
                if source.title and source.title != source.url
                else (extracted.title or source.title),
                "authors": source.authors or extracted.authors[:12],
                "published_date": source.published_date or extracted.published_date,
                "venue": source.venue or extracted.venue,
                "doi": source.doi or extracted.doi,
                "document_format": extracted.format,
            }
        )
        ingest = await ctx.index.ingest(
            research_id=research_id, source=enriched, extracted=extracted
        )
        return enriched.model_copy(
            update={
                "status": "processed",
                "content_origin": "full_text",
                "chunk_count": ingest.chunk_count,
                "content_chars": ingest.content_chars,
                "signals": ingest.signals,
            }
        ), None
    except Exception as exc:
        error = workflow_error(
            "source_processing",
            exc,
            subquestion_id=source.subquestion_ids[0] if source.subquestion_ids else None,
        )
        if len(source.snippet) >= MIN_SNIPPET_CHARS_FOR_FALLBACK:
            # Degrade to the abstract/snippet returned by the search provider.
            try:
                snippet_doc = ExtractedDocument(
                    text=source.snippet, format=DocumentFormat.TEXT, title=source.title
                )
                ingest = await ctx.index.ingest(
                    research_id=research_id, source=source, extracted=snippet_doc
                )
                return source.model_copy(
                    update={
                        "status": "processed",
                        "content_origin": "search_snippet",
                        "chunk_count": ingest.chunk_count,
                        "content_chars": ingest.content_chars,
                        "error": f"{error.error_type}: {error.message}"[:300],
                        "document_format": DocumentFormat.TEXT,
                    }
                ), error
            except Exception as inner:
                logger.warning("Snippet fallback failed for %s: %s", source.url, inner)
        return source.model_copy(
            update={"status": "failed", "error": f"{error.error_type}: {error.message}"[:300]}
        ), error


def _degrade_processing(state: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "scope_source_ids": [],
        "errors": [workflow_error("source_processing", exc, subquestion_id=_sq(state))],
    }


@instrumented(
    "source_processing",
    start="Retrieving and processing sources...",
    timeout_s=600,
    degrade=_degrade_processing,
)
async def source_processing_node(state: WorkerState, runtime: Runtime[ResearchContext]) -> dict:
    ctx, s = runtime.context, runtime.context.settings
    task = state["task"]
    known: dict[str, Source] = state.get("known_sources") or {}
    results = collect_search_results(state.get("messages") or [])
    emit(
        EventType.PROGRESS,
        f"{_label(state)}Found {len(results)} candidate sources.",
        subquestion_id=task.subquestion_id,
        candidates=len(results),
    )

    reused: dict[str, Source] = {}
    fresh: list[tuple[float, Source]] = []
    for result in results:
        source = source_from_result(result, task.subquestion_id)
        if source.id in known:
            reused[source.id] = known[source.id].model_copy(
                update={"subquestion_ids": [task.subquestion_id]}
            )
        else:
            fresh.append((candidate_priority(source, task, result.score), source))
    fresh.sort(key=lambda pair: (pair[0], pair[1].id), reverse=True)

    fetch_budget = max(0, task.tool_budget - state.get("tool_calls_used", 0))
    selected = select_diverse([src for _, src in fresh], min(s.max_sources_per_task, fetch_budget))

    semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def bounded(source: Source) -> tuple[Source, WorkflowError | None]:
        async with semaphore:
            return await _process_source(ctx, state["research_id"], source)

    outcomes = await asyncio.gather(*(bounded(src) for src in selected))
    processed = {src.id: src for src, _ in outcomes}
    errors = [err for _, err in outcomes if err is not None]
    ok = [src for src in processed.values() if src.status == "processed"]
    failed = len(processed) - len(ok)
    emit(
        EventType.PROGRESS,
        f"{_label(state)}Processed {len(ok)} new source(s)"
        + (f", {failed} unavailable" if failed else "")
        + (f", reused {len(reused)} known" if reused else "")
        + ".",
        subquestion_id=task.subquestion_id,
        processed=len(ok),
        failed=failed,
        reused=len(reused),
    )
    for src in processed.values():
        if src.status == "failed":
            emit(
                EventType.WARNING,
                f"{_label(state)}Source unavailable: {src.title[:80]} ({src.error})",
            )
    scope = [src.id for src in ok] + list(reused)
    return {
        "sources": {**reused, **processed},
        "scope_source_ids": scope,
        "tool_calls_used": len(selected),
        "errors": errors,
    }


# --------------------------------------------------------------------------------------
# 3. Passage retrieval (source-aware RAG)
# --------------------------------------------------------------------------------------
def _degrade_retrieval(state: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "passages": [],
        "errors": [workflow_error("passage_retrieval", exc, subquestion_id=_sq(state))],
    }


@instrumented(
    "passage_retrieval",
    start="Retrieving relevant passages...",
    timeout_s=180,
    degrade=_degrade_retrieval,
)
async def passage_retrieval_node(state: WorkerState, runtime: Runtime[ResearchContext]) -> dict:
    ctx, s = runtime.context, runtime.context.settings
    task = state["task"]
    scope = state.get("scope_source_ids") or []
    if not scope:
        emit(EventType.WARNING, f"{_label(state)}No usable sources this round.")
        return {"passages": []}
    queries = [task.question, *(q.query for q in task.queries[:3])]
    if task.focus:
        queries.append(task.focus)
    chunks = await ctx.index.retrieve(
        research_id=state["research_id"],
        queries=queries,
        source_ids=scope,
        k=s.passages_per_task,
        max_per_source=s.max_chunks_per_source,
    )
    passages = [
        Passage(
            id=str(c.document.metadata.get("chunk_id")),
            source_id=c.source_id,
            chunk_index=c.chunk_index,
            text=c.document.page_content,
            score=round(c.score, 4),
        )
        for c in chunks
    ]
    emit(
        EventType.PROGRESS,
        f"{_label(state)}Retrieved {len(passages)} passages from {len({p.source_id for p in passages})} sources.",
    )
    return {"passages": passages}


# --------------------------------------------------------------------------------------
# 4. Evidence extraction
# --------------------------------------------------------------------------------------
def _degrade_extraction(state: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {"errors": [workflow_error("evidence_extraction", exc, subquestion_id=_sq(state))]}


@instrumented(
    "evidence_extraction",
    start="Extracting evidence...",
    timeout_s=300,
    degrade=_degrade_extraction,
)
async def evidence_extraction_node(state: WorkerState, runtime: Runtime[ResearchContext]) -> dict:
    ctx = runtime.context
    task = state["task"]
    passages = state.get("passages") or []
    if not passages:
        return {}
    sources = {**(state.get("known_sources") or {}), **(state.get("sources") or {})}
    result = await extract_evidence(
        ctx, task=task, passages=passages, sources=sources, research_question=state["question"]
    )
    emit(
        EventType.PROGRESS,
        f"{_label(state)}Extracted {len(result.evidence)} evidence items"
        + (
            f" ({result.rejected_unverified} rejected: quote not found in source)"
            if result.rejected_unverified
            else ""
        )
        + ".",
        subquestion_id=task.subquestion_id,
        evidence=len(result.evidence),
        proposed=result.proposed,
        rejected=result.rejected_unverified,
    )
    return {"evidence": {e.id: e for e in result.evidence}}
