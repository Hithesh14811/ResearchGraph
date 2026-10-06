"""Evidence extractor: passages → structured, provenance-carrying evidence.

The model references passages by short IDs (P1, P2, ...) and must quote verbatim. Every
quote is then verified against the passage text; items whose quote cannot be found are
discarded. Source IDs, URLs and titles are attached in code from the passage's provenance,
never taken from model output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from researchgraph.agents.base import (
    UNTRUSTED_CONTENT_POLICY,
    call_structured,
    make_prompt,
    render_block,
)
from researchgraph.core.text import (
    normalize_for_match,
    quote_in_text,
    strip_citation_markers,
    truncate,
)
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.common import stable_id
from researchgraph.schemas.evidence import Evidence, Passage
from researchgraph.schemas.llm import EvidenceExtractionOutput
from researchgraph.schemas.research import ResearchTask
from researchgraph.schemas.sources import Source

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SYSTEM = """You extract evidence for an autonomous research system.

From the passages, extract factual claims that help answer the subquestion.
- The `quote` must be copied EXACTLY from the passage given by `passage_id` (quotes are verified automatically; paraphrased quotes are discarded).
- One specific claim per item (numbers, conditions, datasets, comparisons). The claim must be faithful to the quote and never stronger than it.
- Skip passages that are irrelevant to the research question, purely promotional, or only restate the question. Never extract titles or headings.
- If nothing in the passages bears on the research question, return an empty list — do not stretch.
- confidence = how directly the quote supports the claim; relevance = how much it helps answer the subquestion.
- Extract at most {max_items} items, prioritising empirical and benchmark results.
{policy}"""

HUMAN = """{request}

{passages}"""

PROMPT = make_prompt(SYSTEM, HUMAN)
MAX_ITEMS = 12


@dataclass(frozen=True)
class ExtractionResult:
    evidence: list[Evidence]
    proposed: int
    rejected_unverified: int


def budget_passages(passages: list[Passage], max_chars: int) -> list[Passage]:
    selected: list[Passage] = []
    used = 0
    for passage in passages:
        if used + len(passage.text) > max_chars and selected:
            break
        selected.append(passage)
        used += len(passage.text)
    return selected


async def extract_evidence(
    ctx: ResearchContext,
    *,
    task: ResearchTask,
    passages: list[Passage],
    sources: Mapping[str, Source],
    research_question: str = "",
) -> ExtractionResult:
    passages = budget_passages(passages, ctx.settings.max_extraction_context_chars)
    if not passages:
        return ExtractionResult(evidence=[], proposed=0, rejected_unverified=0)
    short_ids = {f"P{i + 1}": p for i, p in enumerate(passages)}
    passages_block = render_block(
        "passages",
        [
            {
                "passage_id": pid,
                "source_title": truncate(sources[p.source_id].title, 160)
                if p.source_id in sources
                else "",
                "published": str(sources[p.source_id].published_date or "unknown")
                if p.source_id in sources
                else "unknown",
                "text": p.text,
            }
            for pid, p in short_ids.items()
        ],
    )
    request = render_block(
        "extraction_request",
        {
            "research_question": research_question,
            "subquestion_id": task.subquestion_id,
            "subquestion": task.question,
            "search_intent": [q.query for q in task.queries],
            "focus": task.focus,
            "max_items": MAX_ITEMS,
        },
    )
    output = await call_structured(
        ctx,
        tier=ModelTier.FAST,
        prompt=PROMPT,
        variables={
            "request": request,
            "passages": passages_block,
            "max_items": MAX_ITEMS,
            "policy": UNTRUSTED_CONTENT_POLICY,
        },
        schema=EvidenceExtractionOutput,
        operation="evidence_extraction",
    )

    evidence: dict[str, Evidence] = {}
    rejected = 0
    for item in output.evidence[:MAX_ITEMS]:
        passage = short_ids.get(item.passage_id.strip().upper())
        source = sources.get(passage.source_id) if passage else None
        if passage is None or source is None or not quote_in_text(item.quote, passage.text):
            rejected += 1
            continue
        evidence_id = stable_id(
            "E", task.subquestion_id, source.id, normalize_for_match(item.quote)
        )
        publication = ", ".join(
            part
            for part in (
                source.venue,
                str(source.published_date.year) if source.published_date else None,
            )
            if part
        )
        evidence[evidence_id] = Evidence(
            id=evidence_id,
            subquestion_id=task.subquestion_id,
            claim=truncate(strip_citation_markers(item.claim), 600),
            supporting_text=truncate(item.quote, 800),
            source_id=source.id,
            source_url=source.url,
            source_title=source.title,
            publication=publication or None,
            published_date=source.published_date,
            passage_id=passage.id,
            chunk_index=passage.chunk_index,
            confidence=item.confidence,
            relevance=item.relevance,
            evidence_type=item.evidence_type,
            quote_verified=True,
            iteration=task.iteration,
        )
    return ExtractionResult(
        evidence=list(evidence.values()),
        proposed=len(output.evidence),
        rejected_unverified=rejected,
    )
