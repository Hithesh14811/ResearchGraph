"""Query writer: turns subquestions (and evidence gaps) into optimised, non-duplicate searches.

One batched call covers every subquestion in the round, rather than one call per
subquestion, to minimise LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from researchgraph.agents.base import call_structured, make_prompt, render_block
from researchgraph.core.text import normalize_for_match
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.llm import QueryGenerationOutput
from researchgraph.schemas.research import ResearchGap, ResearchSubquestion, SearchQuery

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SYSTEM = """You write web and academic search queries for an autonomous research system.

For each target subquestion, write 1-{max_queries} search queries that maximise the chance of finding credible primary sources:
- Use precise keyword phrases (technical terms, benchmark/dataset names, metric names), not full sentences.
- Vary the queries: different angles, synonyms, or specific sub-aspects. Never repeat a query from <search_history>.
- If a target has a "gap", the previous research round failed to find enough credible evidence for that aspect: write queries that specifically target the gap.
- Choose preferred_sources per target: 'academic' for empirical results, 'web' for documentation/practice, 'mixed' otherwise."""

HUMAN = "{request}"

PROMPT = make_prompt(SYSTEM, HUMAN)


@dataclass(frozen=True)
class QueryTarget:
    subquestion: ResearchSubquestion
    gap: ResearchGap | None = None

    def draft_queries(self) -> list[str]:
        if self.gap is not None and self.gap.suggested_queries:
            return self.gap.suggested_queries
        return [q.query for q in self.subquestion.search_queries]


def dedupe_queries(queries: list[SearchQuery], history: set[str], limit: int) -> list[SearchQuery]:
    seen = set(history)
    unique = []
    for query in queries:
        key = normalize_for_match(query.query)
        if key in seen or len(key) < 3:
            continue
        seen.add(key)
        unique.append(query)
    return unique[:limit]


def fallback_queries(target: QueryTarget, history: set[str], limit: int) -> list[SearchQuery]:
    """Deterministic queries: drafts first, then requirement-specific variants."""
    pref = target.subquestion.preferred_sources
    candidates = [
        SearchQuery(query=q[:300], source_preference=pref)
        for q in target.draft_queries()
        if len(q) >= 3
    ]
    for requirement in target.subquestion.information_requirements:
        candidates.append(
            SearchQuery(
                query=f"{target.subquestion.question[:200]} {requirement}"[:300],
                source_preference=pref,
            )
        )
    return dedupe_queries(candidates, history, limit)


async def write_queries(
    ctx: ResearchContext,
    *,
    objective: str,
    targets: list[QueryTarget],
    history: set[str],
) -> dict[str, list[SearchQuery]]:
    limit = ctx.settings.max_queries_per_subquestion
    request = render_block(
        "query_request",
        {
            "objective": objective,
            "targets": [
                {
                    "subquestion_id": t.subquestion.id,
                    "question": t.subquestion.question,
                    "information_requirements": t.subquestion.information_requirements,
                    "preferred_sources": t.subquestion.preferred_sources,
                    "gap": t.gap.description if t.gap else None,
                    "draft_queries": t.draft_queries(),
                }
                for t in targets
            ],
            "search_history": sorted(history)[-60:],
            "max_queries": limit,
        },
    )
    output = await call_structured(
        ctx,
        tier=ModelTier.FAST,
        prompt=PROMPT,
        variables={"request": request, "max_queries": limit},
        schema=QueryGenerationOutput,
        operation="query_generation",
    )
    by_id = {t.subquestion.id: t for t in targets}
    result: dict[str, list[SearchQuery]] = {}
    for plan in output.plans:
        if plan.subquestion_id not in by_id:
            continue
        queries = [
            SearchQuery(query=q[:300], source_preference=plan.preferred_sources)
            for q in plan.queries
            if len(q.strip()) >= 3
        ]
        result[plan.subquestion_id] = dedupe_queries(queries, history, limit)
    return result
