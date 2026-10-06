"""Planner: decomposes a research question into an approvable, parallelisable plan."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from researchgraph.agents.base import call_structured, make_prompt, render_block
from researchgraph.core.text import keywords
from researchgraph.llm.factory import ModelTier
from researchgraph.schemas.llm import PlannerOutput
from researchgraph.schemas.research import ResearchPlan, ResearchSubquestion, SearchQuery

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SYSTEM = """You are the planning component of an autonomous research system that produces evidence-backed technical reports.

Decompose the user's research question into a research plan:
- Write between 3 and {max_subquestions} subquestions. Each must be answerable independently (they are researched in parallel) and together they must fully cover the question.
- Cover, where relevant: the mechanisms/approaches involved, empirical results and benchmarks, limitations and failure modes, cost and operational tradeoffs, and practical recommendations.
- For each subquestion list concrete information requirements (e.g. "exact-match accuracy on domain QA benchmarks", "inference cost per 1K queries").
- Write 2-3 search-engine queries per subquestion: short keyword phrases, not sentences.
- Choose preferred sources: 'academic' for empirical/benchmark questions, 'web' for documentation and industry practice, 'mixed' otherwise.
- Stopping criteria must be observable (e.g. "each subquestion has evidence from at least two independent credible sources").
Do not answer the question; only plan the research."""

HUMAN = """{request}

If reviewer feedback is included above, revise the plan to address it."""

PROMPT = make_prompt(SYSTEM, HUMAN)

_LEADING_VERBS = re.compile(
    r"^(please\s+)?(compare|analy[sz]e|evaluate|investigate|assess|examine|review|explain|research|study)\s+(the\s+)?",
    re.IGNORECASE,
)
_SOURCE_INSTRUCTIONS = re.compile(
    r"\.\s*(use|using|cite|include|identify|focus)\b.*$", re.IGNORECASE | re.DOTALL
)


_INTERROGATIVE = re.compile(
    r"^(what|which|how|why|when|where|who)\s+(are|is|do|does|did|can|could|should|will|would)?\s*(the\s+)?",
    re.IGNORECASE,
)


def _topic(question: str) -> str:
    topic = _SOURCE_INSTRUCTIONS.sub("", question.strip()).rstrip(" .?")
    topic = _LEADING_VERBS.sub("", topic)
    topic = _INTERROGATIVE.sub("", topic)
    return topic[:1].lower() + topic[1:] if topic else question


def _comparison_entities(topic: str) -> tuple[list[str], str]:
    """Split 'X, Y, and Z for <context>' into (['X','Y','Z'], '<context>')."""
    match = re.match(
        r"^(?:the\s+)?(?:effectiveness|tradeoffs|trade-offs|differences)?\s*(?:of|between)?\s*(.+?)(\s+(?:for|in|on|when|across)\s+.+)?$",
        topic,
        re.IGNORECASE,
    )
    if not match:
        return [], ""
    head, context = match.group(1), (match.group(2) or "").strip()
    parts = [
        p.strip(" ,")
        for p in re.split(r",\s*(?:and\s+)?|\s+and\s+|\s+vs\.?\s+|\s+versus\s+", head)
        if p.strip(" ,")
    ]
    return (parts, context) if 2 <= len(parts) <= 4 else ([], "")


def _queries(text: str, extra: str, limit: int) -> list[SearchQuery]:
    base = keywords(text, 6)
    candidates = [base, f"{base} {extra}".strip(), f"{keywords(text, 4)} evaluation"]
    unique = list(dict.fromkeys(q for q in candidates if len(q) >= 3))
    return [SearchQuery(query=q) for q in unique[:limit]]


def heuristic_plan(question: str, *, max_subquestions: int, max_queries: int) -> ResearchPlan:
    """Deterministic plan used when the planner model is unavailable (graceful degradation)."""
    topic = _topic(question)
    entities, context = _comparison_entities(topic)
    specs: list[tuple[str, str, str]] = []
    if entities:
        for entity in entities:
            specs.append(
                (
                    f"What does the evidence show about the effectiveness of {entity} {context}?".replace(
                        " ?", "?"
                    ),
                    "academic",
                    "benchmark results",
                )
            )
        specs.append(
            (
                f"How do {', '.join(entities)} compare on published benchmarks {context}?".replace(
                    " ?", "?"
                ),
                "academic",
                "benchmark comparison",
            )
        )
        specs.append(
            (
                f"What are the cost, latency and operational tradeoffs of {', '.join(entities)}?",
                "mixed",
                "cost latency",
            )
        )
        specs.append(
            (
                f"What limitations and failure modes are reported for {', '.join(entities)}?",
                "mixed",
                "limitations",
            )
        )
    else:
        specs = [
            (f"What is the current state of knowledge on {topic}?", "mixed", "survey"),
            (
                f"What empirical studies and measurements exist on {topic}?",
                "academic",
                "benchmark results",
            ),
            (
                f"What limitations, risks or uncertainties are reported regarding {topic}?",
                "mixed",
                "limitations",
            ),
            (
                f"What practical implications and tradeoffs follow from the evidence on {topic}?",
                "web",
                "cost tradeoffs",
            ),
            (
                f"What do credible sources recommend regarding {topic}?",
                "web",
                "best practices",
            ),
        ]
    subquestions = [
        ResearchSubquestion(
            id=f"SQ{i + 1}",
            question=text[:500],
            rationale="Generated by the deterministic fallback planner.",
            information_requirements=[extra],
            search_queries=_queries(text, extra, max_queries),
            preferred_sources=pref,
        )
        for i, (text, pref, extra) in enumerate(specs[:max_subquestions])
    ]
    return ResearchPlan(
        objective=f"Produce an evidence-backed assessment of {topic}.",
        scope="Published research and credible technical sources; claims must be supported by retrieved evidence.",
        subquestions=subquestions,
        source_strategy=[
            "Peer-reviewed papers and preprints",
            "Official documentation",
            "Reputable technical publications",
        ],
        stopping_criteria=[
            "Every subquestion has credible evidence from at least two independent sources",
            "The critic reports no critical issues",
        ],
    )


def to_research_plan(
    output: PlannerOutput, *, max_subquestions: int, max_queries: int, version: int = 1
) -> ResearchPlan:
    subquestions = []
    for i, planned in enumerate(output.subquestions[:max_subquestions]):
        queries = [
            SearchQuery(query=q[:300], source_preference=planned.preferred_sources)
            for q in dict.fromkeys(q.strip() for q in planned.search_queries)
            if len(q) >= 3
        ]
        subquestions.append(
            ResearchSubquestion(
                id=f"SQ{i + 1}",
                question=planned.question[:500],
                rationale=planned.rationale,
                information_requirements=planned.information_requirements[:6],
                search_queries=queries[:max_queries] or _queries(planned.question, "", max_queries),
                preferred_sources=planned.preferred_sources,
            )
        )
    return ResearchPlan(
        objective=output.objective,
        scope=output.scope,
        subquestions=subquestions,
        source_strategy=output.source_strategy[:8],
        stopping_criteria=output.stopping_criteria[:6],
        version=version,
    )


async def create_plan(
    ctx: ResearchContext, question: str, *, feedback: list[str] | None = None
) -> ResearchPlan:
    settings = ctx.settings
    request = render_block(
        "planning_request",
        {
            "question": question,
            "max_subquestions": settings.max_subquestions,
            "max_queries_per_subquestion": settings.max_queries_per_subquestion,
            "reviewer_feedback": feedback or [],
        },
    )
    output = await call_structured(
        ctx,
        tier=ModelTier.STRONG,
        prompt=PROMPT,
        variables={"request": request, "max_subquestions": settings.max_subquestions},
        schema=PlannerOutput,
        operation="planner",
    )
    return to_research_plan(
        output,
        max_subquestions=settings.max_subquestions,
        max_queries=settings.max_queries_per_subquestion,
    )
