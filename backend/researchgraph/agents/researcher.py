"""Research worker prompt: a bounded tool-using agent that chooses *how* to search.

The model picks between academic and web search and adapts its queries to what earlier
results returned. It cannot fetch arbitrary URLs — only the pipeline fetches, and only URLs
that a search provider returned.
"""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from researchgraph.agents.base import UNTRUSTED_CONTENT_POLICY, render_block
from researchgraph.schemas.research import ResearchTask

SYSTEM = """You are a research worker gathering sources for ONE subquestion of a larger research project.

Use the search tools to find credible, relevant sources:
- academic_search: papers and preprints — best for empirical results, benchmarks and methods.
- web_search: documentation, technical articles and industry reports — best for practice, tooling and recent developments.
- calculator: arithmetic on numbers you have seen (e.g. relative improvements).
You may make at most {budget} tool calls in total. Prefer several distinct, specific queries over repeating one.
When the results already include several relevant and credible sources, stop calling tools and reply with one sentence summarising what you found.
{policy}"""


def build_research_messages(task: ResearchTask, overall_question: str) -> list[BaseMessage]:
    task_block = render_block(
        "task",
        {
            "subquestion_id": task.subquestion_id,
            "subquestion": task.question,
            "overall_question": overall_question,
            "focus": task.focus,
            "reason": task.reason,
            "suggested_queries": [q.query for q in task.queries],
            "preferred_sources": task.preferred_sources,
            "tool_budget": task.tool_budget,
        },
    )
    return [
        SystemMessage(
            content=SYSTEM.format(budget=task.tool_budget, policy=UNTRUSTED_CONTENT_POLICY)
        ),
        HumanMessage(content=f"{task_block}\n\nFind sources for this subquestion."),
    ]


def format_search_results(results: list[dict[str, object]]) -> str:
    """Compact, model-readable listing of search hits (wrapped as untrusted data)."""
    rows = [
        {
            "rank": i + 1,
            "title": r.get("title"),
            "source": r.get("provider"),
            "published": r.get("published_date"),
            "snippet": str(r.get("snippet") or "")[:400],
        }
        for i, r in enumerate(results)
    ]
    if not rows:
        return "<search_results>\n[]\n</search_results>\nNo results. Try a different query or the other search tool."
    return f"<search_results>\n{json.dumps(rows, ensure_ascii=False, indent=1, default=str)}\n</search_results>"
