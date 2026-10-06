"""LLM-callable tools for the research worker.

Tools are LangChain ``@tool``s executed by LangGraph's prebuilt ``ToolNode``. Dependencies
(search service, fault injector) arrive through ``ToolRuntime`` injection, so tools are
stateless module-level functions. Search tools use ``response_format="content_and_artifact"``:
the model sees a compact text listing (no URLs), while structured ``SearchResult`` data rides
along as the ToolMessage artifact for the source-processing step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import Field

from researchgraph.agents.researcher import format_search_results
from researchgraph.core.errors import SearchProviderError
from researchgraph.schemas.events import EventType
from researchgraph.tools.calculator import CalculatorError, evaluate_expression
from researchgraph.tools.search.base import SearchChannel

if TYPE_CHECKING:
    from researchgraph.graph.context import ResearchContext

SEARCH_TOOL_NAMES = frozenset({"web_search", "academic_search"})

QueryArg = Annotated[
    str, Field(min_length=3, max_length=300, description="Concise keyword search query.")
]
MaxResultsArg = Annotated[
    int, Field(ge=1, le=10, description="Number of results to return (1-10).")
]


async def _search(
    channel: SearchChannel, query: str, max_results: int, runtime: ToolRuntime[ResearchContext, Any]
) -> tuple[str, list[dict[str, Any]]]:
    ctx = runtime.context
    task = runtime.state.get("task")
    query = " ".join(query.split())[:300]
    max_results = max(1, min(max_results, ctx.settings.search_results_per_query))
    fault_key = f"{task.subquestion_id}:{task.iteration}" if task is not None else query
    try:
        if ctx.faults.should_fail("insufficient_evidence", fault_key):
            results = []  # simulated: the search surfaced nothing useful this round
        else:
            results = await ctx.search.search(query, channel=channel, max_results=max_results)
    except SearchProviderError as exc:
        runtime.stream_writer(
            {
                "type": EventType.WARNING.value,
                "node": "execute_tools",
                "message": f"Search failed for '{query}': {exc}",
                "data": {},
            }
        )
        return f"Search failed: {exc}. Try another query or tool.", []
    rows = [r.model_dump(mode="json") for r in results]
    label = f"[{task.subquestion_id}] " if task is not None else ""
    runtime.stream_writer(
        {
            "type": EventType.TOOL.value,
            "node": "execute_tools",
            "message": f"{label}{channel.value} search '{query}' → {len(results)} results",
            "data": {"tool": f"{channel.value}_search", "query": query, "results": len(results)},
        }
    )
    return format_search_results(rows), rows


@tool("web_search", response_format="content_and_artifact")
async def web_search(
    query: QueryArg, runtime: ToolRuntime[Any, Any], max_results: MaxResultsArg = 5
) -> tuple[str, list[dict[str, Any]]]:
    """Search the web for documentation, technical articles, industry reports and recent developments."""
    return await _search(SearchChannel.WEB, query, max_results, runtime)


@tool("academic_search", response_format="content_and_artifact")
async def academic_search(
    query: QueryArg, runtime: ToolRuntime[Any, Any], max_results: MaxResultsArg = 5
) -> tuple[str, list[dict[str, Any]]]:
    """Search academic papers and preprints for empirical results, benchmarks and methods."""
    return await _search(SearchChannel.ACADEMIC, query, max_results, runtime)


@tool("calculator")
def calculator(
    expression: Annotated[
        str,
        Field(max_length=200, description="Arithmetic expression, e.g. '(71.2-64.5)/64.5*100'."),
    ],
) -> str:
    """Evaluate an arithmetic expression exactly (use for comparing numbers found in sources)."""
    try:
        return f"{evaluate_expression(expression):.6g}"
    except CalculatorError as exc:
        return f"Error: {exc}"


RESEARCH_TOOLS = [web_search, academic_search, calculator]
