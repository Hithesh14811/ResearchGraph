"""Graph construction: the research-worker subgraph and the main research workflow.

Main graph::

    START → intake ─(invalid)→ END
              └→ planner → plan_review ⟲ (interrupt: approve / edit / replan / cancel)
                   → query_generation ═(Send × N)═> research_worker ⇉ source_evaluation
                        ↑                                                   │ evidence gate
                        ├────────────── insufficient evidence ──────────────┘
                        │                                  sufficient ↓
                        │      contradiction_detection → synthesis → critic → quality_gate
                        └──────────────────── research_more ─────────────────────┘
                                                              proceed ↓
                        report_writer → citation_verification ⇄ citation_repair
                                              → final_review → END

Worker subgraph (one per subquestion per round, run in parallel)::

    START → search_agent ⇄ execute_tools (ToolNode) → source_processing
          → passage_retrieval → evidence_extraction → END
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from researchgraph.core.retry import is_transient_error
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.nodes.analysis import (
    contradiction_detection_node,
    critic_node,
    quality_gate_node,
    synthesis_node,
)
from researchgraph.graph.nodes.planning import intake_node, plan_review_node, planner_node
from researchgraph.graph.nodes.reporting import (
    citation_repair_node,
    citation_verification_node,
    final_review_node,
    report_writer_node,
)
from researchgraph.graph.nodes.research_round import query_generation_node, source_evaluation_node
from researchgraph.graph.nodes.worker import (
    evidence_extraction_node,
    passage_retrieval_node,
    search_agent_node,
    source_processing_node,
)
from researchgraph.graph.routing import (
    dispatch_research,
    route_after_citation_check,
    route_after_evidence,
    route_after_intake,
    route_after_quality_gate,
    route_search_agent,
)
from researchgraph.graph.state import ResearchState, WorkerInput, WorkerOutput, WorkerState
from researchgraph.tools.research_tools import RESEARCH_TOOLS

# Node-level retries cover unexpected transient failures that escape a node's own handling
# (LLM calls already retry internally and degrade to fallbacks).
TRANSIENT_RETRY = RetryPolicy(max_attempts=2, initial_interval=1.0, retry_on=is_transient_error)


def build_worker_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    builder = StateGraph(
        WorkerState,
        context_schema=ResearchContext,
        input_schema=WorkerInput,
        output_schema=WorkerOutput,
    )
    builder.add_node("search_agent", search_agent_node)
    builder.add_node(
        "execute_tools",
        ToolNode(RESEARCH_TOOLS, handle_tool_errors=True),
        retry_policy=TRANSIENT_RETRY,
    )
    builder.add_node("source_processing", source_processing_node)
    builder.add_node("passage_retrieval", passage_retrieval_node)
    builder.add_node("evidence_extraction", evidence_extraction_node)

    builder.add_edge(START, "search_agent")
    builder.add_conditional_edges(
        "search_agent", route_search_agent, ["execute_tools", "source_processing"]
    )
    builder.add_edge("execute_tools", "search_agent")
    builder.add_edge("source_processing", "passage_retrieval")
    builder.add_edge("passage_retrieval", "evidence_extraction")
    builder.add_edge("evidence_extraction", END)
    return builder.compile(name="research_worker")


def build_research_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    builder = StateGraph(ResearchState, context_schema=ResearchContext)

    builder.add_node("intake", intake_node)
    builder.add_node("planner", planner_node, retry_policy=TRANSIENT_RETRY)
    builder.add_node(
        "plan_review",
        plan_review_node,
        destinations=("query_generation", "planner", "plan_review", END),
    )
    builder.add_node("query_generation", query_generation_node, retry_policy=TRANSIENT_RETRY)
    builder.add_node("research_worker", build_worker_graph())
    builder.add_node("source_evaluation", source_evaluation_node)
    builder.add_node(
        "contradiction_detection", contradiction_detection_node, retry_policy=TRANSIENT_RETRY
    )
    builder.add_node("synthesis", synthesis_node, retry_policy=TRANSIENT_RETRY)
    builder.add_node("critic", critic_node, retry_policy=TRANSIENT_RETRY)
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node("report_writer", report_writer_node, retry_policy=TRANSIENT_RETRY)
    builder.add_node("citation_verification", citation_verification_node)
    builder.add_node("citation_repair", citation_repair_node)
    builder.add_node("final_review", final_review_node)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges("intake", route_after_intake, {"planner": "planner", END: END})
    builder.add_edge("planner", "plan_review")
    # plan_review routes itself with Command(goto=...): approve / edit / replan / cancel.
    builder.add_conditional_edges(
        "query_generation", dispatch_research, ["research_worker", "source_evaluation"]
    )
    builder.add_edge("research_worker", "source_evaluation")
    builder.add_conditional_edges(
        "source_evaluation",
        route_after_evidence,
        {
            "query_generation": "query_generation",
            "contradiction_detection": "contradiction_detection",
        },
    )
    builder.add_edge("contradiction_detection", "synthesis")
    builder.add_edge("synthesis", "critic")
    builder.add_edge("critic", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {"query_generation": "query_generation", "report_writer": "report_writer"},
    )
    builder.add_edge("report_writer", "citation_verification")
    builder.add_conditional_edges(
        "citation_verification",
        route_after_citation_check,
        {"citation_repair": "citation_repair", "final_review": "final_review"},
    )
    builder.add_edge("citation_repair", "citation_verification")
    builder.add_edge("final_review", END)

    return builder.compile(checkpointer=checkpointer, name="researchgraph")
