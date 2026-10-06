"""Mapping from graph nodes to UI workflow stages and progress milestones."""

from __future__ import annotations

NODE_STAGE = {
    "intake": "plan",
    "planner": "plan",
    "plan_review": "plan",
    "query_generation": "research",
    "research_worker": "research",
    "search_agent": "research",
    "execute_tools": "research",
    "source_processing": "research",
    "passage_retrieval": "research",
    "evidence_extraction": "evidence",
    "source_evaluation": "evidence",
    "contradiction_detection": "analysis",
    "synthesis": "analysis",
    "critic": "critic",
    "quality_gate": "quality_gate",
    "report_writer": "report",
    "citation_verification": "verification",
    "citation_repair": "verification",
    "final_review": "verification",
}

# Progress (%) reached when a node *completes*. Progress is reported monotonically even
# though the graph can loop back to research.
NODE_PROGRESS = {
    "intake": 3,
    "planner": 10,
    "plan_review": 14,
    "query_generation": 18,
    "search_agent": 25,
    "execute_tools": 30,
    "source_processing": 40,
    "passage_retrieval": 45,
    "evidence_extraction": 52,
    "research_worker": 55,
    "source_evaluation": 60,
    "contradiction_detection": 66,
    "synthesis": 72,
    "critic": 78,
    "quality_gate": 82,
    "report_writer": 88,
    "citation_verification": 93,
    "citation_repair": 95,
    "final_review": 100,
}
