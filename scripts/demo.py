"""ResearchGraph offline demo.

Runs the complete workflow — planning, human plan review, parallel research workers,
evidence extraction, source scoring, contradiction detection, synthesis, critique, quality
gate, report writing and citation verification — with a deterministic mock model and a
clearly-labelled synthetic corpus. No API keys or network access needed.

Usage:
    python -m scripts.demo                        # sample question 1, auto-approved plan
    python -m scripts.demo --question 3           # pick a sample question (1-3)
    python -m scripts.demo --all                  # run all three sample questions
    python -m scripts.demo --interactive          # approve / revise / cancel the plan yourself
    python -m scripts.demo --simulate-failures all
    python -m scripts.demo --simulate-failures failed_source,critic_rejection
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from researchgraph.config.logging import configure_logging
from researchgraph.config.settings import Settings
from researchgraph.core.aio import use_selector_event_loop_on_windows
from researchgraph.demo.faults import SCENARIOS, parse_scenarios
from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.runtime.bootstrap import AppServices, open_services
from researchgraph.runtime.streaming import follow_run_events
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.events import EventType, WorkflowEvent

WORKER_NODES = {
    "search_agent",
    "execute_tools",
    "source_processing",
    "passage_retrieval",
    "evidence_extraction",
}
COLORS = {
    EventType.NODE_STARTED: "\033[36m",
    EventType.NODE_COMPLETED: "\033[32m",
    EventType.PROGRESS: "\033[0m",
    EventType.TOOL: "\033[35m",
    EventType.WARNING: "\033[33m",
    EventType.ERROR: "\033[31m",
    EventType.PLAN_READY: "\033[1;33m",
    EventType.RUN_STATUS: "\033[1;34m",
    EventType.METRICS: "\033[0m",
}
ICONS = {
    EventType.NODE_STARTED: ">",
    EventType.NODE_COMPLETED: "+",
    EventType.PROGRESS: "*",
    EventType.TOOL: "~",
    EventType.WARNING: "!",
    EventType.ERROR: "x",
    EventType.PLAN_READY: "?",
    EventType.RUN_STATUS: "#",
    EventType.METRICS: "=",
}


def demo_settings(output: Path, latency_ms: int) -> Settings:
    """Force fully-offline settings regardless of any .env file or environment variables."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="mock",
        model_name="mock-research-model",
        search_provider="mock",
        embedding_provider="hashing",
        vector_store="memory",
        database_url=f"sqlite+aiosqlite:///{(output / 'demo.db').as_posix()}",
        resume_runs_on_startup=False,
        mock_llm_latency_ms=latency_ms,
        langchain_tracing_v2=False,
        log_level="WARNING",
    )


class Printer:
    def __init__(self, *, verbose: bool, color: bool) -> None:
        self.verbose = verbose
        self.color = color
        self.start = time.perf_counter()

    def __call__(self, event: WorkflowEvent) -> None:
        lifecycle = event.type in (EventType.NODE_STARTED, EventType.NODE_COMPLETED)
        if (
            not self.verbose
            and lifecycle
            and (event.node in WORKER_NODES or event.type is EventType.NODE_COMPLETED)
        ):
            return
        if not self.verbose and event.type is EventType.TOOL:
            return
        elapsed = time.perf_counter() - self.start
        node = (event.node or "").ljust(24)
        line = f"[{elapsed:6.2f}s] {ICONS.get(event.type, '-')} {node} {event.message}"
        if self.color:
            line = f"{COLORS.get(event.type, '')}{line}\033[0m"
        print(line, flush=True)


def print_plan(plan: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(f"RESEARCH PLAN (v{plan.get('version', 1)})")
    print(f"Objective: {plan['objective']}")
    print(f"Scope:     {plan.get('scope', '')}")
    for sq in plan["subquestions"]:
        print(f"\n  {sq['id']}. {sq['question']}  [{sq['preferred_sources']}]")
        for q in sq["search_queries"]:
            print(f"      search: {q['query']}")
    print("\nSource strategy:  " + "; ".join(plan.get("source_strategy", [])))
    print("Stopping criteria: " + "; ".join(plan.get("stopping_criteria", [])))
    print("=" * 88)


async def follow(
    services: AppServices, research_id: str, printer: Printer, after: int, *, until_plan: bool
) -> int:
    async for event in follow_run_events(
        repository=services.repository,
        broker=services.broker,
        manager=services.manager,
        research_id=research_id,
        after_seq=after,
        keepalive_seconds=2.0,
    ):
        if event is None:
            continue
        printer(event)
        after = event.seq
        if until_plan and event.type is EventType.PLAN_READY:
            break
    return after


async def review_plan(services: AppServices, research_id: str, printer: Printer, after: int) -> int:
    while True:
        run = await services.repository.get_run(research_id)
        if run is None or run.status != RunStatus.AWAITING_APPROVAL.value:
            return after
        plan = await services.manager.get_plan(research_id)
        if plan is not None:
            print_plan(plan.model_dump(mode="json"))
        choice = (
            (await asyncio.to_thread(input, "\n[a]pprove, [r]evise with feedback, or [c]ancel? "))
            .strip()
            .lower()
        )
        if choice.startswith("c"):
            await services.manager.cancel(research_id)
        elif choice.startswith("r"):
            feedback = await asyncio.to_thread(input, "Feedback for the planner: ")
            await services.manager.replan(research_id, feedback or "Add more depth.")
            after = await follow(services, research_id, printer, after, until_plan=True)
            continue
        else:
            await services.manager.approve(research_id)
        return after


async def write_outputs(services: AppServices, research_id: str, output: Path) -> Path:
    folder = output / research_id
    folder.mkdir(parents=True, exist_ok=True)
    repo = services.repository
    run = await repo.get_run(research_id)
    report = await repo.get_report(research_id)
    sources, _ = await repo.get_sources(research_id, limit=500, offset=0)
    evidence, _ = await repo.get_evidence(research_id, subquestion_id=None, limit=500, offset=0)
    events = await repo.list_events(research_id, limit=10_000)
    if report is not None:
        (folder / "report.md").write_text(report.markdown, encoding="utf-8")
        (folder / "executive_summary.md").write_text(
            f"# {report.title}\n\n{report.executive_summary}\n", encoding="utf-8"
        )
    (folder / "sources.json").write_text(
        json.dumps(
            [{"source": s.data, "quality": s.quality} for s in sources], indent=2, default=str
        ),
        encoding="utf-8",
    )
    (folder / "evidence.json").write_text(
        json.dumps([e.data for e in evidence], indent=2, default=str), encoding="utf-8"
    )
    assert run is not None
    (folder / "run.json").write_text(
        json.dumps(
            {
                "id": run.id,
                "question": run.question,
                "status": run.status,
                "iteration": run.iteration,
                "plan": run.plan,
                "critique": run.critique,
                "quality": run.quality,
                "contradictions": run.contradictions,
                "metrics": run.metrics,
                "usage": run.usage,
                "errors": run.errors,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    with (folder / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(
                json.dumps(
                    {
                        "seq": e.seq,
                        "type": e.type,
                        "node": e.node,
                        "message": e.message,
                        "data": e.data,
                    },
                    default=str,
                )
                + "\n"
            )
    (folder / "workflow.mmd").write_text(
        services.graph.get_graph(xray=1).draw_mermaid(), encoding="utf-8"
    )
    return folder


def print_summary(run: Any, folder: Path, wall: float) -> None:
    m = run.metrics or {}
    usage = run.usage or {}
    report = m.get("report") or {}
    print("\n" + "-" * 88)
    print(
        f"Status: {run.status.upper()}   wall time {wall:.1f}s   research iterations: {m.get('iterations', 0)}"
    )
    print(
        f"Tool calls: {m.get('tool_calls_used', 0)}   LLM calls: {usage.get('llm_calls', 0)}   "
        f"tokens: {usage.get('total_tokens', 0):,} (offline estimate)"
    )
    print(
        f"Sources: {m.get('sources_found', 0)} ({m.get('high_quality_sources', 0)} high quality, "
        f"{m.get('sources_failed', 0)} failed, {m.get('sources_degraded', 0)} snippet-only)   evidence: {m.get('evidence_items', 0)}   findings: {m.get('findings', 0)}   "
        f"contradictions: {m.get('contradictions', 0)}"
    )
    print(
        f"Critic: {m.get('critic_issues', 0)} issue(s), verdict {m.get('critic_verdict')}   quality gate score: {m.get('quality_score')}"
    )
    if report:
        print(
            f"Report: {report['total_claims']} verified sentences, {report['citation_coverage']:.0%} citation coverage, "
            f"{report['removed_claims']} removed, {report['sources_cited']} sources cited"
        )
    if m.get("faults_triggered"):
        print(
            "Faults injected and recovered from: "
            + ", ".join(f"{k} x{v}" for k, v in m["faults_triggered"].items())
        )
    if run.errors:
        print(f"Recorded (recoverable) errors: {len(run.errors)}")
    print(f"Outputs: {folder}")
    print("-" * 88)


async def run_one(
    services: AppServices, question: str, args: argparse.Namespace, output: Path
) -> None:
    printer = Printer(verbose=args.verbose, color=sys.stdout.isatty() and not args.no_color)
    failures = parse_scenarios(args.simulate_failures) if args.simulate_failures else frozenset()
    print(f"\nQuestion: {question}")
    if failures:
        print(f"Simulating failures: {', '.join(sorted(failures))}")
    started = time.perf_counter()
    research_id = await services.manager.create_run(
        question, auto_approve=not args.interactive, failure_scenarios=failures
    )
    after = await follow(services, research_id, printer, 0, until_plan=args.interactive)
    if args.interactive:
        after = await review_plan(services, research_id, printer, after)
        await follow(services, research_id, printer, after, until_plan=False)
    await services.manager.wait(research_id)
    run = await services.repository.get_run(research_id)
    assert run is not None
    folder = await write_outputs(services, research_id, output)
    print_summary(run, folder, time.perf_counter() - started)


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--question",
        type=int,
        choices=range(1, len(SAMPLE_QUESTIONS) + 1),
        default=1,
        help="sample question number",
    )
    group.add_argument(
        "--custom", type=str, help="ask your own question (uses the fallback demo planner)"
    )
    group.add_argument("--all", action="store_true", help="run all sample questions")
    parser.add_argument(
        "--interactive", action="store_true", help="review the plan yourself (human-in-the-loop)"
    )
    parser.add_argument(
        "--simulate-failures", default="", help=f"comma-separated: {', '.join(SCENARIOS)}, or 'all'"
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=40,
        help="simulated model latency (makes parallelism visible)",
    )
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    parser.add_argument("--verbose", action="store_true", help="show every node and tool event")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    configure_logging("ERROR")  # workflow warnings are already shown as events
    output: Path = args.output.resolve()
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
    print(
        "ResearchGraph demo — deterministic mock model + SYNTHETIC corpus (no API keys, no network)."
    )
    print("All sources in this demo are fabricated for illustration and labelled [Synthetic].")

    questions = (
        [s.question for s in SAMPLE_QUESTIONS]
        if args.all
        else [args.custom]
        if args.custom
        else [SAMPLE_QUESTIONS[args.question - 1].question]
    )
    async with open_services(demo_settings(output, args.latency_ms), recover=False) as services:
        for question in questions:
            await run_one(services, question, args, output)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    use_selector_event_loop_on_windows()
    asyncio.run(main())
