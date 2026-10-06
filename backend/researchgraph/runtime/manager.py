"""Run lifecycle management.

``RunManager`` executes graph runs as background asyncio tasks and:

* streams LangGraph ``updates`` + ``custom`` events (including from parallel worker
  subgraphs), turning them into persisted, sequence-numbered ``WorkflowEvent``s;
* projects state updates into the relational read model by re-applying the *same reducers*
  the graph uses (so the projection can never disagree with graph semantics);
* resumes interrupted runs with ``Command(resume=...)`` for plan approval/edit/cancel;
* drains running graphs cooperatively on shutdown (``RunControl``) and resumes them from
  their last checkpoint on startup — runs survive a server restart.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphDrained
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import RunControl
from langgraph.types import Command, StreamMode

from researchgraph.config.settings import Settings
from researchgraph.core.errors import InvalidRunStateError, RunNotFoundError, TooManyRunsError
from researchgraph.database.repository import ResearchRepository
from researchgraph.graph.context import ResearchContext
from researchgraph.graph.progress import NODE_PROGRESS, NODE_STAGE
from researchgraph.graph.state import append_unique, merge_by_id, merge_sources
from researchgraph.llm.usage import UsageSnapshot, UsageTracker
from researchgraph.runtime.container import ServiceContainer
from researchgraph.runtime.events import EventBroker
from researchgraph.schemas.common import RunStatus, new_research_id, utc_now
from researchgraph.schemas.events import EventType, WorkflowEvent
from researchgraph.schemas.research import PlanDecision, ResearchPlan
from researchgraph.schemas.sources import QualityTier

logger = logging.getLogger(__name__)

# Reducers mirrored from ResearchState for the read-model projection.
_REDUCERS: dict[str, Any] = {
    "sources": merge_sources,
    "evidence": merge_by_id,
    "source_quality": merge_by_id,
    "search_history": append_unique,
    "tool_calls_used": lambda a, b: (a or 0) + (b or 0),
    "errors": lambda a, b: [*(a or []), *(b or [])],
    "node_metrics": lambda a, b: [*(a or []), *(b or [])],
    "critique_history": lambda a, b: [*(a or []), *(b or [])],
    "quality_history": lambda a, b: [*(a or []), *(b or [])],
}
STREAM_MODES: list[StreamMode] = ["updates", "custom"]
# Keys from inside worker subgraphs that are safe to project early (idempotent merges).
_LIVE_WORKER_KEYS = ("sources", "evidence")


@dataclass
class _ActiveRun:
    task: asyncio.Task[None]
    control: RunControl
    tracker: UsageTracker
    context: ResearchContext


@dataclass
class _RunView:
    """Local merged view of graph state for one run (the projection source)."""

    values: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    active_seconds: float = 0.0

    def apply(self, update: Mapping[str, Any], *, keys: tuple[str, ...] | None = None) -> set[str]:
        changed = set()
        for key, value in update.items():
            if key.startswith("__") or (keys is not None and key not in keys):
                continue
            reducer = _REDUCERS.get(key)
            self.values[key] = reducer(self.values.get(key), value) if reducer else value
            changed.add(key)
        return changed


class RunManager:
    def __init__(
        self,
        *,
        settings: Settings,
        graph: CompiledStateGraph[Any, Any, Any, Any],
        repository: ResearchRepository,
        container: ServiceContainer,
        broker: EventBroker,
    ) -> None:
        self.settings = settings
        self.graph = graph
        self.repo = repository
        self.container = container
        self.broker = broker
        self._active: dict[str, _ActiveRun] = {}
        self._views: dict[str, _RunView] = {}
        self._contexts: dict[str, ResearchContext] = {}  # reused across resumes (e.g. fault state)
        self._seq: dict[str, int] = {}
        self._seq_lock = asyncio.Lock()
        # Serialises state transitions (approve/edit/replan/cancel) per run, so two concurrent
        # requests cannot both pass the "not executing" check.
        self._transition_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ------------------------------------------------------------------ public API
    def is_active(self, research_id: str) -> bool:
        return research_id in self._active

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def create_run(
        self,
        question: str,
        *,
        auto_approve: bool = False,
        failure_scenarios: frozenset[str] = frozenset(),
    ) -> str:
        self._check_capacity()
        research_id = new_research_id()
        await self.repo.create_run(
            research_id=research_id,
            question=question,
            auto_approve=auto_approve,
            failure_scenarios=failure_scenarios,
        )
        await self._emit(
            research_id,
            EventType.RUN_STATUS,
            "Research run created.",
            status=RunStatus.PENDING.value,
        )
        graph_input = {
            "research_id": research_id,
            "original_question": question,
            "auto_approve": auto_approve,
        }
        self._start(research_id, graph_input, failure_scenarios)
        return research_id

    async def get_plan(self, research_id: str) -> ResearchPlan | None:
        snapshot = await self.graph.aget_state(self._config(research_id))
        plan = snapshot.values.get("plan") if snapshot and snapshot.values else None
        return plan if isinstance(plan, ResearchPlan) else None

    async def approve(self, research_id: str) -> None:
        await self._resume(research_id, PlanDecision(action="approve"))

    async def edit_plan(self, research_id: str, plan: ResearchPlan, *, approve: bool) -> None:
        await self._resume(research_id, PlanDecision(action="edit", plan=plan, approve=approve))

    async def replan(self, research_id: str, feedback: str) -> None:
        await self._resume(research_id, PlanDecision(action="replan", feedback=feedback))

    async def cancel(self, research_id: str) -> None:
        async with self._transition_locks[research_id]:
            run = await self.repo.get_run(research_id)
            if run is None:
                raise RunNotFoundError(f"Research run {research_id} not found")
            status = RunStatus(run.status)
            if status.is_terminal:
                raise InvalidRunStateError(f"Run is already {status.value}")
            if status is RunStatus.AWAITING_APPROVAL and not self.is_active(research_id):
                await self._resume_unlocked(research_id, PlanDecision(action="cancel"))
                return
            active = self._active.get(research_id)
            await self._set_status(research_id, RunStatus.CANCELLED, completed_at=utc_now())
            await self._emit(
                research_id,
                EventType.RUN_STATUS,
                "Research cancelled by user.",
                status=RunStatus.CANCELLED.value,
            )
            if active is not None:
                active.task.cancel()

    async def recover(self) -> None:
        """Resume runs that were executing (or paused for shutdown) when the server stopped."""
        runs = await self.repo.runs_with_status(
            [RunStatus.RUNNING, RunStatus.INTERRUPTED, RunStatus.PENDING]
        )
        for run in runs:
            snapshot = await self.graph.aget_state(self._config(run.id))
            failures = frozenset(run.failure_scenarios or [])
            if not snapshot or not snapshot.values:
                # Crashed before the first checkpoint: restart from the original input.
                await self._emit(
                    run.id, EventType.RUN_STATUS, "Restarting run after server restart."
                )
                self._start(
                    run.id,
                    {
                        "research_id": run.id,
                        "original_question": run.question,
                        "auto_approve": run.auto_approve,
                    },
                    failures,
                )
                continue
            if snapshot.interrupts:
                await self._set_status(run.id, RunStatus.AWAITING_APPROVAL)
                continue
            if not snapshot.next:
                await self._finalize(run.id, dict(snapshot.values))
                continue
            logger.info("Resuming run %s from checkpoint (next: %s)", run.id, snapshot.next)
            await self._emit(
                run.id,
                EventType.RUN_STATUS,
                "Resuming from last checkpoint after server restart.",
                status=RunStatus.RUNNING.value,
                next=list(snapshot.next),
            )
            self._start(run.id, None, failures)

    async def shutdown(self, drain_timeout: float | None = None) -> None:
        """Ask every running graph to stop at the next superstep boundary (checkpoint saved)."""
        if not self._active:
            return
        for active in self._active.values():
            active.control.request_drain("server shutdown")
        tasks = [a.task for a in self._active.values()]
        _, pending = await asyncio.wait(
            tasks, timeout=drain_timeout or self.settings.shutdown_drain_timeout_seconds
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def wait(self, research_id: str) -> None:
        """Await the background task of a run (used by tests and the demo CLI)."""
        active = self._active.get(research_id)
        if active is not None:
            await asyncio.gather(active.task, return_exceptions=True)

    # ------------------------------------------------------------------ execution
    def _config(self, research_id: str, tracker: UsageTracker | None = None) -> RunnableConfig:
        config: RunnableConfig = {
            "configurable": {"thread_id": research_id},
            "recursion_limit": self.settings.graph_recursion_limit,
            "max_concurrency": self.settings.max_parallel_workers,
            "run_name": "researchgraph",
            "tags": ["researchgraph"],
            "metadata": {"research_id": research_id},
        }
        if tracker is not None:
            config["callbacks"] = [tracker]
        return config

    def _start(self, research_id: str, graph_input: Any, failures: frozenset[str]) -> None:
        if research_id in self._active:
            raise InvalidRunStateError("Run is already executing")
        tracker = UsageTracker(
            input_cost_per_mtok=self.settings.llm_input_cost_per_mtok,
            output_cost_per_mtok=self.settings.llm_output_cost_per_mtok,
        )
        control = RunControl()
        context = self._contexts.get(research_id) or self.container.context_for_run(
            failures=failures
        )
        self._contexts[research_id] = context
        task = asyncio.create_task(
            self._execute(research_id, graph_input, context, control, tracker),
            name=f"run-{research_id}",
        )
        self._active[research_id] = _ActiveRun(
            task=task, control=control, tracker=tracker, context=context
        )

        def _cleanup(done: asyncio.Task[None]) -> None:
            current = self._active.get(research_id)
            if current is not None and current.task is done:
                self._active.pop(research_id, None)

        task.add_done_callback(_cleanup)

    def _check_capacity(self) -> None:
        """Bound concurrent runs (and therefore concurrent LLM spend) per process."""
        if self.active_count >= self.settings.max_active_runs:
            raise TooManyRunsError(
                f"{self.active_count} runs are already executing "
                f"(MAX_ACTIVE_RUNS={self.settings.max_active_runs}); try again shortly"
            )

    async def _resume(self, research_id: str, decision: PlanDecision) -> None:
        async with self._transition_locks[research_id]:
            await self._resume_unlocked(research_id, decision)

    async def _resume_unlocked(self, research_id: str, decision: PlanDecision) -> None:
        run = await self.repo.get_run(research_id)
        if run is None:
            raise RunNotFoundError(f"Research run {research_id} not found")
        if self.is_active(research_id):
            raise InvalidRunStateError("Run is currently executing")
        if decision.action != "cancel":
            self._check_capacity()
        snapshot = await self.graph.aget_state(self._config(research_id))
        if not snapshot or not snapshot.interrupts:
            raise InvalidRunStateError(f"Run is not awaiting plan approval (status: {run.status})")
        if (
            decision.plan is not None
            and len(decision.plan.subquestions) > self.settings.max_subquestions
        ):
            raise InvalidRunStateError(
                f"A plan may have at most {self.settings.max_subquestions} subquestions"
            )
        await self._emit(
            research_id,
            EventType.RUN_STATUS,
            f"Plan review decision: {decision.action}.",
            action=decision.action,
        )
        self._start(
            research_id,
            Command(resume=decision.model_dump(mode="json")),
            frozenset(run.failure_scenarios or []),
        )

    async def _load_view(self, research_id: str) -> _RunView:
        view = self._views.get(research_id)
        if view is None:
            view = _RunView()
            snapshot = await self.graph.aget_state(self._config(research_id))
            if snapshot and snapshot.values:
                view.values = dict(snapshot.values)
            run = await self.repo.get_run(research_id)
            if run is not None:
                view.progress = run.progress or 0.0
                view.active_seconds = float((run.metrics or {}).get("active_seconds", 0.0))
            self._views[research_id] = view
        return view

    async def _execute(
        self,
        research_id: str,
        graph_input: Any,
        context: ResearchContext,
        control: RunControl,
        tracker: UsageTracker,
    ) -> None:
        view = await self._load_view(research_id)
        run = await self.repo.get_run(research_id)
        baseline = (
            UsageSnapshot.model_validate(run.usage)
            if run is not None and run.usage
            else UsageSnapshot()
        )
        await self._set_status(
            research_id,
            RunStatus.RUNNING,
            started_at=run.started_at if run and run.started_at else utc_now(),
        )
        segment_start = perf_counter()
        config = self._config(research_id, tracker)
        try:
            async for namespace, mode, chunk in self.graph.astream(
                graph_input,
                config,
                context=context,
                stream_mode=STREAM_MODES,
                subgraphs=True,
                control=control,
                durability="sync",
            ):
                if mode == "custom":
                    await self._on_custom(research_id, view, chunk)
                elif mode == "updates":
                    await self._on_updates(research_id, view, tuple(namespace), chunk)
            snapshot = await self.graph.aget_state(self._config(research_id))
            view.values = dict(snapshot.values)
            if snapshot.interrupts:
                plan = snapshot.values.get("plan")
                await self._set_status(research_id, RunStatus.AWAITING_APPROVAL)
                await self._emit(
                    research_id,
                    EventType.PLAN_READY,
                    "Research plan ready for review — approve, edit or cancel.",
                    subquestions=len(plan.subquestions) if plan else 0,
                )
            else:
                await self._finalize(research_id, view.values)
        except GraphDrained:
            await self._set_status(research_id, RunStatus.INTERRUPTED)
            await self._emit(
                research_id,
                EventType.RUN_STATUS,
                "Server shutting down: run paused at a checkpoint and will resume automatically.",
                status=RunStatus.INTERRUPTED.value,
            )
        except asyncio.CancelledError:
            current = await self.repo.get_run(research_id)
            if current is not None and current.status != RunStatus.CANCELLED.value:
                await self._set_status(research_id, RunStatus.INTERRUPTED)
            raise
        except Exception as exc:
            logger.exception("Run %s failed", research_id)
            await self._set_status(
                research_id,
                RunStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}"[:2000],
                completed_at=utc_now(),
            )
            await self._emit(
                research_id, EventType.ERROR, f"Run failed: {type(exc).__name__}: {exc}"[:500]
            )
        finally:
            view.active_seconds += perf_counter() - segment_start
            usage = baseline.plus(tracker.snapshot())
            metrics = self._metrics(view, usage)
            triggered = getattr(context.faults, "triggered", None)
            if triggered:  # the injector is reused across resumes, so its counts are cumulative
                metrics["faults_triggered"] = dict(triggered)
            await self.repo.update_run(
                research_id, usage=usage.model_dump(mode="json"), metrics=metrics
            )
            current = await self.repo.get_run(research_id)
            if current is not None and RunStatus(current.status).is_terminal:
                self._views.pop(research_id, None)
                self._contexts.pop(research_id, None)
                await context.index.release(research_id)

    async def _finalize(self, research_id: str, values: Mapping[str, Any]) -> None:
        status = values.get("status")
        status = status if isinstance(status, RunStatus) else RunStatus.COMPLETED
        if status is RunStatus.RUNNING:
            status = RunStatus.FAILED
        await self._set_status(
            research_id,
            status,
            completed_at=utc_now(),
            progress=100.0 if status is RunStatus.COMPLETED else None,
        )
        message = {
            RunStatus.COMPLETED: "Research complete.",
            RunStatus.CANCELLED: "Research cancelled.",
            RunStatus.REJECTED: "Research question rejected.",
        }.get(status, f"Run ended with status {status.value}.")
        await self._emit(research_id, EventType.RUN_STATUS, message, status=status.value)

    # ------------------------------------------------------------------ projection
    async def _on_custom(self, research_id: str, view: _RunView, chunk: Any) -> None:
        if not isinstance(chunk, dict) or "type" not in chunk:
            return
        try:
            event_type = EventType(chunk["type"])
        except ValueError:
            return
        node = chunk.get("node")
        data = dict(chunk.get("data") or {})
        if node:
            data.setdefault("stage", NODE_STAGE.get(node))
        await self._emit(
            research_id, event_type, str(chunk.get("message", ""))[:1000], node=node, **data
        )
        if node and event_type is EventType.NODE_STARTED and node in NODE_STAGE:
            await self.repo.update_run(
                research_id, current_node=node, current_stage=NODE_STAGE[node]
            )
        elif node and event_type is EventType.NODE_COMPLETED and node in NODE_PROGRESS:
            progress = max(view.progress, float(NODE_PROGRESS[node]))
            if progress > view.progress:
                view.progress = progress
                await self.repo.update_run(research_id, progress=progress)

    async def _on_updates(
        self, research_id: str, view: _RunView, namespace: tuple[str, ...], chunk: Any
    ) -> None:
        if not isinstance(chunk, dict):
            return
        for update in chunk.values():
            if not isinstance(update, Mapping):
                continue
            # Inside a worker subgraph only idempotent keys are applied early (for live UI);
            # the worker's full output (counters, errors) arrives once as a top-level update.
            changed = view.apply(update, keys=_LIVE_WORKER_KEYS if namespace else None)
            new_evidence = (update.get("evidence") or {}) if "evidence" in changed else {}
            await self._project(research_id, view, changed, new_evidence)

    async def _project(
        self, research_id: str, view: _RunView, changed: set[str], new_evidence: Mapping[str, Any]
    ) -> None:
        values = view.values
        if not changed:
            return
        if changed & {"sources", "source_quality"}:
            await self.repo.upsert_sources(
                research_id, values.get("sources") or {}, values.get("source_quality") or {}
            )
        if "evidence" in changed and new_evidence:
            await self.repo.upsert_evidence(research_id, new_evidence)
        if "analysis" in changed and values.get("analysis") is not None:
            await self.repo.replace_findings(research_id, values["analysis"].findings)
        if "final_report" in changed and values.get("final_report") is not None:
            await self.repo.save_report(research_id, values["final_report"])
        fields: dict[str, Any] = {}
        if "plan" in changed and values.get("plan") is not None:
            fields["plan"] = values["plan"].model_dump(mode="json")
        if "critique" in changed and values.get("critique") is not None:
            fields["critique"] = values["critique"].model_dump(mode="json")
        if "quality" in changed and values.get("quality") is not None:
            fields["quality"] = values["quality"].model_dump(mode="json")
        if "contradictions" in changed:
            fields["contradictions"] = [
                c.model_dump(mode="json") for c in values.get("contradictions") or []
            ]
        if "iteration" in changed:
            fields["iteration"] = int(values.get("iteration") or 0)
        if "errors" in changed:
            fields["errors"] = [e.model_dump(mode="json") for e in values.get("errors") or []]
        fields["metrics"] = self._metrics(view, None)
        await self.repo.update_run(research_id, **fields)

    def _metrics(self, view: _RunView, usage: UsageSnapshot | None) -> dict[str, Any]:
        v = view.values
        sources = v.get("sources") or {}
        quality = v.get("source_quality") or {}
        critique = v.get("critique")
        report = v.get("final_report")
        latency: dict[str, float] = {}
        for metric in v.get("node_metrics") or []:
            latency[metric.node] = round(latency.get(metric.node, 0.0) + metric.duration_ms, 1)
        metrics: dict[str, Any] = {
            "sources_found": len(sources),
            "sources_processed": sum(1 for s in sources.values() if s.status == "processed"),
            "sources_failed": sum(1 for s in sources.values() if s.status == "failed"),
            "sources_degraded": sum(
                1 for s in sources.values() if s.content_origin == "search_snippet"
            ),
            "high_quality_sources": sum(1 for q in quality.values() if q.tier is QualityTier.HIGH),
            "evidence_items": len(v.get("evidence") or {}),
            "findings": len(v["analysis"].findings) if v.get("analysis") else 0,
            "contradictions": len(v.get("contradictions") or []),
            "critic_issues": len(critique.issues) if critique else 0,
            "critic_verdict": critique.verdict if critique else None,
            "quality_score": v["quality"].score if v.get("quality") else None,
            "iterations": int(v.get("iteration") or 0),
            "tool_calls_used": int(v.get("tool_calls_used") or 0),
            "errors": len(v.get("errors") or []),
            "active_seconds": round(view.active_seconds, 2),
            "node_latency_ms": latency,
        }
        if report is not None:
            metrics["report"] = report.metrics.model_dump(mode="json")
        if usage is not None:
            metrics["llm_calls"] = usage.llm_calls
            metrics["total_tokens"] = usage.total_tokens
            metrics["estimated_cost_usd"] = usage.estimated_cost_usd
        return metrics

    # ------------------------------------------------------------------ events & status
    async def _next_seq(self, research_id: str) -> int:
        if research_id not in self._seq:
            # The lock stops two first-time emitters from both loading the same last seq.
            async with self._seq_lock:
                if research_id not in self._seq:
                    self._seq[research_id] = await self.repo.last_event_seq(research_id)
        self._seq[research_id] += 1  # no await between read and write: atomic on the event loop
        return self._seq[research_id]

    async def _emit(
        self,
        research_id: str,
        event_type: EventType,
        message: str,
        *,
        node: str | None = None,
        **data: Any,
    ) -> None:
        event = WorkflowEvent(
            seq=await self._next_seq(research_id),
            research_id=research_id,
            type=event_type,
            node=node,
            message=message,
            data={k: v for k, v in data.items() if v is not None},
        )
        await self.repo.append_events([event])
        self.broker.publish(event)

    async def _set_status(self, research_id: str, status: RunStatus, **fields: Any) -> None:
        await self.repo.update_run(
            research_id, status=status.value, **{k: v for k, v in fields.items() if v is not None}
        )
