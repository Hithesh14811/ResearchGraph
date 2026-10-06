"""Evaluation harness.

    python -m evaluation.run                      # offline demo mode (mock model, synthetic corpus)
    python -m evaluation.run --judge              # + LLM-as-judge (mock judge in demo mode)
    python -m evaluation.run --mode live --judge  # real provider/search from your .env
    python -m evaluation.run --only rag-hallucinations --failures all

Each question runs through the full stack (run manager, checkpointing, persistence) with the
plan auto-approved. Results are written to ``evaluation/results/<timestamp>-<mode>.{json,md}``.

Deterministic metrics measure the pipeline (grounding, coverage, source quality, latency,
tokens). Judge scores are model opinions and are reported separately. In demo mode both
describe the behaviour of the *mock* model on a *synthetic* corpus — they validate the
machinery, not real-world research quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from researchgraph.config.logging import configure_logging
from researchgraph.config.settings import Settings
from researchgraph.core.aio import use_selector_event_loop_on_windows
from researchgraph.demo.faults import parse_scenarios
from researchgraph.evaluation.judge import JudgeOutput, judge_report
from researchgraph.evaluation.metrics import (
    CheckResult,
    ExpectedCharacteristics,
    RunMetrics,
    check_expectations,
    compute_run_metrics,
)
from researchgraph.llm.factory import build_model_registry
from researchgraph.runtime.bootstrap import AppServices, open_services
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.events import EventType, WorkflowEvent

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "research_questions.json"
HEADLINE = [
    "citation_coverage",
    "unsupported_claim_rate",
    "first_pass_citation_precision",
    "evidence_quality",
    "evidence_relevance",
    "research_completeness",
    "latency_seconds",
    "total_tokens",
]


class QuestionResult(BaseModel):
    id: str
    question: str
    research_id: str
    metrics: RunMetrics
    checks: list[CheckResult]
    judge: JudgeOutput | None = None
    judge_error: str | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def eval_settings(mode: str, workdir: Path) -> Settings:
    common: dict[str, Any] = {
        "database_url": f"sqlite+aiosqlite:///{(workdir / 'eval.db').as_posix()}",
        "resume_runs_on_startup": False,
        "log_level": "ERROR",
    }
    if mode == "demo":
        return Settings(
            _env_file=None,  # type: ignore[call-arg]
            llm_provider="mock",
            search_provider="mock",
            embedding_provider="hashing",
            vector_store="memory",
            langchain_tracing_v2=False,
            **common,
        )
    return Settings(**common)


async def run_question(
    services: AppServices, item: dict[str, Any], *, failures: frozenset[str], judge: bool
) -> QuestionResult:
    expected = ExpectedCharacteristics.model_validate(item.get("expected", {}))
    started = time.perf_counter()
    research_id = await services.manager.create_run(
        item["question"], auto_approve=True, failure_scenarios=failures
    )
    await services.manager.wait(research_id)
    latency = time.perf_counter() - started

    run = await services.repository.get_run(research_id)
    assert run is not None
    snapshot = await services.graph.aget_state({"configurable": {"thread_id": research_id}})
    state = dict(snapshot.values)
    records = await services.repository.list_events(research_id, limit=20_000)
    events = [
        WorkflowEvent(
            seq=r.seq,
            research_id=research_id,
            type=EventType(r.type),
            node=r.node,
            message=r.message,
            data=r.data or {},
        )
        for r in records
    ]
    metrics = compute_run_metrics(
        state, events, usage=run.usage or {}, latency_seconds=latency, expected=expected
    )
    report = state.get("final_report")
    checks = check_expectations(metrics, expected, report.markdown if report else "")
    result = QuestionResult(
        id=item["id"],
        question=item["question"],
        research_id=research_id,
        metrics=metrics,
        checks=checks,
    )

    if judge and report is not None and run.status == RunStatus.COMPLETED.value:
        evidence = state.get("evidence") or {}
        sample = [
            {
                "id": e.id,
                "claim": e.claim,
                "quote": e.supporting_text[:300],
                "source": e.source_title[:120],
            }
            for e in list(evidence.values())[:25]
        ]
        try:
            models = build_model_registry(services.settings)
            result.judge = await judge_report(
                models,
                question=item["question"],
                report_markdown=report.markdown,
                evidence_sample=sample,
            )
        except Exception as exc:  # judge failures never invalidate deterministic metrics
            result.judge_error = f"{type(exc).__name__}: {exc}"
    return result


def aggregate(results: list[QuestionResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {"questions": len(results), "passed": sum(r.passed for r in results)}
    for name in RunMetrics.model_fields:
        values = [getattr(r.metrics, name) for r in results]
        numeric = [
            float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)
        ]
        if numeric and len(numeric) == len([v for v in values if v is not None]):
            summary[name] = round(statistics.fmean(numeric), 4)
    checks = [c for r in results for c in r.checks]
    summary["check_pass_rate"] = (
        round(sum(c.passed for c in checks) / len(checks), 4) if checks else None
    )
    judged = [r.judge for r in results if r.judge]
    if judged:
        summary["judge_mean"] = round(statistics.fmean(j.mean for j in judged), 3)
    return summary


def to_markdown(
    meta: dict[str, Any], results: list[QuestionResult], summary: dict[str, Any]
) -> str:
    lines = [
        f"# ResearchGraph evaluation — {meta['timestamp']}",
        "",
        f"- Mode: **{meta['mode']}** (LLM: `{meta['llm']}`, search: `{meta['search']}`)",
        f"- Dataset: `{meta['dataset']}` ({summary['questions']} questions)",
        f"- Simulated failures: {', '.join(meta['failures']) or 'none'}",
    ]
    if meta["mode"] == "demo":
        lines.append(
            "- **Note:** demo mode uses a deterministic mock model and a synthetic corpus. These numbers "
            "validate the pipeline's mechanics (grounding, verification, routing), not real research quality."
        )
    lines += [
        "",
        "## Deterministic metrics",
        "",
        "| Question | Status | " + " | ".join(HEADLINE) + " | Checks |",
        "|---|---|" + "---|" * len(HEADLINE) + "---|",
    ]
    for r in results:
        values = []
        for name in HEADLINE:
            value = getattr(r.metrics, name)
            values.append(
                "—"
                if value is None
                else (f"{value:.3f}" if isinstance(value, float) else str(value))
            )
        passed = sum(c.passed for c in r.checks)
        lines.append(
            f"| `{r.id}` | {r.metrics.status} | "
            + " | ".join(values)
            + f" | {passed}/{len(r.checks)} |"
        )
    mean_values = []
    for name in HEADLINE:
        value = summary.get(name)
        mean_values.append("—" if value is None else f"{value:.3f}")
    lines.append(
        "| **mean** | | "
        + " | ".join(mean_values)
        + f" | {summary.get('check_pass_rate', 0):.0%} |"
    )

    failed = [(r.id, c) for r in results for c in r.checks if not c.passed]
    lines += ["", "## Failed checks", ""]
    lines += [f"- `{qid}` — {c.name}: {c.detail}" for qid, c in failed] or ["None."]

    judged = [r for r in results if r.judge or r.judge_error]
    if judged:
        lines += ["", "## LLM-as-judge (subjective; reported separately)", ""]
        if meta["mode"] == "demo":
            lines.append(
                "_Demo mode uses the mock judge, whose scores come from report structure only and are not meaningful._\n"
            )
        lines += [
            "| Question | faithfulness | completeness | balance | actionability | clarity | mean |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in judged:
            if r.judge:
                j = r.judge
                lines.append(
                    f"| `{r.id}` | {j.faithfulness} | {j.completeness} | {j.balance} | {j.actionability} | {j.clarity} | {j.mean:.2f} |"
                )
            else:
                lines.append(f"| `{r.id}` | judge failed: {r.judge_error} | | | | | |")
    return "\n".join(lines) + "\n"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--only", nargs="*", help="question ids to run")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge rubric scores")
    parser.add_argument(
        "--failures", default="", help="simulate failures (demo mode): comma list or 'all'"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    configure_logging("ERROR")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    items = [q for q in dataset["questions"] if args.mode in q.get("modes", ["demo", "live"])]
    if args.only:
        items = [q for q in items if q["id"] in set(args.only)]
    failures = parse_scenarios(args.failures) if args.failures else frozenset()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = args.output / ".work" / timestamp
    workdir.mkdir(parents=True, exist_ok=True)
    settings = eval_settings(args.mode, workdir)

    results: list[QuestionResult] = []
    async with open_services(settings, recover=False) as services:
        for item in items:
            print(f"- {item['id']}: running...", flush=True)
            result = await run_question(services, item, failures=failures, judge=args.judge)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"  {status}  coverage={result.metrics.citation_coverage:.2f} unsupported={result.metrics.unsupported_claim_rate:.3f} "
                f"completeness={result.metrics.research_completeness:.2f} latency={result.metrics.latency_seconds:.1f}s",
                flush=True,
            )

    summary = aggregate(results)
    meta = {
        "timestamp": timestamp,
        "mode": args.mode,
        "llm": f"{settings.llm_provider}/{settings.model_name}",
        "search": settings.search_provider,
        "dataset": str(
            args.dataset.relative_to(ROOT.parent)
            if args.dataset.is_relative_to(ROOT.parent)
            else args.dataset
        ),
        "failures": sorted(failures),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"{timestamp}-{args.mode}"
    payload = {
        "meta": meta,
        "summary": summary,
        "results": [r.model_dump(mode="json") for r in results],
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(to_markdown(meta, results, summary), encoding="utf-8")
    print(
        f"\nPassed {summary['passed']}/{summary['questions']} questions; check pass rate {summary['check_pass_rate']:.0%}"
    )
    print(f"Results: {stem.with_suffix('.md')}")
    return 0 if summary["passed"] == summary["questions"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    use_selector_event_loop_on_windows()
    sys.exit(asyncio.run(main()))
