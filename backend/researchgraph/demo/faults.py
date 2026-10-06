"""Scenario-driven fault injection for demo mode.

Each scenario fails a specific, bounded set of operations so the run demonstrates
recovery rather than collapsing:

* ``failed_source``         — the first two distinct source fetches fail (the worker falls
                              back to the search snippet or records the source as failed).
* ``llm_timeout``           — the first planner call times out (retried with backoff).
* ``invalid_output``        — the first critic response fails schema validation (repaired by
                              re-prompting with the validation error).
* ``insufficient_evidence`` — all searches for one subquestion return nothing in round 1
                              (the evidence gate routes back for another round).
* ``critic_rejection``      — the first critique raises a critical issue (the quality gate
                              triggers targeted research).
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable

SCENARIOS = (
    "failed_source",
    "llm_timeout",
    "invalid_output",
    "insufficient_evidence",
    "critic_rejection",
)


def parse_scenarios(spec: Iterable[str] | str) -> frozenset[str]:
    items = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if s.strip()]
    if "all" in items:
        return frozenset(SCENARIOS)
    unknown = set(items) - set(SCENARIOS)
    if unknown:
        raise ValueError(
            f"Unknown failure scenario(s): {', '.join(sorted(unknown))}. Valid: {', '.join(SCENARIOS)}, all"
        )
    return frozenset(items)


class ScenarioFaultInjector:
    MAX_FAILED_SOURCES = 2

    def __init__(self, scenarios: Iterable[str] | str) -> None:
        self._active = parse_scenarios(scenarios)
        self._lock = threading.Lock()
        self._fired: set[str] = set()
        self._failed_sources: set[str] = set()
        self._targets: dict[str, str] = {}
        self.triggered: Counter[str] = Counter()

    def should_fail(self, point: str, key: str = "") -> bool:
        if point not in self._active:
            return False
        with self._lock:
            fail = self._decide(point, key)
            if fail:
                self.triggered[point] += 1
            return fail

    def _once(self, point: str) -> bool:
        if point in self._fired:
            return False
        self._fired.add(point)
        return True

    def _decide(self, point: str, key: str) -> bool:
        if point == "failed_source":
            if key in self._failed_sources:
                return True
            if len(self._failed_sources) < self.MAX_FAILED_SOURCES:
                self._failed_sources.add(key)
                return True
            return False
        if point == "insufficient_evidence":
            if not key.endswith(":1"):
                return False
            return self._targets.setdefault(point, key) == key
        if point == "llm_timeout":
            return key == "PlannerOutput" and self._once(point)
        if point == "invalid_output":
            return key == "CriticOutput" and self._once(point)
        if point == "critic_rejection":
            return self._once(point)
        return False
