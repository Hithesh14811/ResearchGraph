"""Fault-injection points (chaos-engineering hooks).

Production code calls ``faults.should_fail(point, key)`` at a handful of well-defined
places (fetching a source, running a search, calling a model). The default injector never
fails; demo mode swaps in a scenario-driven injector to prove the graph recovers from
failures instead of only working on the happy path.
"""

from __future__ import annotations

from typing import Protocol


class FaultInjector(Protocol):
    def should_fail(self, point: str, key: str = "") -> bool: ...


class NoFaults:
    """Production default: never injects a failure."""

    def should_fail(self, point: str, key: str = "") -> bool:
        return False
