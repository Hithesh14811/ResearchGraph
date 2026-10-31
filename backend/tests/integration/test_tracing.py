"""Optional Langfuse tracing: off without keys, attached per run with keys, never fatal.

The ``langfuse`` package is replaced by an in-memory fake, so no test touches the network.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from researchgraph.config.observability import configure_langfuse
from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.runtime.bootstrap import open_services
from researchgraph.schemas.common import RunStatus
from tests.conftest import make_settings, wait_for_status

KEYS = {"langfuse_public_key": "pk-lf-test", "langfuse_secret_key": "sk-lf-test"}


class FakeLangfuse:
    instances: ClassVar[list[FakeLangfuse]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.flushes = 0
        FakeLangfuse.instances.append(self)

    @staticmethod
    def create_trace_id(*, seed: str | None = None) -> str:
        return f"trace-{seed}"

    def flush(self) -> None:
        self.flushes += 1


class FakeHandler(BaseCallbackHandler):
    instances: ClassVar[list[FakeHandler]] = []

    def __init__(self, *, public_key: str, trace_context: dict[str, str]) -> None:
        self.public_key = public_key
        self.trace_context = trace_context
        self.llm_starts: list[dict[str, Any]] = []
        FakeHandler.instances.append(self)

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.llm_starts.append(kwargs.get("metadata") or {})


class BrokenHandler(FakeHandler):
    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        raise RuntimeError("langfuse is down")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        raise RuntimeError("langfuse is down")


@pytest.fixture
def fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    FakeLangfuse.instances.clear()
    FakeHandler.instances.clear()
    root = types.ModuleType("langfuse")
    root.Langfuse = FakeLangfuse  # type: ignore[attr-defined]
    integration = types.ModuleType("langfuse.langchain")
    integration.CallbackHandler = FakeHandler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", root)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", integration)
    return integration


@pytest.mark.parametrize("keys", [{}, {"langfuse_public_key": "pk-lf-test"}])
def test_langfuse_is_disabled_without_both_keys(
    tmp_path: Path, fake_langfuse: types.ModuleType, keys: dict[str, str]
) -> None:
    assert configure_langfuse(make_settings(tmp_path, **keys)) is None
    assert FakeLangfuse.instances == []


def test_missing_package_logs_and_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "langfuse", None)  # makes the import raise ImportError
    with caplog.at_level(logging.WARNING):
        assert configure_langfuse(make_settings(tmp_path, **KEYS)) is None
    assert "package is missing" in caplog.text
    assert "sk-lf-test" not in caplog.text


async def test_handler_traces_every_llm_call_of_a_run(
    tmp_path: Path, fake_langfuse: types.ModuleType
) -> None:
    settings = make_settings(tmp_path, **KEYS, langfuse_host="https://lf.example")
    async with open_services(settings, recover=False) as services:
        research_id = await services.manager.create_run(SAMPLE_QUESTIONS[0].question)
        await wait_for_status(services, research_id, RunStatus.AWAITING_APPROVAL)
        await services.manager.approve(research_id)
        await wait_for_status(services, research_id, RunStatus.COMPLETED)
        client = FakeLangfuse.instances[0]
        assert client.flushes == 0

    assert client.kwargs == {
        "public_key": "pk-lf-test",
        "secret_key": "sk-lf-test",
        "base_url": "https://lf.example",
    }
    assert client.flushes == 1  # flushed on shutdown
    # One handler per execution segment (plan, then post-approval), both in the same trace.
    assert len(FakeHandler.instances) == 2
    assert {h.trace_context["trace_id"] for h in FakeHandler.instances} == {f"trace-{research_id}"}
    starts = [m for h in FakeHandler.instances for m in h.llm_starts]
    assert starts and all(m.get("langfuse_session_id") == research_id for m in starts)


async def test_langfuse_failures_never_fail_a_run(
    tmp_path: Path, fake_langfuse: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_langfuse.CallbackHandler = BrokenHandler  # type: ignore[attr-defined]

    def broken_flush(self: FakeLangfuse) -> None:
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(FakeLangfuse, "flush", broken_flush)
    settings = make_settings(tmp_path, **KEYS)
    async with open_services(settings, recover=False) as services:
        research_id = await services.manager.create_run(
            SAMPLE_QUESTIONS[0].question, auto_approve=True
        )
        run = await wait_for_status(services, research_id, RunStatus.COMPLETED)
        assert run.progress == 100
        assert FakeHandler.instances  # the broken handler really was attached
    # Leaving the context ran the failing flush without raising.
