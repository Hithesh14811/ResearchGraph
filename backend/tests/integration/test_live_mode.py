"""Live mode gate: demo by default, real-provider runs only with the password token.

The "live" provider here is configured as OpenAI with a fake key, and its models are swapped
for the mock so nothing leaves the process; what is under test is the gate, not the provider.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from researchgraph.api.live_auth import AttemptLimiter, issue_token, verify_token
from researchgraph.database.engine import create_engine, init_models
from researchgraph.llm.factory import build_model_registry
from researchgraph.main import create_app
from researchgraph.runtime.bootstrap import AppServices, open_services
from researchgraph.schemas.common import RunStatus
from tests.conftest import make_settings, wait_for_status

PASSWORD = "correct horse battery staple"
QUESTION = "Analyze the tradeoffs between small and large language models for production inference."


def _gated_settings(tmp_path: Path, **overrides: object):  # type: ignore[no-untyped-def]
    values: dict[str, object] = {
        "llm_provider": "openai",
        "model_name": "live-model",
        "openai_api_key": "sk-test",
        "public_demo": True,
        "live_mode_password": PASSWORD,
    }
    values.update(overrides)
    return make_settings(tmp_path, **values)


@pytest.fixture
async def gated(tmp_path: Path) -> AsyncIterator[tuple[AppServices, httpx.AsyncClient]]:
    settings = _gated_settings(tmp_path)
    async with open_services(settings, recover=False) as services:
        services.container._models = build_model_registry(make_settings(tmp_path))  # offline
        app = create_app(settings)
        app.state.services = services
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield services, client


async def _unlock(client: httpx.AsyncClient) -> dict[str, str]:
    response = await client.post("/auth/live", json={"password": PASSWORD})
    assert response.status_code == 200
    return {"X-Live-Token": response.json()["token"]}


async def test_demo_by_default_and_health_describes_both_lanes(
    gated: tuple[AppServices, httpx.AsyncClient],
) -> None:
    services, client = gated
    health = (await client.get("/health")).json()
    assert health["llm_provider"] == "mock" and health["search_provider"] == "mock"
    assert health["live_mode"] == {
        "llm_provider": "openai",
        "model": "live-model",
        "search_provider": "mock",
    }
    created = await client.post("/research", json={"question": QUESTION, "auto_approve": True})
    assert created.status_code == 202 and created.json()["mode"] == "demo"
    await wait_for_status(services, created.json()["id"], RunStatus.COMPLETED)
    demo, live = (services.container.context_for_run(live=flag) for flag in (False, True))
    assert demo.settings.llm_provider == "mock" and live.settings.llm_provider == "openai"


async def test_live_runs_need_the_password_token(
    gated: tuple[AppServices, httpx.AsyncClient],
) -> None:
    services, client = gated
    body = {"question": QUESTION, "auto_approve": True, "mode": "live"}
    assert (await client.post("/research", json=body)).status_code == 401
    forged = {"X-Live-Token": "9999999999.deadbeef"}
    assert (await client.post("/research", json=body, headers=forged)).status_code == 401
    assert (await client.post("/auth/live", json={"password": "wrong"})).status_code == 401

    headers = await _unlock(client)
    created = await client.post("/research", json=body, headers=headers)
    assert created.status_code == 202 and created.json()["mode"] == "live"
    await wait_for_status(services, created.json()["id"], RunStatus.COMPLETED)

    # Simulated LLM failures are a demo-lane feature.
    faulty = {**body, "failure_scenarios": ["llm_timeout"]}
    assert (await client.post("/research", json=faulty, headers=headers)).status_code == 409


async def test_resuming_a_live_run_needs_the_token(
    gated: tuple[AppServices, httpx.AsyncClient],
) -> None:
    services, client = gated
    headers = await _unlock(client)
    created = await client.post(
        "/research", json={"question": QUESTION, "mode": "live"}, headers=headers
    )
    research_id = created.json()["id"]
    await wait_for_status(services, research_id, RunStatus.AWAITING_APPROVAL)
    assert (await client.post(f"/research/{research_id}/approve")).status_code == 401
    feedback = {"feedback": "add a subquestion on cost"}
    assert (await client.post(f"/research/{research_id}/replan", json=feedback)).status_code == 401
    approved = await client.post(f"/research/{research_id}/approve", headers=headers)
    assert approved.status_code == 200
    await wait_for_status(services, research_id, RunStatus.COMPLETED)


async def test_password_guessing_is_throttled(
    gated: tuple[AppServices, httpx.AsyncClient],
) -> None:
    _, client = gated
    codes = [
        (await client.post("/auth/live", json={"password": f"guess-{i}"})).status_code
        for i in range(6)
    ]
    assert codes == [401] * 5 + [429]
    # Even the right password is refused while throttled.
    assert (await client.post("/auth/live", json={"password": PASSWORD})).status_code == 429


async def test_public_demo_without_password_never_runs_live(tmp_path: Path) -> None:
    settings = _gated_settings(tmp_path, live_mode_password=None)
    async with open_services(settings, recover=False) as services:
        app = create_app(settings)
        app.state.services = services
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).json()["live_mode"] is None
            assert (await client.post("/auth/live", json={"password": "x"})).status_code == 404
            live = {"question": QUESTION, "mode": "live"}
            assert (await client.post("/research", json=live)).status_code == 403


def test_tokens_expire_and_are_bound_to_the_password() -> None:
    token, expires_at = issue_token(PASSWORD, ttl_seconds=60, now=1_000)
    assert expires_at == 1_060
    assert verify_token(token, PASSWORD, now=1_030)
    assert not verify_token(token, PASSWORD, now=1_061)  # expired
    assert not verify_token(token, "a changed password", now=1_030)  # rotation revokes
    payload, signature = token.split(".")
    assert not verify_token(f"{int(payload) + 9999}.{signature}", PASSWORD, now=1_030)
    assert not verify_token(None, PASSWORD) and not verify_token("garbage", PASSWORD)


def test_attempt_limiter_window_and_global_cap() -> None:
    limiter = AttemptLimiter(per_client=2, overall=3, window_seconds=10)
    limiter.record_failure("a", now=0)
    limiter.record_failure("a", now=1)
    assert limiter.blocked("a", now=2) and not limiter.blocked("b", now=2)
    assert not limiter.blocked("a", now=12)  # window slid past both failures
    limiter.record_failure("c", now=13)
    limiter.record_failure("d", now=13)
    limiter.record_failure("e", now=13)
    assert limiter.blocked("fresh-client", now=14)  # global cap reached


async def test_existing_databases_gain_the_mode_column(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:  # a research_runs table from before the column existed
        db.execute("CREATE TABLE research_runs (id VARCHAR(40) PRIMARY KEY, question TEXT)")
        db.execute("INSERT INTO research_runs VALUES ('rg_old', 'q')")
    engine = create_engine(f"sqlite+aiosqlite:///{path.as_posix()}")
    await init_models(engine)
    await init_models(engine)  # idempotent
    await engine.dispose()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT mode FROM research_runs").fetchone() == ("demo",)
