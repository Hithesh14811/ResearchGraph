"""HTTP API tests (in-process ASGI transport; no network, no real LLM)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.main import create_app
from researchgraph.runtime.bootstrap import AppServices, open_services
from researchgraph.schemas.common import RunStatus
from tests.conftest import make_settings, wait_for_status


async def _client_for(services: AppServices) -> httpx.AsyncClient:
    app = create_app(services.settings)
    app.state.services = services  # ASGITransport does not run lifespan; inject the live services
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client(services: AppServices) -> AsyncIterator[httpx.AsyncClient]:
    async with await _client_for(services) as c:
        yield c


async def test_health_and_graph(client: httpx.AsyncClient) -> None:
    health = (await client.get("/health")).json()
    assert (
        health["status"] == "ok"
        and health["llm_provider"] == "mock"
        and health["database"] == "sqlite"
    )
    mermaid = (await client.get("/graph")).json()["mermaid"]
    assert "plan_review" in mermaid and "research_worker" in mermaid and "quality_gate" in mermaid


async def test_full_lifecycle_with_plan_edit(
    client: httpx.AsyncClient, services: AppServices
) -> None:
    created = await client.post("/research", json={"question": SAMPLE_QUESTIONS[0].question})
    assert created.status_code == 202
    research_id = created.json()["id"]

    await wait_for_status(services, research_id, RunStatus.AWAITING_APPROVAL)
    run = (await client.get(f"/research/{research_id}")).json()
    assert run["awaiting_approval"] and run["status"] == "awaiting_approval"
    plan_response = (await client.get(f"/research/{research_id}/plan")).json()
    assert plan_response["editable"] and len(plan_response["plan"]["subquestions"]) == 5

    # Report is not available yet -> 409.
    assert (await client.get(f"/research/{research_id}/report")).status_code == 409

    plan = plan_response["plan"]
    plan["subquestions"] = plan["subquestions"][:3]
    edited = await client.post(
        f"/research/{research_id}/edit-plan", json={"plan": plan, "approve": True}
    )
    assert edited.status_code == 200
    await wait_for_status(services, research_id, RunStatus.COMPLETED)

    run = (await client.get(f"/research/{research_id}")).json()
    assert (
        run["progress"] == 100
        and run["metrics"]["evidence_items"] > 0
        and run["usage"]["llm_calls"] > 0
    )
    report = (await client.get(f"/research/{research_id}/report")).json()
    assert (
        report["bibliography"]
        and report["metrics"]["citation_coverage"] == 1.0
        and report["quality"]
    )
    markdown = await client.get(f"/research/{research_id}/report", params={"format": "markdown"})
    assert markdown.headers["content-type"].startswith(
        "text/markdown"
    ) and markdown.text.startswith("# ")

    sources = (await client.get(f"/research/{research_id}/sources")).json()
    assert (
        sources["total"] == len(sources["items"]) > 0
        and sources["items"][0]["quality"]["overall"] >= sources["items"][-1]["quality"]["overall"]
    )
    evidence = (
        await client.get(f"/research/{research_id}/evidence", params={"subquestion_id": "SQ1"})
    ).json()
    assert evidence["items"] and all(e["subquestion_id"] == "SQ1" for e in evidence["items"])
    assert {
        e["subquestion_id"]
        for e in (await client.get(f"/research/{research_id}/evidence")).json()["items"]
    } <= {"SQ1", "SQ2", "SQ3"}
    findings = (await client.get(f"/research/{research_id}/findings")).json()["items"]
    assert findings and all(f["evidence_ids"] for f in findings)

    events = (await client.get(f"/research/{research_id}/events")).json()["items"]
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))
    types = {e["type"] for e in events}
    assert {"plan_ready", "node_started", "node_completed", "progress", "run_status"} <= types

    listing = (await client.get("/research")).json()
    assert listing["total"] >= 1 and listing["items"][0]["id"] == research_id


async def test_sse_stream_replays_and_terminates(
    client: httpx.AsyncClient, services: AppServices
) -> None:
    research_id = (
        await client.post(
            "/research", json={"question": SAMPLE_QUESTIONS[2].question, "auto_approve": True}
        )
    ).json()["id"]
    await wait_for_status(services, research_id, RunStatus.COMPLETED)
    async with client.stream("GET", f"/research/{research_id}/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])
    assert body.count("data: ") > 20
    assert '"type": "run_status"' in body and '"status": "completed"' in body
    # Regression: named SSE events collide with EventSource's built-in "error" event.
    assert "event:" not in body
    async with client.stream(
        "GET", f"/research/{research_id}/stream", headers={"Last-Event-ID": "5"}
    ) as response:
        resumed = "".join([chunk async for chunk in response.aiter_text()])
    assert "id: 5\n" not in resumed and "id: 6\n" in resumed


async def test_cancel_and_invalid_state_transitions(
    client: httpx.AsyncClient, services: AppServices
) -> None:
    research_id = (
        await client.post("/research", json={"question": SAMPLE_QUESTIONS[1].question})
    ).json()["id"]
    await wait_for_status(services, research_id, RunStatus.AWAITING_APPROVAL)
    cancelled = await client.post(f"/research/{research_id}/cancel")
    assert cancelled.status_code == 200
    run = await wait_for_status(services, research_id, RunStatus.CANCELLED)
    assert run.status == "cancelled"
    assert (await client.post(f"/research/{research_id}/approve")).status_code == 409
    assert (await client.post(f"/research/{research_id}/cancel")).status_code == 409
    assert (await client.get("/research/rg_missing")).status_code == 404


async def test_request_validation_and_limits(client: httpx.AsyncClient) -> None:
    assert (await client.post("/research", json={"question": "too short"})).status_code == 422
    assert (
        await client.post(
            "/research",
            json={"question": "A valid research question here", "failure_scenarios": ["meteor"]},
        )
    ).status_code == 400
    assert (await client.get("/research", params={"limit": 1000})).status_code == 422
    big = {"question": "x" * 300_000}
    assert (await client.post("/research", json=big)).status_code == 413


async def test_failure_simulation_via_api(client: httpx.AsyncClient, services: AppServices) -> None:
    body = {
        "question": SAMPLE_QUESTIONS[0].question,
        "auto_approve": True,
        "failure_scenarios": ["all"],
    }
    research_id = (await client.post("/research", json=body)).json()["id"]
    run = await wait_for_status(services, research_id, RunStatus.COMPLETED)
    assert run.metrics["faults_triggered"]["critic_rejection"] == 1
    assert run.iteration >= 2 and run.errors


async def test_api_token_is_enforced_when_configured(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, api_token="s3cret-token")
    async with (
        open_services(settings, recover=False) as services,
        await _client_for(services) as client,
    ):
        assert (await client.get("/health")).status_code == 200  # health stays public
        assert (await client.get("/research")).status_code == 401
        ok = await client.get("/research", headers={"Authorization": "Bearer s3cret-token"})
        assert ok.status_code == 200
        assert (await client.get("/research", params={"token": "wrong"})).status_code == 401


async def test_active_run_limit_returns_429(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_active_runs=1, mock_llm_latency_ms=50)
    async with (
        open_services(settings, recover=False) as services,
        await _client_for(services) as client,
    ):
        body = {"question": SAMPLE_QUESTIONS[0].question, "auto_approve": True}
        first = await client.post("/research", json=body)
        assert first.status_code == 202
        second = await client.post("/research", json=body)
        assert second.status_code == 429 and second.json()["code"] == "too_many_runs"
        await wait_for_status(services, first.json()["id"], RunStatus.COMPLETED, within=60)
        assert (await client.post("/research", json=body)).status_code == 202  # capacity freed


async def test_concurrent_approvals_resume_exactly_once(
    client: httpx.AsyncClient, services: AppServices
) -> None:
    import asyncio

    research_id = (
        await client.post("/research", json={"question": SAMPLE_QUESTIONS[1].question})
    ).json()["id"]
    await wait_for_status(services, research_id, RunStatus.AWAITING_APPROVAL)
    responses = await asyncio.gather(
        *(client.post(f"/research/{research_id}/approve") for _ in range(3))
    )
    assert sorted(r.status_code for r in responses) == [200, 409, 409]
    await wait_for_status(services, research_id, RunStatus.COMPLETED)
    events = (await client.get(f"/research/{research_id}/events")).json()["items"]
    assert sum("Plan review decision" in e["message"] for e in events) == 1
