"""HTTP routes."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from researchgraph import __version__
from researchgraph.api.deps import get_services, require_token
from researchgraph.api.live_auth import (
    AttemptLimiter,
    issue_token,
    password_matches,
    verify_token,
)
from researchgraph.api.sse import event_stream
from researchgraph.core.errors import InvalidRunStateError, RunNotFoundError
from researchgraph.database.models import ResearchRun
from researchgraph.demo.faults import parse_scenarios
from researchgraph.runtime.bootstrap import AppServices
from researchgraph.schemas.api import (
    MAX_PAGE_SIZE,
    CreateResearchRequest,
    EditPlanRequest,
    EventItem,
    EventsResponse,
    EvidenceResponse,
    FindingsResponse,
    GraphResponse,
    HealthResponse,
    LiveAuthRequest,
    LiveAuthResponse,
    LiveModeInfo,
    PlanResponse,
    ReplanRequest,
    ReportResponse,
    RunListResponse,
    RunMode,
    RunResponse,
    SourceItem,
    SourcesResponse,
)
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.evidence import Contradiction, Evidence
from researchgraph.schemas.report import FinalReport, ResearchFinding
from researchgraph.schemas.review import Critique, QualityAssessment
from researchgraph.schemas.sources import Source, SourceQuality

Services = Annotated[AppServices, Depends(get_services)]
LiveToken = Annotated[str | None, Header(alias="X-Live-Token")]
router = APIRouter(dependencies=[Depends(require_token)])
public = APIRouter()


def _to_response(run: ResearchRun) -> RunResponse:
    return RunResponse(
        id=run.id,
        question=run.question,
        status=RunStatus(run.status),
        current_node=run.current_node,
        current_stage=run.current_stage,
        progress=run.progress or 0.0,
        iteration=run.iteration or 0,
        auto_approve=run.auto_approve,
        mode="live" if run.mode == "live" else "demo",
        failure_scenarios=list(run.failure_scenarios or []),
        metrics=run.metrics or {},
        usage=run.usage or {},
        error=run.error,
        created_at=run.created_at,
        updated_at=run.updated_at,
        completed_at=run.completed_at,
        awaiting_approval=run.status == RunStatus.AWAITING_APPROVAL.value,
    )


async def _get_run(services: AppServices, research_id: str) -> ResearchRun:
    run = await services.repository.get_run(research_id)
    if run is None:
        raise RunNotFoundError(f"Research run {research_id} not found")
    return run


def _require_live(services: AppServices, token: str | None) -> None:
    """Live runs spend provider credits: only a valid token from POST /auth/live may start them."""
    settings = services.settings
    if not settings.live_mode_available or settings.live_mode_password is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Live mode is not enabled on this server")
    if not verify_token(token, settings.live_mode_password.get_secret_value()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Live mode is locked: unlock it with the password"
        )


def _resolve_mode(services: AppServices, requested: RunMode, token: str | None) -> RunMode:
    settings = services.settings
    if not settings.live_gate_enabled:  # a single lane: whatever the server is configured with
        return "demo" if settings.is_mock_llm else "live"
    if requested == "live":
        _require_live(services, token)
        return "live"
    return "demo"


async def _guard_live_run(services: AppServices, research_id: str, token: str | None) -> None:
    """Resuming a live run spends credits too, so plan decisions on it need the token."""
    run = await _get_run(services, research_id)
    if run.mode == "live" and services.settings.live_gate_enabled:
        _require_live(services, token)


def _limiter(request: Request) -> AttemptLimiter:
    limiter: AttemptLimiter | None = getattr(request.app.state, "live_limiter", None)
    if limiter is None:
        limiter = request.app.state.live_limiter = AttemptLimiter()
    return limiter


@public.get("/health", response_model=HealthResponse, tags=["system"])
async def health(services: Services) -> HealthResponse:
    s = services.settings
    lane = s.demo_variant() if s.live_gate_enabled else s  # what a run gets by default
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=s.environment,
        llm_provider=lane.llm_provider,
        model=lane.model_name,
        search_provider=lane.search_provider,
        embedding_provider=s.embedding_provider,
        vector_store=s.resolved_vector_store,
        database="postgresql" if s.uses_postgres else "sqlite",
        tracing_enabled=services.tracing_enabled,
        active_runs=services.manager.active_count,
        live_mode=LiveModeInfo(
            llm_provider=s.llm_provider, model=s.model_name, search_provider=s.search_provider
        )
        if s.live_mode_available
        else None,
    )


@router.post("/auth/live", response_model=LiveAuthResponse, tags=["system"])
async def unlock_live_mode(
    body: LiveAuthRequest, request: Request, services: Services
) -> LiveAuthResponse:
    """Exchange the live mode password for a short-lived token (failed attempts are throttled)."""
    settings = services.settings
    if not settings.live_mode_available or settings.live_mode_password is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Live mode is not enabled on this server")
    limiter = _limiter(request)
    client = request.client.host if request.client else "unknown"
    if limiter.blocked(client):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed attempts; try again later"
        )
    password = settings.live_mode_password.get_secret_value()
    if not password_matches(body.password, password):
        limiter.record_failure(client)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect password")
    token, expires_at = issue_token(password, ttl_seconds=settings.live_token_ttl_hours * 3600)
    return LiveAuthResponse(token=token, expires_at=expires_at)


@router.get("/graph", response_model=GraphResponse, tags=["system"])
async def graph_diagram(services: Services) -> GraphResponse:
    """Mermaid rendering of the compiled LangGraph (including the worker subgraph)."""
    return GraphResponse(mermaid=services.graph.get_graph(xray=1).draw_mermaid())


@router.post(
    "/research", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED, tags=["research"]
)
async def create_research(
    body: CreateResearchRequest, services: Services, x_live_token: LiveToken = None
) -> RunResponse:
    mode = _resolve_mode(services, body.mode, x_live_token)
    requested = body.failure_scenarios or services.settings.demo_failures  # DEMO_FAILURES default
    failures = parse_scenarios(requested) if requested else frozenset()
    if failures and not (mode == "demo" or failures <= {"failed_source", "insufficient_evidence"}):
        raise InvalidRunStateError("LLM failure scenarios can only be simulated in demo mode")
    research_id = await services.manager.create_run(
        body.question,
        auto_approve=body.auto_approve,
        failure_scenarios=failures,
        live=mode == "live",
    )
    return _to_response(await _get_run(services, research_id))


@router.get("/research", response_model=RunListResponse, tags=["research"])
async def list_research(
    services: Services,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunListResponse:
    runs, total = await services.repository.list_runs(limit=limit, offset=offset)
    return RunListResponse(items=[_to_response(r) for r in runs], total=total)


@router.get("/research/{research_id}", response_model=RunResponse, tags=["research"])
async def get_research(research_id: str, services: Services) -> RunResponse:
    return _to_response(await _get_run(services, research_id))


@router.get("/research/{research_id}/plan", response_model=PlanResponse, tags=["plan"])
async def get_plan(research_id: str, services: Services) -> PlanResponse:
    run = await _get_run(services, research_id)
    plan = await services.manager.get_plan(research_id)
    return PlanResponse(
        research_id=research_id,
        status=RunStatus(run.status),
        editable=run.status == RunStatus.AWAITING_APPROVAL.value,
        plan=plan,
    )


@router.post("/research/{research_id}/approve", response_model=RunResponse, tags=["plan"])
async def approve_plan(
    research_id: str, services: Services, x_live_token: LiveToken = None
) -> RunResponse:
    await _guard_live_run(services, research_id, x_live_token)
    await services.manager.approve(research_id)
    return _to_response(await _get_run(services, research_id))


@router.post("/research/{research_id}/edit-plan", response_model=RunResponse, tags=["plan"])
async def edit_plan(
    research_id: str, body: EditPlanRequest, services: Services, x_live_token: LiveToken = None
) -> RunResponse:
    await _guard_live_run(services, research_id, x_live_token)
    await services.manager.edit_plan(research_id, body.plan, approve=body.approve)
    return _to_response(await _get_run(services, research_id))


@router.post("/research/{research_id}/replan", response_model=RunResponse, tags=["plan"])
async def replan(
    research_id: str, body: ReplanRequest, services: Services, x_live_token: LiveToken = None
) -> RunResponse:
    """Regenerate the plan from natural-language reviewer feedback."""
    await _guard_live_run(services, research_id, x_live_token)
    await services.manager.replan(research_id, body.feedback)
    return _to_response(await _get_run(services, research_id))


@router.post("/research/{research_id}/cancel", response_model=RunResponse, tags=["research"])
async def cancel_research(research_id: str, services: Services) -> RunResponse:
    await services.manager.cancel(research_id)
    return _to_response(await _get_run(services, research_id))


@router.get("/research/{research_id}/sources", response_model=SourcesResponse, tags=["artifacts"])
async def get_sources(
    research_id: str,
    services: Services,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SourcesResponse:
    await _get_run(services, research_id)
    rows, total = await services.repository.get_sources(research_id, limit=limit, offset=offset)
    items = [
        SourceItem(
            source=Source.model_validate(r.data),
            quality=SourceQuality.model_validate(r.quality) if r.quality else None,
        )
        for r in rows
    ]
    return SourcesResponse(items=items, total=total)


@router.get("/research/{research_id}/evidence", response_model=EvidenceResponse, tags=["artifacts"])
async def get_evidence(
    research_id: str,
    services: Services,
    subquestion_id: Annotated[str | None, Query(max_length=16)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EvidenceResponse:
    await _get_run(services, research_id)
    rows, total = await services.repository.get_evidence(
        research_id, subquestion_id=subquestion_id, limit=limit, offset=offset
    )
    return EvidenceResponse(items=[Evidence.model_validate(r.data) for r in rows], total=total)


@router.get("/research/{research_id}/findings", response_model=FindingsResponse, tags=["artifacts"])
async def get_findings(research_id: str, services: Services) -> FindingsResponse:
    await _get_run(services, research_id)
    rows = await services.repository.get_findings(research_id)
    return FindingsResponse(items=[ResearchFinding.model_validate(r.data) for r in rows])


@router.get("/research/{research_id}/report", response_model=ReportResponse, tags=["artifacts"])
async def get_report(
    research_id: str,
    services: Services,
    format: Annotated[Literal["json", "markdown"], Query()] = "json",
) -> ReportResponse | Response:
    run = await _get_run(services, research_id)
    record = await services.repository.get_report(research_id)
    if record is None:
        raise InvalidRunStateError(f"Report not available yet (status: {run.status})")
    if format == "markdown":
        return Response(
            content=record.markdown,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="researchgraph-{research_id}.md"'
            },
        )
    report = FinalReport.model_validate(record.data)
    return ReportResponse(
        research_id=research_id,
        title=report.title,
        markdown=report.markdown,
        executive_summary=report.executive_summary_markdown,
        bibliography=report.bibliography,
        metrics=report.metrics,
        review_notes=report.review_notes,
        critique=Critique.model_validate(run.critique) if run.critique else None,
        quality=QualityAssessment.model_validate(run.quality) if run.quality else None,
        contradictions=[Contradiction.model_validate(c) for c in run.contradictions or []],
        generated_at=report.generated_at,
    )


@router.get("/research/{research_id}/events", response_model=EventsResponse, tags=["events"])
async def get_events(
    research_id: str,
    services: Services,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 2000,
) -> EventsResponse:
    await _get_run(services, research_id)
    rows = await services.repository.list_events(research_id, after_seq=after, limit=limit)
    return EventsResponse(
        items=[
            EventItem(
                seq=r.seq,
                type=r.type,
                node=r.node,
                message=r.message,
                data=r.data,
                timestamp=r.created_at,
            )
            for r in rows
        ]
    )


@router.get("/research/{research_id}/stream", tags=["events"])
async def stream_events(
    research_id: str,
    request: Request,
    services: Services,
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Server-Sent Events: replays persisted events, then follows the run live."""
    await _get_run(services, research_id)
    start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
    return StreamingResponse(
        event_stream(request, services, research_id, start),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
