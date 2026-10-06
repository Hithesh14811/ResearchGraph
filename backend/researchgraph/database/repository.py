"""Data access for research runs (the API's read model)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchgraph.database.models import (
    EventRecord,
    EvidenceRecord,
    FindingRecord,
    ReportRecord,
    ResearchRun,
    SourceRecord,
)
from researchgraph.schemas.common import RunStatus
from researchgraph.schemas.events import WorkflowEvent
from researchgraph.schemas.evidence import Evidence
from researchgraph.schemas.report import FinalReport, ResearchFinding
from researchgraph.schemas.sources import Source, SourceQuality


class ResearchRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    # -- helpers ---------------------------------------------------------------------
    @staticmethod
    async def _upsert(
        session: AsyncSession, model: type[Any], rows: list[dict[str, Any]], keys: Sequence[str]
    ) -> None:
        if not rows:
            return
        dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
        insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
        stmt = insert(model).values(rows)
        updates = {c: stmt.excluded[c] for c in rows[0] if c not in keys}
        await session.execute(stmt.on_conflict_do_update(index_elements=list(keys), set_=updates))

    # -- runs ----------------------------------------------------------------------------
    async def create_run(
        self,
        *,
        research_id: str,
        question: str,
        auto_approve: bool,
        failure_scenarios: Iterable[str],
        mode: str = "demo",
    ) -> ResearchRun:
        run = ResearchRun(
            id=research_id,
            question=question,
            status=RunStatus.PENDING.value,
            auto_approve=auto_approve,
            mode=mode,
            failure_scenarios=sorted(failure_scenarios),
            metrics={},
            usage={},
            errors=[],
            contradictions=[],
        )
        async with self._sessions() as session, session.begin():
            session.add(run)
        return run

    async def get_run(self, research_id: str) -> ResearchRun | None:
        async with self._sessions() as session:
            return await session.get(ResearchRun, research_id)

    async def list_runs(self, *, limit: int, offset: int) -> tuple[list[ResearchRun], int]:
        async with self._sessions() as session:
            total = await session.scalar(select(func.count()).select_from(ResearchRun)) or 0
            rows = await session.scalars(
                select(ResearchRun)
                .order_by(ResearchRun.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(rows), int(total)

    async def runs_with_status(self, statuses: Iterable[RunStatus]) -> list[ResearchRun]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(ResearchRun).where(ResearchRun.status.in_([s.value for s in statuses]))
            )
            return list(rows)

    async def update_run(self, research_id: str, **fields: Any) -> None:
        if not fields:
            return
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(ResearchRun).where(ResearchRun.id == research_id).values(**fields)
            )

    # -- artifacts ---------------------------------------------------------------------------
    async def upsert_sources(
        self, research_id: str, sources: Mapping[str, Source], quality: Mapping[str, SourceQuality]
    ) -> None:
        rows = []
        for source_id, source in sources.items():
            q = quality.get(source_id)
            rows.append(
                {
                    "run_id": research_id,
                    "source_id": source_id,
                    "url": source.url,
                    "title": source.title,
                    "status": source.status,
                    "source_type": q.source_type.value
                    if q
                    else (source.type_hint.value if source.type_hint else None),
                    "quality_overall": q.overall if q else None,
                    "quality_tier": q.tier.value if q else None,
                    "data": source.model_dump(mode="json"),
                    "quality": q.model_dump(mode="json") if q else None,
                }
            )
        async with self._sessions() as session, session.begin():
            await self._upsert(session, SourceRecord, rows, ("run_id", "source_id"))

    async def upsert_evidence(self, research_id: str, evidence: Mapping[str, Evidence]) -> None:
        rows = [
            {
                "run_id": research_id,
                "evidence_id": e.id,
                "subquestion_id": e.subquestion_id,
                "source_id": e.source_id,
                "claim": e.claim,
                "evidence_type": e.evidence_type.value,
                "confidence": e.confidence,
                "relevance": e.relevance,
                "data": e.model_dump(mode="json"),
            }
            for e in evidence.values()
        ]
        async with self._sessions() as session, session.begin():
            await self._upsert(session, EvidenceRecord, rows, ("run_id", "evidence_id"))

    async def replace_findings(self, research_id: str, findings: list[ResearchFinding]) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(FindingRecord).where(FindingRecord.run_id == research_id))
            session.add_all(
                FindingRecord(
                    run_id=research_id,
                    finding_id=f.id,
                    subquestion_id=f.subquestion_id,
                    statement=f.statement,
                    confidence=f.confidence,
                    data=f.model_dump(mode="json"),
                )
                for f in findings
            )

    async def save_report(self, research_id: str, report: FinalReport) -> None:
        row = {
            "run_id": research_id,
            "title": report.title,
            "markdown": report.markdown,
            "executive_summary": report.executive_summary_markdown,
            "data": report.model_dump(mode="json"),
            "created_at": report.generated_at,
        }
        async with self._sessions() as session, session.begin():
            await self._upsert(session, ReportRecord, [row], ("run_id",))

    # -- queries ----------------------------------------------------------------------------------
    async def get_sources(
        self, research_id: str, *, limit: int, offset: int
    ) -> tuple[list[SourceRecord], int]:
        async with self._sessions() as session:
            total = (
                await session.scalar(select(func.count()).where(SourceRecord.run_id == research_id))
                or 0
            )
            rows = await session.scalars(
                select(SourceRecord)
                .where(SourceRecord.run_id == research_id)
                .order_by(SourceRecord.quality_overall.desc().nulls_last(), SourceRecord.id)
                .limit(limit)
                .offset(offset)
            )
            return list(rows), int(total)

    async def get_evidence(
        self, research_id: str, *, subquestion_id: str | None, limit: int, offset: int
    ) -> tuple[list[EvidenceRecord], int]:
        conditions = [EvidenceRecord.run_id == research_id]
        if subquestion_id:
            conditions.append(EvidenceRecord.subquestion_id == subquestion_id)
        async with self._sessions() as session:
            total = await session.scalar(select(func.count()).where(*conditions)) or 0
            rows = await session.scalars(
                select(EvidenceRecord)
                .where(*conditions)
                .order_by(
                    EvidenceRecord.subquestion_id,
                    EvidenceRecord.relevance.desc(),
                    EvidenceRecord.id,
                )
                .limit(limit)
                .offset(offset)
            )
            return list(rows), int(total)

    async def get_findings(self, research_id: str) -> list[FindingRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(FindingRecord)
                .where(FindingRecord.run_id == research_id)
                .order_by(FindingRecord.id)
            )
            return list(rows)

    async def get_report(self, research_id: str) -> ReportRecord | None:
        async with self._sessions() as session:
            return await session.get(ReportRecord, research_id)

    # -- events ---------------------------------------------------------------------------------
    async def last_event_seq(self, research_id: str) -> int:
        async with self._sessions() as session:
            return int(
                await session.scalar(
                    select(func.max(EventRecord.seq)).where(EventRecord.run_id == research_id)
                )
                or 0
            )

    async def append_events(self, events: list[WorkflowEvent]) -> None:
        if not events:
            return
        async with self._sessions() as session, session.begin():
            session.add_all(
                EventRecord(
                    run_id=e.research_id,
                    seq=e.seq,
                    type=e.type.value,
                    node=e.node,
                    message=e.message,
                    data=e.data,
                    created_at=e.timestamp,
                )
                for e in events
            )

    async def list_events(
        self, research_id: str, *, after_seq: int = 0, limit: int = 1000
    ) -> list[EventRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(EventRecord)
                .where(EventRecord.run_id == research_id, EventRecord.seq > after_seq)
                .order_by(EventRecord.seq)
                .limit(limit)
            )
            return list(rows)
