"""Async SQLAlchemy engine/session factory for SQLite (local) and PostgreSQL (Docker/prod)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, event, inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from researchgraph.database.models import Base

logger = logging.getLogger(__name__)


def _ensure_sqlite_dir(url: str) -> None:
    if url.startswith("sqlite") and ":///" in url:
        path = url.split(":///", 1)[1]
        if path and path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def create_engine(url: str) -> AsyncEngine:
    _ensure_sqlite_dir(url)
    if url.startswith("sqlite"):
        engine = create_async_engine(url, connect_args={"timeout": 30})

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_async_engine(url, pool_size=10, max_overflow=10, pool_pre_ping=True)


# Columns added after the first release, created in place on existing databases.
_ADDED_COLUMNS = {"research_runs": {"mode": "VARCHAR(8) NOT NULL DEFAULT 'demo'"}}


def _add_missing_columns(conn: Connection) -> None:
    inspector = inspect(conn)
    for table, columns in _ADDED_COLUMNS.items():
        existing = {column["name"] for column in inspector.get_columns(table)}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


async def init_models(engine: AsyncEngine) -> None:
    """Create tables if missing (a migration tool would replace this in a larger system)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
