"""LangGraph checkpointer factory (SQLite locally, PostgreSQL in Docker/production)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from researchgraph.config.settings import Settings
from researchgraph.graph.serde import state_serializer


def resolve_checkpoint_url(settings: Settings) -> str:
    """Default: a sibling ``checkpoints.sqlite`` next to the SQLite DB, or the same Postgres DB."""
    if settings.checkpoint_url:
        return settings.checkpoint_url
    url = settings.database_url
    if url.startswith("postgresql"):
        return url.replace("postgresql+psycopg://", "postgresql://").replace(
            "postgresql+asyncpg://", "postgresql://"
        )
    if url.startswith("sqlite") and ":///" in url:
        db_path = url.split(":///", 1)[1]
        if db_path in ("", ":memory:"):
            return "memory"
        return str(Path(db_path).with_name("checkpoints.sqlite"))
    return "memory"


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """Yield a ready-to-use checkpointer with the strict, allowlisted serializer."""
    url = resolve_checkpoint_url(settings)
    serde = state_serializer()
    if url == "memory":
        yield InMemorySaver(serde=serde)
        return
    if url.startswith("postgres"):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        async with AsyncConnectionPool(
            conninfo=url,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        ) as pool:
            postgres_saver = AsyncPostgresSaver(pool, serde=serde)  # type: ignore[arg-type]
            await postgres_saver.setup()
            yield postgres_saver
        return

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    await asyncio.to_thread(_ensure_parent_dir, url)
    async with AsyncSqliteSaver.from_conn_string(url) as saver:
        saver.serde = serde
        await saver.setup()
        yield saver
