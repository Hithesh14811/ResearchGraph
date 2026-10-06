"""PostgreSQL + pgvector integration (runs in CI's Postgres service; skipped without a database).

Set TEST_DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db to run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from researchgraph.demo.scenarios import SAMPLE_QUESTIONS
from researchgraph.runtime.bootstrap import open_services
from researchgraph.schemas.common import RunStatus
from tests.conftest import make_settings, wait_for_status

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set"),
]


async def test_full_run_on_postgres_with_pgvector(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, database_url=DATABASE_URL, vector_store="pgvector")
    async with open_services(settings, recover=False) as services:
        research_id = await services.manager.create_run(
            SAMPLE_QUESTIONS[0].question, auto_approve=True
        )
        run = await wait_for_status(services, research_id, RunStatus.COMPLETED, within=120)
        assert run.metrics["evidence_items"] > 0
        report = await services.repository.get_report(research_id)
        assert report is not None and "## References" in report.markdown
