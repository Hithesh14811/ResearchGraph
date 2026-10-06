"""Event-loop setup shared by the command-line entry points."""

from __future__ import annotations

import asyncio
import sys


def use_selector_event_loop_on_windows() -> None:
    """psycopg's async driver (PostgreSQL) cannot run on Windows' default Proactor loop.

    Call before the event loop starts. A no-op on every other platform.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
