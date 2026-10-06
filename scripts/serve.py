"""Run the API in offline demo mode from any working directory.

    python scripts/serve.py            # http://localhost:8000 (docs at /docs)

Loads ``scripts/demo.env`` (mock model, synthetic corpus) unless the variables are
already set, and keeps the SQLite database under the repository's ``data/`` folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from researchgraph.core.aio import use_selector_event_loop_on_windows

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    os.chdir(ROOT)
    for line in (ROOT / "scripts" / "demo.env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    use_selector_event_loop_on_windows()
    uvicorn.run("researchgraph.main:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
