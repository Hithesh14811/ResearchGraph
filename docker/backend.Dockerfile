# syntax=docker/dockerfile:1.7
# ResearchGraph API (FastAPI + LangGraph). Build context: repository root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so source edits don't invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY backend/researchgraph/__init__.py backend/researchgraph/__init__.py
RUN pip install --upgrade pip && pip install ".[all]" && pip uninstall -y researchgraph

COPY backend ./backend
COPY scripts ./scripts
COPY evaluation ./evaluation
RUN pip install --no-deps ".[all]"

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown -R app:app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "researchgraph.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
