<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
  <img src="docs/images/logo.svg" width="52" height="52" alt="ResearchGraph logo">
</picture>

# ResearchGraph

**An evidence-driven research agent built on LangGraph.**
Every sentence in its reports is tied to a quote that was verified against a stored source.

**[Live demo](https://researchgraph-z8r1.onrender.com)** · [API docs](https://researchgraph-api-n4ij.onrender.com/docs) · [Example report](docs/examples/live-report-deepseek.md)

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph 1.2](https://img.shields.io/badge/LangGraph-1.2-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Tests: 122](https://img.shields.io/badge/tests-122-success)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

</div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/run-workflow-dark.png">
  <img src="docs/images/run-workflow.png" alt="ResearchGraph run view: live LangGraph workflow, activity log and metrics">
</picture>
<sub>A demo run with all five failure scenarios injected. The evidence gate looped back once, the critic sent it back
for more research once, citation repair removed one unsupported sentence, and every injected fault was recovered from.</sub>

## What it is

You ask ResearchGraph a question that needs evidence, such as *"Compare RAG, fine-tuning and long-context prompting
for domain-specific QA"*. It then:

- plans the research and waits for you to approve the plan
- researches each subquestion in parallel, using real academic and web search
- extracts claims with exact quotes and checks each quote against the source text
- scores every source for credibility
- looks for contradictions, and has an independent critic review the findings
- goes back for more research when the evidence is thin
- writes a report where every sentence cites verified evidence

The model never writes citations itself. It only refers to evidence IDs, and the citation numbers, URLs and
bibliography are generated in code from stored provenance. A sentence that can't be backed by its cited evidence is
repaired or removed before the report is published.

The hosted demo runs offline (mock model, synthetic corpus), so it's free to try. It's on Render's free tier, so if
it hasn't been used in a while, the first request can take up to a minute while the server wakes up.

<details>
<summary><b>More screenshots</b> (dashboard, plan review, evidence, sources, critique, report)</summary>

![Dashboard: new research, how a run works, run history](docs/images/dashboard.png)
![Plan review: the run is paused at a LangGraph interrupt](docs/images/plan-review.png)
![Evidence tab: every claim with its verbatim, verified quote](docs/images/evidence.png)
![Sources tab: explainable source-quality scores](docs/images/sources.png)
![Critique tab: final critique, contradictions and the quality gate decision](docs/images/critique.png)
![Full report with clickable citations](docs/images/report.png)

</details>

## How it works

```mermaid
flowchart TD
    S([START]) --> intake
    intake -- invalid --> E([END])
    intake --> planner --> plan_review
    plan_review -- "interrupt(): approve / edit" --> query_generation
    plan_review -- "replan with feedback" --> planner
    plan_review -- cancel --> E
    query_generation == "Send() × N subquestions" ==> W
    subgraph W ["research_worker subgraph (one per subquestion, run in parallel)"]
        search_agent <--> execute_tools
        search_agent --> source_processing --> passage_retrieval --> evidence_extraction
    end
    W --> source_evaluation
    source_evaluation -- "not enough evidence & budget left" --> query_generation
    source_evaluation -- "enough (or out of budget)" --> contradiction_detection
    contradiction_detection --> synthesis --> critic --> quality_gate
    quality_gate -- "research_more" --> query_generation
    quality_gate -- "proceed / proceed with limitations" --> report_writer
    report_writer --> citation_verification
    citation_verification -- "issues & repairs left" --> citation_repair --> citation_verification
    citation_verification -- verified --> final_review --> E
```

1. **Plan.** The planner breaks the question into 3 to 6 subquestions, each with search queries. The graph then
   pauses at a LangGraph `interrupt()`, where you can approve the plan, edit it, ask for a new one, or cancel.
2. **Research.** One worker subgraph per subquestion is fanned out with `Send()`, and the workers run in
   parallel. Each worker is a small tool-using agent that chooses between web search (Tavily) and academic search
   (arXiv, Semantic Scholar). It then fetches, parses (PDF, HTML, Markdown), chunks and indexes the sources, and
   retrieves the most relevant passages.
3. **Evidence.** Claims are extracted together with exact quotes. Any quote that isn't found word for word in the
   source is discarded.
4. **Source scoring.** Every source gets an explainable score built from authority, recency, relevance,
   primary-vs-secondary, methodology and citations. Search rank is never treated as credibility.
5. **Evidence gate.** If a subquestion is thinly supported and budget remains, the graph loops back for targeted
   research.
6. **Analysis and critique.** Conflicting evidence is detected and adjudicated, and findings are synthesised with
   calibrated confidence. An independent critic (rule-based checks plus an LLM review) then looks for overclaiming,
   single-source conclusions and gaps.
7. **Quality gate.** A deterministic score decides whether to publish, research more, or publish with the
   limitations stated explicitly.
8. **Report and verification.** The writer drafts sentences tagged with evidence IDs. Each sentence is then
   checked: it must have a citation, the cited evidence must cover its key terms, and every number in it must appear
   in that evidence. Failing sentences are re-attributed or removed, and removals are listed in the report.

Eight steps use an LLM (planner, query writer, research agent, extractor, contradiction judge, synthesizer, critic,
writer). Source scoring, the gates, citation numbering, verification and rendering are plain, unit-tested Python.

Deeper documentation:
- [`docs/architecture.md`](docs/architecture.md): state model, reducers, retrieval pipeline, database schema,
  routing
- [`docs/design-decisions.md`](docs/design-decisions.md): the reasoning behind each major choice

### Engineering highlights

- **Typed state with safe parallel merges.** Parallel workers write through commutative, idempotent reducers, and
  IDs are content-addressed, so two workers that find the same source converge on one entry.
- **Durable runs.** State is checkpointed after every step. On shutdown, running graphs drain to a checkpoint and
  resume on startup. A plan approval still works after a server restart.
- **Live progress.** Workflow events stream to the dashboard over Server-Sent Events, with replay via
  `Last-Event-ID`.
- **Graceful failure.** LLM calls retry with backoff and fall back to deterministic defaults. A failing worker only
  loses its own results. Every loop has a budget, so termination is guaranteed.
- **Built-in fault injection.** Demo runs can simulate a failed source, an LLM timeout, invalid model output,
  insufficient evidence and a critic rejection, to show recovery end to end.

### Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 1.2: StateGraph, `Send()`, `interrupt()`, subgraphs, `ToolNode`, Postgres/SQLite checkpointers |
| Models | Any LangChain `init_chat_model` provider, or any OpenAI-compatible endpoint via `LLM_BASE_URL` |
| API | FastAPI, Server-Sent Events, Pydantic v2 |
| Storage | PostgreSQL + pgvector (SQLite + in-memory vectors for local development), SQLAlchemy 2.0 async |
| Frontend | React 19, TypeScript, Vite, Tailwind |
| Ops | Docker Compose, GitHub Actions, optional LangSmith tracing |

## Demo mode and live mode

The hosted dashboard has a **Demo / Live** switch in the header.

- **Demo mode** is the default and is open to everyone. It runs the full LangGraph workflow with a deterministic
  mock model over a small synthetic corpus, so it's free, fast and repeatable. All demo sources are clearly
  labelled `[Synthetic]`.
- **Live mode** uses a real LLM (DeepSeek V4.1 Flash) with real arXiv and Semantic Scholar search. A live run takes
  4 to 10 minutes and uses about 146k tokens, which costs real API credits.

**Live mode is password-protected** so the public demo can't run up a bill or be used as a free LLM endpoint. The
password is exchanged for a short-lived signed token. That token is required to start a live run and to approve,
edit, replan or cancel one, since each of those actions spends credits or affects someone else's run. Failed
password attempts are rate-limited. Access is available on request. In the meantime,
[an unedited live report](docs/examples/live-report-deepseek.md) and
[the live evaluation results](docs/examples/evaluation-live.md) show what live mode produces.

Running locally with your own API key has no password; see [Running it](#running-it).

## Results

The evaluation harness (`python -m evaluation.run`) runs a set of research questions end to end and checks
behaviour rather than answer keys. Each run is checked for whether everything is cited, whether gaps are admitted,
and whether the expected aspects are covered. One question is a deliberately off-topic **negative control**, where
the correct outcome is no findings at all.

### Live mode

DeepSeek V4.1 Flash, real search and real PDFs, plan auto-approved, no prompt tuning
([full results](docs/examples/evaluation-live.md)):

| Question | citation coverage | unsupported rate | first-pass precision | source quality | completeness | latency (s) | checks |
|---|---|---|---|---|---|---|---|
| rag-vs-finetuning | 1.000 | 0.061 | 0.892 | 0.795 | 1.000 | 593 | 9/10 |
| small-vs-large-models | 1.000 | 0.000 | 1.000 | 0.798 | 1.000 | 249 | 9/9 |
| rag-hallucinations | 1.000 | 0.043 | 0.957 | 0.803 | 0.667 | 308 | 8/9 |
| quantization-edge | 1.000 | 0.091 | 0.886 | 0.767 | 1.000 | 315 | 9/9 |
| **mean** | 1.000 | 0.049 | 0.934 | 0.791 | 0.917 | 366 | 95% |

Two checks failed, and they are reported as-is:

- `rag-vs-finetuning` found no contradiction where the dataset expected one. The papers live search returned don't
  disagree the way the synthetic corpus does.
- `rag-hallucinations` covered 2 of 6 expected aspects. The retrieved papers don't discuss reranking, abstention or
  entailment checking, so the report correctly doesn't either.

Both expectations were written against the demo corpus, which shows that live mode needs its own labelled
dataset. LLM-as-judge scores were 4.0 to 4.6 out of 5, but the judge was the same model that wrote the reports, so
treat them as weak evidence.

The first live run also caught a real scoring bug: arXiv survey papers were being scored as primary research. They
are now treated as secondary sources, and a regression test covers it.

### Demo mode

([full results](docs/examples/evaluation-demo.md))

| Question | citation coverage | unsupported rate | first-pass precision | source quality | completeness | latency (s) | checks |
|---|---|---|---|---|---|---|---|
| rag-vs-finetuning | 1.000 | 0.038 | 0.962 | 0.662 | 1.000 | 1.5 | 10/10 |
| small-vs-large-models | 1.000 | 0.042 | 0.958 | 0.643 | 1.000 | 1.2 | 9/9 |
| rag-hallucinations | 1.000 | 0.048 | 0.955 | 0.667 | 1.000 | 1.3 | 9/9 |
| quantization-edge | 1.000 | 0.043 | 0.957 | 0.639 | 1.000 | 2.0 | 9/9 |
| negative-control-off-topic | 1.000 | 1.000\* | 0.000 | — | 0.000\* | 2.8 | 8/8 |

\* The negative control is meant to produce nothing. Its single generic sentence was removed by verification, and
no off-topic findings were made.

Demo results validate the machinery (grounding, verification, routing, termination), not research quality. The live
numbers are the measure of quality.

**Metric definitions:**
- **Citation coverage:** published sentences with at least one valid citation.
- **Unsupported rate:** written sentences that verification removed.
- **First-pass precision:** sentences that passed before any repair.
- **Source quality:** mean score of the cited sources.
- **Completeness:** average of subquestion coverage and expected-aspect coverage.

## Running it

### Offline demo (no API keys)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m scripts.demo --simulate-failures all
```

This runs the full workflow in the terminal with every failure scenario injected. The report, sources, evidence
and event log are written to `demo_output/<research_id>/`.

Other options:

```bash
python -m scripts.demo --question 3                  # one of the three sample questions
python -m scripts.demo --all                         # all three
python -m scripts.demo --interactive                 # approve, revise or cancel the plan yourself
python -m scripts.demo --custom "Your own question"  # uses the fallback planner
```

### Dashboard

```bash
python scripts/serve.py                            # API on :8000, offline demo config
cd frontend && npm install && npm run dev          # dashboard on http://localhost:5173
```

### With a real model

```bash
pip install -e ".[all,dev]"      # OpenAI/Anthropic integrations and pgvector
cp .env.example .env
```

DeepSeek, the configuration used for the live results:

```bash
LLM_PROVIDER=openai
MODEL_NAME=deepseek-flash
LLM_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=...
SEARCH_PROVIDER=auto              # arXiv + Semantic Scholar, plus Tavily if TAVILY_API_KEY is set
```

Anthropic:

```bash
LLM_PROVIDER=anthropic
MODEL_NAME=claude-sonnet-5-5
FAST_MODEL_NAME=claude-haiku-5-5  # optional cheaper model for queries and extraction
ANTHROPIC_API_KEY=...
SEARCH_PROVIDER=auto
```

Then start the API with `uvicorn researchgraph.main:create_app --factory --port 8000`. On Windows with PostgreSQL,
use `python -m researchgraph.main` instead, because psycopg needs the selector event loop there.

Main settings (the full, commented list is in [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `MODEL_NAME` | `mock` | Any `init_chat_model` provider and model |
| `FAST_MODEL_NAME` | none | Optional cheaper model for high-volume steps |
| `LLM_BASE_URL` | none | Any OpenAI-compatible endpoint (DeepSeek, vLLM, Together, …) |
| `SEARCH_PROVIDER` | `mock` | `auto`, `tavily`, `arxiv` or `semantic_scholar` |
| `EMBEDDING_PROVIDER` | `hashing` | `openai` for semantic embeddings (the default is offline and lexical) |
| `DATABASE_URL` | SQLite file | A PostgreSQL URL enables pgvector automatically |
| `MAX_RESEARCH_ITERATIONS` / `MAX_TOOL_CALLS` | 3 / 60 | Research budgets |
| `LLM_INPUT_COST_PER_MTOK` / `LLM_OUTPUT_COST_PER_MTOK` | none | Cost estimates in the UI |
| `API_TOKEN` | none | Require a bearer token on every request |

To host a public demo like the one above:

| Variable | Purpose |
|---|---|
| `PUBLIC_DEMO=true` | Serve demo mode by default |
| `LIVE_MODE_PASSWORD` | Unlock live mode; without it, live runs are impossible |
| `SEED_DEMO_RUN=true` | Start one sample run on an empty database so visitors have a finished run to explore |
| `TRUSTED_PROXY_HOPS=1` | Read the real client address from the load balancer's `X-Forwarded-For` for rate limiting |

### Docker

```bash
cp .env.example .env    # optional
docker compose up --build
```

This starts PostgreSQL + pgvector, the API (http://localhost:8000/docs) and the dashboard (http://localhost:8080).
In-flight runs get 30 seconds to drain to a checkpoint on `docker compose stop`.

## API

Interactive documentation is served at `/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/research` | Start a run (returns 202) |
| `GET` | `/research` · `/research/{id}` | Run history · status, progress, metrics, token usage |
| `GET` | `/research/{id}/plan` | Current plan |
| `POST` | `/research/{id}/approve` · `/edit-plan` · `/replan` · `/cancel` | Plan review decisions |
| `GET` | `/research/{id}/sources` · `/evidence` · `/findings` | Artifacts (paginated) |
| `GET` | `/research/{id}/report` | Report (`?format=markdown` to download) |
| `GET` | `/research/{id}/stream` | Server-Sent Events with replay |
| `POST` | `/auth/live` | Exchange the live mode password for a token |
| `GET` | `/graph` · `/health` | Mermaid diagram of the compiled graph · configuration and active runs |

## Security

The agent reads arbitrary documents from the internet and passes them to an LLM, so the boundaries are explicit:

- **SSRF protection.** Only public http(s) URLs are fetched. DNS is resolved, private, loopback and
  cloud-metadata addresses are blocked, and every redirect hop is re-validated.
- **Prompt-injection boundary.** Retrieved text is always marked as untrusted data. The model never sees or emits
  URLs, cannot fetch anything itself, and has no shell or code-execution tool. The calculator tool is an AST
  whitelist, not `eval`.
- **Resource limits.** Size caps on documents, PDFs and request bodies (including chunked uploads); iteration,
  tool and repair budgets; and a cap on concurrent runs.
- **Safe checkpoints.** Checkpoints are deserialised against an explicit type allowlist, so write access to the
  checkpoint database does not lead to code execution.
- **Live mode gating** as described [above](#demo-mode-and-live-mode).

## Tests

```bash
pytest                                     # offline: mock LLM, synthetic corpus, mocked HTTP
TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/db pytest -m postgres
ruff check backend scripts evaluation
mypy
cd frontend && npm run build               # strict TypeScript + production build
```

There are 122 tests (84 unit, 38 integration), and none of them call a real API. The integration suite covers:

- the full workflow for every sample question, and every failure scenario
- plan approval surviving a simulated restart, and drain-on-shutdown with resume-on-startup
- the HTTP API end to end, including SSE replay, the live-mode gate and its rate limiting, and concurrent approvals
  (exactly one wins)
- a full run on real PostgreSQL + pgvector

CI (GitHub Actions) runs:

- lint, type checks and tests
- the Postgres suite against a pgvector service
- the demo with every failure injected
- the evaluation harness
- the frontend build
- both Docker image builds

## Project layout

```text
backend/researchgraph/
├── agents/       the eight LLM roles: planner, query writer, researcher, extractor, judge, synthesizer, critic, writer
├── graph/        state and reducers, nodes, routing, worker subgraph, instrumentation
├── services/     deterministic logic: source scoring, coverage, quality gate, citations, rendering
├── retrieval/    loading, cleaning, chunking, embeddings, vector stores, reranking
├── tools/        search providers, SSRF-safe fetcher, document parsing, calculator
├── runtime/      background runs, event streaming, drain and resume
├── api/          FastAPI routes, SSE, auth
├── database/     SQLAlchemy models, repository, checkpointer
├── llm/          provider factory, structured output, mock model, token tracking
└── demo/         synthetic corpus, mock responder, fault injection
frontend/         React dashboard
evaluation/       datasets and the evaluation harness
docs/             architecture, design decisions, example reports
```

## Limitations and next steps

- **Lexical citation verification.** Verification checks term coverage and numbers, not entailment, so an on-topic
  sentence that subtly misstates its source can pass. The next step is an NLI check on top.
- **Limited live evaluation.** Live mode has been measured with one model on four questions. A larger,
  human-labelled live dataset and runs with Claude and GPT are next.
- **Slow live runs.** Live runs take 4 to 10 minutes, mostly because arXiv asks for about 3 seconds between
  requests.
- **Lexical default embeddings.** The default embeddings need no setup but are lexical.
  `EMBEDDING_PROVIDER=openai` enables semantic search.
- **No job queue.** Runs execute inside the API process. Horizontal scaling would need a job queue and per-user
  quotas.
- **Text-only PDF parsing.** There is no OCR, and tables and figures aren't extracted.
- **DNS rebinding window.** The SSRF check resolves DNS before connecting, which leaves a small rebinding window.
  Pinning the resolved address in the HTTP transport would close it.

## License

MIT. See [LICENSE](LICENSE).
