<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
  <img src="docs/images/logo.svg" width="52" height="52" alt="ResearchGraph logo">
</picture>

# ResearchGraph

**An autonomous, evidence-driven research agent built on LangGraph.**

Ask a complex question. ResearchGraph plans the research, investigates subquestions in parallel, scores its sources,
extracts quote-verified evidence, critiques its own analysis, loops back when evidence is weak, and publishes a report in
which **every sentence is tied to a stored, verifiable source.**

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph 1.2](https://img.shields.io/badge/LangGraph-1.2-1C3C3C)
![LangChain 1.x](https://img.shields.io/badge/LangChain-1.x-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Tests: 112](https://img.shields.io/badge/tests-112-success)
![mypy](https://img.shields.io/badge/types-mypy%20clean-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

</div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/run-workflow-dark.png">
  <img src="docs/images/run-workflow.png" alt="ResearchGraph run view: live LangGraph workflow, activity log and metrics">
</picture>
<sub>A run with all five failure scenarios injected: the evidence gate looped back (<i>insufficient ×1</i>), the critic
sent the graph back for targeted research (<i>research more ×1</i>), a citation was repaired, and every simulated fault
was recovered from. Demo mode — synthetic corpus, deterministic mock model.</sub>

<details>
<summary><b>More screenshots</b> — dashboard, evidence, critique and quality gate</summary>

![Dashboard: new research, how a run works, run history](docs/images/dashboard.png)
![Evidence tab: every claim with its verbatim, verified quote](docs/images/evidence.png)
![Critique tab: final critique, contradictions and the quality gate decision](docs/images/critique.png)

</details>

---

## At a glance

**Run it in a minute — no API keys, no network:**

```bash
pip install -e ".[dev]"
python -m scripts.demo --simulate-failures all   # full run in the terminal, with injected failures
```

Dashboard: `python scripts/serve.py`, then `cd frontend && npm install && npm run dev` → http://localhost:5173.

**What is worth a look**

- **A real graph, not a prompt chain.** Typed state with commutative reducers, `Send()` fan-out to parallel worker
  subgraphs, three conditional loops (evidence gate, critic-driven research, citation repair) and an `interrupt()` for
  human plan approval that survives a server restart ([§4](#4-the-langgraph-workflow), [§13](#13-human-in-the-loop)).
- **Citations that cannot be invented.** The model only references evidence IDs; quotes are verified verbatim against
  the source; every published sentence is re-checked (coverage, key terms, every number) and repaired or removed
  ([§8](#8-citations-and-provenance)).
- **Deterministic where it should be.** Source scoring, coverage, the quality gate, numbering and rendering are plain,
  unit-tested Python; only eight judgement steps call a model ([§26](#26-engineering-tradeoffs)).
- **Built to fail safely.** Checkpointed durability, drain-on-shutdown and resume-on-startup, SSE replay with
  `Last-Event-ID`, SSRF-safe fetching, bounded budgets, strict checkpoint deserialisation ([§14](#14-persistence-and-durability),
  [§17](#17-security), [§27](#27-failure-modes)).
- **Measured, with honest limits.** Live evaluation with a real LLM and real papers (100 % citation coverage, 95 % of
  behavioural checks, failures reported as-is); 112 tests including a real PostgreSQL + pgvector run
  ([§15](#15-evaluation), [§28](#28-limitations)).

## Contents

1. [What it does](#1-what-it-does)
2. [Why a single agent is not enough](#2-why-a-single-agent-is-not-enough)
3. [Architecture](#3-architecture)
4. [The LangGraph workflow](#4-the-langgraph-workflow)
5. [State model](#5-state-model)
6. [Tool architecture](#6-tool-architecture)
7. [RAG architecture](#7-rag-architecture)
8. [Citations and provenance](#8-citations-and-provenance)
9. [Source quality scoring](#9-source-quality-scoring)
10. [Critic and reflection loop](#10-critic-and-reflection-loop)
11. [Conditional routing](#11-conditional-routing)
12. [Parallel research](#12-parallel-research)
13. [Human-in-the-loop](#13-human-in-the-loop)
14. [Persistence and durability](#14-persistence-and-durability)
15. [Evaluation](#15-evaluation)
16. [Observability](#16-observability)
17. [Security](#17-security)
18. [Quick start (offline demo)](#18-quick-start-offline-demo)
19. [Installation and configuration](#19-installation-and-configuration)
20. [Running locally](#20-running-locally)
21. [Running with Docker](#21-running-with-docker)
22. [API](#22-api)
23. [Example research question and output](#23-example-research-question-and-output)
24. [Testing](#24-testing)
25. [Project structure](#25-project-structure)
26. [Engineering tradeoffs](#26-engineering-tradeoffs)
27. [Failure modes](#27-failure-modes)
28. [Limitations](#28-limitations)
29. [Future improvements](#29-future-improvements)

Deeper documentation: [`docs/architecture.md`](docs/architecture.md) · [`docs/design-decisions.md`](docs/design-decisions.md)

---

## 1. What it does

Given a question such as *"Compare RAG, fine-tuning and long-context prompting for domain-specific QA"*, ResearchGraph:

1. **Validates and normalises** the question, then **plans**: an objective, scope, 3–6 independent subquestions,
   information requirements, search queries, a source strategy and stopping criteria.
2. **Pauses for human approval** of the plan (approve / edit / request changes / cancel) using LangGraph `interrupt()`.
3. **Researches subquestions in parallel** — one worker subgraph per subquestion, fanned out with `Send()`. Each
   worker is a bounded tool-using agent that chooses between academic and web search.
4. **Retrieves and processes sources** safely (SSRF-validated, size-limited), parses PDF/HTML/Markdown/text, cleans,
   chunks, embeds and indexes them, then retrieves passages with hybrid reranking.
5. **Extracts structured evidence** whose quotes are **verified verbatim** against the source text.
6. **Scores every source** (authority, recency, relevance, primary-ness, methodology, citations) — search rank is
   never trusted as quality.
7. **Checks evidence sufficiency** and **loops back** for targeted research when a subquestion is thinly supported.
8. **Detects contradictions**, **synthesises** calibrated findings, and runs an **independent critic** (rules + LLM).
9. A deterministic **quality gate** decides: publish, research more, or publish with explicit limitations.
10. **Writes the report** with sentence-level evidence IDs, **verifies every citation**, repairs or removes
    unsupported sentences, and renders an executive summary, full report and bibliography.
11. **Persists** everything (PostgreSQL/SQLite), **streams** workflow events to a React dashboard over SSE, and
    **survives server restarts** mid-run.

Everything runs **offline with no API keys** in demo mode (deterministic mock model + clearly-labelled synthetic
corpus), and against real providers (OpenAI, Anthropic, Gemini, Ollama, … via `init_chat_model`; Tavily, arXiv,
Semantic Scholar) by configuration only.

## 2. Why a single agent is not enough

A "prompt → LLM → answer" pipeline, or a single ReAct agent with a search tool, fails at evidence-driven research in
predictable ways:

| Failure of a single agent | How ResearchGraph addresses it |
|---|---|
| Fabricated or mismatched citations | The model only ever attaches **evidence IDs**; citation numbers, URLs and the bibliography are generated deterministically from stored provenance, and every sentence is verified. |
| Paraphrased "quotes" that are not in the source | Every extracted quote is checked against the source chunk; unverifiable evidence is discarded. |
| Treats search rank as credibility | Explainable six-factor source scoring; coverage and the quality gate count only *credible and relevant* evidence. |
| No notion of "enough evidence" | An evidence gate and a quality gate route back to targeted research, bounded by iteration and tool budgets. |
| Grades its own homework | An independent critic (deterministic rules + a separate LLM review) feeds a deterministic gate. |
| Sequential, slow exploration | Subquestions are researched concurrently by parallel worker subgraphs. |
| Context window overflow | Retrieval narrows each worker to the most relevant passages; evidence is selected per subquestion. |
| Fragile JSON | Provider-native structured outputs (JSON schema or tool calling), Pydantic validation, and bounded *repair* re-prompts. |
| Lost work on crash; no human control | Checkpointed state, `interrupt()`-based plan approval, drain-on-shutdown and resume-on-startup. |

The architecture is complex where the problem requires it and simple elsewhere: only the eight steps that need
judgement call an LLM; scoring, gating, numbering, verification and rendering are plain, testable Python.

## 3. Architecture

```mermaid
flowchart LR
    UI["React dashboard<br/>(Vite · TypeScript · Tailwind)"] -- "REST + SSE" --> API["FastAPI"]
    API --> RM["RunManager<br/>background runs · drain/resume"]
    RM -- "astream(updates, custom)" --> G["LangGraph StateGraph<br/>+ research-worker subgraph"]
    G <--> CP[("Checkpointer<br/>Postgres / SQLite")]
    RM -- "projection (same reducers)" --> DB[("Read model<br/>runs · sources · evidence · findings · reports · events")]
    G --> AG["Agents (LLM)<br/>planner · query writer · research agent · extractor<br/>contradiction judge · synthesizer · critic · writer"]
    AG --> LLM[["Chat model<br/>init_chat_model / mock"]]
    G --> TL["LLM tools<br/>web_search · academic_search · calculator"]
    G --> PL["Pipeline<br/>safe fetch · parse · clean · chunk"]
    PL --> VS[("Vector index<br/>pgvector / in-memory")]
    G --> SV["Deterministic services<br/>source scoring · coverage · quality gate · citations · rendering"]
```

| Layer | Package | Responsibility |
|---|---|---|
| API | `researchgraph.api` | FastAPI routes, SSE, optional bearer auth, validation and limits |
| Run lifecycle | `researchgraph.runtime` | Background execution, event stream, read-model projection, drain/resume, DI container |
| Orchestration | `researchgraph.graph` | Typed state + reducers, nodes, routing, worker subgraph, instrumentation, strict serde |
| Reasoning | `researchgraph.agents` | Prompted LLM roles with structured outputs and deterministic fallbacks |
| Domain services | `researchgraph.services` | Source scoring, coverage, contradictions, quality gate, citations, rendering |
| Retrieval | `researchgraph.retrieval` | Loader, cleaning, chunking, embeddings, vector backends, reranking |
| Tools | `researchgraph.tools` | Search providers, safe fetcher, document parsing, metadata, calculator |
| LLM | `researchgraph.llm` | Provider factory, structured-output helper, usage/cost tracking, mock model |
| Persistence | `researchgraph.database` | SQLAlchemy models + repository, checkpointer factory |
| Demo | `researchgraph.demo` | Synthetic corpus, deterministic responder, fault injector |

## 4. The LangGraph workflow

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
    source_evaluation -- "insufficient evidence & budget left" --> query_generation
    source_evaluation -- "sufficient (or budget spent)" --> contradiction_detection
    contradiction_detection --> synthesis --> critic --> quality_gate
    quality_gate -- "research_more" --> query_generation
    quality_gate -- "proceed / proceed with limitations" --> report_writer
    report_writer --> citation_verification
    citation_verification -- "issues & repairs left" --> citation_repair --> citation_verification
    citation_verification -- verified --> final_review --> E
```

The diagram generated from the compiled graph (`graph.get_graph(xray=1)`) is in [`docs/graph.mmd`](docs/graph.mmd)
and served live at `GET /graph`.

| Node | Kind | What it does |
|---|---|---|
| `intake` | deterministic | Sanitises/validates the question; sets per-run limits |
| `planner` | LLM (strong) | Structured `ResearchPlan`; deterministic fallback plan on failure |
| `plan_review` | **interrupt** | Human approval; returns `Command(goto=…)` |
| `query_generation` | LLM (fast) | One batched call writes de-duplicated queries for every target; allocates tool budgets |
| `research_worker` | **subgraph** × N | `search_agent` ⇄ `execute_tools` (`ToolNode`) → `source_processing` → `passage_retrieval` → `evidence_extraction` |
| `source_evaluation` | deterministic | Scores all sources; computes evidence coverage and gaps |
| `contradiction_detection` | rules + LLM | Mines candidate conflicts, LLM adjudicates only those |
| `synthesis` | LLM (strong) | Calibrated findings; drops any finding without real evidence |
| `critic` | rules + LLM | Independent review → structured `Critique` |
| `quality_gate` | deterministic | Weighted score + blocking rules → `proceed` / `research_more` / `proceed_with_limitations` |
| `report_writer` | LLM (strong) | Sentences carry evidence IDs, never citation numbers or URLs |
| `citation_verification` / `citation_repair` | deterministic | Verify every sentence; re-attribute or remove |
| `final_review` | deterministic | De-duplicate, final verification, numbering, bibliography, Markdown |

## 5. State model

The state is a `TypedDict` whose fields hold frozen Pydantic models. Fields written by parallel workers use
reducers that are **commutative and idempotent**, so concurrent updates merge deterministically:

```python
class ResearchState(TypedDict, total=False):
    research_id: str
    question: str
    limits: RunLimits                                   # budgets live in state → routing is pure
    plan: ResearchPlan
    iteration: int
    pending_tasks: list[ResearchTask]
    missing_evidence: list[ResearchGap]
    tool_calls_used: Annotated[int, operator.add]
    search_history: Annotated[list[str], append_unique]
    sources: Annotated[dict[str, Source], merge_sources]          # content-addressed by canonical URL
    source_quality: Annotated[dict[str, SourceQuality], merge_by_id]
    evidence: Annotated[dict[str, Evidence], merge_by_id]         # content-addressed by source + quote
    evidence_assessment: EvidenceAssessment
    contradictions: list[Contradiction]
    analysis: Synthesis
    critique: Critique
    critique_history: Annotated[list[Critique], operator.add]
    quality: QualityAssessment
    quality_history: Annotated[list[QualityAssessment], operator.add]
    draft_report: ReportDraft
    citation_check: CitationCheck
    final_report: FinalReport
    errors: Annotated[list[WorkflowError], operator.add]          # recorded, not raised
    node_metrics: Annotated[list[NodeMetric], operator.add]
```

- **Frozen domain models** (`ConfigDict(frozen=True)`) make in-place mutation of state impossible.
- **Content-addressed IDs** (`S-<hash(url)>`, `E-<hash(subquestion, source, quote)>`) mean two workers that find
  the same source converge on one entry.
- **Strict checkpoint deserialisation:** the checkpointer only reconstructs types on an explicit allowlist (every
  model and enum in `researchgraph.schemas`), closing the "write to the checkpoint DB → code execution" vector.
- LLM-facing output schemas (`schemas/llm.py`) are separate from domain models: they contain no IDs, URLs or
  timestamps — those are assigned in code.

## 6. Tool architecture

| Tool | Exposed to the LLM? | Notes |
|---|---|---|
| `web_search` | yes | Tavily (if `TAVILY_API_KEY`), else falls back to academic providers |
| `academic_search` | yes | arXiv + Semantic Scholar, rate-limited, retried with backoff |
| `calculator` | yes | AST-whitelisted arithmetic with magnitude/exponent bounds — no `eval` |
| safe fetcher | **no** | SSRF validation on every redirect hop, streaming byte cap, content-type allowlist |
| document extraction | **no** | PDF (pypdf), HTML (BeautifulSoup + citation meta tags), Markdown, text |
| metadata extraction | **no** | Titles, authors, dates, DOIs from Highwire/Dublin Core/OpenGraph/PDF metadata |

Tools are LangChain `@tool`s executed by LangGraph's prebuilt `ToolNode`; dependencies arrive through `ToolRuntime`
injection. Search tools use `response_format="content_and_artifact"`: **the model sees a compact listing without
URLs**, while structured results ride along as the message artifact. The model can only *choose among* sources a
search provider returned — it cannot request arbitrary URLs, and there is no shell or code-execution tool. Searches
are cached and concurrent identical requests share one in-flight call.

## 7. RAG architecture

```mermaid
flowchart LR
    A[search results] --> B[candidate ranking<br/>+ diversity slot] --> C[safe fetch] --> D[parse<br/>PDF · HTML · MD · text]
    D --> E[clean<br/>de-hyphenate · reflow · boilerplate] --> F[split references<br/>content signals]
    F --> G[RecursiveCharacterTextSplitter<br/>+ provenance metadata] --> H[batched embeddings] --> I[(vector store)]
    I --> J[multi-query, source-scoped retrieval] --> K[hybrid rerank<br/>vector + lexical, per-source cap] --> L[evidence extraction<br/>+ quote verification]
```

- **Loader:** a LangChain `BaseLoader` (`SourceDocumentLoader`) over already-fetched bytes; every chunk carries
  `research_id, source_id, title, url, author, date, document_type, chunk_index, chunk_id, retrieved_at`.
- **Embeddings:** `BatchingEmbeddings` wraps any LangChain `Embeddings` with batching, bounded retries and a query
  cache. Default `HashingEmbeddings` is deterministic and offline; `EMBEDDING_PROVIDER=openai` for semantic vectors.
- **Vector stores:** LangChain `InMemoryVectorStore` locally; `langchain_postgres.PGVectorStore` on PostgreSQL with
  `research_id`/`source_id` as indexed columns for filtered search.
- **Source-aware retrieval:** each worker searches only the sources it processed (plus relevant known sources),
  so results are scoped per run and per subquestion — no leakage between runs.
- **Hybrid rerank:** 0.6 × vector similarity + 0.4 × lexical coverage, near-duplicate suppression, max chunks per
  source.

This is a component of the research workflow, not an "upload a PDF and chat" interface.

## 8. Citations and provenance

```text
Report sentence  "RAG reached 71.4% exact match …"   → evidence_ids: [E-6b1…]
Evidence E-6b1   claim, verbatim quote, passage S-3fa…:4, confidence, relevance
Source S-3fa…    url, title, authors, date, venue, type, quality 0.79
Citation [1]     numbered by first appearance → bibliography entry
```

- The writer may only attach **evidence IDs** that exist; it never writes `[n]` or URLs (enforced: numeric bracket
  markers are stripped from model text — including a paper's *own* "[14]"-style references).
- **Verification** checks every sentence: `missing_citation`, `unknown_evidence`, `low_relevance` (cited text must
  cover the sentence's key terms), and `numeric_mismatch` (every number in the sentence must appear in the cited
  evidence — a cheap, effective guard against numeric hallucination).
- **Repair** re-attributes a failing sentence to the best-supporting evidence, or removes it. Removed sentences are
  listed in the report ("Claims Removed During Verification") rather than silently published.
- Contradictions are rendered in a "Conflicting Evidence" section with both sides cited.

## 9. Source quality scoring

`overall = 0.25·authority + 0.12·recency + 0.20·relevance + 0.15·primary + 0.18·methodology + 0.10·citations`

| Signal | Basis |
|---|---|
| Source type | Provider hint (e.g. arXiv), curated domain lists, URL heuristics → primary research · official docs · institutional · technical publication · news · blog · forum · unknown |
| Authority | Type prior, adjusted for peer-reviewed venue vs. preprint |
| Recency | Exponential decay, configurable half-life (default 4 years); unknown dates penalised |
| Relevance | Lexical match to the subquestions + the extractor's relevance of the source's evidence |
| Methodology | Methodology-term density, abstract present, numeric density, type prior |
| Citations | Citation count (Semantic Scholar) or reference-list size |

Every score ships with a human-readable rationale (visible in the dashboard). Tiers: high ≥ 0.70, medium ≥ 0.45.
A "credible" evidence item needs a medium+ source **and** relevance ≥ 0.4.

![Sources tab: explainable source-quality scores](docs/images/sources.png)

## 10. Critic and reflection loop

The critic reviews artifacts — it never rewrites the report:

- **Rule-based checks** (always run): findings with no valid evidence, findings resting only on low-quality
  sources, "high confidence" from a single source, contradictions not reflected in any finding, subquestions with
  no findings.
- **LLM review** asks: which claims are unsupported or overgeneralised, are citations relevant, are conclusions
  stronger than the evidence, what research would change the conclusion — returning a structured `Critique` with
  severities and suggested queries.
- Duplicate issues are merged keeping the **most severe** version (a regression test guards this — see §27).

The **quality gate** turns critique + measurable evidence properties into a decision:

| Component | Weight |
|---|---|
| coverage — subquestions with a credible finding | 0.30 |
| support — findings backed by credible, relevant evidence | 0.25 |
| mean quality of cited sources | 0.20 |
| critic penalty (severity-weighted) | 0.15 |
| major contradictions represented in findings | 0.10 |

Blocking conditions (critical issues, uncovered subquestions, no findings) fail the gate regardless of score.
On failure with budget left → **targeted research** on the gaps the critic identified; without budget → the
report is written **with explicit limitations** instead of overclaiming.

## 11. Conditional routing

All routing functions are pure functions of state (limits live in state), so every branch is unit-tested.

| After | Function | Routes |
|---|---|---|
| `intake` | `route_after_intake` | `planner` · `END` (rejected) |
| `plan_review` | `Command(goto=…)` | `query_generation` · `planner` · `plan_review` · `END` |
| `query_generation` | `dispatch_research` | `[Send("research_worker", …) × N]` · `source_evaluation` (no budget) |
| `search_agent` | `route_search_agent` | `execute_tools` · `source_processing` |
| `source_evaluation` | `route_after_evidence` | `query_generation` (insufficient & budget) · `contradiction_detection` |
| `quality_gate` | `route_after_quality_gate` | `query_generation` (research_more) · `report_writer` |
| `citation_verification` | `route_after_citation_check` | `citation_repair` (issues & attempts left) · `final_review` |

Termination is guaranteed by `MAX_RESEARCH_ITERATIONS`, `MAX_TOOL_CALLS` (per-task budgets are allocated from the
remaining global budget), `MAX_CITATION_REPAIR_ATTEMPTS`, bounded search rounds per worker, and LangGraph's
`recursion_limit`.

## 12. Parallel research

`query_generation` returns one `Send("research_worker", WorkerInput)` per task, so LangGraph runs the worker
subgraphs in the same superstep (bounded by `max_concurrency`). Each worker has its own typed `WorkerState` with
an `output_schema` that exposes only `sources`, `evidence`, `tool_calls_used`, `search_history`, `errors` and
`node_metrics` to the parent, where the reducers above merge them. A worker failure is contained: worker nodes
convert unexpected exceptions into recorded `WorkflowError`s and degraded updates rather than failing the run.

## 13. Human-in-the-loop

![Plan review: the run is paused at a LangGraph interrupt](docs/images/plan-review.png)

`plan_review` calls `interrupt({"type": "plan_review", "plan": …})`. The run's state is checkpointed and the
background task ends. The API resumes it with `Command(resume=PlanDecision)`:

| Endpoint | Decision |
|---|---|
| `POST /research/{id}/approve` | approve as-is |
| `POST /research/{id}/edit-plan` | replace the plan (validated, re-numbered, versioned); approve now or review again |
| `POST /research/{id}/replan` | regenerate with natural-language feedback |
| `POST /research/{id}/cancel` | cancel (also works mid-run) |

Because the pause is a real checkpoint, an approval can arrive after a server restart (covered by an integration
test). `auto_approve: true` skips the pause for automation.

## 14. Persistence and durability

- **LangGraph checkpointer** (`AsyncPostgresSaver` with a psycopg pool, or `AsyncSqliteSaver`) is the source of
  truth for workflow state, with `durability="sync"`.
- **Read model** (SQLAlchemy 2.0 async): `research_runs`, `sources`, `evidence`, `findings`, `reports`,
  `run_events`. The `RunManager` projects graph updates into it by re-applying the **same reducers** the graph
  uses, so the projection cannot disagree with graph semantics. Upserts use dialect-specific `ON CONFLICT`.
- **Event log:** every workflow event is persisted with a gap-free sequence number; SSE clients replay history and
  resume with `Last-Event-ID`.
- **Restart survival:** on shutdown, running graphs are asked to stop at the next superstep boundary via
  LangGraph's `RunControl.request_drain()` (state checkpointed, status `interrupted`); on startup `recover()`
  resumes them from their last checkpoint. Paused runs stay paused. Both paths are integration-tested.

## 15. Evaluation

`python -m evaluation.run` runs every question in [`evaluation/datasets/research_questions.json`](evaluation/datasets/research_questions.json)
through the full stack and checks **behavioural expectations** (not answer keys) — grounding, coverage, honesty
about gaps — including a **negative control** (an off-topic question the corpus cannot answer: the system must
produce *no* findings rather than stretch irrelevant evidence).

**Deterministic metrics** (no model involved):

| Metric | Definition |
|---|---|
| `citation_coverage` | published sentences with ≥ 1 valid citation / published sentences |
| `unsupported_claim_rate` | sentences removed by verification / sentences written |
| `first_pass_citation_precision` | sentences that passed verification before any repair |
| `evidence_quality` | mean quality score of cited sources |
| `evidence_relevance` | mean relevance of cited evidence |
| `quote_verification_rate` | extracted quotes found verbatim in the source |
| `research_completeness` | mean of subquestion coverage and expected-aspect coverage |
| latency, LLM calls, tokens, tool calls, estimated cost | measured per run |

**LLM-as-judge** (`--judge`) scores faithfulness, completeness, balance, actionability and clarity on a 1–5 rubric.
It is reported in a **separate section** and never blended into the deterministic metrics.

**Measured results — demo mode** (from [`docs/examples/evaluation-demo.md`](docs/examples/evaluation-demo.md)):

| Question | citation coverage | unsupported rate | first-pass precision | cited-source quality | completeness | latency (s) | checks |
|---|---|---|---|---|---|---|---|
| rag-vs-finetuning | 1.000 | 0.038 | 0.962 | 0.662 | 1.000 | 1.5 | 10/10 |
| small-vs-large-models | 1.000 | 0.042 | 0.958 | 0.643 | 1.000 | 1.2 | 9/9 |
| rag-hallucinations | 1.000 | 0.048 | 0.955 | 0.667 | 1.000 | 1.3 | 9/9 |
| quantization-edge | 1.000 | 0.043 | 0.957 | 0.639 | 1.000 | 2.0 | 9/9 |
| negative-control-off-topic | 1.000 | 1.000\* | 0.000 | — | 0.000\* | 2.8 | 8/8 |

\* The negative control is *supposed* to produce nothing: its one generic sentence was removed by verification and
no off-topic findings were made.

> **What these numbers mean:** demo mode uses a deterministic mock model and a synthetic corpus, so these results
> validate the *machinery* (grounding, verification, routing, termination), **not** real-world research quality.
> As a robustness check, the full workflow was also run against **real arXiv papers** with the mock model: 18 real
> sources parsed, 46 quote-verified evidence items, 100 % citation coverage; three PDFs over the 5 MB limit degraded
> to their abstracts. That run exposed — and led to the fix for — source reference markers leaking into claims.

**Measured results — live mode** (from [`docs/examples/evaluation-live.md`](docs/examples/evaluation-live.md)):
DeepSeek V4.1 Flash (`deepseek-flash`) through its OpenAI-compatible API, real arXiv + Semantic Scholar search,
real PDF fetching, plan auto-approved, no prompt tuning. The negative control is demo-only (its expectations are
defined against the synthetic corpus).

| Question | citation coverage | unsupported rate | first-pass precision | cited-source quality | completeness | latency (s) | checks |
|---|---|---|---|---|---|---|---|
| rag-vs-finetuning | 1.000 | 0.061 | 0.892 | 0.795 | 1.000 | 593 | 9/10 |
| small-vs-large-models | 1.000 | 0.000 | 1.000 | 0.798 | 1.000 | 249 | 9/9 |
| rag-hallucinations | 1.000 | 0.043 | 0.957 | 0.803 | 0.667 | 308 | 8/9 |
| quantization-edge | 1.000 | 0.091 | 0.886 | 0.767 | 1.000 | 315 | 9/9 |
| **mean** | 1.000 | 0.049 | 0.934 | 0.791 | 0.917 | 366 | 95 % |

~146k tokens per run. **Failed checks, unedited:** `rag-vs-finetuning` found no contradiction although the dataset
expects one (the live literature may simply not disagree the way the synthetic corpus does); `rag-hallucinations`
covered only 2 of 6 expected aspects — the papers live search returned do not discuss reranking, abstention or
entailment verification, so the report does not either. Both expectations were written against the demo corpus,
which is itself a finding: behavioural checks need a live-specific dataset. LLM-as-judge means were 4.0–4.6 / 5,
but the judge is the same model that wrote the reports, so treat them as weak evidence.
A full, unedited live report: [`docs/examples/live-report-deepseek.md`](docs/examples/live-report-deepseek.md).

The first live run also surfaced a scoring bug: arXiv **surveys were scored as primary research**. They are now
treated as secondary sources (regression test included).

## 16. Observability

- **LangSmith tracing** is enabled only when `LANGCHAIN_TRACING_V2=true` **and** `LANGCHAIN_API_KEY` are set
  (otherwise explicitly disabled). Runs carry `run_name`, tags and `research_id` metadata; every structured LLM call
  is named after its operation (`planner`, `critic`, …).
- **Token usage and cost:** a LangChain callback (`UsageTracker`) aggregates `usage_metadata` across all model calls,
  including inside parallel workers; set `LLM_INPUT_COST_PER_MTOK` / `LLM_OUTPUT_COST_PER_MTOK` for cost estimates.
- **Workflow events:** each node is wrapped by `@instrumented`, which emits `node_started` / `node_completed`
  events with latency through LangGraph's custom stream, records `NodeMetric`s in state, and converts failures into
  recorded errors. Retries and validation repairs are emitted as warnings. Events never contain model reasoning.
- **Metrics in the UI:** sources, high-quality sources, evidence, findings, contradictions, critic issues,
  iterations, tool calls, LLM calls, tokens, active time, quality score, citation coverage, per-node latency.
- Structured JSON logs with `LOG_JSON=true`.

## 17. Security

- No secrets in code; `SecretStr` settings; `.env` is git-ignored; blank values are treated as unset.
- **SSRF protection:** http(s) only, standard ports, no credentials in URLs, blocked internal hostnames, DNS-resolved
  addresses must be globally routable (blocks loopback, RFC 1918, link-local/cloud-metadata), re-validated on every
  redirect hop.
- **Resource limits:** 5 MB per document, 200 k extracted characters, 40 PDF pages, 256 KB request bodies,
  artifact pages ≤ 500 items, report size cap, iteration/tool/repair budgets, graph recursion limit, per-node timeouts.
- **Prompt-injection boundary:** retrieved text is always placed in blocks declared as untrusted data; the model
  never sees or emits URLs; fetching is not a model tool; no shell/code-execution tools; the calculator is an AST
  whitelist.
- **Input sanitisation** (NFKC, control characters) and Pydantic validation at the API.
- Optional bearer-token auth (`API_TOKEN`, constant-time comparison); CORS allowlist; nginx security headers.
- A cap on concurrently executing runs (`MAX_ACTIVE_RUNS`, HTTP 429) bounds concurrent LLM spend.
- Strict, allowlisted checkpoint deserialisation (see §5).

## 18. Quick start (offline demo)

Requires Python 3.12+. No API keys, no network.

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m scripts.demo
```

Other demo options:

```bash
python -m scripts.demo --question 3                 # sample questions 1-3
python -m scripts.demo --all                        # all three
python -m scripts.demo --interactive                # approve / revise / cancel the plan yourself
python -m scripts.demo --simulate-failures all      # failed source, LLM timeout, invalid output, insufficient evidence, critic rejection
python -m scripts.demo --custom "Your own question"  # uses the fallback planner
```

Each run writes `report.md`, `executive_summary.md`, `sources.json`, `evidence.json`, `run.json`, `events.jsonl` and
the compiled graph (`workflow.mmd`) to `demo_output/<research_id>/`.

## 19. Installation and configuration

```bash
pip install -e ".[all,dev]"      # all = OpenAI + Anthropic integrations + langchain-postgres/pgvector
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, or any `init_chat_model` provider: `openai`, `anthropic`, `google_genai`, `ollama`, `groq`, … |
| `MODEL_NAME` / `FAST_MODEL_NAME` | mock | Strong model (planning, synthesis, critique, writing) / optional cheaper model (queries, extraction, tools) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | — | Provider keys (other providers read their standard env vars) |
| `LLM_STRUCTURED_OUTPUT_METHOD` | `auto` | `auto` = native JSON-schema output for Anthropic/OpenAI (newer Claude models reject forced tool calls), JSON mode with the schema stated in the prompt for OpenAI-compatible endpoints, the integration's default elsewhere |
| `LLM_BASE_URL` | — | Any OpenAI-compatible endpoint (DeepSeek, vLLM, Together, …) with `LLM_PROVIDER=openai` |
| `LLM_INPUT_COST_PER_MTOK`, `LLM_OUTPUT_COST_PER_MTOK` | — | Enables cost estimates |
| `SEARCH_PROVIDER` | `mock` | `auto` (Tavily if keyed + arXiv + Semantic Scholar), `tavily`, `arxiv`, `semantic_scholar` |
| `TAVILY_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY` | — | Search keys (S2 works without a key, rate-limited) |
| `RESEARCH_AGENT_MODE` | `llm` | `deterministic` executes planned queries without the tool-choosing agent |
| `EMBEDDING_PROVIDER` | `hashing` | `openai` for semantic embeddings |
| `DATABASE_URL` | SQLite file | `postgresql+psycopg://…` for Postgres (pgvector used automatically) |
| `MAX_RESEARCH_ITERATIONS` / `MAX_TOOL_CALLS` | 3 / 60 | Research budgets |
| `MAX_ACTIVE_RUNS` | 4 | Concurrently executing runs per process (excess requests get HTTP 429) |
| `QUALITY_GATE_THRESHOLD` | 0.65 | Gate threshold |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | off | LangSmith tracing |
| `API_TOKEN` | — | Require `Authorization: Bearer …` |
| `DEMO_FAILURES` | — | Default failure scenarios for demo runs |

The full list with comments is in [`.env.example`](.env.example). Example — Anthropic + real search:

```bash
LLM_PROVIDER=anthropic
MODEL_NAME=claude-sonnet-5-5
FAST_MODEL_NAME=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=...
SEARCH_PROVIDER=auto
TAVILY_API_KEY=...
```

## 20. Running locally

```bash
# API (demo configuration from scripts/demo.env) — http://localhost:8000/docs
python scripts/serve.py

# or with your .env
uvicorn researchgraph.main:create_app --factory --port 8000

# Dashboard — http://localhost:5173 (proxies /api to :8000)
cd frontend && npm install && npm run dev
```

## 21. Running with Docker

```bash
cp .env.example .env    # optional
docker compose up --build
```

| Service | URL |
|---|---|
| Dashboard (nginx, proxies `/api`) | http://localhost:8080 |
| API + Swagger UI | http://localhost:8000/docs |
| PostgreSQL 16 + pgvector | `postgres:5432` (internal) |

Compose wires the backend to Postgres (checkpointer, read model and pgvector index in one database), waits on health
checks, and gives in-flight runs a 30 s grace period to drain to a checkpoint on `docker compose stop`.

## 22. API

| Method | Path | Description |
|---|---|---|
| `POST` | `/research` | Create a run (`question`, `auto_approve`, `failure_scenarios`) → 202 |
| `GET` | `/research` | Run history |
| `GET` | `/research/{id}` | Status, stage, progress, metrics, usage |
| `GET` | `/research/{id}/plan` | Current plan (+ whether it is editable) |
| `POST` | `/research/{id}/approve` · `/edit-plan` · `/replan` | Resume from the plan interrupt |
| `POST` | `/research/{id}/cancel` | Cancel (paused or running) |
| `GET` | `/research/{id}/sources` · `/evidence` · `/findings` | Artifacts (paginated) |
| `GET` | `/research/{id}/report` | Report JSON (`?format=markdown` to download) |
| `GET` | `/research/{id}/events` | Persisted workflow events |
| `GET` | `/research/{id}/stream` | Server-Sent Events: replay + live, `Last-Event-ID` resume |
| `GET` | `/graph` | Mermaid diagram of the compiled graph |
| `GET` | `/health` | Configuration summary and active runs |

## 23. Example research question and output

> *Compare the effectiveness of RAG, fine-tuning, and long-context prompting for domain-specific question answering.
> Use recent academic papers and credible technical sources. Identify evidence, limitations, benchmarks, and practical
> recommendations.*

Demo run summary (`python -m scripts.demo`):

```text
Status: COMPLETED   wall time 1.4s   research iterations: 1
Tool calls: 34   LLM calls: 21   tokens: 50,144 (offline estimate)
Sources: 8 (4 high quality, 0 failed, 0 snippet-only)   evidence: 46   findings: 15   contradictions: 1
Critic: 0 issue(s), verdict acceptable   quality gate score: 0.943
Report: 25 verified sentences, 100% citation coverage, 1 removed, 7 sources cited
```

![Full report with clickable citations](docs/images/report.png)

The complete generated report — executive summary, sections, recommendations, conflicting evidence, limitations,
methodology, removed claims and bibliography — is in [`docs/examples/sample-report.md`](docs/examples/sample-report.md).
**It is synthetic demo output** (fictional sources on reserved `.example` domains, illustrative figures).

## 24. Testing

```bash
pytest                                    # 111 offline tests (mock LLM, synthetic corpus, mocked HTTP)
TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/db pytest -m postgres
ruff check backend scripts evaluation && ruff format --check backend scripts evaluation
mypy                                      # clean
cd frontend && npm run build              # strict TypeScript + production build
```

112 tests (84 unit, 28 integration), no real API calls:

- **Unit:** text utilities, schemas and reducers (merge semantics, frozen models, strict serde round-trip), document
  parsing (PDF/HTML/Markdown, malformed input), cleaning (heading and hyphenation regressions), SSRF and fetch limits,
  calculator safety, source scoring, coverage, quality gate, citation verification/repair/numbering, every routing
  branch, retries and structured-output repair, critic rules and merge, contradiction mining, embeddings, index
  scoping, search caching/coalescing, provider response parsing, fault injection, `.env.example`.
- **Integration:** full workflow for every sample question with provenance invariants, each failure scenario, HITL
  interrupt/edit/replan/cancel across a simulated restart, the HTTP API end to end (including SSE replay and
  `Last-Event-ID`), the active-run limit (429) and concurrent approvals (exactly one wins), drain-on-shutdown +
  resume-on-startup, and a full run on PostgreSQL + pgvector.

CI (GitHub Actions) runs lint, format, mypy and tests; Postgres integration against a `pgvector/pgvector` service;
the demo with all failures and the evaluation harness; the frontend build; and both Docker image builds.

## 25. Project structure

```text
researchgraph/
├── backend/
│   ├── researchgraph/
│   │   ├── agents/        # planner, query_writer, researcher, extractor, contradiction_judge, synthesizer, critic, writer
│   │   ├── api/           # routes, SSE, auth
│   │   ├── config/        # settings, logging, LangSmith
│   │   ├── core/          # errors, retry/backoff, text utilities, fault-injection points
│   │   ├── database/      # SQLAlchemy models, repository, checkpointer factory
│   │   ├── demo/          # synthetic corpus, deterministic responder, fault injector, PDF writer
│   │   ├── evaluation/    # deterministic metrics, LLM-as-judge
│   │   ├── graph/         # state & reducers, nodes, routing, builder, instrumentation, serde
│   │   ├── llm/           # provider factory, structured-output helper, mock model, usage tracking
│   │   ├── retrieval/     # loader, cleaning, embeddings, vector backends, rerank
│   │   ├── runtime/       # run manager, event broker, streaming, DI container, bootstrap
│   │   ├── schemas/       # domain, LLM-output and API models
│   │   ├── services/      # source quality, coverage, contradictions, quality gate, citations, rendering
│   │   ├── tools/         # search providers, safe fetcher, documents, metadata, calculator, LLM tools
│   │   └── main.py        # FastAPI app factory
│   └── tests/             # unit/ and integration/
├── frontend/              # React 19 + TypeScript + Vite + Tailwind dashboard
├── evaluation/            # datasets/ and run.py (python -m evaluation.run)
├── scripts/               # demo.py (python -m scripts.demo), serve.py
├── docs/                  # architecture, design decisions, examples, screenshots, graph.mmd
├── docker/                # backend/frontend Dockerfiles, nginx.conf
├── .github/workflows/     # CI
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

## 26. Engineering tradeoffs

- **LLM vs. deterministic code.** Eight steps use a model; scoring, gating, numbering, verification and rendering
  do not. Deterministic parts are cheaper, testable and cannot be argued with — but they encode heuristics (e.g.
  lexical relevance) that a model would judge more flexibly.
- **Lexical verification.** Citation relevance uses term coverage and number matching rather than an entailment
  model: fast and transparent, but it can accept a topically-matching sentence that subtly misstates its source.
- **Strict publication policy.** Unsupported sentences are removed, not hedged. Reports can be shorter, but never
  contain uncited claims.
- **Hashing embeddings by default.** Zero setup and fully reproducible; lexical rather than semantic similarity.
  Production should use `EMBEDDING_PROVIDER=openai` (or another embedding model).
- **In-process run execution.** Simple and observable; horizontal scaling needs a job queue (see §29).
- **Single-call batching** (all queries in one call, one extraction call per worker) trades some per-item control for
  far fewer LLM calls.
- **Diversity slot in source selection** deliberately fetches one practitioner source per worker so contradictions
  can be detected; quality scoring then down-weights it.

## 27. Failure modes

| Failure | Handling | Tested |
|---|---|---|
| LLM timeout / rate limit / 5xx | Per-call timeout + exponential backoff with jitter (bounded); provider SDK retries kept low to avoid multiplication | ✔ |
| Invalid structured output | Validation error fed back; bounded repair rounds; then a deterministic fallback (plan, queries, findings, critique, report) | ✔ |
| Failed or oversized source | Recorded as `WorkflowError`; degrade to the provider's abstract/snippet; continue | ✔ |
| Search provider down / 429 | Retried; other providers' results still used; typed error if all fail | ✔ |
| Insufficient evidence | Evidence gate → targeted round; budget exhaustion → limitations in the report | ✔ |
| Critic rejection | Quality gate → targeted research | ✔ |
| Worker crash / timeout | Contained per worker; partial results kept | ✔ |
| Embedding failure | Batched retries; typed `EmbeddingError` | ✔ |
| Server restart mid-run | Drain to checkpoint; resume on startup | ✔ |
| Runaway loops | Iteration, tool, repair and recursion limits | ✔ |
| Off-topic question | Topical relevance floor; no findings rather than stretched evidence (negative control) | ✔ |

Bugs this process caught (each now has a regression test): a critique-merge that downgraded a critical issue, PDF
titles fusing into extracted sentences, compound words losing hyphens, an SSE `error` event colliding with
`EventSource`'s built-in error event, naive SQLite timestamps rendered in the wrong timezone, a paper's own
reference markers being mistaken for ResearchGraph citations, HTML headings fusing into the first sentence after
them, and one conflicting quote pair being adjudicated (and reported) once per subquestion that used it.

## 28. Limitations

- **Demo results are not research results.** The mock model is an extractive stand-in that exercises every code
  path; its prose is not representative of an LLM.
- Live mode has been measured with one provider (DeepSeek V4.1 Flash) on four questions; a larger, human-labelled
  dataset and other models (Claude, GPT) are still to do. Live runs take 4–10 minutes, dominated by arXiv rate limits.
- Citation verification is lexical (no entailment model); relevance and contradiction candidate mining are heuristic.
- The default embeddings are lexical; the in-memory vector store is not durable across restarts (pgvector is).
- Runs execute in the API process; there is no multi-tenant auth or per-user quotas.
- Semantic Scholar's unauthenticated rate limit is low; arXiv requires ~3 s between requests, which dominates latency.
- PDFs are parsed text-only (no OCR, tables or figures).
- Anthropic and OpenAI models were verified offline only (model construction, structured-output schemas, tool
  binding); prompts have not been tuned against any live model's output.
- SSRF checks resolve DNS before connecting, leaving a small DNS-rebinding window; pinning the resolved address in
  the HTTP transport would close it.
- On Windows with PostgreSQL, start the API with `python -m researchgraph.main` (it selects the selector event loop
  psycopg needs); Linux and Docker are unaffected.

## 29. Future improvements

- Entailment/NLI-based citation verification and claim decomposition (atomic claims) behind the current lexical checks.
- Cross-encoder reranking and semantic embeddings by default; persistent embedding cache.
- A distributed job queue (e.g. Redis/Arq or LangGraph Platform) for horizontal scaling; per-user auth and quotas.
- Live-mode evaluation with LLM-as-judge on a larger, human-labelled dataset; regression tracking in CI.
- Prompt caching and token-budget-aware context packing; streaming of report tokens.
- Table/figure extraction and OCR for PDFs; more providers (OpenAlex, PubMed, Crossref).
- Incremental synthesis across rounds instead of full re-synthesis.

## License

MIT — see [LICENSE](LICENSE).
