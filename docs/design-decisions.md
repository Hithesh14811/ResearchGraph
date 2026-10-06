# Design decisions

Each entry records the decision, the alternatives considered, why the choice was made, and what it costs.

---

## 1. LangGraph instead of a chain or a single agent loop

**Decision.** Model the workflow as a LangGraph `StateGraph` with typed state, explicit nodes, conditional edges,
loops, a parallel worker subgraph, checkpointing and `interrupt()`.

**Alternatives.** A LangChain LCEL chain; a single ReAct agent with search tools; a hand-written asyncio orchestrator.

**Reason.** The problem has real control flow: an approval pause, data-dependent loops (insufficient evidence,
critic rejection, citation repair), map-reduce over subquestions, and the need to resume after crashes. A chain
cannot loop or pause; a single agent hides the control flow inside a prompt and cannot guarantee termination,
budgets or verification; a hand-written orchestrator would re-implement checkpointing, streaming and interrupts.

**Tradeoffs.** More concepts (reducers, `Send`, `Command`, namespaces in streams) and a framework dependency that
evolves quickly. Mitigated by keeping routing pure, nodes small and the graph definition in one file.

## 2. Deterministic code wherever judgement is not required

**Decision.** Only eight roles call a model (planning, query writing, tool selection, extraction, contradiction
adjudication, synthesis, critique, writing). Source scoring, coverage, the quality gate, citation numbering,
verification, repair and rendering are plain Python.

**Alternatives.** Make every step an "agent" (e.g. an LLM quality gate, an LLM citation formatter).

**Reason.** Deterministic steps are cheaper, faster, reproducible and unit-testable, and they cannot be talked into a
different answer. The most important guarantees — no fabricated citations, no unsupported published sentences,
guaranteed termination — must not depend on a model behaving.

**Tradeoffs.** Heuristics (lexical relevance, domain lists) are less flexible than model judgement. They are kept
explainable (every score has a rationale) and can be swapped for stronger checks (e.g. NLI) behind the same interfaces.

## 3. PostgreSQL (with SQLite for local development)

**Decision.** PostgreSQL for checkpoints, the read model and pgvector in production; SQLite + in-memory vectors
locally and in tests. Selected by `DATABASE_URL`.

**Alternatives.** Postgres only; a document database; a separate vector database; files.

**Reason.** One durable store for workflow state, queryable artifacts and embeddings keeps operations simple and
transactional. pgvector supports filtered similarity search by `research_id` / `source_id`. SQLite keeps the project
runnable in seconds without Docker, and the same SQLAlchemy code runs on both (dialect-specific upserts).

**Tradeoffs.** Two backends to keep compatible (timezones, upserts, event loops — each covered by tests or a
regression fix); pgvector is not the fastest ANN engine at very large scale.

## 4. Structured outputs, with LLM-facing schemas separate from domain models

**Decision.** Every model call uses `with_structured_output` (provider-native structured output + Pydantic
validation). The schemas the
model fills (`schemas/llm.py`) omit IDs, URLs and timestamps; code converts them into frozen domain models.

**Alternatives.** Ask for JSON in the prompt and parse it; let the model fill domain models directly.

**Reason.** Native structured output plus validation removes most parsing failures, and the remaining ones are repaired by
feeding the validation error back (bounded). Keeping identifiers out of the model's hands is what makes provenance
trustworthy: the model can reference only IDs it was shown, and everything it returns is validated against state.

**Tradeoffs.** Two parallel sets of models and conversion code; provider-specific JSON-schema limitations
(kept simple: no unions or recursive types in LLM schemas). The mode is provider-dependent: forced tool calling is
not supported by the newest Claude models, so `LLM_STRUCTURED_OUTPUT_METHOD=auto` selects native JSON-schema output
for Anthropic and OpenAI and leaves other integrations on their defaults.

## 5. Separate planner and critic

**Decision.** Planning, synthesis and critique are different roles with different prompts and inputs; the critic is
instructed to review, not rewrite, and is complemented by rule-based checks.

**Alternatives.** Self-reflection by the same prompt that wrote the analysis; no critique.

**Reason.** A reviewer that did not produce the artifact, given the evidence it rests on, finds different problems.
Rule-based checks catch certain failures (findings with no evidence, single-source "high confidence", ignored
contradictions, uncovered subquestions) every time, even if the model misses them.

**Tradeoffs.** Extra model call per iteration. Duplicate issues from rules and the LLM must be merged — done by
keeping the most severe version (a bug that once downgraded a critical issue is now covered by a regression test).

## 6. Iterative research with a deterministic gate

**Decision.** Two loops back to research: an evidence gate after each round (coverage) and a quality gate after
critique. Both are bounded by iteration and tool budgets, and both can end in "proceed with limitations".

**Alternatives.** One-shot research; unbounded loops until a model says "done".

**Reason.** Research quality is uneven across subquestions; spending budget only on the gaps is efficient. Budgets in
state keep routing pure and termination guaranteed; writing limitations into the report keeps the output honest when
evidence is genuinely thin (the negative-control evaluation checks this).

**Tradeoffs.** Thresholds and weights are configuration that must be tuned per domain.

## 7. Source provenance and machine-generated citations

**Decision.** Evidence carries `source_id, url, passage_id, chunk_index` and a verbatim quote that is verified
against the passage. The writer attaches evidence IDs; citation numbers, URLs and the bibliography are generated in
code; every sentence is verified (existence, lexical relevance, numbers) and repaired or removed.

**Alternatives.** Let the model write citations; verify only that citations exist.

**Reason.** Fabricated or mismatched citations are the most damaging failure of research assistants. Making them
impossible by construction (rather than unlikely by prompting) is the core design goal. Number matching is a cheap
guard against the most common hallucination (wrong figures).

**Tradeoffs.** Lexical checks can miss subtle misstatements and occasionally remove a valid paraphrase; removals are
listed in the report for transparency.

## 8. Parallel workers via `Send()` into a subgraph

**Decision.** `query_generation` fans out one `Send("research_worker", WorkerInput)` per task; the worker is a
compiled subgraph with its own state and an `output_schema`; results merge through reducers.

**Alternatives.** Sequential loop over subquestions; `asyncio.gather` inside one node; one big agent researching
everything.

**Reason.** Subquestions are independent, so concurrency cuts latency roughly by the number of workers. Using the
graph (rather than `gather` inside a node) keeps each worker's steps visible in traces and streams, checkpointable,
individually instrumented and individually fault-isolated.

**Tradeoffs.** Merge semantics must be designed (content-addressed IDs, commutative reducers); interleaved events
need namespacing in the UI; external rate limits (arXiv) can serialise what the graph parallelises.

## 9. Checkpointing, `interrupt()` and drain/resume

**Decision.** Use LangGraph checkpointers (`durability="sync"`) for all workflow state; human approval via
`interrupt()` + `Command(resume=…)`; cooperative shutdown via `RunControl.request_drain()`; resume on startup.

**Alternatives.** A boolean "approved" flag with polling; re-running from scratch after a crash.

**Reason.** A real pause point means the approval can come minutes or days later, after deploys or restarts, without
holding a process open. Draining at a superstep boundary means a restart loses at most the in-flight superstep.

**Tradeoffs.** Checkpoint storage grows with steps (prune after completion in production); everything in state must
be serialisable — solved with Pydantic models plus a strict deserialisation allowlist.

## 10. A read model projected with the graph's own reducers

**Decision.** The checkpointer is the source of truth; a relational read model (runs, sources, evidence, findings,
reports, events) is maintained by applying stream updates with the same reducer functions.

**Alternatives.** Query checkpoints directly from the API; have nodes write to the database.

**Reason.** Checkpoints are optimised for resuming, not for listing sources or paginating evidence. Nodes stay free of
persistence concerns (and testable without a database). Reusing the reducers makes the projection consistent with
graph semantics by construction.

**Tradeoffs.** Eventual consistency between a node finishing and the read model updating (milliseconds); two copies
of some data.

## 11. A mock model that speaks the real tool-calling protocol

**Decision.** `MockChatModel` subclasses `BaseChatModel` and implements `bind_tools`; a deterministic responder
produces grounded output from the same prompt context a real model receives. Failures are injected through explicit
fault points.

**Alternatives.** Stub agent functions in tests; record/replay real responses.

**Reason.** The project must be runnable and testable without keys, and the tests should exercise the real
structured-output path (including validation failures and repairs), the real `ToolNode`, real retries and real
routing. Record/replay would be brittle with parallel, order-dependent calls.

**Tradeoffs.** The responder is heuristic; demo output demonstrates mechanics, not writing quality. This is stated
wherever demo results appear.

## 12. Offline hashing embeddings by default

**Decision.** Deterministic feature-hashing embeddings (unigrams + bigrams) unless `EMBEDDING_PROVIDER=openai`.

**Alternatives.** Require an embedding API; bundle a local transformer model.

**Reason.** Zero-setup, reproducible tests and demos; no heavyweight dependencies (PyTorch) for a portfolio checkout.
Retrieval quality is backstopped by hybrid reranking.

**Tradeoffs.** Lexical, not semantic, similarity. Production deployments should configure a real embedding model.

## 13. Fetching is a pipeline step, not a model tool

**Decision.** The model can search and pick among results; only pipeline code fetches, and only URLs that a search
provider returned, after SSRF validation on every redirect hop. The model never sees URLs.

**Alternatives.** Give the agent a `fetch_url` tool.

**Reason.** A fetch tool turns prompt injection in any retrieved page into SSRF or data-exfiltration attempts.
Removing the capability is stronger than filtering its use.

**Tradeoffs.** The agent cannot follow links it discovers inside documents.

## 14. Server-Sent Events for live progress

**Decision.** SSE with persisted, sequence-numbered events; replay then follow; `Last-Event-ID` resume.

**Alternatives.** WebSockets; client polling only.

**Reason.** Progress is one-directional; SSE works through proxies, reconnects automatically and is trivial to
consume and test. Persisted events make late joiners and reconnects lossless.

**Tradeoffs.** One open connection per viewer. Messages are unnamed SSE events (type in the payload) after a
workflow `error` event was found to collide with `EventSource`'s built-in `error` event.

## 15. Frozen domain models and content-addressed IDs

**Decision.** Domain models are immutable; sources and evidence get IDs derived from their content.

**Alternatives.** Mutable models; random UUIDs.

**Reason.** Immutability prevents accidental in-place mutation of shared state (a classic source of bugs with
parallel branches). Content addressing makes parallel discoveries of the same source or quote converge instead of
duplicating.

**Tradeoffs.** Updates use `model_copy(update=…)`; IDs change if the canonical form changes.

## 16. A diversity slot in source selection

**Decision.** Each worker fetches its top-ranked candidates plus the best candidate of a source type not yet
represented.

**Alternatives.** Pure ranking by authority and relevance.

**Reason.** Practitioner sources (blogs, forums, news) rarely outrank papers, but without them the system cannot
notice where practice disagrees with research. Quality scoring still down-weights them, and the critic flags findings
that rest on them.

**Tradeoffs.** One fetch per worker is spent on a likely lower-quality source.

## 17. Evaluation measures behaviour, with a negative control

**Decision.** The dataset specifies behavioural expectations (coverage, grounding, primary sources, contradictions,
limitations) instead of reference answers, and includes an off-topic question the corpus cannot answer.
LLM-as-judge is optional and reported separately.

**Alternatives.** Reference answers with similarity scoring; judge-only evaluation.

**Reason.** Research questions rarely have a single right answer, but good research behaviour is checkable. The
negative control caught a real failure mode (irrelevant evidence being stretched to answer an off-topic question)
that positive examples never would.

**Tradeoffs.** Behavioural checks do not measure prose quality; that is what the (clearly separated) judge is for.
