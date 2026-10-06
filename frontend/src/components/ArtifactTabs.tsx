import { ArrowUpRight, Check, ChevronRight } from "lucide-react";
import { Fragment, useMemo, useState, type ReactNode } from "react";

import type { Artifacts } from "../hooks/useArtifacts";
import { SEVERITY_TONE, SOURCE_TYPE_LABEL, TIER_TONE, humanize } from "../lib/format";
import type { Run } from "../types/api";
import { ExecutiveSummary, FullReport } from "./ReportView";
import { Dot, EmptyState, Meter, Pill, SectionLabel, Tag, cx } from "./ui";

export const TABS = ["Executive Summary", "Full Report", "Evidence", "Sources", "Critique", "Research Plan", "Workflow"] as const;
export type Tab = (typeof TABS)[number];

function FilterButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cx(
        "rounded-md px-2 py-1 font-mono text-[12px] transition",
        active ? "bg-ink text-surface" : "text-muted ring-1 ring-line hover:text-ink hover:ring-line-strong",
      )}
    >
      {children}
    </button>
  );
}

function EvidenceTab({ artifacts }: { artifacts: Artifacts }) {
  const [filter, setFilter] = useState<string>("all");
  const subquestions = artifacts.plan?.plan?.subquestions ?? [];
  const items = artifacts.evidence.filter((e) => filter === "all" || e.subquestion_id === filter);
  if (artifacts.evidence.length === 0) return <EmptyState>No evidence extracted yet.</EmptyState>;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1.5">
        {["all", ...subquestions.map((s) => s.id)].map((id) => (
          <FilterButton key={id} active={filter === id} onClick={() => setFilter(id)}>
            {id === "all" ? `All ${artifacts.evidence.length}` : `${id} ${artifacts.evidence.filter((e) => e.subquestion_id === id).length}`}
          </FilterButton>
        ))}
      </div>
      {filter !== "all" && <p className="text-[14px] text-ink-2">{subquestions.find((s) => s.id === filter)?.question}</p>}
      <div className="grid gap-3 lg:grid-cols-2">
        {items.map((e) => (
          <article key={e.id} className="flex flex-col rounded-md border border-line p-4">
            <div className="mb-2.5 flex flex-wrap items-center gap-1.5">
              <Tag>{e.subquestion_id}</Tag>
              <Tag>{humanize(e.evidence_type)}</Tag>
              {e.quote_verified && (
                <span className="inline-flex items-center gap-1 text-[12px] text-ok">
                  <Check className="h-3 w-3" /> quote verified
                </span>
              )}
              <span className="ml-auto font-mono text-[10.5px] text-faint">{e.id}</span>
            </div>
            <p className="text-[14px] leading-relaxed text-ink">{e.claim}</p>
            <blockquote className="mt-2.5 border-l-2 border-line-strong pl-3 font-serif text-[14px] leading-relaxed text-muted italic">
              “{e.supporting_text}”
            </blockquote>
            <div className="mt-auto flex flex-wrap items-end justify-between gap-x-4 gap-y-1 pt-3">
              <a href={e.source_url} target="_blank" rel="noreferrer" className="inline-flex min-w-0 items-center gap-0.5 text-[12.5px] text-accent hover:underline">
                <span className="truncate">{e.source_title}</span>
                <ArrowUpRight className="h-3 w-3 shrink-0" />
              </a>
              <span className="font-mono text-[11px] text-muted">
                conf {e.confidence.toFixed(2)} · rel {e.relevance.toFixed(2)}
              </span>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

const FACTORS = ["authority", "recency", "relevance", "primary_source", "methodology", "citation_signal"] as const;

function SourcesTab({ artifacts }: { artifacts: Artifacts }) {
  const [open, setOpen] = useState<string | null>(null);
  if (artifacts.sources.length === 0) return <EmptyState>No sources retrieved yet.</EmptyState>;
  return (
    <div>
      <p className="mb-3 text-[13px] text-muted">
        Six-factor quality score per source. Search rank is never used as a proxy for credibility. Select a row for the breakdown.
      </p>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full min-w-[760px] text-left text-[13px]">
          <thead className="bg-surface-2 text-[12px] text-muted">
            <tr className="border-b border-line">
              <th className="py-2 pr-3 pl-3 font-normal">Source</th>
              <th className="px-3 font-normal">Type</th>
              <th className="w-36 px-3 font-normal">Score</th>
              <th className="px-3 font-normal">Content</th>
              <th className="px-3 font-normal">Used for</th>
            </tr>
          </thead>
          <tbody>
            {artifacts.sources.map(({ source, quality }) => {
              const expanded = open === source.id;
              return (
                <Fragment key={source.id}>
                  <tr
                    onClick={() => setOpen(expanded ? null : source.id)}
                    className={cx("cursor-pointer border-b border-line last:border-b-0 hover:bg-surface-2", expanded && "bg-surface-2")}
                  >
                    <td className="max-w-md py-2.5 pr-3 pl-3">
                      <div className="flex items-start gap-1.5">
                        <ChevronRight className={cx("mt-0.5 h-3.5 w-3.5 shrink-0 text-faint transition-transform", expanded && "rotate-90")} />
                        <div className="min-w-0">
                          <a href={source.url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()} className="text-ink hover:text-accent hover:underline">
                            {source.title}
                          </a>
                          <p className="truncate font-mono text-[11px] text-faint">{source.url}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 text-[12.5px] text-ink-2">{quality ? SOURCE_TYPE_LABEL[quality.source_type] : "—"}</td>
                    <td className="px-3">
                      {quality && (
                        <div>
                          <div className="flex items-center gap-1.5 font-mono text-[12px] text-ink">
                            <Dot tone={TIER_TONE[quality.tier]} />
                            {quality.overall.toFixed(2)}
                            <span className="font-sans text-[11.5px] text-muted">{quality.tier}</span>
                          </div>
                          <Meter value={quality.overall} className="mt-1.5" />
                        </div>
                      )}
                    </td>
                    <td className="px-3 text-[12.5px]">
                      {source.status === "failed" ? (
                        <span className="text-bad">failed</span>
                      ) : source.content_origin === "search_snippet" ? (
                        <span className="text-warn">snippet only</span>
                      ) : (
                        <span className="text-ink-2">
                          {source.document_format ?? "—"} · {source.chunk_count} chunk{source.chunk_count === 1 ? "" : "s"}
                        </span>
                      )}
                    </td>
                    <td className="px-3 font-mono text-[12px] text-muted">{source.subquestion_ids.join(" ")}</td>
                  </tr>
                  {expanded && quality && (
                    <tr className="border-b border-line bg-surface-2">
                      <td colSpan={5} className="px-3 pt-1 pb-4 pl-8">
                        <div className="grid gap-6 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
                          <div className="grid grid-cols-2 gap-x-5 gap-y-2.5">
                            {FACTORS.map((k) => (
                              <div key={k} className="text-[12px] text-muted">
                                <div className="flex justify-between">
                                  <span>{humanize(k)}</span>
                                  <span className="font-mono text-ink-2">{quality[k].toFixed(2)}</span>
                                </div>
                                <Meter value={quality[k]} className="mt-1" />
                              </div>
                            ))}
                          </div>
                          <ul className="list-disc space-y-1 pl-4 text-[12.5px] leading-relaxed text-ink-2 marker:text-faint">
                            {quality.rationale.map((r) => (
                              <li key={r}>{r}</li>
                            ))}
                            {source.error && <li className="text-warn">{source.error}</li>}
                          </ul>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const VERDICT_TONE = { acceptable: "ok", needs_revision: "warn", major_problems: "bad" } as const;

function CritiqueTab({ artifacts }: { artifacts: Artifacts }) {
  const report = artifacts.report;
  const critique = report?.critique;
  const quality = report?.quality;
  if (!critique || !quality) return <EmptyState>The critique is available when the run completes.</EmptyState>;
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div className="space-y-5">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[14px] font-medium text-ink">Final critique</span>
            <Pill tone={VERDICT_TONE[critique.verdict]}>{humanize(critique.verdict)}</Pill>
            <span className="font-mono text-[11.5px] text-faint">iteration {critique.iteration}</span>
          </div>
          <p className="mt-1.5 text-[14px] leading-relaxed text-ink-2">{critique.summary}</p>
        </div>

        {critique.issues.length === 0 ? (
          <p className="rounded-md border border-line px-4 py-3 text-[13px] text-muted">No open issues in the final critique.</p>
        ) : (
          <ul className="divide-y divide-line rounded-md border border-line">
            {critique.issues.map((issue) => (
              <li key={issue.id} className="p-4">
                <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
                  <Pill tone={SEVERITY_TONE[issue.severity]}>{issue.severity}</Pill>
                  <Tag>{humanize(issue.issue_type)}</Tag>
                  <span className="text-[12px] text-muted">
                    {issue.origin === "rule" ? "rule-based check" : "model review"}
                    {issue.subquestion_id && ` · ${issue.subquestion_id}`}
                  </span>
                </div>
                <p className="text-[14px] text-ink">{issue.description}</p>
                {issue.suggested_action && <p className="mt-1 text-[13px] text-muted">→ {issue.suggested_action}</p>}
              </li>
            ))}
          </ul>
        )}

        {critique.strengths.length > 0 && (
          <div>
            <SectionLabel>Strengths</SectionLabel>
            <ul className="mt-1.5 list-disc space-y-1 pl-5 text-[13px] text-ink-2 marker:text-faint">
              {critique.strengths.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        )}

        {report.contradictions.length > 0 && (
          <div>
            <SectionLabel className="mb-2">Contradictions detected</SectionLabel>
            <ul className="divide-y divide-line rounded-md border border-line">
              {report.contradictions.map((c) => (
                <li key={c.id} className="flex items-start gap-2 p-3 text-[13.5px] text-ink-2">
                  <Pill tone={c.severity === "major" ? "warn" : "neutral"} className="mt-px">
                    {c.severity}
                  </Pill>
                  <span>{c.description}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <aside className="h-fit rounded-md border border-line">
        <div className="border-b border-line px-4 py-3">
          <SectionLabel>Quality gate decision</SectionLabel>
          <div className="mt-1 flex items-baseline justify-between gap-3">
            <span className="font-mono text-[28px] leading-none text-ink tabular-nums">{quality.score.toFixed(2)}</span>
            <Pill tone={quality.decision === "proceed" ? "ok" : "warn"}>{humanize(quality.decision)}</Pill>
          </div>
          <p className="mt-1.5 font-mono text-[11.5px] text-muted">threshold {quality.threshold.toFixed(2)}</p>
        </div>
        <div className="space-y-3 px-4 py-4">
          {Object.entries(quality.components).map(([name, value]) => (
            <div key={name} className="text-[12.5px]">
              <div className="flex justify-between text-muted">
                <span>{humanize(name)}</span>
                <span className="font-mono text-ink-2 tabular-nums">{value.toFixed(2)}</span>
              </div>
              <Meter value={value} className="mt-1" />
            </div>
          ))}
          {quality.blocking_issues.length > 0 && (
            <ul className="list-disc space-y-0.5 pt-1 pl-4 text-[12.5px] text-warn">
              {quality.blocking_issues.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </div>
  );
}

function PlanTab({ artifacts }: { artifacts: Artifacts }) {
  const plan = artifacts.plan?.plan;
  if (!plan) return <EmptyState>No plan yet.</EmptyState>;
  return (
    <div className="space-y-6">
      <div className="grid gap-5 md:grid-cols-2">
        <div>
          <SectionLabel>Objective · plan v{plan.version}</SectionLabel>
          <p className="mt-1 text-[14px] leading-relaxed text-ink">{plan.objective}</p>
        </div>
        <div>
          <SectionLabel>Scope</SectionLabel>
          <p className="mt-1 text-[14px] leading-relaxed text-ink-2">{plan.scope}</p>
        </div>
      </div>
      <ol className="divide-y divide-line rounded-md border border-line">
        {plan.subquestions.map((sq) => {
          const evidenceCount = artifacts.evidence.filter((e) => e.subquestion_id === sq.id).length;
          const findings = artifacts.findings.filter((f) => f.subquestion_id === sq.id);
          return (
            <li key={sq.id} className="grid grid-cols-[40px_minmax(0,1fr)] gap-x-2 p-4">
              <span className="pt-px font-mono text-[12px] text-muted">{sq.id}</span>
              <div>
                <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                  <p className="text-[14px] text-ink">{sq.question}</p>
                  <span className="font-mono text-[11.5px] text-muted">
                    {evidenceCount} evidence · {findings.length} findings · {sq.preferred_sources}
                  </span>
                </div>
                {sq.rationale && <p className="mt-1 text-[13px] text-muted">{sq.rationale}</p>}
                {sq.search_queries.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {sq.search_queries.map((q) => (
                      <Tag key={q.query}>{q.query}</Tag>
                    ))}
                  </div>
                )}
                {findings.length > 0 && (
                  <ul className="mt-3 space-y-1.5 text-[13px] text-ink-2">
                    {findings.map((f) => (
                      <li key={f.id} className="grid grid-cols-[72px_minmax(0,1fr)] gap-x-2">
                        <span className={cx("font-mono text-[11.5px]", f.confidence === "high" ? "text-ok" : f.confidence === "moderate" ? "text-ink-2" : "text-warn")}>
                          {f.confidence}
                        </span>
                        <span>{f.statement}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function WorkflowTab({ run }: { run: Run }) {
  const latency = useMemo(() => Object.entries(run.metrics.node_latency_ms ?? {}).sort((a, b) => b[1] - a[1]), [run.metrics.node_latency_ms]);
  const max = latency[0]?.[1] ?? 1;
  const details: Array<[string, string]> = [
    ["Research ID", run.id],
    ["Iterations", String(run.metrics.iterations ?? run.iteration)],
    ["Tool calls", String(run.metrics.tool_calls_used ?? 0)],
    ["Recoverable errors", String(run.metrics.errors ?? 0)],
    ["LLM calls", `${run.usage.llm_calls ?? 0} (${run.usage.llm_errors ?? 0} failed attempts)`],
    ["Tokens in / out", `${(run.usage.input_tokens ?? 0).toLocaleString()} / ${(run.usage.output_tokens ?? 0).toLocaleString()}`],
    ["Plan approval", run.auto_approve ? "auto-approved" : "human reviewed"],
  ];
  if (run.failure_scenarios.length > 0) details.push(["Simulated failures", run.failure_scenarios.map(humanize).join(", ")]);
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div>
        <SectionLabel className="mb-3">Cumulative node latency — all iterations and workers</SectionLabel>
        {latency.length === 0 ? (
          <EmptyState>No node timings yet.</EmptyState>
        ) : (
          <div className="space-y-2">
            {latency.map(([node, ms]) => (
              <div key={node} className="grid grid-cols-[minmax(0,170px)_minmax(0,1fr)_64px] items-center gap-3 text-[12px]">
                <span className="truncate font-mono text-ink-2">{node}</span>
                <Meter value={ms / max} />
                <span className="text-right font-mono text-muted tabular-nums">{ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div>
        <SectionLabel className="mb-3">Run details</SectionLabel>
        <dl className="divide-y divide-line rounded-md border border-line text-[13px]">
          {details.map(([label, value]) => (
            <div key={label} className="grid grid-cols-[150px_minmax(0,1fr)] gap-3 px-3 py-2">
              <dt className="text-muted">{label}</dt>
              <dd className="font-mono text-[12.5px] break-all text-ink-2">{value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-[12.5px] text-muted">
          The compiled graph, including the worker subgraph, is served as Mermaid at <code className="font-mono text-ink-2">GET /graph</code>.
        </p>
      </div>
    </div>
  );
}

export function ArtifactTabs({ run, artifacts, tab, onTab }: { run: Run; artifacts: Artifacts; tab: Tab; onTab: (tab: Tab) => void }) {
  const report = artifacts.report;
  const counts: Partial<Record<Tab, number>> = {
    Evidence: artifacts.evidence.length,
    Sources: artifacts.sources.length,
    Critique: report?.critique?.issues.length,
  };
  return (
    <section className="rounded-lg border border-line bg-surface">
      <nav className="no-print scrollbar-thin flex gap-5 overflow-x-auto border-b border-line px-5" role="tablist">
        {TABS.map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={tab === name}
            onClick={() => onTab(name)}
            className={cx(
              "-mb-px shrink-0 border-b-2 py-3 text-[13px] transition-colors",
              tab === name ? "border-ink font-medium text-ink" : "border-transparent text-muted hover:text-ink",
            )}
          >
            {name}
            {counts[name] !== undefined && counts[name] > 0 && <span className="ml-1.5 font-mono text-[11px] text-faint">{counts[name]}</span>}
          </button>
        ))}
      </nav>
      <div className="p-5 sm:p-8">
        {tab === "Executive Summary" &&
          (report ? <ExecutiveSummary report={report} /> : <EmptyState>The report is available when the run completes.</EmptyState>)}
        {tab === "Full Report" && (report ? <FullReport report={report} /> : <EmptyState>The report is available when the run completes.</EmptyState>)}
        {tab === "Evidence" && <EvidenceTab artifacts={artifacts} />}
        {tab === "Sources" && <SourcesTab artifacts={artifacts} />}
        {tab === "Critique" && <CritiqueTab artifacts={artifacts} />}
        {tab === "Research Plan" && <PlanTab artifacts={artifacts} />}
        {tab === "Workflow" && <WorkflowTab run={run} />}
      </div>
    </section>
  );
}
