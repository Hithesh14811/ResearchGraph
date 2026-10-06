import type { ReactNode } from "react";

import { humanize, number, percent, plural, seconds } from "../lib/format";
import type { Run } from "../types/api";
import { Card, Dot, Meter, cx } from "./ui";

function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: string }) {
  return (
    <div className="bg-surface px-4 py-3">
      <p className="text-[12px] text-muted">{label}</p>
      <p className={cx("mt-0.5 font-mono text-[19px] leading-tight text-ink tabular-nums", tone)}>{value}</p>
      {hint && <p className="mt-0.5 truncate text-[11.5px] text-faint">{hint}</p>}
    </div>
  );
}

export function MetricsPanel({ run, threshold, maxToolCalls = 60 }: { run: Run; threshold?: number; maxToolCalls?: number }) {
  const m = run.metrics;
  const report = m.report;
  const quality = m.quality_score ?? null;
  const faults = Object.entries(m.faults_triggered ?? {});
  return (
    <Card title="Metrics">
      <div className="grid grid-cols-2 gap-px bg-line">
        <Stat label="Sources" value={number(m.sources_found ?? 0)} hint={`${m.high_quality_sources ?? 0} high quality · ${m.sources_failed ?? 0} failed`} />
        <Stat label="Evidence" value={number(m.evidence_items ?? 0)} hint="quote-verified" />
        <Stat label="Findings" value={number(m.findings ?? 0)} hint={plural(m.contradictions ?? 0, "contradiction")} />
        <Stat
          label="Critic issues"
          value={number(m.critic_issues ?? 0)}
          tone={(m.critic_issues ?? 0) > 0 ? "text-warn" : undefined}
          hint={m.critic_verdict ? humanize(m.critic_verdict) : "—"}
        />
        <Stat label="Iterations" value={number(m.iterations ?? run.iteration)} hint="research rounds" />
        <Stat label="Tool calls" value={number(m.tool_calls_used ?? 0)} hint={`of ${maxToolCalls} budget`} />
        <Stat label="LLM calls" value={number(run.usage.llm_calls ?? m.llm_calls ?? 0)} hint={plural(run.usage.llm_errors ?? 0, "failed attempt")} />
        <Stat
          label="Tokens"
          value={number(run.usage.total_tokens ?? m.total_tokens ?? 0)}
          hint={run.usage.estimated_cost_usd != null ? `≈ $${run.usage.estimated_cost_usd.toFixed(4)}` : "no pricing set"}
        />
      </div>
      <div className="space-y-3.5 border-t border-line px-4 py-4">
        <div>
          <div className="mb-1.5 flex items-baseline justify-between text-[12.5px]">
            <span className="text-muted">Quality gate</span>
            <span className="font-mono text-ink tabular-nums">
              {quality != null ? quality.toFixed(2) : "—"}
              {threshold !== undefined && <span className="text-faint"> / {threshold.toFixed(2)}</span>}
            </span>
          </div>
          <Meter value={quality ?? 0} threshold={threshold} tone={quality != null && threshold !== undefined && quality < threshold ? "warn" : "ink"} />
        </div>
        {report && (
          <div>
            <div className="mb-1.5 flex items-baseline justify-between text-[12.5px]">
              <span className="text-muted">Citation coverage</span>
              <span className="font-mono text-ink tabular-nums">{percent(report.citation_coverage)}</span>
            </div>
            <Meter value={report.citation_coverage} />
            <p className="mt-2 text-[12px] text-muted">
              <span className="font-mono text-ink-2">{report.total_claims}</span> verified sentences ·{" "}
              <span className="font-mono text-ink-2">{report.removed_claims}</span> removed as unsupported
            </p>
          </div>
        )}
        <div className="flex justify-between text-[12.5px]">
          <span className="text-muted">Active time</span>
          <span className="font-mono text-ink tabular-nums">{seconds(m.active_seconds ?? null)}</span>
        </div>
      </div>
      {faults.length > 0 && (
        <div className="border-t border-line px-4 py-3">
          <p className="mb-1.5 flex items-center gap-1.5 text-[12px] text-ink-2">
            <Dot tone="warn" /> Recovered from simulated faults
          </p>
          <ul className="space-y-0.5 text-[12px] text-muted">
            {faults.map(([name, count]) => (
              <li key={name} className="flex justify-between">
                <span>{humanize(name)}</span>
                <span className="font-mono tabular-nums">×{count}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
