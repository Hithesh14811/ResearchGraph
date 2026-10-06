import { Link } from "react-router-dom";

import { relativeTime } from "../lib/format";
import type { Run } from "../types/api";
import { EmptyState, StatusBadge } from "./ui";

const COLUMNS = "grid grid-cols-[minmax(0,1fr)_auto] md:grid-cols-[minmax(0,1fr)_150px_64px_72px_64px_88px] items-center gap-x-4";

export function RunHistory({ runs, loading, total }: { runs: Run[]; loading: boolean; total: number }) {
  return (
    <section className="rounded-lg border border-line bg-surface">
      <header className="flex min-h-11 items-center justify-between border-b border-line px-4 py-2">
        <h2 className="text-[13px] font-medium text-ink">Runs</h2>
        {total > 0 && <span className="font-mono text-[12px] text-muted">{total}</span>}
      </header>
      {runs.length === 0 ? (
        <EmptyState>{loading ? "Loading…" : "No runs yet. Ask a question above to start one."}</EmptyState>
      ) : (
        <div role="table" aria-label="Research runs">
          <div role="row" className={`${COLUMNS} border-b border-line bg-surface-2 px-4 py-2 text-[12px] text-muted`}>
            <span role="columnheader">Question</span>
            <span role="columnheader">Status</span>
            <span role="columnheader" className="hidden text-right md:block">Sources</span>
            <span role="columnheader" className="hidden text-right md:block">Evidence</span>
            <span role="columnheader" className="hidden text-right md:block">Quality</span>
            <span role="columnheader" className="hidden text-right md:block">Started</span>
          </div>
          {runs.map((run) => (
            <Link
              key={run.id}
              to={`/runs/${run.id}`}
              role="row"
              className={`${COLUMNS} border-b border-line px-4 py-3 last:border-b-0 hover:bg-surface-2`}
            >
              <span role="cell" className="min-w-0">
                <span className="block truncate text-[13.5px] text-ink">{run.question}</span>
                <span className="mt-0.5 block font-mono text-[11px] text-faint">{run.id}</span>
              </span>
              <span role="cell">
                <StatusBadge status={run.status} />
              </span>
              <Num value={run.metrics.sources_found} />
              <Num value={run.metrics.evidence_items} />
              <Num value={run.metrics.quality_score} digits={2} />
              <span role="cell" className="hidden text-right text-[12.5px] text-muted md:block">
                {relativeTime(run.created_at)}
              </span>
            </Link>
          ))}
        </div>
      )}
    </section>
  );
}

function Num({ value, digits }: { value: number | null | undefined; digits?: number }) {
  return (
    <span role="cell" className="hidden text-right font-mono text-[12.5px] text-ink-2 tabular-nums md:block">
      {value == null ? <span className="text-faint">—</span> : digits ? value.toFixed(digits) : value}
    </span>
  );
}
