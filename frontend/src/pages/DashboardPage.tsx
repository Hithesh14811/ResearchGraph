import { useEffect, useState } from "react";

import { NewResearchForm } from "../components/NewResearchForm";
import { RunHistory } from "../components/RunHistory";
import { STAGES } from "../lib/workflow";
import { api } from "../services/api";
import { TERMINAL_STATUSES, type Health, type Run } from "../types/api";

const ACTIVE_POLL_MS = 2500;
const IDLE_POLL_MS = 15000;

function PipelineCard({ demoMode }: { demoMode: boolean }) {
  return (
    <section className="rounded-lg border border-line bg-surface">
      <header className="flex min-h-11 items-center border-b border-line px-4 py-2">
        <h2 className="text-[13px] font-medium text-ink">How a run works</h2>
      </header>
      <ol className="divide-y divide-line">
        {STAGES.map((stage, index) => (
          <li key={stage.key} className="grid grid-cols-[28px_minmax(0,1fr)] gap-x-2 px-4 py-2">
            <span className="pt-px font-mono text-[11.5px] text-faint tabular-nums">{String(index + 1).padStart(2, "0")}</span>
            <p className="text-[13px] leading-snug text-ink-2">
              <span className="font-medium text-ink">{stage.label}.</span> {stage.summary}
            </p>
          </li>
        ))}
      </ol>
      {demoMode && (
        <p className="border-t border-line bg-surface-2 px-4 py-3 text-[12.5px] leading-relaxed text-muted">
          <span className="font-medium text-ink-2">Demo mode.</span> A deterministic mock model researches a synthetic, clearly labelled
          corpus. Set <code className="font-mono text-[11.5px] text-ink-2">LLM_PROVIDER</code> and{" "}
          <code className="font-mono text-[11.5px] text-ink-2">SEARCH_PROVIDER</code> to research the live web.
        </p>
      )}
    </section>
  );
}

export function DashboardPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    void api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  // Self-scheduling poll: fast while a run is executing, slow otherwise.
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const load = async () => {
      let delay = IDLE_POLL_MS;
      try {
        const list = await api.listRuns();
        if (!alive) return;
        setRuns(list.items);
        setTotal(list.total);
        const executing = list.items.some((r) => !TERMINAL_STATUSES.includes(r.status) && r.status !== "awaiting_approval");
        if (executing) delay = ACTIVE_POLL_MS;
      } catch {
        /* keep the last list; retry on the idle cadence */
      } finally {
        if (alive) {
          setLoading(false);
          timer = window.setTimeout(() => void load(), delay);
        }
      }
    };
    void load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  const demoMode = health?.llm_provider === "mock";

  return (
    <div className="space-y-8">
      <div className="max-w-2xl">
        <h1 className="text-[24px] leading-tight font-semibold tracking-[-0.02em] text-ink">Research</h1>
        <p className="mt-1.5 text-[14px] leading-relaxed text-muted">
          ResearchGraph plans a question, investigates it in parallel, critiques its own findings, and publishes a report in which
          every sentence is traced to a verified quote from a stored source.
        </p>
      </div>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <NewResearchForm demoMode={demoMode} />
        <PipelineCard demoMode={demoMode} />
      </div>
      <RunHistory runs={runs} loading={loading} total={total} />
    </div>
  );
}
