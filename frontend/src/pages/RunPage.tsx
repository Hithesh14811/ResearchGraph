import { Download, Loader2, Printer, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { ActivityStream } from "../components/ActivityStream";
import { ArtifactTabs, TABS, type Tab } from "../components/ArtifactTabs";
import { MetricsPanel } from "../components/MetricsPanel";
import { PlanReview } from "../components/PlanReview";
import { StageTrack } from "../components/StageTrack";
import { WorkflowGraph } from "../components/WorkflowGraph";
import { Button, StatusBadge, buttonClass } from "../components/ui";
import { useArtifacts } from "../hooks/useArtifacts";
import { useEventStream } from "../hooks/useEventStream";
import { useRun } from "../hooks/useRun";
import { humanize, relativeTime } from "../lib/format";
import { deriveWorkflow } from "../lib/workflow";
import { api } from "../services/api";
import { TERMINAL_STATUSES } from "../types/api";

const REFRESH_TRIGGERS = new Set(["source_evaluation", "synthesis", "critic", "final_review", "planner", "plan_review"]);

export function RunPage() {
  const { researchId } = useParams();
  const [refreshKey, setRefreshKey] = useState(0);
  const { run, error, refresh } = useRun(researchId, refreshKey);
  const { events, connected } = useEventStream(researchId);
  const [searchParams] = useSearchParams();
  // ?tab=full-report deep-links straight to a result tab.
  const [tab, setTab] = useState<Tab>(
    () => TABS.find((t) => t.toLowerCase().replace(/ /g, "-") === searchParams.get("tab")) ?? "Executive Summary",
  );
  const [cancelling, setCancelling] = useState(false);

  // Re-fetch run + artifacts whenever the graph reaches a milestone.
  const milestone = useMemo(
    () => events.filter((e) => e.type === "plan_ready" || e.type === "run_status" || (e.type === "node_completed" && REFRESH_TRIGGERS.has(e.node ?? ""))).length,
    [events],
  );
  useEffect(() => {
    if (milestone > 0) void refresh();
  }, [milestone, refresh]);

  const completed = run?.status === "completed";
  const artifacts = useArtifacts(researchId, `${milestone}:${run?.status ?? ""}`, completed);
  const view = useMemo(() => deriveWorkflow(events, run), [events, run]);

  useEffect(() => {
    if (run) document.title = `${run.question.slice(0, 60)}${run.question.length > 60 ? "…" : ""} — ResearchGraph`;
    return () => {
      document.title = "ResearchGraph";
    };
  }, [run]);

  if (error && !run) return <p className="py-24 text-center text-bad">{error}</p>;
  if (!run || !researchId)
    return (
      <p className="flex items-center justify-center gap-2 py-24 text-muted">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading run…
      </p>
    );

  const terminal = TERMINAL_STATUSES.includes(run.status);
  const plan = artifacts.plan?.plan ?? null;

  const cancel = async () => {
    setCancelling(true);
    try {
      await api.cancel(researchId);
      setRefreshKey((k) => k + 1);
    } finally {
      setCancelling(false);
    }
  };

  const exportPdf = () => {
    setTab("Full Report");
    window.setTimeout(() => window.print(), 150);
  };

  return (
    <div className="space-y-6">
      <div className="no-print space-y-5">
        <nav className="flex items-center gap-1.5 text-[13px] text-muted" aria-label="Breadcrumb">
          <Link to="/" className="hover:text-ink">
            Research
          </Link>
          <span className="text-faint">/</span>
          <span className="truncate font-mono text-[12px] text-ink-2">{run.id}</span>
        </nav>

        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0 max-w-4xl">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12.5px] text-muted">
              <StatusBadge status={run.status} />
              {run.current_stage && !terminal && (
                <span>
                  in <span className="text-ink-2">{humanize(run.current_stage)}</span>
                  {run.current_node && <span className="font-mono text-[11.5px] text-faint"> · {run.current_node}</span>}
                </span>
              )}
              <span>started {relativeTime(run.created_at)}</span>
              {run.failure_scenarios.length > 0 && <span className="text-warn">{run.failure_scenarios.length} simulated failures</span>}
            </div>
            <h1 className="mt-2 text-[21px] leading-snug font-semibold tracking-[-0.015em] text-ink">{run.question}</h1>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            {!terminal && (
              <Button variant="danger" onClick={cancel} disabled={cancelling}>
                {cancelling ? <Loader2 className="animate-spin" /> : <X />} Cancel
              </Button>
            )}
            {completed && (
              <>
                <a href={api.reportMarkdownUrl(researchId)} className={buttonClass("secondary")}>
                  <Download /> Markdown
                </a>
                <Button onClick={exportPdf}>
                  <Printer /> Print / PDF
                </Button>
              </>
            )}
          </div>
        </div>

        <StageTrack view={view} failed={run.status === "failed"} />
        {run.error && <p className="rounded-md bg-bad-soft px-3 py-2 text-[13px] text-bad">{run.error}</p>}
      </div>

      {run.status === "awaiting_approval" && plan && (
        <div className="no-print">
          <PlanReview researchId={researchId} plan={plan} onDecision={() => setRefreshKey((k) => k + 1)} />
        </div>
      )}

      <div className="no-print grid gap-5 xl:grid-cols-[300px_minmax(0,1fr)_320px]">
        <WorkflowGraph view={view} />
        <ActivityStream events={events} connected={connected} />
        <MetricsPanel run={run} threshold={artifacts.report?.quality?.threshold} />
      </div>

      {(completed || artifacts.evidence.length > 0) && <ArtifactTabs run={run} artifacts={artifacts} tab={tab} onTab={setTab} />}
    </div>
  );
}
