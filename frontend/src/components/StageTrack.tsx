import { STAGES, type WorkflowView } from "../lib/workflow";
import { cx } from "./ui";

const BAR = { pending: "bg-line", skipped: "bg-line", done: "bg-ink-2", active: "bg-accent pulse-dot" } as const;
const LABEL = { pending: "text-faint", skipped: "text-faint", done: "text-ink-2", active: "text-accent font-medium" } as const;

/** Where the run is in the workflow: one segment per stage, with loop counts when a stage repeated. */
export function StageTrack({ view, failed }: { view: WorkflowView; failed?: boolean }) {
  const lastVisited = STAGES.reduce((last, stage, index) => (view.visits[stage.key] > 0 ? index : last), -1);
  return (
    <ol className="grid grid-cols-8 gap-1" aria-label="Workflow progress">
      {STAGES.map((stage, index) => {
        const state = view.stages[stage.key];
        const failedHere = failed && index === lastVisited;
        const visits = view.visits[stage.key];
        return (
          <li key={stage.key} className="min-w-0" aria-current={state === "active" ? "step" : undefined}>
            <div className={cx("h-[3px] rounded-full", failedHere ? "bg-bad" : BAR[state])} />
            <p className={cx("mt-1.5 hidden truncate text-[12px] sm:block", failedHere ? "text-bad" : LABEL[state])}>
              {stage.label}
              {visits > 1 && <span className="ml-1 font-mono text-[10.5px] text-muted">×{visits}</span>}
            </p>
          </li>
        );
      })}
    </ol>
  );
}
