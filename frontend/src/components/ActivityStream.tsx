import { useEffect, useMemo, useRef, useState } from "react";

import type { Tone } from "../lib/format";
import type { EventType, WorkflowEvent } from "../types/api";
import { Card, Dot, cx } from "./ui";

const WORKER_NODES = new Set(["search_agent", "execute_tools", "source_processing", "passage_retrieval", "evidence_extraction"]);

const DOT: Record<EventType, Tone> = {
  run_status: "accent",
  node_started: "neutral",
  node_completed: "ok",
  progress: "neutral",
  tool: "neutral",
  warning: "warn",
  error: "bad",
  plan_ready: "warn",
  metrics: "neutral",
};

const TEXT: Partial<Record<EventType, string>> = {
  warning: "text-warn",
  error: "text-bad",
  plan_ready: "text-ink font-medium",
  run_status: "text-ink font-medium",
  node_started: "text-muted",
  node_completed: "text-muted",
  tool: "text-muted",
};

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <button
      type="button"
      aria-pressed={checked}
      onClick={() => onChange(!checked)}
      className={cx(
        "rounded px-1.5 py-0.5 text-[12px] transition",
        checked ? "bg-surface-2 text-ink ring-1 ring-line-strong" : "text-muted hover:text-ink",
      )}
    >
      {label}
    </button>
  );
}

export function ActivityStream({ events, connected }: { events: WorkflowEvent[]; connected: boolean }) {
  const [showLifecycle, setShowLifecycle] = useState(false);
  const [showTools, setShowTools] = useState(true);
  const [follow, setFollow] = useState(true);
  const scroller = useRef<HTMLDivElement>(null);

  const visible = useMemo(
    () =>
      events.filter((event) => {
        if (event.type === "tool") return showTools;
        if (event.type === "node_started" || event.type === "node_completed") {
          return showLifecycle || (event.type === "node_started" && !WORKER_NODES.has(event.node ?? ""));
        }
        return true;
      }),
    [events, showLifecycle, showTools],
  );

  useEffect(() => {
    if (follow && scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight;
  }, [visible, follow]);

  const start = events[0] ? new Date(events[0].timestamp).getTime() : 0;

  return (
    <Card
      title={
        <span className="inline-flex items-center gap-2">
          Activity
          <span className={cx("inline-flex items-center gap-1 font-mono text-[11px] font-normal", connected ? "text-ok" : "text-faint")}>
            <Dot tone={connected ? "ok" : "neutral"} pulse={connected} />
            {connected ? "live" : `${events.length} events`}
          </span>
        </span>
      }
      actions={
        <div className="flex items-center gap-1">
          <Toggle label="Tools" checked={showTools} onChange={setShowTools} />
          <Toggle label="All nodes" checked={showLifecycle} onChange={setShowLifecycle} />
          <Toggle label="Follow" checked={follow} onChange={setFollow} />
        </div>
      }
    >
      <div ref={scroller} className="scrollbar-thin h-[600px] overflow-y-auto py-1.5 font-mono text-[12px] leading-[1.55]">
        {visible.length === 0 && <p className="px-4 py-10 text-center font-sans text-[13px] text-muted">Waiting for workflow events…</p>}
        {visible.map((event) => {
          const subquestion = typeof event.data.subquestion_id === "string" ? event.data.subquestion_id : null;
          return (
            <div
              key={event.seq}
              className="grid grid-cols-[44px_10px_minmax(0,168px)_minmax(0,1fr)] items-baseline gap-x-2.5 px-3 py-[3px] hover:bg-surface-2"
            >
              <span className="text-right text-[11px] text-faint tabular-nums">
                {((new Date(event.timestamp).getTime() - start) / 1000).toFixed(1)}s
              </span>
              <Dot tone={DOT[event.type]} className="mt-[7px] self-start" />
              <span className="truncate text-[11.5px] text-muted" title={event.node ?? undefined}>
                {event.node ?? ""}
                {event.node && subquestion && event.type !== "progress" ? <span className="text-faint">·{subquestion}</span> : null}
              </span>
              <span className={cx("min-w-0 font-sans text-[12.5px] break-words", TEXT[event.type] ?? "text-ink-2")}>{event.message}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
