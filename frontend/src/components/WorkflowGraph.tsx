import { Check, Loader2 } from "lucide-react";
import type { CSSProperties } from "react";

import { STAGES, type StageKey, type StageState, type WorkflowView } from "../lib/workflow";
import { Card, cx } from "./ui";

const NODE_X = 40;
const NODE_W = 188;
const NODE_H = 42;
const GAP = 58;
const TOP = 46;
const CENTER = NODE_X + NODE_W / 2;

const nodeY = (index: number) => TOP + index * GAP;
const indexOf = (key: StageKey) => STAGES.findIndex((s) => s.key === key);

// SVG colours reference the theme tokens through `style`, so the graph follows light/dark.
const v = (name: string) => `var(--${name})`;
const STATE_STYLE: Record<StageState, { fill: string; stroke: string; text: string; sub: string }> = {
  pending: { fill: v("surface"), stroke: v("line-strong"), text: v("muted"), sub: v("faint") },
  active: { fill: v("accent-soft"), stroke: v("accent"), text: v("ink"), sub: v("accent") },
  done: { fill: v("surface"), stroke: v("line-strong"), text: v("ink"), sub: v("muted") },
  skipped: { fill: v("surface"), stroke: v("line"), text: v("faint"), sub: v("faint") },
};
const EDGE: CSSProperties = { stroke: v("line-strong"), fill: "none" };

/** A conditional back-edge drawn in the left or right gutter, labelled on its outer side. */
function LoopEdge({
  from,
  to,
  side,
  offset,
  count,
  label,
}: {
  from: StageKey;
  to: StageKey;
  side: "left" | "right";
  offset: number;
  count: number;
  label: string;
}) {
  const y1 = nodeY(indexOf(from)) + NODE_H / 2;
  const y2 = nodeY(indexOf(to)) + NODE_H / 2;
  const dir = side === "right" ? 1 : -1;
  const x = side === "right" ? NODE_X + NODE_W : NODE_X;
  const peak = x + dir * offset * 0.75; // furthest point of the cubic curve
  const labelX = side === "right" ? peak + 10 : peak - 3; // rotated glyphs extend toward -x
  const taken = count > 0;
  const color = taken ? v("accent") : v("faint");
  const mid = (y1 + y2) / 2;
  return (
    <g>
      <path
        d={`M ${x} ${y1} C ${x + dir * offset} ${y1}, ${x + dir * offset} ${y2}, ${x + dir * 6} ${y2}`}
        style={{ fill: "none", stroke: color }}
        strokeWidth={taken ? 1.5 : 1}
        strokeDasharray={taken ? undefined : "3 3"}
        markerEnd={taken ? "url(#rg-arrow-taken)" : "url(#rg-arrow-faint)"}
      />
      <text x={labelX} y={mid} style={{ fill: color }} fontSize="9.5" textAnchor="middle" transform={`rotate(-90 ${labelX} ${mid})`}>
        {label}
        {taken ? ` ×${count}` : ""}
      </text>
    </g>
  );
}

function Arrow({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: color }} />
    </marker>
  );
}

export function WorkflowGraph({ view }: { view: WorkflowView }) {
  const endY = nodeY(STAGES.length) + 6;
  const height = endY + 22;
  const verifyY = nodeY(indexOf("verification"));
  return (
    <Card title="Workflow" actions={<span className="font-mono text-[11px] text-faint">LangGraph</span>}>
      <div className="px-2 pt-2">
        <svg viewBox={`0 0 320 ${height}`} className="w-full font-sans" role="img" aria-label="Research workflow graph">
          <defs>
            <Arrow id="rg-arrow" color={v("line-strong")} />
            <Arrow id="rg-arrow-faint" color={v("faint")} />
            <Arrow id="rg-arrow-taken" color={v("accent")} />
          </defs>

          <circle cx={CENTER} cy={18} r={5} style={{ fill: v("surface"), stroke: v("faint") }} />
          <text x={CENTER + 12} y={21.5} style={{ fill: v("faint") }} fontSize="9.5" letterSpacing="0.06em">
            START
          </text>
          <line x1={CENTER} y1={23} x2={CENTER} y2={TOP - 2} style={EDGE} markerEnd="url(#rg-arrow)" />

          {STAGES.map((stage, index) => {
            const state = view.stages[stage.key];
            const style = STATE_STYLE[state];
            const y = nodeY(index);
            const visits = view.visits[stage.key];
            return (
              <g key={stage.key}>
                {index < STAGES.length - 1 && (
                  <line x1={CENTER} y1={y + NODE_H} x2={CENTER} y2={nodeY(index + 1) - 2} style={EDGE} markerEnd="url(#rg-arrow)" />
                )}
                <rect
                  x={NODE_X}
                  y={y}
                  width={NODE_W}
                  height={NODE_H}
                  rx={6}
                  style={{ fill: style.fill, stroke: style.stroke }}
                  strokeWidth={state === "active" ? 1.5 : 1}
                />
                <text x={NODE_X + 12} y={y + 17} style={{ fill: style.text }} fontSize="12.5" fontWeight={500}>
                  {stage.label}
                </text>
                <text x={NODE_X + 12} y={y + 32} style={{ fill: style.sub, fontFamily: "var(--mono)" }} fontSize="9">
                  {stage.description}
                </text>
                {visits > 1 && (
                  <text
                    x={NODE_X + NODE_W - (state === "pending" ? 10 : 26)}
                    y={y + 17}
                    style={{ fill: v("muted"), fontFamily: "var(--mono)" }}
                    fontSize="10"
                    textAnchor="end"
                  >
                    ×{visits}
                  </text>
                )}
                {state === "done" && (
                  <path
                    d={`M ${NODE_X + NODE_W - 19} ${y + 13} l 3 3 l 6 -6.5`}
                    style={{ fill: "none", stroke: v("ok") }}
                    strokeWidth={1.6}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                )}
                {state === "active" && <circle cx={NODE_X + NODE_W - 15} cy={y + 13} r={3.5} style={{ fill: v("accent") }} className="pulse-dot" />}
              </g>
            );
          })}

          <LoopEdge from="evidence" to="research" side="left" offset={26} count={view.evidenceLoops} label="insufficient" />
          <LoopEdge from="quality_gate" to="research" side="right" offset={52} count={view.qualityLoops} label="research more" />
          <path
            d={`M ${NODE_X} ${verifyY + 12} C ${NODE_X - 24} ${verifyY + 4}, ${NODE_X - 24} ${verifyY + 38}, ${NODE_X - 1} ${verifyY + 30}`}
            style={{ fill: "none", stroke: view.citationRepairs > 0 ? v("accent") : v("faint") }}
            strokeWidth={view.citationRepairs > 0 ? 1.5 : 1}
            strokeDasharray={view.citationRepairs > 0 ? undefined : "3 3"}
            markerEnd={view.citationRepairs > 0 ? "url(#rg-arrow-taken)" : "url(#rg-arrow-faint)"}
          />
          <text x={4} y={verifyY - 3} style={{ fill: view.citationRepairs > 0 ? v("accent") : v("faint") }} fontSize="9.5">
            repair{view.citationRepairs ? ` ×${view.citationRepairs}` : ""}
          </text>

          <line x1={CENTER} y1={nodeY(STAGES.length - 1) + NODE_H} x2={CENTER} y2={endY - 7} style={EDGE} markerEnd="url(#rg-arrow)" />
          <circle cx={CENTER} cy={endY} r={5} style={{ fill: v("surface"), stroke: v("faint") }} />
          <circle cx={CENTER} cy={endY} r={2.5} style={{ fill: v("faint") }} />
          <text x={CENTER + 12} y={endY + 3.5} style={{ fill: v("faint") }} fontSize="9.5" letterSpacing="0.06em">
            END
          </text>
        </svg>
      </div>
      <div className="border-t border-line px-4 py-3">
        <p className="mb-2 text-[12px] text-muted">Parallel workers{view.workers.length > 0 ? ` · ${view.workers.length}` : ""}</p>
        {view.workers.length === 0 ? (
          <p className="text-[12px] text-faint">Workers start after the plan is approved.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {view.workers.map((worker) => (
              <span
                key={`${worker.iteration}-${worker.subquestionId}`}
                className={cx(
                  "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[11px]",
                  worker.state === "done" ? "border-line text-ink-2" : "border-accent/40 bg-accent-soft text-accent",
                )}
              >
                {worker.state === "done" ? <Check className="h-3 w-3 text-ok" /> : <Loader2 className="h-3 w-3 animate-spin" />}
                {worker.subquestionId}
              </span>
            ))}
          </div>
        )}
        <p className="mt-3 text-[11.5px] leading-relaxed text-faint">Dashed edges are conditional routes; blue edges were taken.</p>
      </div>
    </Card>
  );
}
