import type { Run, WorkflowEvent } from "../types/api";

export type StageKey = "plan" | "research" | "evidence" | "analysis" | "critic" | "quality_gate" | "report" | "verification";
export type StageState = "pending" | "active" | "done" | "skipped";

export interface Stage {
  key: StageKey;
  label: string;
  description: string;
  summary: string;
  nodes: string[];
}

/** UI stages of the LangGraph workflow (each groups one or more graph nodes). */
export const STAGES: Stage[] = [
  {
    key: "plan",
    label: "Plan",
    description: "intake · planner · human review",
    summary: "Break the question into subquestions and searches; pause for approval.",
    nodes: ["intake", "planner", "plan_review"],
  },
  {
    key: "research",
    label: "Research",
    description: "queries · parallel workers",
    summary: "One tool-using worker per subquestion, run in parallel.",
    nodes: ["query_generation", "search_agent", "execute_tools", "source_processing", "passage_retrieval"],
  },
  {
    key: "evidence",
    label: "Evidence",
    description: "extraction · source scoring",
    summary: "Extract quotes verified verbatim; score every source; loop if thin.",
    nodes: ["evidence_extraction", "source_evaluation"],
  },
  {
    key: "analysis",
    label: "Analysis",
    description: "contradictions · synthesis",
    summary: "Detect conflicting evidence and synthesise calibrated findings.",
    nodes: ["contradiction_detection", "synthesis"],
  },
  { key: "critic", label: "Critic", description: "independent review", summary: "Rules plus a separate model review the findings.", nodes: ["critic"] },
  {
    key: "quality_gate",
    label: "Quality gate",
    description: "deterministic decision",
    summary: "Publish, research more, or publish with stated limitations.",
    nodes: ["quality_gate"],
  },
  { key: "report", label: "Report", description: "cited writing", summary: "Write sentences that reference evidence IDs, never URLs.", nodes: ["report_writer"] },
  {
    key: "verification",
    label: "Verification",
    description: "citations · repair · review",
    summary: "Check every sentence against its evidence; repair or remove it.",
    nodes: ["citation_verification", "citation_repair", "final_review"],
  },
];

const NODE_TO_STAGE = new Map<string, StageKey>(STAGES.flatMap((s) => s.nodes.map((n) => [n, s.key] as const)));

// Nodes inside the parallel worker subgraph belong to the Research stage as a whole: their
// interleaved events must not flip the active stage back and forth.
const WORKER_NODES = new Set(["search_agent", "execute_tools", "source_processing", "passage_retrieval", "evidence_extraction"]);

// A stage "visit" is counted when its entry node starts (e.g. each research round starts at
// query_generation), so loop counts reflect real graph iterations.
const ENTRY_NODES = new Map<string, StageKey>([
  ["planner", "plan"],
  ["query_generation", "research"],
  ["source_evaluation", "evidence"],
  ["contradiction_detection", "analysis"],
  ["critic", "critic"],
  ["quality_gate", "quality_gate"],
  ["report_writer", "report"],
  ["citation_verification", "verification"],
]);

export interface WorkerState {
  subquestionId: string;
  state: "active" | "done";
  iteration: number;
}

export interface WorkflowView {
  stages: Record<StageKey, StageState>;
  visits: Record<StageKey, number>;
  evidenceLoops: number;
  qualityLoops: number;
  citationRepairs: number;
  workers: WorkerState[];
}

export function deriveWorkflow(events: WorkflowEvent[], run: Run | null): WorkflowView {
  const visits = Object.fromEntries(STAGES.map((s) => [s.key, 0])) as Record<StageKey, number>;
  const stages = Object.fromEntries(STAGES.map((s) => [s.key, "pending"])) as Record<StageKey, StageState>;
  let active: StageKey | null = null;
  let evidenceLoops = 0;
  let qualityLoops = 0;
  let citationRepairs = 0;
  let iteration = 0;
  const workers = new Map<string, WorkerState>();

  for (const event of events) {
    const node = event.node ?? "";
    const stage = WORKER_NODES.has(node) ? "research" : NODE_TO_STAGE.get(node);
    if (event.type === "node_started" && stage) {
      const entry = ENTRY_NODES.get(node);
      if (entry) visits[entry] += 1;
      if (!WORKER_NODES.has(node)) active = stage;
      if (event.node === "query_generation") iteration += 1;
      if (event.node === "citation_repair") citationRepairs += 1;
      const sq = event.data.subquestion_id;
      if (event.node === "search_agent" && typeof sq === "string") {
        workers.set(`${iteration}:${sq}`, { subquestionId: sq, state: "active", iteration });
      }
    }
    if (event.type === "node_completed" && event.node === "evidence_extraction") {
      const sq = event.data.subquestion_id;
      const key = `${iteration}:${String(sq)}`;
      const worker = workers.get(key);
      if (worker) workers.set(key, { ...worker, state: "done" });
    }
    if (event.type === "progress" && event.node === "source_evaluation" && event.message.includes("performing additional research")) {
      evidenceLoops += 1;
    }
    if (event.type === "progress" && event.node === "quality_gate" && event.message.includes("targeted research")) {
      qualityLoops += 1;
    }
  }

  for (const stage of STAGES) {
    if (visits[stage.key] > 0) stages[stage.key] = "done";
  }
  const status = run?.status;
  if (active && status && !["completed", "failed", "cancelled", "rejected"].includes(status)) {
    stages[active] = "active";
  }
  if (status === "awaiting_approval") stages.plan = "active";

  const latestIteration = Math.max(0, ...Array.from(workers.values()).map((w) => w.iteration));
  return {
    stages,
    visits,
    evidenceLoops,
    qualityLoops,
    citationRepairs,
    workers: Array.from(workers.values()).filter((w) => w.iteration === latestIteration),
  };
}
