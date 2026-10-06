import type {
  Evidence,
  Finding,
  Health,
  PlanResponse,
  Report,
  ResearchPlan,
  Run,
  RunList,
  SourceItem,
  WorkflowEvent,
} from "../types/api";

const BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? "/api").replace(/\/$/, "");
const TOKEN = import.meta.env.VITE_API_TOKEN as string | undefined;

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (init.body) headers["Content-Type"] = "application/json";
  if (TOKEN) headers.Authorization = `Bearer ${TOKEN}`;
  const response = await fetch(`${BASE}${path}`, { ...init, headers: { ...headers, ...(init.headers ?? {}) } });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health"),
  listRuns: (limit = 50) => request<RunList>(`/research?limit=${limit}`),
  createRun: (question: string, autoApprove: boolean, failureScenarios: string[]) =>
    post<Run>("/research", { question, auto_approve: autoApprove, failure_scenarios: failureScenarios }),
  getRun: (id: string) => request<Run>(`/research/${id}`),
  getPlan: (id: string) => request<PlanResponse>(`/research/${id}/plan`),
  approve: (id: string) => post<Run>(`/research/${id}/approve`),
  editPlan: (id: string, plan: ResearchPlan, approve: boolean) => post<Run>(`/research/${id}/edit-plan`, { plan, approve }),
  replan: (id: string, feedback: string) => post<Run>(`/research/${id}/replan`, { feedback }),
  cancel: (id: string) => post<Run>(`/research/${id}/cancel`),
  getSources: (id: string) => request<{ items: SourceItem[]; total: number }>(`/research/${id}/sources?limit=500`),
  getEvidence: (id: string) => request<{ items: Evidence[]; total: number }>(`/research/${id}/evidence?limit=500`),
  getFindings: (id: string) => request<{ items: Finding[] }>(`/research/${id}/findings`),
  getReport: (id: string) => request<Report>(`/research/${id}/report`),
  getEvents: (id: string) => request<{ items: WorkflowEvent[] }>(`/research/${id}/events`),
  graph: () => request<{ mermaid: string }>("/graph"),
  reportMarkdownUrl: (id: string) => `${BASE}/research/${id}/report?format=markdown${TOKEN ? `&token=${encodeURIComponent(TOKEN)}` : ""}`,
  streamUrl: (id: string) => `${BASE}/research/${id}/stream${TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : ""}`,
};
