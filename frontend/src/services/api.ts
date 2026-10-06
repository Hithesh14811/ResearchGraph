import type {
  Evidence,
  Finding,
  Health,
  PlanResponse,
  Report,
  ResearchPlan,
  Run,
  RunList,
  RunMode,
  SourceItem,
  WorkflowEvent,
} from "../types/api";

const BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? "/api").replace(/\/$/, "");
const TOKEN = import.meta.env.VITE_API_TOKEN as string | undefined;
const LIVE_TOKEN_KEY = "rg-live-token";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

interface StoredLiveToken {
  token: string;
  expiresAt: number;
}

/** The token that unlocks live (real-model) runs; kept until it expires or the user locks. */
export const liveToken = {
  get(): string | null {
    try {
      const stored = JSON.parse(localStorage.getItem(LIVE_TOKEN_KEY) ?? "null") as StoredLiveToken | null;
      if (stored && stored.expiresAt * 1000 > Date.now()) return stored.token;
    } catch {
      /* storage unavailable or corrupt: treat as locked */
    }
    return null;
  },
  set(token: string, expiresAt: number) {
    try {
      localStorage.setItem(LIVE_TOKEN_KEY, JSON.stringify({ token, expiresAt } satisfies StoredLiveToken));
    } catch {
      /* not persisted: live mode lasts for this page view only */
    }
    memoryToken = token;
  },
  clear() {
    try {
      localStorage.removeItem(LIVE_TOKEN_KEY);
    } catch {
      /* nothing to clear */
    }
    memoryToken = null;
  },
};
let memoryToken: string | null = null;
const currentLiveToken = () => liveToken.get() ?? memoryToken;

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (init.body) headers["Content-Type"] = "application/json";
  if (TOKEN) headers.Authorization = `Bearer ${TOKEN}`;
  const live = currentLiveToken();
  if (live) headers["X-Live-Token"] = live;
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
  unlockLive: (password: string) => post<{ token: string; expires_at: number }>("/auth/live", { password }),
  listRuns: (limit = 50) => request<RunList>(`/research?limit=${limit}`),
  createRun: (question: string, autoApprove: boolean, failureScenarios: string[], mode: RunMode) =>
    post<Run>("/research", { question, auto_approve: autoApprove, failure_scenarios: failureScenarios, mode }),
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
  docsUrl: `${BASE}/docs`,
  reportMarkdownUrl: (id: string) => `${BASE}/research/${id}/report?format=markdown${TOKEN ? `&token=${encodeURIComponent(TOKEN)}` : ""}`,
  streamUrl: (id: string) => `${BASE}/research/${id}/stream${TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : ""}`,
};
