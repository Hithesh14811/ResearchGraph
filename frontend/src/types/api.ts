// Types mirroring the FastAPI response models (backend/researchgraph/schemas).

export type RunStatus =
  | "pending"
  | "running"
  | "awaiting_approval"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted"
  | "rejected";

export type RunMode = "demo" | "live";

export const TERMINAL_STATUSES: RunStatus[] = ["completed", "failed", "cancelled", "rejected"];

export interface ReportMetrics {
  total_claims: number;
  cited_claims: number;
  citation_coverage: number;
  unsupported_claim_rate: number;
  removed_claims: number;
  sources_cited: number;
  mean_cited_source_quality: number;
  evidence_items: number;
  research_iterations: number;
}

export interface RunMetrics {
  sources_found?: number;
  sources_processed?: number;
  sources_failed?: number;
  sources_degraded?: number;
  high_quality_sources?: number;
  evidence_items?: number;
  findings?: number;
  contradictions?: number;
  critic_issues?: number;
  critic_verdict?: string | null;
  quality_score?: number | null;
  iterations?: number;
  tool_calls_used?: number;
  errors?: number;
  active_seconds?: number;
  node_latency_ms?: Record<string, number>;
  report?: ReportMetrics;
  llm_calls?: number;
  total_tokens?: number;
  estimated_cost_usd?: number | null;
  faults_triggered?: Record<string, number>;
}

export interface Usage {
  llm_calls?: number;
  llm_errors?: number;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  estimated_cost_usd?: number | null;
}

export interface Run {
  id: string;
  question: string;
  status: RunStatus;
  current_node: string | null;
  current_stage: string | null;
  progress: number;
  iteration: number;
  auto_approve: boolean;
  mode: RunMode;
  failure_scenarios: string[];
  metrics: RunMetrics;
  usage: Usage;
  error: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  awaiting_approval: boolean;
}

export interface RunList {
  items: Run[];
  total: number;
}

export type SourcePreference = "academic" | "web" | "mixed";

export interface SearchQuery {
  query: string;
  source_preference: SourcePreference;
}

export interface Subquestion {
  id: string;
  question: string;
  rationale: string;
  information_requirements: string[];
  search_queries: SearchQuery[];
  preferred_sources: SourcePreference;
}

export interface ResearchPlan {
  objective: string;
  scope: string;
  subquestions: Subquestion[];
  source_strategy: string[];
  stopping_criteria: string[];
  version: number;
}

export interface PlanResponse {
  research_id: string;
  status: RunStatus;
  editable: boolean;
  plan: ResearchPlan | null;
}

export type SourceType =
  | "primary_research"
  | "official_documentation"
  | "institutional"
  | "reputable_technical"
  | "news"
  | "blog"
  | "forum"
  | "unknown";

export interface Source {
  id: string;
  url: string;
  title: string;
  provider: string;
  authors: string[];
  published_date: string | null;
  venue: string | null;
  document_format: string | null;
  type_hint: SourceType | null;
  snippet: string;
  content_chars: number;
  chunk_count: number;
  content_origin: "full_text" | "search_snippet" | "none";
  subquestion_ids: string[];
  status: "discovered" | "processed" | "failed";
  error: string | null;
}

export interface SourceQuality {
  source_id: string;
  source_type: SourceType;
  authority: number;
  recency: number;
  relevance: number;
  primary_source: number;
  methodology: number;
  citation_signal: number;
  overall: number;
  tier: "high" | "medium" | "low";
  rationale: string[];
}

export interface SourceItem {
  source: Source;
  quality: SourceQuality | null;
}

export interface Evidence {
  id: string;
  subquestion_id: string;
  claim: string;
  supporting_text: string;
  source_id: string;
  source_url: string;
  source_title: string;
  publication: string | null;
  published_date: string | null;
  confidence: number;
  relevance: number;
  evidence_type: string;
  quote_verified: boolean;
  iteration: number;
}

export interface Finding {
  id: string;
  subquestion_id: string;
  statement: string;
  evidence_ids: string[];
  confidence: "high" | "moderate" | "low";
  caveats: string[];
  contradiction_ids: string[];
}

export interface Citation {
  number: number;
  source_id: string;
  title: string;
  url: string;
  authors: string[];
  published_date: string | null;
  venue: string | null;
  source_type: SourceType;
  quality: number | null;
  evidence_ids: string[];
}

export type Severity = "low" | "medium" | "high" | "critical";

export interface CritiqueIssue {
  id: string;
  issue_type: string;
  severity: Severity;
  description: string;
  finding_ids: string[];
  evidence_ids: string[];
  subquestion_id: string | null;
  suggested_action: string;
  suggested_queries: string[];
  origin: "rule" | "llm";
}

export interface Critique {
  iteration: number;
  issues: CritiqueIssue[];
  strengths: string[];
  summary: string;
  verdict: "acceptable" | "needs_revision" | "major_problems";
}

export interface QualityAssessment {
  iteration: number;
  score: number;
  threshold: number;
  passed: boolean;
  components: Record<string, number>;
  blocking_issues: string[];
  decision: "proceed" | "research_more" | "proceed_with_limitations";
}

export interface Contradiction {
  id: string;
  subquestion_id: string | null;
  evidence_ids: string[];
  description: string;
  severity: "minor" | "major";
}

export interface Report {
  research_id: string;
  title: string;
  markdown: string;
  executive_summary: string;
  bibliography: Citation[];
  metrics: ReportMetrics;
  review_notes: string[];
  critique: Critique | null;
  quality: QualityAssessment | null;
  contradictions: Contradiction[];
  generated_at: string;
}

export type EventType =
  | "run_status"
  | "node_started"
  | "node_completed"
  | "progress"
  | "tool"
  | "warning"
  | "error"
  | "plan_ready"
  | "metrics";


export interface WorkflowEvent {
  seq: number;
  type: EventType;
  node: string | null;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface Health {
  status: string;
  version: string;
  environment: string;
  llm_provider: string;
  model: string;
  search_provider: string;
  embedding_provider: string;
  vector_store: string;
  database: string;
  tracing_enabled: boolean;
  active_runs: number;
  /** Present when a password-protected live lane runs next to the demo lane. */
  live_mode: { llm_provider: string; model: string; search_provider: string } | null;
}
