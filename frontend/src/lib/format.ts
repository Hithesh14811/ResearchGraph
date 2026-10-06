import type { RunStatus, Severity, SourceType } from "../types/api";

export const percent = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;

export const number = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : value.toLocaleString();

export const seconds = (value: number | null | undefined) => {
  if (value === null || value === undefined) return "—";
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
};

export const relativeTime = (iso: string) => {
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

export const plural = (count: number, noun: string) => `${count.toLocaleString()} ${noun}${count === 1 ? "" : "s"}`;

export const humanize = (value: string) => value.replace(/_/g, " ");

/** Semantic colour roles; each maps to a design token (see index.css). */
export type Tone = "neutral" | "accent" | "ok" | "warn" | "bad";

export const TONE_DOT: Record<Tone, string> = {
  neutral: "bg-faint",
  accent: "bg-accent",
  ok: "bg-ok",
  warn: "bg-warn",
  bad: "bg-bad",
};

export const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-muted",
  accent: "text-accent",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
};

export const TONE_SOFT: Record<Tone, string> = {
  neutral: "bg-surface-2 text-ink-2 ring-1 ring-line ring-inset",
  accent: "bg-accent-soft text-accent",
  ok: "bg-ok-soft text-ok",
  warn: "bg-warn-soft text-warn",
  bad: "bg-bad-soft text-bad",
};

export const STATUS_TONE: Record<RunStatus, Tone> = {
  pending: "neutral",
  running: "accent",
  awaiting_approval: "warn",
  completed: "ok",
  failed: "bad",
  cancelled: "neutral",
  interrupted: "warn",
  rejected: "bad",
};

export const SEVERITY_TONE: Record<Severity, Tone> = {
  low: "neutral",
  medium: "warn",
  high: "bad",
  critical: "bad",
};

export const TIER_TONE = {
  high: "ok",
  medium: "accent",
  low: "bad",
} as const satisfies Record<string, Tone>;

export const SOURCE_TYPE_LABEL: Record<SourceType, string> = {
  primary_research: "Primary research",
  official_documentation: "Official docs",
  institutional: "Institutional",
  reputable_technical: "Technical publication",
  news: "News",
  blog: "Blog",
  forum: "Forum",
  unknown: "Unclassified",
};
