import type { ButtonHTMLAttributes, ReactNode } from "react";

import { STATUS_TONE, TONE_DOT, TONE_SOFT, TONE_TEXT, humanize, type Tone } from "../lib/format";
import type { RunStatus } from "../types/api";

const sentence = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

export function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function Card({
  children,
  className,
  title,
  actions,
}: {
  children: ReactNode;
  className?: string;
  title?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className={cx("rounded-lg border border-line bg-surface", className)}>
      {(title || actions) && (
        <header className="flex min-h-11 items-center justify-between gap-3 border-b border-line px-4 py-2">
          <h2 className="text-[13px] font-medium text-ink">{title}</h2>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

type Variant = "primary" | "secondary" | "danger" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-ink text-surface hover:opacity-85 disabled:opacity-35",
  secondary: "border border-line-strong bg-surface text-ink hover:bg-surface-2 disabled:opacity-50",
  danger: "border border-line-strong bg-surface text-bad hover:border-bad/40 hover:bg-bad-soft disabled:opacity-50",
  ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink disabled:opacity-50",
};

export const buttonClass = (variant: Variant = "secondary", className?: string) =>
  cx(
    "inline-flex h-8 items-center justify-center gap-1.5 rounded-md px-3 text-[13px] font-medium whitespace-nowrap transition disabled:cursor-not-allowed [&_svg]:h-3.5 [&_svg]:w-3.5",
    VARIANTS[variant],
    className,
  );

export function Button({
  variant = "secondary",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return <button type="button" {...props} className={buttonClass(variant, className)} />;
}

/** Small neutral label, e.g. a subquestion id or an evidence type. */
export function Tag({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cx("inline-flex items-center gap-1 rounded border border-line px-1.5 py-px font-mono text-[11px] text-muted", className)}>
      {children}
    </span>
  );
}

/** A tinted label for values that carry meaning (severity, tier, verdict). */
export function Pill({ tone, children, className }: { tone: Tone; children: ReactNode; className?: string }) {
  return (
    <span className={cx("inline-flex items-center gap-1 rounded px-1.5 py-px text-[11.5px] font-medium", TONE_SOFT[tone], className)}>
      {children}
    </span>
  );
}

export function Dot({ tone, pulse, className }: { tone: Tone; pulse?: boolean; className?: string }) {
  return <span className={cx("inline-block h-1.5 w-1.5 shrink-0 rounded-full", TONE_DOT[tone], pulse && "pulse-dot", className)} />;
}

export function StatusBadge({ status, className }: { status: RunStatus; className?: string }) {
  const tone = STATUS_TONE[status];
  const live = status === "running" || status === "pending";
  return (
    <span className={cx("inline-flex items-center gap-1.5 text-[12.5px] font-medium", TONE_TEXT[tone], className)}>
      <Dot tone={tone} pulse={live} />
      {sentence(humanize(status))}
    </span>
  );
}

/** Thin horizontal meter for a 0–1 value, with an optional threshold tick. */
export function Meter({
  value,
  tone = "ink",
  threshold,
  className,
}: {
  value: number;
  tone?: Tone | "ink";
  threshold?: number;
  className?: string;
}) {
  const fill = tone === "ink" ? "bg-ink-2" : TONE_DOT[tone];
  return (
    <div className={cx("relative h-1 w-full rounded-full bg-line", className)}>
      <div className={cx("h-full rounded-full transition-[width] duration-500", fill)} style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }} />
      {threshold !== undefined && (
        <span className="absolute -top-1 h-3 w-px bg-ink" style={{ left: `${threshold * 100}%` }} title={`threshold ${threshold.toFixed(2)}`} />
      )}
    </div>
  );
}

export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cx("text-[12px] font-medium text-muted", className)}>{children}</p>;
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="px-4 py-12 text-center text-[13px] text-muted">{children}</div>;
}
