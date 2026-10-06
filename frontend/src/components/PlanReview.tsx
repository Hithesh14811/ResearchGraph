import { Check, Loader2, MessageSquareText, Plus, X } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "../services/api";
import type { ResearchPlan, SourcePreference, Subquestion } from "../types/api";
import { Button, Dot, SectionLabel, Tag } from "./ui";

const PREFS: SourcePreference[] = ["academic", "web", "mixed"];

function renumber(plan: ResearchPlan): ResearchPlan {
  return { ...plan, subquestions: plan.subquestions.map((sq, i) => ({ ...sq, id: `SQ${i + 1}` })) };
}

export function PlanReview({ researchId, plan, onDecision }: { researchId: string; plan: ResearchPlan; onDecision: () => void }) {
  const [draft, setDraft] = useState<ResearchPlan>(plan);
  const [dirty, setDirty] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [showFeedback, setShowFeedback] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(plan);
    setDirty(false);
  }, [plan]);

  const update = (index: number, patch: Partial<Subquestion>) => {
    setDirty(true);
    setDraft((d) => ({ ...d, subquestions: d.subquestions.map((sq, i) => (i === index ? { ...sq, ...patch } : sq)) }));
  };

  const act = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await action();
      onDecision();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(null);
    }
  };

  const valid = draft.subquestions.length > 0 && draft.subquestions.every((sq) => sq.question.trim().length >= 5);

  return (
    <section className="overflow-hidden rounded-lg border border-line bg-surface">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-warn-soft px-4 py-3">
        <div>
          <h2 className="flex items-center gap-2 text-[14px] font-medium text-ink">
            <Dot tone="warn" pulse /> Review the research plan
          </h2>
          <p className="mt-0.5 text-[12.5px] text-ink-2">
            The graph is paused at an <code className="font-mono text-[11.5px]">interrupt()</code>. Edit the subquestions and
            searches, ask the planner for changes, or approve to start research.
          </p>
        </div>
        <Tag>plan v{plan.version}</Tag>
      </header>

      <div className="space-y-5 p-4">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <SectionLabel>Objective</SectionLabel>
            <p className="mt-1 text-[14px] leading-relaxed text-ink">{draft.objective}</p>
          </div>
          <div>
            <SectionLabel>Scope</SectionLabel>
            <p className="mt-1 text-[14px] leading-relaxed text-ink-2">{draft.scope || "—"}</p>
          </div>
        </div>

        <div>
          <SectionLabel className="mb-2">Subquestions — researched in parallel, one worker each</SectionLabel>
          <ol className="divide-y divide-line rounded-md border border-line">
            {draft.subquestions.map((sq, index) => (
              <li key={index} className="p-3">
                <div className="flex items-start gap-2">
                  <span className="mt-1.5 w-8 shrink-0 font-mono text-[12px] text-muted">SQ{index + 1}</span>
                  <textarea
                    value={sq.question}
                    rows={1}
                    aria-label={`Subquestion ${index + 1}`}
                    onChange={(e) => update(index, { question: e.target.value })}
                    className="field-sizing-content min-h-[2.1rem] w-full resize-none rounded-md border border-transparent bg-transparent px-2 py-1 text-[14px] text-ink hover:border-line focus:border-line-strong focus:bg-surface-2 focus:outline-none"
                  />
                  <select
                    value={sq.preferred_sources}
                    aria-label={`Preferred sources for SQ${index + 1}`}
                    onChange={(e) => update(index, { preferred_sources: e.target.value as SourcePreference })}
                    className="mt-0.5 h-7 rounded-md border border-line bg-surface px-1.5 text-[12px] text-ink-2"
                  >
                    {PREFS.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => {
                      setDirty(true);
                      setDraft((d) => ({ ...d, subquestions: d.subquestions.filter((_, i) => i !== index) }));
                    }}
                    className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-md text-faint hover:bg-bad-soft hover:text-bad"
                    aria-label={`Remove SQ${index + 1}`}
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <textarea
                  value={sq.search_queries.map((q) => q.query).join("\n")}
                  rows={Math.max(2, sq.search_queries.length)}
                  aria-label={`Search queries for SQ${index + 1}`}
                  onChange={(e) =>
                    update(index, {
                      search_queries: e.target.value
                        .split("\n")
                        .map((q) => q.trim())
                        .filter((q) => q.length > 0)
                        .map((query) => ({ query, source_preference: sq.preferred_sources })),
                    })
                  }
                  className="field-sizing-content mt-2 ml-10 block w-[calc(100%-2.5rem)] resize-none rounded-md border border-line bg-surface-2 px-2.5 py-1.5 font-mono text-[12px] leading-relaxed text-ink-2 focus:border-line-strong focus:outline-none"
                  placeholder="One search query per line"
                />
              </li>
            ))}
          </ol>
          <Button
            variant="ghost"
            className="mt-2"
            onClick={() => {
              setDirty(true);
              setDraft((d) => ({
                ...d,
                subquestions: [
                  ...d.subquestions,
                  { id: `SQ${d.subquestions.length + 1}`, question: "", rationale: "Added by reviewer.", information_requirements: [], search_queries: [], preferred_sources: "mixed" },
                ],
              }));
            }}
          >
            <Plus /> Add subquestion
          </Button>
        </div>

        <dl className="grid gap-4 text-[13px] md:grid-cols-2">
          <div>
            <dt className="text-[12px] font-medium text-muted">Source strategy</dt>
            <dd className="mt-1 text-ink-2">{draft.source_strategy.join(" · ") || "—"}</dd>
          </div>
          <div>
            <dt className="text-[12px] font-medium text-muted">Stopping criteria</dt>
            <dd className="mt-1 text-ink-2">{draft.stopping_criteria.join(" · ") || "—"}</dd>
          </div>
        </dl>

        {showFeedback && (
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={3}
            aria-label="Feedback for the planner"
            placeholder="Describe what the planner should change, e.g. “add a subquestion on evaluation benchmarks”…"
            className="w-full rounded-md border border-line-strong bg-surface px-3 py-2 text-[14px] text-ink placeholder:text-faint focus:outline-none"
          />
        )}
        {error && <p className="rounded-md bg-bad-soft px-3 py-2 text-[13px] text-bad">{error}</p>}
      </div>

      <footer className="flex flex-wrap items-center justify-end gap-2 border-t border-line bg-surface-2 px-4 py-2.5">
        <Button variant="danger" disabled={busy !== null} onClick={() => act("cancel", () => api.cancel(researchId))} className="mr-auto">
          {busy === "cancel" ? <Loader2 className="animate-spin" /> : <X />} Cancel run
        </Button>
        {showFeedback ? (
          <Button disabled={busy !== null || feedback.trim().length < 3} onClick={() => act("replan", () => api.replan(researchId, feedback.trim()))}>
            {busy === "replan" ? <Loader2 className="animate-spin" /> : <MessageSquareText />} Send to planner
          </Button>
        ) : (
          <Button onClick={() => setShowFeedback(true)}>
            <MessageSquareText /> Request changes
          </Button>
        )}
        {dirty ? (
          <Button variant="primary" disabled={busy !== null || !valid} onClick={() => act("edit", () => api.editPlan(researchId, renumber(draft), true))}>
            {busy === "edit" ? <Loader2 className="animate-spin" /> : <Check />} Save edits and approve
          </Button>
        ) : (
          <Button variant="primary" disabled={busy !== null} onClick={() => act("approve", () => api.approve(researchId))}>
            {busy === "approve" ? <Loader2 className="animate-spin" /> : <Check />} Approve plan
          </Button>
        )}
      </footer>
    </section>
  );
}
