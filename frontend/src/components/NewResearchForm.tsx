import { CornerDownRight, Loader2 } from "lucide-react";
import { useState, type FormEvent, type KeyboardEvent } from "react";
import { useNavigate } from "react-router-dom";

import { useMode } from "../lib/mode";
import { api } from "../services/api";
import { Button, cx } from "./ui";

const SAMPLES = [
  {
    label: "Compare RAG, fine-tuning and long-context prompting for domain-specific QA",
    question:
      "Compare the effectiveness of RAG, fine-tuning, and long-context prompting for domain-specific question answering. Use recent academic papers and credible technical sources. Identify evidence, limitations, benchmarks, and practical recommendations.",
  },
  {
    label: "Tradeoffs between small and large language models for production inference",
    question: "Analyze the tradeoffs between small and large language models for production inference.",
  },
  {
    label: "Current approaches for reducing hallucinations in RAG",
    question: "Evaluate current approaches for reducing hallucinations in retrieval-augmented generation.",
  },
];

const FAILURES = [
  ["failed_source", "Failed source"],
  ["llm_timeout", "LLM timeout"],
  ["invalid_output", "Invalid structured output"],
  ["insufficient_evidence", "Insufficient evidence"],
  ["critic_rejection", "Critic rejection"],
] as const;

export function NewResearchForm({ demoMode }: { demoMode: boolean }) {
  const navigate = useNavigate();
  const { mode, health } = useMode();
  const [question, setQuestion] = useState("");
  const [requireApproval, setRequireApproval] = useState(true);
  const [failures, setFailures] = useState<string[]>([]);
  const [showFaults, setShowFaults] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = question.trim().length >= 10 && !submitting;

  const submit = async (event?: FormEvent) => {
    event?.preventDefault();
    if (!ready) return;
    setSubmitting(true);
    setError(null);
    try {
      const run = await api.createRun(question.trim(), !requireApproval, demoMode ? failures : [], mode);
      navigate(`/runs/${run.id}`);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setSubmitting(false);
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) void submit();
  };

  const toggleFailure = (key: string) =>
    setFailures((current) => (current.includes(key) ? current.filter((f) => f !== key) : [...current, key]));

  return (
    <form onSubmit={submit} className="flex flex-col rounded-lg border border-line bg-surface">
      <div className="flex min-h-11 items-center border-b border-line px-4 py-2">
        <label htmlFor="question" className="text-[13px] font-medium text-ink">
          New research
        </label>
      </div>
      <textarea
        id="question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={onKeyDown}
        rows={4}
        maxLength={4000}
        placeholder="Ask a question that needs evidence — compare approaches, evaluate claims, analyse tradeoffs…"
        className="block min-h-28 w-full flex-1 resize-none bg-transparent px-4 py-3 text-[15px] leading-relaxed text-ink placeholder:text-faint focus:outline-none"
      />

      <div className="px-4 pb-3">
        <p className="mb-1.5 text-[12px] text-muted">Examples</p>
        <ul className="space-y-0.5">
          {SAMPLES.map((sample) => (
            <li key={sample.label}>
              <button
                type="button"
                onClick={() => setQuestion(sample.question)}
                className="group flex w-full items-start gap-2 rounded-md px-1.5 py-1 text-left text-[13px] text-ink-2 hover:bg-surface-2 hover:text-ink"
              >
                <CornerDownRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-faint group-hover:text-muted" />
                {sample.label}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {demoMode && showFaults && (
        <fieldset className="border-t border-line px-4 py-3">
          <legend className="sr-only">Simulated failures</legend>
          <p className="mb-2 text-[12px] text-muted">
            Inject failures to watch the graph recover — retries, fallbacks, and loops back to research.
          </p>
          <div className="grid gap-x-4 gap-y-1.5 sm:grid-cols-2">
            {FAILURES.map(([key, label]) => (
              <label key={key} className="flex cursor-pointer items-center gap-2 text-[13px] text-ink-2">
                <input type="checkbox" checked={failures.includes(key)} onChange={() => toggleFailure(key)} className="h-3.5 w-3.5" />
                {label}
              </label>
            ))}
          </div>
        </fieldset>
      )}

      {mode === "live" && (
        <p className="border-t border-line bg-ok-soft px-4 py-2.5 text-[12.5px] text-ink-2">
          <span className="font-medium text-ok">Live mode.</span> This run uses {health?.live_mode?.model ?? "a real model"} and real
          academic search. Expect several minutes per run.
        </p>
      )}

      {error && <p className="mx-4 mb-3 rounded-md bg-bad-soft px-3 py-2 text-[13px] text-bad">{error}</p>}

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line bg-surface-2 px-4 py-2.5">
        <label className="flex cursor-pointer items-center gap-2 text-[13px] text-ink-2">
          <input type="checkbox" checked={requireApproval} onChange={(e) => setRequireApproval(e.target.checked)} className="h-3.5 w-3.5" />
          Pause for plan review
        </label>
        {demoMode && (
          <button
            type="button"
            onClick={() => setShowFaults((v) => !v)}
            aria-expanded={showFaults}
            className={cx("text-[13px] hover:text-ink", failures.length > 0 ? "text-warn" : "text-ink-2")}
          >
            Simulate failures{failures.length > 0 ? ` (${failures.length})` : ""}
          </button>
        )}
        <div className="ml-auto flex items-center gap-3">
          <span className="hidden font-mono text-[11px] text-faint sm:inline">Ctrl ↵</span>
          <Button variant="primary" type="submit" disabled={!ready}>
            {submitting && <Loader2 className="animate-spin" />}
            Start research
          </Button>
        </div>
      </div>
    </form>
  );
}
