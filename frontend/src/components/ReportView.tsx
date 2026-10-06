import { ArrowUpRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { SOURCE_TYPE_LABEL, percent } from "../lib/format";
import type { Citation, Report } from "../types/api";

/** Turn inline [n] markers into links to the bibliography (skips list items that start with [n]). */
function linkCitations(markdown: string): string {
  return markdown
    .split("\n")
    .map((line) => (/^\[\d+\]\s/.test(line) ? line : line.replace(/\[(\d+)\](?!\()/g, "[[$1]](#ref-$1)")))
    .join("\n");
}

function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) =>
          href?.startsWith("#ref-") ? (
            <a href={href} className="citation">
              {children}
            </a>
          ) : (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
      }}
    >
      {linkCitations(text)}
    </ReactMarkdown>
  );
}

function ReportMeta({ report }: { report: Report }) {
  const m = report.metrics;
  const items = [
    `${m.total_claims} verified sentences`,
    `${percent(m.citation_coverage)} citation coverage`,
    `${m.sources_cited} sources cited`,
    `${m.removed_claims} removed`,
    new Date(report.generated_at).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }),
  ];
  return (
    <p className="mb-8 border-y border-line py-2 font-sans text-[12px] text-muted not-italic">
      {items.map((item, i) => (
        <span key={item}>
          {i > 0 && <span className="mx-2 text-faint">·</span>}
          {item}
        </span>
      ))}
    </p>
  );
}

export function Bibliography({ citations }: { citations: Citation[] }) {
  return (
    <ol className="my-0! list-none! space-y-3 pl-0!">
      {citations.map((c) => (
        <li
          key={c.number}
          id={`ref-${c.number}`}
          className="grid scroll-mt-24 grid-cols-[2.25rem_minmax(0,1fr)] rounded-sm py-0.5 target:bg-accent-soft"
        >
          <span className="pt-px font-sans text-[13px] text-muted tabular-nums">[{c.number}]</span>
          <div>
            <a href={c.url} target="_blank" rel="noreferrer" className="text-ink! no-underline! hover:underline!">
              {c.title}
              <ArrowUpRight className="ml-0.5 inline h-3.5 w-3.5 text-faint" />
            </a>
            <p className="mt-0.5 font-sans text-[12.5px] leading-relaxed text-muted">
              {[
                c.authors.length > 0 ? `${c.authors.slice(0, 3).join(", ")}${c.authors.length > 3 ? " et al." : ""}` : null,
                c.published_date?.slice(0, 4),
                c.venue,
                SOURCE_TYPE_LABEL[c.source_type],
                c.quality != null ? `quality ${c.quality.toFixed(2)}` : null,
                `${c.evidence_ids.length} evidence item${c.evidence_ids.length === 1 ? "" : "s"}`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}

export function ExecutiveSummary({ report }: { report: Report }) {
  return (
    <article className="report-prose print-area">
      <h1>{report.title}</h1>
      <ReportMeta report={report} />
      <Markdown text={report.executive_summary} />
      <h2>Sources cited</h2>
      <Bibliography citations={report.bibliography} />
    </article>
  );
}

export function FullReport({ report }: { report: Report }) {
  const [body] = report.markdown.split(/\n## References\n/);
  const text = body ?? report.markdown;
  // The rendered markdown starts with the report's own H1; show the metadata line right under it.
  const match = /^# .+\n/.exec(text);
  return (
    <article className="report-prose print-area">
      {match ? (
        <>
          <Markdown text={match[0]} />
          <ReportMeta report={report} />
          <Markdown text={text.slice(match[0].length)} />
        </>
      ) : (
        <Markdown text={text} />
      )}
      <h2>References</h2>
      <Bibliography citations={report.bibliography} />
      {report.review_notes.length > 0 && (
        <aside className="no-print mt-10 rounded-md border border-line bg-surface-2 px-4 py-3 font-sans text-[12.5px] text-muted">
          <p className="mb-1 font-medium text-ink-2">Final review notes</p>
          <ul className="my-0! list-disc space-y-0.5 pl-5!">
            {report.review_notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </aside>
      )}
    </article>
  );
}
