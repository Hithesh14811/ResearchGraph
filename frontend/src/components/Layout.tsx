import { ArrowUpRight, Moon, Sun } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { useMode } from "../lib/mode";
import { api } from "../services/api";
import { useTheme } from "../lib/theme";
import { LiveModeDialog } from "./LiveModeDialog";
import { Wordmark } from "./Logo";
import { Dot, cx } from "./ui";

function EnvironmentIndicator() {
  const { health, apiDown, mode } = useMode();

  if (apiDown) {
    return (
      <span className="inline-flex items-center gap-1.5 text-[12px] text-bad">
        <Dot tone="bad" /> API unreachable
      </span>
    );
  }
  if (!health) return null;
  const lane =
    mode === "live" && health.live_mode
      ? { label: "live", model: health.live_mode.model, search: health.live_mode.search_provider }
      : { label: health.llm_provider === "mock" ? "demo mode" : health.llm_provider, model: health.model, search: health.search_provider };
  const isDemo = lane.label === "demo mode";
  return (
    <span
      className="hidden items-center gap-2 font-mono text-[11.5px] text-muted md:inline-flex"
      title={`model: ${lane.model} · search: ${lane.search} · embeddings: ${health.embedding_provider} · vectors: ${health.vector_store} · db: ${health.database}`}
    >
      <Dot tone={isDemo ? "warn" : "ok"} />
      <span className="text-ink-2">{lane.label}</span>
      <span className="text-faint">/</span>
      <span>{lane.model}</span>
      <span className="text-faint">/</span>
      <span>{health.database}</span>
      {health.tracing_enabled && (
        <>
          <span className="text-faint">/</span>
          <span>langsmith</span>
        </>
      )}
    </span>
  );
}

/** Demo ⇄ Live switch, shown only when the server offers a password-protected live lane. */
function ModeSwitch() {
  const { liveAvailable, mode, lock } = useMode();
  const [asking, setAsking] = useState(false);
  if (!liveAvailable) return null;
  const option = (value: "demo" | "live", label: string, onClick: () => void) => (
    <button
      type="button"
      role="radio"
      aria-checked={mode === value}
      onClick={onClick}
      className={cx(
        "rounded px-2 py-0.5 text-[12px] font-medium transition",
        mode === value ? "bg-surface text-ink shadow-sm ring-1 ring-line-strong" : "text-muted hover:text-ink",
      )}
    >
      {label}
    </button>
  );
  return (
    <>
      <div role="radiogroup" aria-label="Research mode" className="inline-flex rounded-md bg-surface-2 p-0.5 ring-1 ring-line">
        {option("demo", "Demo", () => mode === "live" && lock())}
        {option("live", "Live", () => mode === "demo" && setAsking(true))}
      </div>
      {asking && <LiveModeDialog onClose={() => setAsking(false)} />}
    </>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button
      type="button"
      onClick={toggle}
      className="grid h-8 w-8 place-items-center rounded-md text-muted transition hover:bg-surface-2 hover:text-ink"
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      title={theme === "dark" ? "Light theme" : "Dark theme"}
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="no-print sticky top-0 z-20 border-b border-line bg-surface/90 backdrop-blur-sm">
        <div className="mx-auto flex h-14 max-w-[1440px] items-center justify-between gap-4 px-4 sm:px-6">
          <Link to="/" className="rounded-md">
            <Wordmark />
          </Link>
          <div className="flex items-center gap-4">
            <EnvironmentIndicator />
            <ModeSwitch />
            <span className="hidden h-4 w-px bg-line md:block" />
            <a
              href={api.docsUrl}
              target="_blank"
              rel="noreferrer"
              className="hidden items-center gap-0.5 text-[13px] text-ink-2 hover:text-ink sm:inline-flex"
            >
              API <ArrowUpRight className="h-3.5 w-3.5" />
            </a>
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6 sm:py-8">{children}</main>
    </div>
  );
}
