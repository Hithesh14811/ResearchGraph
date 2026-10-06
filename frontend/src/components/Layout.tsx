import { ArrowUpRight, Moon, Sun } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { useTheme } from "../lib/theme";
import { api } from "../services/api";
import type { Health } from "../types/api";
import { Wordmark } from "./Logo";
import { Dot } from "./ui";

function EnvironmentIndicator() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .health()
        .then((h) => alive && (setHealth(h), setDown(false)))
        .catch(() => alive && setDown(true));
    void load();
    const timer = window.setInterval(load, 15000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  if (down) {
    return (
      <span className="inline-flex items-center gap-1.5 text-[12px] text-bad">
        <Dot tone="bad" /> API unreachable
      </span>
    );
  }
  if (!health) return null;
  const demo = health.llm_provider === "mock";
  return (
    <span
      className="hidden items-center gap-2 font-mono text-[11.5px] text-muted md:inline-flex"
      title={`LLM: ${health.llm_provider}/${health.model} · search: ${health.search_provider} · embeddings: ${health.embedding_provider} · vectors: ${health.vector_store} · db: ${health.database}`}
    >
      <Dot tone={demo ? "warn" : "ok"} />
      <span className="text-ink-2">{demo ? "demo mode" : health.llm_provider}</span>
      <span className="text-faint">/</span>
      <span>{health.model}</span>
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
            <span className="hidden h-4 w-px bg-line md:block" />
            <a
              href="/api/docs"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-0.5 text-[13px] text-ink-2 hover:text-ink"
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
