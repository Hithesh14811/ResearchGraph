import { useCallback, useEffect, useState } from "react";

import { api } from "../services/api";
import { TERMINAL_STATUSES, type Run } from "../types/api";

/** Polls a run while it is active; stops once it reaches a terminal state. */
export function useRun(researchId: string | undefined, refreshKey = 0) {
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!researchId) return;
    try {
      setRun(await api.getRun(researchId));
      setError(null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, [researchId]);

  useEffect(() => {
    void refresh();
  }, [refresh, refreshKey]);

  const active = run !== null && !TERMINAL_STATUSES.includes(run.status) && run.status !== "awaiting_approval";
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => void refresh(), 1500);
    return () => window.clearInterval(timer);
  }, [active, refresh]);

  return { run, error, refresh };
}
