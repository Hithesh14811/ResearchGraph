import { Loader2, Lock } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { createPortal } from "react-dom";

import { useMode } from "../lib/mode";
import { Button } from "./ui";

/** Password prompt for switching from demo to live mode (portalled: the header's backdrop blur
 * would otherwise trap a fixed overlay inside it). */
export function LiveModeDialog({ onClose }: { onClose: () => void }) {
  const { unlock, health } = useMode();
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    input.current?.focus();
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await unlock(password);
      onClose();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setBusy(false);
    }
  };

  const live = health?.live_mode;
  return createPortal(
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4" onMouseDown={onClose}>
      <form
        role="dialog"
        aria-modal="true"
        aria-labelledby="live-mode-title"
        onSubmit={submit}
        onMouseDown={(event) => event.stopPropagation()}
        className="w-full max-w-sm rounded-lg border border-line bg-surface shadow-xl"
      >
        <div className="px-5 pt-5">
          <h2 id="live-mode-title" className="flex items-center gap-2 text-[15px] font-semibold text-ink">
            <Lock className="h-4 w-4 text-muted" /> Switch to live mode
          </h2>
          <p className="mt-1.5 text-[13px] leading-relaxed text-muted">
            Live runs use a real model{live ? ` (${live.model})` : ""} and real academic search, and cost API credits. A run
            takes several minutes.
          </p>
          <label htmlFor="live-password" className="mt-4 block text-[12px] font-medium text-ink-2">
            Password
          </label>
          <input
            ref={input}
            id="live-password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1 h-9 w-full rounded-md border border-line-strong bg-surface px-3 text-[14px] text-ink focus:border-ink focus:outline-none"
          />
          {error && <p className="mt-2 text-[12.5px] text-bad">{error}</p>}
        </div>
        <div className="mt-5 flex justify-end gap-2 border-t border-line bg-surface-2 px-5 py-3">
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" type="submit" disabled={busy || password.length === 0}>
            {busy && <Loader2 className="animate-spin" />} Unlock live mode
          </Button>
        </div>
      </form>
    </div>,
    document.body,
  );
}
