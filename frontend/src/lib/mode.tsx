import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { api, liveToken } from "../services/api";
import type { Health, RunMode } from "../types/api";

interface ModeState {
  health: Health | null;
  apiDown: boolean;
  /** Demo unless the user unlocked live mode with the password. */
  mode: RunMode;
  /** True when this server offers a password-protected live lane. */
  liveAvailable: boolean;
  unlock: (password: string) => Promise<void>;
  lock: () => void;
}

const ModeContext = createContext<ModeState | null>(null);

export function ModeProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [unlocked, setUnlocked] = useState(() => liveToken.get() !== null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .health()
        .then((h) => alive && (setHealth(h), setApiDown(false)))
        .catch(() => alive && setApiDown(true));
    void load();
    const timer = window.setInterval(load, 15000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const unlock = useCallback(async (password: string) => {
    const { token, expires_at } = await api.unlockLive(password);
    liveToken.set(token, expires_at);
    setUnlocked(true);
  }, []);

  // Going back to demo never needs a password; returning to live will ask again.
  const lock = useCallback(() => {
    liveToken.clear();
    setUnlocked(false);
  }, []);

  const liveAvailable = health?.live_mode != null;
  const value = useMemo<ModeState>(
    () => ({ health, apiDown, mode: liveAvailable && unlocked ? "live" : "demo", liveAvailable, unlock, lock }),
    [health, apiDown, liveAvailable, unlocked, unlock, lock],
  );
  return <ModeContext.Provider value={value}>{children}</ModeContext.Provider>;
}

export function useMode(): ModeState {
  const value = useContext(ModeContext);
  if (!value) throw new Error("useMode must be used inside ModeProvider");
  return value;
}
