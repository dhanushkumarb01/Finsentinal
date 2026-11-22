// Polls GET /fl/admin/console every 2 s while enabled (live mode). Replay mode never polls.
import { useEffect, useState } from "react";
import type { AppConfig, ConsoleApi } from "../types";
import { getAdmin } from "../config";
import { parseConsole } from "../state/console";

export const CONSOLE_POLL_MS = 2000;

export interface ConsolePoll {
  data: ConsoleApi | null;
  status: "off" | "loading" | "ok" | "error";
  error: string | null;
  lastOk: number | null; // wall-clock ms of the last good response (UI display only)
}

export function useConsolePoll(cfg: AppConfig | null, enabled: boolean): ConsolePoll {
  const [st, setSt] = useState<ConsolePoll>({ data: null, status: "off", error: null, lastOk: null });
  useEffect(() => {
    if (!cfg || !enabled) {
      setSt({ data: null, status: "off", error: null, lastOk: null });
      return;
    }
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const ctl = new AbortController();
    setSt((p) => ({ ...p, status: p.data ? p.status : "loading" }));
    const tick = async () => {
      const r = await getAdmin(cfg, "/fl/admin/console", ctl.signal);
      if (!alive) return;
      const parsed = r.ok ? parseConsole(r.json) : null;
      if (parsed) setSt({ data: parsed, status: "ok", error: null, lastOk: Date.now() });
      else
        setSt((p) => ({
          ...p,
          status: "error",
          error: r.status ? `HTTP ${r.status}${r.ok ? " (unexpected body)" : ""}` : r.text || "unreachable",
        }));
      timer = setTimeout(tick, CONSOLE_POLL_MS);
    };
    tick();
    return () => {
      alive = false;
      ctl.abort();
      if (timer) clearTimeout(timer);
    };
  }, [cfg, enabled]);
  return st;
}
