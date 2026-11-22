import type { AppConfig } from "./types";

export const DEFAULT_CONFIG: AppConfig = {
  mode: "replay",
  ws_url: "ws://127.0.0.1:8200/fl/ws",
  http_url: "http://127.0.0.1:8200",
  admin_token: "",
  replay_url: "/sample-recording.jsonl",
  finsentinel_ui: "http://localhost:5173",
  femnist_url: "/femnist.json",
};

export function mergeConfig(raw: unknown): AppConfig {
  const c = { ...DEFAULT_CONFIG };
  if (raw && typeof raw === "object") {
    const o = raw as Record<string, unknown>;
    if (o.mode === "live" || o.mode === "replay") c.mode = o.mode;
    for (const k of ["ws_url", "http_url", "admin_token", "replay_url", "finsentinel_ui", "femnist_url"] as const) {
      if (typeof o[k] === "string") c[k] = o[k] as string;
    }
  }
  return c;
}

/** Load /config.json; if it is missing (plain `npm run dev`), fall back to the bundled sample replay. */
export async function loadConfig(): Promise<{ config: AppConfig; fromFile: boolean }> {
  try {
    const r = await fetch("/config.json", { cache: "no-store" });
    if (!r.ok) throw new Error(String(r.status));
    return { config: mergeConfig(await r.json()), fromFile: true };
  } catch {
    return { config: { ...DEFAULT_CONFIG }, fromFile: false };
  }
}

export async function postAdmin(
  cfg: AppConfig,
  path: string,
  body: unknown,
): Promise<{ ok: boolean; status: number; text: string }> {
  try {
    const r = await fetch(`${cfg.http_url.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.admin_token}` },
      body: JSON.stringify(body),
    });
    let text = "";
    try {
      text = await r.text();
    } catch {
      /* ignore */
    }
    return { ok: r.ok, status: r.status, text };
  } catch (e) {
    return { ok: false, status: 0, text: e instanceof Error ? e.message : String(e) };
  }
}

export async function getAdmin(
  cfg: AppConfig,
  path: string,
  signal?: AbortSignal,
): Promise<{ ok: boolean; status: number; json: unknown; text: string }> {
  try {
    const r = await fetch(`${cfg.http_url.replace(/\/$/, "")}${path}`, {
      headers: { Authorization: `Bearer ${cfg.admin_token}` },
      cache: "no-store",
      signal,
    });
    const text = await r.text();
    let json: unknown = null;
    try {
      json = JSON.parse(text);
    } catch {
      /* not JSON */
    }
    return { ok: r.ok, status: r.status, json, text };
  } catch (e) {
    return { ok: false, status: 0, json: null, text: e instanceof Error ? e.message : String(e) };
  }
}
