import { useState } from "react";
import type { AppConfig } from "../types";
import type { DashState } from "../state/reducer";
import { pendingReview } from "../state/reducer";
import { postAdmin } from "../config";

export type ToastFn = (kind: "pending" | "ok" | "error", text: string) => void;

interface SceneDef {
  scene: string;
  label: string;
  target: string | null; // default target, null = no target selector
  danger?: boolean;
}

export const SCENES: SceneDef[] = [
  { scene: "poison", label: "Poison (hacked device)", target: "phone-03", danger: true },
  { scene: "replay", label: "Replay attack", target: null, danger: true },
  { scene: "tamper", label: "Tamper", target: null, danger: true },
  { scene: "forge", label: "Forged key", target: null, danger: true },
  { scene: "mitm", label: "Eavesdropper (MITM)", target: null, danger: true },
  { scene: "drift", label: "Make device drift", target: "phone-05" },
  { scene: "weak_signal", label: "Weak signal", target: "phone-02" },
  { scene: "low_battery", label: "Low battery", target: "phone-04" },
  { scene: "offline", label: "Go offline", target: "phone-06" },
  { scene: "restore", label: "Restore", target: "" },
];

const FALLBACK_IDS = ["phone-01", "phone-02", "phone-03", "phone-04", "phone-05", "phone-06", "esp32-01", "esp32-02", "esp32-03"];

export function AttackConsole({
  s,
  cfg,
  enabled,
  toast,
}: {
  s: DashState;
  cfg: AppConfig;
  enabled: boolean;
  toast: ToastFn;
}) {
  const ids = s.deviceOrder.length ? s.deviceOrder : FALLBACK_IDS;
  const [targets, setTargets] = useState<Record<string, string>>(() =>
    Object.fromEntries(SCENES.filter((x) => x.target !== null).map((x) => [x.scene, x.target as string])),
  );
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const pending = pendingReview(s);

  async function fire(def: SceneDef) {
    const target = def.target === null ? undefined : targets[def.scene] || undefined;
    setBusy((b) => ({ ...b, [def.scene]: true }));
    toast("pending", `${def.label}${target ? ` → ${target}` : ""}: sent…`);
    const r = await postAdmin(cfg, `/fl/admin/attack/${def.scene}`, target ? { target } : {});
    setBusy((b) => ({ ...b, [def.scene]: false }));
    if (r.ok) toast("ok", `${def.label}: server accepted the request (outcome will appear in the feed)`);
    else toast("error", `${def.label}: request failed${r.status ? ` (HTTP ${r.status})` : ""} ${r.text.slice(0, 120)}`);
  }

  async function decide(client_id: string, update_id: string, decision: "approve" | "reject") {
    setBusy((b) => ({ ...b, [update_id]: true }));
    toast("pending", `${decision === "approve" ? "Approving" : "Rejecting"} ${update_id}…`);
    const r = await postAdmin(cfg, "/fl/admin/decision", { client_id, update_id, decision });
    setBusy((b) => ({ ...b, [update_id]: false }));
    if (r.ok) toast("ok", `Decision sent for ${update_id} (waiting for confirmation event)`);
    else toast("error", `Decision failed${r.status ? ` (HTTP ${r.status})` : ""} ${r.text.slice(0, 120)}`);
  }

  return (
    <section className={`panel console ${enabled ? "" : "disabled"}`}>
      <h2 className="panel-title">
        Attack console{" "}
        <span className="muted">{enabled ? "· real requests to the real server" : "· disabled in replay"}</span>
        <a className="fs-link" href={cfg.finsentinel_ui} target="_blank" rel="noreferrer">
          Open FinSentinel console ↗
        </a>
      </h2>
      <div className="scene-grid">
        {SCENES.map((def) => (
          <div key={def.scene} className={`scene ${def.danger ? "danger" : ""}`}>
            <button disabled={!enabled || busy[def.scene]} onClick={() => fire(def)}>
              {def.label}
            </button>
            {def.target !== null && (
              <select
                disabled={!enabled}
                value={targets[def.scene] ?? ""}
                onChange={(e) => setTargets((t) => ({ ...t, [def.scene]: e.target.value }))}
                aria-label={`Target for ${def.label}`}
              >
                {def.scene === "restore" && <option value="">all devices</option>}
                {[...new Set([...(def.target ? [def.target] : []), ...ids])].map((id) => (
                  <option key={id} value={id}>
                    {s.devices[id]?.info.label ?? id}
                  </option>
                ))}
              </select>
            )}
          </div>
        ))}
      </div>
      <div className="review">
        <div className="review-title">
          Waiting for review <span className="muted">({pending.length})</span>
        </div>
        {pending.length === 0 && <div className="empty small">Nothing waiting.</div>}
        {pending.slice(0, 4).map((g) => (
          <div key={g.ev.update_id} className="review-row">
            <span className="review-dev">{s.devices[g.ev.device]?.info.label ?? g.ev.device}</span>
            <span className="muted small">
              R{g.ev.round} · {g.ev.severity ?? "?"} · risk {g.ev.risk_score != null ? g.ev.risk_score.toFixed(2) : "—"}
            </span>
            <button
              className="ok"
              disabled={!enabled || busy[g.ev.update_id]}
              onClick={() => decide(g.ev.device, g.ev.update_id, "approve")}
            >
              Approve
            </button>
            <button
              className="bad"
              disabled={!enabled || busy[g.ev.update_id]}
              onClick={() => decide(g.ev.device, g.ev.update_id, "reject")}
            >
              Reject
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
