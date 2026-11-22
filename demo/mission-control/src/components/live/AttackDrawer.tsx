import { useEffect, useState } from "react";
import type { AppConfig } from "../../types";
import type { DashState } from "../../state/reducer";
import { postAdmin } from "../../config";

export type ToastFn = (kind: "pending" | "ok" | "error", text: string) => void;

interface SceneDef {
  scene: string;
  label: string;
  explain: string;
  target: string | null; // default target, null = no target selector
  group: "attack" | "life";
}

export const SCENES: SceneDef[] = [
  { scene: "poison", label: "Hack a device", explain: "It sends weights 8× too large — still signed with its real key.", target: "phone-03", group: "attack" },
  { scene: "replay", label: "Replay an old update", explain: "Re-send a captured genuine update. Expect: “Not a replay” fails (409).", target: null, group: "attack" },
  { scene: "tamper", label: "Tamper in transit", explain: "Change bytes on the way. Expect: “Untampered” fails (422).", target: null, group: "attack" },
  { scene: "forge", label: "Forge a signature", explain: "Sign with a key the server never trusted. Expect: “Sender verified” fails (401).", target: null, group: "attack" },
  { scene: "mitm", label: "Eavesdropper (MITM)", explain: "Tap the line, then pose as the server. Sees only ciphertext; fake certificate refused.", target: null, group: "attack" },
  { scene: "drift", label: "Make a device drift", explain: "Its user changes behaviour: honest updates, but it starts skipping rounds.", target: "phone-05", group: "life" },
  { scene: "weak_signal", label: "Weak signal", explain: "Throttle the device's link to 3 KB/s — its download times out.", target: "phone-02", group: "life" },
  { scene: "low_battery", label: "Low battery", explain: "Battery drops below 15% — the device sits rounds out.", target: "phone-04", group: "life" },
  { scene: "offline", label: "Go offline", explain: "The device disappears from the network.", target: "phone-06", group: "life" },
  { scene: "restore", label: "Restore", explain: "Clear weak signal and drift.", target: "", group: "life" },
];

const FALLBACK_IDS = ["phone-01", "phone-02", "phone-03", "phone-04", "phone-05", "phone-06", "esp32-01", "esp32-02", "esp32-03"];

export function AttackDrawer({
  s,
  cfg,
  enabled,
  open,
  onClose,
  toast,
}: {
  s: DashState;
  cfg: AppConfig;
  enabled: boolean;
  open: boolean;
  onClose: () => void;
  toast: ToastFn;
}) {
  const ids = s.deviceOrder.length ? s.deviceOrder : FALLBACK_IDS;
  const [targets, setTargets] = useState<Record<string, string>>(() =>
    Object.fromEntries(SCENES.filter((x) => x.target !== null).map((x) => [x.scene, x.target as string])),
  );
  const [busy, setBusy] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open, onClose]);

  async function fire(def: SceneDef) {
    const target = def.target === null ? undefined : targets[def.scene] || undefined;
    setBusy((b) => ({ ...b, [def.scene]: true }));
    toast("pending", `${def.label}${target ? ` → ${target}` : ""}: sent…`);
    const r = await postAdmin(cfg, `/fl/admin/attack/${def.scene}`, target ? { target } : {});
    setBusy((b) => ({ ...b, [def.scene]: false }));
    if (r.ok) toast("ok", `${def.label}: the server accepted it — watch the pipeline for the outcome`);
    else toast("error", `${def.label}: request failed${r.status ? ` (HTTP ${r.status})` : ""} ${r.text.slice(0, 120)}`);
  }

  const group = (g: SceneDef["group"]) =>
    SCENES.filter((x) => x.group === g).map((def) => (
      <div key={def.scene} className={`scene sc-${def.group}`}>
        <div className="scene-text">
          <button className="scene-btn" disabled={!enabled || busy[def.scene]} onClick={() => fire(def)}>
            {def.label}
          </button>
          <span className="scene-explain">{def.explain}</span>
        </div>
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
    ));

  return (
    <aside className={`drawer ${open ? "open" : ""}`} aria-hidden={!open} aria-label="Attack console">
      <header className="drawer-head">
        <div>
          <h2>Attack console</h2>
          <p>{enabled ? "Real requests to the real server. Outcomes appear in the pipeline." : "Disabled while playing a recording."}</p>
        </div>
        <button className="x" onClick={onClose} aria-label="Close attack console">
          ×
        </button>
      </header>
      <div className="drawer-scroll">
        <h3>Attacks</h3>
        {group("attack")}
        <h3>Real-life conditions</h3>
        {group("life")}
      </div>
    </aside>
  );
}
