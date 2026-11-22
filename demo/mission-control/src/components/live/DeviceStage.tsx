import type { DashState, DeviceView } from "../../state/reducer";
import { signalBars, verdictOf } from "../../state/reducer";
import { deviceGate, stateWords } from "../../state/derive";
import { Battery, Esp32Art, PhoneArt, SignalBars, type Tone } from "../Art";

export const NET_WORDS: Record<string, string> = {
  wifi: "Wi-Fi",
  "4g": "4G",
  "3g": "3G",
  weak: "Weak signal",
  "wokwi-guest": "Wi-Fi",
};

const BUSY = ["downloading", "training", "signing", "sending"];

export function toneOf(d: DeviceView): Tone {
  if (d.suspended) return "bad";
  if (!d.online || d.state === "offline") return "off";
  if (d.hacked) return "bad";
  if (d.state === "skipped" || d.state === "model_rejected" || d.state === "error") return "warn";
  if (BUSY.includes(d.state)) return "busy";
  if (d.state === "done") return "ok";
  return "idle";
}

export function DeviceCard({ s, d, onOpen }: { s: DashState; d: DeviceView; onOpen: () => void }) {
  const tone = toneOf(d);
  const gate = deviceGate(s, d);
  const verdict = gate ? verdictOf(gate) : null;
  const busy = BUSY.includes(d.state) && d.online && !d.suspended;
  const led = d.suspended
    ? "red"
    : verdict
      ? verdict.kind === "accepted" || verdict.kind === "checked"
        ? "green"
        : verdict.kind === "review"
          ? "amber"
          : "red"
      : "off";
  const cls = [
    "dev",
    `tone-${tone}`,
    d.hacked ? "hacked" : "",
    d.suspended ? "suspended" : "",
    !d.online || d.state === "offline" ? "offline" : "",
    d.state === "sending" ? "sending" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button className={cls} onClick={onOpen} title={`Open "inside the device" for ${d.info.label}`}>
      <div className="dev-top">
        <div className="dev-art">
          {d.info.kind === "wokwi" ? <Esp32Art led={led} busy={busy} /> : <PhoneArt tone={tone} busy={busy} />}
        </div>
        <div className="dev-id">
          <div className="dev-name">{d.info.label}</div>
          {d.hacked && !d.suspended && <span className="flag flag-bad">Hacked</span>}
          {d.suspended && (
            <span className="flag flag-bad" title={d.suspendedRule ? `Suspended by Wazuh rule ${d.suspendedRule}` : "Suspended by Wazuh"}>
              Suspended
            </span>
          )}
        </div>
      </div>
      <span className={`kind ${d.info.kind}`}>{d.info.kind === "wokwi" ? "Wokwi-simulated ESP32" : "Virtual device"}</span>
      <div className="dev-meta">
        <Battery pct={d.battery} />
        <SignalBars bars={signalBars(d.signal_dbm)} dbm={d.signal_dbm} compact />
        <span className="net">{NET_WORDS[d.network] ?? d.network}</span>
      </div>
      <div className={`dev-step st-${d.state}`}>
        {busy && <span className="spin-dot" aria-hidden />}
        <span className="dev-step-text">{stateWords(d)}</span>
        {d.round != null && <span className="dev-round">R{d.round}</span>}
      </div>
      <div className={`dev-verdict v-${verdict?.kind ?? "none"}`} title="Latest security result for this device">
        {verdict && gate ? (
          <>
            <span className="vd-dot" />
            {verdict.text}
            <span className="vd-round">R{gate.ev.round}</span>
          </>
        ) : (
          <span className="muted">No result yet</span>
        )}
      </div>
    </button>
  );
}

export function DeviceStage({ s, onOpen }: { s: DashState; onOpen: (id: string) => void }) {
  const ids = s.deviceOrder;
  const cols = ids.length <= 4 ? 2 : 3;
  const rows = Math.max(1, Math.ceil(ids.length / cols));
  return (
    <div className="stage-body dev-body">
      {ids.length === 0 ? (
        <div className="empty">Waiting for the fleet to come online…</div>
      ) : (
        <div className="dev-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))` }}>
          {ids.map((id) => (
            <DeviceCard key={id} s={s} d={s.devices[id]} onOpen={() => onOpen(id)} />
          ))}
        </div>
      )}
    </div>
  );
}
