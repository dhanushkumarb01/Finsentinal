import type { DashState, DeviceView } from "../state/reducer";
import { signalBars, verdictOf } from "../state/reducer";
import { deviceGate, stateWords } from "../state/derive";
import { Battery, Esp32Art, PhoneArt, SignalBars, Sparkline, type Tone } from "./Art";

const NET_WORDS: Record<string, string> = { wifi: "Wi-Fi", "4g": "4G", "3g": "3G", "wokwi-guest": "Wokwi-GUEST Wi-Fi" };

function toneOf(d: DeviceView): Tone {
  if (!d.online || d.state === "offline") return "off";
  if (d.hacked) return "bad";
  if (d.state === "skipped" || d.state === "model_rejected" || d.state === "error") return "warn";
  if (["downloading", "training", "signing", "sending"].includes(d.state)) return "busy";
  if (d.state === "done") return "ok";
  return "idle";
}

export function DeviceCard({ s, d, onOpen }: { s: DashState; d: DeviceView; onOpen: () => void }) {
  const tone = toneOf(d);
  const gate = deviceGate(s, d);
  const verdict = gate ? verdictOf(gate) : null;
  const busy = ["downloading", "training", "signing", "sending"].includes(d.state) && d.online;
  const led = verdict
    ? verdict.kind === "accepted" || verdict.kind === "checked"
      ? "green"
      : verdict.kind === "review"
        ? "amber"
        : "red"
    : "off";
  const cls = [
    "device-card",
    `tone-${tone}`,
    d.hacked ? "hacked" : "",
    !d.online || d.state === "offline" ? "offline" : "",
    d.state === "sending" ? "sending" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button className={cls} onClick={onOpen} title={`Open "inside the device" for ${d.info.label}`}>
      <div className="dc-art">
        {d.info.kind === "wokwi" ? <Esp32Art led={led} busy={busy} /> : <PhoneArt tone={tone} busy={busy} />}
      </div>
      <div className="dc-body">
        <div className="dc-title">
          <span className="dc-label" title={d.info.label}>{d.info.label}</span>
        </div>
        <div className="dc-badges">
          <span className={`badge ${d.info.kind === "wokwi" ? "badge-wokwi" : "badge-virtual"}`}>
            {d.info.kind === "wokwi" ? "Wokwi-simulated ESP32" : "Virtual device"}
          </span>
          {d.hacked && <span className="tag tag-bad">HACKED</span>}
        </div>
        <div className="dc-meta">
          <Battery pct={d.battery} />
          <SignalBars bars={signalBars(d.signal_dbm)} dbm={d.signal_dbm} />
          <span className="net">{NET_WORDS[d.network] ?? d.network}</span>
        </div>
        <div className={`dc-state st-${d.state}`}>
          {busy && <span className="mini-spin" aria-hidden />}
          <span className="dc-state-text">{stateWords(d)}</span>
          {d.round != null && <span className="dc-round">R{d.round}</span>}
        </div>
        <div className="dc-spark">
          <span className="dc-spark-label">loss</span>
          <Sparkline values={d.losses.map((l) => l.loss)} />
          <span className="dc-spark-val">{d.losses.length ? d.losses[d.losses.length - 1].loss.toFixed(2) : "—"}</span>
        </div>
        <div className={`verdict v-${verdict?.kind ?? "none"}`} title="Latest security-gate result for this device">
          {verdict && gate ? `R${gate.ev.round} · ${verdict.text}` : "No gate result yet"}
        </div>
      </div>
    </button>
  );
}

export function DeviceWall({ s, onOpen }: { s: DashState; onOpen: (id: string) => void }) {
  const ids = s.deviceOrder;
  const cols = ids.length <= 4 ? 2 : ids.length <= 9 ? 3 : 4;
  const rows = Math.max(1, Math.ceil(ids.length / cols));
  return (
    <section className="panel wall">
      <h2 className="panel-title">
        Device fleet <span className="muted">· {ids.length} devices · click one to look inside</span>
      </h2>
      {ids.length === 0 ? (
        <div className="empty">Waiting for the run to start…</div>
      ) : (
        <div
          className="wall-grid"
          style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: `repeat(${rows}, 1fr)` }}
        >
          {ids.map((id) => (
            <DeviceCard key={id} s={s} d={s.devices[id]} onOpen={() => onOpen(id)} />
          ))}
        </div>
      )}
    </section>
  );
}
