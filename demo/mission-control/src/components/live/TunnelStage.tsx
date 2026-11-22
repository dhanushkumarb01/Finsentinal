import type { DashState, Flight } from "../../state/reducer";
import { LockIcon } from "../Art";

const RESOLVED_VISIBLE_S = 6;
const OPEN_VISIBLE_S = 25;
const MAX_SHOWN = 7;

export function visibleFlights(s: DashState): Flight[] {
  const now = s.lastT ?? 0;
  return s.flights.filter((f) =>
    f.outcome == null ? now - f.t <= OPEN_VISIBLE_S : now - (f.tEnd ?? f.t) <= RESOLVED_VISIBLE_S,
  );
}

const OUTCOME_WORDS: Record<string, string> = {
  pass: "through the gate",
  held: "held for a decision",
  quarantine: "quarantined",
  blocked: "blocked",
};

export function TunnelStage({ s }: { s: DashState }) {
  const flights = visibleFlights(s).slice(-MAX_SHOWN).reverse();
  let host = "";
  if (s.tunnel?.url) {
    try {
      host = new URL(s.tunnel.url).host;
    } catch {
      host = s.tunnel.url;
    }
  }
  const inFlight = flights.filter((f) => f.outcome == null).length;
  return (
    <div className="stage-body tun-body">
      <div className="tun-emblem">
        <div className="tun-lock">
          <LockIcon />
        </div>
        <div className="tun-facts">
          <span>TLS encrypted</span>
          <span>Signed per device</span>
        </div>
      </div>
      <div className="tun-count">
        <b>{inFlight}</b> in transit
      </div>
      <div className="tun-list">
        {flights.length === 0 && <div className="tun-empty">Quiet — no updates on the wire</div>}
        {flights.map((f) => (
          <div
            key={f.id}
            className={`env ${f.attacker ? "attacker" : ""} ${f.outcome ? `out-${f.outcome}` : "flying"}`}
            title={f.outcomeLabel ?? "travelling to the gate"}
          >
            <LockIcon />
            <span className="env-label">{f.label}</span>
            <span className="env-out">{f.outcome ? (f.outcome === "blocked" ? f.outcomeLabel : OUTCOME_WORDS[f.outcome]) : "→"}</span>
          </div>
        ))}
      </div>
      <div className={`tun-status ${s.tunnel ? (s.tunnel.status === "up" ? "ok" : "bad") : ""}`}>
        <i />
        {s.tunnel ? (
          <span>
            Public tunnel {s.tunnel.status}
            {host && <small className="mono">{host}</small>}
            {s.tunnel.beacon_published && <small>signed beacon published</small>}
          </span>
        ) : (
          <span>Local links (Toxiproxy)</span>
        )}
      </div>
    </div>
  );
}
