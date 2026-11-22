import type { DashState, Flight } from "../state/reducer";
import { pendingReview, verdictOf } from "../state/reducer";
import { LockIcon } from "./Art";

const RESOLVED_VISIBLE_S = 5;
const OPEN_VISIBLE_S = 25;

export function visibleFlights(s: DashState): Flight[] {
  const now = s.lastT ?? 0;
  return s.flights.filter((f) =>
    f.outcome == null ? now - f.t <= OPEN_VISIBLE_S : now - (f.tEnd ?? f.t) <= RESOLVED_VISIBLE_S,
  );
}

function lane(seq: number): number {
  return (seq * 7) % 4;
}

export function Tunnel({ s }: { s: DashState }) {
  const flights = visibleFlights(s);
  let accepted = 0,
    quarantined = 0,
    blocked = 0;
  for (const g of Object.values(s.gates)) {
    const k = verdictOf(g).kind;
    if (k === "accepted") accepted++;
    else if (k === "quarantined" || k === "rejected") quarantined++;
    else if (k === "blocked") blocked++;
  }
  const held = pendingReview(s).length;
  return (
    <section className="panel tunnel">
      <h2 className="panel-title">
        Secure tunnel <span className="muted">· TLS + signed envelopes</span>
      </h2>
      <div className="tunnel-stage">
        <div className="tube">
          <span className="tube-label">devices →</span>
        </div>
        <div className="gate-bar">
          <span>GATE</span>
        </div>
        <div className="box server-box">
          <div className="box-title">Server</div>
          <div className="box-count">{accepted}</div>
          <div className="box-sub">accepted</div>
        </div>
        <div className="box quarantine-box">
          <div className="box-title">Quarantine</div>
          <div className="box-count">{quarantined}</div>
          <div className="box-sub">{held ? `${held} waiting review` : "held updates"}</div>
        </div>
        <div className="blocked-count">
          blocked <b>{blocked}</b>
        </div>
        {flights.map((f) => {
          const stage = f.outcome == null ? "fly" : `end-${f.outcome}`;
          return (
            <div
              key={f.id}
              className={`env ${f.attacker ? "attacker" : ""} ${stage}`}
              style={{ ["--lane" as string]: lane(f.seq) }}
            >
              <span className="env-body">
                <LockIcon />
                <span className="env-label">{f.label}</span>
              </span>
              {f.outcome === "blocked" && <span className="burst">{f.outcomeLabel}</span>}
              {f.outcome != null && f.outcome !== "blocked" && <span className="env-tag">{f.outcomeLabel}</span>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
