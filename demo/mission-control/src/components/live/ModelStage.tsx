import type { DashState } from "../../state/reducer";
import { latestEval, normAccuracy } from "../../state/reducer";
import { deviceLabelOf, fmtClock } from "../../state/derive";

export function AccuracyChart({ s }: { s: DashState }) {
  const evalRounds = Object.keys(s.evals).map(Number).sort((a, b) => a - b);
  const doneRounds = Object.keys(s.roundsCompleted).map(Number);
  const maxSeen = Math.max(0, ...evalRounds, ...doneRounds, s.round ?? 0);
  const total = Math.max(s.totalRounds ?? 0, maxSeen, 1);
  const fleet = Math.max(1, s.deviceOrder.length);
  const minR = Math.min(1, ...evalRounds, ...doneRounds);
  const slots = total - minR + 1;

  const W = 330,
    H = 230,
    L = 38,
    R = 8,
    T = 12,
    chartB = 150,
    barTop = 166,
    barB = 204;
  const x = (r: number) => L + ((r - minR + 0.5) / slots) * (W - L - R);
  const y = (a: number) => T + (1 - a) * (chartB - T);
  const pts = evalRounds.map((r) => ({ r, a: normAccuracy(s.evals[r].accuracy) }));
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.r).toFixed(1)} ${y(p.a).toFixed(1)}`).join(" ");
  const area = pts.length > 1 ? `${path} L${x(pts[pts.length - 1].r)} ${chartB} L${x(pts[0].r)} ${chartB} Z` : "";
  const barW = Math.max(3, Math.min(18, ((W - L - R) / slots) * 0.5));
  const labelEvery = Math.max(1, Math.ceil(slots / 8));
  const last = pts[pts.length - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="acc-svg" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Global model accuracy per round">
      <defs>
        <linearGradient id="accfill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#7dd3fc" stopOpacity="0.32" />
          <stop offset="1" stopColor="#7dd3fc" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} className="grid" />
          <text x={L - 6} y={y(g) + 4} textAnchor="end" className="ax">
            {g * 100}%
          </text>
        </g>
      ))}
      {s.driftDropRounds.map((r) => (
        <g key={`drift-${r}`}>
          <line x1={x(r)} x2={x(r)} y1={T} y2={barB} className="drift-line" />
        </g>
      ))}
      {area && <path d={area} fill="url(#accfill)" />}
      {path && <path d={path} className="acc-line" />}
      {pts.map((p) => (
        <circle key={p.r} cx={x(p.r)} cy={y(p.a)} r={p === last ? 5 : 3.2} className={p === last ? "acc-dot last" : "acc-dot"} />
      ))}
      {Array.from({ length: slots }, (_, i) => i + minR).map((r) => {
        const rc = s.roundsCompleted[r];
        const n = rc ? rc.participants.length : 0;
        const h = rc ? Math.max(2, (n / fleet) * (barB - barTop)) : 0;
        return (
          <g key={`b${r}`}>
            {rc && <rect x={x(r) - barW / 2} y={barB - h} width={barW} height={h} rx="2" className="part-bar" />}
            {(r - minR) % labelEvery === 0 && (
              <text x={x(r)} y={H - 6} textAnchor="middle" className="ax">
                {r}
              </text>
            )}
          </g>
        );
      })}
      <text x={L - 6} y={barB - 4} textAnchor="end" className="ax small">
        devices
      </text>
    </svg>
  );
}

const SRC_CLS: Record<string, string> = {
  FinSentinel: "src-fs",
  Wazuh: "src-wz",
  Coordinator: "src-co",
  Attack: "src-at",
  "Drift monitor": "src-dm",
  Operator: "src-op",
};

export function ModelStage({ s }: { s: DashState }) {
  const ev = latestEval(s);
  const acc = ev ? normAccuracy(ev.accuracy) * 100 : null;
  const prev = ev ? s.evals[ev.round - 1] : undefined;
  const delta = ev && prev ? (normAccuracy(ev.accuracy) - normAccuracy(prev.accuracy)) * 100 : null;
  const alerts = s.alerts.slice(0, 4);
  return (
    <div className="stage-body model-body">
      <div className="acc-head">
        <div className="acc-big">{acc == null ? "—" : `${acc.toFixed(1)}%`}</div>
        <div className="acc-side">
          <span>test accuracy</span>
          {delta != null && (
            <span className={delta >= 0 ? "up" : "down"}>
              {delta >= 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(1)} since last round
            </span>
          )}
          {ev && <span className="muted">after round {ev.round}</span>}
        </div>
      </div>
      <div className="acc-chart">
        <AccuracyChart s={s} />
      </div>
      <div className="legend">
        <span>
          <i className="lg-acc" /> accuracy
        </span>
        <span>
          <i className="lg-bar" /> devices averaged
        </span>
        {s.driftDropRounds.length > 0 && (
          <span>
            <i className="lg-drift" /> drifting device sat out
          </span>
        )}
      </div>
      <div className="mini-feed">
        <div className="mini-title">Latest alerts</div>
        {alerts.length === 0 && <div className="muted small">No alerts yet</div>}
        {alerts.map((a) => (
          <div key={a.id} className={`mf-item sev-${a.severity}`}>
            <span className={`mf-src ${SRC_CLS[a.source] ?? ""}`}>{a.source === "Drift monitor" ? "Research monitor" : a.source}</span>
            <span className="mf-title" title={`${a.title} — ${a.detail}`}>
              {a.device ? `${deviceLabelOf(s, a.device)}: ` : ""}
              {a.title}
            </span>
            <span className="mf-time mono">{fmtClock(a.t)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
