import type { DashState } from "../state/reducer";
import { normAccuracy } from "../state/reducer";

export function GlobalChart({ s }: { s: DashState }) {
  const evalRounds = Object.keys(s.evals).map(Number);
  const doneRounds = Object.keys(s.roundsCompleted).map(Number);
  const maxSeen = Math.max(0, ...evalRounds, ...doneRounds, s.round ?? 0);
  const total = Math.max(s.totalRounds ?? 0, maxSeen, 1);
  const fleet = Math.max(1, s.deviceOrder.length);

  const W = 680,
    H = 262,
    L = 54,
    R = 12,
    T = 24,
    chartB = 178,
    barTop = 196,
    barB = 240;
  // Round 0 (the initial model, if the coordinator evaluates it) gets its own slot.
  const minR = Math.min(1, ...evalRounds, ...doneRounds);
  const slots = total - minR + 1;
  const x = (r: number) => L + ((r - minR + 0.5) / slots) * (W - L - R);
  const y = (a: number) => T + (1 - a) * (chartB - T);
  const pts = evalRounds.sort((a, b) => a - b).map((r) => ({ r, a: normAccuracy(s.evals[r].accuracy) }));
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.r).toFixed(1)} ${y(p.a).toFixed(1)}`).join(" ");
  const area = pts.length > 1 ? `${path} L${x(pts[pts.length - 1].r)} ${chartB} L${x(pts[0].r)} ${chartB} Z` : "";
  const barW = Math.max(4, Math.min(34, ((W - L - R) / slots) * 0.45));

  return (
    <section className="panel chart">
      <h2 className="panel-title">
        Global model <span className="muted">· test accuracy per round</span>
      </h2>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="accfill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#22d3ee" stopOpacity="0.35" />
            <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="#1e293b" strokeDasharray={g === 0 ? undefined : "4 5"} />
            <text x={L - 8} y={y(g) + 5} textAnchor="end" className="svg-axis">
              {g * 100}%
            </text>
          </g>
        ))}
        {s.driftDropRounds.map((r) => (
          <g key={`drift-${r}`} className="drift-mark">
            <line x1={x(r)} x2={x(r)} y1={T} y2={barB} stroke="#c084fc" strokeWidth="2" strokeDasharray="6 4" />
            <text x={x(r)} y={barTop - 4} textAnchor="middle" className="svg-drift">
              drift dropout
            </text>
          </g>
        ))}
        {area && <path d={area} fill="url(#accfill)" />}
        {path && <path d={path} fill="none" stroke="#22d3ee" strokeWidth="4" strokeLinejoin="round" className="acc-line" />}
        {pts.map((p) => (
          <g key={p.r}>
            <circle cx={x(p.r)} cy={y(p.a)} r="6" fill="#0b1220" stroke="#22d3ee" strokeWidth="3" />
            <text x={x(p.r)} y={y(p.a) - 13} textAnchor="middle" className="svg-val big">
              {(p.a * 100).toFixed(1)}%
            </text>
          </g>
        ))}
        <text x={L - 8} y={barTop + 10} textAnchor="end" className="svg-axis">
          devices
        </text>
        {Array.from({ length: slots }, (_, i) => i + minR).map((r) => {
          const rc = s.roundsCompleted[r];
          const n = rc ? rc.participants.length : 0;
          const h = rc ? Math.max(2, (n / fleet) * (barB - barTop)) : 0;
          return (
            <g key={`b${r}`}>
              {rc && <rect x={x(r) - barW / 2} y={barB - h} width={barW} height={h} rx="2" fill="#475569" />}
              {rc && (
                <text x={x(r) + barW / 2 + 4} y={barB - 2} className="svg-axis small">
                  {n}
                </text>
              )}
              {(slots <= 20 || r % Math.ceil(slots / 10) === 0) && (
                <text x={x(r)} y={H - 2} textAnchor="middle" className="svg-axis">
                  {r}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        <span>
          <i className="lg lg-acc" /> accuracy
        </span>
        <span>
          <i className="lg lg-bar" /> devices aggregated
        </span>
        <span>
          <i className="lg lg-drift" /> round the drifting device sat out
        </span>
      </div>
    </section>
  );
}
