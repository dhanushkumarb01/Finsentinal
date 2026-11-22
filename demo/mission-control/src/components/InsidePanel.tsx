import { useEffect, useState } from "react";
import type { DashState } from "../state/reducer";
import { signalBars, verdictOf } from "../state/reducer";
import { roundsOf, stateWords, stepsFor } from "../state/derive";
import { ACTIVITY_NAMES, GATE_ORDER } from "../types";
import { Battery, Esp32Art, PhoneArt, SignalBars } from "./Art";
import { GATE_LABELS } from "./Gate";

function ClassBars({ counts }: { counts: number[] | null }) {
  if (!counts) return <div className="empty small">No training report yet for this round.</div>;
  const max = Math.max(1, ...counts);
  return (
    <svg viewBox="0 0 420 200" className="chart-svg">
      {ACTIVITY_NAMES.map((name, i) => {
        const v = counts[i] ?? 0;
        const h = (v / max) * 140;
        const x = 12 + i * 68;
        return (
          <g key={name}>
            <rect x={x} y={160 - h} width="50" height={h} rx="3" fill="#38bdf8" opacity="0.85" />
            <text x={x + 25} y={154 - h} textAnchor="middle" className="svg-val">
              {v}
            </text>
            <text x={x + 25} y="178" textAnchor="middle" className="svg-axis" style={{ fontSize: 10 }}>
              {name}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LossCurve({ pts }: { pts: { round: number; loss: number }[] }) {
  if (!pts.length) return <div className="empty small">No loss reported yet.</div>;
  const W = 420,
    H = 200,
    L = 40,
    R = 12,
    T = 14,
    B = 30;
  const maxR = Math.max(...pts.map((p) => p.round), 1);
  const minR = Math.min(...pts.map((p) => p.round), 1);
  const maxL = Math.max(...pts.map((p) => p.loss)) * 1.1 || 1;
  const x = (r: number) => (maxR === minR ? (L + W - R) / 2 : L + ((r - minR) / (maxR - minR)) * (W - L - R));
  const y = (l: number) => T + (1 - l / maxL) * (H - T - B);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.round)} ${y(p.loss)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
      <line x1={L} y1={H - B} x2={W - R} y2={H - B} stroke="#334155" />
      <line x1={L} y1={T} x2={L} y2={H - B} stroke="#334155" />
      <text x={L - 6} y={T + 8} textAnchor="end" className="svg-axis" style={{ fontSize: 10 }}>
        {maxL.toFixed(2)}
      </text>
      <text x={L - 6} y={H - B} textAnchor="end" className="svg-axis">
        0
      </text>
      <path d={d} fill="none" stroke="#f472b6" strokeWidth="3" />
      {pts.map((p) => (
        <g key={p.round}>
          <circle cx={x(p.round)} cy={y(p.loss)} r="4.5" fill="#f472b6" />
          <text x={x(p.round)} y={y(p.loss) - 9} textAnchor="middle" className="svg-val">
            {p.loss.toFixed(2)}
          </text>
          <text x={x(p.round)} y={H - B + 18} textAnchor="middle" className="svg-axis">
            R{p.round}
          </text>
        </g>
      ))}
    </svg>
  );
}

function Heatmap({ grid }: { grid: number[][] | null }) {
  if (!grid || !grid.length) return <div className="empty small">No weight change reported yet.</div>;
  const rows = grid.length;
  const cols = Math.max(...grid.map((r) => r.length));
  return (
    <svg viewBox={`0 0 ${cols * 24} ${rows * 24}`} className="chart-svg heat">
      {grid.map((row, i) =>
        row.map((v, j) => {
          const c = Math.max(0, Math.min(1, v));
          // dark blue → cyan → yellow → red
          const hue = 210 - c * 210;
          const light = 14 + c * 46;
          return (
            <rect key={`${i}-${j}`} x={j * 24 + 1} y={i * 24 + 1} width="22" height="22" rx="3" fill={`hsl(${hue} 85% ${light}%)`}>
              <title>{c.toFixed(3)}</title>
            </rect>
          );
        }),
      )}
    </svg>
  );
}

export function InsidePanel({ s, id, onClose }: { s: DashState; id: string; onClose: () => void }) {
  const d = s.devices[id];
  const rounds = d ? roundsOf(d) : [];
  const latest = rounds.length ? rounds[rounds.length - 1] : null;
  const [sel, setSel] = useState<number | null>(null);
  const round = sel ?? latest;

  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);

  if (!d) return null;
  const training =
    round != null ? d.trainingByRound[round] ?? null : d.lastTrainingRound != null ? d.trainingByRound[d.lastTrainingRound] : null;
  const steps = round != null ? stepsFor(s, d, round) : [];
  const gateId = round != null ? d.gateByRound[round] : undefined;
  const gate = gateId ? s.gates[gateId] : d.lastGateId ? s.gates[d.lastGateId] : undefined;
  const v = gate ? verdictOf(gate) : null;

  return (
    <div className="overlay" onClick={onClose}>
      <div className="inside" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={`Inside ${d.info.label}`}>
        <header className="inside-head">
          <div className="inside-art">{d.info.kind === "wokwi" ? <Esp32Art led="off" /> : <PhoneArt tone="idle" />}</div>
          <div>
            <h2>
              Inside the device — {d.info.label} <span className="muted mono">({d.info.id})</span>
            </h2>
            <div className="inside-sub">
              <span className={`badge ${d.info.kind === "wokwi" ? "badge-wokwi" : "badge-virtual"}`}>
                {d.info.kind === "wokwi" ? "Wokwi-simulated ESP32" : "Virtual device"}
              </span>
              {d.info.subject >= 0 && <span>HAR subject {d.info.subject}</span>}
              <Battery pct={d.battery} />
              <SignalBars bars={signalBars(d.signal_dbm)} dbm={d.signal_dbm} />
              <span>{stateWords(d)}</span>
              {d.hacked && <span className="tag tag-bad">HACKED</span>}
            </div>
          </div>
          <div className="round-pick">
            <span className="muted">Round</span>
            {rounds.length === 0 && <span className="muted">none yet</span>}
            {rounds.map((r) => (
              <button key={r} className={`chip-btn ${r === round ? "on" : ""}`} onClick={() => setSel(r)}>
                {r}
              </button>
            ))}
          </div>
          <button className="close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="steps">
          {steps.length === 0 && <div className="empty small">No round activity from this device yet.</div>}
          {steps.map((st, i) => (
            <div key={st.key} className={`step step-${st.state}`}>
              <div className="step-dot">{i + 1}</div>
              <div className="step-label">{st.label}</div>
              {st.note && <div className="step-note">{st.note}</div>}
            </div>
          ))}
        </div>

        <div className="inside-grid">
          <div className="inside-box">
            <h3>Data sample {training ? `· ${training.samples} samples (round ${training.round})` : ""}</h3>
            <ClassBars counts={training?.class_counts ?? null} />
          </div>
          <div className="inside-box">
            <h3>Training loss across rounds</h3>
            <LossCurve pts={d.losses} />
          </div>
          <div className="inside-box">
            <h3>
              Weight change (|Δw| of first layer, blue = small, red = large){" "}
              {training ? <span className="muted">· L2 {training.delta_l2.toFixed(3)}</span> : null}
            </h3>
            <Heatmap grid={training?.weight_delta ?? null} />
          </div>
          <div className="inside-box">
            <h3>
              Last gate checks {gate ? <span className="muted">· round {gate.ev.round}</span> : null}
            </h3>
            {!gate ? (
              <div className="empty small">No gate result for this device yet.</div>
            ) : (
              <div className="checks-list">
                {GATE_ORDER.map((k) => (
                  <div key={k} className={`check-row c-${gate.ev.checks[k]}`}>
                    <span className="check-dot" />
                    <span>{GATE_LABELS[k].title}</span>
                    <span className="check-state">{gate.ev.checks[k]}</span>
                  </div>
                ))}
                <div className={`verdict v-${v?.kind}`}>{v?.text}</div>
                <div className="muted small">
                  HTTP {gate.ev.http_status}
                  {gate.ev.risk_score != null && ` · risk ${gate.ev.risk_score.toFixed(2)}`}
                  {gate.ev.severity && ` · ${gate.ev.severity}`}
                  {gate.actor && ` · decided by ${gate.actor}`}
                </div>
                {gate.ev.reasons.length > 0 && <div className="small">{gate.ev.reasons.join("; ")}</div>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
