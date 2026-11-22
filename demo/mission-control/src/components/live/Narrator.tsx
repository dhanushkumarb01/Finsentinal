import type { DashState } from "../../state/reducer";
import { narrate } from "../../state/narrator";
import { deviceLabelOf } from "../../state/derive";

const PHASES: { id: "open" | "deciding" | "aggregating"; label: string; hint: string }[] = [
  { id: "open", label: "Open", hint: "devices train and send" },
  { id: "deciding", label: "Deciding", hint: "waiting for decisions" },
  { id: "aggregating", label: "Aggregating", hint: "averaging approved updates" },
];

export function PhaseIndicator({ s }: { s: DashState }) {
  const rp = s.roundPhase;
  const idx = rp ? (rp.phase === "done" ? 3 : PHASES.findIndex((p) => p.id === rp.phase)) : -1;
  const wazuh = s.engine?.engine === "wazuh";
  const waiting = rp?.phase === "deciding" ? rp.waitingFor : [];
  return (
    <div className="phase-ind" aria-label="Round phase">
      <div className="phase-round">
        <span className="phase-k">Round</span>
        <span className="phase-v">
          {s.round ?? "—"}
          <small> / {s.totalRounds ?? "—"}</small>
        </span>
      </div>
      <div className="phase-steps">
        {PHASES.map((p, i) => {
          const st = idx < 0 ? "todo" : i < idx ? "done" : i === idx ? "now" : "todo";
          const inferred = rp != null && !rp.explicit && p.id !== "open";
          return (
            <div key={p.id} className={`pstep ps-${st} ${p.id === "deciding" ? "ps-deciding" : ""}`} title={p.hint}>
              <i />
              <span>{p.label}</span>
              {inferred && p.id === "deciding" && !wazuh && <em>no Wazuh</em>}
            </div>
          );
        })}
        {rp?.phase === "done" && <div className="pstep ps-complete">Round complete</div>}
      </div>
      {waiting.length > 0 && (
        <div className="phase-wait">
          Waiting for {wazuh ? "Wazuh" : "a decision"}: <b>{waiting.map((d) => deviceLabelOf(s, d)).join(", ")}</b>
        </div>
      )}
    </div>
  );
}

export function Narrator({ s }: { s: DashState }) {
  const n = narrate(s);
  const steps = n.headline.split(" → ");
  return (
    <section className={`narrator tone-${n.tone}`} aria-live="polite">
      <div className="narr-icon" aria-hidden>
        <i />
      </div>
      <div className="narr-body">
        <div className="narr-kicker">What's happening now</div>
        <div className="narr-head" key={n.headline}>
          {steps.map((p, i) => (
            <span key={i} className="narr-step">
              {i > 0 && <span className="narr-arrow">→</span>}
              {p}
            </span>
          ))}
        </div>
        {n.detail && (
          <div className="narr-detail" key={`d-${n.detail}`} title={n.detail}>
            {n.detail}
          </div>
        )}
      </div>
      <PhaseIndicator s={s} />
    </section>
  );
}
