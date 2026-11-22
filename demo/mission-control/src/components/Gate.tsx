import type { DashState } from "../state/reducer";
import { effectiveStatus, verdictOf } from "../state/reducer";
import { GATE_ORDER, type GateChecks } from "../types";

export const GATE_LABELS: Record<keyof GateChecks, { title: string; sub: string }> = {
  untampered: { title: "Untampered", sub: "SHA-256 matches" },
  sender: { title: "Sender verified", sub: "Ed25519 signature" },
  shape: { title: "Correct shape & round", sub: "schema + policy" },
  replay: { title: "Not a replay", sub: "never seen before" },
  normal: { title: "Looks normal", sub: "anomaly detector" },
};

const STATUS_WORDS: Record<string, string> = {
  pending_approval: "waiting for review",
  quarantined: "quarantined",
  approved: "approved",
  rejected: "rejected",
};

export function Gate({ s }: { s: DashState }) {
  const g = s.lastGateId ? s.gates[s.lastGateId] : null;
  const v = g ? verdictOf(g) : null;
  const label = g ? (g.ev.source === "attack" ? "Attacker request" : s.devices[g.ev.device]?.info.label ?? g.ev.device) : null;
  const st = g ? effectiveStatus(g) : null;
  return (
    <section className="panel gate">
      <h2 className="panel-title">
        Security gate <span className="muted">· FinSentinel</span>
      </h2>
      <div className="gate-who">
        {g ? (
          <>
            <span className="gate-dev">{label}</span>
            <span className="muted"> · round {g.ev.round} · HTTP {g.ev.http_status}</span>
          </>
        ) : (
          <span className="muted">No update checked yet</span>
        )}
      </div>
      <div className="lights" key={s.lastGateId ?? "none"}>
        {GATE_ORDER.map((k, i) => {
          const state = g ? g.ev.checks[k] : "skip";
          return (
            <div key={k} className={`light l-${state} ${g ? "animate" : ""}`} style={{ animationDelay: `${i * 0.24}s` }}>
              <span className="bulb" style={{ animationDelay: `${i * 0.24}s` }} />
              <span className="light-text">
                <span className="light-title">{GATE_LABELS[k].title}</span>
                <span className="light-sub">{GATE_LABELS[k].sub}</span>
              </span>
              <span className="light-state">{g ? state.toUpperCase() : ""}</span>
            </div>
          );
        })}
      </div>
      {g && (
        <div className="gate-foot">
          <div className={`verdict big v-${v?.kind}`}>{v?.text}</div>
          <div className="gate-nums">
            <span>
              Risk <b>{g.ev.risk_score != null ? g.ev.risk_score.toFixed(2) : "—"}</b>
            </span>
            <span>
              Severity <b>{g.ev.severity ?? "—"}</b>
            </span>
            <span>
              Approval <b>{st ? STATUS_WORDS[st] ?? st : "—"}</b>
              {g.actor ? <span className="muted"> by {g.actor}</span> : null}
            </span>
          </div>
          {(g.ev.reasons.length > 0 || g.ev.detail) && (
            <div className="gate-reasons" title={[...g.ev.reasons, g.ev.detail].join(" · ")}>
              {g.ev.reasons.length ? g.ev.reasons.join(" · ") : g.ev.detail}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
