import type { DashState } from "../../state/reducer";
import { fmtRisk } from "../../state/reducer";
import { spotlightGate } from "../../state/derive";
import { GATE_ORDER, type CheckState, type GateChecks } from "../../types";

export const GATE_LABELS: Record<keyof GateChecks, { title: string; sub: string }> = {
  untampered: { title: "Untampered", sub: "fingerprint (SHA-256) matches" },
  sender: { title: "Sender verified", sub: "device's own Ed25519 signature" },
  shape: { title: "Correct shape & round", sub: "right model, right round" },
  replay: { title: "Not a replay", sub: "never seen this update before" },
  normal: { title: "Looks normal", sub: "anomaly rules, calibrated on clean data" },
};

const STATE_WORDS: Record<CheckState, string> = { pass: "Pass", fail: "Fail", warn: "Warning", skip: "Not reached" };

export function GateStage({ s }: { s: DashState }) {
  const spot = spotlightGate(s);
  const g = spot?.g ?? null;
  const who = g ? (g.ev.source === "attack" ? "Attacker request" : s.devices[g.ev.device]?.info.label ?? g.ev.device) : null;
  const blocked = g ? g.ev.http_status >= 400 : false;
  const flagged = g ? !blocked && (g.ev.checks.normal === "fail" || g.ev.checks.normal === "warn") : false;
  const result = !g ? null : blocked ? "Blocked" : flagged ? "Flagged & reported" : "All checks passed";
  return (
    <div className="stage-body gate-body">
      <div className="who">
        {g ? (
          <>
            <b>{who}</b>
            <span>
              round {g.ev.round} · HTTP {g.ev.http_status}
            </span>
          </>
        ) : (
          <span className="muted">No update checked yet</span>
        )}
      </div>
      <ol className="lights" key={spot?.key ?? "none"}>
        {GATE_ORDER.map((k, i) => {
          const st: CheckState = g ? g.ev.checks[k] : "skip";
          return (
            <li key={k} className={`light l-${g ? st : "idle"}`} style={{ animationDelay: `${i * 0.18}s` }}>
              <span className="bulb">{i + 1}</span>
              <span className="light-text">
                <span className="light-title">{GATE_LABELS[k].title}</span>
                <span className="light-sub">{GATE_LABELS[k].sub}</span>
              </span>
              <span className="light-state">{g ? STATE_WORDS[st] : ""}</span>
            </li>
          );
        })}
      </ol>
      {g && (
        <div className={`gate-result ${blocked ? "r-bad" : flagged ? "r-warn" : "r-ok"}`}>
          <div className="gr-head">{result}</div>
          <div className="gr-nums">
            <span>
              risk <b>{fmtRisk(g.ev.risk_score)}</b>
            </span>
            <span>
              severity <b>{g.ev.severity ?? "—"}</b>
            </span>
          </div>
          {(g.ev.reasons.length > 0 || g.ev.detail) && (
            <div className="gr-reason" title={[...g.ev.reasons, g.ev.detail].join(" · ")}>
              {g.ev.reasons[0] ?? g.ev.detail}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
