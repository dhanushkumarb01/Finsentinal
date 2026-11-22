import type { DashState } from "../../state/reducer";
import { awaitingDecision, deciderOfActor } from "../../state/reducer";
import { deviceLabelOf, spotlightDecision } from "../../state/derive";
import type { WazuhDecisionKind } from "../../types";

export const DECISION_WORDS: Record<WazuhDecisionKind, { word: string; cls: string }> = {
  approve: { word: "Approve", cls: "d-ok" },
  quarantine: { word: "Quarantine", cls: "d-bad" },
  reject: { word: "Reject", cls: "d-bad" },
  suspend_device: { word: "Suspend device", cls: "d-bad" },
  alert: { word: "Alert — human decides", cls: "d-warn" },
};

const APPROVAL_CLS: Record<string, string> = { approved: "d-ok", quarantined: "d-bad", rejected: "d-bad" };

export function EngineBadge({ s }: { s: DashState }) {
  const e = s.engine;
  const wazuhDown = s.components.wazuh?.status === "down";
  if (e?.engine === "wazuh" && !wazuhDown)
    return (
      <div className="engine e-wazuh" title={e.detail}>
        <i /> Wazuh SIEM is deciding
      </div>
    );
  if (e?.engine === "wazuh" && wazuhDown)
    return (
      <div className="engine e-off" title={e.detail}>
        <i /> Wazuh reported down
      </div>
    );
  if (e?.engine === "coordinator")
    return (
      <div className="engine e-off" title={e.detail}>
        <i /> Wazuh offline — coordinator decides
      </div>
    );
  if (s.run)
    return (
      <div className="engine e-off" title="This run has no decision_engine event (v1 recording)">
        <i /> Decided by coordinator (Wazuh not running)
      </div>
    );
  return (
    <div className="engine e-idle">
      <i /> Waiting for the run
    </div>
  );
}

export function WazuhStage({ s }: { s: DashState }) {
  const spot = spotlightDecision(s);
  const waitingIds =
    s.roundPhase?.phase === "deciding" && s.roundPhase.waitingFor.length
      ? s.roundPhase.waitingFor
      : awaitingDecision(s)
          .filter((g) => g.ev.round === s.round)
          .map((g) => g.ev.device);
  const counts = { approve: 0, quarantine: 0, reject: 0, suspend_device: 0, alert: 0 } as Record<WazuhDecisionKind, number>;
  for (const w of s.wazuhDecisions) counts[w.decision]++;
  const isWazuh = s.wazuhDecisions.length > 0 || s.engine?.engine === "wazuh";

  return (
    <div className="stage-body wz-body">
      <EngineBadge s={s} />
      {!spot && <div className="wz-empty">No decision yet</div>}
      {spot?.kind === "wazuh" && (
        <div className={`wz-card ${DECISION_WORDS[spot.w.decision].cls}`} key={spot.w.seq}>
          <div className="wz-who">
            <b>{deviceLabelOf(s, spot.w.device)}</b>
            {spot.w.round != null && <span>round {spot.w.round}</span>}
          </div>
          <div className="wz-decision">{DECISION_WORDS[spot.w.decision].word}</div>
          <div className="wz-rule">
            <div className="wz-rule-id">
              Rule <b>{spot.w.rule_id}</b>
              <span className={`lvl ${spot.w.rule_level >= 12 ? "hi" : spot.w.rule_level >= 7 ? "mid" : "lo"}`}>level {spot.w.rule_level}</span>
            </div>
            <div className="wz-rule-desc">{spot.w.rule_description}</div>
          </div>
          <div className="wz-foot">
            <span className={spot.w.enforced ? "enf yes" : "enf no"}>{spot.w.enforced ? "✓ enforced" : "not enforced"}</span>
            {spot.w.latency_ms != null && <span>decided in {(spot.w.latency_ms / 1000).toFixed(1)} s</span>}
          </div>
        </div>
      )}
      {spot?.kind === "approval" && spot.g.decision && (
        <div className={`wz-card coord ${APPROVAL_CLS[spot.g.decision] ?? ""}`} key={spot.g.ev.update_id + spot.g.decision}>
          <div className="wz-who">
            <b>{deviceLabelOf(s, spot.g.ev.device)}</b>
            <span>round {spot.g.ev.round}</span>
          </div>
          <div className="wz-decision">{spot.g.decision[0].toUpperCase() + spot.g.decision.slice(1)}</div>
          <div className="wz-rule">
            <div className="wz-rule-id">
              decided by <b>{deciderOfActor(spot.g.actor ?? "") === "human" ? "a human operator" : spot.g.actor}</b>
            </div>
            {spot.g.decisionReason && <div className="wz-rule-desc">{spot.g.decisionReason}</div>}
          </div>
        </div>
      )}
      {waitingIds.length > 0 && (
        <div className="wz-wait">
          <span className="pulse-dot" />
          Waiting for {isWazuh ? "Wazuh" : "a decision"}: <b>{waitingIds.map((d) => deviceLabelOf(s, d)).join(", ")}</b>
        </div>
      )}
      {isWazuh && (
        <div className="wz-counts">
          <span className="c-ok">
            <b>{counts.approve}</b> approved
          </span>
          <span className="c-bad">
            <b>{counts.quarantine + counts.reject}</b> stopped
          </span>
          <span className="c-bad">
            <b>{counts.suspend_device}</b> suspended
          </span>
          <span className="c-warn">
            <b>{counts.alert}</b> alerts
          </span>
        </div>
      )}
    </div>
  );
}
