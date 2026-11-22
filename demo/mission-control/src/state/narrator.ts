// "What's happening now" — one or two plain-English sentences generated ONLY from events already
// in the reducer state. Pure: the clock is the last event's `t`, never Date.now().

import { GATE_ORDER, type GateChecks, type WazuhDecisionEv } from "../types";
import type { DashState, GateRecord, Notable } from "./reducer";
import { deciderOfActor, effectiveStatus, fmtRisk, isLowSeverity, normAccuracy, verdictOf } from "./reducer";

export type NarrationTone = "calm" | "ok" | "warn" | "bad" | "wazuh";

export interface Narration {
  /** the main sentence; steps of one story are joined with " → " */
  headline: string;
  /** optional second, smaller sentence (reason, rule description) */
  detail: string | null;
  tone: NarrationTone;
  /** seq of the event the narration is about (null for phase narration) */
  seq: number | null;
}

/** An incident stays on the banner this long (recording seconds) before phase narration returns. */
export const NARRATOR_WINDOW_S = 14;

export const CHECK_WORDS: Record<keyof GateChecks, string> = {
  untampered: "Untampered",
  sender: "Sender verified",
  shape: "Correct shape & round",
  replay: "Not a replay",
  normal: "Looks normal",
};

const SKIP_WORDS: Record<string, string> = {
  low_battery: "its battery is too low",
  weak_signal: "its signal is too weak to download the model in time",
  drift: "its own data drifted — the server only sees an absence",
  offline: "it is offline",
  model_rejected: "it refused a model it could not verify",
  tls_error: "it refused an untrusted connection",
};

/** Scenes whose story is told better by the gate / decision chain than by the attack result. */
const CHAIN_SCENES = ["replay", "tamper", "forge", "poison"];

function label(s: DashState, id: string | null | undefined): string {
  if (!id) return "a device";
  return s.devices[id]?.info.label ?? id;
}

function listWords(xs: string[]): string {
  if (xs.length <= 1) return xs.join("");
  if (xs.length === 2) return `${xs[0]} and ${xs[1]}`;
  if (xs.length > 4) return `${xs.slice(0, 3).join(", ")} and ${xs.length - 3} more`;
  return `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
}

function roundPrefix(r: number | null | undefined): string {
  return r != null ? `Round ${r}: ` : "";
}

export function firstFailedCheck(g: GateRecord): keyof GateChecks | null {
  for (const k of GATE_ORDER) if (g.ev.checks[k] === "fail") return k;
  return null;
}

function attackSceneOf(g: GateRecord): string | null {
  const m = /#(\w+)$/.exec(g.ev.update_id);
  return m ? m[1] : null;
}

const WAZUH_PAST: Record<WazuhDecisionEv["decision"], string> = {
  approve: "approved it",
  quarantine: "quarantined it",
  reject: "rejected it",
  suspend_device: "suspended the device",
  alert: "raised an alert",
};

export function wazuhSentence(w: WazuhDecisionEv): string {
  const base = `Wazuh rule ${w.rule_id} ${WAZUH_PAST[w.decision]}`;
  if (w.decision === "alert") return `${base} — a human decides`;
  return w.enforced ? base : `${base} (not enforced)`;
}

function blockedStory(s: DashState, g: GateRecord): Narration {
  const failed = firstFailedCheck(g);
  const who = label(s, g.ev.device);
  const scene = g.ev.source === "attack" ? attackSceneOf(g) : null;
  const subject =
    scene === "replay"
      ? `An attacker re-sent ${who}'s old update`
      : scene === "tamper"
        ? `An attacker changed bytes in ${who}'s update`
        : scene === "forge"
          ? `An attacker sent an update signed with a fake key`
          : g.ev.source === "attack"
            ? `An attacker sent a forged request as ${who}`
            : `${who}'s update`;
  const verdict = `FinSentinel blocked it${failed ? ` at “${CHECK_WORDS[failed]}”` : ""} (HTTP ${g.ev.http_status})`;
  return {
    headline: `${roundPrefix(g.ev.round)}${subject} → ${verdict}`,
    detail: g.ev.reasons[0] ?? (g.ev.detail || null),
    tone: "bad",
    seq: g.ev.seq,
  };
}

/** The full story of one update: who sent what → what FinSentinel found → who decided what. */
export function updateStory(s: DashState, key: string, decisionOverride?: WazuhDecisionEv): Narration | null {
  const g = s.gates[key];
  if (!g) return null;
  if (g.ev.http_status >= 400) return blockedStory(s, g);
  const d = s.devices[g.ev.device];
  const who = label(s, g.ev.device);
  const normal = g.ev.checks.normal;
  const risky = g.ev.severity != null && !isLowSeverity(g.ev.severity);
  const flagged = normal === "fail" || normal === "warn" || risky;
  const subject = d?.hacked
    ? `${who} sent poisoned weights`
    : normal === "fail"
      ? `${who} sent an unusual update`
      : risky
        ? `${who} sent a ${String(g.ev.severity).toLowerCase()}-risk update`
        : `${who} sent its update`;
  const fs = flagged
    ? `FinSentinel flagged it (risk ${fmtRisk(g.ev.risk_score)}${g.ev.severity ? `, ${g.ev.severity}` : ""})`
    : "FinSentinel passed all 5 checks";

  const w = decisionOverride ?? (g.ev.source === "attack" ? undefined : s.wazuhByUpdate[key]);
  let decision: string;
  let tone: NarrationTone;
  let detail: string | null = g.ev.reasons[0] ?? null;
  const status = effectiveStatus(g);
  if (g.actor && g.decision && (!w || deciderOfActor(g.actor) === "human")) {
    const who2 = deciderOfActor(g.actor);
    const verb = g.decision;
    if (who2 === "human") decision = `the operator ${verb} it by hand`;
    else if (who2 === "wazuh") {
      const m = /rule\s+(\d+)/i.exec(g.actor);
      decision = `Wazuh${m ? ` rule ${m[1]}` : ""} ${verb} it`;
    } else if (who2 === "finsentinel") decision = `FinSentinel's risk policy ${verb} it`;
    else decision = `the coordinator ${verb} it`;
    tone = verb === "approved" ? "ok" : "bad";
    if (!detail && g.decisionReason) detail = g.decisionReason;
  } else if (w) {
    decision = wazuhSentence(w);
    tone = w.decision === "approve" ? "ok" : w.decision === "alert" ? "warn" : "bad";
    detail = w.rule_description + (detail ? ` · ${detail}` : "");
  } else if (status === "pending_approval") {
    decision = s.engine?.engine === "wazuh" ? "waiting for Wazuh to decide…" : "waiting for a decision…";
    tone = "warn";
  } else {
    const v = verdictOf(g);
    decision = v.text.toLowerCase();
    tone = v.kind === "accepted" ? "ok" : v.kind === "review" ? "warn" : "bad";
  }
  return { headline: `${roundPrefix(g.ev.round)}${subject} → ${fs} → ${decision}`, detail, tone, seq: g.ev.seq };
}

function deviceStory(s: DashState, n: Notable): Narration {
  const who = label(s, n.device);
  const p = roundPrefix(n.round);
  switch (n.state) {
    case "skipped": {
      const why = SKIP_WORDS[n.reason ?? ""] ?? (n.reason ? `reason: ${n.reason}` : "no reason given");
      return {
        headline: `${p}${who} sits this round out — ${why}.`,
        detail: n.detail ?? null,
        tone: n.reason === "drift" ? "wazuh" : "warn",
        seq: n.seq,
      };
    }
    case "suspended":
      return {
        headline: `${p}${who} is suspended${n.reason ? ` by Wazuh rule ${n.reason}` : ""} — its updates are no longer accepted.`,
        detail: n.detail ?? null,
        tone: "bad",
        seq: n.seq,
      };
    case "model_rejected":
      return {
        headline: `${p}${who} refused a model whose signature did not verify — it will not train on it.`,
        detail: n.detail ?? null,
        tone: "warn",
        seq: n.seq,
      };
    case "offline":
      return { headline: `${p}${who} went offline.`, detail: n.detail ?? null, tone: "warn", seq: n.seq };
    default:
      return { headline: `${p}${who}: ${n.state ?? "status changed"}.`, detail: n.detail ?? null, tone: "warn", seq: n.seq };
  }
}

function attackStory(s: DashState, n: Notable): Narration {
  if (n.scene === "mitm") {
    if (n.outcome === "observed")
      return {
        headline: "An eavesdropper tapped the line → it captured traffic but saw only encrypted bytes.",
        detail: n.detail ?? null,
        tone: "ok",
        seq: n.seq,
      };
    return {
      headline: "A fake server tried to impersonate ours → the device rejected its certificate and refused to talk.",
      detail: n.detail ?? null,
      tone: "ok",
      seq: n.seq,
    };
  }
  return {
    headline: `${roundPrefix(n.round)}${n.scene ?? "Attack"} attack${n.device ? ` on ${label(s, n.device)}` : ""} → ${n.outcome ?? "result"}.`,
    detail: n.detail ?? null,
    tone: n.outcome === "accepted" ? "bad" : "ok",
    seq: n.seq,
  };
}

function wazuhOnlyStory(s: DashState, w: WazuhDecisionEv): Narration {
  const who = label(s, w.device);
  const verb =
    w.decision === "suspend_device"
      ? `suspended ${who}`
      : w.decision === "alert"
        ? `raised an alert about ${who}`
        : `${w.decision === "approve" ? "approved" : w.decision === "reject" ? "rejected" : "quarantined"} ${who}'s update`;
  return {
    headline: `${roundPrefix(w.round)}Wazuh rule ${w.rule_id} (level ${w.rule_level}) ${verb}${w.enforced ? "" : " (not enforced)"}.`,
    detail: w.rule_description + (w.enforcement_detail ? ` — ${w.enforcement_detail}` : ""),
    tone: w.decision === "alert" ? "warn" : w.decision === "approve" ? "ok" : "bad",
    seq: w.seq,
  };
}

function incidentStory(s: DashState, n: Notable): Narration | null {
  switch (n.kind) {
    case "gate":
    case "decision":
      return n.key ? updateStory(s, n.key) : null;
    case "wazuh": {
      const w = s.wazuhDecisions.find((x) => x.seq === n.seq);
      if (!w) return null;
      if (w.decision !== "suspend_device" && w.update_id && s.gates[w.update_id]) return updateStory(s, w.update_id, w);
      return wazuhOnlyStory(s, w);
    }
    case "device":
      return deviceStory(s, n);
    case "attack":
      return attackStory(s, n);
  }
}

function phaseStory(s: DashState): Narration {
  const n = s.deviceOrder.length;
  if (s.phase === "finished" && s.finished) {
    const acc = s.finished.final_accuracy;
    return {
      headline: `Run complete: ${s.finished.rounds_completed} rounds${acc != null ? ` — final accuracy ${(normAccuracy(acc) * 100).toFixed(1)}%` : ""}.`,
      detail: "Every number on this screen came from the running system.",
      tone: "calm",
      seq: null,
    };
  }
  const rp = s.roundPhase;
  if (!rp) {
    if (!s.run) return { headline: "Waiting for the system to start…", detail: null, tone: "calm", seq: null };
    return {
      headline: `The fleet is coming online: ${n} devices, each holding one person's activity data.`,
      detail: "Their data never leaves the device — only model weights travel.",
      tone: "calm",
      seq: null,
    };
  }
  const r = rp.round;
  const p = roundPrefix(r);
  if (rp.phase === "open") {
    const sent = Object.values(s.devices).filter((d) => d.submittedByRound[r]).length;
    return {
      headline: `${p}the server sent out the signed model — devices are training on their own data.`,
      detail: `${sent} of ${n} updates received so far.`,
      tone: "calm",
      seq: null,
    };
  }
  if (rp.phase === "deciding") {
    const names = rp.waitingFor.map((d) => label(s, d));
    const who = s.engine?.engine === "coordinator" ? "the coordinator" : "Wazuh";
    return {
      headline: names.length
        ? `${p}FinSentinel has checked every update — waiting for ${who} to decide on ${listWords(names)}.`
        : `${p}FinSentinel has checked every update — ${who} is deciding.`,
      detail: null,
      tone: "wazuh",
      seq: null,
    };
  }
  if (rp.phase === "aggregating") {
    const ok = Object.values(s.gates).filter((g) => g.ev.round === r && g.ev.source === "device" && verdictOf(g).kind === "accepted").length;
    return {
      headline: `${p}averaging ${ok} approved update${ok === 1 ? "" : "s"} into the global model.`,
      detail: "Quarantined, rejected and missing updates are left out.",
      tone: "calm",
      seq: null,
    };
  }
  // done
  const rc = s.roundsCompleted[r];
  const ev = s.evals[r] ?? null; // never borrow an older round's accuracy
  const prev = s.evals[r - 1];
  const acc = ev ? normAccuracy(ev.accuracy) * 100 : null;
  const delta = ev && prev ? (normAccuracy(ev.accuracy) - normAccuracy(prev.accuracy)) * 100 : null;
  const k = rc ? rc.participants.length : 0;
  return {
    headline: `${p}complete — ${k} device${k === 1 ? "" : "s"} contributed${
      acc != null
        ? `, global accuracy now ${acc.toFixed(1)}%${delta != null ? ` (${delta >= 0 ? "+" : ""}${delta.toFixed(1)})` : ""}`
        : " — testing the new global model…"
    }${acc != null ? "." : ""}`,
    detail: rc && rc.dropped.length ? `Left out: ${rc.dropped.map((d) => `${label(s, d.device)} (${d.reason.replace(/_/g, " ")})`).join(", ")}.` : null,
    tone: "calm",
    seq: null,
  };
}

/** Generate the banner text. Recent incidents (within NARRATOR_WINDOW_S of the last event) win. */
export function narrate(s: DashState): Narration {
  const now = s.lastT ?? 0;
  const recent = s.notable.filter((n) => now - n.t <= NARRATOR_WINDOW_S);
  const hasChain = recent.some((n) => n.kind === "gate" || n.kind === "wazuh" || n.kind === "decision");
  for (const n of recent) {
    if (n.kind === "attack" && hasChain && CHAIN_SCENES.includes(n.scene ?? "")) continue;
    const story = incidentStory(s, n);
    if (story) return story;
  }
  return phaseStory(s);
}
