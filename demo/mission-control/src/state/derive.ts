// Small pure view-helpers on top of the reducer state (no new facts are created here).

import type { DashState, DeviceView, GateRecord } from "./reducer";
import { verdictOf } from "./reducer";

export const REASON_WORDS: Record<string, string> = {
  low_battery: "battery too low",
  weak_signal: "signal too weak",
  drift: "data drifted (research rule)",
  offline: "offline",
  model_rejected: "refused an unverified model",
  tls_error: "refused an untrusted connection",
  other: "other reason",
};

export function stateWords(d: DeviceView): string {
  switch (d.state) {
    case "waiting":
      return "Waiting for run";
    case "online":
      return "Online";
    case "idle":
      return "Idle";
    case "downloading":
      return "Downloading model";
    case "model_rejected":
      return "Refused bad model";
    case "training":
      return "Training";
    case "signing":
      return "Signing update";
    case "sending":
      return "Sending";
    case "done":
      return "Update sent";
    case "skipped":
      return `Skipped: ${REASON_WORDS[d.reason ?? ""] ?? d.reason ?? "no reason given"}`;
    case "offline":
      return "Offline";
    case "error":
      return "Error";
    default:
      return d.state;
  }
}

export function deviceGate(s: DashState, d: DeviceView): GateRecord | null {
  return d.lastGateId ? s.gates[d.lastGateId] ?? null : null;
}

export type StepState = "done" | "fail" | "warn" | "active" | "idle";
export interface Step {
  key: string;
  label: string;
  state: StepState;
  note?: string;
}

/** The "inside the device" step tracker for one round, lit only by events actually received. */
export function stepsFor(s: DashState, d: DeviceView, round: number): Step[] {
  const st = d.statesByRound[round] ?? [];
  const has = (x: string) => st.includes(x);
  const trained = has("trained") || !!d.trainingByRound[round];
  const submitted = !!d.submittedByRound[round];
  const gateId = d.gateByRound[round];
  const gate = gateId ? s.gates[gateId] : undefined;
  const rejected = has("model_rejected");
  const pastVerify = has("training") || trained || has("signing") || has("sending") || submitted;

  const download: StepState =
    has("downloading") || pastVerify || rejected ? "done" : "idle";
  const verify: StepState = rejected ? "fail" : pastVerify ? "done" : has("downloading") ? "active" : "idle";
  const train: StepState = trained ? "done" : has("training") ? "active" : "idle";
  const sign: StepState = submitted || has("sending") ? "done" : has("signing") ? "active" : "idle";
  const send: StepState = submitted ? "done" : has("sending") ? "active" : "idle";
  let gateState: StepState = "idle";
  let gateNote: string | undefined;
  if (gate) {
    const v = verdictOf(gate);
    gateNote = v.text;
    gateState =
      v.kind === "accepted" || v.kind === "checked" ? "done" : v.kind === "review" ? "warn" : "fail";
  } else if (submitted) {
    gateState = "active";
  }
  const steps: Step[] = [
    { key: "download", label: "Download model", state: download },
    { key: "verify", label: "Verify signature", state: verify, note: rejected ? "model refused" : undefined },
    { key: "train", label: "Train", state: train },
    { key: "sign", label: "Sign", state: sign },
    { key: "send", label: "Send", state: send },
    { key: "gate", label: "Gate result", state: gateState, note: gateNote },
  ];
  if (has("skipped")) {
    steps[2] = { ...steps[2], note: "skipped this round" };
  }
  return steps;
}

export function roundsOf(d: DeviceView): number[] {
  const set = new Set<number>();
  for (const k of Object.keys(d.statesByRound)) set.add(Number(k));
  for (const k of Object.keys(d.trainingByRound)) set.add(Number(k));
  return [...set].filter((n) => Number.isFinite(n)).sort((a, b) => a - b);
}

export function fmtClock(t: number | null | undefined): string {
  if (t == null) return "--:--:--";
  const d = new Date(t * 1000);
  return d.toLocaleTimeString([], { hour12: false });
}
