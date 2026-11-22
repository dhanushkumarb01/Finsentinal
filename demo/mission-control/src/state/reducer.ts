// The ONE pure reducer that every screen element derives from. Live and replay feed the
// exact same events through `reduce`, so they render identically. Nothing in here may read
// the clock, randomness or the network: time comes only from the event's `t` field.

import type {
  ApprovalEv,
  AttackEv,
  DeviceInfo,
  DriftMonitorEv,
  Ev,
  GateResultEv,
  GlobalEvalEv,
  LocalTrainingEv,
  RoundCompletedEv,
  RunFinishedEv,
  RunStartedEv,
  UpdateSubmittedEv,
} from "../types";

export type Phase = "waiting" | "open" | "finished";
export type FlightOutcome = "pass" | "held" | "quarantine" | "blocked";
export type VerdictKind = "accepted" | "review" | "quarantined" | "rejected" | "blocked" | "checked";
export type FeedSeverity = "info" | "low" | "medium" | "high" | "critical";
export type FeedSource = "FinSentinel" | "Coordinator" | "Attack" | "Drift monitor";

export interface GateRecord {
  ev: GateResultEv;
  decision: ApprovalEv["decision"] | null;
  actor: string | null;
  decisionReason: string | null;
}

export interface DeviceView {
  info: DeviceInfo;
  known: boolean; // listed in run_started.devices
  state: string;
  round: number | null;
  battery: number | null;
  signal_dbm: number | null;
  network: string;
  online: boolean;
  hacked: boolean;
  reason: string | null;
  detail: string | null;
  losses: { round: number; loss: number }[];
  trainingByRound: Record<number, LocalTrainingEv>;
  lastTrainingRound: number | null;
  statesByRound: Record<number, string[]>;
  submittedByRound: Record<number, string>; // round -> update_id
  gateByRound: Record<number, string>; // round -> update_id
  lastGateId: string | null;
}

export interface Flight {
  id: string; // update_id, or "attack-<seq>"
  device: string;
  label: string;
  round: number | null;
  attacker: boolean;
  scene: string | null;
  t: number;
  seq: number;
  outcome: FlightOutcome | null;
  outcomeLabel: string | null;
  tEnd: number | null;
}

export interface FeedItem {
  id: number; // seq of the source event
  t: number;
  severity: FeedSeverity;
  source: FeedSource;
  title: string;
  detail: string;
  device: string | null;
  /** consecutive approvals with the same actor/decision/round are shown as one line */
  groupKey?: string;
  count?: number;
  devices?: string[];
}

export interface ComponentView {
  status: "up" | "down";
  detail: string | null;
  t: number;
}

export interface TunnelView {
  url: string | null;
  status: "up" | "down";
  beacon_published: boolean;
}

export interface DashState {
  runId: string | null;
  lastSeq: number;
  lastT: number | null;
  firstT: number | null;
  eventCount: number;
  run: RunStartedEv | null;
  finished: RunFinishedEv | null;
  round: number | null;
  totalRounds: number | null;
  phase: Phase;
  roundDeadline: number | null;
  deviceOrder: string[];
  devices: Record<string, DeviceView>;
  components: Record<string, ComponentView>;
  tunnel: TunnelView | null;
  gates: Record<string, GateRecord>;
  /** approval events whose gate_result has not arrived (yet); applied when it does */
  orphanDecisions: Record<string, ApprovalEv>;
  lastGateId: string | null;
  submissions: Record<string, UpdateSubmittedEv>;
  flights: Flight[];
  roundsCompleted: Record<number, RoundCompletedEv>;
  evals: Record<number, GlobalEvalEv>;
  driftDropRounds: number[];
  driftDevices: string[];
  driftMonitor: Record<string, DriftMonitorEv>; // latest per device
  alerts: FeedItem[]; // newest first
  attacks: AttackEv[];
}

export const MAX_FLIGHTS = 40;
export const MAX_ALERTS = 200;

export function initialState(): DashState {
  return {
    runId: null,
    lastSeq: 0,
    lastT: null,
    firstT: null,
    eventCount: 0,
    run: null,
    finished: null,
    round: null,
    totalRounds: null,
    phase: "waiting",
    roundDeadline: null,
    deviceOrder: [],
    devices: {},
    components: {},
    tunnel: null,
    gates: {},
    orphanDecisions: {},
    lastGateId: null,
    submissions: {},
    flights: [],
    roundsCompleted: {},
    evals: {},
    driftDropRounds: [],
    driftDevices: [],
    driftMonitor: {},
    alerts: [],
    attacks: [],
  };
}

// ---------- helpers (pure, exported for the UI and tests) ----------

export function isLowSeverity(sev: string | null | undefined): boolean {
  return typeof sev === "string" && sev.trim().toUpperCase() === "LOW";
}

/** Effective approval status: a later `approval` event overrides the gate's own status. */
export function effectiveStatus(g: GateRecord): string | null {
  return g.decision ?? g.ev.approval_status;
}

export function verdictOf(g: GateRecord): { kind: VerdictKind; text: string } {
  if (g.ev.http_status >= 400) return { kind: "blocked", text: `Blocked (${g.ev.http_status})` };
  const s = effectiveStatus(g);
  if (s === "approved") return { kind: "accepted", text: "Accepted" };
  if (s === "quarantined") return { kind: "quarantined", text: "Quarantined" };
  if (s === "rejected") return { kind: "rejected", text: "Rejected" };
  if (s === "pending_approval") return { kind: "review", text: "Waiting for review" };
  return { kind: "checked", text: `Checked (${g.ev.http_status})` };
}

/** Where the envelope ends up after the gate. */
export function flightOutcomeOf(g: GateRecord): FlightOutcome {
  if (g.ev.http_status >= 400) return "blocked";
  const s = effectiveStatus(g);
  if (s === "quarantined" || s === "rejected") return "quarantine";
  if (s === "pending_approval" && !isLowSeverity(g.ev.severity)) return "held";
  return "pass";
}

export function flightLabelOf(o: FlightOutcome, httpStatus: number | null): string {
  switch (o) {
    case "blocked":
      return httpStatus != null ? `BLOCKED ${httpStatus}` : "BLOCKED";
    case "quarantine":
      return "QUARANTINED";
    case "held":
      return "HELD FOR REVIEW";
    case "pass":
      return "ACCEPTED INTO GATE";
  }
}

/** Updates that a human must approve/reject: pending, not LOW severity, no decision yet. */
export function pendingReview(s: DashState): GateRecord[] {
  return Object.values(s.gates)
    .filter(
      (g) =>
        g.ev.http_status < 400 &&
        g.ev.approval_status === "pending_approval" &&
        !isLowSeverity(g.ev.severity) &&
        g.decision == null,
    )
    .sort((a, b) => a.ev.seq - b.ev.seq);
}

export function normAccuracy(a: number): number {
  return a > 1 ? a / 100 : a;
}

export function latestEval(s: DashState): GlobalEvalEv | null {
  const rs = Object.keys(s.evals).map(Number);
  if (!rs.length) return null;
  return s.evals[Math.max(...rs)];
}

export function signalBars(dbm: number | null | undefined): number {
  if (dbm == null || Number.isNaN(dbm)) return 0;
  if (dbm >= -60) return 4;
  if (dbm >= -70) return 3;
  if (dbm >= -80) return 2;
  if (dbm >= -90) return 1;
  return 0;
}

function guessDevice(id: string): DeviceInfo {
  const wokwi = id.toLowerCase().startsWith("esp32");
  return {
    id,
    kind: wokwi ? "wokwi" : "virtual",
    label: id,
    subject: -1,
    network: wokwi ? "wokwi-guest" : "wifi",
    battery: 0,
  };
}

function newDevice(info: DeviceInfo, known: boolean): DeviceView {
  return {
    info,
    known,
    state: "waiting",
    round: null,
    battery: known ? info.battery : null,
    signal_dbm: null,
    network: info.network,
    online: true,
    hacked: false,
    reason: null,
    detail: null,
    losses: [],
    trainingByRound: {},
    lastTrainingRound: null,
    statesByRound: {},
    submittedByRound: {},
    gateByRound: {},
    lastGateId: null,
  };
}

function withDevice(s: DashState, id: string, fn: (d: DeviceView) => DeviceView): DashState {
  const existing = s.devices[id];
  const base = existing ?? newDevice(guessDevice(id), false);
  return {
    ...s,
    devices: { ...s.devices, [id]: fn(base) },
    deviceOrder: existing ? s.deviceOrder : [...s.deviceOrder, id],
  };
}

function addState(d: DeviceView, round: number | undefined | null, st: string): DeviceView {
  if (round == null) return d;
  const cur = d.statesByRound[round] ?? [];
  if (cur.includes(st)) return d;
  return { ...d, statesByRound: { ...d.statesByRound, [round]: [...cur, st] } };
}

function pushAlert(s: DashState, item: FeedItem): DashState {
  return { ...s, alerts: [item, ...s.alerts].slice(0, MAX_ALERTS) };
}

function pushFlight(s: DashState, f: Flight): DashState {
  const flights = [...s.flights, f];
  return { ...s, flights: flights.slice(Math.max(0, flights.length - MAX_FLIGHTS)) };
}

function resolveFlight(
  s: DashState,
  idx: number,
  outcome: FlightOutcome,
  label: string,
  t: number,
): DashState {
  const flights = s.flights.slice();
  flights[idx] = { ...flights[idx], outcome, outcomeLabel: label, tEnd: t };
  return { ...s, flights };
}

function addUnique(arr: number[], v: number): number[] {
  return arr.includes(v) ? arr : [...arr, v].sort((a, b) => a - b);
}
function addUniqueStr(arr: string[], v: string): string[] {
  return arr.includes(v) ? arr : [...arr, v];
}

function deviceLabel(s: DashState, id: string | undefined | null): string {
  if (!id) return "";
  return s.devices[id]?.info.label ?? id;
}

export const SCENE_NAMES: Record<string, string> = {
  poison: "Poisoning",
  replay: "Replay",
  tamper: "Tamper",
  forge: "Forged key",
  mitm: "Eavesdropper (MITM)",
  drift: "Make device drift",
  weak_signal: "Weak signal",
  restore: "Restore",
  low_battery: "Low battery",
  offline: "Go offline",
};

/**
 * Scenes whose attack traffic itself hits the gate, so a red "attacker" envelope is drawn.
 * poison travels as the hacked device's own (signed) update; drift / weak_signal / low_battery /
 * offline / restore change device conditions and never reach the gate as attacker traffic.
 */
export const GATE_ATTACK_SCENES = ["replay", "tamper", "forge", "mitm"];

// ---------- the reducer ----------

export function reduce(prev: DashState, ev: Ev): DashState {
  if (!ev || typeof ev.seq !== "number" || typeof ev.type !== "string") return prev;

  let s = prev;
  // A different run → start over from a clean slate.
  if (s.runId !== null && ev.run_id !== s.runId) {
    s = initialState();
  } else if (ev.seq <= s.lastSeq) {
    return prev; // duplicate (e.g. replayed again after a reconnect)
  }
  s = {
    ...s,
    runId: ev.run_id,
    lastSeq: ev.seq,
    lastT: ev.t,
    firstT: s.firstT ?? ev.t,
    eventCount: s.eventCount + 1,
  };

  switch (ev.type) {
    case "run_started": {
      const devices: Record<string, DeviceView> = { ...s.devices };
      const order: string[] = [];
      for (const info of ev.devices) {
        const old = devices[info.id];
        devices[info.id] = old ? { ...old, info, known: true } : newDevice(info, true);
        order.push(info.id);
      }
      for (const id of s.deviceOrder) if (!order.includes(id)) order.push(id);
      return { ...s, run: ev, totalRounds: ev.rounds, devices, deviceOrder: order, phase: "waiting" };
    }

    case "component":
      return {
        ...s,
        components: { ...s.components, [ev.name]: { status: ev.status, detail: ev.detail ?? null, t: ev.t } },
      };

    case "tunnel":
      return { ...s, tunnel: { url: ev.url, status: ev.status, beacon_published: ev.beacon_published } };

    case "round_started":
      return { ...s, round: ev.round, phase: "open", roundDeadline: ev.deadline };

    case "device_status": {
      let out = withDevice(s, ev.device, (d) => {
        let nd: DeviceView = {
          ...d,
          state: ev.state,
          round: ev.round ?? d.round,
          battery: ev.battery ?? d.battery,
          signal_dbm: ev.signal_dbm ?? d.signal_dbm,
          network: ev.network ?? d.network,
          online: ev.online ?? ev.state !== "offline",
          hacked: ev.hacked ?? d.hacked,
          reason: ev.reason ?? null,
          detail: ev.detail ?? null,
        };
        nd = addState(nd, ev.round, ev.state);
        return nd;
      });
      if (ev.state === "skipped" && ev.reason === "drift") {
        if (ev.round != null) out = { ...out, driftDropRounds: addUnique(out.driftDropRounds, ev.round) };
        out = { ...out, driftDevices: addUniqueStr(out.driftDevices, ev.device) };
      }
      return out;
    }

    case "local_training": {
      const last = ev.losses.length ? ev.losses[ev.losses.length - 1] : null;
      return withDevice(s, ev.device, (d) => {
        const losses = d.losses.filter((l) => l.round !== ev.round);
        if (last != null && Number.isFinite(last)) losses.push({ round: ev.round, loss: last });
        losses.sort((a, b) => a.round - b.round);
        return addState(
          {
            ...d,
            losses,
            trainingByRound: { ...d.trainingByRound, [ev.round]: ev },
            lastTrainingRound: Math.max(ev.round, d.lastTrainingRound ?? ev.round),
          },
          ev.round,
          "trained",
        );
      });
    }

    case "update_submitted": {
      // The coordinator also announces attack traffic (e.g. "<id>#replay") as update_submitted.
      // If an attacker envelope aimed at this device is still in the air, that envelope IS this
      // submission: adopt the update_id so the gate_result pairs with it, and leave the honest
      // device's own per-round records untouched.
      const ATTACK_ADOPT_WINDOW_S = 15;
      let adopt = -1;
      for (let i = s.flights.length - 1; i >= 0; i--) {
        const f = s.flights[i];
        if (f.attacker && f.outcome == null && f.id.startsWith("attack-") && f.device === ev.device && ev.t - f.t <= ATTACK_ADOPT_WINDOW_S) {
          adopt = i;
          break;
        }
      }
      if (adopt >= 0) {
        const flights = s.flights.slice();
        flights[adopt] = { ...flights[adopt], id: ev.update_id, round: ev.round };
        return { ...s, flights, submissions: { ...s.submissions, [ev.update_id]: ev } };
      }
      let out = withDevice(s, ev.device, (d) =>
        addState({ ...d, submittedByRound: { ...d.submittedByRound, [ev.round]: ev.update_id } }, ev.round, "submitted"),
      );
      out = { ...out, submissions: { ...out.submissions, [ev.update_id]: ev } };
      if (!out.flights.some((f) => f.id === ev.update_id)) {
        out = pushFlight(out, {
          id: ev.update_id,
          device: ev.device,
          label: deviceLabel(out, ev.device),
          round: ev.round,
          attacker: false,
          scene: null,
          t: ev.t,
          seq: ev.seq,
          outcome: null,
          outcomeLabel: null,
          tEnd: null,
        });
      }
      return out;
    }

    case "gate_result": {
      // A replayed/attack request can reuse a real device's update_id; keep it separate so it
      // never overwrites the honest device's record.
      const key = ev.source === "attack" ? `attack:${ev.seq}` : ev.update_id;
      const old = ev.source === "attack" ? undefined : s.gates[key];
      const orphan = ev.source === "attack" ? undefined : s.orphanDecisions[key];
      const rec: GateRecord = {
        ev,
        decision: old?.decision ?? orphan?.decision ?? null,
        actor: old?.actor ?? orphan?.actor ?? null,
        decisionReason: old?.decisionReason ?? orphan?.reason ?? null,
      };
      let out: DashState = { ...s, gates: { ...s.gates, [key]: rec }, lastGateId: key };
      if (orphan) {
        const rest = { ...out.orphanDecisions };
        delete rest[key];
        out = { ...out, orphanDecisions: rest };
      }
      if (ev.source === "device") {
        out = withDevice(out, ev.device, (d) =>
          addState(
            { ...d, lastGateId: ev.update_id, gateByRound: { ...d.gateByRound, [ev.round]: ev.update_id } },
            ev.round,
            "gate",
          ),
        );
      }
      const outcome = flightOutcomeOf(rec);
      const label = flightLabelOf(outcome, ev.http_status >= 400 ? ev.http_status : null);
      let idx = out.flights.findIndex(
        (f) => f.id === ev.update_id && (ev.source !== "attack" || f.outcome == null),
      );
      if (idx < 0 && ev.source === "attack") {
        // Attack traffic is not announced by update_submitted; pair with the latest open attacker envelope.
        for (let i = out.flights.length - 1; i >= 0; i--) {
          if (out.flights[i].attacker && out.flights[i].outcome == null) {
            idx = i;
            break;
          }
        }
      }
      if (idx >= 0) {
        if (out.flights[idx].outcome == null) out = resolveFlight(out, idx, outcome, label, ev.t);
      } else {
        out = pushFlight(out, {
          id: key,
          device: ev.device,
          label: ev.source === "attack" ? "Attacker" : deviceLabel(out, ev.device),
          round: ev.round,
          attacker: ev.source === "attack",
          scene: null,
          t: ev.t,
          seq: ev.seq,
          outcome,
          outcomeLabel: label,
          tEnd: ev.t,
        });
      }
      return out;
    }

    case "approval": {
      const g = s.gates[ev.update_id];
      let out = s;
      if (!g) out = { ...out, orphanDecisions: { ...out.orphanDecisions, [ev.update_id]: ev } };
      if (g) {
        out = {
          ...out,
          gates: {
            ...out.gates,
            [ev.update_id]: { ...g, decision: ev.decision, actor: ev.actor, decisionReason: ev.reason },
          },
        };
      }
      const groupKey = `approval|${ev.decision}|${ev.actor}|${ev.round}`;
      const top = out.alerts[0];
      if (top && top.groupKey === groupKey) {
        const devices = top.devices?.includes(ev.device) ? top.devices : [...(top.devices ?? []), ev.device];
        const count = (top.count ?? 1) + 1;
        const merged: FeedItem = {
          ...top,
          id: ev.seq,
          t: ev.t,
          count,
          devices,
          device: null,
          title: `${count} updates ${ev.decision} by ${ev.actor}`,
          detail: `round ${ev.round}: ${devices.map((d) => deviceLabel(out, d)).join(", ")}`,
        };
        return { ...out, alerts: [merged, ...out.alerts.slice(1)] };
      }
      return pushAlert(out, {
        id: ev.seq,
        t: ev.t,
        severity: ev.decision === "approved" ? "info" : "high",
        source: "FinSentinel",
        title: `Update ${ev.decision} by ${ev.actor}`,
        detail: ev.reason || ev.update_id,
        device: ev.device,
        groupKey,
        count: 1,
        devices: [ev.device],
      });
    }

    case "drift_monitor": {
      let out: DashState = { ...s, driftMonitor: { ...s.driftMonitor, [ev.device]: ev } };
      if (ev.flagged) {
        out = { ...out, driftDevices: addUniqueStr(out.driftDevices, ev.device) };
        out = pushAlert(out, {
          id: ev.seq,
          t: ev.t,
          severity: "medium",
          source: "Drift monitor",
          title: `Drift flagged (score ${ev.score.toFixed(2)} ≥ ${ev.threshold.toFixed(2)})`,
          detail: ev.finsentinel_flagged
            ? "FinSentinel also flagged this device's update."
            : "FinSentinel's own gate did not flag it.",
          device: ev.device,
        });
      }
      return out;
    }

    case "round_completed": {
      let drift = s.driftDropRounds;
      let driftDevices = s.driftDevices;
      for (const d of ev.dropped) {
        if (d.reason === "drift") {
          drift = addUnique(drift, ev.round);
          driftDevices = addUniqueStr(driftDevices, d.device);
        }
      }
      return {
        ...s,
        roundsCompleted: { ...s.roundsCompleted, [ev.round]: ev },
        driftDropRounds: drift,
        driftDevices,
        round: Math.max(s.round ?? 0, ev.round),
        phase: s.phase === "finished" ? "finished" : "waiting",
      };
    }

    case "global_eval":
      return { ...s, evals: { ...s.evals, [ev.round]: ev } };

    case "alert":
      return pushAlert(s, {
        id: ev.seq,
        t: ev.t,
        severity: ev.severity,
        source: ev.source === "finsentinel" ? "FinSentinel" : ev.source === "attack" ? "Attack" : "Coordinator",
        title: ev.title,
        detail: ev.detail,
        device: ev.device ?? null,
      });

    case "attack": {
      let out: DashState = { ...s, attacks: [...s.attacks, ev].slice(-100) };
      if (ev.phase === "launched") {
        if (!GATE_ATTACK_SCENES.includes(ev.scene)) return out;
        return pushFlight(out, {
          id: `attack-${ev.seq}`,
          device: ev.target ?? "",
          label: `Attacker · ${SCENE_NAMES[ev.scene] ?? ev.scene}`,
          round: null,
          attacker: true,
          scene: ev.scene,
          t: ev.t,
          seq: ev.seq,
          outcome: null,
          outcomeLabel: null,
          tEnd: null,
        });
      }
      // phase === "result"
      let idx = -1;
      for (let i = out.flights.length - 1; i >= 0; i--) {
        const f = out.flights[i];
        if (f.attacker && f.scene === ev.scene && f.outcome == null) {
          idx = i;
          break;
        }
      }
      // "observed" is an eavesdropper's report (e.g. ciphertext only), not where the envelope ended.
      if (idx >= 0 && ev.outcome && ev.outcome !== "observed") {
        const o: FlightOutcome =
          ev.outcome === "accepted" ? "pass" : ev.outcome === "quarantined" ? "quarantine" : "blocked";
        const label =
          ev.http_status != null && ev.http_status >= 400 ? `BLOCKED ${ev.http_status}` : ev.outcome.toUpperCase();
        out = resolveFlight(out, idx, o, label, ev.t);
      }
      const name = SCENE_NAMES[ev.scene] ?? ev.scene;
      return pushAlert(out, {
        id: ev.seq,
        t: ev.t,
        severity: ev.outcome === "accepted" ? "critical" : "medium",
        source: "Attack",
        title: `${name} attack → ${ev.outcome ?? "result"}${ev.http_status != null ? ` (${ev.http_status})` : ""}`,
        detail: ev.detail,
        device: ev.target ?? null,
      });
    }

    case "run_finished":
      return { ...s, finished: ev, phase: "finished" };

    default:
      return s; // unknown event types are ignored, never guessed at
  }
}

export function reduceAll(s: DashState, evs: readonly Ev[]): DashState {
  let out = s;
  for (const e of evs) out = reduce(out, e);
  return out;
}

// Wrapper used with React's useReducer.
export type Action = { kind: "event"; ev: Ev } | { kind: "batch"; evs: Ev[] } | { kind: "reset" };

export function rootReducer(s: DashState, a: Action): DashState {
  switch (a.kind) {
    case "event":
      return reduce(s, a.ev);
    case "batch":
      return reduceAll(s, a.evs);
    case "reset":
      return initialState();
  }
}
