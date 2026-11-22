import { describe, expect, it } from "vitest";
import SAMPLE_RAW from "../../public/sample-recording.jsonl?raw";
import type { Ev, GateChecks } from "../types";
import {
  flightOutcomeOf,
  initialState,
  latestEval,
  pendingReview,
  reduce,
  reduceAll,
  rootReducer,
  signalBars,
  verdictOf,
  type DashState,
} from "./reducer";
import { stepsFor } from "./derive";
import { parseJsonl } from "../replay/engine";

const SAMPLE: string = SAMPLE_RAW;

const RUN = "run-A";
let seq = 0;
// Build an event with auto seq; `as Ev` keeps the test terse while the fields follow EVENTS.md.
function ev(type: string, fields: Record<string, unknown> = {}, run = RUN): Ev {
  seq++;
  return { seq, t: 1000 + seq, run_id: run, type, ...fields } as unknown as Ev;
}
const PASS: GateChecks = { untampered: "pass", sender: "pass", shape: "pass", replay: "pass", normal: "pass" };

function started(): DashState {
  seq = 0;
  return reduce(
    initialState(),
    ev("run_started", {
      model_version: "har-mlp-v1",
      param_count: 2278,
      rounds: 5,
      round_seconds: 30,
      test_samples: 100,
      mode: "live",
      devices: [
        { id: "phone-01", kind: "virtual", label: "Phone 1", subject: 1, network: "wifi", battery: 90 },
        { id: "esp32-01", kind: "wokwi", label: "ESP32 board 1", subject: 2, network: "wokwi-guest", battery: 100 },
      ],
    }),
  );
}

function gate(device: string, update_id: string, over: Record<string, unknown> = {}): Ev {
  return ev("gate_result", {
    device,
    round: 1,
    update_id,
    http_status: 202,
    checks: PASS,
    risk_score: 0.1,
    severity: "LOW",
    approval_status: "pending_approval",
    reasons: [],
    detail: "",
    source: "device",
    ...over,
  });
}

describe("reducer — every event type", () => {
  it("run_started registers devices in order with kind/label/battery", () => {
    const s = started();
    expect(s.runId).toBe(RUN);
    expect(s.totalRounds).toBe(5);
    expect(s.deviceOrder).toEqual(["phone-01", "esp32-01"]);
    expect(s.devices["esp32-01"].info.kind).toBe("wokwi");
    expect(s.devices["phone-01"].battery).toBe(90);
    expect(s.phase).toBe("waiting");
  });

  it("component and tunnel", () => {
    let s = started();
    s = reduce(s, ev("component", { name: "finsentinel", status: "up", detail: "ok" }));
    s = reduce(s, ev("component", { name: "caddy", status: "down" }));
    s = reduce(s, ev("tunnel", { url: "https://x.trycloudflare.com", status: "up", beacon_published: true }));
    expect(s.components.finsentinel.status).toBe("up");
    expect(s.components.caddy).toMatchObject({ status: "down", detail: null });
    expect(s.tunnel).toEqual({ url: "https://x.trycloudflare.com", status: "up", beacon_published: true });
  });

  it("round_started / round_completed / run_finished drive round and phase", () => {
    let s = started();
    s = reduce(s, ev("round_started", { round: 1, deadline: 2000, model_sha256: "aa" }));
    expect(s.round).toBe(1);
    expect(s.phase).toBe("open");
    s = reduce(s, ev("round_completed", { round: 1, participants: ["phone-01"], dropped: [], aggregated: true, model_sha256: "bb" }));
    expect(s.phase).toBe("waiting");
    expect(s.roundsCompleted[1].participants).toEqual(["phone-01"]);
    s = reduce(s, ev("run_finished", { rounds_completed: 1, final_accuracy: 0.7 }));
    expect(s.phase).toBe("finished");
    expect(s.finished?.final_accuracy).toBe(0.7);
  });

  it("device_status updates telemetry, keeps unspecified fields, records per-round states", () => {
    let s = started();
    s = reduce(s, ev("device_status", { device: "phone-01", state: "training", round: 1, battery: 70, signal_dbm: -72, hacked: false }));
    s = reduce(s, ev("device_status", { device: "phone-01", state: "skipped", round: 2, reason: "low_battery" }));
    const d = s.devices["phone-01"];
    expect(d.state).toBe("skipped");
    expect(d.reason).toBe("low_battery");
    expect(d.battery).toBe(70);
    expect(d.signal_dbm).toBe(-72);
    expect(signalBars(d.signal_dbm)).toBe(2);
    expect(d.statesByRound[1]).toEqual(["training"]);
    expect(d.statesByRound[2]).toEqual(["skipped"]);
    s = reduce(s, ev("device_status", { device: "phone-01", state: "offline", online: false }));
    expect(s.devices["phone-01"].online).toBe(false);
    s = reduce(s, ev("device_status", { device: "phone-01", state: "online", hacked: true }));
    expect(s.devices["phone-01"].online).toBe(true);
    expect(s.devices["phone-01"].hacked).toBe(true);
  });

  it("device_status for an unknown device creates a placeholder (kind from id)", () => {
    let s = started();
    s = reduce(s, ev("device_status", { device: "esp32-09", state: "online" }));
    expect(s.deviceOrder).toContain("esp32-09");
    expect(s.devices["esp32-09"].info.kind).toBe("wokwi");
    expect(s.devices["esp32-09"].known).toBe(false);
  });

  it("local_training stores the last loss per round for the sparkline", () => {
    let s = started();
    const base = { device: "phone-01", epochs: 2, samples: 300, class_counts: [1, 2, 3, 4, 5, 6], weight_delta: [[0.1]], delta_l2: 0.3, seconds: 1 };
    s = reduce(s, ev("local_training", { ...base, round: 1, losses: [1.5, 1.2] }));
    s = reduce(s, ev("local_training", { ...base, round: 2, losses: [0.9, 0.8] }));
    expect(s.devices["phone-01"].losses).toEqual([
      { round: 1, loss: 1.2 },
      { round: 2, loss: 0.8 },
    ]);
    expect(s.devices["phone-01"].trainingByRound[2].class_counts).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("update_submitted spawns exactly one envelope per update_id", () => {
    let s = started();
    const sub = { device: "phone-01", round: 1, update_id: "u1", payload_bytes: 9000, payload_sha256: "ab", signed: true, path: "toxiproxy" };
    s = reduce(s, ev("update_submitted", sub));
    s = reduce(s, ev("update_submitted", sub)); // a second event with a new seq but same update_id
    expect(s.flights.filter((f) => f.id === "u1")).toHaveLength(1);
    expect(s.flights[0]).toMatchObject({ device: "phone-01", label: "Phone 1", outcome: null, attacker: false });
    expect(s.devices["phone-01"].submittedByRound[1]).toBe("u1");
  });

  it("global_eval and latestEval", () => {
    let s = started();
    s = reduce(s, ev("global_eval", { round: 1, accuracy: 0.5, loss: 1, per_device_accuracy: {} }));
    s = reduce(s, ev("global_eval", { round: 2, accuracy: 0.66, loss: 0.9, per_device_accuracy: {} }));
    expect(latestEval(s)?.accuracy).toBe(0.66);
  });

  it("alert goes to the feed, newest first, with mapped source", () => {
    let s = started();
    s = reduce(s, ev("alert", { severity: "info", title: "a", detail: "", source: "coordinator" }));
    s = reduce(s, ev("alert", { severity: "high", title: "b", detail: "d", source: "finsentinel", device: "phone-01" }));
    expect(s.alerts.map((a) => a.title)).toEqual(["b", "a"]);
    expect(s.alerts[0]).toMatchObject({ source: "FinSentinel", severity: "high", device: "phone-01" });
    expect(s.alerts[1].source).toBe("Coordinator");
  });

  it("attack launched spawns an attacker envelope; result resolves it and is logged", () => {
    let s = started();
    s = reduce(s, ev("attack", { scene: "replay", phase: "launched", target: "phone-01", detail: "x" }));
    expect(s.flights.at(-1)).toMatchObject({ attacker: true, scene: "replay", outcome: null });
    s = reduce(s, ev("attack", { scene: "replay", phase: "result", http_status: 409, outcome: "blocked", detail: "dup" }));
    expect(s.flights.at(-1)).toMatchObject({ outcome: "blocked", outcomeLabel: "BLOCKED 409" });
    expect(s.alerts[0]).toMatchObject({ source: "Attack" });
    expect(s.alerts[0].title).toContain("409");
    // launched itself is not an alert (only results are)
    expect(s.alerts).toHaveLength(1);
  });

  it("only scenes whose traffic reaches the gate spawn an attacker envelope", () => {
    let s = started();
    for (const scene of ["poison", "drift", "weak_signal", "low_battery", "offline", "restore"]) {
      s = reduce(s, ev("attack", { scene, phase: "launched", target: "phone-01", detail: "" }));
    }
    expect(s.flights).toHaveLength(0);
    expect(s.attacks).toHaveLength(6);
    for (const scene of ["replay", "tamper", "forge", "mitm"]) {
      s = reduce(s, ev("attack", { scene, phase: "launched", detail: "" }));
    }
    expect(s.flights.map((f) => f.scene)).toEqual(["replay", "tamper", "forge", "mitm"]);
    s = reduce(s, ev("attack", { scene: "mitm", phase: "result", outcome: "observed", detail: "ciphertext only" }));
    expect(s.flights.at(-1)!.outcome).toBeNull();
    s = reduce(s, ev("attack", { scene: "mitm", phase: "result", outcome: "refused", detail: "tls" }));
    expect(s.flights.at(-1)).toMatchObject({ outcome: "blocked", outcomeLabel: "REFUSED" });
  });

  it("approval updates the paired gate record and is logged with its actor", () => {
    let s = started();
    s = reduce(s, gate("esp32-01", "u9", { severity: "MEDIUM", risk_score: 0.5 }));
    s = reduce(s, ev("approval", { device: "esp32-01", round: 1, update_id: "u9", decision: "approved", actor: "operator", reason: "ok" }));
    expect(s.gates.u9.decision).toBe("approved");
    expect(verdictOf(s.gates.u9).kind).toBe("accepted");
    expect(s.alerts[0].title).toContain("operator");
  });

  it("consecutive approvals by the same actor in a round collapse into one feed line", () => {
    let s = started();
    const ap = (device: string, update_id: string, actor = "coord") =>
      ev("approval", { device, round: 1, update_id, decision: "approved", actor, reason: "clean" });
    s = reduce(s, ap("phone-01", "a"));
    s = reduce(s, ap("esp32-01", "b"));
    expect(s.alerts).toHaveLength(1);
    expect(s.alerts[0].title).toBe("2 updates approved by coord");
    expect(s.alerts[0].detail).toContain("Phone 1, ESP32 board 1");
    s = reduce(s, ev("alert", { severity: "high", title: "x", detail: "", source: "finsentinel" }));
    s = reduce(s, ap("phone-01", "c"));
    expect(s.alerts).toHaveLength(3); // an alert in between breaks the group
    s = reduce(s, ap("phone-01", "d", "operator"));
    expect(s.alerts).toHaveLength(4); // different actor → new line
  });

  it("drift_monitor: only flagged ones reach the feed, labelled as the research monitor", () => {
    let s = started();
    s = reduce(s, ev("drift_monitor", { device: "phone-01", round: 1, score: 0.2, threshold: 2, flagged: false, finsentinel_flagged: false }));
    expect(s.alerts).toHaveLength(0);
    s = reduce(s, ev("drift_monitor", { device: "phone-01", round: 2, score: 3, threshold: 2, flagged: true, finsentinel_flagged: false }));
    expect(s.alerts[0].source).toBe("Drift monitor");
    expect(s.driftMonitor["phone-01"].score).toBe(3);
    expect(s.driftDevices).toEqual(["phone-01"]);
  });

  it("unknown event types are ignored but still advance seq", () => {
    const s0 = started();
    const s1 = reduce(s0, ev("something_new", { x: 1 }));
    expect(s1.lastSeq).toBe(s0.lastSeq + 1);
    expect(s1.devices).toBe(s0.devices);
  });
});

describe("reducer — dedupe and run changes", () => {
  it("drops events with seq <= lastSeq (reconnect replays history)", () => {
    const events = parseJsonl(SAMPLE).events;
    const once = reduceAll(initialState(), events);
    const twice = reduceAll(once, events);
    expect(twice).toBe(once); // identical object: nothing applied
    // replaying a prefix then the full stream gives the same result as the full stream
    const partial = reduceAll(reduceAll(initialState(), events.slice(0, 100)), events);
    expect(partial.lastSeq).toBe(once.lastSeq);
    expect(partial.alerts).toEqual(once.alerts);
    expect(partial.gates).toEqual(once.gates);
  });

  it("a different run_id resets all state", () => {
    let s = started();
    s = reduce(s, ev("alert", { severity: "info", title: "old", detail: "", source: "coordinator" }));
    expect(s.alerts).toHaveLength(1);
    const fresh = reduce(s, { seq: 1, t: 5, run_id: "run-B", type: "component", name: "coordinator", status: "up" } as Ev);
    expect(fresh.runId).toBe("run-B");
    expect(fresh.alerts).toHaveLength(0);
    expect(fresh.deviceOrder).toEqual([]);
    expect(fresh.lastSeq).toBe(1);
    expect(fresh.components.coordinator.status).toBe("up");
  });

  it("rootReducer reset / batch", () => {
    const evs = parseJsonl(SAMPLE).events.slice(0, 10);
    const s = rootReducer(initialState(), { kind: "batch", evs });
    expect(s.lastSeq).toBe(10);
    expect(rootReducer(s, { kind: "reset" })).toEqual(initialState());
  });
});

describe("reducer — gate pairing", () => {
  it("gate_result resolves the envelope with the same update_id", () => {
    let s = started();
    s = reduce(s, ev("update_submitted", { device: "phone-01", round: 1, update_id: "A", payload_bytes: 1, payload_sha256: "x", signed: true, path: "toxiproxy" }));
    s = reduce(s, ev("update_submitted", { device: "esp32-01", round: 1, update_id: "B", payload_bytes: 1, payload_sha256: "y", signed: true, path: "tunnel" }));
    s = reduce(s, gate("esp32-01", "B", { http_status: 202, approval_status: "quarantined", severity: "HIGH", checks: { ...PASS, normal: "fail" } }));
    const a = s.flights.find((f) => f.id === "A")!;
    const b = s.flights.find((f) => f.id === "B")!;
    expect(a.outcome).toBeNull();
    expect(b.outcome).toBe("quarantine");
    expect(s.lastGateId).toBe("B");
    expect(s.devices["esp32-01"].lastGateId).toBe("B");
    s = reduce(s, gate("phone-01", "A", { http_status: 401, approval_status: null, severity: null }));
    expect(s.flights.find((f) => f.id === "A")!).toMatchObject({ outcome: "blocked", outcomeLabel: "BLOCKED 401" });
  });

  it("classifies outcomes: pass / held / quarantine / blocked", () => {
    let s = started();
    s = reduce(s, gate("phone-01", "p", { severity: "LOW", approval_status: "pending_approval" }));
    s = reduce(s, gate("phone-01", "h", { severity: "MEDIUM", approval_status: "pending_approval" }));
    s = reduce(s, gate("phone-01", "q", { severity: "HIGH", approval_status: "quarantined" }));
    s = reduce(s, gate("phone-01", "b", { http_status: 422 }));
    expect(flightOutcomeOf(s.gates.p)).toBe("pass");
    expect(flightOutcomeOf(s.gates.h)).toBe("held");
    expect(flightOutcomeOf(s.gates.q)).toBe("quarantine");
    expect(flightOutcomeOf(s.gates.b)).toBe("blocked");
    expect(verdictOf(s.gates.p).text).toBe("Waiting for review");
    expect(verdictOf(s.gates.q).text).toBe("Quarantined");
    expect(verdictOf(s.gates.b).text).toBe("Blocked (422)");
  });

  it("attack gate_result reusing a device's update_id does not overwrite the honest record", () => {
    let s = started();
    s = reduce(s, gate("phone-01", "U", {}));
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "U", decision: "approved", actor: "c", reason: "" }));
    s = reduce(s, ev("attack", { scene: "replay", phase: "launched", target: "phone-01", detail: "" }));
    s = reduce(s, gate("phone-01", "U", { http_status: 409, source: "attack", approval_status: null, severity: null }));
    expect(verdictOf(s.gates.U).kind).toBe("accepted");
    expect(s.devices["phone-01"].lastGateId).toBe("U");
    const atk = s.gates[s.lastGateId!];
    expect(atk.ev.http_status).toBe(409);
    expect(s.flights.at(-1)).toMatchObject({ attacker: true, outcome: "blocked", outcomeLabel: "BLOCKED 409" });
  });
});

describe("reducer — attack traffic announced as update_submitted (real coordinator behaviour)", () => {
  it("the attacker envelope adopts the submission and pairs with its gate_result; device records untouched", () => {
    let s = started();
    s = reduce(s, ev("update_submitted", { device: "phone-01", round: 3, update_id: "U3", payload_bytes: 1, payload_sha256: "x", signed: true, path: "toxiproxy" }));
    s = reduce(s, gate("phone-01", "U3", { round: 3 }));
    s = reduce(s, ev("attack", { scene: "replay", phase: "launched", target: "phone-01", detail: "" }));
    s = reduce(s, ev("update_submitted", { device: "phone-01", round: 3, update_id: "U3#replay", payload_bytes: 1, payload_sha256: "x", signed: true, path: "toxiproxy" }));
    expect(s.flights).toHaveLength(2);
    expect(s.flights[1]).toMatchObject({ id: "U3#replay", attacker: true, outcome: null });
    expect(s.devices["phone-01"].submittedByRound[3]).toBe("U3");
    s = reduce(s, gate("phone-01", "U3#replay", { round: 3, http_status: 409, source: "attack", severity: null, approval_status: null }));
    expect(s.flights[1]).toMatchObject({ outcome: "blocked", outcomeLabel: "BLOCKED 409" });
    s = reduce(s, ev("attack", { scene: "replay", phase: "result", target: "phone-01", http_status: 409, outcome: "blocked", detail: "" }));
    expect(s.flights.every((f) => f.outcome !== null)).toBe(true);
    expect(s.devices["phone-01"].lastGateId).toBe("U3");
    expect(verdictOf(s.gates.U3).kind).toBe("review");
  });
});

describe("reducer — pending review list", () => {
  it("lists pending, non-LOW, undecided, non-blocked updates only", () => {
    let s = started();
    s = reduce(s, gate("phone-01", "low", { severity: "LOW" }));
    s = reduce(s, gate("phone-01", "med", { severity: "MEDIUM" }));
    s = reduce(s, gate("esp32-01", "high", { severity: "high" }));
    s = reduce(s, gate("esp32-01", "quar", { severity: "HIGH", approval_status: "quarantined" }));
    s = reduce(s, gate("esp32-01", "blk", { severity: "HIGH", http_status: 409 }));
    expect(pendingReview(s).map((g) => g.ev.update_id)).toEqual(["med", "high"]);
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "med", decision: "rejected", actor: "op", reason: "" }));
    expect(pendingReview(s).map((g) => g.ev.update_id)).toEqual(["high"]);
    s = reduce(s, ev("approval", { device: "esp32-01", round: 1, update_id: "high", decision: "approved", actor: "op", reason: "" }));
    expect(pendingReview(s)).toHaveLength(0);
  });

  it("an approval that arrives before its gate_result is not lost", () => {
    let s = started();
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "z", decision: "approved", actor: "op", reason: "" }));
    s = reduce(s, gate("phone-01", "z", { severity: "MEDIUM" }));
    expect(pendingReview(s)).toHaveLength(0);
    expect(verdictOf(s.gates.z).kind).toBe("accepted");
    expect(s.orphanDecisions).toEqual({});
  });
});

describe("reducer — drift / dropout markers", () => {
  it("marks rounds where a device dropped because of drift (round_completed and device_status)", () => {
    let s = started();
    s = reduce(s, ev("round_completed", { round: 2, participants: [], dropped: [{ device: "phone-01", reason: "low_battery" }], aggregated: true, model_sha256: "" }));
    expect(s.driftDropRounds).toEqual([]);
    s = reduce(s, ev("device_status", { device: "phone-01", state: "skipped", round: 4, reason: "drift" }));
    s = reduce(s, ev("round_completed", { round: 3, participants: [], dropped: [{ device: "esp32-01", reason: "drift" }], aggregated: true, model_sha256: "" }));
    s = reduce(s, ev("round_completed", { round: 4, participants: [], dropped: [{ device: "phone-01", reason: "drift" }], aggregated: true, model_sha256: "" }));
    expect(s.driftDropRounds).toEqual([3, 4]);
    expect(s.driftDevices.sort()).toEqual(["esp32-01", "phone-01"]);
  });
});

describe("reducer — full sample recording", () => {
  const { events, skipped } = parseJsonl(SAMPLE);
  const s = reduceAll(initialState(), events);

  it("parses cleanly and covers the scripted story", () => {
    expect(skipped).toBe(0);
    expect(s.deviceOrder).toHaveLength(9);
    expect(s.deviceOrder.filter((id) => s.devices[id].info.kind === "wokwi")).toHaveLength(3);
    expect(s.phase).toBe("finished");
    expect(Object.keys(s.evals)).toHaveLength(5); // round 0 (initial model) + 4 rounds
    const gates = Object.values(s.gates);
    expect(gates.some((g) => g.ev.http_status === 409 && g.ev.checks.replay === "fail")).toBe(true);
    expect(gates.some((g) => g.ev.http_status === 422 && g.ev.checks.untampered === "fail")).toBe(true);
    expect(gates.some((g) => verdictOf(g).kind === "quarantined" && g.ev.device === "phone-03")).toBe(true);
    expect(s.devices["phone-04"].statesByRound[2]).toContain("skipped");
    expect(s.devices["phone-04"].reason).toBe("low_battery");
    expect(s.driftDropRounds).toEqual([3, 4]);
    expect(s.alerts.some((a) => a.source === "Drift monitor")).toBe(true);
    // the MEDIUM update was approved by a human in round 3 → nothing left waiting
    expect(pendingReview(s)).toHaveLength(0);
    // every envelope found its ending
    expect(s.flights.every((f) => f.outcome !== null)).toBe(true);
  });

  it("mid-run the MEDIUM update is waiting for review", () => {
    const mid = events.findIndex((e) => e.type === "approval" && e.actor.startsWith("presenter"));
    const m = reduceAll(initialState(), events.slice(0, mid));
    expect(pendingReview(m).map((g) => g.ev.device)).toEqual(["esp32-02"]);
  });

  it("step tracker lights from real events only", () => {
    const d = s.devices["phone-01"];
    expect(stepsFor(s, d, 1).map((x) => x.state)).toEqual(["done", "done", "done", "done", "done", "done"]);
    const skippedDev = s.devices["phone-05"];
    expect(stepsFor(s, skippedDev, 3).map((x) => x.state)).toEqual(["idle", "idle", "idle", "idle", "idle", "idle"]);
    const q = stepsFor(s, s.devices["phone-03"], 2);
    expect(q[5]).toMatchObject({ state: "fail", note: "Quarantined" });
  });
});
