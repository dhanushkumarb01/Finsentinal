// v2 — Wazuh decides (EVENTS.md "v2" section): decision_engine, round_phase, wazuh_decision,
// gate_result.decided_by, device_status "suspended", alert source "wazuh".
import { describe, expect, it } from "vitest";
import SAMPLE_RAW from "../../public/sample-recording.jsonl?raw";
import type { Ev, GateChecks } from "../types";
import {
  awaitingDecision,
  deciderOf,
  deciderOfActor,
  initialState,
  reduce,
  reduceAll,
  verdictOf,
  type DashState,
} from "./reducer";
import { parseJsonl } from "../replay/engine";

const RUN = "run-v2";
let seq = 0;
function ev(type: string, fields: Record<string, unknown> = {}): Ev {
  seq++;
  return { seq, t: 1000 + seq, run_id: RUN, type, ...fields } as unknown as Ev;
}
const PASS: GateChecks = { untampered: "pass", sender: "pass", shape: "pass", replay: "pass", normal: "pass" };

export function started(): DashState {
  seq = 0;
  return reduce(
    initialState(),
    ev("run_started", {
      model_version: "har-mlp-v1",
      param_count: 2278,
      rounds: 9,
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
    http_status: 200,
    checks: PASS,
    risk_score: 0,
    severity: "LOW",
    approval_status: "pending_approval",
    reasons: [],
    detail: "",
    source: "device",
    ...over,
  });
}

function wz(over: Record<string, unknown> = {}): Ev {
  return ev("wazuh_decision", {
    device: "phone-01",
    update_id: "U",
    round: 1,
    decision: "approve",
    rule_id: "100201",
    rule_level: 3,
    rule_description: "clean update",
    event_type: "security.update_validated",
    enforced: true,
    enforcement_detail: "approved via operator API",
    latency_ms: 900,
    ...over,
  });
}

describe("v2 — decision engine and round phases", () => {
  it("decision_engine is recorded with the Wazuh dashboard link", () => {
    let s = started();
    expect(s.engine).toBeNull();
    s = reduce(s, ev("decision_engine", { engine: "wazuh", detail: "Wazuh decides", wazuh_dashboard: "https://localhost" }));
    expect(s.engine).toEqual({ engine: "wazuh", detail: "Wazuh decides", dashboard: "https://localhost" });
    s = reduce(s, ev("decision_engine", { engine: "coordinator", detail: "fallback" }));
    expect(s.engine).toMatchObject({ engine: "coordinator", dashboard: null });
  });

  it("round_phase drives open → deciding (with waiting list) → aggregating; round events are inferred", () => {
    let s = started();
    s = reduce(s, ev("round_started", { round: 2, deadline: 0, model_sha256: "" }));
    expect(s.roundPhase).toMatchObject({ round: 2, phase: "open", explicit: false });
    s = reduce(s, ev("round_phase", { round: 2, phase: "deciding", waiting_for: ["phone-01", "esp32-01"] }));
    expect(s.roundPhase).toMatchObject({ round: 2, phase: "deciding", waitingFor: ["phone-01", "esp32-01"], explicit: true });
    s = reduce(s, ev("round_phase", { round: 2, phase: "aggregating" }));
    expect(s.roundPhase).toMatchObject({ phase: "aggregating", waitingFor: [] });
    s = reduce(s, ev("round_completed", { round: 2, participants: [], dropped: [], aggregated: true, model_sha256: "" }));
    expect(s.roundPhase).toMatchObject({ round: 2, phase: "done" });
  });
});

describe("v2 — wazuh_decision pairing by update_id", () => {
  it("pairs a decision that arrives AFTER its gate_result", () => {
    let s = started();
    s = reduce(s, gate("phone-01", "U", { decided_by: "pending", severity: "CRITICAL", checks: { ...PASS, normal: "fail" } }));
    expect(deciderOf(s.gates.U, s)).toBe("pending");
    s = reduce(s, wz({ decision: "quarantine", rule_id: "100211", rule_level: 12 }));
    expect(s.wazuhByUpdate.U.rule_id).toBe("100211");
    expect(deciderOf(s.gates.U, s)).toBe("wazuh");
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "U", decision: "quarantined", actor: "wazuh rule 100211 (level 12)", reason: "" }));
    expect(verdictOf(s.gates.U).kind).toBe("quarantined");
    expect(deciderOf(s.gates.U, s)).toBe("wazuh");
  });

  it("pairs a decision (and its approval) that arrive BEFORE the gate_result", () => {
    let s = started();
    s = reduce(s, wz({ update_id: "E" }));
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "E", decision: "approved", actor: "wazuh rule 100201 (level 3)", reason: "" }));
    s = reduce(s, gate("phone-01", "E", { decided_by: "pending" }));
    expect(s.wazuhByUpdate.E.decision).toBe("approve");
    expect(verdictOf(s.gates.E).kind).toBe("accepted");
    expect(deciderOf(s.gates.E, s)).toBe("wazuh");
  });

  it("an un-enforced decision does not count as decided", () => {
    let s = started();
    s = reduce(s, gate("phone-01", "M", { decided_by: "pending", severity: "MEDIUM" }));
    s = reduce(s, wz({ update_id: "M", decision: "alert", enforced: false }));
    expect(deciderOf(s.gates.M, s)).toBe("pending");
    expect(awaitingDecision(s).map((g) => g.ev.update_id)).toEqual(["M"]);
  });

  it("a null update_id is logged but not paired; newest first; feed source is Wazuh", () => {
    let s = started();
    s = reduce(s, wz({ update_id: null, decision: "alert", rule_id: "100230", enforced: false }));
    s = reduce(s, wz({ update_id: "X", decision: "reject", rule_id: "100212" }));
    expect(s.wazuhDecisions.map((w) => w.rule_id)).toEqual(["100212", "100230"]);
    expect(Object.keys(s.wazuhByUpdate)).toEqual(["X"]);
    expect(s.alerts[0]).toMatchObject({ source: "Wazuh", severity: "high" });
    expect(s.alerts[1].title).toMatch(/not enforced/);
  });

  it("consecutive Wazuh approvals in a round collapse into one feed line; Wazuh approvals are not double-logged", () => {
    let s = started();
    s = reduce(s, wz({ update_id: "A", device: "phone-01" }));
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "A", decision: "approved", actor: "wazuh rule 100201 (level 3)", reason: "" }));
    s = reduce(s, wz({ update_id: "B", device: "esp32-01" }));
    expect(s.alerts).toHaveLength(1);
    expect(s.alerts[0]).toMatchObject({ count: 2, title: "Rule 100201 approved 2 updates" });
  });

  it("alert source 'wazuh' maps to the Wazuh feed source", () => {
    let s = started();
    s = reduce(s, ev("alert", { severity: "critical", title: "x", detail: "", source: "wazuh" }));
    expect(s.alerts[0].source).toBe("Wazuh");
  });
});

describe("v2 — decided_by", () => {
  it("is stored from gate_result and is null for v1 events", () => {
    let s = started();
    s = reduce(s, gate("phone-01", "v1"));
    s = reduce(s, gate("phone-01", "v2", { decided_by: "pending" }));
    expect(s.gates.v1.decidedBy).toBeNull();
    expect(s.gates.v2.decidedBy).toBe("pending");
  });

  it("classifies actors: wazuh / human / coordinator / finsentinel; blocked requests = finsentinel", () => {
    expect(deciderOfActor("wazuh rule 100201 (level 3)")).toBe("wazuh");
    expect(deciderOfActor("presenter (Mission Control)")).toBe("human");
    expect(deciderOfActor("demo-coordinator")).toBe("coordinator");
    expect(deciderOfActor("finsentinel (risk policy)")).toBe("finsentinel");
    let s = started();
    s = reduce(s, gate("phone-01", "b", { http_status: 409, decided_by: "pending" }));
    expect(deciderOf(s.gates.b, s)).toBe("finsentinel");
    s = reduce(s, gate("phone-01", "c", { decided_by: "coordinator", approval_status: "approved" }));
    expect(deciderOf(s.gates.c, s)).toBe("coordinator");
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "c", decision: "rejected", actor: "presenter (Mission Control)", reason: "" }));
    expect(deciderOf(s.gates.c, s)).toBe("human");
  });

  it("awaitingDecision lists undecided device updates of any severity (Wazuh decides LOW ones too)", () => {
    let s = started();
    s = reduce(s, gate("phone-01", "low", { decided_by: "pending" }));
    s = reduce(s, gate("esp32-01", "hi", { decided_by: "pending", severity: "HIGH" }));
    s = reduce(s, gate("esp32-01", "blk", { http_status: 422 }));
    expect(awaitingDecision(s).map((g) => g.ev.update_id)).toEqual(["low", "hi"]);
    s = reduce(s, ev("approval", { device: "phone-01", round: 1, update_id: "low", decision: "approved", actor: "wazuh rule 1 (level 3)", reason: "" }));
    expect(awaitingDecision(s).map((g) => g.ev.update_id)).toEqual(["hi"]);
  });
});

describe("v2 — suspended devices", () => {
  it("device_status 'suspended' marks the device with rule and round; a later normal status lifts it", () => {
    let s = started();
    s = reduce(s, ev("round_started", { round: 7, deadline: 0, model_sha256: "" }));
    s = reduce(s, ev("device_status", { device: "phone-01", state: "suspended", round: 7, reason: "100213" }));
    expect(s.devices["phone-01"]).toMatchObject({ suspended: true, suspendedRule: "100213", suspendedRound: 7, state: "suspended" });
    expect(s.notable[0]).toMatchObject({ kind: "device", state: "suspended", reason: "100213" });
    // being skipped *because* it is suspended keeps it suspended
    s = reduce(s, ev("device_status", { device: "phone-01", state: "skipped", round: 8, reason: "suspended" }));
    expect(s.devices["phone-01"]).toMatchObject({ suspended: true, suspendedRound: 7 });
    s = reduce(s, ev("device_status", { device: "phone-01", state: "downloading", round: 9 }));
    expect(s.devices["phone-01"]).toMatchObject({ suspended: false, suspendedRule: null });
  });
});

describe("v2 — full sample recording", () => {
  const { events, skipped } = parseJsonl(SAMPLE_RAW as string);
  const s = reduceAll(initialState(), events);
  it("contains every v2 decision kind and ends with Phone 3 suspended", () => {
    expect(skipped).toBe(0);
    expect(s.engine?.engine).toBe("wazuh");
    const kinds = new Set<string>(s.wazuhDecisions.map((w) => w.decision));
    for (const k of ["approve", "quarantine", "reject", "suspend_device", "alert"]) expect(kinds.has(k)).toBe(true);
    expect(s.devices["phone-03"]).toMatchObject({ suspended: true, suspendedRule: "100213" });
    expect(events.some((e) => e.type === "round_phase" && e.phase === "deciding")).toBe(true);
    const med = Object.values(s.gates).find((g) => g.ev.severity === "MEDIUM")!;
    expect(deciderOf(med, s)).toBe("human");
    const poisoned = Object.values(s.gates).find((g) => g.ev.device === "phone-03" && g.ev.round === 2)!;
    expect(deciderOf(poisoned, s)).toBe("wazuh");
    expect(verdictOf(poisoned).kind).toBe("quarantined");
    const r4 = Object.values(s.gates).find((g) => g.ev.device === "phone-03" && g.ev.round === 4)!;
    expect(verdictOf(r4).kind).toBe("rejected");
  });
});
