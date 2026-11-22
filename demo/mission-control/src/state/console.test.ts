import { describe, expect, it } from "vitest";
import SAMPLE_RAW from "../../public/sample-recording.jsonl?raw";
import type { Ev, GateChecks } from "../types";
import { initialState, reduce, reduceAll, type DashState } from "./reducer";
import { buildConsoleView, filterRows, mergeUpdates, normDecision, parseConsole, rowsFromEvents } from "./console";
import { parseJsonl } from "../replay/engine";

let seq = 0;
function ev(type: string, fields: Record<string, unknown> = {}): Ev {
  seq++;
  return { seq, t: 1000 + seq, run_id: "run-C", type, ...fields } as unknown as Ev;
}
const PASS: GateChecks = { untampered: "pass", sender: "pass", shape: "pass", replay: "pass", normal: "pass" };

function base(): DashState {
  seq = 0;
  let s = reduce(
    initialState(),
    ev("run_started", {
      model_version: "m",
      param_count: 1,
      rounds: 9,
      round_seconds: 30,
      test_samples: 1,
      mode: "live",
      devices: [
        { id: "phone-01", kind: "virtual", label: "Phone 1", subject: 1, network: "wifi", battery: 90 },
        { id: "phone-03", kind: "virtual", label: "Phone 3", subject: 3, network: "wifi", battery: 90 },
      ],
    }),
  );
  s = reduce(s, ev("decision_engine", { engine: "wazuh", detail: "Wazuh decides", wazuh_dashboard: "https://wazuh.local" }));
  s = reduce(s, ev("gate_result", { device: "phone-01", round: 1, update_id: "A", http_status: 200, checks: PASS, risk_score: 0, severity: "LOW", approval_status: "pending_approval", reasons: [], detail: "", source: "device", decided_by: "pending" }));
  s = reduce(
    s,
    ev("wazuh_decision", { device: "phone-01", update_id: "A", round: 1, decision: "approve", rule_id: "100201", rule_level: 3, rule_description: "clean", event_type: "x", enforced: true, enforcement_detail: "ok", latency_ms: 800 }),
  );
  s = reduce(s, ev("gate_result", { device: "phone-03", round: 1, update_id: "B", http_status: 200, checks: { ...PASS, normal: "fail" }, risk_score: 75, severity: "CRITICAL", approval_status: "pending_approval", reasons: ["norm"], detail: "", source: "device", decided_by: "pending" }));
  s = reduce(s, ev("device_status", { device: "phone-03", state: "suspended", round: 1, reason: "100213" }));
  return s;
}

const API = {
  decision_engine: "wazuh",
  wazuh: { up: true, decisions: 41, last_decision_t: 1795.5, dashboard_url: "https://localhost" },
  finsentinel: { approvals: { pending_count: 1 }, outbox: { pending_count: 0, dead_letter_count: 0 } },
  pending: [
    { update_id: "B", client_id: "phone-03", round_number: 1, risk_score: 75, risk_severity: "CRITICAL", anomaly_reasons: ["norm"], status: "pending_approval", model_version: "m" },
    { bogus: true },
  ],
  suspended: [{ device: "esp32-02", rule_id: "100213", since_round: 7 }],
  thresholds: { FINSENTINEL_MAX_L2: 11.6 },
  updates: [
    { update_id: "B", device: "phone-03", round: 1, http_status: 200, finsentinel_status: "quarantined", risk_score: 75, severity: "CRITICAL", reasons: ["norm"], decision: "quarantine", decided_by: "wazuh", rule_id: "100211", t: 2000 },
    { update_id: "Z", device: "phone-01", round: 0, http_status: 200, finsentinel_status: "approved", risk_score: 0, severity: "LOW", reasons: [], decision: "approved", decided_by: "coordinator", rule_id: null, t: 500 },
  ],
};

describe("console — parsing the REST snapshot", () => {
  it("normalises a full payload and drops malformed items", () => {
    const c = parseConsole(API)!;
    expect(c.decision_engine).toBe("wazuh");
    expect(c.wazuh).toEqual({ up: true, decisions: 41, last_decision_t: 1795.5, dashboard_url: "https://localhost" });
    expect(c.pending).toHaveLength(1);
    expect(c.pending[0]).toMatchObject({ update_id: "B", client_id: "phone-03", round_number: 1, risk_score: 75, risk_severity: "CRITICAL", anomaly_reasons: ["norm"] });
    expect(c.suspended).toEqual([{ device: "esp32-02", rule_id: "100213", since_round: 7 }]);
    expect(c.updates.map((u) => u.update_id)).toEqual(["B", "Z"]);
  });

  it("tolerates missing sections and rejects non-objects", () => {
    expect(parseConsole(null)).toBeNull();
    expect(parseConsole([1, 2])).toBeNull();
    const c = parseConsole({ decision_engine: "coordinator" })!;
    expect(c).toMatchObject({ decision_engine: "coordinator", wazuh: null, finsentinel: null, pending: [], suspended: [], updates: [] });
  });

  it("normDecision maps both verb and status spellings", () => {
    expect(normDecision("approved")).toBe("approve");
    expect(normDecision("quarantine")).toBe("quarantine");
    expect(normDecision("suspend_device")).toBe("suspend");
    expect(normDecision("pending_approval")).toBe("pending");
    expect(normDecision("weird")).toBeNull();
  });
});

describe("console — merging events with the API", () => {
  it("event rows carry Wazuh rule details paired by update_id", () => {
    const rows = rowsFromEvents(base());
    const a = rows.find((r) => r.updateId === "A")!;
    expect(a).toMatchObject({ decision: "approve", ruleId: "100201", ruleLevel: 3, enforced: true, decidedBy: "wazuh", from: "events" });
    const b = rows.find((r) => r.updateId === "B")!;
    expect(b).toMatchObject({ decision: "pending", decidedBy: "pending", ruleId: null });
  });

  it("API fields win, events fill the gaps, API-only rows are kept, newest first", () => {
    const rows = mergeUpdates(base(), parseConsole(API));
    expect(rows.map((r) => r.updateId)).toEqual(["B", "A", "Z"]);
    expect(rows[0]).toMatchObject({ decision: "quarantine", decidedBy: "wazuh", ruleId: "100211", fsStatus: "quarantined", from: "both", t: 2000 });
    expect(rows[2]).toMatchObject({ updateId: "Z", decision: "approve", from: "api" });
  });

  it("filterRows by device and decision", () => {
    const rows = mergeUpdates(base(), parseConsole(API));
    expect(filterRows(rows, { device: "phone-01", decision: "" }).map((r) => r.updateId)).toEqual(["A", "Z"]);
    expect(filterRows(rows, { device: "", decision: "quarantine" }).map((r) => r.updateId)).toEqual(["B"]);
    expect(filterRows(rows, { device: "phone-03", decision: "approve" })).toEqual([]);
  });

  it("without the API (replay): pending, suspended and engine come from events", () => {
    const v = buildConsoleView(base(), null);
    expect(v.apiAvailable).toBe(false);
    expect(v.engine).toBe("wazuh");
    expect(v.dashboardUrl).toBe("https://wazuh.local");
    expect(v.decisions).toBe(1);
    expect(v.pending.map((p) => p.updateId)).toEqual(["B"]);
    expect(v.suspended).toEqual([{ device: "phone-03", ruleId: "100213", sinceRound: 1 }]);
    expect(v.finsentinel).toBeNull();
  });

  it("with the API: the API is authoritative for pending, suspended, counts and links", () => {
    const v = buildConsoleView(base(), parseConsole(API));
    expect(v.apiAvailable).toBe(true);
    expect(v.decisions).toBe(41);
    expect(v.wazuhUp).toBe(true);
    expect(v.dashboardUrl).toBe("https://localhost");
    expect(v.pending).toHaveLength(1);
    expect(v.suspended.map((x) => x.device)).toEqual(["esp32-02"]);
    expect(v.thresholds).toEqual({ FINSENTINEL_MAX_L2: 11.6 });
  });

  it("v1 run without decision_engine: engine is the coordinator", () => {
    seq = 0;
    const s = reduce(initialState(), ev("run_started", { model_version: "m", param_count: 1, rounds: 1, round_seconds: 1, test_samples: 1, mode: "live", devices: [] }));
    expect(buildConsoleView(s, null).engine).toBe("coordinator");
    expect(buildConsoleView(initialState(), null).engine).toBe("unknown");
  });

  it("the sample recording yields one row per gate result, attacks included", () => {
    const s = reduceAll(initialState(), parseJsonl(SAMPLE_RAW as string).events);
    const v = buildConsoleView(s, null);
    expect(v.rows).toHaveLength(Object.keys(s.gates).length);
    expect(v.rows.filter((r) => r.attack).every((r) => r.decision === "blocked")).toBe(true);
    expect(v.suspended.map((x) => x.device)).toEqual(["phone-03"]);
    expect(v.pending).toHaveLength(0);
  });
});
