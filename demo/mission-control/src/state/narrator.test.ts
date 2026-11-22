import { describe, expect, it } from "vitest";
import type { Ev, GateChecks } from "../types";
import { initialState, reduce, reduceAll, type DashState } from "./reducer";
import { narrate, NARRATOR_WINDOW_S } from "./narrator";
import { parseJsonl } from "../replay/engine";

// The real v1 recording of a live run (no Wazuh). Optional: an empty glob skips the test.
const GOLDEN = Object.values(
  import.meta.glob("../../../recordings/golden.jsonl", { query: "?raw", import: "default", eager: true }),
)[0] as string | undefined;

let seq = 0;
function ev(type: string, fields: Record<string, unknown> = {}, dt = 1): Ev {
  seq++;
  return { seq, t: 1000 + seq * dt, run_id: "run-N", type, ...fields } as unknown as Ev;
}
const PASS: GateChecks = { untampered: "pass", sender: "pass", shape: "pass", replay: "pass", normal: "pass" };

function started(engine: "wazuh" | "coordinator" | null = "wazuh"): DashState {
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
        { id: "esp32-01", kind: "wokwi", label: "ESP32 board 1", subject: 2, network: "wokwi-guest", battery: 100 },
      ],
    }),
  );
  if (engine) s = reduce(s, ev("decision_engine", { engine, detail: "" }));
  return s;
}

function gate(update_id: string, over: Record<string, unknown> = {}): Ev {
  return ev("gate_result", {
    device: "phone-01",
    round: 5,
    update_id,
    http_status: 200,
    checks: PASS,
    risk_score: 0,
    severity: "LOW",
    approval_status: "pending_approval",
    reasons: [],
    detail: "",
    source: "device",
    decided_by: "pending",
    ...over,
  });
}

describe("narrator — phase narration", () => {
  it("before anything happens", () => {
    expect(narrate(initialState()).headline).toBe("Waiting for the system to start…");
  });

  it("fleet coming online, then round open", () => {
    let s = started();
    expect(narrate(s).headline).toMatch(/fleet is coming online: 2 devices/);
    s = reduce(s, ev("round_started", { round: 3, deadline: 0, model_sha256: "" }));
    expect(narrate(s).headline).toBe("Round 3: the server sent out the signed model — devices are training on their own data.");
    expect(narrate(s).detail).toBe("0 of 2 updates received so far.");
  });

  it("deciding lists the devices Wazuh is still deciding on", () => {
    let s = started();
    s = reduce(s, ev("round_phase", { round: 4, phase: "deciding", waiting_for: ["phone-01", "esp32-01"] }));
    expect(narrate(s)).toMatchObject({
      headline: "Round 4: FinSentinel has checked every update — waiting for Wazuh to decide on Phone 1 and ESP32 board 1.",
      tone: "wazuh",
    });
  });

  it("round complete reports participants and the accuracy change", () => {
    let s = started();
    s = reduce(s, ev("global_eval", { round: 1, accuracy: 0.6, loss: 1, per_device_accuracy: {} }));
    s = reduce(s, ev("round_completed", { round: 2, participants: ["phone-01"], dropped: [{ device: "esp32-01", reason: "low_battery" }], aggregated: true, model_sha256: "" }));
    s = reduce(s, ev("global_eval", { round: 2, accuracy: 0.65, loss: 1, per_device_accuracy: {} }));
    const n = narrate(s);
    expect(n.headline).toBe("Round 2: complete — 1 device contributed, global accuracy now 65.0% (+5.0).");
    expect(n.detail).toBe("Left out: ESP32 board 1 (low battery).");
  });
});

describe("narrator — incident stories built from real events", () => {
  it("poisoned update: flagged → waiting for Wazuh → Wazuh rule quarantines it", () => {
    let s = started();
    s = reduce(s, ev("device_status", { device: "phone-01", state: "sending", round: 5, hacked: true }));
    s = reduce(s, gate("P", { severity: "CRITICAL", risk_score: 75, checks: { ...PASS, normal: "fail" }, reasons: ["norm too high"] }));
    expect(narrate(s)).toMatchObject({
      headline: "Round 5: Phone 1 sent poisoned weights → FinSentinel flagged it (risk 75, CRITICAL) → waiting for Wazuh to decide…",
      tone: "warn",
    });
    s = reduce(
      s,
      ev("wazuh_decision", {
        device: "phone-01",
        update_id: "P",
        round: 5,
        decision: "quarantine",
        rule_id: "100211",
        rule_level: 12,
        rule_description: "anomalous update — quarantine",
        event_type: "security.anomaly_detected",
        enforced: true,
        enforcement_detail: "",
        latency_ms: 1200,
      }),
    );
    const n = narrate(s);
    expect(n.headline).toBe("Round 5: Phone 1 sent poisoned weights → FinSentinel flagged it (risk 75, CRITICAL) → Wazuh rule 100211 quarantined it");
    expect(n.tone).toBe("bad");
    expect(n.detail).toBe("anomalous update — quarantine · norm too high");
  });

  it("replay attack: blocked at 'Not a replay' with the HTTP code", () => {
    let s = started();
    s = reduce(s, ev("attack", { scene: "replay", phase: "launched", target: "phone-01", detail: "" }));
    s = reduce(
      s,
      gate("U#replay", {
        http_status: 409,
        source: "attack",
        severity: null,
        risk_score: null,
        approval_status: null,
        checks: { untampered: "pass", sender: "pass", shape: "pass", replay: "fail", normal: "skip" },
      }),
    );
    s = reduce(s, ev("attack", { scene: "replay", phase: "result", target: "phone-01", http_status: 409, outcome: "blocked", detail: "" }));
    expect(narrate(s)).toMatchObject({
      headline: "Round 5: An attacker re-sent Phone 1's old update → FinSentinel blocked it at “Not a replay” (HTTP 409)",
      tone: "bad",
    });
  });

  it("suspension by a Wazuh rule", () => {
    let s = started();
    s = reduce(
      s,
      ev("wazuh_decision", {
        device: "phone-01",
        update_id: null,
        round: 7,
        decision: "suspend_device",
        rule_id: "100213",
        rule_level: 14,
        rule_description: "repeat offender",
        event_type: "x",
        enforced: true,
        enforcement_detail: "",
        latency_ms: null,
      }),
    );
    expect(narrate(s).headline).toBe("Round 7: Wazuh rule 100213 (level 14) suspended Phone 1.");
    s = reduce(s, ev("device_status", { device: "phone-01", state: "suspended", round: 7, reason: "100213" }));
    expect(narrate(s).headline).toBe("Round 7: Phone 1 is suspended by Wazuh rule 100213 — its updates are no longer accepted.");
  });

  it("drift dropout and the eavesdropper", () => {
    let s = started();
    s = reduce(s, ev("device_status", { device: "esp32-01", state: "skipped", round: 6, reason: "drift" }));
    expect(narrate(s).headline).toBe("Round 6: ESP32 board 1 sits this round out — its own data drifted — the server only sees an absence.");
    s = reduce(s, ev("attack", { scene: "mitm", phase: "result", outcome: "observed", detail: "ciphertext only" }));
    expect(narrate(s).headline).toMatch(/^An eavesdropper tapped the line → it captured traffic but saw only encrypted bytes/);
  });

  it("a human override is named as such", () => {
    let s = started();
    s = reduce(s, gate("M", { severity: "MEDIUM", risk_score: 40, checks: { ...PASS, normal: "warn" } }));
    s = reduce(s, ev("approval", { device: "phone-01", round: 5, update_id: "M", decision: "approved", actor: "presenter (Mission Control)", reason: "benign" }));
    expect(narrate(s).headline).toBe("Round 5: Phone 1 sent a medium-risk update → FinSentinel flagged it (risk 40, MEDIUM) → the operator approved it by hand");
  });

  it("v1 (no Wazuh): the coordinator / FinSentinel policy is named, never Wazuh", () => {
    let s = started(null);
    s = reduce(s, gate("Q", { severity: "CRITICAL", risk_score: 75, approval_status: "quarantined", checks: { ...PASS, normal: "fail" }, decided_by: undefined }));
    s = reduce(s, ev("approval", { device: "phone-01", round: 5, update_id: "Q", decision: "quarantined", actor: "finsentinel (risk policy)", reason: "" }));
    expect(narrate(s).headline).toBe("Round 5: Phone 1 sent an unusual update → FinSentinel flagged it (risk 75, CRITICAL) → FinSentinel's risk policy quarantined it");
  });

  it(`incidents expire after ${NARRATOR_WINDOW_S}s of recording time`, () => {
    let s = started();
    s = reduce(s, ev("device_status", { device: "esp32-01", state: "offline", round: 2 }));
    expect(narrate(s).headline).toMatch(/went offline/);
    s = reduce(s, ev("round_started", { round: 3, deadline: 0, model_sha256: "" }, 1));
    // jump the clock forward by more than the window
    s = reduce(s, { seq: 999, t: (s.lastT ?? 0) + NARRATOR_WINDOW_S + 1, run_id: "run-N", type: "round_phase", round: 3, phase: "open" } as Ev);
    expect(narrate(s).headline).toMatch(/^Round 3: the server sent out the signed model/);
  });
});

describe("narrator — real v1 recording (golden.jsonl)", () => {
  it.skipIf(!GOLDEN)("narrates every prefix without inventing Wazuh", () => {
    const { events, skipped } = parseJsonl(GOLDEN!);
    expect(skipped).toBe(0);
    let s = initialState();
    const seen = new Set<string>();
    for (const e of events) {
      s = reduce(s, e);
      const n = narrate(s);
      expect(n.headline.length).toBeGreaterThan(0);
      expect(n.headline).not.toMatch(/wazuh/i);
      seen.add(n.headline);
    }
    expect([...seen].some((h) => /poisoned weights/.test(h))).toBe(true);
    expect([...seen].some((h) => /Not a replay/.test(h))).toBe(true);
    expect(reduceAll(initialState(), events).engine).toBeNull();
  });
});
