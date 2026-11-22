// Generates public/sample-recording.jsonl — a deterministic, EVENTS.md-conformant FIXTURE (v2:
// Wazuh decides) used for tests and as the dev default replay. It is NOT a recording of a real
// run; the real demo replays demo/recordings/<run_id>.jsonl produced by the coordinator.
//   node scripts/gen-sample.mjs
//
// Story: R1 everyone learns (Wazuh approves) · R2 low battery, poisoned Phone 3 quarantined by
// Wazuh, MEDIUM update left for a human, replay blocked · R3 drift dropout, presenter approves the
// MEDIUM update by hand, tamper blocked · R4 Phone 3 poisons again → Wazuh rejects and suspends it.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, "..", "public", "sample-recording.jsonl");

let seed = 20261122;
function rnd() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const hex = (n) => Array.from({ length: n }, () => Math.floor(rnd() * 16).toString(16)).join("");
const r3 = (x) => Math.round(x * 1000) / 1000;

const RUN = "sample-fixture-0002";
let seq = 0;
let t = 1795356000.0;
const out = [];
function emit(type, fields, dt = 0.25) {
  t = Math.round((t + dt) * 1000) / 1000;
  out.push({ seq: ++seq, t, run_id: RUN, type, ...fields });
}

const devices = [
  { id: "phone-01", kind: "virtual", label: "Phone 1", subject: 2, network: "wifi", battery: 88, dbm: -55 },
  { id: "phone-02", kind: "virtual", label: "Phone 2", subject: 4, network: "4g", battery: 71, dbm: -76 },
  { id: "phone-03", kind: "virtual", label: "Phone 3", subject: 6, network: "wifi", battery: 93, dbm: -58 },
  { id: "phone-04", kind: "virtual", label: "Phone 4", subject: 9, network: "3g", battery: 19, dbm: -84 },
  { id: "phone-05", kind: "virtual", label: "Phone 5", subject: 11, network: "4g", battery: 80, dbm: -66 },
  { id: "phone-06", kind: "virtual", label: "Phone 6", subject: 14, network: "wifi", battery: 65, dbm: -61 },
  { id: "esp32-01", kind: "wokwi", label: "ESP32 board 1", subject: 17, network: "wokwi-guest", battery: 100, dbm: -62 },
  { id: "esp32-02", kind: "wokwi", label: "ESP32 board 2", subject: 19, network: "wokwi-guest", battery: 100, dbm: -64 },
  { id: "esp32-03", kind: "wokwi", label: "ESP32 board 3", subject: 23, network: "wokwi-guest", battery: 100, dbm: -63 },
];
const byId = Object.fromEntries(devices.map((d) => [d.id, d]));
const battery = Object.fromEntries(devices.map((d) => [d.id, d.battery]));
const lastLoss = Object.fromEntries(devices.map((d) => [d.id, 1.55 + rnd() * 0.25]));
const updates = {}; // `${dev}:${round}` -> update_id
const PASS = { untampered: "pass", sender: "pass", shape: "pass", replay: "pass", normal: "pass" };

const RULE_APPROVE = { decision: "approve", rule_id: "100201", rule_level: 3, rule_description: "FinSentinel: update passed every check (LOW risk) — approve", event_type: "security.update_validated" };

function status(d, state, round, extra = {}) {
  emit(
    "device_status",
    {
      device: d.id,
      state,
      round,
      battery: battery[d.id],
      signal_dbm: d.dbm + Math.round((rnd() - 0.5) * 4),
      network: d.network,
      online: state !== "offline",
      hacked: !!extra.hacked,
      reason: extra.reason ?? null,
      detail: extra.detail ?? null,
    },
    extra.dt ?? 0.15 + rnd() * 0.3,
  );
}

function classCounts(skew) {
  if (skew) return [11, 9, 8, 104, 117, 109];
  return Array.from({ length: 6 }, () => 48 + Math.floor(rnd() * 18));
}
function weightDelta(strength) {
  return Array.from({ length: 8 }, () =>
    Array.from({ length: 16 }, () => r3(Math.min(1, Math.pow(rnd(), strength > 1 ? 0.4 : 2.2) * (strength > 1 ? 1 : 0.85)))),
  );
}

/** Wazuh decides on an update (and the coordinator enforces it as an `approval`). */
function wazuhDecide(device, round, update_id, rule, { enforced = true, enforcement_detail, latency } = {}) {
  emit(
    "wazuh_decision",
    {
      device,
      update_id,
      round,
      decision: rule.decision,
      rule_id: rule.rule_id,
      rule_level: rule.rule_level,
      rule_description: rule.rule_description,
      event_type: rule.event_type,
      enforced,
      enforcement_detail:
        enforcement_detail ??
        (rule.decision === "approve" ? "approved via FinSentinel operator API" : `${rule.decision} via FinSentinel operator API`),
      latency_ms: latency ?? Math.round(700 + rnd() * 1400),
    },
    0.6 + rnd() * 0.6,
  );
  const map = { approve: "approved", quarantine: "quarantined", reject: "rejected" };
  if (enforced && map[rule.decision] && update_id) {
    emit(
      "approval",
      { device, round, update_id, decision: map[rule.decision], actor: `wazuh rule ${rule.rule_id} (level ${rule.rule_level})`, reason: rule.rule_description },
      0.1,
    );
  }
}

// One device's honest/poisoned contribution for a round. Returns a function that emits Wazuh's
// decision, so callers can defer some decisions into the "deciding" phase.
function contribute(d, round, opts = {}) {
  const cc = classCounts(opts.skew);
  const samples = cc.reduce((a, b) => a + b, 0);
  const loss = r3(lastLoss[d.id] * (0.62 + rnd() * 0.08));
  lastLoss[d.id] = loss;
  battery[d.id] = Math.max(0, battery[d.id] - (d.kind === "wokwi" ? 0 : 2));
  status(d, "training", round, { hacked: opts.hacked, detail: "epoch 1/1", dt: 0.4 + rnd() * 0.5 });
  emit(
    "local_training",
    {
      device: d.id,
      round,
      epochs: 1,
      losses: [loss],
      samples,
      class_counts: cc,
      weight_delta: weightDelta(opts.hacked ? 8 : 1),
      delta_l2: r3(opts.hacked ? 3.1 + rnd() * 0.6 : 0.28 + rnd() * 0.2),
      seconds: r3(d.kind === "wokwi" ? 4.2 + rnd() * 1.4 : 0.6 + rnd() * 0.5),
    },
    d.kind === "wokwi" ? 1.2 : 0.6,
  );
  status(d, "signing", round, { hacked: opts.hacked, dt: 0.2 });
  status(d, "sending", round, { hacked: opts.hacked, dt: 0.2 });
  const update_id = `${d.id}-r${String(round).padStart(4, "0")}-${hex(8)}`;
  updates[`${d.id}:${round}`] = update_id;
  emit(
    "update_submitted",
    { device: d.id, round, update_id, payload_bytes: 9688, payload_sha256: hex(64), signed: true, path: d.kind === "wokwi" ? "tunnel" : "toxiproxy" },
    0.15,
  );
  const g = opts.gate ?? {
    http_status: 200,
    checks: PASS,
    risk_score: Math.floor(rnd() * 8),
    severity: "LOW",
    approval_status: "pending_approval",
    reasons: [],
    detail: "Security validation passed; LOW risk. Approval is still required.",
  };
  emit("gate_result", { device: d.id, round, update_id, ...g, source: "device", decided_by: "pending" }, 0.9 + rnd() * 0.5);
  status(d, "done", round, { hacked: opts.hacked, dt: 0.2 });
  const rule = opts.rule ?? RULE_APPROVE;
  return () => wazuhDecide(d.id, round, update_id, rule, opts.wazuh ?? {});
}

function evalRound(round, acc) {
  emit(
    "global_eval",
    {
      round,
      accuracy: acc,
      loss: r3(1.9 - acc * 1.5),
      per_device_accuracy: Object.fromEntries(devices.map((d) => [d.id, r3(acc - 0.06 + rnd() * 0.12)])),
    },
    0.8,
  );
}

function startRound(round) {
  emit("round_started", { round, deadline: r3(t + 40), model_sha256: hex(64) }, 1.5);
  emit("round_phase", { round, phase: "open" }, 0.05);
}

/** Run contributions; decisions for `deferIds` wait until the "deciding" phase. */
function runContributions(round, plan, deferIds) {
  const deferred = [];
  for (const [id, opts] of plan) {
    const decide = contribute(byId[id], round, opts);
    if (opts.after) opts.after();
    if (deferIds.includes(id)) deferred.push(decide);
    else decide();
    if (opts.afterDecide && !deferIds.includes(id)) opts.afterDecide();
  }
  return deferred;
}

function decidingPhase(round, waiting, deferred) {
  emit("round_phase", { round, phase: "deciding", waiting_for: waiting }, 0.4);
  for (const d of deferred) d();
  emit("round_phase", { round, phase: "aggregating", waiting_for: [] }, 0.5);
}

// ---------------- prelude ----------------
emit("run_started", {
  model_version: "har-mlp-v1",
  param_count: 2278,
  rounds: 4,
  round_seconds: 40,
  devices: devices.map(({ dbm, ...d }) => d),
  test_samples: 2947,
  mode: "live",
}, 0);
emit("global_eval", { round: 0, accuracy: 0.219, loss: 1.763, per_device_accuracy: {} }, 0.05);
for (const [name, detail] of [
  ["finsentinel", "127.0.0.1:8000"],
  ["coordinator", "127.0.0.1:8200"],
  ["tls-proxy", "https://127.0.0.1:8443"],
  ["toxiproxy", "6 device links"],
  ["wazuh", "manager + active response"],
]) emit("component", { name, status: "up", detail }, 0.3);
emit("decision_engine", {
  engine: "wazuh",
  detail: "FinSentinel checks and reports; Wazuh rules decide; the coordinator enforces through FinSentinel's operator API",
  wazuh_dashboard: "https://localhost",
}, 0.2);
emit("tunnel", { url: "https://sample-fixture.trycloudflare.com", status: "up", beacon_published: true }, 0.6);
emit("alert", { severity: "info", title: "Fleet coming online", detail: "9 devices registered with signing keys", source: "coordinator" }, 0.3);
for (const d of devices) status(d, "online", undefined, { dt: 0.35 });

// ---------------- round 1: everyone learns together ----------------
startRound(1);
for (const d of devices) status(d, "downloading", 1, { dt: 0.12 });
{
  const deferred = runContributions(1, devices.map((d) => [d.id, {}]), ["esp32-02", "esp32-03"]);
  decidingPhase(1, ["esp32-02", "esp32-03"], deferred);
}
emit("round_completed", { round: 1, participants: devices.map((d) => d.id), dropped: [], aggregated: true, model_sha256: hex(64) }, 1.0);
emit("alert", { severity: "info", title: "Round 1 aggregated", detail: "9 approved updates averaged", source: "coordinator" }, 0.2);
evalRound(1, 0.612);

// ---------------- round 2: low battery, poisoned device, medium risk, replay ----------------
battery["phone-04"] = 12;
emit("attack", { scene: "poison", phase: "launched", target: "phone-03", detail: "phone-03 is compromised: its next update is boosted ×8 (still signed with its real key)" }, 1.2);
startRound(2);
const r2 = ["phone-01", "phone-02", "phone-03", "phone-05", "phone-06", "esp32-01", "esp32-02", "esp32-03"];
for (const id of r2) status(byId[id], "downloading", 2, { hacked: id === "phone-03", dt: 0.12 });
status(byId["phone-04"], "skipped", 2, { reason: "low_battery", detail: "battery 12% < 15%" });
emit("alert", { severity: "low", title: "Phone 4 sat out this round", detail: "Battery 12% is below the 15% limit", device: "phone-04", source: "coordinator" }, 0.2);
{
  const plan = r2.map((id) => {
    if (id === "phone-03")
      return [id, {
        hacked: true,
        gate: {
          http_status: 200,
          checks: { ...PASS, normal: "fail" },
          risk_score: 75,
          severity: "CRITICAL",
          approval_status: "pending_approval",
          reasons: [
            "Weight L2 norm is unusually high: observed 49.63 exceeds configured threshold 11.61.",
            "Mean absolute weight is unusually high: observed 0.745 exceeds configured threshold 0.155.",
          ],
          detail: "Security validation passed; ANOMALY FLAGGED (score=75, CRITICAL). Reported to the SIEM.",
        },
        rule: { decision: "quarantine", rule_id: "100211", rule_level: 12, rule_description: "FinSentinel: anomalous update (CRITICAL risk) — quarantine", event_type: "security.anomaly_detected" },
        wazuh: { latency: 1240 },
        after: () =>
          emit("alert", { severity: "high", title: "Anomalous update reported", detail: "Phone 3's weights are far outside the calibrated range", device: id, source: "finsentinel", event_type: "security.anomaly_detected" }, 0.2),
      }];
    if (id === "esp32-02")
      return [id, {
        gate: {
          http_status: 200,
          checks: { ...PASS, normal: "warn" },
          risk_score: 40,
          severity: "MEDIUM",
          approval_status: "pending_approval",
          reasons: ["Loss improvement unusually small for this client."],
          detail: "Security validation passed; risk_score=40 (MEDIUM). Approval is required.",
        },
        rule: { decision: "alert", rule_id: "100205", rule_level: 7, rule_description: "FinSentinel: medium-risk update — operator review", event_type: "security.approval_required" },
        wazuh: { enforced: false, enforcement_detail: "left pending for a human decision" },
      }];
    return [id, { skew: id === "phone-05" }];
  });
  const deferred = runContributions(2, plan, ["phone-03", "esp32-03"]);
  // Replay attack: resend Phone 1's captured round-1 body (announced like the real coordinator does).
  emit("attack", { scene: "replay", phase: "launched", target: "phone-01", detail: "Resending Phone 1's captured round-1 request" }, 0.8);
  const replayId = `${updates["phone-01:1"]}#replay`;
  emit("update_submitted", { device: "phone-01", round: 1, update_id: replayId, payload_bytes: 9688, payload_sha256: hex(64), signed: true, path: "toxiproxy" }, 0.3);
  emit("gate_result", {
    device: "phone-01", round: 1, update_id: replayId, http_status: 409,
    checks: { untampered: "pass", sender: "pass", shape: "pass", replay: "fail", normal: "skip" },
    risk_score: null, severity: null, approval_status: null,
    reasons: ["update_id already received"], detail: "Duplicate update rejected", source: "attack",
  }, 0.7);
  emit("attack", { scene: "replay", phase: "result", target: "phone-01", http_status: 409, outcome: "blocked", detail: "FinSentinel refused the duplicate (409 Conflict)" }, 0.15);
  emit("alert", { severity: "high", title: "Replay attempt blocked", detail: "Same update_id seen before - 409", device: "phone-01", source: "finsentinel", event_type: "security.replay_detected" }, 0.15);
  wazuhDecide("phone-01", 1, replayId,
    { decision: "reject", rule_id: "100220", rule_level: 10, rule_description: "FinSentinel: replay detected — reject", event_type: "security.replay_detected" },
    { enforced: false, enforcement_detail: "already refused by FinSentinel (HTTP 409); logged" });
  emit("alert", { severity: "high", title: "Wazuh: replay attack on the FL server", detail: "rule 100220 (level 10) fired on security.replay_detected", device: "phone-01", source: "wazuh", event_type: "security.replay_detected" }, 0.2);
  emit("drift_monitor", { device: "phone-05", round: 2, score: 3.42, threshold: 2.5, flagged: true, finsentinel_flagged: false }, 0.6);
  emit("drift_monitor", { device: "phone-01", round: 2, score: 0.61, threshold: 2.5, flagged: false, finsentinel_flagged: false }, 0.1);
  decidingPhase(2, ["phone-03", "esp32-03"], deferred);
}
emit("round_completed", {
  round: 2,
  participants: ["phone-01", "phone-02", "phone-05", "phone-06", "esp32-01", "esp32-03"],
  dropped: [
    { device: "phone-04", reason: "low_battery" },
    { device: "phone-03", reason: "quarantined" },
    { device: "esp32-02", reason: "pending_review" },
  ],
  aggregated: true,
  model_sha256: hex(64),
}, 1.0);
emit("alert", { severity: "info", title: "Round 2 aggregated", detail: "6 approved updates averaged; 1 quarantined, 1 waiting", source: "coordinator" }, 0.2);
evalRound(2, 0.704);

// ---------------- round 3: tamper, drift dropout, human override ----------------
startRound(3);
const r3ids = ["phone-01", "phone-02", "phone-03", "phone-06", "esp32-01", "esp32-02", "esp32-03"];
for (const id of r3ids) status(byId[id], "downloading", 3, { dt: 0.12 });
status(byId["phone-04"], "skipped", 3, { reason: "low_battery", detail: "battery 11% < 15%" });
status(byId["phone-05"], "skipped", 3, { reason: "drift", detail: "local data drifted from the fleet (research rule)" });
emit("alert", { severity: "low", title: "Phone 5 sat out: data drift", detail: "Drift-causal dropout - the research mechanism", device: "phone-05", source: "coordinator" }, 0.2);
emit("approval", { device: "esp32-02", round: 2, update_id: updates["esp32-02:2"], decision: "approved", actor: "presenter (Mission Control)", reason: "manual override: board warm-up, benign" }, 0.5);
emit("attack", { scene: "tamper", phase: "launched", target: "phone-02", detail: "Flipping bytes in Phone 2's payload in transit" }, 0.4);
const tamperId = `phone-02-r0003-${hex(8)}#tamper`;
emit("update_submitted", { device: "phone-02", round: 3, update_id: tamperId, payload_bytes: 9688, payload_sha256: hex(64), signed: true, path: "toxiproxy" }, 0.3);
emit("gate_result", {
  device: "phone-02", round: 3, update_id: tamperId, http_status: 422,
  checks: { untampered: "fail", sender: "skip", shape: "skip", replay: "skip", normal: "skip" },
  risk_score: null, severity: null, approval_status: null,
  reasons: ["payload SHA-256 does not match envelope"], detail: "Payload hash mismatch", source: "attack",
}, 0.7);
emit("attack", { scene: "tamper", phase: "result", target: "phone-02", http_status: 422, outcome: "blocked", detail: "Hash mismatch - rejected before anything else ran (422)" }, 0.15);
emit("alert", { severity: "high", title: "Tampered payload blocked", detail: "SHA-256 mismatch - 422", device: "phone-02", source: "finsentinel", event_type: "security.payload_integrity_failed" }, 0.15);
wazuhDecide("phone-02", 3, tamperId,
  { decision: "alert", rule_id: "100230", rule_level: 10, rule_description: "FinSentinel: payload integrity failure — alert", event_type: "security.payload_integrity_failed" },
  { enforced: false, enforcement_detail: "already refused by FinSentinel (HTTP 422); logged" });
{
  const deferred = runContributions(3, r3ids.map((id) => [id, {}]), ["esp32-01"]);
  decidingPhase(3, ["esp32-01"], deferred);
}
emit("round_completed", {
  round: 3,
  participants: r3ids,
  dropped: [
    { device: "phone-04", reason: "low_battery" },
    { device: "phone-05", reason: "drift" },
  ],
  aggregated: true,
  model_sha256: hex(64),
}, 1.0);
evalRound(3, 0.768);

// ---------------- round 4: repeat offender → reject + suspend ----------------
emit("attack", { scene: "poison", phase: "launched", target: "phone-03", detail: "phone-03 is compromised again: next update boosted ×8" }, 1.0);
startRound(4);
const r4ids = ["phone-01", "phone-02", "phone-03", "phone-06", "esp32-01", "esp32-02", "esp32-03"];
for (const id of r4ids) status(byId[id], "downloading", 4, { hacked: id === "phone-03", dt: 0.12 });
status(byId["phone-04"], "skipped", 4, { reason: "low_battery", detail: "battery 10% < 15%" });
status(byId["phone-05"], "skipped", 4, { reason: "drift", detail: "local data drifted from the fleet (research rule)" });
{
  const plan = r4ids.map((id) => {
    if (id !== "phone-03") return [id, {}];
    return [id, {
      hacked: true,
      gate: {
        http_status: 200,
        checks: { ...PASS, normal: "fail" },
        risk_score: 80,
        severity: "CRITICAL",
        approval_status: "pending_approval",
        reasons: ["Weight L2 norm is unusually high: observed 51.20 exceeds configured threshold 11.61."],
        detail: "Security validation passed; ANOMALY FLAGGED (score=80, CRITICAL). Reported to the SIEM.",
      },
      rule: { decision: "reject", rule_id: "100212", rule_level: 13, rule_description: "FinSentinel: anomalous update from a device already quarantined — reject", event_type: "security.anomaly_detected" },
      wazuh: { latency: 980 },
      afterDecide: () => {
        wazuhDecide("phone-03", 4, updates["phone-03:4"],
          { decision: "suspend_device", rule_id: "100213", rule_level: 14, rule_description: "Repeat offender: 2 anomalous updates in 3 rounds — suspend device", event_type: "security.anomaly_detected" },
          { enforcement_detail: "phone-03 suspended; its next updates are refused", latency: 1105 });
        status(byId["phone-03"], "suspended", 4, { hacked: true, reason: "100213", detail: "suspended by Wazuh rule 100213" });
        emit("alert", { severity: "critical", title: "Wazuh suspended Phone 3", detail: "rule 100213 (level 14): repeat offender", device: "phone-03", source: "wazuh", event_type: "security.anomaly_detected" }, 0.2);
      },
    }];
  });
  const deferred = runContributions(4, plan, ["esp32-03"]);
  decidingPhase(4, ["esp32-03"], deferred);
}
emit("round_completed", {
  round: 4,
  participants: ["phone-01", "phone-02", "phone-06", "esp32-01", "esp32-02", "esp32-03"],
  dropped: [
    { device: "phone-03", reason: "rejected" },
    { device: "phone-04", reason: "low_battery" },
    { device: "phone-05", reason: "drift" },
  ],
  aggregated: true,
  model_sha256: hex(64),
}, 1.0);
evalRound(4, 0.791);
emit("run_finished", { rounds_completed: 4, final_accuracy: 0.791 }, 1.0);

writeFileSync(OUT, out.map((e) => JSON.stringify(e)).join("\n") + "\n", "utf8");
console.log(`wrote ${out.length} events to ${OUT}`);
