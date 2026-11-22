// Generates public/sample-recording.jsonl — a deterministic, EVENTS.md-conformant FIXTURE used
// for tests and as the dev default replay. It is NOT a recording of a real run; the real demo
// replays demo/recordings/<run_id>.jsonl produced by the coordinator.
//   node scripts/gen-sample.mjs
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

const RUN = "sample-fixture-0001";
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
const battery = Object.fromEntries(devices.map((d) => [d.id, d.battery]));
const lastLoss = Object.fromEntries(devices.map((d) => [d.id, 1.55 + rnd() * 0.25]));
const updates = {}; // `${dev}:${round}` -> update_id
const PASS = { untampered: "pass", sender: "pass", shape: "pass", replay: "pass", normal: "pass" };

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

// One device's honest/poisoned contribution for a round. Returns the gate fields.
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
    {
      device: d.id,
      round,
      update_id,
      payload_bytes: 9688,
      payload_sha256: hex(64),
      signed: true,
      path: d.kind === "wokwi" ? "tunnel" : "toxiproxy",
    },
    0.15,
  );
  const g = opts.gate ?? {
    http_status: 202,
    checks: PASS,
    risk_score: r3(0.04 + rnd() * 0.1),
    severity: "LOW",
    approval_status: "pending_approval",
    reasons: [],
    detail: "Accepted for review: all checks passed",
  };
  emit("gate_result", { device: d.id, round, update_id, ...g, source: "device" }, 0.9 + rnd() * 0.5);
  if (g.severity === "LOW" && g.approval_status === "pending_approval") {
    emit(
      "approval",
      { device: d.id, round, update_id, decision: "approved", actor: "demo-coordinator (assisted policy)", reason: "clean LOW-risk update" },
      0.2,
    );
  }
  status(d, "done", round, { hacked: opts.hacked, dt: 0.2 });
  return update_id;
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

// ---------------- prelude ----------------
emit("run_started", {
  model_version: "har-mlp-v1",
  param_count: 2278,
  rounds: 3,
  round_seconds: 40,
  devices: devices.map(({ dbm, ...d }) => d),
  test_samples: 2947,
  mode: "live",
}, 0);
for (const [name, detail] of [
  ["finsentinel", "127.0.0.1:8000"],
  ["coordinator", "127.0.0.1:8200"],
  ["caddy", "https://127.0.0.1:8443"],
  ["toxiproxy", "6 device links"],
]) emit("component", { name, status: "up", detail }, 0.3);
emit("tunnel", { url: "https://sample-fixture.trycloudflare.com", status: "up", beacon_published: true }, 0.6);
emit("alert", { severity: "info", title: "Fleet coming online", detail: "9 devices registered with signing keys", source: "coordinator" }, 0.3);
for (const d of devices) status(d, "online", undefined, { dt: 0.35 });

// ---------------- round 1: everyone learns together ----------------
function startRound(round) {
  emit("round_started", { round, deadline: r3(t + 40), model_sha256: hex(64) }, 1.5);
}
startRound(1);
for (const d of devices) status(d, "downloading", 1, { dt: 0.12 });
for (const d of devices) contribute(d, 1);
emit("round_completed", { round: 1, participants: devices.map((d) => d.id), dropped: [], aggregated: true, model_sha256: hex(64) }, 1.0);
emit("alert", { severity: "info", title: "Round 1 aggregated", detail: "9 approved updates averaged", source: "coordinator" }, 0.2);
evalRound(1, 0.612);

// ---------------- round 2: low battery, hacked device, review, replay ----------------
const byId = Object.fromEntries(devices.map((d) => [d.id, d]));
battery["phone-04"] = 12;
emit("attack", { scene: "poison", phase: "launched", target: "phone-03", detail: "Phone 3 will send weights x8" }, 1.2);
startRound(2);
const r2 = ["phone-01", "phone-02", "phone-03", "phone-05", "phone-06", "esp32-01", "esp32-02", "esp32-03"];
for (const id of r2) status(byId[id], "downloading", 2, { hacked: id === "phone-03", dt: 0.12 });
status(byId["phone-04"], "skipped", 2, { reason: "low_battery", detail: "battery 12% < 15%" });
emit("alert", { severity: "low", title: "Phone 4 sat out this round", detail: "Battery 12% is below the 15% limit", device: "phone-04", source: "coordinator" }, 0.2);
for (const id of r2) {
  if (id === "phone-03") {
    const uid = contribute(byId[id], 2, {
      hacked: true,
      gate: {
        http_status: 202,
        checks: { ...PASS, normal: "fail" },
        risk_score: 0.94,
        severity: "HIGH",
        approval_status: "quarantined",
        reasons: ["update norm 8.1x the fleet median", "cosine distance to fleet 0.91"],
        detail: "Anomaly detector quarantined the update",
      },
    });
    emit("approval", { device: id, round: 2, update_id: uid, decision: "quarantined", actor: "FinSentinel anomaly policy", reason: "risk 0.94 (HIGH)" }, 0.2);
    emit("alert", { severity: "high", title: "Poisoned update quarantined", detail: "Phone 3 sent weights far outside the fleet's normal range", device: id, source: "finsentinel", event_type: "update_quarantined" }, 0.2);
    emit("attack", { scene: "poison", phase: "result", target: id, http_status: 202, outcome: "quarantined", detail: "Signed and well-formed, but the anomaly detector held it" }, 0.2);
  } else if (id === "esp32-02") {
    contribute(byId[id], 2, {
      gate: {
        http_status: 202,
        checks: { ...PASS, normal: "warn" },
        risk_score: 0.47,
        severity: "MEDIUM",
        approval_status: "pending_approval",
        reasons: ["loss improvement unusually small for this client"],
        detail: "Needs a human decision",
      },
    });
    emit("alert", { severity: "medium", title: "Update waiting for human review", detail: "ESP32 board 2: risk 0.47 (MEDIUM)", device: id, source: "finsentinel", event_type: "approval_required" }, 0.2);
  } else {
    contribute(byId[id], 2, { skew: id === "phone-05" });
  }
}
// Replay attack: resend Phone 1's captured round-1 body.
emit("attack", { scene: "replay", phase: "launched", target: "phone-01", detail: "Resending Phone 1's captured round-1 request" }, 0.8);
emit("gate_result", {
  device: "phone-01", round: 1, update_id: updates["phone-01:1"], http_status: 409,
  checks: { untampered: "pass", sender: "pass", shape: "pass", replay: "fail", normal: "skip" },
  risk_score: null, severity: null, approval_status: null,
  reasons: ["update_id already received"], detail: "Duplicate update rejected", source: "attack",
}, 0.7);
emit("attack", { scene: "replay", phase: "result", target: "phone-01", http_status: 409, outcome: "blocked", detail: "FinSentinel refused the duplicate (409 Conflict)" }, 0.15);
emit("alert", { severity: "high", title: "Replay attempt blocked", detail: "Same update_id seen before - 409", device: "phone-01", source: "finsentinel", event_type: "replay_detected" }, 0.15);
emit("drift_monitor", { device: "phone-05", round: 2, score: 3.42, threshold: 2.5, flagged: true, finsentinel_flagged: false }, 0.6);
emit("drift_monitor", { device: "phone-01", round: 2, score: 0.61, threshold: 2.5, flagged: false, finsentinel_flagged: false }, 0.1);
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

// ---------------- round 3: tamper, drift dropout, human approval ----------------
startRound(3);
const r3ids = ["phone-01", "phone-02", "phone-03", "phone-06", "esp32-01", "esp32-02", "esp32-03"];
for (const id of r3ids) status(byId[id], "downloading", 3, { dt: 0.12 });
status(byId["phone-04"], "skipped", 3, { reason: "low_battery", detail: "battery 11% < 15%" });
status(byId["phone-05"], "skipped", 3, { reason: "drift", detail: "local data drifted from the fleet (research rule)" });
emit("alert", { severity: "low", title: "Phone 5 sat out: data drift", detail: "Drift-causal dropout - the research mechanism", device: "phone-05", source: "coordinator" }, 0.2);
emit("approval", { device: "esp32-02", round: 2, update_id: updates["esp32-02:2"], decision: "approved", actor: "operator (Mission Control)", reason: "reviewed: board warm-up, benign" }, 0.5);
emit("attack", { scene: "tamper", phase: "launched", target: "phone-02", detail: "Flipping bytes in Phone 2's payload in transit" }, 0.4);
// Like the real coordinator, announce the tampered request itself as an update_submitted.
const tamperId = `phone-02-r0003-${hex(8)}#tamper`;
emit("update_submitted", { device: "phone-02", round: 3, update_id: tamperId, payload_bytes: 9688, payload_sha256: hex(64), signed: true, path: "toxiproxy" }, 0.3);
emit("gate_result", {
  device: "phone-02", round: 3, update_id: tamperId, http_status: 422,
  checks: { untampered: "fail", sender: "skip", shape: "skip", replay: "skip", normal: "skip" },
  risk_score: null, severity: null, approval_status: null,
  reasons: ["payload SHA-256 does not match envelope"], detail: "Payload hash mismatch", source: "attack",
}, 0.7);
emit("attack", { scene: "tamper", phase: "result", target: "phone-02", http_status: 422, outcome: "blocked", detail: "Hash mismatch - rejected before anything else ran (422)" }, 0.15);
emit("alert", { severity: "high", title: "Tampered payload blocked", detail: "SHA-256 mismatch - 422", device: "phone-02", source: "finsentinel", event_type: "integrity_failure" }, 0.15);
for (const id of r3ids) contribute(byId[id], 3);
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
emit("run_finished", { rounds_completed: 3, final_accuracy: 0.768 }, 1.0);

writeFileSync(OUT, out.map((e) => JSON.stringify(e)).join("\n") + "\n", "utf8");
console.log(`wrote ${out.length} events to ${OUT}`);
