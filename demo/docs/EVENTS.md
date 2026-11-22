# Live event stream (contract between coordinator and Mission Control)

Source: coordinator, `ws://127.0.0.1:8200/fl/ws` (on connect it sends every event of the current
run from seq 1, then live events) and `GET http://127.0.0.1:8200/fl/events?since=<seq>` (JSON
array, polling fallback). Every event is also appended as one JSON line to
`demo/recordings/<run_id>.jsonl` — a recording is exactly this stream, replayable offline using
the `t` field for timing.

Every event: `{"seq": int (1..), "t": float unix seconds, "run_id": str, "type": str, ...fields}`.
Only real system events are emitted — the UI must never invent outcomes.

```ts
type DeviceKind = "virtual" | "wokwi";
type CheckState = "pass" | "fail" | "warn" | "skip";   // skip = not reached
// Gate lights in FinSentinel's REAL pipeline order:
type GateChecks = { untampered: CheckState; sender: CheckState; shape: CheckState;
                    replay: CheckState; normal: CheckState };

interface DeviceInfo { id: string; kind: DeviceKind; label: string;   // "Phone 1", "ESP32 board 1"
  subject: number; network: "wifi" | "4g" | "3g" | "wokwi-guest"; battery: number }

type Ev =
 | { type: "run_started"; model_version: string; param_count: number; rounds: number;
     round_seconds: number; devices: DeviceInfo[]; test_samples: number; mode: "live" }
 | { type: "component"; name: string; status: "up" | "down"; detail?: string }
 | { type: "tunnel"; url: string | null; status: "up" | "down"; beacon_published: boolean }
 | { type: "round_started"; round: number; deadline: number; model_sha256: string }
 | { type: "device_status"; device: string; state: string;           // see PROTOCOL.md §4
     round?: number; battery?: number; signal_dbm?: number; network?: string;
     online?: boolean; hacked?: boolean; reason?: string | null; detail?: string | null }
 | { type: "local_training"; device: string; round: number; epochs: number; losses: number[];
     samples: number; class_counts: number[];        // data sample for the "inside" panel
     weight_delta: number[][];                       // 8×16 grid, |Δw| of fc1 vs global, 0..1 normalised
     delta_l2: number; seconds: number }
 | { type: "update_submitted"; device: string; round: number; update_id: string;
     payload_bytes: number; payload_sha256: string; signed: true; path: "toxiproxy" | "tunnel" }
 | { type: "gate_result"; device: string; round: number; update_id: string; http_status: number;
     checks: GateChecks; risk_score: number | null; severity: string | null;
     approval_status: string | null;                 // pending_approval | quarantined | approved | rejected
     reasons: string[]; detail: string; source: "device" | "attack" }
 | { type: "approval"; device: string; round: number; update_id: string;
     decision: "approved" | "rejected" | "quarantined"; actor: string; reason: string }
 | { type: "drift_monitor"; device: string; round: number; score: number; threshold: number;
     flagged: boolean; finsentinel_flagged: boolean }  // coordinator research layer, labelled as such
 | { type: "round_completed"; round: number; participants: string[];
     dropped: { device: string; reason: string }[]; aggregated: boolean; model_sha256: string }
 | { type: "global_eval"; round: number; accuracy: number; loss: number;
     per_device_accuracy: Record<string, number> }
 | { type: "alert"; severity: "info" | "low" | "medium" | "high" | "critical"; title: string;
     detail: string; device?: string; source: "finsentinel" | "coordinator" | "attack";
     event_type?: string }
 | { type: "attack"; scene: string; phase: "launched" | "result"; target?: string;
     http_status?: number; outcome?: "blocked" | "quarantined" | "refused" | "accepted" | "observed";
     detail: string }
 | { type: "run_finished"; rounds_completed: number; final_accuracy: number | null };
```

Attack scenes (`POST /fl/admin/attack/<scene>` with JSON `{"target"?: device}` and header
`Authorization: Bearer <admin_token>`): `poison`, `replay`, `tamper`, `forge`, `mitm`, `drift`,
`weak_signal`, `restore` (clears weak_signal/drift), `low_battery`, `offline`.
Approve/reject a waiting update: `POST /fl/admin/decision` `{"client_id","update_id","decision":"approve"|"reject"}`.
`GET /fl/state` returns `{"run_id", "round", "state", "admin": false}` for health checks.

Mission Control reads `/config.json` from its own origin:
```json
{"mode": "live" | "replay", "ws_url": "ws://127.0.0.1:8200/fl/ws",
 "http_url": "http://127.0.0.1:8200", "admin_token": "...", "replay_url": "/recordings/golden.jsonl",
 "finsentinel_ui": "http://localhost:5173", "femnist_url": "/femnist.json"}
```
`/femnist.json` (optional, research slide for scene 7): `{"title": str, "note": str,
"series": [{"label": str, "final_accuracy": number, "drift_cohort_accuracy"?: number}],
"source": str}`.

## Notes from integration (2026-11-22)

* Attack traffic (replay / tamper / forge) is emitted as its own `update_submitted` + `gate_result`
  with `update_id` suffixed `#replay`, `#tamper`, `#forge` and `source: "attack"`; the honest
  device's records are never overwritten.
* `mitm` produces two `attack` results: `outcome: "observed"` (tap saw only ciphertext) and then
  `outcome: "refused"` (impostor certificate rejected).
* Only replay, tamper, forge and mitm are drawn as red attacker envelopes. Poison travels as the
  hacked device's own signed update; drift / weak_signal / low_battery / offline / restore never
  reach the gate.
* `global_eval` starts at `round: 0` (the initial model). Accuracy is a 0–1 fraction.
* `device_status.network` may be `"weak"` while the weak-signal scene is active.
* An `approval` can arrive before its `gate_result`; consumers must pair by `update_id`.
* There is no separate "aggregating" event: a round is aggregating between the end of its
  submissions and its `round_completed`.
* The coordinator allows CORS from `http://localhost:5180` and `http://127.0.0.1:5180`.

---

## v2 — Wazuh decides, one unified UI (2026-11-22)

**Decision engine.** With `run_demo.py --wazuh` (default when Wazuh is up) FinSentinel only checks
and reports (`FINSENTINEL_HIGH_RISK_ACTION=pending_approval`, the coordinator never auto-approves).
FinSentinel's audit events → SIEM log → Wazuh rules → Wazuh **active response** writes a decision
→ the coordinator enforces it through FinSentinel's operator API. Without Wazuh the engine is
`coordinator` (v1 behaviour) and the UI must say so.

New / extended events:
```ts
 | { type: "decision_engine"; engine: "wazuh" | "coordinator"; detail: string;
     wazuh_dashboard?: string }                       // emitted once after run_started
 | { type: "round_phase"; round: number; phase: "open" | "deciding" | "aggregating";
     waiting_for?: string[] }                         // devices whose decision is still pending
 | { type: "wazuh_decision"; device: string; update_id: string | null; round: number | null;
     decision: "approve" | "quarantine" | "reject" | "suspend_device" | "alert";
     rule_id: string; rule_level: number; rule_description: string; event_type: string;
     enforced: boolean; enforcement_detail: string; latency_ms: number | null }
// gate_result gains:   decided_by: "wazuh" | "coordinator" | "human" | "pending"
// device_status.state may be "suspended" (Wazuh suspended the device; reason = rule id)
// alert.source may be "wazuh"; component.name may be "wazuh"
```
`approval.actor` is e.g. `"wazuh rule 100201 (level 3)"`.

**Security Console API** (coordinator, local only, `Authorization: Bearer <admin_token>`):
- `GET /fl/admin/console` →
  ```json
  {"decision_engine": "wazuh", "wazuh": {"up": true, "decisions": 41, "last_decision_t": 1795.., "dashboard_url": "https://localhost"},
   "finsentinel": {<FinSentinel GET /api/v1/operator/dashboard JSON, verbatim>},
   "pending": [<FinSentinel GET /api/v1/updates/pending items, verbatim>],
   "suspended": [{"device": "phone-03", "rule_id": "100211", "since_round": 7}],
   "thresholds": {<config/thresholds.json finsentinel_env>},
   "updates": [{"update_id","device","round","http_status","finsentinel_status","risk_score","severity",
                "reasons","decision","decided_by","rule_id","t"}]}   // newest first, ≤ 300
  ```
- `POST /fl/admin/decision` (unchanged) = manual override, actor "presenter (Mission Control)".
- `POST /fl/admin/unsuspend` `{"device"}`.
