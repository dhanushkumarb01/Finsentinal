// Event-stream contract — mirrors demo/docs/EVENTS.md exactly. Do not extend silently.

export type DeviceKind = "virtual" | "wokwi";
export type CheckState = "pass" | "fail" | "warn" | "skip";

export interface GateChecks {
  untampered: CheckState;
  sender: CheckState;
  shape: CheckState;
  replay: CheckState;
  normal: CheckState;
}

export const GATE_ORDER: (keyof GateChecks)[] = ["untampered", "sender", "shape", "replay", "normal"];

export interface DeviceInfo {
  id: string;
  kind: DeviceKind;
  label: string;
  subject: number;
  network: "wifi" | "4g" | "3g" | "wokwi-guest";
  battery: number;
}

export interface EvBase {
  seq: number;
  t: number;
  run_id: string;
}

export interface RunStartedEv extends EvBase {
  type: "run_started";
  model_version: string;
  param_count: number;
  rounds: number;
  round_seconds: number;
  devices: DeviceInfo[];
  test_samples: number;
  mode: "live";
}
export interface ComponentEv extends EvBase {
  type: "component";
  name: string;
  status: "up" | "down";
  detail?: string;
}
export interface TunnelEv extends EvBase {
  type: "tunnel";
  url: string | null;
  status: "up" | "down";
  beacon_published: boolean;
}
export interface RoundStartedEv extends EvBase {
  type: "round_started";
  round: number;
  deadline: number;
  model_sha256: string;
}
export interface DeviceStatusEv extends EvBase {
  type: "device_status";
  device: string;
  state: string;
  round?: number;
  battery?: number;
  signal_dbm?: number;
  network?: string;
  online?: boolean;
  hacked?: boolean;
  reason?: string | null;
  detail?: string | null;
}
export interface LocalTrainingEv extends EvBase {
  type: "local_training";
  device: string;
  round: number;
  epochs: number;
  losses: number[];
  samples: number;
  class_counts: number[];
  weight_delta: number[][];
  delta_l2: number;
  seconds: number;
}
export interface UpdateSubmittedEv extends EvBase {
  type: "update_submitted";
  device: string;
  round: number;
  update_id: string;
  payload_bytes: number;
  payload_sha256: string;
  signed: true;
  path: "toxiproxy" | "tunnel";
}
export interface GateResultEv extends EvBase {
  type: "gate_result";
  device: string;
  round: number;
  update_id: string;
  http_status: number;
  checks: GateChecks;
  risk_score: number | null;
  severity: string | null;
  approval_status: string | null;
  reasons: string[];
  detail: string;
  source: "device" | "attack";
  /** v2: who makes / made the decision. Absent in v1 recordings. */
  decided_by?: DecidedBy;
}
export type DecidedBy = "wazuh" | "coordinator" | "human" | "pending";
export interface ApprovalEv extends EvBase {
  type: "approval";
  device: string;
  round: number;
  update_id: string;
  decision: "approved" | "rejected" | "quarantined";
  actor: string;
  reason: string;
}
export interface DriftMonitorEv extends EvBase {
  type: "drift_monitor";
  device: string;
  round: number;
  score: number;
  threshold: number;
  flagged: boolean;
  finsentinel_flagged: boolean;
}
export interface RoundCompletedEv extends EvBase {
  type: "round_completed";
  round: number;
  participants: string[];
  dropped: { device: string; reason: string }[];
  aggregated: boolean;
  model_sha256: string;
}
export interface GlobalEvalEv extends EvBase {
  type: "global_eval";
  round: number;
  accuracy: number;
  loss: number;
  per_device_accuracy: Record<string, number>;
}
export interface AlertEv extends EvBase {
  type: "alert";
  severity: "info" | "low" | "medium" | "high" | "critical";
  title: string;
  detail: string;
  device?: string;
  source: "finsentinel" | "coordinator" | "attack" | "wazuh";
  event_type?: string;
}
export interface AttackEv extends EvBase {
  type: "attack";
  scene: string;
  phase: "launched" | "result";
  target?: string;
  http_status?: number;
  outcome?: "blocked" | "quarantined" | "refused" | "accepted" | "observed";
  detail: string;
}
export interface RunFinishedEv extends EvBase {
  type: "run_finished";
  rounds_completed: number;
  final_accuracy: number | null;
}

// ---------- v2: Wazuh decides ----------
export interface DecisionEngineEv extends EvBase {
  type: "decision_engine";
  engine: "wazuh" | "coordinator";
  detail: string;
  wazuh_dashboard?: string;
}
export type RoundPhaseName = "open" | "deciding" | "aggregating";
export interface RoundPhaseEv extends EvBase {
  type: "round_phase";
  round: number;
  phase: RoundPhaseName;
  waiting_for?: string[];
}
export type WazuhDecisionKind = "approve" | "quarantine" | "reject" | "suspend_device" | "alert";
export interface WazuhDecisionEv extends EvBase {
  type: "wazuh_decision";
  device: string;
  update_id: string | null;
  round: number | null;
  decision: WazuhDecisionKind;
  rule_id: string;
  rule_level: number;
  rule_description: string;
  event_type: string;
  enforced: boolean;
  enforcement_detail: string;
  latency_ms: number | null;
}

export type Ev =
  | RunStartedEv
  | ComponentEv
  | TunnelEv
  | RoundStartedEv
  | DeviceStatusEv
  | LocalTrainingEv
  | UpdateSubmittedEv
  | GateResultEv
  | ApprovalEv
  | DriftMonitorEv
  | RoundCompletedEv
  | GlobalEvalEv
  | AlertEv
  | AttackEv
  | RunFinishedEv
  | DecisionEngineEv
  | RoundPhaseEv
  | WazuhDecisionEv;

export interface AppConfig {
  mode: "live" | "replay";
  ws_url: string;
  http_url: string;
  admin_token: string;
  replay_url: string;
  finsentinel_ui: string;
  femnist_url: string;
}

// ---------- v2: Security Console REST API (GET /fl/admin/console) ----------
export interface ConsoleUpdateRow {
  update_id: string;
  device: string;
  round: number | null;
  http_status: number | null;
  finsentinel_status: string | null;
  risk_score: number | null;
  severity: string | null;
  reasons: string[];
  decision: string | null;
  decided_by: string | null;
  rule_id: string | null;
  t: number | null;
}
export interface ConsolePending {
  update_id: string;
  client_id: string;
  round_number: number | null;
  risk_score: number | null;
  risk_severity: string | null;
  anomaly_reasons: string[];
  status: string | null;
  raw: Record<string, unknown>;
}
export interface ConsoleApi {
  decision_engine: "wazuh" | "coordinator" | string;
  wazuh: { up: boolean; decisions: number | null; last_decision_t: number | null; dashboard_url: string | null } | null;
  finsentinel: Record<string, unknown> | null;
  pending: ConsolePending[];
  suspended: { device: string; rule_id: string | null; since_round: number | null }[];
  thresholds: Record<string, unknown> | null;
  updates: ConsoleUpdateRow[];
}

export interface FemnistData {
  title: string;
  note: string;
  series: { label: string; final_accuracy: number; drift_cohort_accuracy?: number }[];
  source: string;
}

export const ACTIVITY_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"];
