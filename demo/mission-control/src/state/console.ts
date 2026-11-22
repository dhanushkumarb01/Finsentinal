// Security Console data: merges the polled REST snapshot (GET /fl/admin/console, EVENTS.md v2)
// with what the live event stream already told us. Pure functions only.

import type { ConsoleApi, ConsolePending, ConsoleUpdateRow, WazuhDecisionEv } from "../types";
import type { DashState, Decider } from "./reducer";
import { awaitingDecision, deciderOf, effectiveStatus } from "./reducer";

export type DecisionKind = "approve" | "quarantine" | "reject" | "suspend" | "alert" | "pending" | "blocked";

export interface UpdateRow {
  key: string;
  updateId: string;
  device: string;
  round: number | null;
  t: number | null;
  http: number | null;
  fsStatus: string | null;
  risk: number | null;
  severity: string | null;
  reasons: string[];
  decision: DecisionKind | null;
  decisionRaw: string | null;
  decidedBy: string | null;
  ruleId: string | null;
  ruleLevel: number | null;
  ruleDescription: string | null;
  enforced: boolean | null;
  enforcementDetail: string | null;
  attack: boolean;
  from: "api" | "events" | "both";
}

export interface PendingRow {
  updateId: string;
  device: string;
  round: number | null;
  risk: number | null;
  severity: string | null;
  reasons: string[];
  status: string | null;
  wazuh: WazuhDecisionEv | null;
}

export interface SuspendedRow {
  device: string;
  ruleId: string | null;
  sinceRound: number | null;
}

export interface ConsoleView {
  apiAvailable: boolean;
  engine: "wazuh" | "coordinator" | "unknown";
  engineDetail: string | null;
  wazuhUp: boolean | null;
  decisions: number;
  lastDecisionT: number | null;
  dashboardUrl: string | null;
  finsentinel: Record<string, unknown> | null;
  thresholds: Record<string, unknown> | null;
  pending: PendingRow[];
  suspended: SuspendedRow[];
  rows: UpdateRow[];
}

// ---------- parsing (defensive: the API is a different process) ----------

const str = (v: unknown): string | null => (typeof v === "string" ? v : typeof v === "number" ? String(v) : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const obj = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);

export function parseConsole(raw: unknown): ConsoleApi | null {
  const o = obj(raw);
  if (!o) return null;
  const w = obj(o.wazuh);
  const pending: ConsolePending[] = (Array.isArray(o.pending) ? o.pending : [])
    .map(obj)
    .filter((x): x is Record<string, unknown> => !!x && typeof x.update_id === "string")
    .map((x) => ({
      update_id: x.update_id as string,
      client_id: str(x.client_id) ?? str(x.device) ?? "",
      round_number: num(x.round_number) ?? num(x.round),
      risk_score: num(x.risk_score),
      risk_severity: str(x.risk_severity) ?? str(x.severity),
      anomaly_reasons: strs(x.anomaly_reasons ?? x.reasons),
      status: str(x.status),
      raw: x,
    }));
  const updates: ConsoleUpdateRow[] = (Array.isArray(o.updates) ? o.updates : [])
    .map(obj)
    .filter((x): x is Record<string, unknown> => !!x && typeof x.update_id === "string")
    .map((x) => ({
      update_id: x.update_id as string,
      device: str(x.device) ?? str(x.client_id) ?? "",
      round: num(x.round),
      http_status: num(x.http_status),
      finsentinel_status: str(x.finsentinel_status),
      risk_score: num(x.risk_score),
      severity: str(x.severity),
      reasons: strs(x.reasons),
      decision: str(x.decision),
      decided_by: str(x.decided_by),
      rule_id: str(x.rule_id),
      t: num(x.t),
    }));
  const suspended = (Array.isArray(o.suspended) ? o.suspended : [])
    .map(obj)
    .filter((x): x is Record<string, unknown> => !!x && typeof x.device === "string")
    .map((x) => ({ device: x.device as string, rule_id: str(x.rule_id), since_round: num(x.since_round) }));
  return {
    decision_engine: str(o.decision_engine) ?? "unknown",
    wazuh: w
      ? {
          up: w.up === true,
          decisions: num(w.decisions),
          last_decision_t: num(w.last_decision_t),
          dashboard_url: str(w.dashboard_url),
        }
      : null,
    finsentinel: obj(o.finsentinel),
    pending,
    suspended,
    thresholds: obj(o.thresholds),
    updates,
  };
}

// ---------- merging ----------

export function normDecision(d: string | null | undefined): DecisionKind | null {
  if (!d) return null;
  const x = d.toLowerCase();
  if (x.startsWith("approv")) return "approve";
  if (x.startsWith("quarantin")) return "quarantine";
  if (x.startsWith("reject")) return "reject";
  if (x.startsWith("suspend")) return "suspend";
  if (x === "alert") return "alert";
  if (x.startsWith("pending")) return "pending";
  if (x.startsWith("block") || x.startsWith("refus")) return "blocked";
  return null;
}

function deciderWord(d: Decider): string | null {
  return d === "none" ? null : d;
}

/** One row per update the event stream has seen (attack requests included, keyed separately). */
export function rowsFromEvents(s: DashState): UpdateRow[] {
  return Object.entries(s.gates).map(([key, g]) => {
    const attack = g.ev.source === "attack";
    const w = attack ? undefined : s.wazuhByUpdate[g.ev.update_id];
    let decision: DecisionKind | null;
    let raw: string | null;
    if (g.ev.http_status >= 400) {
      decision = "blocked";
      raw = `blocked (${g.ev.http_status})`;
    } else if (g.decision) {
      decision = normDecision(g.decision);
      raw = g.decision;
    } else if (w) {
      decision = w.decision === "suspend_device" ? "suspend" : w.decision === "alert" ? "pending" : normDecision(w.decision);
      raw = w.decision;
    } else {
      const st = effectiveStatus(g);
      decision = normDecision(st);
      raw = st;
    }
    return {
      key,
      updateId: g.ev.update_id,
      device: g.ev.device,
      round: g.ev.round,
      t: g.ev.t,
      http: g.ev.http_status,
      fsStatus: g.ev.approval_status,
      risk: g.ev.risk_score,
      severity: g.ev.severity,
      reasons: g.ev.reasons,
      decision,
      decisionRaw: raw,
      decidedBy: deciderWord(deciderOf(g, s)),
      ruleId: w?.rule_id ?? null,
      ruleLevel: w?.rule_level ?? null,
      ruleDescription: w?.rule_description ?? null,
      enforced: w ? w.enforced : null,
      enforcementDetail: w?.enforcement_detail ?? null,
      attack,
      from: "events",
    };
  });
}

/** API rows win for the fields they carry; events fill rule level / description / enforcement. */
export function mergeUpdates(s: DashState, api: ConsoleApi | null): UpdateRow[] {
  const byKey = new Map<string, UpdateRow>();
  for (const r of rowsFromEvents(s)) byKey.set(r.key, r);
  for (const a of api?.updates ?? []) {
    const ev = byKey.get(a.update_id);
    const w = s.wazuhByUpdate[a.update_id];
    const merged: UpdateRow = {
      key: a.update_id,
      updateId: a.update_id,
      device: a.device || ev?.device || "",
      round: a.round ?? ev?.round ?? null,
      t: a.t ?? ev?.t ?? null,
      http: a.http_status ?? ev?.http ?? null,
      fsStatus: a.finsentinel_status ?? ev?.fsStatus ?? null,
      risk: a.risk_score ?? ev?.risk ?? null,
      severity: a.severity ?? ev?.severity ?? null,
      reasons: a.reasons.length ? a.reasons : ev?.reasons ?? [],
      decision: normDecision(a.decision) ?? ev?.decision ?? null,
      decisionRaw: a.decision ?? ev?.decisionRaw ?? null,
      decidedBy: a.decided_by ?? ev?.decidedBy ?? null,
      ruleId: a.rule_id ?? ev?.ruleId ?? w?.rule_id ?? null,
      ruleLevel: ev?.ruleLevel ?? w?.rule_level ?? null,
      ruleDescription: ev?.ruleDescription ?? w?.rule_description ?? null,
      enforced: ev?.enforced ?? (w ? w.enforced : null),
      enforcementDetail: ev?.enforcementDetail ?? w?.enforcement_detail ?? null,
      attack: ev?.attack ?? a.update_id.includes("#"),
      from: ev ? "both" : "api",
    };
    byKey.set(a.update_id, merged);
  }
  return [...byKey.values()].sort((x, y) => (y.t ?? 0) - (x.t ?? 0) || y.key.localeCompare(x.key));
}

export interface RowFilter {
  device: string; // "" = all
  decision: DecisionKind | ""; // "" = all
}

export function filterRows(rows: UpdateRow[], f: RowFilter): UpdateRow[] {
  return rows.filter((r) => (!f.device || r.device === f.device) && (!f.decision || r.decision === f.decision));
}

export function buildConsoleView(s: DashState, api: ConsoleApi | null): ConsoleView {
  // pending review
  let pending: PendingRow[];
  if (api) {
    pending = api.pending.map((p) => ({
      updateId: p.update_id,
      device: p.client_id,
      round: p.round_number,
      risk: p.risk_score,
      severity: p.risk_severity,
      reasons: p.anomaly_reasons,
      status: p.status,
      wazuh: s.wazuhByUpdate[p.update_id] ?? null,
    }));
  } else {
    pending = awaitingDecision(s).map((g) => ({
      updateId: g.ev.update_id,
      device: g.ev.device,
      round: g.ev.round,
      risk: g.ev.risk_score,
      severity: g.ev.severity,
      reasons: g.ev.reasons,
      status: g.ev.approval_status,
      wazuh: s.wazuhByUpdate[g.ev.update_id] ?? null,
    }));
  }
  // suspended: API list ∪ devices the event stream shows as suspended
  const susp = new Map<string, SuspendedRow>();
  for (const x of api?.suspended ?? []) susp.set(x.device, { device: x.device, ruleId: x.rule_id, sinceRound: x.since_round });
  if (!api) {
    for (const id of s.deviceOrder) {
      const d = s.devices[id];
      if (d?.suspended) susp.set(id, { device: id, ruleId: d.suspendedRule, sinceRound: d.suspendedRound });
    }
  }
  const apiEngine = api?.decision_engine;
  const engine: ConsoleView["engine"] =
    apiEngine === "wazuh" || apiEngine === "coordinator" ? apiEngine : s.engine?.engine ?? (s.run ? "coordinator" : "unknown");
  const lastEv = s.wazuhDecisions[0];
  return {
    apiAvailable: !!api,
    engine,
    engineDetail: s.engine?.detail ?? null,
    wazuhUp: api?.wazuh ? api.wazuh.up : s.components.wazuh ? s.components.wazuh.status === "up" : null,
    decisions: api?.wazuh?.decisions ?? s.wazuhDecisions.length,
    lastDecisionT: api?.wazuh?.last_decision_t ?? lastEv?.t ?? null,
    dashboardUrl: api?.wazuh?.dashboard_url ?? s.engine?.dashboard ?? null,
    finsentinel: api?.finsentinel ?? null,
    thresholds: api?.thresholds ?? null,
    pending,
    suspended: [...susp.values()],
    rows: mergeUpdates(s, api),
  };
}
