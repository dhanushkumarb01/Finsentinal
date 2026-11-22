import { useMemo, useState } from "react";
import type { AppConfig } from "../../types";
import type { DashState } from "../../state/reducer";
import { fmtRisk } from "../../state/reducer";
import { buildConsoleView, filterRows, type DecisionKind, type UpdateRow } from "../../state/console";
import { deviceLabelOf, fmtClock } from "../../state/derive";
import { postAdmin } from "../../config";
import type { ConsolePoll } from "../../console/useConsole";
import { DECISION_WORDS } from "../live/WazuhStage";
import type { ToastFn } from "../live/AttackDrawer";

const DECISION_LABEL: Record<DecisionKind, string> = {
  approve: "Approved",
  quarantine: "Quarantined",
  reject: "Rejected",
  suspend: "Suspended",
  alert: "Alert",
  pending: "Pending",
  blocked: "Blocked",
};
const DECISION_CLS: Record<DecisionKind, string> = {
  approve: "ok",
  quarantine: "bad",
  reject: "bad",
  suspend: "bad",
  alert: "warn",
  pending: "warn",
  blocked: "bad",
};

/** Flatten nested JSON to "a.b.c" → scalar pairs (for unknown shapes). */
export function flatten(o: unknown, prefix = "", depth = 0, out: [string, unknown][] = []): [string, unknown][] {
  if (o && typeof o === "object" && !Array.isArray(o) && depth < 4) {
    for (const [k, v] of Object.entries(o as Record<string, unknown>)) flatten(v, prefix ? `${prefix}.${k}` : k, depth + 1, out);
  } else {
    out.push([prefix, o]);
  }
  return out;
}

function fmtVal(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  if (Array.isArray(v)) return v.length <= 6 ? v.map((x) => fmtVal(x)).join(", ") : `${v.length} items`;
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function KVTable({ data, empty }: { data: Record<string, unknown> | null; empty: string }) {
  if (!data) return <div className="muted small pad">{empty}</div>;
  const rows = flatten(data);
  if (!rows.length) return <div className="muted small pad">(empty)</div>;
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k} className="kv-row">
          <dt title={k}>{k}</dt>
          <dd className="mono">{fmtVal(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

function Tile({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone: "ok" | "warn" | "bad" | "wz" | "dim" }) {
  return (
    <div className={`tile t-${tone}`}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

function UpdatesTable({ s, rows }: { s: DashState; rows: UpdateRow[] }) {
  return (
    <div className="table-wrap">
      <table className="utable">
        <thead>
          <tr>
            <th>Time</th>
            <th>Device</th>
            <th>Rnd</th>
            <th>FinSentinel result</th>
            <th>Wazuh decision</th>
            <th>Enforced</th>
            <th>Decided by</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={7} className="muted center">
                No updates match.
              </td>
            </tr>
          )}
          {rows.map((r) => (
            <tr key={r.key} className={`row-${r.decision ? DECISION_CLS[r.decision] : "none"}`}>
              <td className="mono nowrap">{fmtClock(r.t)}</td>
              <td className="nowrap">
                {r.attack ? <span className="atk">Attacker</span> : null} {deviceLabelOf(s, r.device)}
              </td>
              <td className="mono">{r.round ?? "—"}</td>
              <td>
                <div className="fs-cell">
                  <span className={`http ${r.http != null && r.http >= 400 ? "bad" : ""}`}>{r.http ?? "—"}</span>
                  <span>{r.fsStatus ? r.fsStatus.replace(/_/g, " ") : r.http != null && r.http >= 400 ? "refused" : "—"}</span>
                  <span className="muted">
                    risk {fmtRisk(r.risk)}
                    {r.severity ? ` · ${r.severity}` : ""}
                  </span>
                </div>
                {r.reasons.length > 0 && (
                  <div className="reasons" title={r.reasons.join("\n")}>
                    {r.reasons.join(" · ")}
                  </div>
                )}
              </td>
              <td>
                <div className="wz-cell">
                  {r.decision && <span className={`dpill ${DECISION_CLS[r.decision]}`}>{DECISION_LABEL[r.decision]}</span>}
                  {r.ruleId && (
                    <span className="mono">
                      rule {r.ruleId}
                      {r.ruleLevel != null && <span className="muted"> · L{r.ruleLevel}</span>}
                    </span>
                  )}
                </div>
                {r.ruleDescription && <div className="reasons">{r.ruleDescription}</div>}
              </td>
              <td className="center">{r.enforced == null ? <span className="muted">—</span> : r.enforced ? <span className="yes">✓</span> : <span className="no">no</span>}</td>
              <td className={`by by-${r.decidedBy ?? "none"}`}>{r.decidedBy ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function SecurityConsole({
  s,
  cfg,
  live,
  poll,
  toast,
}: {
  s: DashState;
  cfg: AppConfig;
  live: boolean;
  poll: ConsolePoll;
  toast: ToastFn;
}) {
  const v = useMemo(() => buildConsoleView(s, poll.data), [s, poll.data]);
  const [devF, setDevF] = useState("");
  const [decF, setDecF] = useState<DecisionKind | "">("");
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const rows = filterRows(v.rows, { device: devF, decision: decF });
  const devices = [...new Set([...s.deviceOrder, ...v.rows.map((r) => r.device)])].filter(Boolean);

  async function decide(client_id: string, update_id: string, decision: "approve" | "reject") {
    setBusy((b) => ({ ...b, [update_id]: true }));
    toast("pending", `${decision === "approve" ? "Approving" : "Rejecting"} ${update_id}…`);
    const r = await postAdmin(cfg, "/fl/admin/decision", { client_id, update_id, decision });
    setBusy((b) => ({ ...b, [update_id]: false }));
    if (r.ok) toast("ok", `Override sent for ${update_id} — waiting for the confirmation event`);
    else toast("error", `Override failed${r.status ? ` (HTTP ${r.status})` : ""} ${r.text.slice(0, 120)}`);
  }
  async function unsuspend(device: string) {
    setBusy((b) => ({ ...b, [`u:${device}`]: true }));
    toast("pending", `Unsuspending ${device}…`);
    const r = await postAdmin(cfg, "/fl/admin/unsuspend", { device });
    setBusy((b) => ({ ...b, [`u:${device}`]: false }));
    if (r.ok) toast("ok", `${device} unsuspended`);
    else toast("error", `Unsuspend failed${r.status ? ` (HTTP ${r.status})` : ""} ${r.text.slice(0, 120)}`);
  }

  const apiNote = !live
    ? "Console API not available in replay — built from the recorded events"
    : poll.status === "ok"
      ? "Console API connected · refreshes every 2 s"
      : poll.status === "error"
        ? `Console API unreachable (${poll.error}) — showing live events only`
        : "Connecting to the console API…";

  const engineTile =
    v.engine === "wazuh"
      ? v.wazuhUp === false
        ? { value: "Wazuh down", sub: "decisions are stalled — check the Wazuh manager", tone: "bad" as const }
        : { value: "Wazuh", sub: "SIEM rules decide; coordinator enforces", tone: "wz" as const }
      : v.engine === "coordinator"
        ? { value: "Coordinator", sub: "Wazuh offline — coordinator decides", tone: "warn" as const }
        : { value: "—", sub: "no run yet", tone: "dim" as const };

  const fsComp = s.components.finsentinel;
  const fsNums = v.finsentinel ? flatten(v.finsentinel).filter(([, x]) => typeof x === "number").slice(0, 3) : [];
  const fsTile = {
    value: fsComp ? (fsComp.status === "up" ? "Healthy" : "Down") : v.finsentinel ? "Reachable" : "—",
    sub: fsNums.length ? fsNums.map(([k, x]) => `${k.split(".").slice(-2).join(" ")}: ${fmtVal(x)}`).join(" · ") : live ? "dashboard data via console API" : "no dashboard data in replay",
    tone: fsComp?.status === "down" ? ("bad" as const) : fsComp || v.finsentinel ? ("ok" as const) : ("dim" as const),
  };

  return (
    <div className="console">
      <header className="con-head">
        <div>
          <h1>Security Console</h1>
          <p>Every update FinSentinel checked, every decision Wazuh made, and the controls to override them.</p>
        </div>
        <div className="con-head-right">
          <span className={`api-pill ${live ? (poll.status === "ok" ? "ok" : poll.status === "error" ? "bad" : "warn") : "dim"}`}>
            <i />
            {apiNote}
          </span>
          {v.dashboardUrl && (
            <a className="wz-link" href={v.dashboardUrl} target="_blank" rel="noreferrer">
              Open Wazuh dashboard ↗
            </a>
          )}
        </div>
      </header>

      <div className="tiles">
        <Tile label="Decision engine" value={engineTile.value} sub={v.engineDetail ?? engineTile.sub} tone={engineTile.tone} />
        <Tile label="FinSentinel" value={fsTile.value} sub={fsTile.sub} tone={fsTile.tone} />
        <Tile
          label="Wazuh decisions"
          value={String(v.decisions)}
          sub={v.lastDecisionT ? `last at ${fmtClock(v.lastDecisionT)}` : "none yet"}
          tone={v.decisions ? "wz" : "dim"}
        />
        <Tile label="Pending review" value={String(v.pending.length)} sub={v.pending.length ? "a human can decide below" : "nothing waiting"} tone={v.pending.length ? "warn" : "ok"} />
        <Tile
          label="Suspended devices"
          value={String(v.suspended.length)}
          sub={v.suspended.length ? v.suspended.map((x) => deviceLabelOf(s, x.device)).join(", ") : "none"}
          tone={v.suspended.length ? "bad" : "ok"}
        />
      </div>

      <div className="con-grid">
        <section className="card updates">
          <header className="card-head">
            <h2>
              Updates <span className="muted">· {rows.length} of {v.rows.length}</span>
            </h2>
            <div className="filters">
              <select value={devF} onChange={(e) => setDevF(e.target.value)} aria-label="Filter by device">
                <option value="">All devices</option>
                {devices.map((d) => (
                  <option key={d} value={d}>
                    {deviceLabelOf(s, d)}
                  </option>
                ))}
              </select>
              <select value={decF} onChange={(e) => setDecF(e.target.value as DecisionKind | "")} aria-label="Filter by decision">
                <option value="">All decisions</option>
                {(Object.keys(DECISION_LABEL) as DecisionKind[]).map((k) => (
                  <option key={k} value={k}>
                    {DECISION_LABEL[k]}
                  </option>
                ))}
              </select>
            </div>
          </header>
          <UpdatesTable s={s} rows={rows.slice(0, 200)} />
        </section>

        <div className="con-side">
          <section className="card">
            <header className="card-head">
              <h2>
                Pending review <span className="muted">· manual override</span>
              </h2>
            </header>
            {v.pending.length === 0 && <div className="muted small pad">Nothing waiting for a decision.</div>}
            {v.pending.map((p) => (
              <div key={p.updateId} className="pend">
                <div className="pend-main">
                  <b>{deviceLabelOf(s, p.device)}</b>
                  <span className="muted">
                    round {p.round ?? "?"} · risk {fmtRisk(p.risk)}
                    {p.severity ? ` · ${p.severity}` : ""}
                  </span>
                  {p.wazuh && (
                    <span className="pend-wz">
                      Wazuh rule {p.wazuh.rule_id}: {DECISION_WORDS[p.wazuh.decision].word.toLowerCase()}
                    </span>
                  )}
                  {p.reasons[0] && <span className="reasons">{p.reasons[0]}</span>}
                </div>
                <div className="pend-btns">
                  <button className="btn ok" disabled={!live || busy[p.updateId]} onClick={() => decide(p.device, p.updateId, "approve")}>
                    Approve
                  </button>
                  <button className="btn bad" disabled={!live || busy[p.updateId]} onClick={() => decide(p.device, p.updateId, "reject")}>
                    Reject
                  </button>
                </div>
              </div>
            ))}
          </section>

          <section className="card">
            <header className="card-head">
              <h2>Suspended devices</h2>
            </header>
            {v.suspended.length === 0 && <div className="muted small pad">No device is suspended.</div>}
            {v.suspended.map((x) => (
              <div key={x.device} className="pend">
                <div className="pend-main">
                  <b>{deviceLabelOf(s, x.device)}</b>
                  <span className="muted">
                    {x.ruleId ? `Wazuh rule ${x.ruleId}` : "suspended"}
                    {x.sinceRound != null ? ` · since round ${x.sinceRound}` : ""}
                  </span>
                </div>
                <div className="pend-btns">
                  <button className="btn" disabled={!live || busy[`u:${x.device}`]} onClick={() => unsuspend(x.device)}>
                    Unsuspend
                  </button>
                </div>
              </div>
            ))}
          </section>

          <section className="card log">
            <header className="card-head">
              <h2>
                Wazuh decision log <span className="muted">· {s.wazuhDecisions.length}</span>
              </h2>
            </header>
            <div className="log-list">
              {s.wazuhDecisions.length === 0 && (
                <div className="muted small pad">{s.engine?.engine === "wazuh" ? "No decisions yet." : "No Wazuh decisions in this run (coordinator decides)."}</div>
              )}
              {s.wazuhDecisions.slice(0, 80).map((w) => (
                <div key={w.seq} className={`log-item ${DECISION_WORDS[w.decision].cls}`}>
                  <div className="log-top">
                    <span className="mono">{fmtClock(w.t)}</span>
                    <b>{DECISION_WORDS[w.decision].word}</b>
                    <span>{deviceLabelOf(s, w.device)}</span>
                    <span className="mono muted">
                      rule {w.rule_id} · L{w.rule_level}
                    </span>
                  </div>
                  <div className="log-desc">
                    {w.rule_description}
                    {w.enforcement_detail && <span className="muted"> — {w.enforcement_detail}</span>}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>

      <div className="con-grid3">
        <section className="card log">
          <header className="card-head">
            <h2>Alert &amp; audit feed</h2>
          </header>
          <div className="log-list">
            {s.alerts.length === 0 && <div className="muted small pad">No alerts yet.</div>}
            {s.alerts.slice(0, 80).map((a) => (
              <div key={a.id} className={`log-item sev-${a.severity}`}>
                <div className="log-top">
                  <span className="mono">{fmtClock(a.t)}</span>
                  <span className="src-tag">{a.source === "Drift monitor" ? "Coordinator drift monitor (research layer)" : a.source}</span>
                  <b>{a.title}</b>
                </div>
                {a.detail && <div className="log-desc">{a.detail}</div>}
              </div>
            ))}
          </div>
        </section>
        <section className="card">
          <header className="card-head">
            <h2>Calibrated thresholds</h2>
          </header>
          <KVTable data={v.thresholds} empty={live ? "Not reported by the console API yet." : "Not available in replay."} />
        </section>
        <section className="card">
          <header className="card-head">
            <h2>FinSentinel dashboard</h2>
          </header>
          <KVTable data={v.finsentinel} empty={live ? "Not reported by the console API yet." : "Not available in replay."} />
        </section>
      </div>
    </div>
  );
}
