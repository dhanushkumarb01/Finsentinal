import type { DashState } from "../state/reducer";
import { latestEval, normAccuracy } from "../state/reducer";
import type { ConnStatus } from "../live/socket";

export type Mode = "live" | "replay";

export function Header({
  s,
  mode,
  conn,
  onMode,
  onResearch,
}: {
  s: DashState;
  mode: Mode;
  conn: ConnStatus | "replay";
  onMode: (m: Mode) => void;
  onResearch: () => void;
}) {
  const ev = latestEval(s);
  const acc = ev ? normAccuracy(ev.accuracy) * 100 : null;
  const comps = Object.entries(s.components);
  const phaseWord = s.phase === "open" ? "open" : s.phase === "finished" ? "finished" : "waiting";
  let tunnelHost = "";
  if (s.tunnel?.url) {
    try {
      tunnelHost = new URL(s.tunnel.url).host;
    } catch {
      tunnelHost = s.tunnel.url;
    }
  }
  return (
    <header className="hdr">
      <div className="hdr-title">
        <svg viewBox="0 0 24 24" className="shield" aria-hidden>
          <path d="M12 2l8 3v6c0 5-3.4 9.2-8 11-4.6-1.8-8-6-8-11V5z" fill="#0ea5e9" />
          <path d="M8 12l3 3 5-6" stroke="#fff" strokeWidth="2.2" fill="none" strokeLinecap="round" />
        </svg>
        <div>
          <h1>Mission Control — Secured Federated Learning</h1>
          <div className="hdr-sub">
            {s.run ? (
              <>
                model <b className="mono">{s.run.model_version}</b> · {s.run.param_count.toLocaleString()} params · UCI HAR ·{" "}
                {s.run.test_samples.toLocaleString()} test samples
              </>
            ) : (
              "waiting for run_started"
            )}
            {s.runId && <span className="muted mono"> · run {s.runId}</span>}
          </div>
        </div>
      </div>

      <div className="hdr-stat">
        <div className="stat-label">Round</div>
        <div className="stat-val">
          {s.round ?? "—"}
          <span className="stat-of"> / {s.totalRounds ?? "—"}</span>
        </div>
      </div>
      <div className="hdr-stat acc">
        <div className="stat-label">Global accuracy</div>
        <div className="stat-val big">{acc == null ? "—" : `${acc.toFixed(1)}%`}</div>
      </div>
      <div className="hdr-stat">
        <div className="stat-label">Phase</div>
        <div className={`phase ph-${phaseWord}`}>{phaseWord}</div>
      </div>

      <div className="hdr-status">
        <div className="row">
          <div className="mode-toggle" role="group" aria-label="Data source">
            <button className={mode === "live" ? "on" : ""} onClick={() => onMode("live")}>
              Live
            </button>
            <button className={mode === "replay" ? "on" : ""} onClick={() => onMode("replay")}>
              Replay
            </button>
          </div>
          <span className={`conn-dot c-${conn}`} title={`connection: ${conn}`} />
          <span className="muted small">{conn === "replay" ? "recording" : conn}</span>
          <button className="research-btn" onClick={onResearch}>
            FEMNIST results
          </button>
        </div>
        <div className="row">
          <span className={`pill ${s.tunnel ? (s.tunnel.status === "up" ? "pill-ok" : "pill-bad") : "pill-off"}`}>
            Tunnel {s.tunnel ? s.tunnel.status : "—"}
            {tunnelHost && <span className="mono pill-host"> {tunnelHost}</span>}
            {s.tunnel?.beacon_published && <span title="Signed beacon published"> · beacon ✓</span>}
          </span>
          {comps.map(([name, c]) => (
            <span key={name} className="comp" title={c.detail ?? ""}>
              <span className={`dot ${c.status === "up" ? "dot-ok" : "dot-bad"}`} />
              {name}
            </span>
          ))}
        </div>
      </div>
    </header>
  );
}
