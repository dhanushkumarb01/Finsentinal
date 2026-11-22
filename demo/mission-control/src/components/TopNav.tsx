import type { DashState } from "../state/reducer";
import type { ConnStatus } from "../live/socket";

export type Mode = "live" | "replay";
export type View = "live" | "console" | "research";

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "live", label: "Live", hint: "The story, for the audience" },
  { id: "console", label: "Security Console", hint: "Operator view: every update and decision" },
  { id: "research", label: "Research", hint: "FEMNIST study results" },
];

export function TopNav({
  s,
  view,
  onView,
  mode,
  conn,
  onMode,
  attackOpen,
  onAttack,
  pendingCount,
}: {
  s: DashState;
  view: View;
  onView: (v: View) => void;
  mode: Mode;
  conn: ConnStatus | "replay";
  onMode: (m: Mode) => void;
  attackOpen: boolean;
  onAttack: () => void;
  pendingCount: number;
}) {
  const comps = Object.entries(s.components);
  const down = comps.filter(([, c]) => c.status !== "up").map(([n]) => n);
  const connWord = conn === "replay" ? "Playing a recording" : conn === "open" ? "Connected" : conn === "lost" ? "Reconnecting…" : "Connecting…";
  return (
    <nav className="nav">
      <div className="brand">
        <svg viewBox="0 0 24 24" className="brand-mark" aria-hidden>
          <path d="M12 2l8 3v6c0 5-3.4 9.2-8 11-4.6-1.8-8-6-8-11V5z" fill="url(#bm)" />
          <path d="M8 12l3 3 5-6" stroke="#06101f" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <defs>
            <linearGradient id="bm" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0" stopColor="#7dd3fc" />
              <stop offset="1" stopColor="#a78bfa" />
            </linearGradient>
          </defs>
        </svg>
        <div className="brand-text">
          <div className="brand-name">Mission Control</div>
          <div className="brand-sub">Secured federated learning · FinSentinel + Wazuh</div>
        </div>
      </div>

      <div className="tabs" role="tablist" aria-label="Views">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            role="tab"
            aria-selected={view === v.id}
            className={`tab ${view === v.id ? "on" : ""}`}
            onClick={() => onView(v.id)}
            title={v.hint}
          >
            {v.label}
            {v.id === "console" && pendingCount > 0 && <span className="tab-badge">{pendingCount}</span>}
          </button>
        ))}
      </div>

      <div className="nav-right">
        {comps.length > 0 && (
          <span
            className={`sys ${down.length ? "down" : "up"}`}
            title={comps.map(([name, c]) => `${name}: ${c.status}${c.detail ? ` — ${c.detail}` : ""}`).join("\n")}
          >
            <i />
            {down.length ? `${down.join(", ")} down` : `All ${comps.length} systems up`}
          </span>
        )}
        {view === "live" && (
          <button className={`attack-btn ${attackOpen ? "on" : ""}`} onClick={onAttack} aria-expanded={attackOpen}>
            <svg viewBox="0 0 16 16" aria-hidden>
              <path d="M8 1.5l6.5 12h-13z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
              <path d="M8 6v3.6M8 11.4v.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
            Attack console
          </button>
        )}
        <div className="source">
          <div className="seg" role="group" aria-label="Data source">
            <button className={mode === "live" ? "on" : ""} onClick={() => onMode("live")}>
              Live stream
            </button>
            <button className={mode === "replay" ? "on" : ""} onClick={() => onMode("replay")}>
              Recording
            </button>
          </div>
          <span className={`conn c-${conn}`} title={connWord}>
            <i />
            {connWord}
          </span>
        </div>
      </div>
    </nav>
  );
}
