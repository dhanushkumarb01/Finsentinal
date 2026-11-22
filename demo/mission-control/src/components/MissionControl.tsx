import { useCallback, useState, type ReactNode } from "react";
import type { AppConfig } from "../types";
import type { DashState } from "../state/reducer";
import { awaitingDecision } from "../state/reducer";
import type { ConnStatus } from "../live/socket";
import type { ConsolePoll } from "../console/useConsole";
import { TopNav, type Mode, type View } from "./TopNav";
import { LiveView } from "./live/LiveView";
import { AttackDrawer, type ToastFn } from "./live/AttackDrawer";
import { InsidePanel } from "./InsidePanel";
import { SecurityConsole } from "./console/SecurityConsole";
import { ResearchView } from "./ResearchView";

const OFF_POLL: ConsolePoll = { data: null, status: "off", error: null, lastOk: null };

function initialView(): View {
  if (typeof window === "undefined") return "live";
  const v = new URLSearchParams(window.location.search).get("view");
  return v === "console" || v === "research" ? v : "live";
}

/** The whole screen, driven only by props (App owns the data sources). */
export function MissionControl({
  s,
  cfg,
  mode,
  conn,
  onMode,
  toast,
  poll = OFF_POLL,
  view: viewProp,
  onView,
  bottomBar,
  banner,
}: {
  s: DashState;
  cfg: AppConfig;
  mode: Mode;
  conn: ConnStatus | "replay";
  onMode: (m: Mode) => void;
  toast: ToastFn;
  poll?: ConsolePoll;
  view?: View;
  onView?: (v: View) => void;
  bottomBar?: ReactNode;
  banner?: ReactNode;
}) {
  const [ownView, setOwnView] = useState<View>(initialView);
  const view = viewProp ?? ownView;
  const setView = onView ?? setOwnView;
  // ?inside=<device id> opens the "inside the device" panel on load (rehearsal helper)
  const [inside, setInside] = useState<string | null>(() =>
    typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("inside"),
  );
  const [attackOpen, setAttackOpen] = useState(false);
  const closeInside = useCallback(() => setInside(null), []);
  const closeAttack = useCallback(() => setAttackOpen(false), []);
  const pendingCount = poll.data ? poll.data.pending.length : awaitingDecision(s).length;

  return (
    <div className={`app view-${view}`}>
      <TopNav
        s={s}
        view={view}
        onView={(v) => {
          setView(v);
          if (v !== "live") setAttackOpen(false);
        }}
        mode={mode}
        conn={conn}
        onMode={onMode}
        attackOpen={attackOpen}
        onAttack={() => setAttackOpen((o) => !o)}
        pendingCount={pendingCount}
      />
      {banner}
      <main className="content">
        {view === "live" && <LiveView s={s} onOpenDevice={setInside} />}
        {view === "console" && <SecurityConsole s={s} cfg={cfg} live={mode === "live"} poll={poll} toast={toast} />}
        {view === "research" && <ResearchView url={cfg.femnist_url} />}
      </main>
      {bottomBar}
      {view === "live" && <AttackDrawer s={s} cfg={cfg} enabled={mode === "live"} open={attackOpen} onClose={closeAttack} toast={toast} />}
      {inside && view === "live" && <InsidePanel s={s} id={inside} onClose={closeInside} />}
    </div>
  );
}
