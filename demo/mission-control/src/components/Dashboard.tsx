import { useCallback, useState, type ReactNode } from "react";
import type { AppConfig } from "../types";
import type { DashState } from "../state/reducer";
import type { ConnStatus } from "../live/socket";
import { Header, type Mode } from "./Header";
import { DeviceWall } from "./DeviceWall";
import { Tunnel } from "./Tunnel";
import { Gate } from "./Gate";
import { GlobalChart } from "./GlobalChart";
import { AlertFeed } from "./AlertFeed";
import { AttackConsole, type ToastFn } from "./AttackConsole";
import { InsidePanel } from "./InsidePanel";
import { ResearchOverlay } from "./ResearchOverlay";

export function Dashboard({
  s,
  cfg,
  mode,
  conn,
  onMode,
  toast,
  bottomBar,
  banner,
}: {
  s: DashState;
  cfg: AppConfig;
  mode: Mode;
  conn: ConnStatus | "replay";
  onMode: (m: Mode) => void;
  toast: ToastFn;
  bottomBar?: ReactNode;
  banner?: ReactNode;
}) {
  // ?inside=<device id> opens the "inside the device" panel on load (rehearsal helper)
  const [inside, setInside] = useState<string | null>(() =>
    typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("inside"),
  );
  const [research, setResearch] = useState(false);
  const closeInside = useCallback(() => setInside(null), []);
  const closeResearch = useCallback(() => setResearch(false), []);
  return (
    <div className="app">
      <Header s={s} mode={mode} conn={conn} onMode={onMode} onResearch={() => setResearch(true)} />
      {banner}
      <main className="main">
        <DeviceWall s={s} onOpen={setInside} />
        <div className="col-mid">
          <Tunnel s={s} />
          <Gate s={s} />
        </div>
        <div className="col-right">
          <GlobalChart s={s} />
          <AlertFeed s={s} />
          <AttackConsole s={s} cfg={cfg} enabled={mode === "live"} toast={toast} />
        </div>
      </main>
      {bottomBar}
      <footer className="foot">
        Simulated fleet: virtual devices run on the laptop; ESP32 boards run in the Wokwi simulator. All numbers come from
        the live system or a recording of it.
      </footer>
      {inside && <InsidePanel s={s} id={inside} onClose={closeInside} />}
      {research && <ResearchOverlay url={cfg.femnist_url} onClose={closeResearch} />}
    </div>
  );
}
