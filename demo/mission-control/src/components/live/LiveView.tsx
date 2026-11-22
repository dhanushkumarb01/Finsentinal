import type { ReactNode } from "react";
import type { DashState } from "../../state/reducer";
import { Narrator } from "./Narrator";
import { DeviceStage } from "./DeviceStage";
import { TunnelStage } from "./TunnelStage";
import { GateStage } from "./GateStage";
import { WazuhStage } from "./WazuhStage";
import { ModelStage } from "./ModelStage";

function Stage({
  n,
  title,
  caption,
  cls,
  children,
}: {
  n: number;
  title: string;
  caption: string;
  cls: string;
  children: ReactNode;
}) {
  return (
    <section className={`stage ${cls}`}>
      <header className="stage-head">
        <span className="stage-n">{n}</span>
        <div>
          <h2>{title}</h2>
          <p>{caption}</p>
        </div>
      </header>
      {children}
    </section>
  );
}

export function LiveView({ s, onOpenDevice }: { s: DashState; onOpenDevice: (id: string) => void }) {
  const n = s.deviceOrder.length;
  return (
    <div className="live">
      <Narrator s={s} />
      <div className="pipeline">
        <Stage n={1} cls="st-devices" title={`Devices${n ? ` · ${n}` : ""}`} caption="Each trains on its own data — the data never leaves the device">
          <DeviceStage s={s} onOpen={onOpenDevice} />
        </Stage>
        <Stage n={2} cls="st-tunnel" title="Encrypted tunnel" caption="Only signed, encrypted weights travel">
          <TunnelStage s={s} />
        </Stage>
        <Stage n={3} cls="st-gate" title="FinSentinel gate" caption="Five checks on every update, in order">
          <GateStage s={s} />
        </Stage>
        <Stage n={4} cls="st-wazuh" title="Wazuh decision" caption="Wazuh SIEM watches every security event and decides in real time">
          <WazuhStage s={s} />
        </Stage>
        <Stage n={5} cls="st-model" title="Global model" caption="Approved updates are averaged into one shared model">
          <ModelStage s={s} />
        </Stage>
      </div>
      <footer className="honest">
        Simulated fleet: virtual devices run on this laptop; ESP32 boards run real firmware in the Wokwi simulator. Every value on screen is a
        real event from the running system or a recording of it. Click a device to look inside.
      </footer>
    </div>
  );
}
