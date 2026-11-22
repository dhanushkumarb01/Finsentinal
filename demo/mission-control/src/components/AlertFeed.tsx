import type { DashState } from "../state/reducer";
import { fmtClock } from "../state/derive";

const SOURCE_LABEL: Record<string, string> = {
  FinSentinel: "FinSentinel",
  Coordinator: "Coordinator",
  Attack: "Attack",
  "Drift monitor": "Coordinator drift monitor (research layer)",
};

export function AlertFeed({ s }: { s: DashState }) {
  const items = s.alerts.slice(0, 40);
  return (
    <section className="panel feed">
      <h2 className="panel-title">
        Live alert feed <span className="muted">· newest first</span>
      </h2>
      <div className="feed-list">
        {items.length === 0 && <div className="empty small">No alerts yet.</div>}
        {items.map((a) => (
          <div key={a.id} className={`feed-item sev-${a.severity}`}>
            <div className="feed-top">
              <span className={`src src-${a.source.replace(/\s/g, "")}`}>{SOURCE_LABEL[a.source]}</span>
              <span className="feed-sev">{a.severity}</span>
              {a.device && <span className="feed-dev">{s.devices[a.device]?.info.label ?? a.device}</span>}
              <span className="feed-time mono">{fmtClock(a.t)}</span>
            </div>
            <div className="feed-title">{a.title}</div>
            {a.detail && <div className="feed-detail">{a.detail}</div>}
          </div>
        ))}
      </div>
    </section>
  );
}
