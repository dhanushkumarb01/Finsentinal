import { useEffect, useState } from "react";
import type { FemnistData } from "../types";

export function parseFemnist(raw: unknown): FemnistData | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.series)) return null;
  const series = o.series
    .filter((x): x is Record<string, unknown> => !!x && typeof x === "object")
    .filter((x) => typeof x.label === "string" && typeof x.final_accuracy === "number")
    .map((x) => ({
      label: x.label as string,
      final_accuracy: x.final_accuracy as number,
      drift_cohort_accuracy: typeof x.drift_cohort_accuracy === "number" ? x.drift_cohort_accuracy : undefined,
    }));
  if (!series.length) return null;
  return {
    title: typeof o.title === "string" ? o.title : "FEMNIST results",
    note: typeof o.note === "string" ? o.note : "",
    source: typeof o.source === "string" ? o.source : "",
    series,
  };
}

const pct = (a: number) => (a > 1 ? a : a * 100);

export function ResearchOverlay({ url, onClose }: { url: string; onClose: () => void }) {
  const [data, setData] = useState<FemnistData | null | "loading">("loading");
  useEffect(() => {
    let live = true;
    fetch(url, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j) => live && setData(parseFemnist(j)))
      .catch(() => live && setData(null));
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => {
      live = false;
      window.removeEventListener("keydown", k);
    };
  }, [url, onClose]);

  const W = 1200,
    H = 520,
    L = 70,
    B = 70,
    T = 30;
  return (
    <div className="overlay research" onClick={onClose}>
      <div className="research-card" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <div className="research-kicker">Separate FEMNIST simulation study — research results, not produced by this live demo</div>
        {data === "loading" && <div className="empty">Loading…</div>}
        {data === null && <div className="empty">FEMNIST results not available.</div>}
        {data && data !== "loading" && (
          <>
            <h1>{data.title}</h1>
            <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg research-svg">
              {[0, 25, 50, 75, 100].map((g) => {
                const yy = T + (1 - g / 100) * (H - T - B);
                return (
                  <g key={g}>
                    <line x1={L} x2={W - 10} y1={yy} y2={yy} stroke="#1e293b" />
                    <text x={L - 10} y={yy + 6} textAnchor="end" className="svg-axis">
                      {g}%
                    </text>
                  </g>
                );
              })}
              {data.series.map((sr, i) => {
                const slot = (W - L - 20) / data.series.length;
                const x0 = L + i * slot + slot * 0.15;
                const bw = slot * 0.7;
                const hasCohort = sr.drift_cohort_accuracy != null;
                const w = hasCohort ? bw / 2 - 4 : bw;
                const bar = (v: number, x: number, color: string) => {
                  const h = (pct(v) / 100) * (H - T - B);
                  return (
                    <g>
                      <rect x={x} y={H - B - h} width={w} height={h} rx="4" fill={color} />
                      <text x={x + w / 2} y={H - B - h - 10} textAnchor="middle" className="svg-val big">
                        {pct(v).toFixed(1)}%
                      </text>
                    </g>
                  );
                };
                return (
                  <g key={sr.label}>
                    {bar(sr.final_accuracy, x0, "#22d3ee")}
                    {hasCohort && bar(sr.drift_cohort_accuracy as number, x0 + w + 8, "#c084fc")}
                    <text x={x0 + bw / 2} y={H - B + 32} textAnchor="middle" className="svg-axis big">
                      {sr.label}
                    </text>
                  </g>
                );
              })}
            </svg>
            <div className="chart-legend big">
              <span>
                <i className="lg lg-acc" /> final accuracy
              </span>
              {data.series.some((x) => x.drift_cohort_accuracy != null) && (
                <span>
                  <i className="lg lg-drift" /> drift-cohort accuracy
                </span>
              )}
            </div>
            {data.note && <p className="research-note">{data.note}</p>}
            {data.source && <p className="research-source">Source: {data.source}</p>}
            <p className="research-disclaimer">
              The live HAR demo illustrates the idea; it does not prove that these FEMNIST results transfer.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
