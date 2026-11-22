// Inline SVG artwork (no external assets): phone, ESP32 dev board, battery, signal, sparkline.

export type Tone = "idle" | "busy" | "ok" | "warn" | "bad" | "off";

const TONE_FILL: Record<Tone, string> = {
  idle: "#1e3a5f",
  busy: "#0ea5e9",
  ok: "#16a34a",
  warn: "#d97706",
  bad: "#dc2626",
  off: "#1f2937",
};

export function PhoneArt({ tone, busy }: { tone: Tone; busy?: boolean }) {
  return (
    <svg viewBox="0 0 60 110" className="art" aria-hidden>
      <rect x="2" y="2" width="56" height="106" rx="10" fill="#0b1220" stroke="#64748b" strokeWidth="2.5" />
      <rect x="7" y="12" width="46" height="84" rx="3" fill={TONE_FILL[tone]} opacity={tone === "off" ? 0.6 : 0.9} />
      <rect x="22" y="5" width="16" height="3" rx="1.5" fill="#334155" />
      <circle cx="30" cy="102" r="2.6" fill="#334155" />
      {busy && (
        <g className="spin-origin">
          <circle cx="30" cy="54" r="11" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="4" />
          <path d="M30 43 a11 11 0 0 1 11 11" fill="none" stroke="#fff" strokeWidth="4" strokeLinecap="round" className="spin" />
        </g>
      )}
    </svg>
  );
}

export function Esp32Art({ led, busy }: { led: "green" | "red" | "amber" | "off"; busy?: boolean }) {
  const ledColor = led === "green" ? "#22c55e" : led === "red" ? "#ef4444" : led === "amber" ? "#f59e0b" : "#1f2937";
  const pins = Array.from({ length: 12 }, (_, i) => i);
  return (
    <svg viewBox="0 0 70 110" className="art" aria-hidden>
      <rect x="10" y="4" width="50" height="102" rx="4" fill="#0f2a4a" stroke="#3b82f6" strokeWidth="1.5" />
      {pins.map((i) => (
        <g key={i}>
          <rect x="3" y={14 + i * 7.4} width="9" height="3.4" rx="1" fill="#cbd5e1" />
          <rect x="58" y={14 + i * 7.4} width="9" height="3.4" rx="1" fill="#cbd5e1" />
        </g>
      ))}
      <rect x="18" y="10" width="34" height="40" rx="2" fill="#94a3b8" stroke="#e2e8f0" strokeWidth="1" />
      <text x="35" y="28" textAnchor="middle" fontSize="6.5" fill="#0f172a" fontFamily="monospace" fontWeight="700">ESP32</text>
      <text x="35" y="37" textAnchor="middle" fontSize="4.5" fill="#0f172a" fontFamily="monospace">WROOM-32</text>
      <path d="M22 14 h6 v4 h4 v-4 h6 v4 h4 v-4 h4" stroke="#475569" strokeWidth="0.8" fill="none" />
      <rect x="24" y="60" width="22" height="14" rx="1.5" fill="#111827" />
      {busy && <rect x="26" y="62" width="18" height="10" fill="#0ea5e9" className="blink" />}
      <circle cx="22" cy="86" r="4.2" fill={ledColor} className={led !== "off" ? "led-glow" : undefined} />
      <text x="30" y="88" fontSize="5" fill="#93c5fd" fontFamily="monospace">LED</text>
      <rect x="26" y="98" width="18" height="8" rx="1" fill="#9ca3af" />
    </svg>
  );
}

export function Battery({ pct }: { pct: number | null }) {
  const p = pct == null ? 0 : Math.max(0, Math.min(100, pct));
  const color = pct == null ? "#475569" : p < 15 ? "#ef4444" : p < 35 ? "#f59e0b" : "#22c55e";
  return (
    <span className="battery" title="Battery">
      <svg viewBox="0 0 30 14" aria-hidden>
        <rect x="1" y="1" width="25" height="12" rx="2.5" fill="none" stroke="#94a3b8" strokeWidth="1.6" />
        <rect x="27" y="4.5" width="2.4" height="5" rx="1" fill="#94a3b8" />
        <rect x="3" y="3" width={(21 * p) / 100} height="8" rx="1.2" fill={color} />
      </svg>
      <span>{pct == null ? "—" : `${Math.round(p)}%`}</span>
    </span>
  );
}

export function SignalBars({ bars, dbm }: { bars: number; dbm: number | null }) {
  return (
    <span className="signal" title={dbm == null ? "No signal reading yet" : `${dbm} dBm`}>
      <svg viewBox="0 0 22 16" aria-hidden>
        {[0, 1, 2, 3].map((i) => (
          <rect
            key={i}
            x={1 + i * 5.4}
            y={12 - i * 3.6}
            width="4"
            height={4 + i * 3.6}
            rx="0.8"
            fill={i < bars ? (bars <= 1 ? "#ef4444" : bars === 2 ? "#f59e0b" : "#22c55e") : "#334155"}
          />
        ))}
      </svg>
      <span>{dbm == null ? "—" : `${dbm}`}</span>
    </span>
  );
}

export function Sparkline({ values, width = 120, height = 30 }: { values: number[]; width?: number; height?: number }) {
  if (values.length === 0) {
    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="spark" aria-hidden>
        <line x1="0" y1={height - 2} x2={width} y2={height - 2} stroke="#1e293b" strokeDasharray="3 3" />
      </svg>
    );
  }
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = values.length === 1 ? width / 2 : (i / (values.length - 1)) * (width - 6) + 3;
    const y = 4 + ((max - v) / span) * (height - 8);
    return [x, y] as const;
  });
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const last = pts[pts.length - 1];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spark" aria-hidden>
      <path d={d} fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinejoin="round" />
      <circle cx={last[0]} cy={last[1]} r="3" fill="#38bdf8" />
    </svg>
  );
}

export function LockIcon({ color = "currentColor" }: { color?: string }) {
  return (
    <svg viewBox="0 0 16 16" className="lock" aria-hidden>
      <rect x="3" y="7" width="10" height="8" rx="1.5" fill={color} />
      <path d="M5 7 V5 a3 3 0 0 1 6 0 V7" fill="none" stroke={color} strokeWidth="1.8" />
    </svg>
  );
}
