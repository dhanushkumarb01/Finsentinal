import type { ReplayProgress } from "../replay/engine";

function mmss(s: number): string {
  const v = Math.max(0, Math.floor(s));
  return `${Math.floor(v / 60)}:${String(v % 60).padStart(2, "0")}`;
}

export function ReplayBar({
  progress,
  source,
  error,
  skipped,
  onPlay,
  onPause,
  onRestart,
  onSpeed,
  onSeek,
  onFile,
  onUseUrl,
}: {
  progress: ReplayProgress | null;
  source: string;
  error: string | null;
  skipped: number;
  onPlay: () => void;
  onPause: () => void;
  onRestart: () => void;
  onSpeed: (n: number) => void;
  onSeek: (sec: number) => void;
  onFile: (f: File) => void;
  onUseUrl: () => void;
}) {
  const p = progress;
  return (
    <div className="replay-bar">
      <span className="rb-tag">REPLAY</span>
      <button className="rb-btn" disabled={!p || p.total === 0} onClick={() => (p?.playing ? onPause() : onPlay())}>
        {p?.playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <button className="rb-btn" disabled={!p} onClick={onRestart}>
        ⟲ Restart
      </button>
      <div className="rb-speed" role="group" aria-label="Speed">
        {[1, 2, 4, 8].map((n) => (
          <button key={n} className={p?.speed === n ? "on" : ""} onClick={() => onSpeed(n)}>
            {n}×
          </button>
        ))}
      </div>
      <span className="mono small">{mmss(p?.position ?? 0)}</span>
      <input
        className="rb-scrub"
        type="range"
        min={0}
        max={Math.max(0.001, p?.duration ?? 0)}
        step={0.1}
        value={p?.position ?? 0}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-label="Scrub"
      />
      <span className="mono small">{mmss(p?.duration ?? 0)}</span>
      <span className="muted small rb-src" title={source}>
        {p ? `${p.index}/${p.total} events` : ""} · {source}
        {skipped > 0 && ` · ${skipped} bad lines skipped`}
      </span>
      {error && <span className="rb-err">{error}</span>}
      <label className="rb-btn file">
        Load .jsonl
        <input
          type="file"
          accept=".jsonl,.json,.txt,application/json"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = "";
          }}
        />
      </label>
      <button className="rb-btn" onClick={onUseUrl}>
        Use configured recording
      </button>
    </div>
  );
}
