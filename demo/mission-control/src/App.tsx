import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { AppConfig, Ev } from "./types";
import { initialState, rootReducer } from "./state/reducer";
import { DEFAULT_CONFIG, loadConfig } from "./config";
import { LiveClient, type ConnStatus } from "./live/socket";
import { parseJsonl, ReplayEngine, type ReplayProgress } from "./replay/engine";
import { Dashboard } from "./components/Dashboard";
import type { Mode } from "./components/Header";
import { ReplayBar } from "./components/ReplayBar";

interface Toast {
  id: number;
  kind: "pending" | "ok" | "error";
  text: string;
}

const REPLAY_MAX_GAP_S = 15; // long idle gaps in a recording are shortened to this

export default function App() {
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [mode, setMode] = useState<Mode>("replay");
  const [state, dispatch] = useReducer(rootReducer, undefined, initialState);
  const [conn, setConn] = useState<ConnStatus>("connecting");
  const [attempt, setAttempt] = useState(0);
  const [progress, setProgress] = useState<ReplayProgress | null>(null);
  const [replaySrc, setReplaySrc] = useState<{ kind: "url"; url: string } | { kind: "file"; name: string; text: string } | null>(
    null,
  );
  const [replayErr, setReplayErr] = useState<string | null>(null);
  const [skipped, setSkipped] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const engineRef = useRef<ReplayEngine | null>(null);
  const toastId = useRef(0);

  useEffect(() => {
    loadConfig().then(({ config }) => {
      setCfg(config);
      setMode(config.mode);
      setReplaySrc({ kind: "url", url: config.replay_url });
    });
  }, []);

  const toast = useCallback((kind: Toast["kind"], text: string) => {
    const id = ++toastId.current;
    setToasts((t) => [...t.slice(-3), { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 9000 : 5000);
  }, []);

  // ---- live mode ----
  useEffect(() => {
    if (!cfg || mode !== "live") return;
    dispatch({ kind: "reset" });
    const client = new LiveClient(cfg.ws_url, {
      onEvents: (evs: Ev[]) => dispatch({ kind: "batch", evs }),
      onStatus: (st, n) => {
        setConn(st);
        setAttempt(n);
      },
    });
    client.start();
    return () => client.stop();
  }, [cfg, mode]);

  // ---- replay mode ----
  useEffect(() => {
    if (!cfg || mode !== "replay" || !replaySrc) return;
    let cancelled = false;
    dispatch({ kind: "reset" });
    setProgress(null);
    setReplayErr(null);
    const textP =
      replaySrc.kind === "file"
        ? Promise.resolve(replaySrc.text)
        : fetch(replaySrc.url, { cache: "no-store" }).then((r) => {
            if (!r.ok) throw new Error(`could not load ${replaySrc.url} (HTTP ${r.status})`);
            return r.text();
          });
    textP
      .then((text) => {
        if (cancelled) return;
        const { events, skipped: bad } = parseJsonl(text);
        setSkipped(bad);
        if (!events.length) {
          setReplayErr("recording contains no valid events");
          return;
        }
        const eng = new ReplayEngine(
          events,
          {
            onApply: (evs, reset) => {
              if (reset) dispatch({ kind: "reset" });
              if (evs.length) dispatch({ kind: "batch", evs });
            },
            onProgress: (p) => setProgress(p),
          },
          { maxGap: REPLAY_MAX_GAP_S },
        );
        engineRef.current = eng;
        setProgress(eng.progress());
        // Rehearsal helpers: ?seek=<seconds> jumps into the recording, ?paused=1 starts paused.
        const q = new URLSearchParams(window.location.search);
        const seekTo = Number(q.get("seek"));
        if (q.has("seek") && Number.isFinite(seekTo)) eng.seek(seekTo);
        if (q.get("paused") !== "1") eng.play();
      })
      .catch((e: unknown) => {
        if (!cancelled) setReplayErr(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      engineRef.current?.dispose();
      engineRef.current = null;
    };
  }, [cfg, mode, replaySrc]);

  if (!cfg) return <div className="boot">Loading Mission Control…</div>;

  const banner =
    mode === "live" && conn !== "open" ? (
      <div className={`banner ${conn === "lost" ? "banner-bad" : "banner-warn"}`}>
        {conn === "lost"
          ? `Connection lost to ${cfg.ws_url} — reconnecting (attempt ${attempt})…`
          : `Connecting to ${cfg.ws_url}…`}
      </div>
    ) : null;

  const bottomBar =
    mode === "replay" ? (
      <ReplayBar
        progress={progress}
        source={replaySrc ? (replaySrc.kind === "file" ? replaySrc.name : replaySrc.url) : ""}
        error={replayErr}
        skipped={skipped}
        onPlay={() => engineRef.current?.play()}
        onPause={() => engineRef.current?.pause()}
        onRestart={() => {
          engineRef.current?.restart();
          engineRef.current?.play();
        }}
        onSpeed={(n) => engineRef.current?.setSpeed(n)}
        onSeek={(sec) => engineRef.current?.seek(sec)}
        onFile={(f) => f.text().then((text) => setReplaySrc({ kind: "file", name: f.name, text }))}
        onUseUrl={() => setReplaySrc({ kind: "url", url: cfg.replay_url || DEFAULT_CONFIG.replay_url })}
      />
    ) : null;

  return (
    <>
      <Dashboard
        s={state}
        cfg={cfg}
        mode={mode}
        conn={mode === "replay" ? "replay" : conn}
        onMode={(m) => {
          if (m !== mode) {
            setConn("connecting");
            setMode(m);
          }
        }}
        toast={toast}
        bottomBar={bottomBar}
        banner={banner}
      />
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`}>
            {t.text}
          </div>
        ))}
      </div>
    </>
  );
}
