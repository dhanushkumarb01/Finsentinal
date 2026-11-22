// Replay engine: plays a recording (JSONL, one event per line) back using the `t` deltas.
// It only decides WHEN events are emitted; what they mean is the reducer's job.

import type { Ev } from "../types";

export interface ParseResult {
  events: Ev[];
  skipped: number;
}

function isEvent(x: unknown): x is Ev {
  if (!x || typeof x !== "object" || Array.isArray(x)) return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.seq === "number" &&
    Number.isFinite(o.seq) &&
    typeof o.t === "number" &&
    Number.isFinite(o.t) &&
    typeof o.type === "string" &&
    typeof o.run_id === "string"
  );
}

/** Parse JSONL text. Blank lines are ignored; malformed or non-event lines are skipped and counted. */
export function parseJsonl(text: string): ParseResult {
  const events: Ev[] = [];
  let skipped = 0;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    try {
      const obj: unknown = JSON.parse(line);
      if (isEvent(obj)) events.push(obj);
      else skipped++;
    } catch {
      skipped++;
    }
  }
  // Stable sort: recordings are written in seq order, but be robust to small reorders.
  const sorted = events
    .map((e, i) => ({ e, i }))
    .sort((a, b) => (a.e.run_id === b.e.run_id ? a.e.seq - b.e.seq || a.i - b.i : a.i - b.i))
    .map((x) => x.e);
  return { events: sorted, skipped };
}

export interface ReplayCallbacks {
  /** Called with events to apply. `reset` = clear state first (restart / seek). */
  onApply: (evs: Ev[], reset: boolean) => void;
  /** Called whenever position / playing / speed changes. */
  onProgress?: (p: ReplayProgress) => void;
}

export interface ReplayProgress {
  index: number; // number of events applied
  total: number;
  position: number; // seconds since the first event (recording time)
  duration: number; // seconds, first → last event
  playing: boolean;
  speed: number;
  ended: boolean;
}

export interface ReplayOptions {
  /** Longest pause (recording seconds) between two events; longer gaps are shortened. */
  maxGap?: number;
  now?: () => number;
}

export class ReplayEngine {
  readonly events: Ev[];
  private cb: ReplayCallbacks;
  private idx = 0;
  private playing = false;
  private speed = 1;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private remaining = 0; // recording-seconds left to wait before events[idx]
  private scheduledAt = 0; // real ms when the current wait started
  private readonly maxGap: number;
  private readonly now: () => number;
  private readonly t0: number;

  constructor(events: Ev[], cb: ReplayCallbacks, opts: ReplayOptions = {}) {
    this.events = events;
    this.cb = cb;
    this.maxGap = opts.maxGap ?? Infinity;
    this.now = opts.now ?? (() => Date.now());
    this.t0 = events.length ? events[0].t : 0;
  }

  get index(): number {
    return this.idx;
  }
  get isPlaying(): boolean {
    return this.playing;
  }
  get currentSpeed(): number {
    return this.speed;
  }
  get duration(): number {
    return this.events.length ? this.events[this.events.length - 1].t - this.t0 : 0;
  }

  /** Recording time of the playhead (seconds since first event). */
  get position(): number {
    if (this.idx === 0) return 0;
    return this.events[this.idx - 1].t - this.t0;
  }

  progress(): ReplayProgress {
    return {
      index: this.idx,
      total: this.events.length,
      position: this.position,
      duration: this.duration,
      playing: this.playing,
      speed: this.speed,
      ended: this.idx >= this.events.length,
    };
  }

  private emitProgress(): void {
    this.cb.onProgress?.(this.progress());
  }

  private gapBefore(i: number): number {
    if (i <= 0 || i >= this.events.length) return 0;
    const g = this.events[i].t - this.events[i - 1].t;
    return Math.min(Math.max(0, g), this.maxGap);
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private schedule(): void {
    this.clearTimer();
    if (!this.playing) return;
    if (this.idx >= this.events.length) {
      this.playing = false;
      this.emitProgress();
      return;
    }
    this.scheduledAt = this.now();
    this.timer = setTimeout(() => this.tick(), (this.remaining * 1000) / this.speed);
  }

  private tick(): void {
    this.timer = null;
    if (!this.playing) return;
    // Emit this event plus any that share its timestamp.
    const batch: Ev[] = [this.events[this.idx]];
    this.idx++;
    while (this.idx < this.events.length && this.gapBefore(this.idx) === 0) {
      batch.push(this.events[this.idx]);
      this.idx++;
    }
    this.cb.onApply(batch, false);
    this.remaining = this.gapBefore(this.idx);
    this.emitProgress();
    this.schedule();
  }

  play(): void {
    if (this.playing) return;
    if (this.idx >= this.events.length) return;
    this.playing = true;
    this.emitProgress();
    this.schedule();
  }

  pause(): void {
    if (!this.playing) return;
    const elapsed = ((this.now() - this.scheduledAt) / 1000) * this.speed;
    this.remaining = Math.max(0, this.remaining - elapsed);
    this.playing = false;
    this.clearTimer();
    this.emitProgress();
  }

  setSpeed(speed: number): void {
    if (!(speed > 0)) return;
    if (this.playing) {
      const elapsed = ((this.now() - this.scheduledAt) / 1000) * this.speed;
      this.remaining = Math.max(0, this.remaining - elapsed);
      this.speed = speed;
      this.schedule();
    } else {
      this.speed = speed;
    }
    this.emitProgress();
  }

  restart(): void {
    this.seekIndex(0);
  }

  /** Jump so that exactly the first `n` events are applied (state is rebuilt from scratch). */
  seekIndex(n: number): void {
    const target = Math.max(0, Math.min(this.events.length, Math.floor(n)));
    this.clearTimer();
    this.idx = target;
    this.cb.onApply(this.events.slice(0, target), true);
    this.remaining = this.gapBefore(this.idx);
    this.emitProgress();
    if (this.playing) this.schedule();
  }

  /** Jump to a recording time (seconds since first event): every event with t ≤ that time is applied. */
  seek(seconds: number): void {
    let n = 0;
    while (n < this.events.length && this.events[n].t - this.t0 <= seconds) n++;
    this.seekIndex(n);
  }

  dispose(): void {
    this.playing = false;
    this.clearTimer();
  }
}
