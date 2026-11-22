import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SAMPLE_RAW from "../../public/sample-recording.jsonl?raw";
import type { Ev } from "../types";
import { parseJsonl, ReplayEngine } from "./engine";
import { initialState, reduceAll, type DashState } from "../state/reducer";

const SAMPLE: string = SAMPLE_RAW;

function mk(seq: number, t: number): Ev {
  return { seq, t, run_id: "r", type: "alert", severity: "info", title: `e${seq}`, detail: "", source: "coordinator" };
}
// events at t = 100, 101, 103, 103, 107  (gaps 1, 2, 0, 4)
const EVS = [mk(1, 100), mk(2, 101), mk(3, 103), mk(4, 103), mk(5, 107)];

function harness(events: Ev[] = EVS, opts = {}) {
  const applied: number[] = [];
  let resets = 0;
  const eng = new ReplayEngine(
    events,
    {
      onApply: (evs, reset) => {
        if (reset) {
          resets++;
          applied.length = 0;
        }
        for (const e of evs) applied.push(e.seq);
      },
    },
    opts,
  );
  return { eng, applied, resets: () => resets };
}

describe("parseJsonl", () => {
  it("parses events, skips bad lines and blank lines, sorts by seq", () => {
    const text = [
      JSON.stringify(mk(2, 101)),
      "",
      "not json at all",
      '{"seq": "x", "t": 1, "run_id": "r", "type": "alert"}',
      "[1,2,3]",
      JSON.stringify(mk(1, 100)),
      '{"t": 5, "run_id": "r", "type": "alert"}',
      "   ",
      JSON.stringify(mk(3, 102)) + "\r",
    ].join("\n");
    const r = parseJsonl(text);
    expect(r.events.map((e) => e.seq)).toEqual([1, 2, 3]);
    expect(r.skipped).toBe(4);
  });

  it("parses the bundled sample recording with no bad lines", () => {
    const r = parseJsonl(SAMPLE);
    expect(r.skipped).toBe(0);
    expect(r.events.length).toBeGreaterThan(150);
    expect(r.events[0].type).toBe("run_started");
    for (let i = 1; i < r.events.length; i++) {
      expect(r.events[i].seq).toBe(r.events[i - 1].seq + 1);
      expect(r.events[i].t).toBeGreaterThanOrEqual(r.events[i - 1].t);
    }
  });
});

describe("ReplayEngine timing", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => vi.useRealTimers());

  it("emits the first event immediately, then follows t deltas at 1×", () => {
    const { eng, applied } = harness();
    eng.play();
    vi.advanceTimersByTime(0);
    expect(applied).toEqual([1]);
    vi.advanceTimersByTime(999);
    expect(applied).toEqual([1]);
    vi.advanceTimersByTime(1);
    expect(applied).toEqual([1, 2]);
    vi.advanceTimersByTime(2000); // events 3 and 4 share a timestamp → same batch
    expect(applied).toEqual([1, 2, 3, 4]);
    vi.advanceTimersByTime(3999);
    expect(applied).toEqual([1, 2, 3, 4]);
    vi.advanceTimersByTime(1);
    expect(applied).toEqual([1, 2, 3, 4, 5]);
    expect(eng.isPlaying).toBe(false);
    expect(eng.progress().ended).toBe(true);
  });

  it("scales timing by speed (4× plays 7 s of recording in 1.75 s)", () => {
    const { eng, applied } = harness();
    eng.setSpeed(4);
    eng.play();
    vi.advanceTimersByTime(0);
    vi.advanceTimersByTime(249);
    expect(applied).toEqual([1]);
    vi.advanceTimersByTime(1);
    expect(applied).toEqual([1, 2]);
    vi.advanceTimersByTime(1500);
    expect(applied).toEqual([1, 2, 3, 4, 5]);
  });

  it("changing speed mid-wait keeps the elapsed part", () => {
    const { eng, applied } = harness();
    eng.play();
    vi.advanceTimersByTime(1000); // → events 1,2 ; now waiting 2 s for event 3
    expect(applied).toEqual([1, 2]);
    vi.advanceTimersByTime(1000); // half of the wait done at 1×
    eng.setSpeed(2); // remaining 1 s of recording → 0.5 s real
    vi.advanceTimersByTime(499);
    expect(applied).toEqual([1, 2]);
    vi.advanceTimersByTime(1);
    expect(applied).toEqual([1, 2, 3, 4]);
  });

  it("pause stops emission and resume continues with the remaining wait", () => {
    const { eng, applied } = harness();
    eng.play();
    vi.advanceTimersByTime(1000);
    expect(applied).toEqual([1, 2]);
    vi.advanceTimersByTime(500);
    eng.pause();
    expect(eng.isPlaying).toBe(false);
    vi.advanceTimersByTime(10_000);
    expect(applied).toEqual([1, 2]);
    eng.play();
    vi.advanceTimersByTime(1499);
    expect(applied).toEqual([1, 2]);
    vi.advanceTimersByTime(1);
    expect(applied).toEqual([1, 2, 3, 4]);
  });

  it("seek rebuilds from scratch to a recording time; works while paused and playing", () => {
    const h = harness();
    h.eng.seek(3); // t0+3 → events with t ≤ 103 → 1..4
    expect(h.resets()).toBe(1);
    expect(h.applied).toEqual([1, 2, 3, 4]);
    expect(h.eng.position).toBe(3);
    expect(h.eng.isPlaying).toBe(false);
    h.eng.seek(0.5);
    expect(h.applied).toEqual([1]);
    h.eng.play();
    vi.advanceTimersByTime(500); // remaining gap after a seek is the full gap (1 s)
    expect(h.applied).toEqual([1]);
    vi.advanceTimersByTime(500);
    expect(h.applied).toEqual([1, 2]);
    h.eng.seek(100);
    expect(h.applied).toEqual([1, 2, 3, 4, 5]);
    expect(h.eng.progress().ended).toBe(true);
  });

  it("restart clears and replays from the beginning", () => {
    const h = harness();
    h.eng.play();
    vi.advanceTimersByTime(10_000);
    expect(h.applied).toHaveLength(5);
    h.eng.restart();
    expect(h.applied).toEqual([]);
    h.eng.play();
    vi.advanceTimersByTime(0);
    expect(h.applied).toEqual([1]);
  });

  it("maxGap shortens long idle gaps", () => {
    const h = harness([mk(1, 0), mk(2, 600)], { maxGap: 5 });
    h.eng.play();
    vi.advanceTimersByTime(5000);
    expect(h.applied).toEqual([1, 2]);
  });

  it("replaying the sample through the reducer equals reducing it directly", () => {
    const { events } = parseJsonl(SAMPLE);
    let state: DashState = initialState();
    const eng = new ReplayEngine(events, {
      onApply: (evs, reset) => {
        if (reset) state = initialState();
        state = reduceAll(state, evs);
      },
    });
    eng.setSpeed(8);
    eng.play();
    vi.advanceTimersByTime(60 * 60 * 1000);
    expect(state).toEqual(reduceAll(initialState(), events));
  });
});
