// Live event stream over WebSocket with auto-reconnect + exponential backoff.
// The server replays every event of the run on each (re)connect; duplicates are dropped by
// the reducer (seq), and a changed run_id resets state there too.

import type { Ev } from "../types";

export type ConnStatus = "connecting" | "open" | "lost";

export interface LiveCallbacks {
  onEvents: (evs: Ev[]) => void;
  onStatus: (s: ConnStatus, attempt: number) => void;
}

export function backoffMs(attempt: number): number {
  return Math.min(8000, 500 * 2 ** Math.max(0, attempt));
}

export function decodeMessage(data: unknown): Ev[] {
  if (typeof data !== "string") return [];
  try {
    const obj: unknown = JSON.parse(data);
    const arr = Array.isArray(obj) ? obj : [obj];
    return arr.filter(
      (e): e is Ev =>
        !!e && typeof e === "object" && typeof (e as Ev).seq === "number" && typeof (e as Ev).type === "string",
    );
  } catch {
    return [];
  }
}

export class LiveClient {
  private ws: WebSocket | null = null;
  private attempt = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;

  constructor(
    private url: string,
    private cb: LiveCallbacks,
  ) {}

  start(): void {
    this.closed = false;
    this.connect();
  }

  private connect(): void {
    if (this.closed) return;
    this.cb.onStatus("connecting", this.attempt);
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.cb.onStatus("open", 0);
    };
    ws.onmessage = (m: MessageEvent) => {
      const evs = decodeMessage(m.data);
      if (evs.length) this.cb.onEvents(evs);
    };
    ws.onerror = () => {
      /* onclose follows */
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      if (!this.closed) this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    this.cb.onStatus("lost", this.attempt);
    const delay = backoffMs(this.attempt);
    this.attempt++;
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.connect(), delay);
  }

  stop(): void {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
