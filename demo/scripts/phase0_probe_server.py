"""Phase-0 probe server: logs what the Wokwi probe board requests, with wall-clock times.

Usage:  demo/.venv/Scripts/python demo/scripts/phase0_probe_server.py [--port 8790]

Then expose it (pick one):
  * Cloudflare quick tunnel:  demo/tools/cloudflared.exe tunnel --url http://127.0.0.1:8790
    and put the printed https://<name>.trycloudflare.com into sketch.ino's PROBE_URL;
  * Wokwi Private Gateway:   PROBE_URL "http://host.wokwi.internal:8790".
Results are appended to demo/.state/probe_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fldemo.paths import STATE_DIR  # noqa: E402

LOG = STATE_DIR / "probe_log.jsonl"
_bench_start: float | None = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        global _bench_start
        now = time.time()
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        rec = {"t": now, "path": u.path, "query": q, "peer": self.client_address[0],
               "via": self.headers.get("cf-connecting-ip") and "cloudflare" or "direct"}
        if u.path.endswith("/bench_start"):
            _bench_start = now
        elif u.path.endswith("/bench_end") and _bench_start is not None:
            wall = now - _bench_start
            epochs = int(q.get("epochs", "1"))
            rec["wall_s"] = round(wall, 3)
            rec["wall_s_per_epoch"] = round(wall / epochs, 3)
            if "sim_ms" in q:
                rec["sim_speed_pct"] = round(100 * int(q["sim_ms"]) / 1000 / wall, 1)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        body = b"hello-from-laptop" if u.path.endswith("/hello") else b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    print(f"probe server on http://127.0.0.1:{args.port}  (log: {LOG})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
