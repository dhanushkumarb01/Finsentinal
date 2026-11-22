"""One command to run the live federated-learning demo.

    demo\\.venv\\Scripts\\python demo\\run_demo.py                 # live, virtual devices + Wokwi tunnel
    demo\\.venv\\Scripts\\python demo\\run_demo.py --wokwi off     # virtual devices only (no internet needed)
    demo\\.venv\\Scripts\\python demo\\run_demo.py --replay demo\\recordings\\golden.jsonl   # safety net

Starts and supervises: FinSentinel (own SQLite DB) → coordinator → Caddy (TLS) → Toxiproxy (one
link per virtual phone) → Mission Control → 6 virtual phones → public tunnel + signed beacon for
the Wokwi boards → optional FinSentinel UI and Wazuh forwarder. Ctrl+C stops everything it started.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))

import httpx  # noqa: E402

from fldemo import keys as K  # noqa: E402
from fldemo.paths import FINSENTINEL_DIR, RECORDINGS_DIR, STATE_DIR  # noqa: E402
from fldemo.settings import (  # noqa: E402
    CADDY_HTTP_PORT, CADDY_TLS_PORT, CA_CERT, COORD_PORT, EXE, FORWARDER_PORT, FS_PORT, FS_UI_PORT, LOG_DIR,
    MC_DIST, MC_PORT, SERVER_CERT, SERVER_KEY, TOXI_API_PORT, TUNNEL_URL_FILE, load_devices, load_secrets,
)

PY = sys.executable
IS_WIN = os.name == "nt"
GREEN, RED, YELLOW, BOLD, END = ("\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m")
if IS_WIN:
    os.system("")  # enable ANSI colours in the Windows console
for _stream in (sys.stdout, sys.stderr):  # ✓/✗ must never crash a cp1252 console or a redirect
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def say(msg: str, colour: str = "") -> None:
    print(f"{colour}{msg}{END if colour else ''}", flush=True)


# =============================================================================== processes
class Proc:
    def __init__(self, name: str, cmd: list[str], *, cwd: Path | None = None, env: dict | None = None,
                 restart: bool = False, critical: bool = True, capture: bool = False) -> None:
        self.name, self.cmd, self.cwd, self.restart, self.critical = name, cmd, cwd, restart, critical
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", **(env or {})}
        self.log_path = LOG_DIR / f"{name}.log"
        self.capture = capture
        self.p: subprocess.Popen | None = None
        self.restarts = 0

    def start(self) -> None:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0
        if self.capture:
            self.p = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                                      encoding="utf-8", errors="replace", creationflags=flags)
        else:
            log = open(self.log_path, "a", encoding="utf-8")
            log.write(f"\n===== {time.ctime()} starting {self.cmd}\n")
            log.flush()
            self.p = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, creationflags=flags)

    def alive(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def stop(self) -> None:
        if not self.p or self.p.poll() is not None:
            return
        if IS_WIN:  # terminate this process tree only (by PID — never by image name)
            subprocess.run(["taskkill", "/PID", str(self.p.pid), "/T", "/F"], capture_output=True)
        else:
            self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()


def wait_http(url: str, *, timeout: float = 60, verify=False, ok=(200,)) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=3, verify=verify).status_code in ok:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


# =============================================================================== static site
def serve_static(root: Path, config: dict, port: int) -> ThreadingHTTPServer:
    extra = {"/config.json": json.dumps(config).encode()}
    femnist = DEMO / "config" / "femnist.json"

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path in extra:
                return self._send(extra[path], "application/json")
            if path == "/femnist.json" and femnist.exists():
                return self._send(femnist.read_bytes(), "application/json")
            if path.startswith("/recordings/"):
                f = (RECORDINGS_DIR / Path(path).name)
                if f.exists():
                    return self._send(f.read_bytes(), "application/x-ndjson")
            if path.startswith("/replay-file/") and config.get("_replay_path"):
                return self._send(Path(config["_replay_path"]).read_bytes(), "application/x-ndjson")
            target = root / path.lstrip("/")
            if path == "/" or not target.exists():
                self.path = "/index.html"
            return super().do_GET()

        def _send(self, data: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(H, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# =============================================================================== Caddy
def caddyfile() -> Path:
    routes = f"""
    @fs path /api/v1/updates /api/v1/health
    handle @fs {{
        reverse_proxy 127.0.0.1:{FS_PORT}
    }}
    @fl path /fl/round /fl/model/* /fl/pubkey /fl/status /fl/report
    handle @fl {{
        reverse_proxy 127.0.0.1:{COORD_PORT}
    }}
    handle {{
        respond "not found" 404
    }}"""
    text = f"""{{
    admin off
    auto_https off
    persist_config off
}}

https://localhost:{CADDY_TLS_PORT}, https://127.0.0.1:{CADDY_TLS_PORT} {{
    bind 127.0.0.1
    tls "{SERVER_CERT.as_posix()}" "{SERVER_KEY.as_posix()}"
{routes}
}}

http://:{CADDY_HTTP_PORT} {{
    bind 127.0.0.1
{routes}
}}
"""
    p = STATE_DIR / "Caddyfile"
    p.write_text(text)
    return p


# =============================================================================== tunnel
class Tunnel(threading.Thread):
    """Keeps a public URL pointing at Caddy's plain-HTTP origin; tells the coordinator on every change."""

    def __init__(self, kind: str, admin_token: str) -> None:
        super().__init__(daemon=True)
        self.kind, self.admin_token = kind, admin_token
        self.stop_flag = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None

    def _cmd(self) -> tuple[list[str], str]:
        if self.kind == "cloudflare":
            return ([str(EXE["cloudflared"]), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{CADDY_HTTP_PORT}"],
                    r"https://[a-z0-9-]+\.trycloudflare\.com")
        return (["ssh", "-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={'NUL' if IS_WIN else '/dev/null'}",
                 "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "ExitOnForwardFailure=yes",
                 "-R", f"80:127.0.0.1:{CADDY_HTTP_PORT}", "nokey@localhost.run"],
                r"https://[a-z0-9.-]+\.lhr\.life")

    def publish(self, url: str | None) -> None:
        self.url = url
        TUNNEL_URL_FILE.write_text(url or "")
        try:
            httpx.post(f"http://127.0.0.1:{COORD_PORT}/fl/admin/tunnel", json={"url": url}, timeout=15,
                       headers={"Authorization": f"Bearer {self.admin_token}"})
        except httpx.HTTPError:
            pass
        if url:
            say(f"  ↳ Wokwi tunnel: {url}  (published to the boards' signed beacon)", GREEN)

    def run(self) -> None:
        cmd, pattern = self._cmd()
        log = open(LOG_DIR / "tunnel.log", "a", encoding="utf-8")
        while not self.stop_flag.is_set():
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0)
            except OSError as exc:
                say(f"  ✗ tunnel could not start ({exc}); boards will not connect", RED)
                return
            for line in self.proc.stdout:  # type: ignore[union-attr]
                log.write(line)
                log.flush()
                m = re.search(pattern, line)
                if m and m.group(0) != self.url and "api.trycloudflare.com" not in m.group(0):
                    self.publish(m.group(0))
            if self.stop_flag.is_set():
                break
            self.publish(None)
            say("  ! tunnel dropped — reconnecting in 3 s", YELLOW)
            time.sleep(3)

    def stop(self) -> None:
        self.stop_flag.set()
        if self.proc and self.proc.poll() is None:
            if IS_WIN:
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            else:
                self.proc.terminate()


# =============================================================================== main
def port_in_use(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def preflight(args) -> list[str]:
    problems = []
    busy = [p for p in ((MC_PORT,) if args.replay else (FS_PORT, COORD_PORT, CADDY_TLS_PORT, MC_PORT))
            if port_in_use(p)]
    if busy:
        problems.append(f"port(s) {busy} already in use — another copy of the demo (or FinSentinel's own dev "
                        "server on 8000) is running. Stop it first (Ctrl+C in its window).")
    if not (DEMO / ".data" / "extract" / "UCI HAR Dataset").exists():
        problems.append("UCI HAR dataset missing — run: python demo/setup.py")
    if not (MC_DIST / "index.html").exists():
        if args.replay:
            problems.append("Mission Control not built — run: python demo/setup.py (or npm run build in demo/mission-control)")
        else:
            say("! Mission Control not built yet — backend will run, the big screen will not "
                "(npm run build in demo/mission-control)", YELLOW)
    if not args.replay:
        for tool in ("caddy", "toxiproxy"):
            if not EXE[tool].exists():
                problems.append(f"{EXE[tool].name} missing in demo/tools — run: python demo/setup.py")
        if args.wokwi == "cloudflare" and not EXE["cloudflared"].exists():
            problems.append("cloudflared.exe missing in demo/tools")
        if args.wokwi == "tunnel" and not shutil.which("ssh"):
            problems.append("ssh not found (needed for the localhost.run tunnel) — use --wokwi cloudflare or off")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the live FL demo")
    ap.add_argument("--replay", help="play a recording in Mission Control (no backend needed)")
    ap.add_argument("--wokwi", choices=["tunnel", "cloudflare", "gateway", "off"], default="tunnel",
                    help="how Wokwi boards reach the laptop (tunnel = localhost.run)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--round-seconds", type=float, default=None, help="default 25 (90 with Wokwi boards)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--wazuh", action="store_true",
                    help="start Wazuh (WSL) and let it make every security decision in real time")
    ap.add_argument("--no-devices", action="store_true", help="do not start the virtual phones")
    ap.add_argument("--no-fs-ui", action="store_true", help=argparse.SUPPRESS)  # kept for old commands
    ap.add_argument("--fs-ui", action="store_true",
                    help="also start FinSentinel's own web UI (not needed: Mission Control's Security Console replaces it)")
    args = ap.parse_args()

    problems = preflight(args)
    if problems:
        for p in problems:
            say(f"✗ {p}", RED)
        return 1

    procs: list[Proc] = []
    tunnel: Tunnel | None = None
    stopping = threading.Event()

    def shutdown(*_):
        if stopping.is_set():
            return
        stopping.set()
        say("\nStopping everything…", YELLOW)
        if tunnel:
            tunnel.stop()
        for pr in reversed(procs):
            pr.stop()
        say("All stopped. Bye.", GREEN)

    # ---------------------------------------------------------------- replay mode
    if args.replay:
        rec = Path(args.replay).resolve()
        if not rec.exists():
            say(f"✗ recording not found: {rec}", RED)
            return 1
        cfg = {"mode": "replay", "replay_url": f"/replay-file/{rec.name}", "_replay_path": str(rec),
               "ws_url": "", "http_url": "", "admin_token": "", "finsentinel_ui": "",
               "femnist_url": "/femnist.json"}
        serve_static(MC_DIST, cfg, MC_PORT)
        say(f"{BOLD}REPLAY READY{END}  Mission Control → http://localhost:{MC_PORT}   (Ctrl+C to stop)", GREEN)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    # ---------------------------------------------------------------- live
    devices = load_devices()
    virtual = [d["id"] for d in devices if d["kind"] == "virtual"]
    wokwi_on = args.wokwi != "off"
    round_s = args.round_seconds or (90.0 if wokwi_on else 25.0)
    say(f"{BOLD}Live FL demo{END} — {args.rounds} rounds × {round_s:.0f}s, "
        f"{len(virtual)} virtual phones, Wokwi: {args.wokwi}")
    sec = K.ensure_secrets()
    K.ensure_device_keys([d["id"] for d in devices])
    K.ensure_tls()
    K.ensure_beacon_topic()
    if (DEMO / "scripts" / "gen_firmware.py").exists():
        subprocess.run([PY, str(DEMO / "scripts" / "gen_firmware.py")], capture_output=True)
    for f in ("finsentinel_demo.db", "finsentinel_demo.db-wal", "finsentinel_demo.db-shm"):
        (STATE_DIR / f).unlink(missing_ok=True)

    def up(pr: Proc, url: str, label: str, **kw) -> bool:
        procs.append(pr)
        pr.start()
        ok = wait_http(url, **kw)
        say(f"  {'✓' if ok else '✗'} {label}" + ("" if ok else f" did not come up — see {pr.log_path}"),
            GREEN if ok else RED)
        return ok

    def component(name: str, status: str, detail: str = "") -> None:
        try:
            httpx.post(f"http://127.0.0.1:{COORD_PORT}/fl/admin/component", timeout=5,
                       headers={"Authorization": f"Bearer {sec['admin_token']}"},
                       json={"name": name, "status": status, "detail": detail})
        except httpx.HTTPError:
            pass

    engine = "coordinator"
    if args.wazuh:
        say("  … starting Wazuh in WSL (first start can take several minutes)")
        wsl_script = "/mnt/" + DEMO.drive[0].lower() + DEMO.as_posix()[2:] + "/wazuh/wazuh.sh"
        r = subprocess.run(["wsl", "-d", "Ubuntu", "bash", wsl_script, "up"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
        (LOG_DIR / "wazuh.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode == 0:
            engine = "wazuh"
            say("  ✓ Wazuh manager up — Wazuh makes every security decision", GREEN)
        else:
            say("  ✗ Wazuh did not come up (see demo/.state/logs/wazuh.log). Continuing with the coordinator "
                "as decision engine — Mission Control will say so. One-time setup: "
                "wsl -d Ubuntu bash " + wsl_script + " setup", RED)

    try:
        siem_env = {}
        if args.wazuh:
            siem_log = STATE_DIR / "siem" / "finsentinel-siem.jsonl"
            siem_log.parent.mkdir(parents=True, exist_ok=True)
            siem_env = {"FINSENTINEL_SIEM_LOG": str(siem_log)}
            fwd = Proc("siem-forwarder", [PY, str(FINSENTINEL_DIR / "scripts" / "siem_forwarder.py")],
                       cwd=FINSENTINEL_DIR, critical=False,
                       env={**siem_env, "FINSENTINEL_AUDIT_WEBHOOK_TOKEN": sec["forwarder_token"],
                            "FINSENTINEL_FORWARDER_PORT": str(FORWARDER_PORT)})
            procs.append(fwd)
            fwd.start()
        if not up(Proc("finsentinel", [PY, "-m", "fldemo.fs_server", "--rounds", str(args.rounds), "--fresh"]
                       + (["--defer-decisions"] if engine == "wazuh" else []),
                       cwd=DEMO, env=siem_env),
                  f"http://127.0.0.1:{FS_PORT}/api/v1/health", "FinSentinel (security gate)", timeout=90):
            raise SystemExit(2)
        coord_cmd = [PY, "-m", "fldemo.coordinator", "--rounds", str(args.rounds), "--round-seconds", str(round_s),
                     "--epochs", str(args.epochs), "--min-devices", "0" if args.no_devices else str(len(virtual)),
                     "--decision-engine", engine] + (["--wokwi"] if wokwi_on else []) + (["--wazuh"] if args.wazuh else [])
        if not up(Proc("coordinator", coord_cmd, cwd=DEMO), f"http://127.0.0.1:{COORD_PORT}/fl/state",
                  "Coordinator (rounds, signed models, event stream)", timeout=60):
            raise SystemExit(2)
        component("finsentinel", "up")
        component("coordinator", "up")
        if not up(Proc("caddy", [str(EXE["caddy"]), "run", "--config", str(caddyfile()), "--adapter", "caddyfile"],
                       cwd=STATE_DIR, env={"XDG_DATA_HOME": str(STATE_DIR / "caddy-data"),
                                           "XDG_CONFIG_HOME": str(STATE_DIR / "caddy-config")}),
                  f"https://localhost:{CADDY_TLS_PORT}/api/v1/health", "Caddy HTTPS front door",
                  verify=str(CA_CERT), timeout=30):
            raise SystemExit(2)
        component("tls-proxy", "up")
        if not up(Proc("toxiproxy", [str(EXE["toxiproxy"]), "-host", "127.0.0.1", "-port", str(TOXI_API_PORT)]),
                  f"http://127.0.0.1:{TOXI_API_PORT}/version", "Toxiproxy (real network conditions)", timeout=20):
            raise SystemExit(2)
        from fldemo import toxi
        for d in devices:
            if d["kind"] == "virtual":
                toxi.ensure_link(d["id"])
                toxi.apply_profile(d["id"], d["network"])
        component("toxiproxy", "up")

        mc_cfg = {"mode": "live", "ws_url": f"ws://127.0.0.1:{COORD_PORT}/fl/ws",
                  "http_url": f"http://127.0.0.1:{COORD_PORT}", "admin_token": sec["admin_token"],
                  "replay_url": "/recordings/golden.jsonl", "finsentinel_ui": f"http://localhost:{FS_UI_PORT}",
                  "femnist_url": "/femnist.json"}
        serve_static(MC_DIST, mc_cfg, MC_PORT)
        say(f"  ✓ Mission Control on http://localhost:{MC_PORT}", GREEN)

        if args.fs_ui and (FINSENTINEL_DIR / "frontend" / "node_modules").exists():
            npm = shutil.which("npm.cmd" if IS_WIN else "npm") or "npm"
            ui = Proc("finsentinel-ui", [npm, "run", "dev", "--", "--port", str(FS_UI_PORT), "--strictPort"],
                      cwd=FINSENTINEL_DIR / "frontend", critical=False)
            procs.append(ui)
            ui.start()
            say(f"  ✓ FinSentinel UI starting on http://localhost:{FS_UI_PORT}", GREEN)
        elif args.fs_ui:
            say("  - FinSentinel UI not installed (optional: npm ci in finsentinel/frontend)", YELLOW)

        if not args.no_devices:
            for did in virtual:
                pr = Proc(f"device-{did}", [PY, "-m", "fldemo.device", "--device", did], cwd=DEMO,
                          restart=True, critical=False)
                procs.append(pr)
                pr.start()
            say(f"  ✓ {len(virtual)} virtual phones started", GREEN)

        if args.wokwi in ("tunnel", "cloudflare"):
            tunnel = Tunnel(args.wokwi, sec["admin_token"])
            tunnel.start()
        elif args.wokwi == "gateway":
            httpx.post(f"http://127.0.0.1:{COORD_PORT}/fl/admin/tunnel", timeout=15,
                       headers={"Authorization": f"Bearer {sec['admin_token']}"},
                       json={"url": f"http://host.wokwi.internal:{CADDY_HTTP_PORT}"})

        say(f"\n{BOLD}{GREEN}ALL SYSTEMS READY{END}")
        say(f"  Mission Control  → http://localhost:{MC_PORT}   (Live · Security Console · Research)")
        say(f"  Decision engine  → {'Wazuh (real-time SIEM decisions)' if engine == 'wazuh' else 'coordinator (Wazuh off)'}")
        if engine == "wazuh":
            say("  Wazuh dashboard  → https://localhost   (admin / see finsentinel/deploy/wazuh/.env.example)")
        if args.fs_ui:
            say(f"  FinSentinel UI   → http://localhost:{FS_UI_PORT}   operator token in {STATE_DIR / 'secrets.json'}")
        say(f"  Attacks (CLI)    → demo\\.venv\\Scripts\\python demo\\attack.py <scene>")
        say(f"  Logs             → {LOG_DIR}")
        say("  Ctrl+C stops everything.\n")

        # ------------------------------------------------------------ supervise
        while True:
            time.sleep(2)
            for pr in procs:
                if pr.alive() or stopping.is_set():
                    continue
                if pr.restart and pr.restarts < 20:
                    pr.restarts += 1
                    say(f"  ! {pr.name} exited — restarting ({pr.restarts})", YELLOW)
                    pr.start()
                elif pr.critical and not getattr(pr, "reported", False):
                    pr.reported = True  # type: ignore[attr-defined]
                    say(f"  ✗ {pr.name} STOPPED — see {pr.log_path}. Use the replay safety net if needed.", RED)
                    component(pr.name, "down", "process exited")
    except (KeyboardInterrupt, SystemExit) as exc:
        shutdown()
        return exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 0
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
