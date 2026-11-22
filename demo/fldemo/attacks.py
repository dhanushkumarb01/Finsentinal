"""Attack scenes — each one sends REAL traffic at the REAL server and reports what actually happened.

Used by the coordinator's admin endpoint (Mission Control buttons) and by demo/attack.py (CLI).

  poison       the target device's next update is boosted ×8 (it is honestly signed by the device's
               real key) → FinSentinel's calibrated anomaly rules fire → quarantined
  replay       re-send a captured, previously accepted submission byte-for-byte      → 409
  tamper       captured submission with one payload byte flipped, envelope unchanged  → 422
  forge        update for the target signed with an attacker key under its key id    → 401
  mitm         (a) passive tap relays real TLS traffic and searches it for plaintext → none;
               (b) interceptor presents a certificate not issued by the demo CA       → device refuses
  drift        target's local data shifts (label shift); FinSentinel sees an honest update;
               the device's own drift score drives drift-causal dropout
  weak_signal  Toxiproxy degrades the target's real network link (latency + 3 KB/s)
  low_battery  target's simulated battery set to 8 %;  offline  target goes offline
  restore      clears directives and restores the target's normal network profile
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import socket
import ssl
import threading
import time
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fldemo import MODEL_VERSION
from fldemo.settings import CA_CERT, CADDY_TLS_PORT, TLS_DIR
from fldemo.wire import build_npz_stored, make_envelope, submission_body

GATEWAY = f"https://localhost:{CADDY_TLS_PORT}"
DEFAULT_TARGET = {"poison": "phone-03", "forge": "phone-02", "drift": "phone-05", "weak_signal": "phone-02",
                  "low_battery": "phone-06", "offline": "phone-04"}


def _ctx() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(CA_CERT))


def _latest_accepted(coord) -> dict | None:
    recs = [r for r in coord.updates.values() if r["http_status"] == 200]
    return max(recs, key=lambda r: r["t"]) if recs else None


async def _post_update(coord, body: dict) -> tuple[int, Any]:
    gateway = getattr(coord, "gateway", GATEWAY)
    async with httpx.AsyncClient(verify=_ctx() if gateway.startswith("https") else False, timeout=20) as c:
        r = await c.post(f"{gateway}/api/v1/updates", json=body)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"text": r.text}


async def _gate_for_attack(coord, scene: str, target: str, update_id: str, rnd: int, status: int, body: Any,
                           payload: bytes, env: dict) -> None:
    from fldemo.coordinator import detail_text, gate_checks
    detail = detail_text(body)
    coord.bus.emit("update_submitted", device=target, round=rnd, update_id=f"{update_id}#{scene}",
                   payload_bytes=len(payload), payload_sha256=env.get("payload_sha256", ""), signed=True,
                   path="toxiproxy")
    coord.bus.emit("gate_result", device=target, round=rnd, update_id=f"{update_id}#{scene}", http_status=status,
                   checks=gate_checks(status, detail, None), risk_score=None, severity=None,
                   approval_status=None, reasons=[], detail=detail[:300], source="attack")
    outcome = "blocked" if status >= 400 else "accepted"
    coord.bus.emit("attack", scene=scene, phase="result", target=target, http_status=status, outcome=outcome,
                   detail=f"FinSentinel answered {status}: {detail[:200]}")


# ----------------------------------------------------------------------------- scenes
async def replay(coord, target: str | None) -> dict:
    rec = _latest_accepted(coord)
    if not rec:
        return {"ok": False, "detail": "no accepted update captured yet — wait for a round"}
    body = rec["body"]
    coord.bus.emit("attack", scene="replay", phase="launched", target=rec["device"],
                   detail=f"re-sending {rec['device']}'s captured update {rec['update_id']} unchanged")
    status, resp = await _post_update(coord, body)
    await _gate_for_attack(coord, "replay", rec["device"], rec["update_id"], rec["round"], status, resp,
                           rec["payload"], body["envelope"])
    return {"ok": True, "http_status": status, "response": resp}


async def tamper(coord, target: str | None) -> dict:
    rec = _latest_accepted(coord)
    if not rec:
        return {"ok": False, "detail": "no accepted update captured yet — wait for a round"}
    payload = bytearray(rec["payload"])
    i = len(payload) // 2
    payload[i] ^= 0x40   # nudge one weight byte
    body = {"envelope": copy.deepcopy(rec["body"]["envelope"]),
            "payload_base64": base64.b64encode(bytes(payload)).decode()}
    body["envelope"]["update_id"] = rec["update_id"][:-8] + "7a3f0c11"  # fresh id so it is not just a replay
    coord.bus.emit("attack", scene="tamper", phase="launched", target=rec["device"],
                   detail=f"changed byte {i} of {rec['device']}'s weights in transit, kept its signed envelope")
    status, resp = await _post_update(coord, body)
    await _gate_for_attack(coord, "tamper", rec["device"], body["envelope"]["update_id"], rec["round"], status,
                           resp, bytes(payload), body["envelope"])
    return {"ok": True, "http_status": status, "response": resp}


async def forge(coord, target: str | None) -> dict:
    target = target or DEFAULT_TARGET["forge"]
    attacker = Ed25519PrivateKey.generate()
    payload = build_npz_stored(coord.global_state)
    env = make_envelope(update_id=f"{target}-r{max(coord.round, 1):04d}-f0f0f0f0", client_id=target,
                        model_version=MODEL_VERSION, round_number=max(coord.round, 1), sample_count=300,
                        payload=payload, signing_key_id=f"{target}-key", private_key=attacker)
    coord.bus.emit("attack", scene="forge", phase="launched", target=target,
                   detail=f"impersonating {target}: update signed with an attacker's key under {target}-key")
    status, resp = await _post_update(coord, submission_body(env, payload))
    await _gate_for_attack(coord, "forge", target, env["update_id"], max(coord.round, 1), status, resp, payload, env)
    return {"ok": True, "http_status": status, "response": resp}


def _relay(listen: socket.socket, captured: bytearray, stop: threading.Event, upstream_port: int) -> None:
    """Passive tap: forward bytes both ways to Caddy, keep a copy of everything."""
    listen.settimeout(5)
    try:
        cli, _ = listen.accept()
    except OSError:
        return
    up = socket.create_connection(("127.0.0.1", upstream_port), timeout=5)

    def pump(a, b):
        try:
            while not stop.is_set():
                data = a.recv(65536)
                if not data:
                    break
                captured.extend(data)
                b.sendall(data)
        except OSError:
            pass
    threads = [threading.Thread(target=pump, args=(cli, up), daemon=True),
               threading.Thread(target=pump, args=(up, cli), daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    cli.close()
    up.close()


def _fake_server(listen: socket.socket, stop: threading.Event, result: dict) -> None:
    """Active MITM: presents a self-signed certificate that the demo CA did not issue."""
    from fldemo.selfsigned import make_untrusted_cert
    cert, key = make_untrusted_cert(TLS_DIR)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    listen.settimeout(5)
    try:
        cli, _ = listen.accept()
        with ctx.wrap_socket(cli, server_side=True) as s:
            s.recv(1024)
            result["server_got_data"] = True
    except (OSError, ssl.SSLError) as exc:
        result["server_error"] = type(exc).__name__


def _mitm_sync(upstream_port: int = CADDY_TLS_PORT) -> dict:
    out: dict = {}
    # (a) passive tap on a real TLS session carrying a real device-style request
    secret_marker = f"envelope-marker-{int(time.time())}"
    captured, stop = bytearray(), threading.Event()
    ls = socket.socket()
    ls.bind(("127.0.0.1", 0))
    ls.listen(1)
    port = ls.getsockname()[1]
    t = threading.Thread(target=_relay, args=(ls, captured, stop, upstream_port), daemon=True)
    t.start()
    try:
        with httpx.Client(verify=_ctx(), timeout=10) as c:
            r = c.post(f"https://localhost:{port}/api/v1/updates",
                       json={"envelope": {"client_id": secret_marker}, "payload_base64": "AA=="})
        out["tap_request_status"] = r.status_code
    except httpx.HTTPError as exc:
        out["tap_request_status"] = f"error {type(exc).__name__}"
    stop.set()
    t.join(timeout=5)
    ls.close()
    out["tap_bytes_captured"] = len(captured)
    out["tap_plaintext_found"] = secret_marker.encode() in captured or b"client_id" in captured
    # (b) active interceptor with an untrusted certificate
    ls2 = socket.socket()
    ls2.bind(("127.0.0.1", 0))
    ls2.listen(1)
    port2 = ls2.getsockname()[1]
    res: dict = {}
    t2 = threading.Thread(target=_fake_server, args=(ls2, threading.Event(), res), daemon=True)
    t2.start()
    try:
        with httpx.Client(verify=_ctx(), timeout=10) as c:
            c.get(f"https://localhost:{port2}/fl/round")
        out["device_refused"] = False
    except httpx.ConnectError as exc:
        out["device_refused"] = "CERTIFICATE_VERIFY_FAILED" in str(exc) or "certificate" in str(exc).lower()
        out["device_error"] = str(exc)[:160]
    t2.join(timeout=5)
    ls2.close()
    out["interceptor"] = res
    return out


async def mitm(coord, target: str | None) -> dict:
    coord.bus.emit("attack", scene="mitm", phase="launched", target=target,
                   detail="eavesdropper taps the encrypted link, then tries to impersonate the server")
    out = await asyncio.to_thread(_mitm_sync)
    coord.bus.emit("attack", scene="mitm", phase="result", target=target, outcome="observed",
                   detail=f"tap captured {out['tap_bytes_captured']} bytes; readable data found: "
                          f"{'YES' if out['tap_plaintext_found'] else 'none (TLS ciphertext only)'}")
    coord.bus.emit("attack", scene="mitm", phase="result", target=target,
                   outcome="refused" if out.get("device_refused") else "accepted",
                   detail="device refused the impostor's certificate (not signed by the pinned CA)"
                   if out.get("device_refused") else "WARNING: device accepted an untrusted certificate")
    return {"ok": True, **out}


async def poison(coord, target: str | None) -> dict:
    target = target or DEFAULT_TARGET["poison"]
    coord.sim.setdefault(target, {})["hacked"] = True
    coord.attack_watch[target] = "poison"
    coord.bus.emit("attack", scene="poison", phase="launched", target=target,
                   detail=f"{target} is compromised: its next update is boosted ×8 (still signed with its real key)")
    return {"ok": True, "target": target, "detail": "takes effect in the device's next round"}


async def drift(coord, target: str | None) -> dict:
    target = target or DEFAULT_TARGET["drift"]
    coord.sim.setdefault(target, {})["drift"] = True
    coord.attack_watch[target] = "drift"
    coord.bus.emit("attack", scene="drift", phase="launched", target=target,
                   detail=f"{target}'s user changes behaviour (label shift): the device stays honest")
    return {"ok": True, "target": target}


async def weak_signal(coord, target: str | None) -> dict:
    target = target or DEFAULT_TARGET["weak_signal"]
    ok = await coord.set_network(target, "weak")
    coord.bus.emit("attack", scene="weak_signal", phase="launched", target=target,
                   detail=f"{target}'s network link degraded (1.5 s latency, 3 KB/s) via Toxiproxy" if ok
                   else "Toxiproxy not reachable — network unchanged")
    return {"ok": ok, "target": target}


async def low_battery(coord, target: str | None) -> dict:
    target = target or DEFAULT_TARGET["low_battery"]
    coord.sim.setdefault(target, {}).update(battery=8, battery_at=time.time())
    coord.bus.emit("attack", scene="low_battery", phase="launched", target=target,
                   detail=f"{target}'s battery drops to 8 %")
    return {"ok": True, "target": target}


async def offline(coord, target: str | None) -> dict:
    target = target or DEFAULT_TARGET["offline"]
    coord.sim.setdefault(target, {})["offline"] = True
    coord.bus.emit("attack", scene="offline", phase="launched", target=target, detail=f"{target} goes offline")
    return {"ok": True, "target": target}


async def restore(coord, target: str | None) -> dict:
    targets = [target] if target else [d for d, v in coord.devices.items() if v["kind"] == "virtual"]
    for t in targets:
        coord.sim[t] = {"restore": time.time()}
        coord.attack_watch.pop(t, None)
        await coord.set_network(t, coord.devices[t]["network"])
    coord.bus.emit("attack", scene="restore", phase="launched", target=target,
                   detail=f"restored {', '.join(targets)} to normal")
    return {"ok": True, "targets": targets}


SCENES = {"poison": poison, "replay": replay, "tamper": tamper, "forge": forge, "mitm": mitm, "drift": drift,
          "weak_signal": weak_signal, "low_battery": low_battery, "offline": offline, "restore": restore}


async def run_attack(coord, scene: str, target: str | None) -> dict:
    fn = SCENES.get(scene)
    if not fn:
        return {"ok": False, "detail": f"unknown scene {scene!r}; choose from {sorted(SCENES)}"}
    if target is not None and target not in coord.devices:
        return {"ok": False, "detail": f"unknown device {target!r}"}
    try:
        return await fn(coord, target)
    except Exception as exc:  # surface the failure on screen instead of hiding it
        coord.bus.emit("alert", severity="high", title=f"Attack scene '{scene}' failed to run",
                       detail=repr(exc)[:300], source="attack")
        return {"ok": False, "detail": repr(exc)}


__all__ = ["run_attack", "SCENES", "json"]
