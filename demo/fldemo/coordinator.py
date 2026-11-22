"""Demo FL coordinator (FastAPI) — round management, signed model downlink, assisted approval,
aggregation through FinSentinel, live event stream + recording.

Run: python -m fldemo.coordinator --rounds 30 --round-seconds 25 [--wokwi] [--wazuh]

Security model:
  * Devices submit updates straight to FinSentinel; the coordinator never vouches for them.
  * Every global model is signed with the coordinator's Ed25519 key; devices pin the public key.
  * "Assisted" approval: an update FinSentinel scored LOW risk is approved through FinSentinel's
    operator API with reason "[actor=demo-coordinator] ..."; anything flagged waits for a human
    (FinSentinel UI or Mission Control's approve button, which calls the same operator API).
  * Aggregation is FinSentinel's /api/v1/aggregation/run: it re-checks approval and the exact bytes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import secrets as pysecrets
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from fldemo import MODEL_VERSION  # noqa: E402
from fldemo import keys as K  # noqa: E402
from fldemo.fleet import (  # noqa: E402
    bytes_to_state, delta_l2, sha256_hex, state_to_bytes, weight_delta_grid,
)
from fldemo.har import load_split, subject_arrays  # noqa: E402
from fldemo.model import PARAM_COUNT, TENSOR_SPECS, new_model, to_numpy_state  # noqa: E402
from fldemo import npmodel  # noqa: E402
from fldemo.paths import RECORDINGS_DIR  # noqa: E402
from fldemo.settings import (  # noqa: E402
    FORWARDER_PORT, FS_PORT, MC_PORT, THRESHOLDS_FILE,
    RunConfig, TrainConfig, load_devices, load_secrets,
)
from fldemo.wire import build_npz_stored  # noqa: E402

FS = f"http://127.0.0.1:{FS_PORT}"
NTFY = "https://ntfy.sh"


# =============================================================================== helpers
def npz_to_state(payload: bytes) -> "OrderedDict[str, np.ndarray]":
    """Parse a (trusted-after-FinSentinel) NPZ payload into a numpy state, in TENSOR_SPECS order."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        manifest = json.loads(z["__manifest__"].tobytes().decode())
        by_name = {t["name"]: np.array(z[t["archive_key"]]) for t in manifest["tensors"]}
    return OrderedDict((n, by_name[n].astype("<f4")) for n, _ in TENSOR_SPECS)


def gate_checks(http_status: int, detail: str, severity: str | None) -> dict[str, str]:
    """Map FinSentinel's outcome to the 5 lights, in its REAL pipeline order:
    integrity → signature → schema/policy → replay → anomaly detection."""
    order = ["untampered", "sender", "shape", "replay", "normal"]
    if http_status == 200:
        normal = {"LOW": "pass", "MEDIUM": "warn"}.get(severity or "LOW", "fail")
        return {"untampered": "pass", "sender": "pass", "shape": "pass", "replay": "pass", "normal": normal}
    d = (detail or "").lower()
    if http_status == 422 and ("payload" in d and ("match" in d or "size" in d or "empty" in d)) \
            and "serialized" not in d and "schema" not in d:
        failed = "untampered"
    elif http_status == 401:
        failed = "sender"
    elif http_status in (403, 422):
        failed = "shape"
    elif http_status == 409:
        failed = "replay"
    else:
        return {k: "skip" for k in order}
    out, hit = {}, False
    for k in order:
        out[k] = "skip" if hit else ("fail" if k == failed else "pass")
        hit = hit or k == failed
    return out


def detail_text(body: Any) -> str:
    if isinstance(body, dict):
        d = body.get("detail", body.get("text", ""))
        if isinstance(d, dict):
            return str(d.get("message") or d.get("code") or d)
        return str(d)
    return str(body)


AUDIT_TITLES = {
    "security.payload_integrity_failed": "Tampered payload rejected",
    "security.unknown_signing_key": "Update signed with an unknown key",
    "security.invalid_signature": "Forged signature rejected",
    "security.replay_detected": "Replay attack blocked",
    "security.model_schema_rejected": "Wrong model shape rejected",
    "security.update_quarantined": "Suspicious update quarantined",
    "security.anomaly_detected": "Anomalous update detected",
}


def audit_title(event_type: str) -> str:
    if event_type in AUDIT_TITLES:
        return AUDIT_TITLES[event_type]
    if event_type.startswith("security.policy_rejected"):
        return "Update rejected by policy"
    return event_type.replace(".", " · ").replace("_", " ")


# =============================================================================== event bus
class EventBus:
    def __init__(self, run_id: str, record: bool = True) -> None:
        self.run_id = run_id
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.path = RECORDINGS_DIR / f"{run_id}.jsonl" if record else None

    def emit(self, type_: str, **fields: Any) -> dict:
        ev = {"seq": len(self.events) + 1, "t": round(time.time(), 3), "run_id": self.run_id,
              "type": type_, **fields}
        self.events.append(ev)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, separators=(",", ":")) + "\n")
        for q in list(self.subscribers):
            q.put_nowait(ev)
        return ev


# =============================================================================== coordinator
class Coordinator:
    def __init__(self, cfg: RunConfig, *, wazuh: bool = False, record: bool = True,
                 fs_client: httpx.AsyncClient | None = None, min_devices: int = 0) -> None:
        self.cfg = cfg
        self.min_devices = min_devices
        self.wazuh = wazuh
        self.devices = {d["id"]: d for d in load_devices()}
        self.secrets = load_secrets()
        self.key = K.load_or_create_key(K.COORDINATOR)
        self.run_id = time.strftime("run-%Y%m%d-%H%M%S")
        self.bus = EventBus(self.run_id, record=record)
        self.fs = fs_client or httpx.AsyncClient(base_url=FS, timeout=30.0)
        self.op_headers = {"Authorization": f"Bearer {self.secrets['operator_token']}"}

        self.global_state = to_numpy_state(new_model(seed=cfg.seed))
        self.round = 0
        self.state = "waiting"
        self.deadline = 0.0
        self.model_cache: dict[int, bytes] = {}
        self.reports: dict[int, dict[str, dict]] = defaultdict(dict)     # round -> device -> report
        self.skips: dict[int, dict[str, str]] = defaultdict(dict)       # round -> device -> reason
        self.updates: dict[str, dict] = {}                               # update_id -> record
        self.last_seen: dict[str, float] = {}
        self.device_state: dict[str, dict] = {}
        self.sim: dict[str, dict] = {d: {} for d in self.devices}       # simulated-world directives
        self.attack_watch: dict[str, str] = {}                           # device -> scene (poison/drift)
        self.seen_audit: set[str] = set()
        self.tunnel_url: str | None = None
        self.monitor_threshold = self._monitor_threshold()
        split = load_split()
        self.test_x, self.test_y = subject_arrays(split["test"])
        self.device_data = {d: subject_arrays([v["subject"]]) for d, v in self.devices.items()}
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _monitor_threshold() -> float:
        if THRESHOLDS_FILE.exists():
            return float(json.loads(THRESHOLDS_FILE.read_text())["coordinator_drift_monitor"]["threshold"])
        return 0.0

    # ------------------------------------------------------------------ model downlink
    def model_bytes(self, r: int) -> bytes:
        if r not in self.model_cache:
            raise KeyError(r)
        return self.model_cache[r]

    def round_info(self) -> dict:
        blob = self.model_cache.get(self.round)
        sha = sha256_hex(blob) if blob else ""
        return {
            "round": self.round, "state": self.state, "model_version": MODEL_VERSION,
            "model_sha256": sha,
            "model_sig": K.sign_b64(self.key, K.model_sig_message(MODEL_VERSION, self.round, sha)) if blob else "",
            "deadline": self.deadline, "server_time": time.time(),
            "epochs": self.cfg.train.epochs, "lr": self.cfg.train.lr, "batch_size": self.cfg.train.batch_size,
        }

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.bus.emit(
            "run_started", model_version=MODEL_VERSION, param_count=PARAM_COUNT, rounds=self.cfg.rounds,
            round_seconds=self.cfg.round_seconds, test_samples=int(len(self.test_y)), mode="live",
            devices=[{"id": d["id"], "kind": d["kind"], "label": d["label"], "subject": d["subject"],
                      "network": d["network"], "battery": d["battery"]}
                     for d in self.devices.values() if d["kind"] == "virtual" or self.cfg.wokwi],
        )
        acc, loss = npmodel.loss_acc(self.global_state, self.test_x, self.test_y)[::-1]
        self.bus.emit("global_eval", round=0, accuracy=round(acc, 4), loss=round(loss, 4),
                      per_device_accuracy=self._per_device_acc())
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            t0 = time.time()  # wait (≤ 90 s) until the expected virtual devices are online
            while len(self.last_seen) < self.min_devices and time.time() - t0 < 90:
                await asyncio.sleep(0.5)
            await asyncio.sleep(self.cfg.pause_seconds)
            for r in range(1, self.cfg.rounds + 1):
                await self._one_round(r)
                await asyncio.sleep(self.cfg.pause_seconds)
            self.state = "finished"
            last = [e for e in self.bus.events if e["type"] == "global_eval"]
            self.bus.emit("run_finished", rounds_completed=self.cfg.rounds,
                          final_accuracy=last[-1]["accuracy"] if last else None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # keep the stage alive and visible
            self.bus.emit("alert", severity="critical", title="Coordinator error", detail=repr(exc),
                          source="coordinator")
            raise

    def _expected_devices(self) -> list[str]:
        now = time.time()
        out = []
        for d, v in self.devices.items():
            seen = self.last_seen.get(d, 0)
            st = self.device_state.get(d, {})
            if now - seen < (20 if v["kind"] == "virtual" else 45) and st.get("online", True) \
                    and st.get("state") != "offline":
                out.append(d)
        return out

    async def _one_round(self, r: int) -> None:
        self.round = r
        blob = state_to_bytes(self.global_state)
        self.model_cache[r] = blob
        self.model_cache.pop(r - 3, None)
        start = time.time()
        self.deadline = start + self.cfg.round_seconds
        self.state = "open"
        self.bus.emit("round_started", round=r, deadline=round(self.deadline, 3), model_sha256=sha256_hex(blob))
        while time.time() < self.deadline:
            await asyncio.sleep(0.5)
            done = set(self.reports[r]) | set(self.skips[r])
            exp = self._expected_devices()
            if time.time() - start >= self.cfg.min_round_seconds and exp and all(d in done for d in exp):
                break
        self.state = "aggregating"
        await asyncio.sleep(self.cfg.grace_seconds)
        await self._aggregate(r)

    # ------------------------------------------------------------------ reports & gate
    async def handle_report(self, rep: dict) -> dict:
        cid, r = rep["client_id"], int(rep["round"])
        if cid not in self.devices:
            raise HTTPException(404, "unknown device")
        sub = rep["submission"]
        env = sub["envelope"]
        payload = base64.b64decode(sub["payload_base64"], validate=True)
        if hashlib.sha256(payload).hexdigest() != env.get("payload_sha256"):
            raise HTTPException(422, "payload does not match envelope hash")
        if env.get("client_id") != cid:
            raise HTTPException(422, "envelope client_id mismatch")
        self.last_seen[cid] = time.time()
        uid = env["update_id"]
        fsr = rep.get("finsentinel", {})
        status = int(fsr.get("http_status", 0))
        train = rep.get("train", {})
        rec = {"device": cid, "round": r, "update_id": uid, "payload": payload, "envelope": env,
               "sample_count": int(env["sample_count"]), "http_status": status, "approval": None,
               "body": sub, "t": time.time()}
        self.updates[uid] = rec
        if r == self.round:
            self.reports[r][cid] = rec

        try:
            state = npz_to_state(payload)
            base = bytes_to_state(self.model_cache[r]) if r in self.model_cache else self.global_state
            rec["delta"] = np.concatenate([(np.asarray(state[k], np.float64) - np.asarray(base[k], np.float64)).ravel()
                                           for k, _ in TENSOR_SPECS])
            grid, dl2 = weight_delta_grid(base, state), delta_l2(state, base)
        except Exception:
            grid, dl2 = [], 0.0
        self.bus.emit("local_training", device=cid, round=r, epochs=int(train.get("epochs", 0)),
                      losses=[round(float(v), 4) for v in train.get("losses", [])],
                      samples=int(train.get("samples", env["sample_count"])),
                      class_counts=list(train.get("class_counts", [])), weight_delta=grid,
                      delta_l2=round(dl2, 4), seconds=round(float(train.get("seconds", 0)), 2))
        self.bus.emit("update_submitted", device=cid, round=r, update_id=uid, payload_bytes=len(payload),
                      payload_sha256=env["payload_sha256"], signed=True,
                      path="tunnel" if self.devices[cid]["kind"] == "wokwi" else "toxiproxy")
        await self._gate_from_finsentinel(rec, fsr.get("body"), source="device")
        return {"ok": True}

    async def _gate_from_finsentinel(self, rec: dict, body: Any, *, source: str) -> None:
        cid, uid, status = rec["device"], rec["update_id"], rec["http_status"]
        risk = severity = approval = None
        reasons: list[str] = []
        detail = detail_text(body)
        if status == 200:
            # Trust FinSentinel's own record, not the device's copy of the response.
            st = await self.fs.get(f"/api/v1/updates/{cid}/{uid}/status", headers=self.op_headers)
            if st.status_code == 200:
                s = st.json()
                risk, severity, approval = s.get("risk_score"), s.get("risk_severity"), s.get("status")
                reasons = list(s.get("anomaly_reasons") or [])
                detail = s.get("reason") or "accepted for review"
            else:
                detail = f"FinSentinel status lookup failed ({st.status_code})"
        checks = gate_checks(status, detail, severity)
        rec["approval"] = approval
        self.bus.emit("gate_result", device=cid, round=rec["round"], update_id=uid, http_status=status,
                      checks=checks, risk_score=risk, severity=severity, approval_status=approval,
                      reasons=reasons, detail=detail[:300], source=source)
        scene = self.attack_watch.get(cid)
        if scene == "poison" and source == "device":
            outcome = "quarantined" if approval == "quarantined" else ("blocked" if status != 200 else "accepted")
            self.bus.emit("attack", scene="poison", phase="result", target=cid, http_status=status,
                          outcome=outcome, detail=f"risk {risk} {severity or ''}: " + "; ".join(reasons)[:200])
            self.attack_watch.pop(cid, None)
        if approval == "quarantined":
            self.bus.emit("approval", device=cid, round=rec["round"], update_id=uid, decision="quarantined",
                          actor="finsentinel (risk policy)", reason="; ".join(reasons)[:300] or detail)
        elif approval == "pending_approval" and severity == "LOW":
            await self.decide(cid, uid, "approve",
                              actor="demo-coordinator",
                              reason="[actor=demo-coordinator] assisted approval: LOW risk, all ingestion checks passed")

    async def decide(self, cid: str, uid: str, decision: str, *, actor: str, reason: str) -> dict:
        path = {"approve": "approve", "reject": "reject"}[decision]
        resp = await self.fs.post(f"/api/v1/updates/{cid}/{uid}/{path}", headers=self.op_headers,
                                  json={"reason": reason})
        if resp.status_code != 200:
            self.bus.emit("alert", severity="medium", title=f"{decision} failed for {cid}",
                          detail=detail_text(resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text),
                          device=cid, source="coordinator")
            return {"ok": False, "status": resp.status_code}
        rec = self.updates.get(uid)
        if rec:
            rec["approval"] = "approved" if decision == "approve" else "rejected"
        self.bus.emit("approval", device=cid, round=rec["round"] if rec else self.round, update_id=uid,
                      decision="approved" if decision == "approve" else "rejected", actor=actor, reason=reason)
        return {"ok": True}

    # ------------------------------------------------------------------ aggregation
    async def _aggregate(self, r: int) -> None:
        reps = self.reports[r]
        approved = [rec for rec in reps.values() if rec.get("approval") == "approved"]
        dropped = [{"device": d, "reason": why} for d, why in self.skips[r].items() if d not in reps]
        for d in self._expected_devices():
            if d not in reps and d not in self.skips[r]:
                dropped.append({"device": d, "reason": "no_report"})
        for rec in reps.values():
            if rec.get("approval") != "approved":
                dropped.append({"device": rec["device"], "reason": rec.get("approval") or f"blocked_{rec['http_status']}"})

        # Coordinator research-layer drift monitor (cosine to the cohort), labelled as such.
        with_delta = [rec for rec in reps.values() if rec.get("delta") is not None and rec.get("http_status") == 200]
        if len(with_delta) >= 3 and self.monitor_threshold:
            for rec in with_delta:
                others = np.mean([o["delta"] for o in with_delta if o is not rec], axis=0)
                cos = float(rec["delta"] @ others / (np.linalg.norm(rec["delta"]) * np.linalg.norm(others) + 1e-12))
                flagged = cos < self.monitor_threshold
                if flagged or self.attack_watch.get(rec["device"]) == "drift":
                    self.bus.emit("drift_monitor", device=rec["device"], round=r, score=round(cos, 3),
                                  threshold=self.monitor_threshold, flagged=flagged,
                                  finsentinel_flagged=rec.get("approval") == "quarantined")

        aggregated = False
        if approved:
            ref = build_npz_stored(self.global_state)
            body = {"model_version": MODEL_VERSION, "round_number": r,
                    "reference_payload_base64": base64.b64encode(ref).decode(),
                    "updates": [{"client_id": a["device"], "update_id": a["update_id"],
                                 "model_version": MODEL_VERSION, "round_number": r,
                                 "sample_count": a["sample_count"],
                                 "payload_base64": base64.b64encode(a["payload"]).decode()} for a in approved]}
            resp = await self.fs.post("/api/v1/aggregation/run", headers=self.op_headers, json=body)
            if resp.status_code == 200:
                new = base64.b64decode(resp.json()["aggregated_payload_base64"])
                self.global_state = npz_to_state(new)
                aggregated = True
            else:
                self.bus.emit("alert", severity="high", title="Aggregation refused by FinSentinel",
                              detail=detail_text(resp.json())[:300], source="finsentinel")
        participants = [a["device"] for a in approved] if aggregated else []
        self.bus.emit("round_completed", round=r, participants=participants, dropped=dropped,
                      aggregated=aggregated, model_sha256=sha256_hex(state_to_bytes(self.global_state)))
        loss, acc = npmodel.loss_acc(self.global_state, self.test_x, self.test_y)
        self.bus.emit("global_eval", round=r, accuracy=round(acc, 4), loss=round(loss, 4),
                      per_device_accuracy=self._per_device_acc())

    def _per_device_acc(self) -> dict[str, float]:
        return {d: round(npmodel.loss_acc(self.global_state, x, y)[1], 3) for d, (x, y) in self.device_data.items()}

    # ------------------------------------------------------------------ telemetry
    def handle_status(self, st: dict) -> None:
        cid = st.get("client_id")
        if cid not in self.devices:
            raise HTTPException(404, "unknown device")
        self.last_seen[cid] = time.time()
        prev = self.device_state.get(cid, {})
        cur = {k: st.get(k) for k in ("state", "round", "battery", "signal_dbm", "network", "online",
                                      "hacked", "reason", "detail") if st.get(k) is not None}
        self.device_state[cid] = {**prev, **cur}
        if st.get("state") == "skipped" and st.get("round") == self.round:
            self.skips[self.round][cid] = st.get("reason") or "other"
        if st.get("state") in ("offline",):
            self.device_state[cid]["online"] = False
        elif st.get("state"):
            self.device_state[cid]["online"] = True
        if cur != {k: prev.get(k) for k in cur}:
            self.bus.emit("device_status", device=cid, **cur)
        if self.attack_watch.get(cid) == "drift" and st.get("reason") == "drift":
            self.bus.emit("attack", scene="drift", phase="result", target=cid, outcome="observed",
                          detail=f"{cid} skipped round {st.get('round')} because its own data drifted "
                                 f"(drift-causal dropout; {st.get('detail') or ''})")

    # ------------------------------------------------------------------ audit webhook (FinSentinel → us)
    async def handle_audit(self, payload: dict) -> None:
        key = payload.get("idempotency_key") or payload.get("event_id")
        if key in self.seen_audit:
            return
        self.seen_audit.add(key)
        sev = str(payload.get("severity", "info")).lower()
        et = str(payload.get("event_type", ""))
        if sev in ("low", "medium", "high", "critical"):  # routine info events stay in FinSentinel
            self.bus.emit("alert", severity=sev if sev in ("low", "medium", "high", "critical") else "info",
                          title=audit_title(et), detail=str(payload.get("description", ""))[:300],
                          device=payload.get("client_id") or None, source="finsentinel", event_type=et)
        if self.wazuh:
            try:
                async with httpx.AsyncClient(timeout=5) as c:
                    await c.post(f"http://127.0.0.1:{FORWARDER_PORT}/ingest", json=payload,
                                 headers={"Authorization": f"Bearer {self.secrets['forwarder_token']}"})
            except httpx.HTTPError:
                pass

    # ------------------------------------------------------------------ tunnel beacon
    async def set_tunnel(self, url: str | None) -> None:
        self.tunnel_url = url
        published = False
        if url:
            ts = int(time.time())
            line = f"{K.beacon_message(url, ts)}|{K.sign_b64(self.key, K.beacon_message(url, ts))}"
            topic = K.ensure_beacon_topic()
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    published = (await c.post(f"{NTFY}/{topic}", content=line.encode())).status_code == 200
            except httpx.HTTPError:
                published = False
        self.bus.emit("tunnel", url=url, status="up" if url else "down", beacon_published=published)

    # ------------------------------------------------------------------ toxiproxy (real network effects)
    async def set_network(self, device: str, profile: str) -> bool:
        from fldemo import toxi
        try:
            await asyncio.to_thread(toxi.apply_profile, device, profile)
        except (httpx.HTTPError, ValueError, KeyError, OSError):
            return False
        self.sim.setdefault(device, {})["network"] = profile
        return True


# =============================================================================== FastAPI app
def create_app(coord: Coordinator, *, autostart: bool = True) -> FastAPI:
    app = FastAPI(title="FL demo coordinator", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=[f"http://localhost:{MC_PORT}", f"http://127.0.0.1:{MC_PORT}"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.coord = coord
    sec = coord.secrets

    def admin(request: Request) -> None:
        if request.headers.get("authorization", "") != f"Bearer {sec['admin_token']}":
            raise HTTPException(401, "admin token required")
        if request.client and request.client.host not in ("127.0.0.1", "::1", "testclient"):
            raise HTTPException(403, "local only")

    @app.on_event("startup")
    async def _startup() -> None:
        if autostart:
            coord.start()

    # ---- public device API (exposed through Caddy) ----
    @app.get("/fl/round")
    async def get_round() -> dict:
        return coord.round_info()

    @app.get("/fl/model/{r}")
    async def get_model(r: int) -> Response:
        try:
            return Response(coord.model_bytes(r), media_type="application/octet-stream")
        except KeyError:
            raise HTTPException(404, "no such round") from None

    @app.get("/fl/pubkey")
    async def pubkey() -> dict:
        return {"coordinator_ed25519_pub_hex": K.coordinator_pub_hex()}

    @app.post("/fl/status")
    async def status(request: Request) -> dict:
        coord.handle_status(await request.json())
        return {"ok": True}

    @app.post("/fl/report")
    async def report(request: Request) -> dict:
        return await coord.handle_report(await request.json())

    # ---- local-only: simulated world for virtual devices ----
    @app.get("/fl/sim/{device}")
    async def sim(device: str, request: Request) -> dict:
        if request.client and request.client.host not in ("127.0.0.1", "::1", "testclient"):
            raise HTTPException(403, "local only")
        return coord.sim.get(device, {})

    # ---- local-only: events, state, admin ----
    @app.get("/fl/state")
    async def state() -> dict:
        return {"run_id": coord.run_id, "round": coord.round, "state": coord.state,
                "events": len(coord.bus.events), "tunnel": coord.tunnel_url}

    @app.get("/fl/events")
    async def events(since: int = 0) -> JSONResponse:
        return JSONResponse(coord.bus.events[since:])

    @app.websocket("/fl/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        backlog = list(coord.bus.events)
        coord.bus.subscribers.add(q)
        try:
            for ev in backlog:
                await sock.send_text(json.dumps(ev))
            last = backlog[-1]["seq"] if backlog else 0
            while True:
                ev = await q.get()
                if ev["seq"] > last:
                    await sock.send_text(json.dumps(ev))
                    last = ev["seq"]
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            coord.bus.subscribers.discard(q)

    @app.post("/fl/admin/decision")
    async def decision(request: Request) -> dict:
        admin(request)
        b = await request.json()
        return await coord.decide(b["client_id"], b["update_id"], b["decision"], actor="presenter (Mission Control)",
                                  reason=f"[actor=presenter via Mission Control] manual {b['decision']}")

    @app.post("/fl/admin/attack/{scene}")
    async def attack(scene: str, request: Request) -> dict:
        admin(request)
        from fldemo.attacks import run_attack
        body = await request.json() if (await request.body()) else {}
        return await run_attack(coord, scene, body.get("target"))

    @app.post("/fl/admin/tunnel")
    async def tunnel(request: Request) -> dict:
        admin(request)
        await coord.set_tunnel((await request.json()).get("url"))
        return {"ok": True}

    @app.post("/fl/admin/component")
    async def component(request: Request) -> dict:
        admin(request)
        b = await request.json()
        coord.bus.emit("component", name=b["name"], status=b["status"], detail=b.get("detail"))
        return {"ok": True}

    @app.post("/fl/internal/audit")
    async def audit(request: Request) -> dict:
        if request.headers.get("authorization", "") != f"Bearer {sec['webhook_token']}":
            raise HTTPException(401, "bad webhook token")
        await coord.handle_audit(await request.json())
        return {"ok": True}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--round-seconds", type=float, default=25.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wokwi", action="store_true")
    ap.add_argument("--wazuh", action="store_true")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--min-devices", type=int, default=0, help="wait for this many devices before round 1")
    args = ap.parse_args()
    cfg = RunConfig(rounds=args.rounds, round_seconds=args.round_seconds, wokwi=args.wokwi,
                    train=TrainConfig(epochs=args.epochs, lr=args.lr))
    coord = Coordinator(cfg, wazuh=args.wazuh, min_devices=args.min_devices)
    print(f"[coordinator] {coord.run_id} recording to {coord.bus.path}", flush=True)
    import uvicorn
    uvicorn.run(create_app(coord), host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
