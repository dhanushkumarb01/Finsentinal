"""Virtual device agent — one OS process per phone.

    python -m fldemo.device --device phone-01

Talks to the server only through its own Toxiproxy link (https://localhost:<9100+i>, demo-CA
pinned), so network profiles and the weak-signal scene are real TCP effects. Follows
docs/PROTOCOL.md exactly like the ESP32 firmware. Dropout causes are physical and reported:
offline, low battery (drains when training), weak signal (model download misses its time
budget), model rejected (bad signature) and drift-causal dropout (research mechanism).
"""

from __future__ import annotations

import argparse
import base64
import os
import secrets
import ssl
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from fldemo import MODEL_VERSION, npmodel  # noqa: E402
from fldemo import keys as K  # noqa: E402
from fldemo.drift import DriftTracker  # noqa: E402
from fldemo.fleet import MODEL_BYTES, bytes_to_state, class_counts, device_data, poison, sha256_hex, sigmoid  # noqa: E402
from fldemo.settings import CA_CERT, COORD_PORT, DEVICES_FILE, TOXI_BASE_PORT  # noqa: E402
from fldemo.wire import build_npz_stored, make_envelope, submission_body  # noqa: E402

import json  # noqa: E402

DOWNLOAD_BUDGET_S = 4.0   # a phone gives up on a round if the model takes longer than this


class Device:
    def __init__(self, device_id: str, *, base_url: str | None = None, sim_url: str | None = None,
                 verify: ssl.SSLContext | bool | None = None, poll_s: float = 1.5) -> None:
        cfg = json.loads(DEVICES_FILE.read_text())
        self.spec = next(d for d in cfg["devices"] if d["id"] == device_id)
        self.profiles = cfg["network_profiles"]
        self.dd = cfg["drift_dropout"]
        self.low_batt = cfg["low_battery_percent"]
        self.id = device_id
        idx = int(device_id.split("-")[1])
        self.base = base_url or f"https://localhost:{TOXI_BASE_PORT + idx}"
        self.sim_url = sim_url or f"http://127.0.0.1:{COORD_PORT}/fl/sim/{device_id}"
        ctx = verify if verify is not None else ssl.create_default_context(cafile=str(CA_CERT))
        self.http = httpx.Client(base_url=self.base, verify=ctx, timeout=httpx.Timeout(10.0, connect=5.0))
        self.local = httpx.Client(timeout=3.0)
        self.key = K.load_or_create_key(device_id)
        self.coord_pub = K.public_key(K.COORDINATOR)
        self.battery = float(self.spec["battery"])
        self.network = self.spec["network"]
        self.poll_s = poll_s
        self.last_round = 0
        self.drift = DriftTracker()
        self.sim: dict = {}
        self.last_restore = 0.0
        self.battery_applied_at = 0.0
        self.offline_reported = False

    # ---------------------------------------------------------------- helpers
    @property
    def signal_dbm(self) -> int:
        return self.profiles[self.sim.get("network", self.network)]["signal_dbm"]

    def status(self, state: str, **kw) -> None:
        body = {"client_id": self.id, "state": state, "battery": int(round(self.battery)),
                "signal_dbm": self.signal_dbm, "network": self.sim.get("network", self.network),
                "online": state != "offline", "hacked": bool(self.sim.get("hacked")), **kw}
        try:
            self.http.post("/fl/status", json=body, timeout=5.0)
        except httpx.HTTPError:
            pass

    def refresh_sim(self) -> None:
        try:
            self.sim = self.local.get(self.sim_url).json()
        except (httpx.HTTPError, ValueError):
            return
        if self.sim.get("restore", 0) > self.last_restore:
            self.last_restore = self.sim["restore"]
            self.battery = max(self.battery, float(self.spec["battery"]))
        if self.sim.get("battery_at", 0) > self.battery_applied_at:  # one-shot per trigger
            self.battery_applied_at = self.sim["battery_at"]
            self.battery = float(self.sim["battery"])

    # ---------------------------------------------------------------- one round
    def run_round(self, info: dict) -> None:
        r = info["round"]
        if self.battery < self.low_batt:
            self.battery = min(100.0, self.battery + self.spec.get("recharge", 1) * 3)
            self.status("skipped", round=r, reason="low_battery", detail=f"battery {self.battery:.0f}% — charging")
            return
        self.status("downloading", round=r)
        t0 = time.time()
        try:
            resp = self.http.get(f"/fl/model/{r}", timeout=DOWNLOAD_BUDGET_S)
            blob = resp.content
            if resp.status_code != 200 or len(blob) != MODEL_BYTES or time.time() - t0 > DOWNLOAD_BUDGET_S:
                raise httpx.ReadTimeout("slow")
        except httpx.HTTPError:
            self.status("skipped", round=r, reason="weak_signal",
                        detail=f"model download exceeded {DOWNLOAD_BUDGET_S:.0f}s on {self.sim.get('network', self.network)}")
            return
        sha = sha256_hex(blob)
        if sha != info["model_sha256"] or not K.verify_b64(
                self.coord_pub, K.model_sig_message(MODEL_VERSION, r, sha), info["model_sig"]):
            self.status("model_rejected", round=r, reason="model_rejected",
                        detail="global model signature/hash invalid — refused")
            self.status("skipped", round=r, reason="model_rejected")
            return
        g = bytes_to_state(blob)

        drifted = bool(self.sim.get("drift")) or (
            self.spec.get("drift", {}).get("auto_start_round") not in (None, 0)
            and r >= self.spec["drift"]["auto_start_round"])
        x, y = device_data(self.spec, drifted=drifted, round_number=r)
        d_score = self.drift.score(x)
        if d_score >= self.dd["min_score"]:
            p = sigmoid(self.dd["alpha"] + self.dd["beta"] * d_score)
            u = np.random.default_rng(zlib.crc32(f"{self.id}|{r}".encode())).random()
            if u < p:
                self.status("skipped", round=r, reason="drift",
                            detail=f"own data drifted (D_indep={d_score:.3f}, p_skip={p:.2f})")
                return

        self.status("training", round=r, detail=f"{info['epochs']} epochs on {len(y)} samples")
        t1 = time.time()
        new, losses = npmodel.train_sgd(g, x, y, epochs=int(info["epochs"]), lr=float(info["lr"]),
                                        batch_size=int(info["batch_size"]), seed=r * 100 + int(self.id[-2:]))
        train_s = time.time() - t1
        self.battery = max(0.0, self.battery - self.spec.get("drain", 3))
        hacked = bool(self.sim.get("hacked"))
        if hacked:
            new = poison(new)

        self.status("signing", round=r)
        payload = build_npz_stored(new)
        env = make_envelope(update_id=f"{self.id}-r{r:04d}-{secrets.token_hex(4)}", client_id=self.id,
                            model_version=MODEL_VERSION, round_number=r, sample_count=int(len(y)),
                            payload=payload, signing_key_id=f"{self.id}-key", private_key=self.key)
        body = submission_body(env, payload)
        self.status("sending", round=r)
        try:
            fr = self.http.post("/api/v1/updates", json=body, timeout=20.0)
            try:
                fbody = fr.json()
            except ValueError:
                fbody = {"text": fr.text[:500]}
            fstatus = fr.status_code
        except httpx.HTTPError as exc:
            self.status("skipped", round=r, reason="weak_signal", detail=f"upload failed: {type(exc).__name__}")
            return
        report = {"client_id": self.id, "round": r, "submission": body,
                  "finsentinel": {"http_status": fstatus, "body": fbody},
                  "train": {"epochs": int(info["epochs"]), "losses": [round(v, 4) for v in losses],
                            "samples": int(len(y)), "class_counts": class_counts(y), "seconds": round(train_s, 3)},
                  "battery": int(round(self.battery)), "signal_dbm": self.signal_dbm, "hacked": hacked}
        try:
            self.http.post("/fl/report", json=report, timeout=20.0)
        except httpx.HTTPError:
            pass
        verdict = fbody.get("status") if isinstance(fbody, dict) else None
        self.status("done", round=r, detail=f"FinSentinel {fstatus} {verdict or ''}".strip())

    # ---------------------------------------------------------------- main loop
    def loop(self, stop_after_round: int | None = None) -> None:
        self.status("online")
        while True:
            self.refresh_sim()
            if self.sim.get("offline"):
                if not self.offline_reported:
                    self.status("offline", reason="offline", detail="switched off by presenter")
                    self.offline_reported = True
                time.sleep(self.poll_s)
                continue
            if self.offline_reported:
                self.offline_reported = False
                self.status("online")
            try:
                info = self.http.get("/fl/round", timeout=5.0).json()
            except (httpx.HTTPError, ValueError):
                time.sleep(self.poll_s)
                continue
            if info.get("state") == "finished":
                if stop_after_round:
                    return
                if self.last_round >= 0:
                    self.status("idle", detail="run finished")
                    self.last_round = -1  # report once; stay alive (the supervisor would restart us)
                time.sleep(5)
                continue
            if info.get("state") == "open" and info["round"] > max(self.last_round, 0):
                self.last_round = info["round"]
                try:
                    self.run_round(info)
                except Exception as exc:  # never die on stage; report it
                    self.status("error", round=info["round"], reason="other", detail=repr(exc)[:200])
                if stop_after_round and self.last_round >= stop_after_round:
                    return
            else:
                self.battery = min(100.0, self.battery + 0.02 * self.spec.get("recharge", 1))
            time.sleep(self.poll_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    args = ap.parse_args()
    Device(args.device).loop()


if __name__ == "__main__":
    main()
