"""Automated rehearsal of the 7-scene script against a RUNNING demo (run_demo.py), with checks.

    demo\\.venv\\Scripts\\python demo\\scripts\\rehearse.py [--save-golden]

Scenes are triggered through the same admin API as Mission Control's buttons, at fixed rounds,
and every expected outcome is verified from the live event stream (nothing is assumed).
With --save-golden and all checks passing, the run's recording becomes recordings/golden.jsonl.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO))

import httpx  # noqa: E402

from fldemo.paths import RECORDINGS_DIR  # noqa: E402
from fldemo.settings import COORD_PORT, load_secrets  # noqa: E402

BASE = f"http://127.0.0.1:{COORD_PORT}"
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


class Rehearsal:
    def __init__(self) -> None:
        self.h = {"Authorization": f"Bearer {load_secrets()['admin_token']}"}
        self.events: list[dict] = []
        self.results: list[tuple[str, bool, str]] = []

    def pull(self) -> None:
        self.events += httpx.get(f"{BASE}/fl/events", params={"since": len(self.events)}, timeout=10).json()

    def round_done(self) -> int:
        self.pull()
        done = [e["round"] for e in self.events if e["type"] == "round_completed"]
        return max(done) if done else 0

    def wait_round(self, r: int, timeout: float = 600) -> None:
        t0 = time.time()
        while self.round_done() < r:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"round {r} not completed")
            time.sleep(1)

    def attack(self, scene: str, target: str | None = None) -> dict:
        r = httpx.post(f"{BASE}/fl/admin/attack/{scene}", headers=self.h, json={"target": target}, timeout=60)
        return r.json()

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    def find(self, pred) -> list[dict]:
        self.pull()
        return [e for e in self.events if pred(e)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-golden", action="store_true")
    args = ap.parse_args()
    R = Rehearsal()
    run_id = httpx.get(f"{BASE}/fl/state", timeout=10).json()["run_id"]
    print(f"Rehearsing on {run_id}")

    print("Scene 1-2: fleet online, learning together")
    R.wait_round(2)
    acc = [e["accuracy"] for e in R.find(lambda e: e["type"] == "global_eval")]
    R.check("devices online", len({e["device"] for e in R.find(lambda e: e["type"] == "device_status")}) >= 6)
    R.check("updates approved by assisted policy",
            len(R.find(lambda e: e["type"] == "approval" and e["decision"] == "approved")) >= 6)
    R.check("accuracy rising", acc[-1] > acc[0] + 0.15, f"{acc[0]:.2f} → {acc[-1]:.2f}")

    print("Scene 3: real-life dropout")
    R.attack("low_battery", "phone-06")
    R.attack("weak_signal", "phone-02")
    r0 = R.round_done()
    R.wait_round(r0 + 2)
    skips = R.find(lambda e: e["type"] == "device_status" and e.get("state") == "skipped")
    R.check("low battery skip", any(e["device"] == "phone-06" and e.get("reason") == "low_battery" for e in skips))
    R.check("weak signal skip (Toxiproxy)", any(e["device"] == "phone-02" and e.get("reason") == "weak_signal" for e in skips))
    R.attack("restore", "phone-02")

    print("Scene 4: hacked device")
    R.attack("poison", "phone-03")
    r0 = R.round_done()
    R.wait_round(r0 + 2)
    q = R.find(lambda e: e["type"] == "approval" and e["decision"] == "quarantined" and e["device"] == "phone-03")
    R.check("poisoned update quarantined", bool(q), q[0]["reason"][:80] if q else "")
    R.attack("restore", "phone-03")

    print("Scene 5: replay + tamper (+ forged key)")
    for scene, code in (("replay", 409), ("tamper", 422), ("forge", 401)):
        res = R.attack(scene, "phone-02" if scene == "forge" else None)
        R.check(f"{scene} blocked", res.get("http_status") == code, f"HTTP {res.get('http_status')}")

    print("Scene 6: eavesdropper")
    res = R.attack("mitm")
    R.check("tap sees no plaintext", res.get("tap_plaintext_found") is False, f"{res.get('tap_bytes_captured')} bytes")
    R.check("impostor certificate refused", res.get("device_refused") is True)

    print("Scene 7: research twist (drift)")
    R.attack("drift", "phone-05")
    r0 = R.round_done()
    R.wait_round(r0 + 3)
    drift_skips = R.find(lambda e: e["type"] == "device_status" and e.get("reason") == "drift" and e["device"] == "phone-05")
    flagged = R.find(lambda e: e["type"] == "approval" and e["decision"] == "quarantined" and e["device"] == "phone-05")
    R.check("drifting device drops out (drift-causal)", bool(drift_skips), f"{len(drift_skips)} skipped rounds")
    R.check("honest drifting device NOT quarantined by FinSentinel (expected)", not flagged)
    R.attack("restore")

    passed = all(ok for _, ok, _ in R.results)
    print(f"\n{'ALL SCENES PASSED' if passed else 'SOME SCENES FAILED'} "
          f"({sum(ok for _, ok, _ in R.results)}/{len(R.results)} checks)")
    if passed and args.save_golden:
        print("waiting for the run to finish before saving the golden recording…")
        while not R.find(lambda e: e["type"] == "run_finished"):
            time.sleep(2)
        src = RECORDINGS_DIR / f"{run_id}.jsonl"
        shutil.copyfile(src, RECORDINGS_DIR / "golden.jsonl")
        print(f"golden recording ← {src.name}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
