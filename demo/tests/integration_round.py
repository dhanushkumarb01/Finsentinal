"""In-process end-to-end run: real FinSentinel app + real coordinator + real device agents.

Run as a subprocess by tests/test_integration_round.py (FinSentinel's boot is process-global).
Needs FLDEMO_STATE_DIR / FLDEMO_RECORDINGS_DIR set to temp dirs. Prints one JSON summary line.

Wiring: one ASGI dispatcher (/api/v1/* → FinSentinel, everything else → coordinator) behind a
single Starlette TestClient; device agents use that client instead of Toxiproxy/TLS, and the
coordinator reaches FinSentinel through httpx.ASGITransport. Everything else is the real code.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO))

assert "FLDEMO_STATE_DIR" in os.environ, "set FLDEMO_STATE_DIR to a temp dir"

from fldemo import keys as K  # noqa: E402
from fldemo import fs_boot  # noqa: E402
from fldemo.settings import FS_DB, load_devices  # noqa: E402

ROUNDS = int(os.environ.get("IT_ROUNDS", "3"))
VIRTUAL = [d["id"] for d in load_devices() if d["kind"] == "virtual"]


def main() -> None:
    K.ensure_secrets()
    K.ensure_device_keys([d["id"] for d in load_devices()])
    from fldemo.fs_server import build_env
    env = build_env(ROUNDS)
    env["FINSENTINEL_AUDIT_DELIVERY_WORKER_ENABLED"] = "0"
    fs_boot.boot(FS_DB, env)

    import httpx
    import torch
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from fastapi.testclient import TestClient

    from app.main import app as fs_app
    from core.key_registry import register_trusted_key
    from core.model_registry import register_model_schema
    from fldemo import MODEL_VERSION
    from fldemo.coordinator import Coordinator, create_app
    from fldemo.device import Device
    from fldemo.model import new_model, to_numpy_state
    from fldemo.settings import REGISTRY_FILE, RunConfig, TrainConfig

    st = to_numpy_state(new_model(seed=0))
    register_model_schema(MODEL_VERSION, {k: torch.from_numpy(v) for k, v in st.items()})
    for cid, pub in json.loads(REGISTRY_FILE.read_text()).items():
        register_trusted_key(key_id=f"{cid}-key", client_id=cid,
                             public_key=Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)), max_sample_count=2000)

    cfg = RunConfig(rounds=ROUNDS, round_seconds=20, min_round_seconds=1.0, grace_seconds=0.5, pause_seconds=0.3,
                    train=TrainConfig(epochs=2, lr=0.1))
    fs_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fs_app), base_url="http://fs")
    coord = Coordinator(cfg, fs_client=fs_client)
    coord.gateway = "http://fs"            # attacks go straight to FinSentinel (no Caddy here)
    coord_app = create_app(coord)

    async def dispatcher(scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api/v1"):
            await fs_app(scope, receive, send)
        else:
            await coord_app(scope, receive, send)

    # Attacks need an HTTP client that reaches FinSentinel inside this process.
    import fldemo.attacks as A
    _orig = A._post_update

    async def _post_inproc(c, body):
        r = await fs_client.post("/api/v1/updates", json=body)
        return r.status_code, r.json()
    A._post_update = _post_inproc

    results: dict = {}
    with TestClient(dispatcher) as client:
        devices = []
        for did in VIRTUAL:
            d = Device(did, base_url="http://testserver", verify=False, poll_s=0.3)
            d.http, d.local, d.sim_url = client, client, f"/fl/sim/{did}"
            devices.append(d)
        threads = [threading.Thread(target=d.loop, kwargs={"stop_after_round": ROUNDS}, daemon=True) for d in devices]
        for t in threads:
            t.start()
        admin = {"Authorization": f"Bearer {coord.secrets['admin_token']}"}
        attacked = False
        t0 = time.time()
        while coord.state != "finished" and time.time() - t0 < 240:
            time.sleep(0.5)
            if not attacked and coord.round >= 2 and any(u["http_status"] == 200 for u in coord.updates.values()):
                for scene, target in (("replay", None), ("tamper", None), ("forge", "phone-02")):
                    r = client.post(f"/fl/admin/attack/{scene}", headers=admin, json={"target": target})
                    results[scene] = r.json().get("http_status")
                r = client.post("/fl/admin/attack/poison", headers=admin, json={"target": "phone-03"})
                attacked = True
        for t in threads:
            t.join(timeout=30)
        evs = coord.bus.events

    A._post_update = _orig
    types = [e["type"] for e in evs]
    evals = [e for e in evs if e["type"] == "global_eval"]
    gates = [e for e in evs if e["type"] == "gate_result"]
    summary = {
        "state": coord.state,
        "rounds_completed": sum(1 for e in evs if e["type"] == "round_completed"),
        "aggregated_rounds": sum(1 for e in evs if e["type"] == "round_completed" and e["aggregated"]),
        "accuracy": [e["accuracy"] for e in evals],
        "gate_statuses": sorted({g["http_status"] for g in gates}),
        "approvals": sum(1 for e in evs if e["type"] == "approval" and e["decision"] == "approved"),
        "quarantined": [e["device"] for e in evs if e["type"] == "approval" and e["decision"] == "quarantined"],
        "attacks": results,
        "event_types": sorted(set(types)),
        "recording": str(coord.bus.path),
        "recording_lines": sum(1 for _ in open(coord.bus.path, encoding="utf-8")),
        "events": len(evs),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    asyncio.run(fs_client.aclose())


if __name__ == "__main__":
    main()
