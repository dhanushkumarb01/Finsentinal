"""Run FinSentinel (unmodified) for the demo on 127.0.0.1:8000 with the demo's own SQLite DB.

Started by run_demo.py as a separate process:
    python -m fldemo.fs_server --rounds 30 [--wazuh]
Registers the har-mlp-v1 schema and every device's public key (from .state/registry.json),
applies calibrated detection thresholds (config/thresholds.json) through env vars, and points
FinSentinel's audit webhook at the coordinator so its real audit events feed the alert stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fldemo import MODEL_VERSION, fs_boot  # noqa: E402
from fldemo.settings import (  # noqa: E402
    COORD_PORT, FS_DB, FS_PORT, REGISTRY_FILE, STATE_DIR, THRESHOLDS_FILE, load_secrets,
)


def build_env(rounds: int) -> dict[str, str]:
    sec = load_secrets()
    extra = {
        "FINSENTINEL_AUDIT_DELIVERY_WORKER_ENABLED": "1",
        "FINSENTINEL_AUDIT_WEBHOOK_URL": f"http://127.0.0.1:{COORD_PORT}/fl/internal/audit",
        "FINSENTINEL_AUDIT_WEBHOOK_TOKEN": sec["webhook_token"],
    }
    if THRESHOLDS_FILE.exists():
        extra.update({k: str(v) for k, v in json.loads(THRESHOLDS_FILE.read_text())["finsentinel_env"].items()})
    # +50 spare rounds so a long rehearsal never hits the policy's round allow-list.
    return fs_boot.base_env(operator_token=sec["operator_token"], rounds=rounds + 50, extra=extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--fresh", action="store_true", help="delete the demo DB first")
    args = ap.parse_args()

    os.chdir(STATE_DIR)  # never pick up a .env from FinSentinel's folder
    if args.fresh:
        for suffix in ("", "-wal", "-shm"):
            Path(str(FS_DB) + suffix).unlink(missing_ok=True)
    fs_boot.boot(FS_DB, build_env(args.rounds))

    import torch

    from core.key_registry import register_trusted_key
    from core.model_registry import register_model_schema
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from fldemo.model import new_model, to_numpy_state

    state = to_numpy_state(new_model(seed=0))
    register_model_schema(MODEL_VERSION, {k: torch.from_numpy(v) for k, v in state.items()})
    for client_id, pub_hex in json.loads(REGISTRY_FILE.read_text()).items():
        register_trusted_key(
            key_id=f"{client_id}-key", client_id=client_id,
            public_key=Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex)),
            max_sample_count=2000,
        )
    print(f"[finsentinel] schema {MODEL_VERSION} + {len(json.loads(REGISTRY_FILE.read_text()))} keys registered",
          flush=True)

    import uvicorn

    from app.main import app

    uvicorn.run(app, host="127.0.0.1", port=FS_PORT, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
