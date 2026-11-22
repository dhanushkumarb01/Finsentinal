"""End-to-end: a device-built STORED-zip update is accepted by the real FinSentinel API.

Runs FinSentinel's unmodified FastAPI app in a subprocess-free way via
TestClient, booted on a throwaway SQLite file (never FinSentinel's dev DB).
Each test module that boots FinSentinel must run in its own process because
the boot is process-global; pytest runs this file in-process, so it is the
only test here that boots FinSentinel.
"""

import secrets

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fldemo import MODEL_VERSION, fs_boot
from fldemo.model import new_model, to_numpy_state
from fldemo.wire import build_npz_stored, make_envelope, submission_body

TOKEN = secrets.token_hex(32)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("fs") / "finsentinel_demo_test.db"
    fs_boot.boot(db, fs_boot.base_env(operator_token=TOKEN, rounds=10))
    from fastapi.testclient import TestClient

    from app.main import app
    from core.key_registry import register_trusted_key
    from core.model_registry import register_model_schema

    state = to_numpy_state(new_model(seed=0))
    register_model_schema(MODEL_VERSION, {k: torch.from_numpy(v) for k, v in state.items()})
    key = Ed25519PrivateKey.generate()
    register_trusted_key(key_id="phone-01-key", client_id="phone-01", public_key=key.public_key(),
                         max_sample_count=1000)
    with TestClient(app) as c:
        c.key = key
        yield c


def _submit(client, *, update_id="phone-01-r0001-a", round_number=1, state=None):
    state = state or to_numpy_state(new_model(seed=1))
    payload = build_npz_stored(state)
    env = make_envelope(
        update_id=update_id, client_id="phone-01", model_version=MODEL_VERSION,
        round_number=round_number, sample_count=347, payload=payload,
        signing_key_id="phone-01-key", private_key=client.key,
    )
    return client.post("/api/v1/updates", json=submission_body(env, payload)), env, payload


def test_stored_update_is_accepted_and_pending(client):
    r, _, _ = _submit(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("approval_status", body.get("status")) in {"pending_approval"}, body


def test_replay_is_rejected(client):
    r1, env, payload = _submit(client, update_id="phone-01-r0001-b")
    assert r1.status_code == 200
    r2 = client.post("/api/v1/updates", json=submission_body(env, payload))
    assert r2.status_code == 409, r2.text


def test_tampered_payload_is_rejected(client):
    payload = build_npz_stored(to_numpy_state(new_model(seed=2)))
    env = make_envelope(
        update_id="phone-01-r0001-c", client_id="phone-01", model_version=MODEL_VERSION,
        round_number=1, sample_count=347, payload=payload,
        signing_key_id="phone-01-key", private_key=client.key,
    )
    bad = bytearray(payload)
    bad[200] ^= 0x01  # flip one bit inside a weight
    r = client.post("/api/v1/updates", json=submission_body(env, bytes(bad)))
    assert r.status_code == 422, r.text


def test_unknown_key_is_rejected(client):
    payload = build_npz_stored(to_numpy_state(new_model(seed=3)))
    env = make_envelope(
        update_id="phone-01-r0001-d", client_id="phone-01", model_version=MODEL_VERSION,
        round_number=1, sample_count=347, payload=payload,
        signing_key_id="phone-01-key", private_key=Ed25519PrivateKey.generate(),
    )
    r = client.post("/api/v1/updates", json=submission_body(env, payload))
    assert r.status_code == 401, r.text
