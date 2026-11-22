"""The device-side wire format must match FinSentinel byte-for-byte."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.crypto import canonical_signed_data, verify_envelope_signature
from features.model_schema import validate_model_schema  # noqa: F401  (import check)
from ingestion.schemas import UpdateEnvelope
from ingestion.serialization import deserialize_state_dict, serialize_state_dict

from fldemo import MODEL_VERSION
from fldemo.model import TENSOR_SPECS, new_model, to_numpy_state
from fldemo.testvectors import deterministic_state
from fldemo.wire import build_npz_stored, canonical_envelope, make_envelope, npy_bytes

VECTORS = json.loads((Path(__file__).parent / "vectors" / "finsentinel_vectors.json").read_text())


def _test_key():
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS["test_private_seed_hex"]))


def test_npy_bytes_match_numpy_save():
    for arr in (np.zeros((32, 64), "<f4"), np.arange(6, dtype="<f4"), np.arange(300, dtype=np.uint8)):
        buf = io.BytesIO()
        np.save(buf, arr)
        assert npy_bytes(arr) == buf.getvalue()


def test_stored_npz_entries_are_uncompressed_and_loadable():
    state = to_numpy_state(new_model(seed=3))
    payload = build_npz_stored(state)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert zf.testzip() is None
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        assert np.array_equal(npz["tensor_000000"], state["fc1.weight"])


def test_finsentinel_deserializer_accepts_stored_npz():
    state = to_numpy_state(new_model(seed=4))
    restored = deserialize_state_dict(build_npz_stored(state))
    assert list(restored) == [n for n, _ in TENSOR_SPECS]
    for name, arr in state.items():
        assert restored[name].dtype.__str__() == "torch.float32"
        assert np.array_equal(restored[name].numpy(), arr)


def test_stored_and_compressed_payloads_decode_identically():
    import torch
    state = to_numpy_state(new_model(seed=5))
    compressed = serialize_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    a, b = deserialize_state_dict(compressed), deserialize_state_dict(build_npz_stored(state))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_payload_vector_is_reproduced_exactly():
    payload = build_npz_stored(deterministic_state())
    assert hashlib.sha256(payload).hexdigest() == VECTORS["payload"]["sha256"]
    assert base64.b64encode(payload).decode() == VECTORS["payload"]["base64"]


@pytest.mark.parametrize("vec", VECTORS["envelopes"], ids=lambda v: v["created_at_sent"])
def test_server_canonical_form_matches_vectors(vec):
    env = UpdateEnvelope(**_vector_fields(vec["created_at_sent"]), signature=vec["signature_base64"])
    assert canonical_signed_data(env).decode() == vec["canonical_signed_data"]
    assert verify_envelope_signature(env, _test_key().public_key())


def test_device_canonical_form_matches_server_for_device_timestamp():
    vec = VECTORS["envelopes"][0]  # devices always send whole-second UTC "Z" timestamps
    fields = _vector_fields(vec["created_at_sent"])
    assert canonical_envelope(fields).decode() == vec["canonical_signed_data"]


def test_device_envelope_verifies_on_server():
    key = _test_key()
    payload = build_npz_stored(to_numpy_state(new_model(seed=6)))
    env = make_envelope(
        update_id="phone-01-r0003-x", client_id="phone-01", model_version=MODEL_VERSION,
        round_number=3, sample_count=347, payload=payload, signing_key_id="phone-01-key",
        private_key=key,
    )
    server_env = UpdateEnvelope(**env)
    assert verify_envelope_signature(server_env, key.public_key())
    tampered = server_env.model_copy(update={"sample_count": 348})
    assert not verify_envelope_signature(tampered, key.public_key())


def _vector_fields(created_at: str) -> dict:
    p = VECTORS["payload"]
    return {
        "update_id": "esp32-01-r0001-a1b2c3", "client_id": "esp32-01", "model_version": MODEL_VERSION,
        "round_number": 1, "sample_count": 341, "payload_sha256": p["sha256"],
        "payload_size_bytes": p["size_bytes"], "created_at": created_at, "signing_key_id": "esp32-01-key",
    }
