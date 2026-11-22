"""Unit tests: NumPy model vs PyTorch, gate-light mapping, model bytes, drift estimator, keys/beacon."""

import numpy as np
import pytest

from fldemo import keys as K
from fldemo import npmodel
from fldemo.coordinator import detail_text, gate_checks, npz_to_state
from fldemo.drift import DriftTracker
from fldemo.fleet import bytes_to_state, device_data, local_update, poison, sha256_hex, state_to_bytes, weight_delta_grid
from fldemo.har import subject_arrays
from fldemo.model import new_model, to_numpy_state
from fldemo.settings import load_devices
from fldemo.wire import build_npz_stored


@pytest.fixture(scope="module")
def data():
    return subject_arrays([1])


def test_numpy_training_matches_pytorch(data):
    x, y = data
    g = to_numpy_state(new_model(seed=0))
    a, la = local_update(g, x, y, epochs=2, lr=0.1, batch_size=32, seed=7)
    b, lb = npmodel.train_sgd(g, x, y, epochs=2, lr=0.1, batch_size=32, seed=7)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=2e-5)
    np.testing.assert_allclose(la, lb, atol=1e-4)


def test_numpy_eval_matches_pytorch(data):
    from fldemo.fleet import eval_state
    x, y = data
    g = to_numpy_state(new_model(seed=3))
    acc_t, loss_t = eval_state(g, x, y)
    loss_n, acc_n = npmodel.loss_acc(g, x, y)
    assert abs(acc_t - acc_n) < 1e-9 and abs(loss_t - loss_n) < 1e-5


def test_model_bytes_roundtrip():
    g = to_numpy_state(new_model(seed=1))
    blob = state_to_bytes(g)
    assert len(blob) == 9112
    back = bytes_to_state(blob)
    assert all(np.array_equal(g[k], back[k]) for k in g)
    with pytest.raises(ValueError):
        bytes_to_state(blob[:-4])


def test_npz_parse_roundtrip():
    g = to_numpy_state(new_model(seed=2))
    back = npz_to_state(build_npz_stored(g))
    assert all(np.array_equal(g[k], back[k]) for k in g)


@pytest.mark.parametrize("status,detail,sev,expected", [
    (200, "", "LOW", {"untampered": "pass", "sender": "pass", "shape": "pass", "replay": "pass", "normal": "pass"}),
    (200, "", "MEDIUM", {"untampered": "pass", "sender": "pass", "shape": "pass", "replay": "pass", "normal": "warn"}),
    (200, "", "CRITICAL", {"untampered": "pass", "sender": "pass", "shape": "pass", "replay": "pass", "normal": "fail"}),
    (422, "Payload SHA-256 does not match envelope", None,
     {"untampered": "fail", "sender": "skip", "shape": "skip", "replay": "skip", "normal": "skip"}),
    (401, "Invalid update signature", None,
     {"untampered": "pass", "sender": "fail", "shape": "skip", "replay": "skip", "normal": "skip"}),
    (422, "Invalid serialized model update: bad", None,
     {"untampered": "pass", "sender": "pass", "shape": "fail", "replay": "skip", "normal": "skip"}),
    (403, "stale update", None,
     {"untampered": "pass", "sender": "pass", "shape": "fail", "replay": "skip", "normal": "skip"}),
    (409, "Duplicate update", None,
     {"untampered": "pass", "sender": "pass", "shape": "pass", "replay": "fail", "normal": "skip"}),
])
def test_gate_checks_follow_finsentinel_pipeline_order(status, detail, sev, expected):
    assert gate_checks(status, detail, sev) == expected


def test_detail_text_handles_policy_dict():
    assert detail_text({"detail": {"code": "stale", "message": "too old"}}) == "too old"
    assert detail_text({"detail": "x"}) == "x"


def test_drift_estimator_zero_when_unchanged_positive_when_drifted():
    dev = next(d for d in load_devices() if d.get("drift"))
    t = DriftTracker()
    x0, _ = device_data(dev, drifted=False, round_number=1)
    assert t.score(x0) == 0.0
    assert t.score(device_data(dev, drifted=False, round_number=2)[0]) == 0.0
    xd, yd = device_data(dev, drifted=True, round_number=3)
    assert t.score(xd) > 0.05
    assert set(np.unique(yd)) <= set(dev["drift"]["keep_classes"])


def test_poison_scales_weights():
    g = to_numpy_state(new_model(seed=0))
    p = poison(g)
    assert np.allclose(p["fc1.weight"], g["fc1.weight"] * 8)


def test_weight_delta_grid_shape_and_range():
    a = to_numpy_state(new_model(seed=0))
    b = to_numpy_state(new_model(seed=1))
    grid = weight_delta_grid(a, b)
    assert len(grid) == 8 and all(len(r) == 16 for r in grid)
    assert max(max(r) for r in grid) == 1.0 and min(min(r) for r in grid) >= 0.0


def test_model_and_beacon_signatures(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate()
    sha = sha256_hex(b"abc")
    msg = K.model_sig_message("har-mlp-v1", 3, sha)
    sig = K.sign_b64(key, msg)
    assert K.verify_b64(key.public_key(), msg, sig)
    assert not K.verify_b64(key.public_key(), K.model_sig_message("har-mlp-v1", 4, sha), sig)
    b = K.beacon_message("https://x.lhr.life", 1795356000)
    assert b == "FLB1|https://x.lhr.life|1795356000"
    assert K.verify_b64(key.public_key(), b, K.sign_b64(key, b))
