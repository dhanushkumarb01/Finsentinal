"""Host tests for the ESP32 firmware's pure C++ headers (demo/wokwi/common/).

Compiles demo/tests/firmware/host_test.cpp with g++ (PATH, then /c/MinGW/bin/g++,
then `wsl g++`) and checks it against the Python reference implementation:
  (a) forward logits + one SGD epoch vs PyTorch fldemo.model (1e-4)
  (b) NPZ bytes of deterministic_state() == the FinSentinel vector payload
  (c) canonical envelope == vector canonical_signed_data
  (d) SHA-256 / base64 known answers
  (e) model-bytes load/store round trip
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parent.parent
COMMON = DEMO / "wokwi" / "common"
DRIVER = DEMO / "tests" / "firmware" / "host_test.cpp"
VECTORS = json.loads((DEMO / "tests" / "vectors" / "finsentinel_vectors.json").read_text())


def _to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = f"/mnt/{s[0].lower()}{s[2:]}"
    return s


class Runner:
    def __init__(self, exe: Path, wsl: bool):
        self.exe, self.wsl = exe, wsl

    def path(self, p: Path) -> str:
        return _to_wsl(p) if self.wsl else str(p)

    def __call__(self, *args: str) -> str:
        cmd = (["wsl", _to_wsl(self.exe)] if self.wsl else [str(self.exe)]) + list(args)
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        assert r.returncode == 0, f"{cmd} failed ({r.returncode}): {r.stderr.decode(errors='replace')}"
        return r.stdout.decode()


def _candidates():
    found = shutil.which("g++")
    if found:
        yield ("native", found)
    for p in (r"C:\MinGW\bin\g++.exe", "/c/MinGW/bin/g++"):
        if os.path.exists(p):
            yield ("native", p)
    if shutil.which("wsl"):
        yield ("wsl", "g++")


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Runner:
    out = tmp_path_factory.mktemp("fwbuild")
    errors = []
    flags = ["-std=c++11", "-O1", "-Wall", "-D__USE_MINGW_ANSI_STDIO=1"]
    for kind, gxx in _candidates():
        if kind == "native":
            exe = out / "host_test.exe"
            cmd = [gxx, *flags, "-I", str(COMMON), str(DRIVER), "-o", str(exe)]
            if os.name == "nt":
                cmd.append("-static")  # avoid depending on MinGW DLLs on PATH
        else:
            exe = out / "host_test"
            cmd = ["wsl", gxx, *flags, "-I", _to_wsl(COMMON), _to_wsl(DRIVER), "-o", _to_wsl(exe)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"{cmd[0]}: {e}")
            continue
        if r.returncode == 0 and exe.exists():
            runner = Runner(exe, kind == "wsl")
            try:
                runner("sha256", runner.path(DRIVER))
                return runner
            except AssertionError as e:  # built but cannot run (e.g. missing DLL)
                errors.append(f"{gxx}: built but failed to run: {e}")
                continue
        errors.append(f"{' '.join(cmd[:2])}: {r.stderr.decode(errors='replace')[-2000:]}")
    pytest.skip("No working g++ found (PATH, C:/MinGW, wsl). Errors:\n" + "\n".join(errors))


def _model_bytes(state) -> bytes:
    from fldemo.model import TENSOR_SPECS

    return b"".join(np.asarray(state[n], dtype="<f4").tobytes() for n, _ in TENSOR_SPECS)


def _har_like(n: int, seed: int):
    rng = np.random.default_rng(seed)
    q = np.clip(np.round(rng.uniform(-1, 1, size=(n, 64)) * 127), -127, 127).astype(np.int8)
    y = rng.integers(0, 6, size=n).astype(np.uint8)
    x = q.astype(np.float32) * np.float32(1.0 / 127.0)  # same ops as the firmware
    return q, x, y


# --------------------------------------------------------------------------- (a)
def test_forward_matches_torch(run, tmp_path):
    import torch
    from fldemo.model import new_model, to_numpy_state

    m = new_model(seed=3)
    q, x, _ = _har_like(50, seed=1)
    (tmp_path / "m.bin").write_bytes(_model_bytes(to_numpy_state(m)))
    (tmp_path / "x.bin").write_bytes(q.tobytes())
    run("forward", run.path(tmp_path / "m.bin"), run.path(tmp_path / "x.bin"), "50", run.path(tmp_path / "o.bin"))
    got = np.frombuffer((tmp_path / "o.bin").read_bytes(), dtype="<f4").reshape(50, 6)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=0)


@pytest.mark.parametrize("shuffled", [False, True])
def test_sgd_epoch_matches_torch(run, tmp_path, shuffled):
    import torch
    from torch import nn
    from fldemo.model import new_model, to_numpy_state, train_sgd

    n, lr, bs, seed = 341, 0.05, 32, 7
    m = new_model(seed=0)
    q, x, y = _har_like(n, seed=2)
    (tmp_path / "m.bin").write_bytes(_model_bytes(to_numpy_state(m)))
    (tmp_path / "x.bin").write_bytes(q.tobytes())
    (tmp_path / "y.bin").write_bytes(y.tobytes())
    if shuffled:
        # Reproduce train_sgd's permutation exactly and hand it to the C++ side.
        perm = np.random.default_rng(seed).permutation(n)
        (tmp_path / "ord.bin").write_bytes(perm.astype("<u2").tobytes())
        order = run.path(tmp_path / "ord.bin")
        losses = train_sgd(m, x, y.astype(np.int64), epochs=1, lr=lr, batch_size=bs, seed=seed)
        ref_loss = losses[0]
    else:
        # Deterministic order (no shuffle): plain sequential mini-batches.
        order = "-"
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
        total = 0.0
        for i in range(0, n, bs):
            opt.zero_grad()
            loss = loss_fn(m(xt[i : i + bs]), yt[i : i + bs])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yt[i : i + bs])
        ref_loss = total / n
    out = run("epoch", run.path(tmp_path / "m.bin"), run.path(tmp_path / "x.bin"), run.path(tmp_path / "y.bin"),
              str(n), repr(lr), str(bs), order, run.path(tmp_path / "out.bin"))
    got_loss = float(out.split()[1])
    got = np.frombuffer((tmp_path / "out.bin").read_bytes(), dtype="<f4")
    ref = np.frombuffer(_model_bytes(to_numpy_state(m)), dtype="<f4")
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=0)
    assert abs(got_loss - ref_loss) < 1e-4, (got_loss, ref_loss)


# --------------------------------------------------------------------------- (b)
def test_npz_matches_vector(run, tmp_path):
    from fldemo.testvectors import deterministic_state
    from fldemo.wire import build_npz_stored

    state = deterministic_state()
    (tmp_path / "m.bin").write_bytes(_model_bytes(state))
    run("npz", run.path(tmp_path / "m.bin"), run.path(tmp_path / "p.npz"))
    got = (tmp_path / "p.npz").read_bytes()
    assert got == build_npz_stored(state)
    assert len(got) == VECTORS["payload"]["size_bytes"]
    assert hashlib.sha256(got).hexdigest() == VECTORS["payload"]["sha256"]
    assert base64.b64encode(got).decode() == VECTORS["payload"]["base64"]
    # And the firmware's own SHA-256 + base64 agree on the payload.
    assert run("sha256", run.path(tmp_path / "p.npz")).strip() == VECTORS["payload"]["sha256"]
    assert run("b64enc", run.path(tmp_path / "p.npz")) == VECTORS["payload"]["base64"]


def test_npz_trained_weights_match_wire(run, tmp_path):
    from fldemo.model import new_model, to_numpy_state
    from fldemo.wire import build_npz_stored

    state = to_numpy_state(new_model(seed=11))
    (tmp_path / "m.bin").write_bytes(_model_bytes(state))
    run("npz", run.path(tmp_path / "m.bin"), run.path(tmp_path / "p.npz"))
    assert (tmp_path / "p.npz").read_bytes() == build_npz_stored(state)


# --------------------------------------------------------------------------- (c)
def test_canonical_envelope_matches_vector(run):
    v = VECTORS["envelopes"][0]
    env = json.loads(v["canonical_signed_data"])
    got = run("envelope", env["client_id"], env["created_at"], env["model_version"], env["payload_sha256"],
              str(env["payload_size_bytes"]), str(env["round_number"]), str(env["sample_count"]),
              env["signing_key_id"], env["update_id"])
    assert got == v["canonical_signed_data"]


def test_canonical_envelope_matches_wire(run):
    from fldemo.wire import canonical_envelope

    env = {"update_id": "esp32-03-r0012-0badf00d", "client_id": "esp32-03", "model_version": "har-mlp-v1",
           "round_number": 12, "sample_count": 408, "payload_sha256": "ab" * 32, "payload_size_bytes": 10687,
           "created_at": "2026-11-22T23:59:59Z", "signing_key_id": "esp32-03-key"}
    got = run("envelope", env["client_id"], env["created_at"], env["model_version"], env["payload_sha256"],
              str(env["payload_size_bytes"]), str(env["round_number"]), str(env["sample_count"]),
              env["signing_key_id"], env["update_id"])
    assert got.encode() == canonical_envelope(env)


# --------------------------------------------------------------------------- (d)
@pytest.mark.parametrize("data", [b"", b"abc", b"a" * 55, b"a" * 56, b"a" * 64, bytes(range(256)) * 41])
def test_sha256_and_base64_known_answers(run, tmp_path, data):
    f = tmp_path / "d.bin"
    f.write_bytes(data)
    assert run("sha256", run.path(f)).strip() == hashlib.sha256(data).hexdigest()
    enc = base64.b64encode(data).decode()
    assert run("b64enc", run.path(f)) == enc
    (tmp_path / "e.txt").write_text(enc)
    out = run("b64dec", run.path(tmp_path / "e.txt"), run.path(tmp_path / "back.bin"))
    assert out.startswith("ok")
    assert (tmp_path / "back.bin").read_bytes() == data


def test_sha256_nist_vector(run, tmp_path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"abc")
    assert run("sha256", run.path(f)).strip() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("bad", ["abc", "ab=c", "a===", "####"])
def test_base64_rejects_malformed(run, tmp_path, bad):
    (tmp_path / "e.txt").write_text(bad)
    assert run("b64dec", run.path(tmp_path / "e.txt"), run.path(tmp_path / "o.bin")).strip() == "invalid"


# --------------------------------------------------------------------------- (e)
def test_model_bytes_roundtrip(run, tmp_path):
    from fldemo.model import new_model, to_numpy_state

    raw = _model_bytes(to_numpy_state(new_model(seed=5)))
    assert len(raw) == 9112
    (tmp_path / "m.bin").write_bytes(raw)
    run("roundtrip", run.path(tmp_path / "m.bin"), run.path(tmp_path / "o.bin"))
    assert (tmp_path / "o.bin").read_bytes() == raw


# --------------------------------------------------------------------------- parsing helpers
def test_parse_round_json(run, tmp_path):
    body = {"round": 3, "state": "open", "model_version": "har-mlp-v1", "model_sha256": "cd" * 32,
            "model_sig": base64.b64encode(bytes(64)).decode(), "deadline": 1795356400.0,
            "server_time": 1795356340.2, "epochs": 2, "lr": 0.05, "batch_size": 16}
    for text in (json.dumps(body, separators=(",", ":")), json.dumps(body, indent=1)):
        (tmp_path / "r.json").write_text(text)
        out = run("round", run.path(tmp_path / "r.json"))
        assert f"round=3 state=open version=har-mlp-v1 sha={'cd' * 32} sig={body['model_sig']} " in out
        assert "epochs=2 lr=0.0500 batch=16 deadline=1795356400.0" in out


def test_parse_beacon(run, tmp_path):
    line = "FLB1|https://abc.trycloudflare.com|1795356340|" + base64.b64encode(bytes(range(64))).decode()
    (tmp_path / "b.txt").write_text(line + "\n")
    out = run("beacon", run.path(tmp_path / "b.txt"))
    assert out.strip() == (f"url=https://abc.trycloudflare.com ts=1795356340 "
                           f"sig={line.rsplit('|', 1)[1]} signed_len={len(line.rsplit('|', 1)[0])}")
    (tmp_path / "b.txt").write_text("FLB1|http://plain|1|x")
    assert run("beacon", run.path(tmp_path / "b.txt")).strip() == "invalid"


# --------------------------------------------------------------------------- generated board data
@pytest.mark.parametrize("n", [1, 2, 3])
def test_board_data_matches_har_subject(n):
    import re

    path = DEMO / "wokwi" / f"board{n}" / "board_data.h"
    if not path.exists():
        pytest.skip("run demo/scripts/gen_firmware.py first")
    try:
        from fldemo.har import subject_arrays
        split = json.loads((DEMO / "config" / "har_split.json").read_text())["wokwi_boards"]
        x, y = subject_arrays([split[f"esp32-{n:02d}"]])
    except FileNotFoundError:
        pytest.skip("UCI HAR dataset not downloaded")
    text = path.read_text()
    assert int(re.search(r"#define FL_N (\d+)", text).group(1)) == len(y)
    ys = re.search(r"FL_Y\[FL_N\] PROGMEM = \{([^}]*)\}", text).group(1)
    assert [int(v) for v in ys.split(",")] == y.tolist()
    xs = text.split("FL_X[FL_N][64] PROGMEM = {", 1)[1]
    rows = re.findall(r"\{([-0-9,]+)\}", xs)
    q = np.array([[int(v) for v in r.split(",")] for r in rows], dtype=np.int8)
    assert q.shape == (len(y), 64)
    assert np.abs(q.astype(np.float32) / 127 - x).max() <= 0.5 / 127 + 1e-6
