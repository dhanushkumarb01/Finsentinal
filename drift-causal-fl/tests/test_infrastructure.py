"""Step 8 infrastructure tests: config validation, run expansion, locking,
skip-completed, failure recording, interrupted-run resume via the sweep layer.
Synthetic data only (load_run_data is monkeypatched)."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.drift.injection import make_drift_schedule  # noqa: E402
from src.experiments import sweep  # noqa: E402
from src.experiments.config import ConfigError, expand_runs, load_study, validate_study  # noqa: E402
from test_steps1_5 import _client  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _mvp():
    return load_study(CONFIGS / "mvp.yaml")


def test_mvp_matrix_is_exactly_the_spec_350():
    runs = expand_runs(_mvp())
    assert len(runs) == 350
    assert len({r.run_id for r in runs}) == 350
    conds = {r.condition for r in runs}
    assert conds == {"mcar", "mar", "mnar_b0.5", "mnar_b1", "mnar_b2", "mnar_b4", "mnar_b8"}
    assert {r.method for r in runs} == {"fedavg", "fedipw_oracle", "fedipw_estimated", "flfdms", "feddrift"}
    assert {r.seed for r in runs} == {0, 1, 2, 3, 4}
    assert {r.drift_estimator for r in runs} == {"model_relative", "model_independent"}
    # every (seed, estimator, condition, method) cell exactly once
    cells = {(r.seed, r.drift_estimator, r.condition, r.method) for r in runs}
    assert len(cells) == 350


def test_operating_point_matches_spec():
    cfg = _mvp()
    assert cfg["data"]["n_clients"] == 100 and cfg["training"]["rounds"] == 100
    assert cfg["training"]["local_epochs"] == 1 and cfg["training"]["eval_every"] == 5
    assert cfg["participation"]["target_rate"] == 0.3 and cfg["drift"]["fraction"] == 0.3
    assert sorted(c["beta"] for c in cfg["conditions"] if c["mechanism"] == "mnar") == [0.5, 1, 2, 4, 8]


def test_run_kwargs_and_hash_are_deterministic():
    a, b = expand_runs(_mvp()), expand_runs(_mvp())
    assert [r.config_hash() for r in a] == [r.config_hash() for r in b]
    r = next(x for x in a if x.run_id == "s3__mind__mnar_b4__feddrift")
    kw = r.runner_kwargs()
    assert kw["beta"] == 4.0 and kw["drift_estimator"] == "model_independent" and kw["seed"] == 3
    assert kw["method_kwargs"] == {"delta": 0.04, "max_models": 10}
    assert len({x.config_hash() for x in a}) == 350


@pytest.mark.parametrize("mutate, msg", [
    (lambda c: c.pop("seeds"), "missing top-level"),
    (lambda c: c["conditions"].append({"mechanism": "mnar"}), "beta"),
    (lambda c: c["conditions"].append({"mechanism": "mcar", "beta": 1}), "beta only"),
    (lambda c: c["conditions"].append({"mechanism": "mcar"}), "duplicate condition"),
    (lambda c: c["methods"].append({"name": "scaffold"}), "unknown method"),
    (lambda c: c["participation"].update(target_rate=0.99), "positivity"),
    (lambda c: c["drift"].update(mode="rotation"), "label_shift"),
    (lambda c: c["data"].update(n_reserve=5), "n_reserve"),
    (lambda c: c.update(expected_total_runs=175), "expected_total_runs"),
    (lambda c: c["methods"][0].update(kwargs={"delta": 1}), "takes no kwargs"),
    (lambda c: c.update(seeds=[0, 0, 1]), "unique"),
])
def test_invalid_configs_rejected(mutate, msg):
    cfg = copy.deepcopy(_mvp())
    mutate(cfg)
    with pytest.raises(ConfigError, match=msg):
        validate_study(cfg)


# --------------------------------------------------------------- sweep layer
def _tiny_cfg():
    cfg = copy.deepcopy(_mvp())
    cfg["study"] = "tiny"
    cfg["data"].update(n_clients=6, n_reserve=4)
    cfg["drift"].update(fraction=0.5, start_round=2, ramp_rounds=2)
    cfg["training"].update(rounds=4, eval_every=2, batch_size=6, lr=0.01)
    cfg["conditions"] = [{"mechanism": "mcar"}, {"mechanism": "mnar", "beta": 2}]
    cfg["methods"] = [{"name": "fedavg"}, {"name": "feddrift", "kwargs": {"delta": 0.01, "max_models": 3}}]
    cfg["seeds"] = [0]
    cfg["drift_estimators"] = ["model_relative"]
    cfg.pop("expected_total_runs")
    return cfg


@pytest.fixture
def synthetic_data(monkeypatch):
    def fake(spec):
        clients = [_client(i, 12, 5, i) for i in range(6)]
        reserve = [_client(-(i + 1), 12, 5, 100 + i, n_test=6) for i in range(4)]
        sched = make_drift_schedule([c.client_id for c in clients], 0.5, 2, np.random.default_rng(spec.seed),
                                    ramp_rounds=2, reserve_pool=reserve)
        return clients, reserve, 5, sched, {"synthetic": True}
    monkeypatch.setattr(sweep, "load_run_data", fake)


def test_execute_skip_done_and_report(tmp_path, synthetic_data):
    runs = expand_runs(_tiny_cfg())
    for r in runs:
        assert sweep.execute_run(r, tmp_path, checkpoint_every=2) == "done"
    d = tmp_path / runs[0].run_id
    for f in ("spec.json", "metrics.jsonl", "trace.npz", "result.json", "attempts.jsonl"):
        assert (d / f).exists(), f
    assert not (d / "lock").exists() and not (d / "checkpoint.pt").exists()
    spec = json.loads((d / "spec.json").read_text())
    assert spec["config_hash"] == runs[0].config_hash() and "torch" in spec["environment"]
    mtime = (d / "result.json").stat().st_mtime_ns
    assert sweep.execute_run(runs[0], tmp_path) == "done"          # not re-executed
    assert (d / "result.json").stat().st_mtime_ns == mtime
    rep = sweep.sweep_report(runs, tmp_path)
    assert rep["counts"] == {"done": len(runs)} and rep["planned"] == len(runs)


def test_lock_prevents_duplicate_execution_and_stale_lock_reclaimed(tmp_path, synthetic_data):
    r = expand_runs(_tiny_cfg())[0]
    d = tmp_path / r.run_id
    assert sweep.try_claim(d)
    assert not sweep.try_claim(d)                                   # second claimant refused
    assert sweep.execute_run(r, tmp_path) == "running"
    sweep.release(d)
    # a lock left by a dead process on this host is stale and reclaimable
    (d / "lock").write_text(json.dumps({"pid": 2**22 + 12345, "host": sweep.socket.gethostname(), "time": 0}))
    assert sweep.run_status(d) == "stale_lock"
    assert sweep.execute_run(r, tmp_path) == "done"


def test_failure_recorded_not_silently_retried(tmp_path, synthetic_data, monkeypatch):
    r = expand_runs(_tiny_cfg())[0]
    calls = {"n": 0}

    def boom(**kw):
        calls["n"] += 1
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(sweep, "run_mechanism_experiment", boom)
    assert sweep.execute_run(r, tmp_path) == "failed"
    d = tmp_path / r.run_id
    assert "synthetic failure" in (d / "error.txt").read_text()
    att = [json.loads(x) for x in (d / "attempts.jsonl").read_text().splitlines()]
    assert att[-1]["status"] == "failed" and "synthetic failure" in att[-1]["error"]
    assert sweep.execute_run(r, tmp_path) == "failed" and calls["n"] == 1   # not retried
    monkeypatch.undo()
    monkeypatch.setattr(sweep, "load_run_data", synthetic_data_loader())
    assert sweep.execute_run(r, tmp_path, retry_failed=True) == "done"     # explicit retry
    assert not (d / "error.txt").exists()
    rep = sweep.sweep_report([r], tmp_path)
    assert rep["recovered_or_rerun"] == [r.run_id]


def synthetic_data_loader():
    def fake(spec):
        clients = [_client(i, 12, 5, i) for i in range(6)]
        reserve = [_client(-(i + 1), 12, 5, 100 + i, n_test=6) for i in range(4)]
        sched = make_drift_schedule([c.client_id for c in clients], 0.5, 2, np.random.default_rng(spec.seed),
                                    ramp_rounds=2, reserve_pool=reserve)
        return clients, reserve, 5, sched, {"synthetic": True}
    return fake


def test_interrupted_run_resumes_identically(tmp_path, synthetic_data, monkeypatch):
    r = next(x for x in expand_runs(_tiny_cfg()) if x.method == "feddrift" and x.mechanism == "mnar")
    ref = tmp_path / "ref"
    sweep.execute_run(r, ref, checkpoint_every=2)
    real = sweep.run_mechanism_experiment

    def interrupted(**kw):
        real(**kw, stop_after_round=3)
        raise KeyboardInterrupt  # simulate the process being killed after round 3
    monkeypatch.setattr(sweep, "run_mechanism_experiment", interrupted)
    with pytest.raises(KeyboardInterrupt):
        sweep.execute_run(r, tmp_path / "x", checkpoint_every=2)
    d = tmp_path / "x" / r.run_id
    (d / "lock").unlink(missing_ok=True)   # the killed process's lock (pid would be dead in reality)
    assert sweep.run_status(d) == "interrupted"
    monkeypatch.setattr(sweep, "run_mechanism_experiment", real)
    assert sweep.execute_run(r, tmp_path / "x", checkpoint_every=2) == "done"
    a = json.loads((ref / r.run_id / "result.json").read_text())
    b = json.loads((d / "result.json").read_text())
    strip = lambda rows: [{k: v for k, v in e.items() if k != "elapsed_sec"} for e in rows]  # noqa: E731
    assert strip(a["rounds"]) == strip(b["rounds"])
    assert b["summary"]["resumed_from_round"] == 2


def test_spec_hash_mismatch_refused(tmp_path, synthetic_data):
    r = expand_runs(_tiny_cfg())[0]
    d = tmp_path / r.run_id
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"config_hash": "different"}))
    assert sweep.execute_run(r, tmp_path) == "failed"
    assert "different config hash" in (d / "error.txt").read_text()


def test_configs_parse_with_yaml_safe_load():
    for f in CONFIGS.glob("*.yaml"):
        validate_study(yaml.safe_load(f.read_text()))
    assert os.path.exists(CONFIGS / "mvp.yaml")


def test_concurrency_slots_cap_memory_heavy_methods(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_sweep
    lim = {"flfdms": 1}
    assert run_sweep._claim_slot(tmp_path, "fedavg", lim) is True        # uncapped
    s1 = run_sweep._claim_slot(tmp_path, "flfdms", lim)
    assert isinstance(s1, Path)
    assert run_sweep._claim_slot(tmp_path, "flfdms", lim) is None         # cap reached
    sweep.release(s1)
    assert isinstance(run_sweep._claim_slot(tmp_path, "flfdms", lim), Path)
