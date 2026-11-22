"""Tests for Steps 6-8: Strategy refactor, FL-FDMS, FedDrift, information
invariant, checkpoint/resume. Synthetic data only.

Run from drift-causal-fl/:  python -m pytest tests -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.methods.base import RoundContext, Strategy, StreamingWeightedAverage  # noqa: E402
from src.methods.feddrift import FedDriftStrategy  # noqa: E402
from src.methods.flfdms import FLFDMSStrategy  # noqa: E402
from src.simulation import mechanism_runner  # noqa: E402
from src.simulation.fedavg_baseline import fedavg_aggregate  # noqa: E402
from src.simulation.mechanism_runner import _jsonable, build_strategy  # noqa: E402
from test_steps1_5 import _run, _setup  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_runner_outputs.json"


class Lin(nn.Module):
    """Tiny model for strategy-level unit tests."""

    def __init__(self, d=3, k=2):
        super().__init__()
        self.fc = nn.Linear(d, k, bias=False)

    def forward(self, x):
        return self.fc(x)


def _state(vals):
    return {"fc.weight": torch.tensor(vals, dtype=torch.float32).reshape(2, 3)}


# ------------------------------------------------------- refactor regression
@pytest.mark.parametrize("key", sorted(json.loads(FIXTURE.read_text())))
def test_refactor_bit_identical_to_legacy_loop(key):
    """FedAvg / FedIPW results must not change because of the Strategy refactor."""
    exp = json.loads(FIXTURE.read_text())[key]
    mech, meth, est = key.split("|")
    c, r, s = _setup()
    res = _run(c, r, s, mechanism=mech, method=meth, drift_estimator=est, rounds=4, eval_every=1,
               threshold_exclude_frac=0.15)
    rows = [{k: v for k, v in e.items() if k not in ("elapsed_sec", "method_log")} for e in res["rounds"]]
    summ = {k: v for k, v in res["summary"].items() if k != "wall_clock_sec"}
    got = json.loads(json.dumps(_jsonable({"rounds": rows, "summary": summ, "per_client_final": res["per_client_final"]})))
    assert got == exp


def test_streaming_average_equals_fedavg_aggregate():
    g = torch.Generator().manual_seed(0)
    sds = [{"a": torch.randn(4, 3, generator=g), "b": torch.randn(5, generator=g)} for _ in range(7)]
    w = [3, 1, 4, 1, 5, 9, 2]
    acc = StreamingWeightedAverage(sum(w))
    for sd, wi in zip(sds, w):
        acc.add(sd, wi)
    ref = fedavg_aggregate(sds, w)
    for k in ref:
        assert torch.equal(acc.result()[k], ref[k])


# ------------------------------------------------------------------ FL-FDMS
def test_flfdms_similarity_eq14_and_running_mean_eq15():
    s = FLFDMSStrategy(Lin(), n_clients=3)
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    s.update_similarity([0, 1, 2], d)
    assert s.R[0, 1] == pytest.approx(0.5)      # cos 0  -> 0.5
    assert s.R[0, 2] == pytest.approx(0.0)      # cos -1 -> 0
    assert s.R[1, 0] == s.R[0, 1] and s.N[0, 1] == 1
    d2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    s.update_similarity([0, 1], d2)             # cos 1 -> 1; running mean (0.5 + 1)/2
    assert s.R[0, 1] == pytest.approx(0.75) and s.N[0, 1] == 2
    assert s.R[0, 2] == pytest.approx(0.0) and s.N[0, 2] == 1  # pair not co-active: unchanged
    for _ in range(2):
        s.update_similarity([0, 1], torch.tensor([[0.0, 1.0], [0.0, -1.0]]))  # r = 0 twice
    assert s.R[0, 1] == pytest.approx((0.5 + 1 + 0 + 0) / 4)  # exact arithmetic mean, not an EMA


def test_flfdms_zero_update_pair_skipped():
    s = FLFDMSStrategy(Lin(), n_clients=2)
    s.update_similarity([0, 1], torch.tensor([[0.0, 0.0], [1.0, 2.0]]))
    assert s.N[0, 1] == 0


def test_flfdms_friend_selection_argmax_over_active_with_history():
    s = FLFDMSStrategy(Lin(), n_clients=5)
    s.last_n[:] = 10
    s.R[4, 0], s.N[4, 0] = 0.6, 3
    s.R[4, 1], s.N[4, 1] = 0.9, 2
    s.R[4, 2], s.N[4, 2] = 0.95, 1   # best overall, but client 2 is dropped this round
    s.R[3, 0], s.N[3, 0] = 0.7, 1
    s.R[3, 1], s.N[3, 1] = 0.7, 1    # tie -> lowest index
    friends, no_friend = s.select_friends([0, 1])
    assert friends == {4: 1, 3: 0}
    assert no_friend == [2]          # never co-active with 0 or 1


def test_flfdms_cold_start_and_unknown_size_not_substituted():
    s = FLFDMSStrategy(Lin(), n_clients=3)
    s.R[2, 0], s.N[2, 0] = 0.8, 1
    friends, no_friend = s.select_friends([0, 1])
    assert friends == {} and no_friend == [2]     # client 2 never reported a sample count


def test_flfdms_substitution_weights_toy():
    """Dropped client 2 (n=30, friend 0) adds its weight to the friend's
    current update: w = (10+30)/(10+20+30) * s0 + 20/60 * s1."""
    model = Lin()
    s = FLFDMSStrategy(model, n_clients=3)
    s.last_n[2] = 30
    s.R[2, 0], s.N[2, 0] = 0.9, 1
    s.R[2, 1], s.N[2, 1] = 0.1, 1
    ctx = RoundContext(rnd=1, n_clients=3, participants=[0, 1], n_samples={0: 10, 1: 20})
    s.begin_round(ctx)
    s0, s1 = _state([1, 0, 0, 0, 0, 0]), _state([0, 1, 0, 0, 0, 0])
    s.receive(0, s0)
    s.receive(1, s1)
    s.end_round(ctx)
    exp = s0["fc.weight"] * (40 / 60) + s1["fc.weight"] * (20 / 60)
    assert torch.allclose(model.fc.weight, exp)
    assert s.round_log()["friend_map"] == {2: 0}


def test_flfdms_full_participation_equals_fedavg():
    c, r, s = _setup()
    a = _run(c, r, s, mechanism="mcar", method="fedavg", target_rate=0.05, rounds=3, eval_every=1)
    c, r, s = _setup()
    b = _run(c, r, s, mechanism="mcar", method="flfdms", target_rate=0.05, rounds=3, eval_every=1)
    # identical whenever no-one dropped; check the first round (rate 0.05 with 6 clients)
    if a["rounds"][0]["n_participants"] == 6:
        assert a["rounds"][0]["acc_global"] == b["rounds"][0]["acc_global"]
    c, r, s = _setup()
    # direct check: no dropped clients => weights == sample counts
    st = FLFDMSStrategy(Lin(), 3)
    st.begin_round(RoundContext(rnd=1, n_clients=3, participants=[0, 1, 2], n_samples={0: 5, 1: 7, 2: 9}))
    assert st._weights == {0: 5.0, 1: 7.0, 2: 9.0}


# ----------------------------------------------------------------- FedDrift
def test_feddrift_cluster_distance_formula():
    L = {(0, 0): 1.0, (0, 1): 1.5, (1, 0): 1.1, (1, 1): 0.4}
    # max(L01 - L00, L10 - L11, 0) = max(0.5, 0.7, 0)
    assert FedDriftStrategy.cluster_distance(L, 0, 1) == pytest.approx(0.7)
    L2 = {(0, 0): 1.0, (0, 1): 0.5, (1, 0): 0.2, (1, 1): 0.4}
    assert FedDriftStrategy.cluster_distance(L2, 0, 1) == 0.0


def _xy(seed, n=20, d=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randint(0, 2, (n,), generator=g)


def test_feddrift_drift_test_spawns_singleton_copy_of_best():
    model = Lin()
    st = FedDriftStrategy(model, n_clients=2, delta=0.05, max_models=5)
    x, y = _xy(0)
    st.prev_min_loss[:] = [0.0, 100.0]       # client 0 will look drifted, client 1 not
    ctx = RoundContext(rnd=2, n_clients=2, participants=[0, 1], n_samples={0: 20, 1: 20}, data={0: (x, y), 1: (x, y)})
    st.begin_round(ctx)
    assert st.assign.tolist() == [1, 0]
    assert len(st.models) == 2 and st.n_models_created == 1
    assert torch.equal(st.models[1].fc.weight, st.models[0].fc.weight)   # F5: copy of best
    assert st.models[1] is not st.models[0]
    assert st.round_log()["flagged"] == [0]


def test_feddrift_first_participation_records_reference_only():
    st = FedDriftStrategy(Lin(), n_clients=1, delta=1e-9)
    x, y = _xy(1)
    st.begin_round(RoundContext(rnd=1, n_clients=1, participants=[0], n_samples={0: 20}, data={0: (x, y)}))
    assert st.round_log()["n_flagged"] == 0 and not math.isnan(st.prev_min_loss[0])


def test_feddrift_cap_blocks_new_models():
    st = FedDriftStrategy(Lin(), n_clients=2, delta=0.05, max_models=1)
    x, y = _xy(2)
    st.prev_min_loss[:] = 0.0
    st.begin_round(RoundContext(rnd=2, n_clients=2, participants=[0, 1], n_samples={0: 20, 1: 20},
                                data={0: (x, y), 1: (x, y)}))
    assert len(st.models) == 1 and st.round_log()["n_cap_blocked"] == 2 and st.assign.tolist() == [0, 0]


def _inject_losses(st, losses):
    """losses[c][m] = loss of model m on client c's data; client identified by y[0]."""
    st._loss = lambda model, x, y, device: losses[int(y[0])][next(m for m, mod in st.models.items() if mod is model)]
    return {c: (torch.zeros(1, 3), torch.tensor([c])) for c in losses}


def test_feddrift_merge_close_clusters_and_reassign_absent_clients():
    model = Lin()
    st = FedDriftStrategy(model, n_clients=3, delta=0.5)
    import copy
    st.models[1] = copy.deepcopy(model)
    st.data_count = {0: 10.0, 1: 30.0}
    st.next_id = 2
    st.assign[:] = [0, 1, 1]                 # client 2 absent this round, identity 1
    st.prev_min_loss[:] = 1e9
    data = _inject_losses(st, {0: {0: 1.00, 1: 1.01}, 1: {0: 1.01, 1: 1.00}})  # D(0,1) = 0.01
    st.begin_round(RoundContext(rnd=5, n_clients=3, participants=[0, 1], n_samples={0: 1, 1: 1}, data=data))
    assert sorted(st.models) == [0]
    assert st.assign.tolist() == [0, 0, 0]   # absent client 2 follows its merged cluster
    assert st.round_log()["merges"][0][:2] == [0, 1]


def test_feddrift_no_merge_when_cluster_has_no_participating_data():
    """F3: D(i,j) needs data from both clusters this round."""
    st = FedDriftStrategy(Lin(), n_clients=2, delta=0.5)
    import copy
    st.models[1] = copy.deepcopy(st.models[0])
    st.data_count = {0: 1.0, 1: 1.0}
    st.next_id = 2
    st.assign[:] = [0, 1]
    st.prev_min_loss[:] = 1e9
    data = _inject_losses(st, {0: {0: 1.0, 1: 1.0}})
    st.begin_round(RoundContext(rnd=5, n_clients=2, participants=[0], n_samples={0: 1}, data=data))
    assert sorted(st.models) == [0, 1] and st.assign.tolist() == [0, 1]


def test_feddrift_merge_weights_by_training_data():
    st = FedDriftStrategy(Lin(), n_clients=1, delta=0.1)
    st.models[0].load_state_dict(_state([1, 1, 1, 1, 1, 1]))
    m1 = Lin()
    m1.load_state_dict(_state([5, 5, 5, 5, 5, 5]))
    st.models[1], st.data_count = m1, {0: 30.0, 1: 10.0}
    st._merge(0, 1)
    assert torch.allclose(st.models[0].fc.weight, torch.full((2, 3), 2.0))  # 0.75*1 + 0.25*5
    assert st.data_count == {0: 40.0}


def test_feddrift_complete_linkage_update():
    """Three clusters: D(0,1)=0.01 merges; D(0,2)=0.02 but D(1,2)=0.3, so
    complete linkage gives D({0,1},2)=0.3 >= delta and 2 is NOT merged."""
    st = FedDriftStrategy(Lin(), n_clients=3, delta=0.1)
    import copy
    st.models[1], st.models[2] = copy.deepcopy(st.models[0]), copy.deepcopy(st.models[0])
    st.data_count = {0: 1.0, 1: 1.0, 2: 1.0}
    st.next_id = 3
    st.assign[:] = [0, 1, 2]
    st.prev_min_loss[:] = 1e9
    # inject losses: L_ij = loss of model i on cluster j's data = losses[j][i]
    st._loss = lambda model, x, y, device: losses[int(y[0])][next(m for m, mod in st.models.items() if mod is model)]
    data = {c: (torch.zeros(1, 3), torch.tensor([c])) for c in range(3)}
    # each client's argmin is its own cluster; D(0,1)=0.01, D(0,2)=0.02, D(1,2)=0.30
    losses = {0: {0: 1.00, 1: 1.01, 2: 1.02}, 1: {0: 1.005, 1: 1.00, 2: 1.30}, 2: {0: 1.02, 1: 1.30, 2: 1.00}}
    st.begin_round(RoundContext(rnd=3, n_clients=3, participants=[0, 1, 2], n_samples={0: 1, 1: 1, 2: 1}, data=data))
    log = st.round_log()
    assert [m[:2] for m in log["merges"]] == [[0, 1]]
    assert sorted(st.models) == [0, 2]


def test_feddrift_without_drift_flags_equals_fedavg():
    c, r, s = _setup()
    a = _run(c, r, s, mechanism="mnar", method="fedavg", rounds=4, eval_every=1)
    c, r, s = _setup()
    b = _run(c, r, s, mechanism="mnar", method="feddrift", rounds=4, eval_every=1, method_kwargs={"delta": 1e6})
    assert [e["acc_global"] for e in a["rounds"]] == [e["acc_global"] for e in b["rounds"]]


# -------------------------------------------------------- information invariant
class SpyStrategy(Strategy):
    """Wraps a real strategy and records everything it is given."""

    def __init__(self, inner):
        self.inner = inner
        self.needs_participant_data = inner.needs_participant_data
        self.needs_oracle_propensity = inner.needs_oracle_propensity
        self.seen = []
        self.received = []

    def begin_round(self, ctx):
        self.seen.append(ctx)
        self.inner.begin_round(ctx)

    def start_state(self, idx):
        return self.inner.start_state(idx)

    def receive(self, idx, sd):
        self.received.append((self.seen[-1].rnd, idx))
        self.inner.receive(idx, sd)

    def end_round(self, ctx):
        self.inner.end_round(ctx)

    def eval_model(self, idx):
        return self.inner.eval_model(idx)

    def snapshot(self):
        return self.inner.snapshot()

    def round_log(self):
        return self.inner.round_log()

    def final_log(self):
        return self.inner.final_log()

    def get_state(self):
        return self.inner.get_state()


@pytest.mark.parametrize("method", ["fedavg", "fedipw_oracle", "fedipw_estimated", "flfdms", "feddrift"])
def test_strategies_only_see_participants(method, monkeypatch):
    spies = []

    def factory(m, model, n, kw=None):
        spy = SpyStrategy(build_strategy(m, model, n, kw))
        if m in ("fedavg", "fedipw_oracle", "fedipw_estimated", "flfdms"):
            # keep the single-model plumbing (lagged model) of the inner strategy
            spy.__class__ = type("Spy" + m, (SpyStrategy, type(spy.inner)), {})
            spy.global_model = spy.inner.global_model
        spies.append(spy)
        return spy

    monkeypatch.setattr(mechanism_runner, "build_strategy", factory)
    c, r, s = _setup()
    out = _run(c, r, s, mechanism="mnar", method=method, rounds=4, eval_every=1)
    spy = spies[0]
    for ctx in spy.seen:
        parts = set(ctx.participants)
        assert set(ctx.n_samples) == parts
        assert set(ctx.data) <= parts
        if method == "fedipw_oracle":
            assert set(ctx.oracle_propensity) == parts
        else:
            assert ctx.oracle_propensity is None
        for name in vars(ctx):
            assert "drift" not in name.lower() and name not in ("D", "D_true", "p_miss")
    assert {(rnd, i) for rnd, i in spy.received} == {(ctx.rnd, i) for ctx in spy.seen for i in ctx.participants}
    assert out["summary"]["final_acc_global"] is not None


# ------------------------------------------------------------- checkpointing
@pytest.mark.parametrize("method", ["fedavg", "fedipw_estimated", "flfdms", "feddrift"])
def test_checkpoint_resume_is_bit_identical(method, tmp_path):
    kw = {"delta": 0.01} if method == "feddrift" else None
    c, r, s = _setup()
    full = _run(c, r, s, mechanism="mnar", method=method, rounds=6, eval_every=2, method_kwargs=kw,
                run_dir=tmp_path / "full")
    c, r, s = _setup()
    part = _run(c, r, s, mechanism="mnar", method=method, rounds=6, eval_every=2, method_kwargs=kw,
                run_dir=tmp_path / "split", checkpoint_every=2, stop_after_round=3)
    assert part == {"interrupted_at": 3}
    assert (tmp_path / "split" / "checkpoint.pt").exists()
    c, r, s = _setup()
    resumed = _run(c, r, s, mechanism="mnar", method=method, rounds=6, eval_every=2, method_kwargs=kw,
                   run_dir=tmp_path / "split", checkpoint_every=2)
    strip = lambda rows: [{k: v for k, v in e.items() if k != "elapsed_sec"} for e in rows]  # noqa: E731
    assert strip(resumed["rounds"]) == strip(full["rounds"])
    assert resumed["per_client_final"] == full["per_client_final"]
    assert resumed["summary"]["resumed_from_round"] == 2
    assert not (tmp_path / "split" / "checkpoint.pt").exists()
    lines = (tmp_path / "split" / "metrics.jsonl").read_text().strip().splitlines()
    eval_rounds = [json.loads(line)["round"] for line in lines if "acc_global" in json.loads(line)]
    assert eval_rounds == [e["round"] for e in full["rounds"]]   # no duplicated rows after resume
    tr_full = np.load(tmp_path / "full" / "trace.npz")
    tr_split = np.load(tmp_path / "split" / "trace.npz")
    for k in tr_full.files:
        assert np.array_equal(tr_full[k], tr_split[k], equal_nan=True)


def test_resume_refuses_mismatched_config(tmp_path):
    c, r, s = _setup()
    _run(c, r, s, mechanism="mcar", rounds=4, run_dir=tmp_path, checkpoint_every=2, stop_after_round=2)
    c, r, s = _setup()
    with pytest.raises(ValueError, match="does not match"):
        _run(c, r, s, mechanism="mcar", rounds=4, lr=0.5, run_dir=tmp_path, checkpoint_every=2)


@pytest.mark.parametrize("method", ["fedavg", "fedipw_oracle", "fedipw_estimated", "flfdms", "feddrift"])
@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_all_methods_run_under_all_mechanisms(method, mechanism):
    c, r, s = _setup()
    out = _run(c, r, s, mechanism=mechanism, method=method, rounds=3, eval_every=1, drift_estimator="model_relative")
    for k in ("acc_global", "acc_drift_cohort", "acc_stable_cohort", "acc_clean", "acc_worst_client"):
        assert out["rounds"][-1][k] is not None
    assert abs(out["summary"]["mean_expected_dropout_rate"] - 0.3) < 1e-6


def test_runs_are_reproducible():
    for method in ("flfdms", "feddrift"):
        c, r, s = _setup()
        a = _run(c, r, s, mechanism="mnar", method=method, rounds=3, eval_every=1)
        c, r, s = _setup()
        b = _run(c, r, s, mechanism="mnar", method=method, rounds=3, eval_every=1)
        strip = lambda rows: [{k: v for k, v in e.items() if k != "elapsed_sec"} for e in rows]  # noqa: E731
        assert strip(a["rounds"]) == strip(b["rounds"])
