"""Unit tests for the Step 4/5 fixes (calibration, threshold mechanism, FedIPW,
drift-aware evaluation, class handling, runner invariants).

Synthetic data only — no FEMNIST download needed. Run from drift-causal-fl/:
    python -m pytest tests -q
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import ClientData, validate_labels
from src.drift.injection import apply_drift, apply_drift_to_test, make_drift_schedule
from src.mechanisms.dropout import (
    POSITIVITY_HI,
    POSITIVITY_LO,
    mar_dropout,
    mcar_dropout,
    mnar_drift_dropout,
    mnar_threshold_dropout,
)
from src.methods.fedipw import EstimatedPropensityTracker, ipw_aggregate
from src.simulation.fedavg_baseline import fedavg_aggregate
from src.simulation.mechanism_runner import run_mechanism_experiment


def _tied_drift(n=40, n_drift=16, seed=0):
    """Shaped like real model-independent D: many stable clients at exactly 0."""
    rng = np.random.default_rng(seed)
    D = np.zeros(n)
    D[:n_drift] = rng.uniform(0.002, 0.014, n_drift)
    return D


# ---------------------------------------------------------------- calibration
@pytest.mark.parametrize("target", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
def test_mnar_logistic_expected_rate_matches_target_under_clipping(target, beta):
    _, p_miss = mnar_drift_dropout(_tied_drift(), target, beta, np.random.default_rng(0))
    assert p_miss.min() >= POSITIVITY_LO - 1e-12 and p_miss.max() <= POSITIVITY_HI + 1e-12
    assert abs(p_miss.mean() - target) < 1e-6


@pytest.mark.parametrize("target", [0.1, 0.3, 0.7])
@pytest.mark.parametrize("gamma", [0.5, 2.0, 8.0])
def test_mar_expected_rate_matches_target(target, gamma):
    X = np.random.default_rng(1).normal(size=40)
    _, p_miss = mar_dropout(X, target, gamma, np.random.default_rng(0))
    assert abs(p_miss.mean() - target) < 1e-6


def test_mnar_realized_rate_is_unbiased_monte_carlo():
    D = _tied_drift()
    rng = np.random.default_rng(42)
    rates = [1 - mnar_drift_dropout(D, 0.3, 8.0, rng)[0].mean() for _ in range(20000)]
    # SE of the mean ~ sqrt(.3*.7/40/20000) ~ 5e-4
    assert abs(np.mean(rates) - 0.3) < 2.5e-3


def test_target_outside_positivity_band_rejected():
    with pytest.raises(ValueError):
        mnar_drift_dropout(_tied_drift(), 0.99, 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        mcar_dropout(10, 0.0, np.random.default_rng(0))


# ---------------------------------------------------------- threshold dropout
@pytest.mark.parametrize("target,excl", [(0.3, 0.0), (0.3, 0.2), (0.3, 0.3), (0.5, 0.25), (0.7, 0.5)])
def test_threshold_expected_rate_exact(target, excl):
    D = _tied_drift()
    _, p_miss = mnar_threshold_dropout(D, target, excl, np.random.default_rng(0))
    assert abs(p_miss.mean() - target) < 1e-12


def test_threshold_excludes_top_drift_and_realized_rate_unbiased():
    D = _tied_drift()
    rng = np.random.default_rng(3)
    rates = []
    for _ in range(5000):
        part, p_miss = mnar_threshold_dropout(D, 0.3, 0.2, rng)
        excluded = np.where(p_miss == 1.0)[0]
        assert len(excluded) == 8 and not part[excluded].any()
        assert set(excluded) <= set(np.argsort(-D)[:8])  # 16 distinct positive D -> top-8 is unique
        rates.append(1 - part.mean())
    assert abs(np.mean(rates) - 0.3) < 3e-3


def test_threshold_breaks_ties_randomly():
    """BUG FIX: ties used to be broken by client index, so the same stable
    clients were deterministically dropped every round."""
    D = _tied_drift()  # 16 positive, 24 tied at 0; exclusion of 20 reaches into the ties
    rng = np.random.default_rng(7)
    hits = np.zeros(len(D))
    for _ in range(4000):
        _, p_miss = mnar_threshold_dropout(D, 0.5, 0.5, rng)
        hits += p_miss == 1.0
    tied = hits[16:] / 4000
    assert np.all(hits[:16] == 4000)
    assert np.allclose(tied, 4 / 24, atol=0.03), tied  # each tied client excluded ~1/6 of rounds


def test_threshold_exclusion_above_target_rejected():
    with pytest.raises(ValueError):
        mnar_threshold_dropout(_tied_drift(), 0.3, 0.4, np.random.default_rng(0))


# --------------------------------------------------------------------- FedIPW
def _sd(v):
    return {"w": torch.tensor([float(v)])}


def test_ipw_aggregate_toy_known_output():
    # weights n/p: 10/0.5=20, 30/0.75=40 -> (20*1 + 40*4)/60 = 3.0
    out = ipw_aggregate([_sd(1), _sd(4)], [10, 30], [0.5, 0.75])
    assert torch.allclose(out["w"], torch.tensor([3.0]))


def test_ipw_equal_propensities_equals_fedavg():
    sds, n = [_sd(1), _sd(2), _sd(7)], [5, 10, 20]
    assert torch.allclose(ipw_aggregate(sds, n, [0.7] * 3)["w"], fedavg_aggregate(sds, n)["w"])


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_ipw_rejects_invalid_propensity(bad):
    with pytest.raises(ValueError):
        ipw_aggregate([_sd(1), _sd(2)], [1, 1], [0.5, bad])


def test_estimated_propensity_uses_shared_floor_and_no_lookahead():
    tr = EstimatedPropensityTracker(decay=0.8, prior=0.5)
    assert tr.observe_and_get(0, True) == 0.5  # this round's outcome not used for this round
    for _ in range(60):
        p = tr.observe_and_get(1, False)
    assert p == pytest.approx(0.05)  # was floored at 1e-3 before the fix
    for _ in range(60):
        p = tr.observe_and_get(2, True)
    assert p == pytest.approx(0.95)


def test_estimated_propensity_never_sees_drift():
    """Spec Part 1 invariant: the estimated propensity model's only input is
    participation history."""
    params = list(inspect.signature(EstimatedPropensityTracker.observe_and_get).parameters)
    assert params == ["self", "client_id", "participated_this_round"]


# ------------------------------------------------------ data / drifted test set
def _client(cid, n, num_classes, seed, n_test=None):
    g = torch.Generator().manual_seed(seed)
    n_test = n_test or max(2, n // 4)
    return ClientData(
        cid,
        torch.rand(n, 1, 28, 28, generator=g),
        torch.randint(0, num_classes, (n,), generator=g),
        torch.rand(n_test, 1, 28, 28, generator=g),
        torch.randint(0, num_classes, (n_test,), generator=g),
    )


def test_validate_labels_catches_out_of_range():
    c = _client(0, 8, 5, 0)
    validate_labels([c], 5)
    c.test_y[0] = 5
    with pytest.raises(ValueError):
        validate_labels([c], 5)


def _setup(n_clients=6, num_classes=5):
    clients = [_client(i, 12, num_classes, i) for i in range(n_clients)]
    reserve = [_client(-(i + 1), 12, num_classes, 100 + i, n_test=6) for i in range(4)]
    sched = make_drift_schedule([c.client_id for c in clients], 0.5, 2, np.random.default_rng(0),
                                ramp_rounds=2, reserve_pool=reserve)
    return clients, reserve, sched


def test_drifted_test_set_uses_only_heldout_source_samples_and_is_deterministic():
    clients, reserve, sched = _setup()
    rby = {r.client_id: r for r in reserve}
    cid = next(iter(sched.drift_start_round))
    c = clients[cid]
    src = rby[sched.drift_source_id[cid]]
    x1, y1 = apply_drift_to_test(c, 1.0, sched, 5, rby, eval_seed=0)
    x2, _ = apply_drift_to_test(c, 1.0, sched, 5, rby, eval_seed=0)
    assert torch.equal(x1, x2)
    for img in x1:  # every swapped-in image comes from the source's TEST split, never its train split
        assert any(torch.equal(img, t) for t in src.test_x)
        assert not any(torch.equal(img, t) for t in src.train_x)
    x0, y0 = apply_drift_to_test(c, 0.0, sched, 5, rby)
    assert torch.equal(x0, c.test_x) and torch.equal(y0, c.test_y)
    # training-side drift draws only from the source's train split
    xt, _ = apply_drift(c.train_x, c.train_y, 1.0, sched, cid, np.random.default_rng(0), rby)
    for img in xt:
        assert any(torch.equal(img, t) for t in src.train_x)


# --------------------------------------------------------------------- runner
def _run(clients, reserve, sched, **kw):
    base = dict(clients=clients, num_classes=5, schedule=sched, target_rate=0.3, rounds=3,
                reserve_pool=reserve, local_epochs=1, lr=0.01, batch_size=6, seed=0, verbose=False)
    base.update(kw)
    return run_mechanism_experiment(**base)


def test_evaluation_does_not_change_training():
    clients, reserve, sched = _setup()
    a = _run(clients, reserve, sched, mechanism="mnar", method="fedavg", eval_every=1)
    b = _run(clients, reserve, sched, mechanism="mnar", method="fedavg", eval_every=100)
    assert a["rounds"][-1]["acc_global"] == b["rounds"][-1]["acc_global"]
    assert a["summary"]["mean_realized_dropout_rate"] == b["summary"]["mean_realized_dropout_rate"]


def test_oracle_ipw_equals_fedavg_under_mcar():
    clients, reserve, sched = _setup()
    a = _run(clients, reserve, sched, mechanism="mcar", method="fedavg")
    b = _run(clients, reserve, sched, mechanism="mcar", method="fedipw_oracle")
    assert a["rounds"][-1]["acc_global"] == b["rounds"][-1]["acc_global"]


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar", "mnar_threshold"])
@pytest.mark.parametrize("estimator", ["model_relative", "model_independent"])
def test_runner_all_mechanisms_and_estimators(mechanism, estimator):
    clients, reserve, sched = _setup()
    out = _run(clients, reserve, sched, mechanism=mechanism, method="fedipw_oracle",
               drift_estimator=estimator, threshold_exclude_frac=0.15)
    s = out["summary"]
    assert abs(s["mean_expected_dropout_rate"] - 0.3) < 1e-6
    for k in ("acc_global", "acc_drift_cohort", "acc_stable_cohort", "acc_clean", "acc_worst_client"):
        assert out["rounds"][-1][k] is not None
    assert len(out["per_client_final"]) == len(clients)


def test_runner_rejects_out_of_range_reserve_labels():
    clients, reserve, sched = _setup()
    reserve[0].train_y[0] = 5
    with pytest.raises(ValueError):
        _run(clients, reserve, sched, mechanism="mcar")
