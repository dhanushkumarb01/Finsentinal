"""Unit tests for DS-DR's core maths (Lemma 0, Lemma 1 / identification, Proposition 1)."""

import numpy as np
import pytest

from dsdr_core import dsdr_coefficients, link, propensity, shadow_gmm, sigmoid, standardise, tilted_kernel


def _world(rng, n=4000, rho=0.95, gamma=0.8, eta=-2.0):
    """Persistent drift stat: Z = stale copy, S = current; selection depends on S only (MNAR)."""
    z = rng.normal(size=n)
    s = rho * z + np.sqrt(1 - rho**2) * rng.normal(size=n)
    pi = link(gamma + eta * s)
    r = (rng.random(n) < pi).astype(int)
    return r, s[:, None], z[:, None], pi


def test_gmm_recovers_true_propensity_from_shadow():
    rng = np.random.default_rng(0)
    r, s, z, _ = _world(rng, n=20000)
    s_obs = np.where(r[:, None] == 1, s, np.nan)   # current drift seen ONLY for participants
    res = shadow_gmm(r, s_obs, z)
    assert res.converged
    np.testing.assert_allclose(res.theta, [0.8, -2.0], atol=0.15)
    assert res.wald_p < 1e-6


def test_naive_mar_on_stale_report_is_biased_but_shadow_is_not():
    """A logistic MLE of R on Z (lag-MAR) mis-estimates eta; the shadow GMM does not."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    r, s, z, _ = _world(rng, n=20000, rho=0.8)
    lag = LogisticRegression(C=1e6).fit(z, r)
    res = shadow_gmm(r, np.where(r[:, None] == 1, s, np.nan), z)
    assert abs(lag.coef_[0, 0] - (-2.0)) > 0.4          # attenuated
    assert abs(res.theta[1] - (-2.0)) < 0.2


def test_gate_statistic_under_no_selection():
    rng = np.random.default_rng(2)
    rejections = 0
    for _ in range(100):
        r, s, z, _ = _world(rng, n=1500, eta=0.0)
        rejections += shadow_gmm(r, np.where(r[:, None] == 1, s, np.nan), z).wald_p < 0.05
    assert rejections <= 12   # ~5% nominal, with Monte-Carlo slack


def test_lemma0_coefficients_sum_to_one():
    rng = np.random.default_rng(3)
    N = 60
    w = rng.random(N)
    w /= w.sum()
    Z = rng.normal(size=(N, 2))
    parts = np.sort(rng.choice(N, 25, replace=False))
    pi_p = rng.uniform(0.1, 0.9, size=25)
    kap = tilted_kernel(standardise(Z), parts, pi_p)
    np.testing.assert_allclose(kap.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(kap[parts, np.arange(25)] == 0)        # leave-one-out
    c = dsdr_coefficients(w, parts, pi_p, kap)
    assert abs(c.sum() - 1.0) < 1e-12


def test_proposition1_exact_conditional_bias():
    """Fix G_t = (Delta, S, Z, theta_hat); randomise only R. Monte-Carlo E[Delta_hat] - Delta_bar
    must equal sum_i w_i (pi_i/pi_hat_i - 1)(Delta_i - mu_i), mu_i = E[m_i | G_t]."""
    rng = np.random.default_rng(4)
    N, d, reps = 40, 3, 60000
    w = np.full(N, 1 / N)
    S = rng.normal(size=(N, 1))
    Z = 0.9 * S + 0.3 * rng.normal(size=(N, 1))
    Delta = rng.normal(size=(N, d)) + 2.0 * S          # updates covary with drift
    pi = np.clip(sigmoid(0.7 - 1.5 * S[:, 0]), 0.05, 0.95)
    theta_hat = np.array([0.5, -1.0])                  # deliberately misspecified propensity
    pi_hat_all = propensity(theta_hat, S)
    Zs = standardise(Z)
    est, m_sum = np.zeros(d), np.zeros((N, d))
    count = 0
    for _ in range(reps):
        R = rng.random(N) < pi
        parts = np.flatnonzero(R)
        if len(parts) < 2:
            continue
        kap = tilted_kernel(Zs, parts, pi_hat_all[parts])
        c = dsdr_coefficients(w, parts, pi_hat_all[parts], kap)
        est += c @ Delta[parts]
        m_sum += kap @ Delta[parts]
        count += 1
    mean_est, mu = est / count, m_sum / count
    target = w @ Delta
    formula = (w * (pi / pi_hat_all - 1)) @ (Delta - mu)
    emp = mean_est - target
    # relative error of the bias identity (plan: < 5%)
    assert np.linalg.norm(emp - formula) / np.linalg.norm(formula) < 0.05, (emp, formula)


def test_correct_propensity_gives_unbiased_aggregate():
    """Corollary 1(a): with pi_hat = pi, DS-DR is unbiased for ANY outcome model."""
    rng = np.random.default_rng(5)
    N, d, reps = 40, 3, 40000
    w = np.full(N, 1 / N)
    S = rng.normal(size=(N, 1))
    Z = 0.9 * S + 0.3 * rng.normal(size=(N, 1))
    Delta = rng.normal(size=(N, d)) + 2.0 * S
    pi = np.clip(sigmoid(0.7 - 1.5 * S[:, 0]), 0.05, 0.95)
    Zs = standardise(Z)
    est, n = np.zeros(d), 0
    for _ in range(reps):
        parts = np.flatnonzero(rng.random(N) < pi)
        if len(parts) < 2:
            continue
        kap = tilted_kernel(Zs, parts, pi[parts])
        est += dsdr_coefficients(w, parts, pi[parts], kap) @ Delta[parts]
        n += 1
    bias = est / n - w @ Delta
    fedavg_bias = None
    assert np.linalg.norm(bias) < 0.05 * np.linalg.norm(w @ Delta - Delta[pi > np.median(pi)].mean(0)) + 0.05
