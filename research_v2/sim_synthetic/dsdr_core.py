"""DS-DR (Drift-Shadow Doubly Robust aggregation) — pure-numpy core.

Implements research_v2/CANDIDATE_METHODS.md §1.3 exactly:
  (a) temporal shadow-variable GMM for the drift-dependent propensity pi(S) = sigmoid(gamma + eta^T S),
      moments  mean_{s in window, i} ( R_is / pi(S_is) - 1 ) h(Z_is),  h(z) = (1, z)  (just identified);
      the summand needs S only where R = 1 (else it is -h(Z)), so it is computable from observed data;
  (b) selection gate: sandwich Wald test of eta = 0 (closed -> exact FedAvg weights);
  (c) tilted, leave-one-out kernel outcome model  m_i = sum_{k in S \\ i} kappa_ik Delta_k,
      kappa_ik ∝ K(Z_i, Z_k) * omega_k,  omega_k = (1 - pi_k)/pi_k;
  (d)/(e) DS-DR aggregate in coefficient form  Delta_hat = sum_{k in S} c_k Delta_k,
      c_k = sum_i w_i kappa_ik + w_k/pi_k - sum_{j in S} (w_j/pi_j) kappa_jk   (Lemma 0: sum c = 1);
  (f) leverage cap.
No torch, no model memory: everything is scalar coefficients. Used by the synthetic testbed (Stage 0)
and, later, by drift-causal-fl/src/methods/dsdr.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PI_CLIP = (0.05, 0.95)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def link(a):
    """Bounded logistic pi = lo + (hi - lo) * sigmoid(a): the clipped-logistic family of assumption A3.
    It can represent the [0.05, 0.95] plateau exactly, so the GMM does not diverge when the true
    mechanism is near-deterministic (strong beta)."""
    lo, hi = PI_CLIP
    return lo + (hi - lo) * sigmoid(a)


def dlink(a):
    lo, hi = PI_CLIP
    s_ = sigmoid(a)
    return (hi - lo) * s_ * (1 - s_)


@dataclass
class GMMResult:
    theta: np.ndarray        # (gamma, eta_1..eta_p)
    cov: np.ndarray          # sandwich covariance of theta
    wald_p: float            # p-value of H0: eta = 0
    cond: float              # condition number of the Jacobian
    converged: bool
    n: int


def shadow_gmm(R: np.ndarray, S: np.ndarray, Z: np.ndarray, *, iters: int = 50, tol: float = 1e-10) -> GMMResult:
    """Temporal shadow-variable GMM.

    R: (n,) 0/1 participation over the pooled window (client-rounds);
    S: (n, p) current drift stats — only rows with R = 1 are read (others may be NaN);
    Z: (n, p) stale (shadow) reports — fully observed.
    """
    from scipy.stats import chi2

    n, p = Z.shape
    H = np.hstack([np.ones((n, 1)), Z])                      # instruments h(Z)
    X = np.hstack([np.ones((n, 1)), np.where(R[:, None] == 1, S, 0.0)])  # regressors (only used where R=1)
    rbar = float(np.clip(R.mean(), 1e-3, 1 - 1e-3))
    theta = np.zeros(p + 1)
    theta[0] = np.log(rbar / (1 - rbar))
    converged = False
    J = np.eye(p + 1)
    for _ in range(iters):
        a = X @ theta
        pi = link(a)
        inv = np.where(R == 1, 1.0 / pi, 0.0)
        G = ((inv - 1.0)[:, None] * H).mean(axis=0)         # moment vector
        # d(1/pi)/dtheta = -pi'(a)/pi^2 * x   (only where R = 1)
        dinv = np.where(R == 1, -dlink(a) / pi**2, 0.0)
        J = (H * dinv[:, None]).T @ X / n                    # (p+1) x (p+1): dG/dtheta
        try:
            step = np.linalg.solve(J, G)
        except np.linalg.LinAlgError:
            break
        # damped Newton for robustness
        lam = 1.0
        while lam > 1e-4:
            cand = theta - lam * step
            pc = link(X @ cand)
            Gc = ((np.where(R == 1, 1.0 / pc, 0.0) - 1.0)[:, None] * H).mean(axis=0)
            if np.linalg.norm(Gc) <= np.linalg.norm(G) or lam <= 1e-3:
                break
            lam *= 0.5
        theta = cand
        if np.linalg.norm(lam * step) < tol:
            converged = True
            break
    pi = link(X @ theta)
    g = ((np.where(R == 1, 1.0 / pi, 0.0) - 1.0)[:, None] * H)
    omega = g.T @ g / n
    try:
        Jinv = np.linalg.inv(J)
        cov = Jinv @ omega @ Jinv.T / n
        cond = float(np.linalg.cond(J))
        eta, V = theta[1:], cov[1:, 1:]
        wald = float(eta @ np.linalg.solve(V, eta))
        pval = float(chi2.sf(wald, df=p))
    except np.linalg.LinAlgError:
        cov, cond, pval = np.full((p + 1, p + 1), np.nan), np.inf, 1.0
    return GMMResult(theta, cov, pval, cond, converged, n)


def propensity(theta: np.ndarray, S: np.ndarray) -> np.ndarray:
    return link(theta[0] + S @ theta[1:])


def standardise(Z: np.ndarray) -> np.ndarray:
    sd = Z.std(axis=0)
    return (Z - Z.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)


def tilted_kernel(Zs: np.ndarray, parts: np.ndarray, pi_parts: np.ndarray, kappa_bw: float = 1.0) -> np.ndarray:
    """kappa (N x |S|): row i = tilted, leave-one-out kernel weights over participants."""
    zp = Zs[parts]
    d2 = ((Zs[:, None, :] - zp[None, :, :]) ** 2).sum(-1)    # N x |S|
    # Bandwidth from ALL clients' stale reports (fully observed, F_{t-1}-measurable). Using only the
    # participants would make m_i depend on R_i through h and break the leave-one-out independence
    # that Proposition 1 needs (found by test_proposition1_exact_conditional_bias: 5% bias otherwise).
    pair = np.sqrt(((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1))
    med = np.median(pair[np.triu_indices(len(Zs), 1)]) if len(Zs) > 2 else 1.0
    h = kappa_bw * (med if med > 1e-12 else 1.0)
    K = np.exp(-d2 / (2 * h * h)) if np.isfinite(kappa_bw) else np.ones_like(d2)
    K[parts, np.arange(len(parts))] = 0.0                     # leave-one-out
    W = K * ((1 - pi_parts) / pi_parts)[None, :]
    rs = W.sum(axis=1, keepdims=True)
    # if a row has no mass (all tilts zero), fall back to uniform over the other participants
    fallback = np.ones_like(W)
    fallback[parts, np.arange(len(parts))] = 0.0
    fallback /= fallback.sum(axis=1, keepdims=True)
    return np.where(rs > 1e-300, W / np.where(rs > 1e-300, rs, 1.0), fallback)


def dsdr_coefficients(w: np.ndarray, parts: np.ndarray, pi_parts: np.ndarray, kappa: np.ndarray,
                      outcome: bool = True) -> np.ndarray:
    """c over participants. outcome=False gives the m≡0 (Horvitz–Thompson IPW) ablation."""
    wp = w[parts]
    if not outcome:
        return wp / pi_parts
    return w @ kappa + wp / pi_parts - (wp / pi_parts) @ kappa[parts]


def leverage_cap(c: np.ndarray, c_fa: np.ndarray, rho_max: float) -> tuple[np.ndarray, float]:
    """Smallest lambda in [0,1] with max|c_lambda| <= rho_max/|S|; c_lambda = (1-lambda) c + lambda c_FA."""
    if not np.isfinite(rho_max):
        return c, 0.0
    cap = rho_max / len(c)
    if np.max(np.abs(c)) <= cap:
        return c, 0.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.max(np.abs((1 - mid) * c + mid * c_fa)) <= cap:
            hi = mid
        else:
            lo = mid
    return (1 - hi) * c + hi * c_fa, hi


# =============================================================================== round-intercept GMM
@dataclass
class RoundGMMResult:
    eta: np.ndarray          # common drift slope(s), in raw S units
    intercepts: np.ndarray   # one gamma per window round
    cov_eta: np.ndarray      # sandwich covariance of eta
    wald_p: float
    cond: float
    converged: bool
    scale: float


ETA_MAX = 25.0  # bound on |eta| in standardised-Z units (numerical safeguard; logged when active)


def shadow_gmm_rounds(rounds: list[tuple[np.ndarray, np.ndarray, np.ndarray]], *, iters: int = 100,
                      tol: float = 1e-9) -> RoundGMMResult:
    """Temporal shadow GMM with ROUND-SPECIFIC intercepts and a common slope.

    The simulator's mechanism (drift-causal-fl/src/mechanisms/dropout.py) re-solves its intercept every
    round so the mean miss rate stays 0.30; pooling rounds under one intercept violates assumption A3
    (found in Stage 0: GMM failed to converge in ~1/3 of windows). Model: pi_s(z) = link(gamma_s + eta^T s).
    Moments (just identified, W + p equations / unknowns):
      for each window round s:  mean_i ( R_is / pi_s(S_is) - 1 )             = 0
      pooled:                   mean_{s,i} ( R_is / pi_s(S_is) - 1 ) Z_is    = 0
    rounds: list of (R (N,), S (N,p) NaN where R=0, Z (N,p)).
    """
    from scipy.stats import chi2

    Wn = len(rounds)
    p = rounds[0][2].shape[1]
    allZ = np.vstack([r[2] for r in rounds])
    scale = float(allZ.std()) or 1.0
    R = np.concatenate([r[0] for r in rounds]).astype(float)
    Sx = np.vstack([np.where(r[0][:, None] == 1, r[1], 0.0) for r in rounds]) / scale
    Zx = allZ / scale
    rid = np.concatenate([np.full(len(r[0]), k) for k, r in enumerate(rounds)])
    n = len(R)
    E = np.zeros((n, Wn))
    E[np.arange(n), rid] = 1.0                                  # round indicators
    H = np.hstack([E, Zx])                                     # instruments
    X = np.hstack([E, Sx])                                     # regressors
    th = np.zeros(Wn + p)
    for k, r in enumerate(rounds):
        m = float(np.clip(r[0].mean(), 0.06, 0.94))
        q = (m - PI_CLIP[0]) / (PI_CLIP[1] - PI_CLIP[0])
        th[k] = np.log(q / (1 - q))

    def moments(theta):
        a = X @ theta
        pi = link(a)
        inv = np.where(R == 1, 1.0 / pi, 0.0)
        return a, pi, ((inv - 1.0)[:, None] * H)

    converged = False
    for _ in range(iters):
        a, pi, g = moments(th)
        G = g.mean(axis=0)
        dinv = np.where(R == 1, -dlink(a) / pi**2, 0.0)
        J = (H * dinv[:, None]).T @ X / n
        try:
            step = np.linalg.solve(J, G)
        except np.linalg.LinAlgError:
            break
        lam = 1.0
        while True:
            cand = th - lam * step
            cand[Wn:] = np.clip(cand[Wn:], -ETA_MAX, ETA_MAX)
            if np.linalg.norm(moments(cand)[2].mean(axis=0)) <= np.linalg.norm(G) or lam < 1e-3:
                break
            lam *= 0.5
        delta = np.linalg.norm(cand - th)
        th = cand
        if delta < tol:
            converged = True
            break
    a, pi, g = moments(th)
    G = g.mean(axis=0)
    if np.linalg.norm(G) < 1e-6:
        converged = True
    dinv = np.where(R == 1, -dlink(a) / pi**2, 0.0)
    J = (H * dinv[:, None]).T @ X / n
    try:
        Jinv = np.linalg.inv(J)
        cov = Jinv @ (g.T @ g / n) @ Jinv.T / n
        cov_eta = cov[Wn:, Wn:]
        eta_s = th[Wn:]
        wald = float(eta_s @ np.linalg.solve(cov_eta, eta_s))
        pval = float(chi2.sf(wald, df=p))
        cond = float(np.linalg.cond(J))
    except np.linalg.LinAlgError:
        cov_eta, pval, cond = np.full((p, p), np.nan), 1.0, np.inf
    return RoundGMMResult(th[Wn:] / scale, th[:Wn], cov_eta / scale**2, pval, cond, converged, scale)


def calibrate_intercept(eta: np.ndarray, S_parts: np.ndarray, n_total: int) -> float:
    """Current-round intercept from the calibration equation  sum_{j in S_t} 1/pi(S_j) = N
    (the round-t intercept moment). Hajek-type: it uses this round's participation, so pi_hat is not
    exactly F_{t-1}-measurable; its effect on Proposition 1 is O(1/N) (documented deviation)."""
    lin = S_parts @ eta
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.sum(1.0 / link(mid + lin)) > n_total:   # too few implied clients -> raise pi
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def score_gate(rounds: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """Score (LM) test of H0: eta = 0 — no selection on drift — from the shadow moments.

    Under H0 each round's propensity is its participation rate rbar_s (the intercept moment fitted
    exactly), and the pooled shadow moment g = (R/rbar_s - 1)(Z - Zbar_s) has mean zero. Statistic
    LM = (sum g)^T (sum g g^T)^{-1} (sum g) ~ chi2_p. Unlike the Wald test it is evaluated at eta = 0,
    so it does not lose power when the true |eta| is large (Hauck-Donner effect, found in Stage 0:
    Wald gate opened in only ~16% of strong-MNAR rounds).
    """
    from scipy.stats import chi2

    gs = []
    for R, _S, Z in rounds:
        rbar = float(np.clip(R.mean(), 1e-3, 1 - 1e-3))
        gs.append(((R / rbar) - 1.0)[:, None] * (Z - Z.mean(axis=0)))
    g = np.vstack(gs)
    tot = g.sum(axis=0)
    V = g.T @ g
    try:
        lm = float(tot @ np.linalg.solve(V, tot))
    except np.linalg.LinAlgError:
        return 1.0
    return float(chi2.sf(lm, df=g.shape[1]))
