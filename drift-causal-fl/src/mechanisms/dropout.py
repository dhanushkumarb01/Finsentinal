"""The three dropout mechanisms (Days 3-4 of the project plan).

Each mechanism returns a boolean "participates this round" array over
clients, all calibrated so the average dropout rate matches a shared
target — this is the core methodological trick of the whole project: we're
isolating *who* is missing, not *how much* is missing.
"""
from __future__ import annotations


import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """The 'squishing function' — turns any real number into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


# Positivity bounds on the per-client MISS probability (spec Part 2). Because
# P(participate) = 1 - P(miss), the same [0.05, 0.95] band also bounds every
# participation propensity — FedIPW (methods/fedipw.py) reuses these bounds so
# the oracle and estimated variants share one, consistent propensity floor.
POSITIVITY_LO = 0.05
POSITIVITY_HI = 0.95


def _clip_positivity(p: np.ndarray, lo: float = POSITIVITY_LO, hi: float = POSITIVITY_HI) -> np.ndarray:
    """Keep every participation probability strictly away from 0 and 1.
    Required for FedIPW's own math (inverse-probability weights explode as
    p -> 0), and keeps 'ignorability violation bias' separable from
    'IPW variance blowup from near-zero propensities' as two distinct,
    non-confounded effects (see project spec, Part 2)."""
    return np.clip(p, lo, hi)


def mcar_dropout(num_clients: int, target_rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """P(miss) = p — a single fixed number, same for every client, every
    round. No relationship to anything about the client.

    Returns (participates, p_miss) — p_miss is trivially the same constant
    for everyone, but returned for interface consistency with the other
    mechanisms (needed by oracle-IPW, which needs the TRUE participation
    probability regardless of which mechanism generated it)."""
    _check_target(target_rate)
    p_miss = np.full(num_clients, target_rate)
    participates = rng.random(num_clients) >= target_rate
    return participates, p_miss


def mar_dropout(
    X: np.ndarray, target_rate: float, gamma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """P(miss) = sigmoid(alpha + gamma * X)

    X is a FIXED, drift-independent covariate per client (assigned once at
    setup — e.g. a device-class label), standardized before use. alpha is
    solved so the *average* miss probability across clients equals
    target_rate, exactly like the MNAR case below — this keeps MAR and
    MNAR fairly comparable.

    Returns (participates, p_miss), same reasoning as mcar_dropout above.
    """
    X_std = (X - X.mean()) / (X.std() + 1e-8)
    alpha = _solve_alpha(X_std, gamma, target_rate)
    p_miss = _miss_prob(alpha, gamma, X_std)
    participates = rng.random(num_clients_from(X)) >= p_miss
    return participates, p_miss


def mnar_drift_dropout(
    D: np.ndarray, target_rate: float, beta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """P(miss) = sigmoid(alpha + beta * D)  — the core mechanism of the project.

    D = each client's CURRENT drift score, standardized within-round before
    use so beta is comparably scaled across rounds/datasets. alpha is
    re-solved every round (numeric bisection) so the *marginal* dropout
    rate matches target_rate regardless of beta — beta only changes WHO is
    likely to be missing, never HOW MANY are missing on average.

    Returns (participates_mask, realized_miss_probability) — the second
    value is kept so we can later compute the realized correlation
    rho = corr(D, missing) as required by the project's coupling-axis
    design (Part 5/6 of the spec).
    """
    D_std = (D - D.mean()) / (D.std() + 1e-8)
    alpha = _solve_alpha(D_std, beta, target_rate)
    p_miss = _miss_prob(alpha, beta, D_std)
    participates = rng.random(len(D)) >= p_miss
    return participates, p_miss


def mnar_threshold_dropout(
    D: np.ndarray, target_rate: float, exclude_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Robustness-check mechanism (required alongside the logistic version,
    per the project spec's Part 2): R=0 deterministically for the
    top-`exclude_frac` of clients by drift this round (the within-round
    percentile threshold tau), else Bernoulli(p_base), with p_base
    CALIBRATED so that the deterministic exclusion and the base rate
    COMBINED hit `target_rate` exactly in expectation:

        k/n + p_base * (n - k)/n = target_rate,   k = round(exclude_frac * n)

    Ties in D (e.g. many stable clients at exactly D=0) are broken uniformly
    at random so no client is chronically excluded because of its index.

    Returns (participates, p_miss). p_miss = 1 for excluded clients (they
    never participate this round, so no IPW weight is ever computed for
    them) and p_base for everyone else — needed by oracle-IPW.
    """
    _check_target(target_rate)
    if not 0.0 <= exclude_frac <= target_rate:
        raise ValueError(f"exclude_frac must be in [0, target_rate={target_rate}], got {exclude_frac}")
    n = len(D)
    k = int(round(exclude_frac * n))
    p_base = (target_rate * n - k) / (n - k) if n > k else 0.0
    if not 0.0 <= p_base < 1.0:
        raise ValueError(f"infeasible threshold calibration: k={k}, n={n}, target={target_rate} -> p_base={p_base}")
    # Sort by D descending, breaking ties randomly (lexsort: last key is primary).
    order = np.lexsort((rng.random(n), -np.asarray(D, dtype=float)))
    p_miss = np.full(n, p_base)
    p_miss[order[:k]] = 1.0
    participates = rng.random(n) >= p_miss
    return participates, p_miss


def _check_target(target_rate: float) -> None:
    if not POSITIVITY_LO <= target_rate <= POSITIVITY_HI:
        raise ValueError(f"target_rate must lie in the positivity band [{POSITIVITY_LO}, {POSITIVITY_HI}], got {target_rate}")


def _miss_prob(alpha: float, coef: float, x: np.ndarray) -> np.ndarray:
    """Clipped logistic miss probability — the exact quantity whose mean is
    calibrated, so clipping can no longer shift the realized marginal rate."""
    return _clip_positivity(sigmoid(alpha + coef * x))


def _solve_alpha(x: np.ndarray, coef: float, target_rate: float, iters: int = 100) -> float:
    """Numeric bisection for alpha so mean(clip(sigmoid(alpha + coef*x))) == target_rate.

    BUG FIX: alpha used to be solved on the UNCLIPPED sigmoid and the
    [0.05, 0.95] positivity clip applied afterwards. Clipping raises the
    smallest probabilities and lowers the largest, so the realized marginal
    rate drifted away from the target as |coef| grew (e.g. 0.316 instead of
    0.30 at beta=8, 0.682 instead of 0.70 at target 0.7). Calibrating the
    clipped function directly keeps every mechanism's average dropout rate
    identical — the whole point of the matched-rate benchmark design. The
    clipped mean is continuous and non-decreasing in alpha, and spans
    [0.05, 0.95], so bisection converges for any target in that band."""
    _check_target(target_rate)
    lo, hi = -50.0, 50.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = _miss_prob(mid, coef, x).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def num_clients_from(x: np.ndarray) -> int:
    return len(x)


def realized_correlation(D: np.ndarray, participates: np.ndarray) -> float:
    """rho = point-biserial correlation between drift and missingness —
    the project's primary reported coupling-strength axis (spec Part 5).
    Sanity check: this should climb toward 1.0 as beta increases."""
    missing = (~participates).astype(float)
    if missing.std() == 0 or D.std() == 0:
        return 0.0
    return float(np.corrcoef(D, missing)[0, 1])
