"""Stage 0 — synthetic known-truth testbed for DS-DR (research_v2/EVALUATION_PLAN.md §Stage 0).

Every client's would-be update is known, so bias is measured exactly. Design:
  N clients, d-dim model, quadratic local losses: Delta_i = -lr (theta - theta_i*) + noise,
  theta_i* = theta0 + xi_i + D_it * u  (drift cohort: D ramps 0 -> 1 after onset),
  drift statistic S_it = D_it + e_it, e AR(1) with persistence rho_S (the client-computable D_indep),
  participation as in drift-causal-fl/src/mechanisms/dropout.py: p_miss = clip(sigmoid(a_t + beta*std_t(S)),
  .05, .95) with a_t solved so mean p_miss = 0.30; 'threshold' = top 20% by S excluded (positivity fails).
Estimators share the same participation draws (paired). Each runs its OWN trajectory (closed loop);
open-loop one-step errors are measured on the full-participation trajectory.

Usage: python stage0_synthetic.py [--reps 200] [--workers 6] [--out results_stage0.json]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from dsdr_core import (calibrate_intercept, dsdr_coefficients, leverage_cap, link, propensity, shadow_gmm,
                       score_gate, shadow_gmm_rounds, sigmoid, standardise, tilted_kernel)

N, D_MODEL, T, N_DRIFT, ONSET, RAMP = 100, 20, 40, 30, 10, 5
LR, NOISE, HET, SHIFT, S_NOISE = 0.5, 0.05, 0.3, 3.0, 0.3
W, W_MIN, ALPHA_GATE, KAPPA, RHO_MAX = 5, 3, 0.01, 1.0, 10.0
TARGET_MISS = 0.30
METHODS = ["fedavg", "ipw_oracle_ht", "ipw_oracle_hajek", "ipw_ema", "fdms_sub", "lag_mar",
           "dsdr", "dsdr_ipw", "dsdr_nogate", "dsdr_oraclepi"]


def p_miss_for(S: np.ndarray, mech: str, beta: float) -> np.ndarray:
    if mech == "threshold":  # checked first: threshold cells carry beta = 0 (bug fixed 2026-11-22)
        p = np.full(len(S), (TARGET_MISS - 0.2) / 0.8)
        p[np.argsort(-S)[: int(0.2 * len(S))]] = 1.0
        return p
    if mech == "mcar" or beta == 0:
        return np.full(len(S), TARGET_MISS)
    z = (S - S.mean()) / (S.std() + 1e-12)
    lo, hi = -20.0, 20.0
    for _ in range(60):  # solve a_t so that the mean clipped miss rate is 0.30
        a = (lo + hi) / 2
        if np.clip(sigmoid(a + beta * z), 0.05, 0.95).mean() > TARGET_MISS:
            hi = a
        else:
            lo = a
    return np.clip(sigmoid((lo + hi) / 2 + beta * z), 0.05, 0.95)


def drift_level(t: int) -> float:
    return float(np.clip((t - ONSET + 1) / RAMP, 0.0, 1.0)) if t >= ONSET else 0.0


def lag_mar_propensity(Rw, Zw, Zt):
    """A3 ablation: logistic MLE of R on the stale report Z (treats selection as MAR on Z)."""
    from sklearn.linear_model import LogisticRegression
    if Rw.min() == Rw.max():
        return np.full(len(Zt), Rw.mean())
    m = LogisticRegression(C=1e6, max_iter=200).fit(Zw, Rw)
    return np.clip(m.predict_proba(Zt)[:, 1], 0.05, 0.95)


def run_cell(args) -> dict:
    mech, beta, rho, reps, seed = args
    rng = np.random.default_rng(seed)
    w = np.full(N, 1 / N)
    drift = np.arange(N) < N_DRIFT
    acc = {m: {"bias": np.zeros((T, D_MODEL)), "sq": np.zeros(T), "final": [], "cohort_gap": []} for m in METHODS}
    diag = {"corr_gmm": [], "corr_ema": [], "corr_lagmar": [], "gate_open": [], "lambda": [],
            "theta_eta": [], "covers": []}
    for rep in range(reps):
        theta0 = rng.normal(size=D_MODEL)
        xi = rng.normal(scale=HET, size=(N, D_MODEL))
        u = rng.normal(size=D_MODEL)
        u = SHIFT * u / np.linalg.norm(u)
        e = rng.normal(scale=S_NOISE, size=N)
        Z = np.zeros(N)                         # stale reports (warm-up report S = 0)
        ema = np.full(N, 1 - TARGET_MISS)
        buf: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        th = {m: theta0.copy() for m in METHODS}
        th_full = theta0.copy()
        for t in range(T):
            D = np.where(drift, drift_level(t), 0.0)
            e = rho * e + np.sqrt(1 - rho**2) * rng.normal(scale=S_NOISE, size=N)
            S = D + e
            star = theta0 + xi + D[:, None] * u
            pm = p_miss_for(S, mech, beta)
            pi = 1 - pm
            R = rng.random(N) < pi
            parts = np.flatnonzero(R)
            noise = rng.normal(scale=NOISE, size=(N, D_MODEL))

            # nuisance estimates (all F_{t-1}-measurable or participant-only)
            gate, th_hat, lam = False, None, 0.0
            if len(buf) >= W_MIN:
                Rw = np.concatenate([b[0] for b in buf]).astype(int)
                Zw = np.concatenate([b[2] for b in buf])[:, None]
                res = shadow_gmm_rounds([(b[0], b[1][:, None], b[2][:, None]) for b in buf])
                gate = bool(score_gate([(b[0].astype(float), None, b[2][:, None]) for b in buf]) < ALPHA_GATE)
                if len(parts) >= 2:
                    g_t = calibrate_intercept(res.eta, S[parts][:, None], N)
                    th_hat = np.concatenate([[g_t], res.eta])
                if t >= ONSET + RAMP and mech != "mcar":
                    diag["theta_eta"].append(float(res.eta[0]))
                pi_lag = lag_mar_propensity(Rw, Zw, Z[:, None])
            else:
                pi_lag = np.full(N, 1 - TARGET_MISS)
            diag["gate_open"].append(gate)
            pi_gmm = propensity(th_hat, S[:, None]) if th_hat is not None else np.full(N, 1 - TARGET_MISS)
            if t >= ONSET + RAMP and len(parts) > 2 and pi.std() > 0 and mech != "threshold":
                for k, est in (("corr_gmm", pi_gmm), ("corr_ema", ema), ("corr_lagmar", pi_lag)):
                    diag[k].append(float(np.corrcoef(est, pi)[0, 1]) if est.std() > 1e-9 else 0.0)
            Zs = standardise(Z[:, None]) if Z.std() > 1e-12 else np.zeros((N, 1))

            def coefs(method: str) -> np.ndarray:
                c_fa = w[parts] / w[parts].sum()
                if method == "fedavg" or len(parts) < 2:
                    return c_fa
                if method == "ipw_oracle_ht":
                    return w[parts] / pi[parts]
                if method == "ipw_oracle_hajek":
                    c = w[parts] / pi[parts]
                    return c / c.sum()
                if method == "ipw_ema":
                    c = w[parts] / ema[parts]
                    return c / c.sum()
                if method == "lag_mar":
                    c = w[parts] / pi_lag[parts]
                    return c / c.sum()
                if method == "fdms_sub":  # absent client takes its nearest participant (by stale report)
                    c = np.zeros(len(parts))
                    for i in range(N):
                        if R[i]:
                            c[np.searchsorted(parts, i)] += w[i]
                        else:
                            c[np.argmin(np.abs(Z[parts] - Z[i]))] += w[i]
                    return c
                if method == "dsdr_oraclepi":
                    kap = tilted_kernel(Zs, parts, pi[parts], KAPPA)
                    return dsdr_coefficients(w, parts, pi[parts], kap)
                if th_hat is None or (method in ("dsdr", "dsdr_ipw") and not gate):
                    return c_fa
                pp = pi_gmm[parts]
                if method == "dsdr_ipw":
                    return dsdr_coefficients(w, parts, pp, None, outcome=False)
                kap = tilted_kernel(Zs, parts, pp, KAPPA)
                c = dsdr_coefficients(w, parts, pp, kap)
                if method == "dsdr":
                    c, lam_ = leverage_cap(c, c_fa, RHO_MAX)
                    diag["lambda"].append(lam_)
                return c

            # open loop on the full-participation trajectory
            delta_full_path = -LR * (th_full - star) + noise
            target = w @ delta_full_path
            for m in METHODS:
                c = coefs(m)
                err = c @ delta_full_path[parts] - target
                acc[m]["bias"][t] += err
                acc[m]["sq"][t] += err @ err
                # closed loop: each method on its own trajectory
                dm = -LR * (th[m] - star) + noise
                th[m] = th[m] + c @ dm[parts]
            th_full = th_full + target

            buf.append((R.copy(), np.where(R, S, np.nan), Z.copy()))
            buf = buf[-W:]
            Z = np.where(R, S, Z)
            ema = np.clip(0.8 * ema + 0.2 * R, 0.05, 0.95)
        star_T = theta0 + xi + np.where(drift, 1.0, 0.0)[:, None] * u
        full_cohort = np.mean(np.sum((th_full - star_T[drift]) ** 2, axis=1))
        for m in METHODS:
            acc[m]["final"].append(float(np.linalg.norm(th[m] - th_full)))
            acc[m]["cohort_gap"].append(float(np.mean(np.sum((th[m] - star_T[drift]) ** 2, axis=1)) - full_cohort))

    post = slice(ONSET + RAMP, T)
    out = {"mech": mech, "beta": beta, "rho": rho, "reps": reps, "methods": {}}
    for m in METHODS:
        b = acc[m]["bias"] / reps
        out["methods"][m] = {
            "bias_norm_post": float(np.mean(np.linalg.norm(b[post], axis=1))),
            "rmse_post": float(np.sqrt(np.mean(acc[m]["sq"][post] / reps))),
            "final_dist_mean": float(np.mean(acc[m]["final"])),
            "final_dist_se": float(np.std(acc[m]["final"]) / np.sqrt(reps)),
            "cohort_gap_mean": float(np.mean(acc[m]["cohort_gap"])),
            "cohort_gap_se": float(np.std(acc[m]["cohort_gap"]) / np.sqrt(reps)),
        }
    out["diag"] = {k: (float(np.median(v) if k == "theta_eta" else np.mean(v)) if len(v) else None)
                   for k, v in diag.items()}
    out["diag"]["gate_open_rate"] = out["diag"].pop("gate_open")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results_stage0.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="run only this mechanism (e.g. threshold)")
    args = ap.parse_args()
    cells = []
    rhos = [0.9] if args.quick else [0.5, 0.9, 0.99]
    mechs = [("mcar", 0.0)] + [("mnar", b) for b in ([4.0] if args.quick else [1.0, 2.0, 4.0, 8.0])] + \
            ([] if args.quick else [("threshold", 0.0)])
    for rho in rhos:
        for mech, beta in mechs:
            cells.append((mech, beta, rho, args.reps, 1000 + len(cells)))
    if args.only:
        cells = [c for c in cells if c[0] == args.only]
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_cell, cells)
    json.dump({"design": {"N": N, "d": D_MODEL, "T": T, "n_drift": N_DRIFT, "onset": ONSET, "ramp": RAMP,
                          "lr": LR, "noise": NOISE, "het": HET, "shift": SHIFT, "s_noise": S_NOISE,
                          "W": W, "alpha_gate": ALPHA_GATE, "kappa": KAPPA, "rho_max": RHO_MAX,
                          "target_miss": TARGET_MISS, "reps": args.reps},
               "seconds": round(time.time() - t0, 1), "cells": results}, open(args.out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
