"""Statistical analysis for the MVP sweep (Step 10).

Design facts that determine the tests (spec Parts 6 and 8):
  * Fully crossed: seed x estimator x condition x method, every cell once.
  * Paired by seed: within a seed, every condition/method sees the identical
    client pool, drift cohort/sources and evaluation sets; the seed is the
    experimental unit ("subject"). n = 5 seeds per cell.

Choices and why:
  * Descriptives: mean, SD over seeds; 95% CI = percentile bootstrap over seeds
    (spec Part 6: 5 seeds is too thin for normal-theory CIs). With n=5 the
    bootstrap CI is itself coarse; it is reported as such, never as exact.
  * Primary model (spec Part 6): linear mixed model with a random intercept per
    seed, fitted by ML, with likelihood-ratio tests for the condition main effect
    and the condition x method interaction. The random intercept absorbs the
    shared per-seed difficulty (different writers per seed).
  * Effect sizes: partial eta^2 from the balanced two-way repeated-measures
    ANOVA (within factors condition, method; subject = seed); Cohen's d_z for
    paired differences.
  * Pairwise comparisons (method vs method within a condition, MNAR vs MCAR):
    seed-paired differences with (i) exact sign-flip permutation test,
    (ii) paired t-test, (iii) Wilcoxon signed-rank. IMPORTANT power fact: with
    n = 5 pairs, the smallest attainable two-sided p of any exact rank/sign test
    is 2/32 = 0.0625 (one-sided 1/32 = 0.031). Single-cell nonparametric tests
    therefore CANNOT reach two-sided alpha = 0.05 by construction; pooled models
    are where the design has power. Holm correction within each family.
  * Practical importance: |difference| >= 1 percentage point of accuracy
    (MIN_PRACTICAL), stated a priori here, reported separately from p-values.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

MIN_PRACTICAL = 0.01
N_BOOT = 10000
BOOT_SEED = 20261119


# ------------------------------------------------------------------ helpers
def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = stat(x[idx], axis=1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def sign_flip_p(d: np.ndarray, alternative: str = "two-sided") -> float:
    """Exact sign-flip permutation test of mean(d) = 0 (all 2^n flips)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    obs = d.mean()
    flips = np.array(list(itertools.product([1, -1], repeat=n)))
    null = (flips * np.abs(d)).mean(axis=1)
    eps = 1e-12
    if alternative == "greater":
        return float(np.mean(null >= obs - eps))
    if alternative == "less":
        return float(np.mean(null <= obs + eps))
    return float(np.mean(np.abs(null) >= abs(obs) - eps))


def holm(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    ok = p.notna()
    out = pd.Series(np.nan, index=p.index)
    if ok.sum() == 0:
        return out
    pv = p[ok].values
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def paired_stats(d: np.ndarray, alternative: str = "two-sided") -> dict:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()) if n else np.nan, "sd_diff": float(d.std(ddof=1)) if n > 1 else np.nan}
    out["ci_lo"], out["ci_hi"] = bootstrap_ci(d)
    out["d_z"] = float(d.mean() / d.std(ddof=1)) if n > 1 and d.std(ddof=1) > 0 else np.nan
    out["p_signflip"] = sign_flip_p(d, alternative)
    if n > 1 and d.std(ddof=1) > 0:
        out["p_ttest"] = float(st.ttest_1samp(d, 0.0, alternative=alternative).pvalue)
    else:
        out["p_ttest"] = np.nan
    try:
        out["p_wilcoxon"] = float(st.wilcoxon(d, alternative=alternative, method="exact").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        out["p_wilcoxon"] = np.nan
    out["n_positive"] = int((d > 0).sum())
    out["practically_important"] = bool(abs(out["mean_diff"]) >= MIN_PRACTICAL) if n else False
    return out


# ------------------------------------------------------------ completeness
def complete_blocks(runs: pd.DataFrame, keys=("estimator",), cells=("condition", "method")) -> tuple[pd.DataFrame, dict]:
    """Keep only seeds that completed every (condition x method) cell within each
    `keys` group; returns (filtered, report). Never imputes."""
    report = {}
    keep = []
    for k, g in runs.groupby(list(keys)):
        expected = g[list(cells)].drop_duplicates().shape[0]
        counts = g.groupby("seed").size()
        full = counts[counts == expected].index.tolist()
        name = "/".join(map(str, k)) if isinstance(k, tuple) else str(k)
        report[name] = {"seeds_complete": sorted(int(s) for s in full),
                          "seeds_incomplete": sorted(int(s) for s in counts.index if s not in full)}
        keep.append(g[g.seed.isin(full)])
    return (pd.concat(keep) if keep else runs.iloc[0:0]), report


# ------------------------------------------------------------- descriptives
def describe(runs: pd.DataFrame, metrics: list[str], by=("estimator", "condition", "method")) -> pd.DataFrame:
    rows = []
    for k, g in runs.groupby(list(by)):
        for m in metrics:
            x = g[m].dropna().values
            lo, hi = bootstrap_ci(x)
            rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), metric=m, n=len(x),
                             mean=float(np.mean(x)) if len(x) else np.nan,
                             sd=float(np.std(x, ddof=1)) if len(x) > 1 else np.nan, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------- pairwise tests
def method_vs_method(runs: pd.DataFrame, metric: str, a: str, b: str, by=("estimator", "condition")) -> pd.DataFrame:
    """b - a per seed within each (by) cell. Positive = b better."""
    rows = []
    for k, g in runs.groupby(list(by)):
        pa = g[g.method == a].set_index("seed")[metric]
        pb = g[g.method == b].set_index("seed")[metric]
        common = pa.index.intersection(pb.index)
        if len(common) == 0:
            continue
        d = (pb.loc[common] - pa.loc[common]).values
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), metric=metric, a=a, b=b, **paired_stats(d)))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p_signflip"])
    return df


def condition_vs_condition(runs: pd.DataFrame, metric: str, ref: str, other: str, alternative: str = "two-sided",
                           by=("estimator", "method")) -> pd.DataFrame:
    """other - ref per seed, within each (by) cell."""
    rows = []
    for k, g in runs.groupby(list(by)):
        pr = g[g.condition == ref].set_index("seed")[metric]
        po = g[g.condition == other].set_index("seed")[metric]
        common = pr.index.intersection(po.index)
        if len(common) == 0:
            continue
        d = (po.loc[common] - pr.loc[common]).values
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), metric=metric, ref=ref, other=other,
                         alternative=alternative, **paired_stats(d, alternative)))
    return pd.DataFrame(rows)


# --------------------------------------------------------- pooled models
def rm_anova(runs: pd.DataFrame, metric: str, within=("condition", "method")) -> pd.DataFrame:
    """Two-way repeated-measures ANOVA, subject = seed, with partial eta^2."""
    from statsmodels.stats.anova import AnovaRM
    d = runs[["seed", *within, metric]].dropna()
    res = AnovaRM(d, depvar=metric, subject="seed", within=list(within)).fit().anova_table.reset_index()
    res = res.rename(columns={"index": "effect", "F Value": "F", "Num DF": "df1", "Den DF": "df2", "Pr > F": "p"})
    res["partial_eta2"] = res.F * res.df1 / (res.F * res.df1 + res.df2)
    res["metric"] = metric
    return res


def _fe(m):
    """Fixed-effect parameters of a MixedLM or OLS result (seed dummies excluded)."""
    fe = m.fe_params if hasattr(m, "fe_params") else m.params
    return fe[[t for t in fe.index if not t.startswith("C(seed)")]]


def mixed_model(runs: pd.DataFrame, metric: str, formula_rhs: str, reduced: dict[str, str]) -> dict:
    """ML-fitted LMM `metric ~ formula_rhs + (1|seed)`, plus likelihood-ratio
    tests against each reduced model (name -> rhs). Returns coefficients with
    95% Wald CIs and LRT results."""
    import statsmodels.formula.api as smf
    d = runs.dropna(subset=[metric]).copy()
    out = {"metric": metric, "formula": f"{metric} ~ {formula_rhs} + (1|seed)", "n_obs": len(d),
           "n_seeds": int(d.seed.nunique())}

    def fit(rhs):
        # With few seeds the seed variance can sit on its boundary (0); some
        # optimizers then hit a singular Hessian and raise. Try each optimizer in
        # turn and record which one produced the fit. If no LMM fit succeeds, fall
        # back to OLS with seed fixed effects (the same blocked design without a
        # variance component) and say so in the output.
        errors = []
        for opt in ("lbfgs", "powell", "nm", "bfgs"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    m = smf.mixedlm(f"{metric} ~ {rhs}", d, groups=d["seed"]).fit(reml=False, method=[opt])
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{opt}: {type(e).__name__}: {e}")
                    continue
            return m, [str(x.message)[:160] for x in w], f"mixedlm/{opt}", errors
        m = smf.ols(f"{metric} ~ {rhs} + C(seed)", d).fit()
        return m, [], "ols_seed_fixed_effects (all LMM optimizers failed)", errors

    full, warns, how, errs = fit(formula_rhs)
    out["fit_method"] = how
    out["optimizer_errors"] = errs
    is_lmm = how.startswith("mixedlm")
    out["converged"] = bool(full.converged) if is_lmm else True
    out["warnings"] = sorted(set(warns))
    out["llf"] = float(full.llf)
    out["seed_variance"] = float(full.cov_re.iloc[0, 0]) if is_lmm else None
    out["seed_variance_at_boundary"] = bool(is_lmm and out["seed_variance"] < 1e-6)
    out["resid_variance"] = float(full.scale)
    ci = full.conf_int()
    out["coefficients"] = [
        {"term": t, "estimate": float(_fe(full)[t]), "ci_lo": float(ci.loc[t, 0]), "ci_hi": float(ci.loc[t, 1]),
         "p": float(full.pvalues[t])} for t in _fe(full).index]
    lrts = []
    for name, rhs in reduced.items():
        red, _, red_how, _ = fit(rhs)
        lr = 2 * (full.llf - red.llf)
        df = len(_fe(full)) - len(_fe(red))
        lrts.append({"test": name, "reduced": f"{metric} ~ {rhs}", "reduced_fit": red_how, "LR": float(lr), "df": int(df),
                     # an LRT is only valid between models of the same family
                     "p": float(st.chi2.sf(lr, df)) if df > 0 and (red_how.startswith("mixedlm") == is_lmm) else np.nan})
    out["lrt"] = lrts
    return out


def dose_response(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Within MNAR runs: per (estimator, method) slope of metric on log2(beta),
    estimated per seed then summarised across seeds (paired design)."""
    rows = []
    mn = runs[runs.mechanism == "mnar"]
    for (est, meth), g in mn.groupby(["estimator", "method"]):
        slopes = []
        for s, gs in g.groupby("seed"):
            if gs.beta.nunique() >= 2:
                slopes.append(np.polyfit(np.log2(gs.beta.values), gs[metric].values, 1)[0])
        rows.append(dict(estimator=est, method=meth, metric=metric, **paired_stats(np.array(slopes))))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.rename(columns={"mean_diff": "mean_slope_per_doubling_beta", "sd_diff": "sd_slope"})
        df["p_holm"] = holm(df["p_signflip"])
    return df


def did_ipw_boundary(runs: pd.DataFrame, metric: str, ipw: str) -> pd.DataFrame:
    """FedIPW boundary (spec Part 11): difference-in-differences per seed,
    [IPW - FedAvg]_{MNAR beta} - [IPW - FedAvg]_{MCAR}. Negative = IPW's
    advantage shrinks (or its cost grows) under MNAR relative to MCAR."""
    rows = []
    for est, g in runs.groupby("estimator"):
        piv = g[g.method.isin(["fedavg", ipw])].pivot_table(index=["seed", "condition"], columns="method", values=metric)
        if ipw not in piv or "fedavg" not in piv:
            continue
        gain = (piv[ipw] - piv["fedavg"]).unstack("condition")
        for ref in ("mcar", "mar"):
            if ref not in gain:
                continue
            for c in [c for c in gain.columns if c.startswith("mnar")]:
                d = (gain[c] - gain[ref]).dropna().values
                rows.append(dict(estimator=est, ipw=ipw, metric=metric, ref=ref, condition=c,
                                 gain_ref=float(gain[ref].mean()), gain_cond=float(gain[c].mean()), **paired_stats(d)))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p_signflip"])
    return df


def reproducibility_check(runs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """MCAR/MAR never read D, so their model-relative and model-independent runs
    must be bit-identical replicates. Returns the max |difference| per cell."""
    rows = []
    sub = runs[runs.mechanism.isin(["mcar", "mar"])]
    for (s, c, m), g in sub.groupby(["seed", "condition", "method"]):
        if g.estimator.nunique() < 2:
            continue
        a = g[g.estimator == "model_relative"].iloc[0]
        b = g[g.estimator == "model_independent"].iloc[0]
        rows.append({"seed": s, "condition": c, "method": m,
                     "max_abs_diff": float(max(abs(a[x] - b[x]) for x in metrics if pd.notna(a[x])))})
    return pd.DataFrame(rows)
