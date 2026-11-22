"""Publication figures for the MVP sweep (Step 10D). Static matplotlib.

Encoding rules (dataviz method): methods use categorical slots 1-5 in a FIXED
order (colour follows the method everywhere) plus a distinct marker per method
as secondary encoding; one y-axis per panel; legends on every multi-series
panel; recessive grid; text in neutral ink. Error bars are 95% percentile-
bootstrap CIs over seeds (n shown in captions/titles).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.collect import (  # noqa: E402
    CONDITION_LABELS,
    CONDITION_ORDER,
    ESTIMATOR_LABELS,
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_LABELS,
)
from analysis.stats import bootstrap_ci  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots 1-5
MARKERS = ["o", "s", "D", "^", "v"]
METHOD_COLOR = dict(zip(METHOD_ORDER, SERIES))
METHOD_MARKER = dict(zip(METHOD_ORDER, MARKERS))
INK, INK2, GRID = "#1f1f1e", "#5c5b55", "#e4e3de"
COHORT_COLOR = {"Drift cohort": "#2a78d6", "Stable cohort": "#eb6834"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def _ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x.mean() if len(x) else np.nan
    lo, hi = bootstrap_ci(x)
    return m, lo, hi


def _save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


def _conds(runs):
    return [c for c in CONDITION_ORDER if c in set(runs.condition)]


def _methods(runs):
    return [m for m in METHOD_ORDER if m in set(runs.method)]


def fig_method_by_condition(runs: pd.DataFrame, metric: str, out: Path, name: str) -> list[str]:
    """Points + 95% CI per method across conditions, one panel per estimator."""
    ests = [e for e in ESTIMATOR_LABELS if e in set(runs.estimator)]
    conds, meths = _conds(runs), _methods(runs)
    fig, axes = plt.subplots(1, len(ests), figsize=(5.2 * len(ests), 3.6), sharey=True, squeeze=False)
    width = 0.75
    for ax, est in zip(axes[0], ests):
        g = runs[runs.estimator == est]
        for k, m in enumerate(meths):
            xs, ys, lo, hi = [], [], [], []
            for i, c in enumerate(conds):
                v = g[(g.condition == c) & (g.method == m)][metric]
                if len(v) == 0:
                    continue
                mu, l_, h_ = _ci(v)
                xs.append(i - width / 2 + width * (k + 0.5) / len(meths))
                ys.append(mu)
                lo.append(mu - l_ if not np.isnan(l_) else 0)
                hi.append(h_ - mu if not np.isnan(h_) else 0)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=METHOD_MARKER[m], color=METHOD_COLOR[m], ms=5, capsize=0,
                        elinewidth=1.5, label=METHOD_LABELS[m])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conds], rotation=30, ha="right")
        n = int(g.groupby(["condition", "method"]).size().min()) if len(g) else 0
        ax.set_title(f"{ESTIMATOR_LABELS[est]} (n={n} seeds/cell)")
        ax.set_xlabel("Participation mechanism (matched 30% dropout)")
    axes[0][0].set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (fraction correct)")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Method")
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} at round {_final_round(runs)}, mean ± 95% bootstrap CI over seeds", y=1.02)
    return _save(fig, out, name)


def fig_dose_response(runs: pd.DataFrame, metric: str, out: Path, name: str) -> list[str]:
    """metric vs beta (log2 x) for MNAR, per method; MCAR mean as dashed reference."""
    ests = [e for e in ESTIMATOR_LABELS if e in set(runs.estimator)]
    meths = _methods(runs)
    fig, axes = plt.subplots(1, len(ests), figsize=(5.2 * len(ests), 3.6), sharey=True, squeeze=False)
    for ax, est in zip(axes[0], ests):
        g = runs[runs.estimator == est]
        mn = g[g.mechanism == "mnar"]
        for m in meths:
            gm = mn[mn.method == m]
            betas = sorted(gm.beta.dropna().unique())
            if not betas:
                continue
            stats_ = [_ci(gm[gm.beta == b][metric]) for b in betas]
            mu = np.array([s[0] for s in stats_])
            lo = np.array([s[1] for s in stats_])
            hi = np.array([s[2] for s in stats_])
            ax.plot(betas, mu, marker=METHOD_MARKER[m], color=METHOD_COLOR[m], ms=5, label=METHOD_LABELS[m])
            ax.fill_between(betas, lo, hi, color=METHOD_COLOR[m], alpha=0.12, linewidth=0)
            ref = g[(g.condition == "mcar") & (g.method == m)][metric]
            if len(ref):
                ax.axhline(ref.mean(), color=METHOD_COLOR[m], lw=1, ls="--", alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("MNAR coupling strength β (log₂ scale; dashed = same method under MCAR)")
        ax.set_title(ESTIMATOR_LABELS[est])
    axes[0][0].set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (fraction correct)")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Method")
    fig.suptitle(f"Dose-response: {METRIC_LABELS.get(metric, metric)} vs drift-dropout coupling (mean, 95% CI band)", y=1.02)
    return _save(fig, out, name)


def _final_round(runs: pd.DataFrame) -> str:
    """Final evaluation round, read from the data (never hard-coded)."""
    if "final_round" not in runs or runs.final_round.isna().all():
        return "end of training"
    r = sorted(set(runs.final_round.dropna().astype(int)))
    return str(r[0]) if len(r) == 1 else "/".join(map(str, r))


def fig_cohort_gap(runs: pd.DataFrame, out: Path, name: str, method: str = "fedavg") -> list[str]:
    """Drift-cohort vs stable-cohort accuracy per condition for one method."""
    ests = [e for e in ESTIMATOR_LABELS if e in set(runs.estimator)]
    conds = _conds(runs)
    fig, axes = plt.subplots(1, len(ests), figsize=(5.2 * len(ests), 3.4), sharey=True, squeeze=False)
    for ax, est in zip(axes[0], ests):
        g = runs[(runs.estimator == est) & (runs.method == method)]
        for k, (lab, col) in enumerate([("Drift cohort", "acc_drift_cohort"), ("Stable cohort", "acc_stable_cohort")]):
            xs, ys, lo, hi = [], [], [], []
            for i, c in enumerate(conds):
                v = g[g.condition == c][col]
                if len(v) == 0:
                    continue
                mu, l_, h_ = _ci(v)
                xs.append(i + (k - 0.5) * 0.28)
                ys.append(mu)
                lo.append(mu - l_ if not np.isnan(l_) else 0)
                hi.append(h_ - mu if not np.isnan(h_) else 0)
            ax.bar(xs, ys, width=0.26, color=COHORT_COLOR[lab], label=lab, edgecolor="white", linewidth=1)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conds], rotation=30, ha="right")
        ax.set_title(f"{METHOD_LABELS[method]} — {ESTIMATOR_LABELS[est]}")
        ax.set_ylim(bottom=max(0, np.nanmin(g[["acc_drift_cohort", "acc_stable_cohort"]].values) - 0.1) if len(g) else 0)
    axes[0][0].set_ylabel(f"Accuracy at round {_final_round(runs)} (fraction correct)")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Drift-cohort vs stable-cohort accuracy by mechanism (mean ± 95% CI)", y=1.02)
    return _save(fig, out, name)


def fig_estimators(runs: pd.DataFrame, out: Path, name: str) -> list[str]:
    """(a) estimator fidelity corr(D, D_true); (b) realized coupling of dropout with
    TRUE drift vs beta, per estimator driving the mechanism."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    fed = runs[(runs.method == "fedavg")]
    data = [fed.corr_Dmodel_Dtrue.dropna().values, fed.corr_Dindep_Dtrue.dropna().values]
    parts = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops={"color": INK})
    for patch, col in zip(parts["boxes"], ["#2a78d6", "#eb6834"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)), d, s=10, color=INK2, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Model-relative D", "Model-independent D"])
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("Mean per-round corr(D, D_true)")
    ax.set_title("(a) Estimator fidelity to injected drift (FedAvg runs)")
    ax = axes[1]
    mn = runs[(runs.mechanism == "mnar") & (runs.method == "fedavg")]
    for est, col, mk in [("model_relative", "#2a78d6", "o"), ("model_independent", "#eb6834", "s")]:
        g = mn[mn.estimator == est]
        betas = sorted(g.beta.unique())
        if not betas:
            continue
        s = [_ci(g[g.beta == b].rho_true_mean_post_onset) for b in betas]
        ax.plot(betas, [x[0] for x in s], marker=mk, color=col, ms=5, label=ESTIMATOR_LABELS[est])
        ax.fill_between(betas, [x[1] for x in s], [x[2] for x in s], color=col, alpha=0.15, linewidth=0)
    ref = runs[(runs.mechanism == "mcar") & (runs.method == "fedavg")].rho_true_mean_post_onset
    if len(ref):
        ax.axhline(ref.mean(), color=INK2, lw=1, ls="--", label="MCAR reference")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MNAR coupling strength β (log₂ scale)")
    ax.set_ylabel("Realized corr(D_true, dropout), post-onset")
    ax.set_title("(b) How much dropout actually tracks TRUE drift")
    ax.legend(loc="upper left")
    return _save(fig, out, name)


def fig_curves(curves: pd.DataFrame, out: Path, name: str, metric: str = "acc_global",
               conditions=("mcar", "mnar_b8")) -> list[str]:
    ests = [e for e in ESTIMATOR_LABELS if e in set(curves.estimator)]
    conds = [c for c in conditions if c in set(curves.condition)]
    meths = [m for m in METHOD_ORDER if m in set(curves.method)]
    fig, axes = plt.subplots(len(ests), len(conds), figsize=(4.6 * len(conds), 3.0 * len(ests)), sharey=True,
                             sharex=True, squeeze=False)
    for r, est in enumerate(ests):
        for c_i, c in enumerate(conds):
            ax = axes[r][c_i]
            g = curves[(curves.estimator == est) & (curves.condition == c)]
            for m in meths:
                gm = g[g.method == m].groupby("round")[metric]
                mu, sd, n = gm.mean(), gm.std(ddof=1), gm.count()
                se = sd / np.sqrt(n)
                ax.plot(mu.index, mu.values, color=METHOD_COLOR[m], lw=1.6, label=METHOD_LABELS[m])
                ax.fill_between(mu.index, (mu - 2 * se).values, (mu + 2 * se).values, color=METHOD_COLOR[m], alpha=0.1, linewidth=0)
            ax.set_title(f"{CONDITION_LABELS[c]} — {ESTIMATOR_LABELS[est]}")
            if r == len(ests) - 1:
                ax.set_xlabel("Communication round")
        axes[r][0].set_ylabel(f"{METRIC_LABELS[metric]}\n(fraction correct)")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Method")
    fig.suptitle("Training curves (mean ± 2 SE over seeds)", y=1.01)
    return _save(fig, out, name)


def fig_feddrift_clusters(curves: pd.DataFrame, clients: pd.DataFrame, out: Path, name: str) -> list[str]:
    fd_c = curves[(curves.method == "feddrift") & curves.feddrift_n_models.notna()] if "feddrift_n_models" in curves else curves.iloc[0:0]
    fd_k = clients[(clients.method == "feddrift")] if "cluster" in clients else clients.iloc[0:0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    conds = [c for c in CONDITION_ORDER if c in set(fd_c.condition)]
    cmap = plt.get_cmap("Blues")
    for i, c in enumerate(conds):
        g = fd_c[fd_c.condition == c].groupby("round").feddrift_n_models.mean()
        ax.plot(g.index, g.values, color=cmap(0.35 + 0.65 * i / max(len(conds) - 1, 1)), lw=1.6, label=CONDITION_LABELS[c])
    ax.set_xlabel("Communication round")
    ax.set_ylabel("FedDrift models\n(mean over seeds × estimators)")
    ax.set_title("(a) FedDrift model count over training")
    if conds:
        ax.legend(loc="upper left", fontsize=7)
    ax = axes[1]
    if len(fd_k):
        fd_k = fd_k.copy()
        sizes = fd_k.groupby(["run_id", "cluster"]).client.transform("count")
        fd_k["cluster_kind"] = np.where(sizes == 1, "Singleton cluster", "Shared cluster")
        for k, (lab, col) in enumerate([("Shared cluster", "#2a78d6"), ("Singleton cluster", "#eb6834")]):
            for j, coh in enumerate([False, True]):
                v = fd_k[(fd_k.cluster_kind == lab) & (fd_k.drift_cohort == coh)].accuracy.dropna()
                if len(v) == 0:
                    continue
                mu, lo, hi = _ci(v)
                x = j + (k - 0.5) * 0.3
                ax.bar(x, mu, width=0.28, color=col, label=lab if j == 0 else None, edgecolor="white")
                ax.errorbar([x], [mu], yerr=[[mu - lo], [hi - mu]], fmt="none", ecolor=INK2, elinewidth=1)
                ax.text(x, 0.02, f"n={len(v)}", ha="center", fontsize=7, color="white")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stable-cohort clients", "Drift-cohort clients"])
        ax.set_ylabel("Final per-client accuracy (fraction correct)")
        ax.set_title("(b) FedDrift per-client accuracy by cluster type")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return _save(fig, out, name)


def fig_seed_variability(runs: pd.DataFrame, metric: str, out: Path, name: str) -> list[str]:
    """Every seed's value (dots) behind the mean and CI, per method (pooled over
    MNAR conditions vs MCAR/MAR), shows run-to-run variability directly."""
    meths = _methods(runs)
    ests = [e for e in ESTIMATOR_LABELS if e in set(runs.estimator)]
    fig, axes = plt.subplots(1, len(ests), figsize=(5.2 * len(ests), 3.4), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, est in zip(axes[0], ests):
        g = runs[runs.estimator == est]
        for k, m in enumerate(meths):
            for j, (lab, mask) in enumerate([("MCAR/MAR", g.mechanism.isin(["mcar", "mar"])), ("MNAR (all β)", g.mechanism == "mnar")]):
                v = g[mask & (g.method == m)][metric].dropna().values
                if len(v) == 0:
                    continue
                x = k + (j - 0.5) * 0.36
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=8, color=METHOD_COLOR[m], alpha=0.45 if j else 0.8,
                           marker=METHOD_MARKER[m], linewidths=0)
                mu, lo, hi = _ci(v)
                ax.errorbar([x], [mu], yerr=[[mu - lo], [hi - mu]], fmt="_", color=INK, ms=14, elinewidth=1.5)
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in meths], rotation=20, ha="right")
        ax.set_title(f"{ESTIMATOR_LABELS[est]} (left = MCAR/MAR, right = MNAR)")
    axes[0][0].set_ylabel(f"{METRIC_LABELS[metric]} (fraction correct)")
    fig.suptitle("Run-level variability: every run (dots), mean and 95% CI (black)", y=1.02)
    return _save(fig, out, name)
