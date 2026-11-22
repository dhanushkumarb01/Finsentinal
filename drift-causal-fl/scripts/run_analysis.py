"""Statistical analysis of a completed (or partially completed) sweep.

    python scripts/run_analysis.py --study mvp
    python scripts/run_analysis.py --root results/smoke --out results/smoke/analysis

Writes CSV tables, model outputs (JSON) and figures into <out>. Only completed
runs are analysed; incomplete seed blocks are dropped from paired/pooled tests
and reported in coverage.json (never imputed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import figures as F  # noqa: E402
from analysis import stats as S  # noqa: E402
from analysis.collect import collect  # noqa: E402

PRIMARY = "acc_global"
OUTCOMES = ["acc_global", "acc_drift_cohort", "acc_stable_cohort", "acc_worst_client", "acc_clean"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = Path(__file__).resolve().parents[1]
    root = Path(args.root) if args.root else base / "results" / args.study
    out = Path(args.out) if args.out else root / "analysis"
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)

    runs, curves, clients = collect(root)
    if runs.empty:
        print("no completed runs")
        return
    runs.to_csv(tab / "runs.csv", index=False)
    curves.to_csv(tab / "curves.csv", index=False)
    clients.to_csv(tab / "clients_final.csv", index=False)

    full, coverage = S.complete_blocks(runs)
    coverage["n_runs_completed"] = int(len(runs))
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    results: dict = {"coverage": coverage, "min_practical_difference": S.MIN_PRACTICAL}

    # A. descriptives (all completed runs, n per cell reported)
    S.describe(runs, OUTCOMES + [f"{m}_lastk" for m in OUTCOMES]).to_csv(tab / "descriptives.csv", index=False)
    diag_cols = ["realized_dropout", "near_exclusion", "participation_freq_std", "rho_mean_post_onset",
                 "rho_true_mean_post_onset", "mi_Dtrue_R_bits", "part_rate_drifting_clientrounds",
                 "part_rate_stable_clientrounds", "corr_Dmodel_Dtrue", "corr_Dindep_Dtrue"]
    diag_cols = [c for c in diag_cols if c in runs]
    runs.groupby(["estimator", "condition", "method"])[diag_cols].mean().reset_index().to_csv(
        tab / "mechanism_diagnostics.csv", index=False)
    mspec = [c for c in runs.columns if c.startswith(("flfdms_", "ipw_", "feddrift_"))]
    if mspec:
        runs.groupby(["estimator", "condition", "method"])[mspec].mean().dropna(how="all").reset_index().to_csv(
            tab / "method_diagnostics.csv", index=False)

    if full.empty:
        print("no complete seed blocks yet; descriptives only")
        return

    # B. pooled models per estimator: LMM (spec primary) + RM-ANOVA effect sizes
    lmm, anova = [], []
    for est, g in full.groupby("estimator"):
        for m in OUTCOMES:
            try:
                r = S.mixed_model(
                    g, m, "C(condition, Treatment('mcar')) * C(method, Treatment('fedavg'))",
                    {"condition_main_and_interaction": "C(method, Treatment('fedavg'))",
                     "condition_x_method_interaction": "C(condition, Treatment('mcar')) + C(method, Treatment('fedavg'))",
                     "method_main_and_interaction": "C(condition, Treatment('mcar'))"})
                r["estimator"] = est
                lmm.append(r)
            except Exception as e:  # noqa: BLE001
                lmm.append({"estimator": est, "metric": m, "error": f"{type(e).__name__}: {e}"})
            if g.seed.nunique() >= 2:
                try:
                    a = S.rm_anova(g, m)
                    a["estimator"] = est
                    anova.append(a)
                except Exception as e:  # noqa: BLE001
                    print("rm_anova failed", est, m, e)
    results["lmm"] = lmm
    if anova:
        pd.concat(anova).to_csv(tab / "rm_anova.csv", index=False)
    # estimator robustness: does the mechanism effect differ by drift definition? (FedAvg, MNAR + MCAR)
    try:
        sub = full[full.method == "fedavg"]
        results["estimator_x_condition_fedavg"] = [S.mixed_model(
            sub, m, "C(condition, Treatment('mcar')) * C(estimator)",
            {"estimator_x_condition": "C(condition, Treatment('mcar')) + C(estimator)"}) for m in OUTCOMES]
    except Exception as e:  # noqa: BLE001
        results["estimator_x_condition_fedavg"] = f"error: {e}"

    # C. pairwise method comparisons, per estimator x condition (Holm within family)
    pairs = [("fedavg", "fedipw_oracle"), ("fedavg", "fedipw_estimated"), ("fedavg", "flfdms"), ("fedavg", "feddrift"),
             ("fedipw_oracle", "flfdms"), ("fedipw_estimated", "flfdms"), ("flfdms", "feddrift"),
             ("fedipw_oracle", "feddrift"), ("fedipw_estimated", "feddrift"), ("fedipw_estimated", "fedipw_oracle")]
    pw = [S.method_vs_method(full, m, a, b) for a, b in pairs for m in OUTCOMES]
    pw = pd.concat([p for p in pw if len(p)])
    pw.to_csv(tab / "pairwise_methods.csv", index=False)
    # consistency: across the 14 (estimator, condition) cells, how often b beats a
    cons = (pw[pw.metric == PRIMARY].groupby(["a", "b"])
            .agg(cells=("mean_diff", "size"), b_better=("mean_diff", lambda x: int((x > 0).sum())),
                 mean_of_diffs=("mean_diff", "mean"),
                 ci_excludes_zero_pos=("ci_lo", lambda x: int((x > 0).sum())),
                 ci_excludes_zero_neg=("ci_hi", lambda x: int((x < 0).sum())),
                 practically_important=("practically_important", "sum")).reset_index())
    cons.to_csv(tab / "pairwise_consistency.csv", index=False)
    # pooled method contrasts across all conditions (seed-paired mean over conditions)
    pooled = []
    for est, g in full.groupby("estimator"):
        per_seed = g.groupby(["seed", "method"])[PRIMARY].mean().unstack()
        for a, b in pairs:
            if a in per_seed and b in per_seed:
                pooled.append(dict(estimator=est, a=a, b=b, metric=PRIMARY, scope="mean over 7 conditions",
                                   **S.paired_stats((per_seed[b] - per_seed[a]).values)))
    pooled = pd.DataFrame(pooled)
    if len(pooled):
        pooled["p_holm"] = S.holm(pooled.p_signflip)
    pooled.to_csv(tab / "pairwise_methods_pooled.csv", index=False)

    # D. H1a (one-sided: drift-cohort acc lower under MNAR than MCAR / MAR) and H1b (two-sided, global)
    h = []
    for ref in ("mcar", "mar"):
        for c in sorted(c for c in full.condition.unique() if c.startswith("mnar")):
            h.append(S.condition_vs_condition(full, "acc_drift_cohort", ref, c, alternative="less").assign(hypothesis="H1a"))
            h.append(S.condition_vs_condition(full, PRIMARY, ref, c, alternative="two-sided").assign(hypothesis="H1b"))
    h = pd.concat([x for x in h if len(x)])
    h["p_holm"] = np.nan
    for (hyp, est, meth), idx in h.groupby(["hypothesis", "estimator", "method"]).groups.items():
        h.loc[idx, "p_holm"] = S.holm(h.loc[idx, "p_signflip"]).values
    h.to_csv(tab / "h1_mechanism_contrasts.csv", index=False)

    # E. dose-response and FedIPW boundary (difference-in-differences)
    dr = pd.concat([S.dose_response(full, m) for m in OUTCOMES])
    dr.to_csv(tab / "dose_response.csv", index=False)
    did = pd.concat([S.did_ipw_boundary(full, m, ipw) for m in (PRIMARY, "acc_drift_cohort")
                     for ipw in ("fedipw_oracle", "fedipw_estimated")])
    did.to_csv(tab / "fedipw_boundary_did.csv", index=False)

    # F. reproducibility: MCAR/MAR duplicates across estimators
    rep = S.reproducibility_check(runs, OUTCOMES)
    rep.to_csv(tab / "reproducibility_mcar_mar_duplicates.csv", index=False)
    results["reproducibility"] = {"cells_compared": int(len(rep)),
                                  "max_abs_diff": float(rep.max_abs_diff.max()) if len(rep) else None,
                                  "identical_cells": int((rep.max_abs_diff == 0).sum()) if len(rep) else 0}

    # G. figures
    figs = []
    for m in OUTCOMES:
        figs += F.fig_method_by_condition(full, m, fig, f"methods_by_condition_{m}")
    figs += F.fig_dose_response(full, PRIMARY, fig, "dose_response_acc_global")
    figs += F.fig_dose_response(full, "acc_drift_cohort", fig, "dose_response_acc_drift_cohort")
    for meth in sorted(full.method.unique()):
        figs += F.fig_cohort_gap(full, fig, f"cohort_gap_{meth}", method=meth)
    figs += F.fig_estimators(full, fig, "estimator_comparison")
    figs += F.fig_curves(curves[curves.run_id.isin(full.run_id)], fig, "curves_acc_global")
    figs += F.fig_curves(curves[curves.run_id.isin(full.run_id)], fig, "curves_acc_drift_cohort", metric="acc_drift_cohort")
    if "feddrift" in set(full.method):
        figs += F.fig_feddrift_clusters(curves[curves.run_id.isin(full.run_id)], clients[clients.run_id.isin(full.run_id)],
                                        fig, "feddrift_clusters")
    figs += F.fig_seed_variability(full, PRIMARY, fig, "seed_variability_acc_global")
    results["figures"] = [str(Path(p).relative_to(out)) for p in figs if p.endswith(".png")]
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({"runs": len(runs), "complete_blocks": coverage, "figures": len(results["figures"]),
                      "reproducibility": results["reproducibility"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
