"""Collect a sweep's run directories into tidy DataFrames (Step 10).

Reads ONLY completed runs (result.json present). Never imputes: a missing run
is reported as missing, and the statistical layer drops incomplete seed blocks
explicitly (see stats.complete_blocks).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["acc_global", "acc_drift_cohort", "acc_stable_cohort", "acc_worst_client", "acc_clean", "acc_client_std"]
METRIC_LABELS = {
    "acc_global": "Global accuracy (primary)",
    "acc_drift_cohort": "Drift-cohort accuracy",
    "acc_stable_cohort": "Stable-cohort accuracy",
    "acc_worst_client": "Worst-client accuracy",
    "acc_clean": "Clean (undrifted) accuracy",
    "acc_client_std": "Across-client accuracy SD",
}
METHOD_ORDER = ["fedavg", "fedipw_oracle", "fedipw_estimated", "flfdms", "feddrift"]
METHOD_LABELS = {"fedavg": "FedAvg", "fedipw_oracle": "FedIPW-oracle", "fedipw_estimated": "FedIPW-estimated",
                 "flfdms": "FL-FDMS", "feddrift": "FedDrift"}
CONDITION_ORDER = ["mcar", "mar", "mnar_b0.5", "mnar_b1", "mnar_b2", "mnar_b4", "mnar_b8"]
CONDITION_LABELS = {"mcar": "MCAR", "mar": "MAR", "mnar_b0.5": "MNAR β=0.5", "mnar_b1": "MNAR β=1",
                    "mnar_b2": "MNAR β=2", "mnar_b4": "MNAR β=4", "mnar_b8": "MNAR β=8"}
ESTIMATOR_LABELS = {"model_relative": "Model-relative D", "model_independent": "Model-independent D"}
LAST_K = 3  # eval points averaged for the "late-training" robustness variant of each metric


def _load(run_dir: Path) -> tuple[dict, dict] | None:
    res = run_dir / "result.json"
    if not res.exists():
        return None
    return json.loads((run_dir / "spec.json").read_text()), json.loads(res.read_text())


def collect(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (runs, curves, clients):
    runs    one row per completed run (final + last-k metrics, diagnostics)
    curves  one row per (run, eval round)
    clients one row per (run, client) at the final round
    """
    root = Path(root)
    rows, curves, clients = [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        loaded = _load(d)
        if loaded is None:
            continue
        spec_doc, res = loaded
        sp = spec_doc["spec"]
        key = dict(run_id=sp["run_id"], seed=sp["seed"], estimator=sp["drift_estimator"], condition=sp["condition"],
                   mechanism=sp["mechanism"], beta=sp["beta"], method=sp["method"])
        rnds = res["rounds"]
        last = rnds[-1]
        row = dict(key)
        row["final_round"] = int(last["round"])
        for m in METRICS:
            row[m] = last[m]
            vals = [e[m] for e in rnds[-LAST_K:] if e[m] is not None]
            row[f"{m}_lastk"] = float(np.mean(vals)) if vals else np.nan
        s = res["summary"]
        row.update(
            realized_dropout=s["mean_realized_dropout_rate"], expected_dropout=s["mean_expected_dropout_rate"],
            near_exclusion=s["near_exclusion_frac"], participation_freq_std=s["participation_freq_std"],
            corr_Dmodel_Dtrue=s["mean_corr_Dmodel_Dtrue"], corr_Dindep_Dtrue=s["mean_corr_Dindep_Dtrue"],
            wall_clock_sec=s["wall_clock_sec"], resumed=s.get("resumed_from_round") is not None,
            config_hash=spec_doc["config_hash"],
        )
        # mechanism diagnostics averaged over the post-onset (drifted) evaluation rounds
        post = [e for e in rnds if e["n_drifting_total"] > 0]
        row["rho_mean_post_onset"] = float(np.nanmean([e["realized_rho"] for e in post])) if post else np.nan
        row["rho_true_mean_post_onset"] = float(np.nanmean([e["realized_rho_true"] for e in post])) if post else np.nan
        row["drift_participation_share_post_onset"] = (
            float(np.mean([e["n_drifting_participated"] / max(e["n_drifting_total"], 1) for e in post])) if post else np.nan)
        # trace-based diagnostics (all rounds, not only eval rounds)
        tr_path = d / "trace.npz"
        if tr_path.exists():
            tr = np.load(tr_path)
            R, Dt = tr["R"].astype(float), tr["D_true"]
            drifted = Dt > 0
            row["part_rate_drifting_clientrounds"] = float(R[drifted].mean()) if drifted.any() else np.nan
            row["part_rate_stable_clientrounds"] = float(R[~drifted].mean())
            row["mi_Dtrue_R_bits"] = _binned_mi(Dt[drifted.any(axis=1)], R[drifted.any(axis=1)])
        mlogs = [e.get("method_log", {}) for e in rnds]
        if sp["method"] == "flfdms":
            fr = [ml.get("friend_same_cohort_frac") for ml in mlogs if ml.get("friend_same_cohort_frac") is not None]
            row["flfdms_friend_same_cohort"] = float(np.mean(fr)) if fr else np.nan
            row["flfdms_no_friend_mean"] = float(np.mean([ml.get("n_no_friend", np.nan) for ml in mlogs]))
        if sp["method"] in ("fedipw_oracle", "fedipw_estimated"):
            row["ipw_ess_mean"] = float(np.nanmean([ml.get("ipw_weight_ess", np.nan) for ml in mlogs]))
            row["ipw_max_ratio_mean"] = float(np.nanmean([ml.get("ipw_weight_max_ratio", np.nan) or np.nan for ml in mlogs]))
        if sp["method"] == "feddrift":
            mf = res.get("method_final", {})
            row["feddrift_models_final"] = mf.get("n_models_final")
            row["feddrift_models_created"] = mf.get("n_models_created")
            row["feddrift_merges"] = mf.get("n_merges")
        rows.append(row)
        for e in rnds:
            c = dict(key, round=e["round"], **{m: e[m] for m in METRICS}, n_participants=e["n_participants"],
                     realized_rho=e["realized_rho"], realized_rho_true=e["realized_rho_true"])
            ml = e.get("method_log") or {}
            if "n_models" in ml:
                c["feddrift_n_models"] = ml["n_models"]
            curves.append(c)
        assign = res.get("method_final", {}).get("assign_final")
        for i, pc in enumerate(res["per_client_final"]):
            cr = dict(key, client=pc["client_id"], drift_cohort=pc["drift_cohort"], accuracy=pc["accuracy"],
                      n_test=pc["n_test"], participation_freq=pc.get("participation_freq"))
            if assign is not None:
                cr["cluster"] = assign[i]
            clients.append(cr)
    runs = pd.DataFrame(rows)
    return runs, pd.DataFrame(curves), pd.DataFrame(clients)


def _binned_mi(D: np.ndarray, R: np.ndarray, bins: int = 5) -> float:
    """Plug-in mutual information I(D_true; R) in bits over client-rounds after
    drift onset (spec Part 5 secondary coupling measure). D_true is a
    deterministic ramp shared by the cohort, so bins = distinct quantile levels."""
    d, r = D.ravel(), R.ravel()
    if len(d) == 0 or r.std() == 0:
        return 0.0
    edges = np.unique(np.quantile(d, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    db = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2)
    joint = np.zeros((len(edges) - 1, 2))
    for a, b in zip(db, r.astype(int)):
        joint[a, b] += 1
    p = joint / joint.sum()
    pd_, pr = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pd_ @ pr)[nz])).sum())
