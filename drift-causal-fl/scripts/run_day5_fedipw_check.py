"""Day 5 — the actual theoretical test this whole project is built around.

Per the spec (Part 11): FedIPW's Mean Ignorability assumption should hold
under MCAR/MAR (participation depends only on things outside D), so its
correction should behave normally there. Under MNAR, participation depends
on D specifically — which FedIPW-estimated never gets to see — so its
propensity estimates should be systematically wrong for high/low-drift
clients, and its correction should degrade relative to how it performs
under MCAR/MAR.

This script runs FedAvg / FedIPW-oracle / FedIPW-estimated under both MCAR
and MNAR (single seed, reduced scale — a first look, not the final sweep)
and reports the mechanism x method pattern directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.loaders import load_clients, load_femnist_reserve_pool
from src.drift.injection import make_drift_schedule
from src.simulation.mechanism_runner import run_mechanism_experiment

METHODS = ["fedavg", "fedipw_oracle", "fedipw_estimated"]


def main() -> None:
    ap = argparse.ArgumentParser()
    # Defaults = the original Day-5 v2 pilot configuration (PROGRESS.md).
    ap.add_argument("--clients", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-rate", type=float, default=0.30)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--threshold-exclude-frac", type=float, default=0.2)
    ap.add_argument("--drift-estimator", choices=["model_relative", "model_independent"], default="model_independent")
    ap.add_argument("--mechanisms", nargs="+", default=["mcar", "mnar"], choices=["mcar", "mar", "mnar", "mnar_threshold"])
    ap.add_argument("--out-prefix", default="results/day5_v3")
    args = ap.parse_args()

    seed = args.seed
    rng = np.random.default_rng(seed)

    print("=" * 70)
    print("DAY 5 — FedAvg vs FedIPW-oracle vs FedIPW-estimated, by mechanism")
    print(f"config: {vars(args)}")
    print("=" * 70)

    clients, num_classes, name = load_clients(num_clients=args.clients, seed=seed)
    client_ids = [c.client_id for c in clients]
    reserve_pool = load_femnist_reserve_pool(n_reserve=15, exclude_seed=seed, exclude_num_clients=args.clients)
    schedule = make_drift_schedule(client_ids, drift_fraction=0.4, drift_start_round=2, rng=rng, ramp_rounds=6, mode="label_shift", reserve_pool=reserve_pool)
    print(f"[drift] mode={schedule.mode} (real-writer label shift, no pixel corruption), "
          f"{len(schedule.drift_start_round)}/{len(clients)} clients drift; MNAR driven by D_{args.drift_estimator}")

    results = {}
    for mechanism in args.mechanisms:
        for method in METHODS:
            key = f"{mechanism}_{method}"
            print(f"\n--- {mechanism.upper()} + {method} ---")
            results[key] = run_mechanism_experiment(
                clients=clients,
                num_classes=num_classes,
                schedule=schedule,
                mechanism=mechanism,
                method=method,
                target_rate=args.target_rate,
                rounds=args.rounds,
                beta=args.beta,
                threshold_exclude_frac=args.threshold_exclude_frac,
                drift_estimator=args.drift_estimator,
                local_epochs=1,
                lr=0.03,
                batch_size=10,
                eval_every=5,
                seed=seed,
                reserve_pool=reserve_pool,
                log_path=f"{args.out_prefix}_{args.drift_estimator}_{key}.json",
            )

    print("\n" + "=" * 70)
    print(f"SUMMARY — final round {args.rounds}, drift-aware evaluation (D estimator: {args.drift_estimator})")
    print("acc = global (primary, drifted test sets) | drift = drift cohort (H1a) | stable | clean = legacy undrifted")
    print("=" * 70)
    print(f"{'condition':<28} {'acc':>7} {'drift':>7} {'stable':>7} {'worst':>7} {'clean':>7} | {'E[drop]':>7} {'drop':>6} {'nearX':>6}")
    for key, r in results.items():
        s = r["summary"]
        print(f"{key:<28} {s['final_acc_global']:>7.4f} {s['final_acc_drift_cohort']:>7.4f} {s['final_acc_stable_cohort']:>7.4f} "
              f"{s['final_acc_worst_client']:>7.4f} {s['final_acc_clean']:>7.4f} | {s['mean_expected_dropout_rate']:>7.4f} "
              f"{s['mean_realized_dropout_rate']:>6.3f} {s['near_exclusion_frac']:>6.3f}")
    any_s = next(iter(results.values()))["summary"]
    print(f"\nmean corr(D, D_true) over rounds: model-relative = {any_s['mean_corr_Dmodel_Dtrue']}, "
          f"model-independent = {any_s['mean_corr_Dindep_Dtrue']}")
    print("\nFedIPW gain over FedAvg (global acc | drift-cohort acc):")
    for method in METHODS[1:]:
        parts = []
        for mechanism in args.mechanisms:
            g = results[f"{mechanism}_{method}"]["summary"]["final_acc_global"] - results[f"{mechanism}_fedavg"]["summary"]["final_acc_global"]
            d = results[f"{mechanism}_{method}"]["summary"]["final_acc_drift_cohort"] - results[f"{mechanism}_fedavg"]["summary"]["final_acc_drift_cohort"]
            parts.append(f"{mechanism.upper()} {g:+.4f} | {d:+.4f}")
        print(f"  {method:<18} " + "   ".join(parts))
    print("=" * 70)
    with open(f"{args.out_prefix}_{args.drift_estimator}_summary.json", "w") as f:
        json.dump({"config": vars(args), "summaries": {k: r["summary"] for k, r in results.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
