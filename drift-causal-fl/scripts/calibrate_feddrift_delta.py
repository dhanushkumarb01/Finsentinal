"""Choose FedDrift's delta from the paper's grid {0.02, 0.04, ..., 0.20}
(Appendix A) using the held-out tuning-seed pilot (configs/pilot_calibration.yaml).

The pilot runs FedDrift with delta = 1e6 (never flags, never merges, so it is
exactly FedAvg) and logs the drift-test statistic
    s_{c,t} = min_m l_{c,m}^t - min_m l_{c,m}^{t_prev}
for every tested participant. For each candidate delta, the test flags iff
s > delta:
  * TPR = flag rate over drift-cohort client-rounds inside the drift window
    [onset, onset + ramp + slack] (the rounds in which their distribution changes);
  * FPR = flag rate over stable-cohort client-rounds after onset (the regime in
    which a spurious flag would spawn a useless singleton model).
Selection rule (stated before looking at the result): maximise Youden's
J = TPR - FPR; ties -> the larger delta (fewer spurious models). The paper
selects delta by test accuracy, which would need one full run per grid value;
this detection-based rule is the documented, affordable substitute. Uses only
the tuning seed (100), never a sweep seed.

    python scripts/calibrate_feddrift_delta.py [--run-dir results/pilot_calibration/s100__mrel__mcar__feddrift]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID = [round(0.02 * k, 2) for k in range(1, 11)]


def main() -> None:
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    ap.add_argument("--run-dir", default=str(base / "results/pilot_calibration/s100__mrel__mcar__feddrift"))
    ap.add_argument("--slack", type=int, default=5, help="rounds after the ramp still counted as the drift window")
    args = ap.parse_args()
    d = Path(args.run_dir)
    spec = json.loads((d / "spec.json").read_text())
    cohort = set(spec["data"]["drift_cohort"])
    onset, ramp = spec["spec"]["drift"]["start_round"], spec["spec"]["drift"]["ramp_rounds"]
    window = (onset, onset + ramp - 1 + args.slack)
    pos, neg, last_round = [], [], 0
    per_round_stable = []
    for line in (d / "metrics.jsonl").read_text().splitlines():
        e = json.loads(line)
        ts = (e.get("method_log") or {}).get("drift_test_stat")
        if ts is None:
            continue
        r = e["round"]
        last_round = max(last_round, r)
        stable_r = []
        for c, s in ts.items():
            c = int(c)
            if c in cohort and window[0] <= r <= window[1]:
                pos.append(s)
            elif c not in cohort and r >= onset:
                neg.append(s)
                stable_r.append(s)
        if r >= onset:
            per_round_stable.append(len(stable_r))
    pos, neg = np.array(pos), np.array(neg)
    rows = []
    for delta in GRID:
        tpr = float((pos > delta).mean()) if len(pos) else float("nan")
        fpr = float((neg > delta).mean()) if len(neg) else float("nan")
        rows.append({"delta": delta, "TPR": round(tpr, 4), "FPR": round(fpr, 4), "J": round(tpr - fpr, 4),
                     "expected_spurious_flags_per_round": round(fpr * float(np.mean(per_round_stable or [0])), 2)})
    best = max(rows, key=lambda x: (x["J"], x["delta"]))
    out = {"run_dir": str(d), "rounds_observed": last_round, "drift_window": window,
           "n_positive_clientrounds": int(len(pos)), "n_negative_clientrounds": int(len(neg)),
           "stat_quantiles_drift_window": np.round(np.quantile(pos, [0.1, 0.5, 0.9]), 4).tolist() if len(pos) else None,
           "stat_quantiles_stable": np.round(np.quantile(neg, [0.5, 0.9, 0.99]), 4).tolist() if len(neg) else None,
           "grid": rows, "selected_delta": best["delta"], "rule": "max Youden J, ties -> larger delta"}
    print(json.dumps(out, indent=2))
    (d.parent / "delta_calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
