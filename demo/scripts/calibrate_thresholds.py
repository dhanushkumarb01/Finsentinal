"""Calibrate FinSentinel's detection thresholds for the HAR model (closes his open item S5-10 for the demo).

Method (recorded in demo/config/thresholds.json and demo/PROGRESS.md):
  1. Simulate R clean FedAvg rounds with the exact device training (all 9 device subjects,
     1 epoch, lr 0.05, bs 32, seeds as in the live run) starting from the same initial model.
  2. For every clean client update compute FinSentinel's own features
     (features.update_features.extract_update_features).
  3. threshold = max over clean updates × MARGIN for weight_l2_norm, weight_abs_mean, weight_std;
     max_weight_zero_fraction = 0.5 (clean updates have ~0 exact zeros).
  4. Measure, without changing thresholds: poisoned updates (×8) and an honest drifting device
     (label shift from DRIFT_ROUND) — are they flagged? Report truthfully.
  5. Also calibrate the coordinator's research-layer drift monitor (update distance relative to
     the cohort median) and the device-side drift score used for drift-causal dropout.

Usage: demo/.venv/Scripts/python demo/scripts/calibrate_thresholds.py [--rounds 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parent / "finsentinel"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from features.update_features import extract_update_features  # noqa: E402
from fldemo.fleet import (  # noqa: E402
    delta_l2, device_data, eval_state, local_update, poison,
)
from fldemo.har import load_split, subject_arrays  # noqa: E402
from fldemo.model import new_model, to_numpy_state  # noqa: E402
from fldemo.settings import THRESHOLDS_FILE, load_devices  # noqa: E402

MARGIN = 1.25
EPOCHS, LR = 3, 0.1
METRICS = ("weight_l2_norm", "weight_abs_mean", "weight_std")
DRIFT_DEVICE = "phone-05"


def feats(state) -> dict:
    return extract_update_features({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()})


def fedavg(states, weights):
    w = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    return {k: np.sum([wi * np.asarray(s[k], np.float64) for wi, s in zip(w, states)], axis=0).astype("<f4")
            for k in states[0]}


def flat(state):
    return np.concatenate([np.asarray(v, np.float64).ravel() for v in state.values()])


_REF = {}


def indep(device_id, x):
    from fldemo.drift import drift_embedding_mean
    m = drift_embedding_mean(x)
    if device_id not in _REF:
        _REF[device_id] = m
        return 0.0
    return float(np.sum((m - _REF[device_id]) ** 2))


def simulate(rounds: int, drift_round: int | None, devices: list[dict]):
    torch.set_num_threads(1)
    _REF.clear()
    split = load_split()
    xte, yte = subject_arrays(split["test"])
    g = to_numpy_state(new_model(seed=0))
    rows = []
    for r in range(1, rounds + 1):
        states, ns = [], []
        for i, d in enumerate(devices):
            drifted = drift_round is not None and d["id"] == DRIFT_DEVICE and r >= drift_round
            x, y = device_data(d, drifted=drifted, round_number=r)
            pre_loss = eval_state(g, x, y)[1]
            s, losses = local_update(g, x, y, epochs=EPOCHS, lr=LR, batch_size=32, seed=r * 100 + i)
            f = feats(s)
            rows.append({"round": r, "device": d["id"], "drifted": drifted, "pre_loss": pre_loss,
                         "d_indep": indep(d["id"], x), "_delta": flat(s) - flat(g),
                         "delta_l2": delta_l2(s, g), **{m: f[m] for m in (*METRICS, "weight_zero_fraction")},
                         "poison": {m: feats(poison(s))[m] for m in METRICS} if r % 5 == 0 else None})
            states.append(s)
            ns.append(len(y))
        cur = [x for x in rows if x.get("round") == r and "device" in x]
        for x in cur:
            others = [o["_delta"] for o in cur if o is not x]
            m = np.mean(others, axis=0)
            x["cos"] = float(x["_delta"] @ m / (np.linalg.norm(x["_delta"]) * np.linalg.norm(m) + 1e-12))
        for x in cur:
            x.pop("_delta")
        g = fedavg(states, ns)
        acc, loss = eval_state(g, xte, yte)
        rows.append({"round": r, "global_acc": acc, "global_loss": loss})
        print(f"round {r:2d} acc {acc:.3f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--drift-round", type=int, default=12)
    args = ap.parse_args()
    devices = load_devices()

    clean = simulate(args.rounds, None, devices)
    upd = [r for r in clean if "device" in r]
    thr = {f"max_{m}": max(r[m] for r in upd) * MARGIN for m in METRICS}
    thr["max_weight_zero_fraction"] = 0.5

    def flagged(row) -> int:
        return sum(row[m] > thr[f"max_{m}"] for m in METRICS) + (row["weight_zero_fraction"] > 0.5)

    poison_rules = [sum(p[m] > thr[f"max_{m}"] for m in METRICS) for r in upd if (p := r["poison"])]

    # Research-layer drift monitor: update distance / cohort median, per round.
    def ratios(rows):
        out = []
        for r in sorted({x["round"] for x in rows if "device" in x}):
            rr = [x for x in rows if x.get("round") == r and "device" in x]
            med = float(np.median([x["delta_l2"] for x in rr]))
            out += [(x, x["delta_l2"] / med) for x in rr]
        return out

    clean_cos = [r["cos"] for r in upd if r["round"] > 3]
    monitor_thr = float(np.percentile(clean_cos, 0.5)) - 0.05

    drift = simulate(args.rounds, args.drift_round, devices)
    dr = [r for r in drift if "device" in r]
    drift_dev = [r for r in dr if r["drifted"]]
    drift_cos = [r["cos"] for r in drift_dev]

    # Device-side drift score: pre-training loss on current data minus the device's clean baseline.
    base = {}
    for r in upd:
        base.setdefault(r["device"], []).append(r["pre_loss"])
    clean_scores = [r["d_indep"] for r in upd]
    drift_scores = [r["d_indep"] for r in drift_dev]

    result = {
        "_doc": "Generated by demo/scripts/calibrate_thresholds.py. FinSentinel thresholds = max of "
                f"clean-update features over {args.rounds} simulated rounds x {MARGIN}. Passed to FinSentinel "
                "via FINSENTINEL_MAX_* env vars by the launcher (his code is not modified).",
        "finsentinel_env": {
            "FINSENTINEL_MAX_WEIGHT_L2_NORM": round(thr["max_weight_l2_norm"], 4),
            "FINSENTINEL_MAX_WEIGHT_ABS_MEAN": round(thr["max_weight_abs_mean"], 6),
            "FINSENTINEL_MAX_WEIGHT_STD": round(thr["max_weight_std"], 6),
            "FINSENTINEL_MAX_WEIGHT_ZERO_FRACTION": 0.5,
        },
        "clean_feature_max": {m: max(r[m] for r in upd) for m in METRICS},
        "clean_feature_min": {m: min(r[m] for r in upd) for m in METRICS},
        "clean_false_positives": sum(flagged(r) > 0 for r in upd),
        "clean_updates": len(upd),
        "poison_x8_rules_triggered": {"min": min(poison_rules), "max": max(poison_rules), "n": len(poison_rules)},
        "drift_device_updates": len(drift_dev),
        "drift_device_finsentinel_flagged": sum(flagged(r) > 0 for r in drift_dev),
        "drift_device_feature_max": {m: max(r[m] for r in drift_dev) for m in METRICS},
        "coordinator_drift_monitor": {
            "statistic": "cosine(update delta, mean delta of the other clients this round); flagged when below threshold",
            "threshold": round(monitor_thr, 3),
            "clean_p0_5": round(float(np.percentile(clean_cos, 0.5)), 3),
            "clean_median": round(float(np.median(clean_cos)), 3),
            "drift_device_cos_mean": round(float(np.mean(drift_cos)), 3),
            "drift_device_flag_rate": round(float(np.mean([c < monitor_thr for c in drift_cos])), 3),
        },
        "device_drift_score": {
            "statistic": "D_indep (research estimator #2): squared L2 between mean frozen-random-feature embedding of current data and the device's first-seen reference",
            "clean_max": round(float(np.max(clean_scores)), 4),
            "drift_mean": round(float(np.mean(drift_scores)), 4),
            "drift_min": round(float(np.min(drift_scores)), 4),
        },
        "global_accuracy_clean": [round(r["global_acc"], 4) for r in clean if "global_acc" in r],
        "global_accuracy_with_drift": [round(r["global_acc"], 4) for r in drift if "global_acc" in r],
        "epochs": EPOCHS, "lr": LR, "rounds": args.rounds, "drift_round": args.drift_round, "drift_device": DRIFT_DEVICE,
    }
    THRESHOLDS_FILE.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("global_")}, indent=1))


if __name__ == "__main__":
    main()
