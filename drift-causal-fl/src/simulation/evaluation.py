"""Drift-aware, subpopulation-level evaluation (spec Part 1 target estimand +
Part 9 metrics).

Why this exists (bug in the original pipeline): `fedavg_baseline.evaluate`
scores the global model on every client's ORIGINAL, never-drifted test split.
Drifting clients were trained on shifted data but evaluated as if nothing had
changed, so "accuracy" measured performance on the pre-drift population — the
estimand the spec explicitly rejects as (b)-like, which would make H1a/H1b
untestable. Here every client is scored on its test set *as it looks at the
evaluated round* (injection.apply_drift_to_test), built only from held-out
samples, and the result is broken down by subpopulation.

God-view note: this module uses the simulator's ground truth (which clients
are scheduled to drift, D_true). That is permitted for EVALUATION only (spec
Part 1); nothing here is ever fed back into a mechanism or an FL algorithm.

Metrics (all on the current-round, drift-injected held-out test sets):
  acc_global          PRIMARY — pooled accuracy over all clients (estimand (a))
  acc_drift_cohort    pooled over clients scheduled to drift (H1a subpopulation)
  acc_stable_cohort   pooled over never-drifting clients
  acc_client_mean     unweighted mean of per-client accuracies
  acc_client_std      dispersion across clients (fairness)
  acc_worst_client    min per-client accuracy (fairness)
  acc_clean           LEGACY metric — pooled accuracy on the original,
                      undrifted test splits. Identical to what the old
                      `evaluate()` reported as "accuracy"; kept only for
                      comparability with pre-fix logs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.data.loaders import ClientData
from src.drift.injection import DriftSchedule, apply_drift_to_test


@torch.no_grad()
def _correct(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> int:
    if len(y) == 0:
        return 0
    return int((model(x.to(device)).argmax(dim=1) == y.to(device)).sum().item())


def _pooled(correct: np.ndarray, total: np.ndarray, mask: np.ndarray) -> float | None:
    t = total[mask].sum()
    return float(correct[mask].sum() / t) if t > 0 else None


@torch.no_grad()
def evaluate_drift_aware(
    model: nn.Module | Callable[[int], nn.Module],
    clients: list[ClientData],
    schedule: DriftSchedule,
    round_num: int,
    reserve_by_id: dict[int, ClientData] | None,
    device: str,
    eval_seed: int = 0,
) -> tuple[dict, list[dict]]:
    """Returns (aggregate_metrics, per_client_rows).

    `model` is either one model used by every client (FedAvg/FedIPW/FL-FDMS)
    or a callable idx -> model giving each client's own inference model
    (FedDrift: the client's cluster model)."""
    model_for = (lambda i: model) if isinstance(model, nn.Module) else model
    n = len(clients)
    used = {}
    correct, total = np.zeros(n), np.zeros(n)
    clean_correct, clean_total = np.zeros(n), np.zeros(n)
    in_drift_cohort = np.zeros(n, dtype=bool)
    rows = []
    for i, c in enumerate(clients):
        mag = schedule.true_drift_magnitude(c.client_id, round_num)
        x, y = apply_drift_to_test(c, mag, schedule, round_num, reserve_by_id, eval_seed)
        m = model_for(i)
        if id(m) not in used:
            used[id(m)] = (m, m.training)
            m.eval()
        correct[i], total[i] = _correct(m, x, y, device), len(y)
        clean_correct[i], clean_total[i] = _correct(m, c.test_x, c.test_y, device), len(c.test_y)
        in_drift_cohort[i] = c.client_id in schedule.drift_start_round
        rows.append({
            "client_id": c.client_id,
            "drift_cohort": bool(in_drift_cohort[i]),
            "d_true": mag,
            "n_test": int(total[i]),
            "accuracy": float(correct[i] / total[i]) if total[i] else None,
        })
    for m, was_training in used.values():
        if was_training:
            m.train()

    has_test = total > 0
    per_client = correct[has_test] / total[has_test]
    metrics = {
        "acc_global": _pooled(correct, total, np.ones(n, dtype=bool)),
        "acc_drift_cohort": _pooled(correct, total, in_drift_cohort),
        "acc_stable_cohort": _pooled(correct, total, ~in_drift_cohort),
        "acc_client_mean": float(per_client.mean()) if len(per_client) else None,
        "acc_client_std": float(per_client.std()) if len(per_client) else None,
        "acc_worst_client": float(per_client.min()) if len(per_client) else None,
        "acc_clean": _pooled(clean_correct, clean_total, np.ones(n, dtype=bool)),
        "n_test_global": int(total.sum()),
        "n_test_drift_cohort": int(total[in_drift_cohort].sum()),
    }
    return metrics, rows
