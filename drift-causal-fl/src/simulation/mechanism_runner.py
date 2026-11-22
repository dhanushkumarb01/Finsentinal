"""The MVP loop: drift injection + a dropout mechanism + an aggregation
method (Strategy), combined into one experiment.

Everything here was validated independently first (Day 1: FedAvg baseline
works; Day 2: drift estimators detect real drift; Day 3/4: mechanisms
produce correctly-calibrated, correctly-correlated dropout on real data).

Step-4/5 fixes (see PROGRESS.md):
  - Evaluation is drift-aware and subpopulation-level (simulation/evaluation.py).
  - The drift score that drives MNAR is selectable: "model_relative" (spec
    Part 3A, lagged model theta_{t-1}) or "model_independent" (Part 3B). BOTH
    are computed every round and logged, whichever drives the mechanism.
  - Threshold MNAR mechanism wired in; one shared positivity band.
  - Per-round calibration diagnostics.

Step-6/7/8 changes:
  - Aggregation is delegated to a `Strategy` (src/methods/): FedAvg,
    FedIPW-oracle, FedIPW-estimated, FL-FDMS, FedDrift. The loop itself is
    method-agnostic; the RNG consumption order is unchanged, so FedAvg/FedIPW
    results are bit-identical to the pre-refactor loop (regression-tested
    against tests/fixtures/legacy_runner_outputs.json).
  - Aggregation is streamed (no list of 100 state_dicts in memory).
  - The model-relative estimator's lagged model theta_{t-1} is the model each
    client used in the previous round (its cluster model under FedDrift).
  - Checkpoint / resume (bit-identical continuation), incremental JSONL
    metrics, and a per-round trace (R, p_miss, D_true, D_model, D_indep).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import ClientData, validate_labels
from src.drift.injection import DriftSchedule, apply_drift
from src.drift.model_independent import FrozenFeatureExtractor, ModelIndependentDriftTracker
from src.drift.model_relative import ModelRelativeDriftTracker
from src.mechanisms.dropout import (
    mar_dropout,
    mcar_dropout,
    mnar_drift_dropout,
    mnar_threshold_dropout,
    realized_correlation,
)
from src.methods.base import RoundContext, Strategy
from src.methods.fedavg import FedAvgStrategy, GlobalModelStrategy
from src.methods.feddrift import FedDriftStrategy
from src.methods.fedipw import FedIPWEstimatedStrategy, FedIPWOracleStrategy
from src.methods.flfdms import FLFDMSStrategy
from src.models.cnn import SmallCNN
from src.simulation.evaluation import evaluate_drift_aware
from src.simulation.fedavg_baseline import local_train

from src.constants import DRIFT_ESTIMATORS, MECHANISMS, METHODS  # noqa: E402,F401  (re-exported)
NEAR_EXCLUSION_P = 0.10  # spec Part 2: report client-rounds with P(participate) < 0.1
CHECKPOINT_VERSION = 1


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def build_strategy(method: str, model: torch.nn.Module, n_clients: int, method_kwargs: dict | None = None) -> Strategy:
    kw = dict(method_kwargs or {})
    if method == "fedavg":
        return FedAvgStrategy(model)
    if method == "fedipw_oracle":
        return FedIPWOracleStrategy(model)
    if method == "fedipw_estimated":
        return FedIPWEstimatedStrategy(model)
    if method == "flfdms":
        return FLFDMSStrategy(model, n_clients)
    if method == "feddrift":
        return FedDriftStrategy(model, n_clients, **kw)
    raise ValueError(method)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def run_mechanism_experiment(
    clients: list[ClientData],
    num_classes: int,
    schedule: DriftSchedule,
    mechanism: str,  # see MECHANISMS
    target_rate: float,
    rounds: int,
    method: str = "fedavg",  # see METHODS
    reserve_pool: list[ClientData] | None = None,  # required if schedule.mode == "label_shift"
    drift_estimator: str = "model_independent",  # see DRIFT_ESTIMATORS
    beta: float = 2.0,
    gamma: float = 2.0,
    threshold_exclude_frac: float = 0.2,
    model_relative_ema_decay: float = 0.7,
    local_epochs: int = 1,
    lr: float = 0.02,
    batch_size: int = 10,
    eval_every: int = 4,
    seed: int = 0,
    eval_seed: int | None = None,
    device: str | None = None,
    log_path: str | Path | None = None,
    verbose: bool = True,
    method_kwargs: dict | None = None,
    run_dir: str | Path | None = None,  # if set: metrics.jsonl, trace.npz, checkpoint.pt live here
    checkpoint_every: int = 0,  # rounds between checkpoints (0 = never); requires run_dir
    resume: bool = True,
    save_final_model: bool = False,
    stop_after_round: int | None = None,  # testing hook: simulate an interruption
) -> dict:
    """Run one full FL simulation under ONE dropout mechanism AND ONE
    aggregation method, with drift injection active throughout.

    CRITICAL invariant (spec Part 1): D (any estimator) and D_true are
    simulator god-view. They are used ONLY to generate R (the mechanism is
    "nature") and for evaluation/diagnostics. No Strategy ever receives D:
    strategies only see participants' sample counts/data/updates
    (RoundContext), and FedIPW-oracle additionally the true generating
    propensity (its explicit, documented privilege).

    Returns {"config", "rounds", "summary", "per_client_final", "method_final"};
    the same object is written to `log_path` as JSON.
    """
    assert mechanism in MECHANISMS, mechanism
    assert method in METHODS, method
    assert drift_estimator in DRIFT_ESTIMATORS, drift_estimator
    validate_labels(clients, num_classes, "clients")
    if reserve_pool:
        validate_labels(reserve_pool, num_classes, "reserve pool")
    eval_seed = seed if eval_seed is None else eval_seed
    config = {k: v for k, v in locals().items() if k in (
        "mechanism", "method", "drift_estimator", "target_rate", "rounds", "beta", "gamma",
        "threshold_exclude_frac", "model_relative_ema_decay", "local_epochs", "lr", "batch_size",
        "eval_every", "seed", "eval_seed", "num_classes")}
    config.update(n_clients=len(clients), drift_mode=schedule.mode,
                  n_drift_cohort=len(schedule.drift_start_round), ramp_rounds=schedule.ramp_rounds)
    if method_kwargs:
        config["method_kwargs"] = dict(method_kwargs)

    run_dir = Path(run_dir) if run_dir is not None else None
    if checkpoint_every and run_dir is None:
        raise ValueError("checkpoint_every requires run_dir")
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt" if run_dir else None
    jsonl_path = run_dir / "metrics.jsonl" if run_dir else None

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or "cpu"

    # NOTE: model construction order matters for the torch RNG stream (kept
    # identical to the pre-refactor loop).
    global_model = SmallCNN(num_classes=num_classes).to(device)
    lagged_model = SmallCNN(num_classes=num_classes).to(device)  # theta_{t-1}, single-model methods
    lagged_model.load_state_dict(global_model.state_dict())
    lag_for = lambda idx: lagged_model  # noqa: E731
    indep_tracker = ModelIndependentDriftTracker(FrozenFeatureExtractor(out_dim=32, seed=1234), device=device)
    rel_tracker = ModelRelativeDriftTracker(ema_decay=model_relative_ema_decay)
    strategy = build_strategy(method, global_model, len(clients), method_kwargs)
    single_model = isinstance(strategy, GlobalModelStrategy)

    n = len(clients)
    fixed_X = rng.normal(size=n)  # fixed MAR covariate, drift-independent by construction
    reserve_by_id = {c.client_id: c for c in reserve_pool} if reserve_pool else {}
    drift_cohort = np.array([c.client_id in schedule.drift_start_round for c in clients])

    log: list[dict] = []
    per_round_expected, per_round_realized, per_round_near_excl = [], [], []
    corr_rel, corr_indep = [], []
    participation_counts = np.zeros(n, dtype=int)
    trace = {k: np.full((rounds, n), np.nan, dtype=np.float32) for k in ("D_true", "D_model", "D_indep", "p_miss")}
    trace["R"] = np.zeros((rounds, n), dtype=np.int8)
    per_client: list[dict] = []
    start_round = 1
    elapsed_before = 0.0
    resumed_from = None

    if ckpt_path is not None and resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        if ck.get("version") != CHECKPOINT_VERSION or ck["config"] != _jsonable(config):
            raise ValueError(f"checkpoint at {ckpt_path} does not match this run's config; refusing to resume")
        strategy.set_state(ck["strategy"])
        if single_model:
            lagged_model.load_state_dict(ck["lagged"])
        else:
            lag_for = strategy.snapshot() if ck["lagged"] is None else ck["lagged"]
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        rel_tracker._ema = dict(ck["rel_ema"])
        indep_tracker._reference_mean = {k: v.clone() for k, v in ck["indep_ref"].items()}
        fixed_X = ck["fixed_X"]
        log = ck["log"]
        per_round_expected, per_round_realized, per_round_near_excl = ck["per_round"]
        corr_rel, corr_indep = ck["corr"]
        participation_counts = ck["participation_counts"]
        trace = ck["trace"]
        per_client = ck["per_client"]
        start_round = ck["round"] + 1
        elapsed_before = ck["elapsed"]
        resumed_from = ck["round"]
        if verbose:
            print(f"[resume] {run_dir} from round {ck['round']}")
    else:
        # Warm-up (round 0): reference window for the model-independent estimator
        # and initial loss EMA for the model-relative one, both on pre-drift data
        # under the initial model. Consumes no RNG. No dropout yet.
        for client in clients:
            indep_tracker.compute(client.client_id, client.train_x)
            rel_tracker.compute(client.client_id, lagged_model, client.train_x, client.train_y, device)

    if jsonl_path is not None:  # rewrite completed entries (drops any written after the checkpoint)
        with open(jsonl_path, "w") as f:
            for e in log:
                f.write(json.dumps(_jsonable(e)) + "\n")

    t0 = time.time()

    for rnd in range(start_round, rounds + 1):
        # 1. this round's drift-affected data + drift scores per client (god view).
        D_true = np.zeros(n)
        D_rel, D_indep = np.zeros(n), np.zeros(n)
        round_x, round_y = [], []
        for idx, client in enumerate(clients):
            D_true[idx] = schedule.true_drift_magnitude(client.client_id, rnd)
            x, y = apply_drift(client.train_x, client.train_y, D_true[idx], schedule, client.client_id, rng, reserve_by_id)
            round_x.append(x)
            round_y.append(y)
            D_indep[idx] = indep_tracker.compute(client.client_id, x)
            D_rel[idx] = rel_tracker.compute(client.client_id, lag_for(idx), x, y, device)
        D = D_rel if drift_estimator == "model_relative" else D_indep
        c_rel, c_ind = _safe_corr(D_rel, D_true), _safe_corr(D_indep, D_true)
        if c_rel is not None:
            corr_rel.append(c_rel)
        if c_ind is not None:
            corr_indep.append(c_ind)

        # 2. who participates. p_miss is the TRUE generating miss-probability.
        if mechanism == "mcar":
            participates, p_miss = mcar_dropout(n, target_rate, rng)
        elif mechanism == "mar":
            participates, p_miss = mar_dropout(fixed_X, target_rate, gamma, rng)
        elif mechanism == "mnar":
            participates, p_miss = mnar_drift_dropout(D, target_rate, beta, rng)
        else:
            participates, p_miss = mnar_threshold_dropout(D, target_rate, threshold_exclude_frac, rng)
        participation_counts += participates.astype(int)
        per_round_expected.append(float(p_miss.mean()))
        per_round_realized.append(float(1 - participates.mean()))
        per_round_near_excl.append(float(np.mean(1 - p_miss < NEAR_EXCLUSION_P)))
        rho = realized_correlation(D, participates)
        rho_true = realized_correlation(D_true, participates)
        r0 = rnd - 1
        trace["D_true"][r0], trace["D_model"][r0], trace["D_indep"][r0] = D_true, D_rel, D_indep
        trace["p_miss"][r0], trace["R"][r0] = p_miss, participates.astype(np.int8)

        # 3. the strategy sees ONLY participants (+ oracle propensity for FedIPW-oracle).
        parts = [int(i) for i in np.flatnonzero(participates)]
        ctx = RoundContext(
            rnd=rnd, n_clients=n, participants=parts,
            n_samples={i: int(len(round_y[i])) for i in parts},
            data={i: (round_x[i], round_y[i]) for i in parts} if strategy.needs_participant_data else {},
            oracle_propensity={i: float(1.0 - p_miss[i]) for i in parts} if strategy.needs_oracle_propensity else None,
            device=device,
        )
        strategy.begin_round(ctx)

        drifting_participants = 0
        for idx in parts:
            local_model = SmallCNN(num_classes=num_classes).to(device)  # (consumes torch RNG, as before)
            local_model.load_state_dict(strategy.start_state(idx))
            client = clients[idx]
            sd = local_train(local_model, ClientData(client.client_id, round_x[idx], round_y[idx], client.test_x, client.test_y),
                             local_epochs, lr, batch_size, device)
            strategy.receive(idx, sd)
            if D_true[idx] > 0:
                drifting_participants += 1

        # theta_{t-1} for next round's model-relative D = the models used this round.
        if single_model:
            lagged_model.load_state_dict(strategy.global_model.state_dict())
        else:
            lag_for = strategy.snapshot()
        strategy.end_round(ctx)
        mlog = strategy.round_log()
        if "friend_map" in mlog and mlog["friend_map"]:  # FL-FDMS diagnostic (evaluation-side god view)
            fm = mlog["friend_map"]
            mlog["friend_same_cohort_frac"] = float(np.mean([drift_cohort[k] == drift_cohort[f] for k, f in fm.items()]))

        if rnd % eval_every == 0 or rnd == 1 or rnd == rounds:
            metrics, per_client = evaluate_drift_aware(strategy.eval_model, clients, schedule, rnd, reserve_by_id, device, eval_seed)
            entry = {
                "round": rnd,
                **metrics,
                "n_participants": int(participates.sum()),
                "expected_dropout_rate": per_round_expected[-1],
                "realized_dropout_rate": per_round_realized[-1],
                "realized_rho": rho,
                "realized_rho_true": rho_true,
                "corr_Dmodel_Dtrue": c_rel,
                "corr_Dindep_Dtrue": c_ind,
                "n_drifting_total": int((D_true > 0).sum()),
                "n_drifting_participated": drifting_participants,
                "elapsed_sec": round(elapsed_before + time.time() - t0, 1),
            }
            if mlog:
                entry["method_log"] = _jsonable(mlog)
            log.append(entry)
            if jsonl_path is not None:
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(_jsonable(entry)) + "\n")
            if verbose:
                extra = f" models={mlog['n_models']}" if "n_models" in mlog else ""
                print(
                    f"[{mechanism:>4}/{method} r{rnd:>3}/{rounds}] acc={metrics['acc_global']:.4f} "
                    f"drift={metrics['acc_drift_cohort']:.4f} stable={metrics['acc_stable_cohort']:.4f} "
                    f"clean={metrics['acc_clean']:.4f}  part={int(participates.sum())}/{n} "
                    f"rho={rho:+.3f}  drifting_part={drifting_participants}/{int((D_true > 0).sum())}{extra}",
                    flush=True,
                )
        elif mlog and jsonl_path is not None:  # method diagnostics for non-eval rounds too
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(_jsonable({"round": rnd, "method_log": mlog})) + "\n")

        if ckpt_path is not None and checkpoint_every and (rnd % checkpoint_every == 0) and rnd < rounds:
            ck = {
                "version": CHECKPOINT_VERSION, "config": _jsonable(config), "round": rnd,
                "strategy": strategy.get_state(),
                "lagged": lagged_model.state_dict() if single_model else None,
                "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "rel_ema": dict(rel_tracker._ema),
                "indep_ref": {k: v.clone() for k, v in indep_tracker._reference_mean.items()},
                "fixed_X": fixed_X, "log": log,
                "per_round": (per_round_expected, per_round_realized, per_round_near_excl),
                "corr": (corr_rel, corr_indep), "participation_counts": participation_counts,
                "trace": trace, "per_client": per_client,
                "elapsed": elapsed_before + time.time() - t0,
            }
            if not single_model:
                # FedDrift's lagged snapshot == the models used in this round, which are
                # recoverable only from the snapshot itself; store its states.
                ck["lagged"] = lag_for
            _atomic_torch_save(ck, ckpt_path)
        if stop_after_round is not None and rnd >= stop_after_round:
            return {"interrupted_at": rnd}

    part_freq = participation_counts / rounds
    final = log[-1]
    summary = {
        "final_acc_global": final["acc_global"],
        "final_acc_drift_cohort": final["acc_drift_cohort"],
        "final_acc_stable_cohort": final["acc_stable_cohort"],
        "final_acc_worst_client": final["acc_worst_client"],
        "final_acc_client_std": final["acc_client_std"],
        "final_acc_clean": final["acc_clean"],
        "mean_expected_dropout_rate": float(np.mean(per_round_expected)),
        "mean_realized_dropout_rate": float(np.mean(per_round_realized)),
        "near_exclusion_frac": float(np.mean(per_round_near_excl)),
        "participation_freq_std": float(part_freq.std()),
        "mean_corr_Dmodel_Dtrue": float(np.mean(corr_rel)) if corr_rel else None,
        "mean_corr_Dindep_Dtrue": float(np.mean(corr_indep)) if corr_indep else None,
        "wall_clock_sec": round(elapsed_before + time.time() - t0, 1),
    }
    if resumed_from is not None:
        summary["resumed_from_round"] = resumed_from
    for row, freq in zip(per_client, part_freq):
        row["participation_freq"] = float(freq)
    result = {"config": config, "rounds": log, "summary": summary, "per_client_final": per_client}
    mfinal = strategy.final_log()
    if mfinal:
        result["method_final"] = _jsonable(mfinal)

    if run_dir is not None:
        np.savez_compressed(run_dir / "trace.npz", **trace)
        if save_final_model:
            _atomic_torch_save(strategy.get_state(), run_dir / "final_model.pt")
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(log_path) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(_jsonable(result), f, indent=2)
        os.replace(tmp, log_path)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()  # run complete; final state is in result.json (+ final_model.pt if requested)
    return result
