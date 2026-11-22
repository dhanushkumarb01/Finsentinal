"""Sweep execution: one run = one directory (Step 8B/8C).

<root>/<run_id>/
    spec.json        RunSpec + config hash + environment/provenance (written first)
    metrics.jsonl    per-eval-round metrics + method diagnostics (incremental)
    checkpoint.pt    every `checkpoint_every` rounds while running (deleted on success)
    trace.npz        per-round R, p_miss, D_true, D_model, D_indep  [rounds x clients]
    result.json      final result (its presence == run complete)
    attempts.jsonl   one line per attempt: start/end, status, duration, peak RSS, error
    error.txt        traceback of the most recent failed attempt
    lock             exclusive claim (pid, host, start time) while a worker owns the run

Guarantees:
  * Completed runs are never re-executed (result.json present -> skip).
  * Two workers can never execute the same run (O_EXCL lock file; a lock whose
    pid is dead on this host is stale and may be reclaimed).
  * Interrupted runs resume from their last checkpoint, bit-identically.
  * A failed run is recorded and NOT silently retried; it is re-attempted
    only when the operator passes retry_failed=True after fixing the cause.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

from src.data.cache import load_cached_femnist
from src.drift.injection import make_drift_schedule
from src.experiments.config import RunSpec
from src.experiments.status import release, run_status, sweep_report, try_claim  # noqa: F401  (re-exported)
from src.simulation.mechanism_runner import run_mechanism_experiment

REPO = Path(__file__).resolve().parents[2]


def _git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "scripts"], cwd=REPO,
                               capture_output=True, text=True, timeout=10).stdout.strip() != ""
        return {"commit": commit, "src_dirty": dirty}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def environment() -> dict:
    return {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(), "host": socket.gethostname(),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "device": "cpu",
        "flush_denormal": os.environ.get("DCFL_FLUSH_DENORMAL") == "1",
        "git": _git_info(),
    }


_DATA_CACHE: dict = {}


def load_run_data(spec: RunSpec):
    """Clients + reserve pool + drift schedule for a seed. Within a seed every
    mechanism x method x estimator shares the identical client pool and drift
    schedule, so comparisons are paired by seed."""
    d, dr = spec.data, spec.drift
    key = (d["n_clients"], spec.seed, d["n_reserve"], spec.seed + d["reserve_seed_offset"])
    if key not in _DATA_CACHE:
        _DATA_CACHE.clear()  # keep one seed in memory
        _DATA_CACHE[key] = load_cached_femnist(*key)
    clients, reserve, num_classes, prov = _DATA_CACHE[key]
    schedule = make_drift_schedule([c.client_id for c in clients], drift_fraction=dr["fraction"],
                                   drift_start_round=dr["start_round"], rng=np.random.default_rng(spec.seed),
                                   ramp_rounds=dr["ramp_rounds"], mode=dr["mode"], reserve_pool=reserve)
    prov = dict(prov, drift_cohort=sorted(int(c) for c in schedule.drift_start_round),
                drift_source=dict(sorted((int(k), int(v)) for k, v in schedule.drift_source_id.items())))
    return clients, reserve, num_classes, schedule, prov


def execute_run(spec: RunSpec, root: Path, checkpoint_every: int = 10, retry_failed: bool = False,
                verbose: bool = False) -> str:
    """Execute one run if it is not done/claimed. Returns the resulting status."""
    run_dir = root / spec.run_id
    st = run_status(run_dir)
    if st == "done":
        return "done"
    if st == "failed" and not retry_failed:
        return "failed"
    if not try_claim(run_dir):
        return "running"
    t0 = time.time()
    proc = psutil.Process()
    attempt = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "status": "interrupted"}
    try:
        clients, reserve, num_classes, schedule, prov = load_run_data(spec)
        spec_doc = {"spec": spec.to_dict(), "config_hash": spec.config_hash(), "environment": environment(), "data": prov}
        spec_path = run_dir / "spec.json"
        if spec_path.exists():
            old = json.loads(spec_path.read_text())
            if old["config_hash"] != spec_doc["config_hash"]:
                raise RuntimeError(f"{spec.run_id}: existing spec.json has a different config hash; refusing to mix configs")
        spec_path.write_text(json.dumps(spec_doc, indent=2))
        attempt["resumed_from_checkpoint"] = (run_dir / "checkpoint.pt").exists()
        result = run_mechanism_experiment(
            clients=clients, num_classes=num_classes, schedule=schedule, reserve_pool=reserve,
            **spec.runner_kwargs(), log_path=run_dir / "result.json", run_dir=run_dir,
            checkpoint_every=checkpoint_every, verbose=verbose,
        )
        attempt.update(status="done", final_acc_global=result["summary"]["final_acc_global"])
        (run_dir / "error.txt").unlink(missing_ok=True)
        status = "done"
    except Exception as e:  # noqa: BLE001 — recorded, never swallowed silently
        (run_dir / "error.txt").write_text(traceback.format_exc())
        attempt.update(status="failed", error=f"{type(e).__name__}: {e}")
        status = "failed"
    finally:
        attempt.update(end=time.strftime("%Y-%m-%dT%H:%M:%S"), duration_sec=round(time.time() - t0, 1),
                       rss_mb=round(proc.memory_info().rss / 2**20, 1))
        with open(run_dir / "attempts.jsonl", "a") as f:
            f.write(json.dumps(attempt) + "\n")
        release(run_dir)
    return status
