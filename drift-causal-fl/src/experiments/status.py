"""Torch-free run-state helpers: status, locking, sweep report (see sweep.py
for the run-directory layout and guarantees)."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np
import psutil

if TYPE_CHECKING:
    from src.experiments.config import RunSpec


def run_status(run_dir: Path) -> str:
    if (run_dir / "result.json").exists():
        return "done"
    if (run_dir / "lock").exists():
        return "stale_lock" if _lock_is_stale(run_dir / "lock") else "running"
    if (run_dir / "error.txt").exists():
        return "failed"
    if (run_dir / "checkpoint.pt").exists():
        return "interrupted"
    return "pending"


def _lock_is_stale(lock: Path) -> bool:
    try:
        info = json.loads(lock.read_text())
    except (OSError, ValueError):
        return False  # being written right now
    if info.get("host") != socket.gethostname():
        return False  # cannot judge another machine's pid
    pid = int(info.get("pid", -1))
    if not psutil.pid_exists(pid):
        return True
    try:
        return psutil.Process(pid).create_time() > info.get("time", 0) + 1  # pid reused by another process
    except psutil.Error:
        return True


def try_claim(run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "lock"
    if lock.exists() and _lock_is_stale(lock):
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()}, f)
    return True


def release(run_dir: Path) -> None:
    (run_dir / "lock").unlink(missing_ok=True)


def sweep_report(runs: list[RunSpec], root: Path) -> dict:
    counts: dict[str, int] = {}
    by_status: dict[str, list[str]] = {}
    recovered = []
    durations = []
    for r in runs:
        d = root / r.run_id
        st = run_status(d)
        counts[st] = counts.get(st, 0) + 1
        by_status.setdefault(st, []).append(r.run_id)
        att = d / "attempts.jsonl"
        if att.exists():
            lines = [json.loads(x) for x in att.read_text().splitlines() if x.strip()]
            if st == "done" and (len(lines) > 1 or any(a.get("resumed_from_checkpoint") for a in lines)):
                recovered.append(r.run_id)
            durations += [a["duration_sec"] for a in lines if a.get("status") == "done"]
    return {"planned": len(runs), "counts": counts, "failed": by_status.get("failed", []),
            "interrupted": by_status.get("interrupted", []), "stale_lock": by_status.get("stale_lock", []),
            "recovered_or_rerun": recovered,
            "run_duration_sec": {"n": len(durations), "mean": float(np.mean(durations)) if durations else None,
                                 "total": float(np.sum(durations)) if durations else 0.0}}
