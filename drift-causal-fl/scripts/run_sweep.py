"""Run (or resume) a study sweep defined by a YAML config.

    python scripts/run_sweep.py --config configs/mvp.yaml --workers 4 --threads 2
    python scripts/run_sweep.py --config configs/mvp.yaml --status
    python scripts/run_sweep.py --config configs/mvp.yaml --dry-run

Safe to kill and restart at any time: completed runs are skipped, claimed runs
are locked, interrupted runs resume from their checkpoint. Failed runs are not
retried unless --retry-failed is given (fix the cause first).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config import expand_runs, load_study  # noqa: E402
from src.experiments.status import release, run_status, sweep_report, try_claim  # noqa: E402  (torch-free parent)


def _claim_slot(root: Path, method: str, limits: dict[str, int]) -> Path | None | bool:
    """Concurrency cap for memory-heavy methods. Returns the claimed slot dir,
    True if the method is uncapped, or None if every slot is busy."""
    if method not in limits:
        return True
    for i in range(limits[method]):
        slot = root / "_slots" / f"{method}_{i}"
        if try_claim(slot):
            return slot
    return None


def _worker(wid: int, config: str, root: str, threads: int, only: str | None, checkpoint_every: int,
            retry_failed: bool, verbose: bool, order: str, limits: dict[str, int] | None = None) -> None:
    import os

    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    # Flush subnormal floats to zero (x86 FTZ/DAZ): avoids large CPU slowdowns on
    # subnormal arithmetic; affects only values < 1.2e-38. Applied uniformly to
    # every sweep run and recorded in each run's spec.json.
    torch.set_flush_denormal(True)
    os.environ["DCFL_FLUSH_DENORMAL"] = "1"
    from src.experiments.sweep import execute_run  # torch-heavy import only in workers
    runs = _select(expand_runs(load_study(config)), only, order)
    root_p = Path(root)
    limits = limits or {}
    while True:
        progressed, blocked = False, False
        for spec in runs:
            st = run_status(root_p / spec.run_id)
            if st == "done" or st == "running" or (st == "failed" and not retry_failed):
                continue
            slot = _claim_slot(root_p, spec.method, limits)
            if slot is None:
                blocked = True  # memory-heavy method at its concurrency cap; try other runs
                continue
            t0 = time.time()
            try:
                out = execute_run(spec, root_p, checkpoint_every=checkpoint_every, retry_failed=retry_failed, verbose=verbose)
            finally:
                if isinstance(slot, Path):
                    release(slot)
            if out in ("done", "failed"):
                progressed = True
                print(f"[w{wid}] {spec.run_id}: {out} in {time.time() - t0:.0f}s", flush=True)
                if out == "failed":
                    retry_failed = False  # one retry per worker invocation, never a loop
        if not progressed:
            if not blocked:
                return
            time.sleep(30)  # only capped runs remain; wait for a slot


def _select(runs, only, order):
    if only:
        pat = re.compile(only)
        runs = [r for r in runs if pat.search(r.run_id)]
    if order == "unique_first":
        # MCAR/MAR never read D, so their model-independent runs replicate the
        # model-relative ones exactly; execute them last (they still run and are
        # used as an end-to-end reproducibility check).
        runs = sorted(runs, key=lambda r: (r.drift_estimator == "model_independent" and r.mechanism in ("mcar", "mar")))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None, help="results root (default results/<study>)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2, help="torch intra-op threads per worker")
    ap.add_argument("--only", default=None, help="regex on run_id")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--order", choices=["config", "unique_first"], default="unique_first")
    ap.add_argument("--max-concurrent", nargs="*", default=[], metavar="METHOD=N",
                    help="cap concurrent runs of memory-heavy methods, e.g. flfdms=1 feddrift=2")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    limits = {k: int(v) for k, v in (x.split("=") for x in args.max_concurrent)}
    cfg = load_study(args.config)
    runs = _select(expand_runs(cfg), args.only, args.order)
    root = Path(args.root or Path(__file__).resolve().parents[1] / "results" / cfg["study"])
    root.mkdir(parents=True, exist_ok=True)
    if args.status or args.dry_run:
        rep = sweep_report(runs, root)
        if args.dry_run:
            rep["first_runs"] = [r.run_id for r in runs[:10]]
        print(json.dumps(rep, indent=2))
        return

    (root / "study_config.yaml").write_text(Path(args.config).read_text())
    print(f"[sweep] {len(runs)} runs, {args.workers} workers x {args.threads} threads, caps={limits} -> {root}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(w, args.config, str(root), args.threads, args.only,
                                                args.checkpoint_every, args.retry_failed, args.verbose, args.order, limits))
             for w in range(args.workers)]
    for i, p in enumerate(procs):
        p.start()
        time.sleep(2 if i == 0 else 0.5)
    for p in procs:
        p.join()
    rep = sweep_report(runs, root)
    rep["sweep_wall_clock_sec"] = round(time.time() - t0, 1)
    print(json.dumps(rep, indent=2))
    with open(root / "sweep_report.json", "w") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
