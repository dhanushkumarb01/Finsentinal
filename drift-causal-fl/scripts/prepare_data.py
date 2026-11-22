"""Materialise the per-seed FEMNIST tensor caches for a study before launching
workers (so parallel workers never decode the same seed concurrently).

    python scripts/prepare_data.py --config configs/mvp.yaml [--seeds 0 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import load_cached_femnist  # noqa: E402
from src.experiments.config import load_study  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_study(args.config)
    d = cfg["data"]
    for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
        t0 = time.time()
        clients, reserve, k, prov = load_cached_femnist(d["n_clients"], seed, d["n_reserve"], seed + d["reserve_seed_offset"])
        print(json.dumps({"seed": seed, "sec": round(time.time() - t0, 1), "n_clients": len(clients),
                          "n_reserve": len(reserve), "num_classes": k, **prov}), flush=True)


if __name__ == "__main__":
    main()
