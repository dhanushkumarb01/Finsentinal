"""Per-seed tensor cache for the FEMNIST client pool + reserve pool.

Decoding ~25k PIL images from the HuggingFace cache takes minutes; the sweep
needs the same (seed -> clients, reserve) materialisation for 70 runs per seed.
The first call builds it with the existing loaders and saves tensors to
`data_cache/`; later calls load the .pt file. The cached content is exactly what
`load_femnist_clients` / `load_femnist_reserve_pool` return (verified by a
content hash stored alongside and re-checked on load).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from src.data.loaders import ClientData, load_femnist_clients, load_femnist_reserve_pool, validate_labels

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _digest(clients: list[ClientData]) -> str:
    h = hashlib.sha256()
    for c in clients:
        h.update(str(c.client_id).encode())
        for t in (c.train_x, c.train_y, c.test_x, c.test_y):
            h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def _pack(clients):
    return [(c.client_id, c.train_x, c.train_y, c.test_x, c.test_y) for c in clients]


def _unpack(rows):
    return [ClientData(cid, tx, ty, sx, sy) for cid, tx, ty, sx, sy in rows]


def load_cached_femnist(n_clients: int, data_seed: int, n_reserve: int, reserve_seed: int,
                        cache_dir: str | Path | None = None) -> tuple[list[ClientData], list[ClientData], int, dict]:
    """Returns (clients, reserve_pool, num_classes, provenance)."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"femnist_n{n_clients}_s{data_seed}_r{n_reserve}_rs{reserve_seed}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=False)
        clients, reserve = _unpack(blob["clients"]), _unpack(blob["reserve"])
        if _digest(clients) != blob["clients_sha256"] or _digest(reserve) != blob["reserve_sha256"]:
            raise ValueError(f"data cache {path} failed its integrity check; delete it to rebuild")
    else:
        clients, num_classes, _name = load_femnist_clients(num_clients=n_clients, seed=data_seed)
        reserve = load_femnist_reserve_pool(n_reserve=n_reserve, exclude_seed=data_seed,
                                            exclude_num_clients=n_clients, seed=reserve_seed)
        blob = {"clients": _pack(clients), "reserve": _pack(reserve), "num_classes": num_classes,
                "clients_sha256": _digest(clients), "reserve_sha256": _digest(reserve),
                "dataset": "flwrlabs/femnist (writer-partitioned)"}
        tmp = path.with_suffix(".pt.tmp")
        torch.save(blob, tmp)
        os.replace(tmp, path)
    num_classes = int(blob["num_classes"])
    validate_labels(clients, num_classes, "cached clients")
    validate_labels(reserve, num_classes, "cached reserve")
    prov = {"cache_file": path.name, "clients_sha256": blob["clients_sha256"], "reserve_sha256": blob["reserve_sha256"],
            "n_train_total": int(sum(len(c.train_y) for c in clients)),
            "n_test_total": int(sum(len(c.test_y) for c in clients))}
    return clients, reserve, num_classes, prov
