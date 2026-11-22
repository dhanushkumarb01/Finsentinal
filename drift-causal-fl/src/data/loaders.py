"""Dataset loading + client partitioning.

Day-1 goal (per the project plan): get a *trustworthy* full-participation
FedAvg baseline running before touching any dropout mechanism. This module
tries the real target dataset first (FEMNIST, writer-partitioned, via
flwr-datasets) and transparently falls back to a fast MNIST-based smoke test
if FEMNIST isn't available yet (e.g. still downloading) — this keeps the
pipeline runnable today while the full FEMNIST pull finishes in the
background. Every run logs which dataset it actually used, so results are
never silently ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ClientData:
    client_id: int
    train_x: torch.Tensor  # (N, 1, 28, 28) float32, normalized
    train_y: torch.Tensor  # (N,) long
    test_x: torch.Tensor
    test_y: torch.Tensor


class TensorDS(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x, self.y = x, y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _to_client_split(x: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return train_idx, test_idx


FEMNIST_NUM_CLASSES = 62  # digits (10) + upper (26) + lower (26)


def _femnist_num_classes(fds) -> int:
    """Class count from the dataset's own `character` ClassLabel schema (62
    for flwrlabs/femnist), NOT from whichever labels a sampled subset of
    writers happens to contain."""
    feat = fds.load_partition(0).features["character"]
    n = int(getattr(feat, "num_classes", FEMNIST_NUM_CLASSES))
    if n != FEMNIST_NUM_CLASSES:
        raise ValueError(f"unexpected FEMNIST class count {n} (expected {FEMNIST_NUM_CLASSES})")
    return n


def validate_labels(clients: list[ClientData], num_classes: int, what: str = "clients") -> None:
    """Every train/test label must index a valid output unit of the model —
    otherwise CrossEntropyLoss raises (or, worse, silently mis-indexes on
    some backends)."""
    for c in clients:
        for split, y in (("train", c.train_y), ("test", c.test_y)):
            if len(y) and (int(y.min()) < 0 or int(y.max()) >= num_classes):
                raise ValueError(
                    f"{what}: client {c.client_id} {split} labels span [{int(y.min())}, {int(y.max())}], "
                    f"outside [0, {num_classes})"
                )


def load_femnist_clients(num_clients: int = 100, seed: int = 0) -> tuple[list[ClientData], int, str]:
    """Real FEMNIST, partitioned by writer (natural, non-synthetic non-IID),
    via flwr-datasets pulling `flwrlabs/femnist` from the Hugging Face Hub.

    Returns (clients, num_classes, dataset_name).
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
    )
    num_partitions = fds.partitioners["train"].num_partitions
    rng = np.random.default_rng(seed)
    chosen = rng.choice(num_partitions, size=min(num_clients, num_partitions), replace=False)

    clients: list[ClientData] = []
    for cid, pid in enumerate(chosen):
        part = fds.load_partition(int(pid))
        imgs = np.stack([np.array(im, dtype=np.float32) / 255.0 for im in part["image"]])
        labels = np.array(part["character"], dtype=np.int64)
        x = torch.tensor(imgs).unsqueeze(1)  # (N,1,28,28)
        y = torch.tensor(labels)
        train_idx, test_idx = _to_client_split(imgs, labels, seed=seed + cid)
        clients.append(
            ClientData(
                client_id=cid,
                train_x=x[train_idx],
                train_y=y[train_idx],
                test_x=x[test_idx],
                test_y=y[test_idx],
            )
        )
    # BUG FIX: num_classes used to be max(label seen in the sampled clients' TRAIN
    # splits) + 1. A subset of writers that never wrote the last class(es) ('y',
    # 'z') produced a model with < 62 outputs, and any test sample or drift-source
    # (reserve-pool) sample carrying such a label then crashed CrossEntropyLoss.
    num_classes = _femnist_num_classes(fds)
    validate_labels(clients, num_classes, "FEMNIST clients")
    return clients, num_classes, "FEMNIST (real, writer-partitioned)"


def load_femnist_reserve_pool(
    n_reserve: int, exclude_seed: int, exclude_num_clients: int, seed: int = 999
) -> list[ClientData]:
    """A small pool of EXTRA real FEMNIST writers, disjoint from the main
    client pool (drawn with the same seed/num_clients used to build the
    main pool, so we can compute and exclude the same writer indices).

    Purpose: drift-source material for the label-shift drift injection
    (see injection.py) — mixing in another real writer's real, undegraded
    samples, instead of corrupting the original writer's own images. This
    is what fixes the rotation confound found in Day 5 (PROGRESS.md):
    the samples used to simulate drift are always real, natural FEMNIST
    handwriting, never artificially distorted.

    Each reserve writer is split 80/20 into train/test (same splitter as the
    main clients). Drift injection into a client's TRAINING data only ever
    draws from the reserve writer's train split; the held-out test split is
    used only to build that client's drifted TEST set (see
    injection.apply_drift_to_test). The two never overlap, so drifted
    evaluation never scores the model on samples it trained on.
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
    )
    num_partitions = fds.partitioners["train"].num_partitions

    main_rng = np.random.default_rng(exclude_seed)
    excluded = set(main_rng.choice(num_partitions, size=min(exclude_num_clients, num_partitions), replace=False).tolist())

    reserve_rng = np.random.default_rng(seed)
    available = [p for p in range(num_partitions) if p not in excluded]
    chosen = reserve_rng.choice(available, size=min(n_reserve, len(available)), replace=False)

    reserve: list[ClientData] = []
    for rid, pid in enumerate(chosen):
        part = fds.load_partition(int(pid))
        imgs = np.stack([np.array(im, dtype=np.float32) / 255.0 for im in part["image"]])
        labels = np.array(part["character"], dtype=np.int64)
        x = torch.tensor(imgs).unsqueeze(1)
        y = torch.tensor(labels)
        train_idx, test_idx = _to_client_split(imgs, labels, seed=seed + rid)
        reserve.append(
            ClientData(client_id=-(rid + 1), train_x=x[train_idx], train_y=y[train_idx], test_x=x[test_idx], test_y=y[test_idx])
        )
    validate_labels(reserve, _femnist_num_classes(fds), "FEMNIST reserve pool")
    return reserve


def load_mnist_smoketest_clients(num_clients: int = 100, shards_per_client: int = 2, seed: int = 0):
    """Fast fallback: MNIST, sharded into non-IID client partitions the
    standard way (sort by label, cut into shards, assign shards to clients).
    Used ONLY as a pipeline smoke test while the real FEMNIST pull finishes —
    every result generated with this fallback is labeled as such.
    """
    from torchvision import datasets

    train_ds = datasets.MNIST(root="data_cache", train=True, download=True)
    test_ds = datasets.MNIST(root="data_cache", train=False, download=True)

    train_x = train_ds.data.numpy().astype(np.float32) / 255.0
    train_y = train_ds.targets.numpy().astype(np.int64)
    test_x = test_ds.data.numpy().astype(np.float32) / 255.0
    test_y = test_ds.targets.numpy().astype(np.int64)

    rng = np.random.default_rng(seed)
    order = np.argsort(train_y, kind="stable")
    num_shards = num_clients * shards_per_client
    shard_size = len(order) // num_shards
    shard_ids = rng.permutation(num_shards)

    clients: list[ClientData] = []
    test_order = rng.permutation(len(test_y))
    test_chunks = np.array_split(test_order, num_clients)

    for cid in range(num_clients):
        my_shards = shard_ids[cid * shards_per_client : (cid + 1) * shards_per_client]
        idxs = np.concatenate([order[s * shard_size : (s + 1) * shard_size] for s in my_shards])
        x = torch.tensor(train_x[idxs]).unsqueeze(1)
        y = torch.tensor(train_y[idxs])
        tx = torch.tensor(test_x[test_chunks[cid]]).unsqueeze(1)
        ty = torch.tensor(test_y[test_chunks[cid]])
        clients.append(ClientData(client_id=cid, train_x=x, train_y=y, test_x=tx, test_y=ty))

    return clients, 10, "MNIST (smoke-test fallback, non-IID sharded — NOT the real target dataset)"


def load_clients(num_clients: int = 100, seed: int = 0, prefer_femnist: bool = True):
    """Try real FEMNIST first; fall back to the MNIST smoke test if FEMNIST
    isn't reachable yet (still downloading, offline, etc). Always prints
    which one was actually used so results are never mislabeled.
    """
    if prefer_femnist:
        try:
            clients, num_classes, name = load_femnist_clients(num_clients, seed)
            print(f"[data] Loaded dataset: {name} ({len(clients)} clients, {num_classes} classes)")
            return clients, num_classes, name
        except ValueError:
            # Data-integrity failures (e.g. invalid labels) must never be
            # papered over by silently switching datasets.
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[data] FEMNIST not available yet ({type(e).__name__}: {e}) — falling back to MNIST smoke test.")

    clients, num_classes, name = load_mnist_smoketest_clients(num_clients, seed=seed)
    print(f"[data] Loaded dataset: {name} ({len(clients)} clients, {num_classes} classes)")
    return clients, num_classes, name
