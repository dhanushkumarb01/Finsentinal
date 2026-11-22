"""Shared FL logic: per-device data (incl. label-shift drift), local training, model bytes."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from fldemo.har import NUM_CLASSES, subject_arrays
from fldemo.model import TENSOR_SPECS, HarMLP, load_numpy_state, to_numpy_state, train_sgd

MODEL_BYTES = sum(int(np.prod(s)) for _, s in TENSOR_SPECS) * 4  # 9112


# ----------------------------------------------------------------------------- model bytes
def state_to_bytes(state) -> bytes:
    return b"".join(np.asarray(state[name], dtype="<f4").tobytes() for name, _ in TENSOR_SPECS)


def bytes_to_state(blob: bytes) -> "OrderedDict[str, np.ndarray]":
    if len(blob) != MODEL_BYTES:
        raise ValueError(f"model must be {MODEL_BYTES} bytes, got {len(blob)}")
    out, off = OrderedDict(), 0
    for name, shape in TENSOR_SPECS:
        n = int(np.prod(shape)) * 4
        out[name] = np.frombuffer(blob[off:off + n], dtype="<f4").reshape(shape).copy()
        off += n
    return out


def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------------- data
def device_data(device: dict, *, drifted: bool, round_number: int) -> tuple[np.ndarray, np.ndarray]:
    """The device's local dataset this round.

    Normal: its own subject's samples. Drifted (label shift, as in the research): only the
    activities in ``keep_classes`` remain, topped up with the same activities from the
    drift-source subjects, so the class mix changes sharply while the device stays honest.
    """
    x, y = subject_arrays([device["subject"]])
    if not drifted or not device.get("drift"):
        return x, y
    spec = device["drift"]
    keep = np.isin(y, spec["keep_classes"])
    xm, ym = subject_arrays(spec.get("mix_subjects", []))
    mk = np.isin(ym, spec["keep_classes"])
    rng = np.random.default_rng(1000 + round_number)
    extra = rng.permutation(np.flatnonzero(mk))[: len(y) - int(keep.sum())]
    return np.concatenate([x[keep], xm[extra]]), np.concatenate([y[keep], ym[extra]])


# ----------------------------------------------------------------------------- training
def local_update(global_state, x, y, *, epochs: int, lr: float, batch_size: int, seed: int):
    model = HarMLP()
    load_numpy_state(model, global_state)
    losses = train_sgd(model, x, y, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    return to_numpy_state(model), losses


@torch.no_grad()
def eval_state(state, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    model = HarMLP()
    load_numpy_state(model, state)
    model.eval()
    logits = model(torch.from_numpy(x))
    loss = float(nn.functional.cross_entropy(logits, torch.from_numpy(y)))
    acc = float((logits.argmax(1).numpy() == y).mean())
    return acc, loss


def poison(state, factor: float = 8.0):
    return OrderedDict((k, (np.asarray(v) * np.float32(factor)).astype("<f4")) for k, v in state.items())


def weight_delta_grid(global_state, new_state, rows: int = 8, cols: int = 16) -> list[list[float]]:
    """|Δ fc1.weight| pooled to rows×cols, normalised to 0..1 (for the 'inside the device' heatmap)."""
    d = np.abs(np.asarray(new_state["fc1.weight"]) - np.asarray(global_state["fc1.weight"]))  # 32×64
    r, c = d.shape[0] // rows, d.shape[1] // cols
    pooled = d[: rows * r, : cols * c].reshape(rows, r, cols, c).mean(axis=(1, 3))
    m = float(pooled.max()) or 1.0
    return [[round(float(v) / m, 3) for v in row] for row in pooled]


def delta_l2(a, b) -> float:
    return float(math.sqrt(sum(float(np.sum((np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)) ** 2))
                               for k, _ in TENSOR_SPECS)))


def class_counts(y: np.ndarray) -> list[int]:
    return np.bincount(y, minlength=NUM_CLASSES).astype(int).tolist()


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))
