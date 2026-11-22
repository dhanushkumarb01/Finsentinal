"""NumPy implementation of har-mlp-v1 training for virtual devices.

Same math as fldemo.model (PyTorch) and the ESP32 firmware: Linear(64,32) → ReLU →
Linear(32,6), softmax cross-entropy (mean over the batch), plain SGD, shuffling with
numpy.random.default_rng(seed).permutation per epoch. Lets six device processes run without
importing torch (~100 MB RAM each). tests/test_npmodel.py checks it against PyTorch.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


def _params(state):
    return (np.array(state["fc1.weight"], np.float32), np.array(state["fc1.bias"], np.float32),
            np.array(state["fc2.weight"], np.float32), np.array(state["fc2.bias"], np.float32))


def forward(state, x: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2 = _params(state)
    return np.maximum(x @ w1.T + b1, 0.0) @ w2.T + b2


def loss_acc(state, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = forward(state, x).astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean()), float((z.argmax(1) == y).mean())


def train_sgd(state, x: np.ndarray, y: np.ndarray, *, epochs: int, lr: float, batch_size: int, seed: int):
    w1, b1, w2, b2 = _params(state)
    rng = np.random.default_rng(seed)
    lr = np.float32(lr)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            b = perm[i:i + batch_size]
            xb, yb, n = x[b], y[b], len(b)
            a = xb @ w1.T + b1
            h = np.maximum(a, 0.0)
            z = h @ w2.T + b2
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            total += float(-np.log(p[np.arange(n), yb] + 1e-12).sum())
            dz = p
            dz[np.arange(n), yb] -= 1.0
            dz /= n
            gw2, gb2 = dz.T @ h, dz.sum(axis=0)
            dh = (dz @ w2) * (a > 0)
            gw1, gb1 = dh.T @ xb, dh.sum(axis=0)
            w1 -= lr * gw1.astype(np.float32)
            b1 -= lr * gb1.astype(np.float32)
            w2 -= lr * gw2.astype(np.float32)
            b2 -= lr * gb2.astype(np.float32)
        losses.append(total / len(x))
    new = OrderedDict([("fc1.weight", w1), ("fc1.bias", b1), ("fc2.weight", w2), ("fc2.bias", b2)])
    return new, losses
