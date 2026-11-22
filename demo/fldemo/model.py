"""The shared HAR model (MLP 64→32→6) and its weight (de)serialization.

Every device — PyTorch on laptop devices, hand-written C++ on the ESP32 —
implements exactly this network and this tensor naming/order.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from fldemo.har import NUM_CLASSES, NUM_FEATURES

HIDDEN = 32
# Fixed tensor order — the firmware writes the NPZ in this order too.
TENSOR_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("fc1.weight", (HIDDEN, NUM_FEATURES)),
    ("fc1.bias", (HIDDEN,)),
    ("fc2.weight", (NUM_CLASSES, HIDDEN)),
    ("fc2.bias", (NUM_CLASSES,)),
)
PARAM_COUNT = sum(int(np.prod(s)) for _, s in TENSOR_SPECS)


class HarMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def new_model(seed: int = 0) -> HarMLP:
    g = torch.Generator().manual_seed(seed)
    m = HarMLP()
    with torch.no_grad():
        for layer in (m.fc1, m.fc2):
            # Same scheme as nn.Linear's default (uniform ±1/sqrt(fan_in)), but seeded.
            bound = 1.0 / np.sqrt(layer.in_features)
            layer.weight.uniform_(-bound, bound, generator=g)
            layer.bias.uniform_(-bound, bound, generator=g)
    return m


def to_numpy_state(model: nn.Module) -> "OrderedDict[str, np.ndarray]":
    sd = model.state_dict()
    return OrderedDict(
        (name, sd[name].detach().cpu().numpy().astype("<f4").copy()) for name, _ in TENSOR_SPECS
    )


def load_numpy_state(model: nn.Module, state) -> None:
    model.load_state_dict(
        {name: torch.from_numpy(np.asarray(state[name], dtype=np.float32).copy()) for name, _ in TENSOR_SPECS}
    )


@torch.no_grad()
def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    pred = model(torch.from_numpy(x)).argmax(dim=1).numpy()
    return float((pred == y).mean())


def train_sgd(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> list[float]:
    """Plain mini-batch SGD (no momentum) — the ESP32 firmware implements the same."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(len(x))
        total, n = 0.0, 0
        for i in range(0, len(x), batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
            n += len(b)
        losses.append(total / n)
    return losses
