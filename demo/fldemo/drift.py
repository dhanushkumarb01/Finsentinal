"""Model-independent drift estimate on the device (mirrors the research's estimator #2).

drift-causal-fl/src/drift/model_independent.py: a FROZEN random feature extractor embeds the
client's current data; D_indep = squared L2 distance between the current mean embedding and
the client's first-seen reference mean. Here the extractor is a seeded random ReLU projection
of the 64 HAR features (the research uses a small frozen random CNN for images).
"""

from __future__ import annotations

import numpy as np

_EMB_DIM = 32
_rng = np.random.default_rng(1234)
_W = (_rng.standard_normal((64, _EMB_DIM)) / np.sqrt(64)).astype(np.float32)
_B = (_rng.standard_normal(_EMB_DIM) * 0.1).astype(np.float32)


def drift_embedding_mean(x: np.ndarray) -> np.ndarray:
    return np.maximum(x @ _W + _B, 0.0).mean(axis=0).astype(np.float64)


class DriftTracker:
    def __init__(self) -> None:
        self.reference: np.ndarray | None = None

    def score(self, x: np.ndarray) -> float:
        m = drift_embedding_mean(x)
        if self.reference is None:
            self.reference = m
            return 0.0
        return float(np.sum((m - self.reference) ** 2))
