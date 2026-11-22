"""Deterministic model state used for cross-implementation test vectors."""

import numpy as np

from fldemo.model import TENSOR_SPECS


def deterministic_state() -> dict[str, np.ndarray]:
    """Exactly representable float32 values (i mod 97 - 48) / 64 over the flattened tensors."""
    state, k = {}, 0
    for name, shape in TENSOR_SPECS:
        n = int(np.prod(shape))
        vals = ((np.arange(k, k + n) % 97) - 48).astype(np.float32) / np.float32(64.0)
        state[name] = vals.reshape(shape)
        k += n
    return state
