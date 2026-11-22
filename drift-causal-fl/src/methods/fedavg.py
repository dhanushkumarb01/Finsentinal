"""FedAvg as a `Strategy` (the canonical floor, spec Part 7).

Participants train from the single global model; the server averages their
local models weighted by local sample count. Non-participants are simply
skipped ("Dropout" baseline in FL-FDMS terminology). Numerically identical to
`simulation.fedavg_baseline.fedavg_aggregate` (see StreamingWeightedAverage).
"""
from __future__ import annotations

import torch.nn as nn

from src.methods.base import RoundContext, Strategy, StreamingWeightedAverage, clone_state


class GlobalModelStrategy(Strategy):
    """Shared plumbing for single-global-model methods."""

    def __init__(self, model: nn.Module):
        self.global_model = model
        self._acc: StreamingWeightedAverage | None = None
        self._weights: dict[int, float] = {}
        self._log: dict = {}

    def start_state(self, idx: int) -> dict:
        return self.global_model.state_dict()

    def _set_weights(self, weights: dict[int, float]) -> None:
        """weights: participant idx -> aggregation weight, iterated in
        ascending idx order (the order participants are trained/received)."""
        self._weights = weights
        total = sum(weights[i] for i in sorted(weights))
        self._acc = StreamingWeightedAverage(total) if weights else None

    def receive(self, idx: int, local_state: dict) -> None:
        self._acc.add(local_state, self._weights[idx])

    def end_round(self, ctx: RoundContext) -> None:
        if self._acc is not None and self._acc.count > 0:
            if self._acc.count != len(self._weights):
                raise RuntimeError(f"expected {len(self._weights)} updates, got {self._acc.count}")
            self.global_model.load_state_dict(self._acc.result())
        self._acc = None

    def eval_model(self, idx: int) -> nn.Module:
        return self.global_model

    def round_log(self) -> dict:
        return dict(self._log)

    def get_state(self) -> dict:
        return {"global": clone_state(self.global_model.state_dict())}

    def set_state(self, state: dict) -> None:
        self.global_model.load_state_dict(state["global"])


class FedAvgStrategy(GlobalModelStrategy):
    name = "fedavg"

    def begin_round(self, ctx: RoundContext) -> None:
        # same weighting as the legacy loop: max(sample count, 1)
        self._set_weights({i: float(max(ctx.n_samples[i], 1)) for i in ctx.participants})
