"""Common interface for every aggregation method (Steps 6-8).

The round loop in `simulation/mechanism_runner.py` is method-agnostic: it
injects drift, draws the participation mask R_t from a dropout mechanism, and
then hands control to a `Strategy` through four hooks:

    begin_round(ctx)                 -> server/client-side work before training
                                        (FedDrift: drift test + clustering)
    start_state(idx)                 -> the model a participant downloads
    receive(idx, local_state)        -> one participant's locally-trained model
    end_round(ctx)                   -> aggregation

and, for evaluation, `eval_model(idx)` returns the model client `idx` would use
for inference (one global model for FedAvg/FedIPW/FL-FDMS; the client's
cluster model for FedDrift).

THE INFORMATION INVARIANT (spec Part 1) is enforced by what `RoundContext`
carries, not by convention:
  * `participants` / `n_samples` / `data` contain ONLY clients with R_{i,t}=1.
    A strategy never receives a non-participant's data, sample count or update
    for the current round, and never receives any drift score D or D_true.
  * `oracle_propensity` is populated ONLY for FedIPW-oracle (its documented
    privilege: the true generating P(R=1) of each participant). It is None for
    every other method.
Tests in tests/test_methods.py assert this.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RoundContext:
    rnd: int
    n_clients: int
    participants: list[int]                      # indices with R_{i,t}=1, ascending
    n_samples: dict[int, int]                    # participant idx -> local sample count (as reported)
    data: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)  # participant idx -> current local (x, y)
    oracle_propensity: dict[int, float] | None = None  # FedIPW-oracle only
    device: str = "cpu"


class Strategy:
    """Base class. Subclasses must implement start_state/receive/end_round/eval_model."""

    name = "base"
    needs_participant_data = False  # FedDrift sets True (client-side loss evaluation)
    needs_oracle_propensity = False  # FedIPW-oracle sets True

    def begin_round(self, ctx: RoundContext) -> None:  # noqa: D401 - hook
        pass

    def start_state(self, idx: int) -> dict:
        raise NotImplementedError

    def receive(self, idx: int, local_state: dict) -> None:
        raise NotImplementedError

    def end_round(self, ctx: RoundContext) -> None:
        raise NotImplementedError

    def eval_model(self, idx: int) -> nn.Module:
        raise NotImplementedError

    # -- logging ---------------------------------------------------------------
    def round_log(self) -> dict:
        """Method-specific diagnostics for the round just finished (JSON-safe)."""
        return {}

    def final_log(self) -> dict:
        return {}

    # -- checkpointing -----------------------------------------------------------
    def get_state(self) -> dict:
        raise NotImplementedError

    def set_state(self, state: dict) -> None:
        raise NotImplementedError


class StreamingWeightedAverage:
    """Weighted average of state_dicts accumulated one at a time, so a round
    never holds all participants' models in memory (100 clients x 26 MB).

    Bit-compatibility: `fedavg_aggregate` computed
        sum(sd[k].float() * (w / total) for sd, w in ...)
    Python's sum starts at int 0 and 0 + t == t exactly, so accumulating
    acc = t_1; acc = acc + t_2; ... in the same order with the same
    pre-computed scalars (w / total) gives bit-identical tensors.
    `total` must therefore be known up front (it is: participants and their
    sample counts / propensities are known before local training starts).
    """

    def __init__(self, total_weight: float):
        if not total_weight > 0:
            raise ValueError(f"total weight must be positive, got {total_weight}")
        self.total = total_weight
        self.acc: dict | None = None
        self.count = 0

    def add(self, state: dict, weight: float) -> None:
        scale = weight / self.total
        if self.acc is None:
            self.acc = {k: v.float() * scale for k, v in state.items()}
        else:
            for k, v in state.items():
                self.acc[k] = self.acc[k] + v.float() * scale
        self.count += 1

    def result(self) -> dict:
        if self.acc is None:
            raise ValueError("no states were added")
        return self.acc


def clone_state(state: dict) -> dict:
    return {k: v.detach().clone() for k, v in state.items()}


def flatten_delta(local_state: dict, base_state: dict) -> torch.Tensor:
    """Full d-dimensional update vector w_local - w_global, all parameters
    concatenated in state_dict order (SmallCNN has no buffers)."""
    return torch.cat([(local_state[k].float() - base_state[k].float()).reshape(-1) for k in base_state])


def model_from_state(template: nn.Module, state: dict) -> nn.Module:
    m = copy.deepcopy(template)
    m.load_state_dict(state)
    return m


def rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def set_rng_state(rng: np.random.Generator, state: dict) -> None:
    rng.bit_generator.state = state
