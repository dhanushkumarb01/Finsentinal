"""FedDrift — "Federated Learning under Distributed Concept Drift"
(Jothimurugesan, Hsieh, Wang, Joshi, Gibbons; AISTATS 2023, arXiv:2206.00799).
Step 7. Implements the paper's Algorithm 4 ("FedDrift", with Merge) on top of
Algorithm 1 (multiple-model training), adapted to partial participation.

Paper, per time step tau (Algorithm 4):
  1. Every client c computes l_{c,m} = loss of each global model h_m on its new
     data S_c^tau.
  2. Local drift test (Eq. 4):  min_m l_{c,m}^tau > min_m l_{c,m}^{tau-1} + delta.
     A flagged client gets a NEW model and its own singleton cluster; every
     other client joins argmin_m l_{c,m}^tau.
  3. Cluster distance between existing models i, j:
        D(i,j) = max(L_ij - L_ii, L_ji - L_jj, 0),
     L_ij = loss of h_i on the data assigned to cluster j.
  4. Hierarchical clustering with COMPLETE linkage: while min_{i!=j} D(i,j) < delta,
     merge the closest pair (i, j) into h_k = data-size-weighted average of h_i
     and h_j, set D(k,l) = max(D(i,l), D(j,l)), clients of i and j join k.
     The merge tolerance re-uses delta.
  5. Algorithm 1: each model is trained by FedAvg over the clients assigned to
     it, weighted by their sample counts.
  Clients without new data keep their previous cluster identity; models with
  no clients are retained (to serve recurring concepts). Each client's
  inference model is its cluster's model.

Mapping onto this benchmark and partial participation (documented decisions,
see docs/METHODS.md; D = decision):

  F1 Time step = communication round (R = 1 round per time step). The paper
     defines a time step as "the granularity at which new data may arrive";
     in our simulator a client's local data is re-drawn from its (drifting)
     distribution every round.
  F2 Partial participation. FedDrift assumes full participation (Sec 2.1). Here
     only participants (R_{c,t}=1) evaluate losses, run the drift test, are
     re-assigned, and train. Non-participants keep their cluster identity —
     the paper's own rule for clients without new data (Sec 3, footnote 2).
     The drift test's reference value min_m l^{tau-1} is the client's own
     min-loss from its most recent participation (clients are stateful in
     FedDrift; this is client-side information about the client's own data,
     so the "no D for absent rounds" invariant is respected). A client's
     first participation only records the reference (no test).
  F3 L_ij uses the data of THIS round's participants assigned to cluster j
     (sample-weighted mean of the participants' l_{c,i}, which they already
     computed in step 1), instead of a subsample of the full history across
     all clients — the server cannot query absent clients. If cluster i or j
     has no participating data this round, D(i,j) is undefined and the pair
     is not merged this round (it can merge in a later round).
  F4 Training window = current round's data ("win-1"). All methods in the
     benchmark train on the current local data only; the official code also
     ships a `softclusterwin-1` variant.
  F5 New-model initialisation: a copy of the flagged client's current best
     model (argmin loss), which the client then trains locally this round.
     The paper does not specify the initialisation; this is what the
     official implementation does (`find_unused_model_lru` copies
     `models[best_model]`; the random re-init is commented out). New models
     join the distance computation from the next round ("added to the set of
     global models at tau+1", Algorithm 4).
  F6 Loss = mean cross-entropy, as written in the paper (the official code
     uses accuracy differences instead; we follow the paper).
  F7 Merge weights = cumulative number of training samples the model has been
     trained on (sum_{c,t} w_{c,m}^t N_c^t), as in the paper's Merge().
  F8 Model cap `max_models` (resource guard, default 10). The paper has no cap;
     the official code caps models at `concept_num`. If the cap is reached a
     flagged client cannot spawn a model and joins its argmin model; every
     occurrence is counted (`n_cap_blocked`).
  F9 Ties in argmin / closest pair are broken by lowest model id
     (deterministic; the paper breaks ties randomly — exact float ties do not
     occur in practice).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import RoundContext, Strategy, StreamingWeightedAverage, clone_state


class FedDriftStrategy(Strategy):
    name = "feddrift"
    needs_participant_data = True

    def __init__(self, model: nn.Module, n_clients: int, delta: float = 0.04, max_models: int = 10,
                 eval_batch: int = 512):
        if delta <= 0:
            raise ValueError("delta must be positive")
        if max_models < 1:
            raise ValueError("max_models must be >= 1")
        self.n = n_clients
        self.delta = float(delta)
        self.max_models = int(max_models)
        self.eval_batch = eval_batch
        self.template = model
        self.models: dict[int, nn.Module] = {0: model}       # model id -> module
        self.data_count: dict[int, float] = {0: 0.0}          # cumulative training samples
        self.assign = np.zeros(n_clients, dtype=np.int64)      # cluster identity per client
        self.prev_min_loss = np.full(n_clients, np.nan)        # client-side drift-test reference
        self.next_id = 1
        self._acc: dict[int, StreamingWeightedAverage] = {}
        self._weights: dict[int, float] = {}
        self._log: dict = {}
        self.n_models_created = 0
        self.n_merges = 0

    # ------------------------------------------------------------ utilities
    @torch.no_grad()
    def _loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
        if len(y) == 0:
            return float("nan")
        model.eval()
        total = 0.0
        for s in range(0, len(y), self.eval_batch):
            xb, yb = x[s:s + self.eval_batch].to(device), y[s:s + self.eval_batch].to(device)
            total += float(F.cross_entropy(model(xb), yb, reduction="sum"))
        return total / len(y)

    @staticmethod
    def cluster_distance(L: dict[tuple[int, int], float], i: int, j: int) -> float:
        return max(L[(i, j)] - L[(i, i)], L[(j, i)] - L[(j, j)], 0.0)

    def _merge(self, keep: int, drop: int) -> None:
        wi, wj = self.data_count[keep], self.data_count[drop]
        if wi + wj > 0:
            ai, aj = wi / (wi + wj), wj / (wi + wj)
        else:
            ai = aj = 0.5
        si, sj = self.models[keep].state_dict(), self.models[drop].state_dict()
        merged = {k: si[k].float() * ai + sj[k].float() * aj for k in si}
        self.models[keep].load_state_dict(merged)
        self.data_count[keep] = wi + wj
        self.assign[self.assign == drop] = keep
        del self.models[drop], self.data_count[drop]
        self.n_merges += 1

    # ---------------------------------------------------------------- round
    def begin_round(self, ctx: RoundContext) -> None:
        S = sorted(ctx.participants)
        existing = sorted(self.models)
        # 1. client-side losses of every existing model on the current local data
        loss = {c: {m: self._loss(self.models[m], *ctx.data[c], ctx.device) for m in existing} for c in S}

        # 2. drift test + assignment
        flagged, cap_blocked, new_models = [], [], []
        test_stat = {}  # min_m l^tau - min_m l^{tau-1} per tested participant (for delta calibration)
        for c in S:
            if ctx.n_samples[c] == 0:
                continue  # no new data: identity carried over (paper Sec 3, fn 2)
            lmin = min(loss[c].values())
            best = min(existing, key=lambda m: (loss[c][m], m))
            if not math.isnan(self.prev_min_loss[c]):
                test_stat[int(c)] = round(float(lmin - self.prev_min_loss[c]), 5)
            drift = (not math.isnan(self.prev_min_loss[c])) and lmin > self.prev_min_loss[c] + self.delta
            self.prev_min_loss[c] = lmin
            if drift:
                flagged.append(c)
                if len(self.models) < self.max_models:
                    mid = self.next_id
                    self.next_id += 1
                    self.models[mid] = copy.deepcopy(self.models[best])  # F5
                    self.data_count[mid] = 0.0
                    self.assign[c] = mid
                    new_models.append(mid)
                    self.n_models_created += 1
                    continue
                cap_blocked.append(c)
            self.assign[c] = best

        # 3. distances among pre-existing models, from this round's participant data (F3)
        cluster_members = {m: [c for c in S if self.assign[c] == m and ctx.n_samples[c] > 0] for m in existing}
        with_data = [m for m in existing if cluster_members[m]]
        L = {}
        for j in with_data:
            n_tot = sum(ctx.n_samples[c] for c in cluster_members[j])
            for i in with_data:
                L[(i, j)] = sum(ctx.n_samples[c] * loss[c][i] for c in cluster_members[j]) / n_tot
        D = {}
        for a_idx, i in enumerate(with_data):
            for j in with_data[a_idx + 1:]:
                D[(i, j)] = self.cluster_distance(L, i, j)
        dist_snapshot = {f"{i}-{j}": round(v, 5) for (i, j), v in D.items()}

        # 4. complete-linkage agglomeration while min D < delta
        merges = []
        alive = set(with_data)
        while True:
            cand = [(v, i, j) for (i, j), v in D.items() if i in alive and j in alive]
            if not cand:
                break
            v, i, j = min(cand)
            if not v < self.delta:
                break
            keep, drop = min(i, j), max(i, j)
            self._merge(keep, drop)
            merges.append([int(keep), int(drop), round(float(v), 5)])
            alive.discard(drop)
            for l in alive:
                if l == keep:
                    continue
                key_kl = (min(keep, l), max(keep, l))
                d_il = D.get((min(i, l), max(i, l)), math.inf)
                d_jl = D.get((min(j, l), max(j, l)), math.inf)
                D[key_kl] = max(d_il, d_jl)
            D = {k: v for k, v in D.items() if drop not in k}

        # 5. training plan: FedAvg within each cluster, weighted by sample count
        self._weights = {c: float(max(ctx.n_samples[c], 1)) for c in S}
        per_model: dict[int, float] = {}
        for c in S:
            per_model[int(self.assign[c])] = per_model.get(int(self.assign[c]), 0.0) + self._weights[c]
        self._acc = {m: StreamingWeightedAverage(t) for m, t in per_model.items()}
        for c in S:
            self.data_count[int(self.assign[c])] += float(ctx.n_samples[c])

        sizes = {int(m): int((self.assign == m).sum()) for m in sorted(self.models)}
        self._log = {
            "n_models": len(self.models),
            "cluster_sizes": sizes,
            "n_flagged": len(flagged),
            "flagged": [int(c) for c in flagged],
            "new_models": [int(m) for m in new_models],
            "n_cap_blocked": len(cap_blocked),
            "merges": merges,
            "distances": dist_snapshot,
            "assign": [int(a) for a in self.assign],
            "drift_test_stat": test_stat,
        }

    def start_state(self, idx: int) -> dict:
        return self.models[int(self.assign[idx])].state_dict()

    def receive(self, idx: int, local_state: dict) -> None:
        self._acc[int(self.assign[idx])].add(local_state, self._weights[idx])

    def end_round(self, ctx: RoundContext) -> None:
        for m, acc in self._acc.items():
            if acc.count > 0:
                self.models[m].load_state_dict(acc.result())
        self._acc = {}

    def eval_model(self, idx: int) -> nn.Module:
        return self.models[int(self.assign[idx])]

    def snapshot(self) -> "FedDriftSnapshot":
        return FedDriftSnapshot({m: clone_state(mod.state_dict()) for m, mod in self.models.items()},
                                self.assign.copy(), self.template)

    def round_log(self) -> dict:
        return dict(self._log)

    def final_log(self) -> dict:
        return {"n_models_final": len(self.models), "n_models_created": self.n_models_created,
                "n_merges": self.n_merges, "assign_final": [int(a) for a in self.assign]}

    # ------------------------------------------------------------ checkpoint
    def get_state(self) -> dict:
        return {"models": {m: clone_state(mod.state_dict()) for m, mod in self.models.items()},
                "data_count": dict(self.data_count), "assign": self.assign.copy(),
                "prev_min_loss": self.prev_min_loss.copy(), "next_id": self.next_id,
                "n_models_created": self.n_models_created, "n_merges": self.n_merges}

    def set_state(self, state: dict) -> None:
        self.models = {}
        for m, sd in state["models"].items():
            mod = copy.deepcopy(self.template)
            mod.load_state_dict(sd)
            self.models[int(m)] = mod
        self.data_count = {int(k): float(v) for k, v in state["data_count"].items()}
        self.assign = state["assign"].copy()
        self.prev_min_loss = state["prev_min_loss"].copy()
        self.next_id = int(state["next_id"])
        self.n_models_created = int(state["n_models_created"])
        self.n_merges = int(state["n_merges"])


class FedDriftSnapshot:
    """Frozen copy of every cluster model + the assignment, used as the
    lagged model theta_{t-1} for the model-relative drift estimator."""

    def __init__(self, states: dict, assign: np.ndarray, template: nn.Module):
        self.assign = assign
        self.modules = {}
        for m, sd in states.items():
            mod = copy.deepcopy(template)
            mod.load_state_dict(sd)
            mod.eval()
            self.modules[m] = mod

    def __call__(self, idx: int) -> nn.Module:
        return self.modules[int(self.assign[idx])]
