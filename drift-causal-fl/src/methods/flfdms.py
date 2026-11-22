"""FL-FDMS — "Combating Client Dropout in Federated Learning via Friend Model
Substitution" (Wang & Xu, arXiv:2205.13222v3). Step 6.

Paper method (Section 5.1), implemented exactly:

  * Each active client k in S_t uploads its local update
        Delta^k_t = w^k_{t+1} - w_t                                (Eq. 3)
  * For every pair of ACTIVE clients i, j in S_t the server computes
        r^{i,j}_t = 1/2 * ( <Delta^i_t, Delta^j_t> / (||Delta^i_t|| ||Delta^j_t||) + 1 )   (Eq. 14)
    over the full flattened parameter vector.
  * Similarity scores are a RUNNING ARITHMETIC MEAN over co-active rounds
        R^{i,j}_t = N/(N+1) * R^{i,j}_{t-1} + 1/(N+1) * r^{i,j}_t   if i, j in S_t
                  = R^{i,j}_{t-1}                                     otherwise   (Eq. 15)
    with N = N^{i,j}_{t-1}, the number of earlier rounds in which both were
    active. (Note: the project spec's "EMA score tracking" wording is
    inaccurate — the paper has no decay coefficient; Eq. 15 is an exact mean.)
  * Each dropped client k gets a friend
        phi_t(k) = argmax_{i in S_t} R^{k,i}_t
    and the friend's CURRENT-round update substitutes for k's missing one:
        Delta_t = sum_{k in S_t} a_k Delta^k_t + sum_{k notin S_t} a_k Delta^{phi_t(k)}_t   (Eq. 5)
  * w_{t+1} = w_t + Delta_t (server learning rate 1, as in our FedAvg).
  * If every client drops out, the round is skipped (paper Section 3).

Documented implementation decisions (the paper is silent or the benchmark
requires consistency) — see docs/METHODS.md:

  D1 Aggregation weights a_k. The paper writes Eq. 5 with uniform 1/K weights
     on equal-size partitions. Every method in this benchmark weights clients
     by local sample count (FedAvg), so FL-FDMS uses a_k proportional to n_k,
     where a dropped client's n_k is its most recently REPORTED sample count.
     With equal n_k this is exactly the paper's rule, and with no dropout
     FL-FDMS reduces exactly to our FedAvg (unit-tested).
  D2 Cold start / no friend. Before clients k and i have ever been co-active,
     R^{k,i} is undefined. The argmax is taken only over active clients with
     at least one co-active round (N^{k,i} >= 1). If no such client exists,
     or k has never reported (unknown n_k), k is not substituted — exactly
     the FedAvg "Dropout" treatment. Counted and logged every round
     (`n_no_friend`). The paper's Assumption 5 (friend presence) excludes this
     case; Appendix B (relaxed case) was not accessible.
  D3 Ties in the argmax are broken by lowest client index (deterministic).
  D4 A zero-norm update (client with no data) gives an undefined cosine; such
     pairs are not updated that round.
  D5 Section 5.3's optional complexity-reduction (candidate elimination with
     threshold Theta_t) is NOT used: it is an efficiency device; with K=100
     the full pairwise computation is affordable and exact.

Because a dropped client's friend is chosen from R_{t-1} restricted to S_t,
the friend map — and therefore every aggregation weight — is known before
local training starts, which lets the server aggregate in a streaming
fashion. Only the d-dimensional updates needed for this round's pairwise
similarities are held (|S_t| x d float32).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.methods.base import RoundContext, clone_state
from src.methods.fedavg import GlobalModelStrategy


class FLFDMSStrategy(GlobalModelStrategy):
    name = "flfdms"

    def __init__(self, model: nn.Module, n_clients: int):
        super().__init__(model)
        self.n = n_clients
        self.R = np.zeros((n_clients, n_clients), dtype=np.float64)   # running-mean similarity
        self.N = np.zeros((n_clients, n_clients), dtype=np.int64)     # co-active round counts
        self.last_n = np.full(n_clients, -1, dtype=np.int64)          # last reported sample count (-1 = never)
        self._base_flat: torch.Tensor | None = None
        self._deltas: torch.Tensor | None = None
        self._slot: dict[int, int] = {}
        self._friend: dict[int, int] = {}

    # ---------------------------------------------------------------- friends
    def select_friends(self, participants: list[int]) -> tuple[dict[int, int], list[int]]:
        """phi(k) for every dropped k (Eq. above + D2/D3). Uses only R_{t-1},
        N_{t-1} and the set S_t."""
        S = np.asarray(sorted(participants), dtype=np.int64)
        part = set(participants)
        friends, no_friend = {}, []
        for k in range(self.n):
            if k in part:
                continue
            if len(S) == 0 or self.last_n[k] < 0:
                no_friend.append(k)
                continue
            known = S[self.N[k, S] > 0]
            if len(known) == 0:
                no_friend.append(k)
                continue
            scores = self.R[k, known]
            friends[k] = int(known[int(np.argmax(scores))])  # argmax returns first max -> lowest index (D3)
        return friends, no_friend

    def begin_round(self, ctx: RoundContext) -> None:
        S = sorted(ctx.participants)
        for i in S:
            self.last_n[i] = int(ctx.n_samples[i])
        self._friend, no_friend = self.select_friends(S)
        weights = {i: float(max(ctx.n_samples[i], 1)) for i in S}
        for k, f in self._friend.items():
            weights[f] += float(max(self.last_n[k], 1))
        self._set_weights(weights)
        self._base_flat = torch.cat([p.detach().float().reshape(-1) for p in self.global_model.state_dict().values()])
        self._slot = {i: s for s, i in enumerate(S)}
        self._deltas = torch.empty((len(S), self._base_flat.numel()), dtype=torch.float32) if S else None
        self._log = {
            "n_dropped": self.n - len(S),
            "n_substituted": len(self._friend),
            "n_no_friend": len(no_friend),
            "friend_map": {int(k): int(f) for k, f in self._friend.items()},
        }

    def receive(self, idx: int, local_state: dict) -> None:
        flat = torch.cat([v.detach().float().reshape(-1) for v in local_state.values()])
        self._deltas[self._slot[idx]] = flat - self._base_flat
        super().receive(idx, local_state)

    def end_round(self, ctx: RoundContext) -> None:
        S = sorted(ctx.participants)
        if len(S) >= 2:
            self.update_similarity(S, self._deltas)
        super().end_round(ctx)
        self._deltas, self._base_flat = None, None
        if len(S) >= 2:
            iu = np.triu_indices(len(S), 1)
            sub = self.R[np.ix_(S, S)][iu]
            self._log["mean_R_active_pairs"] = float(sub.mean())

    def update_similarity(self, S: list[int], deltas: torch.Tensor) -> None:
        """Eq. (14) + Eq. (15) for all active pairs."""
        norms = deltas.norm(dim=1)
        gram = deltas @ deltas.T
        S_arr = np.asarray(S)
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                na, nb = float(norms[a]), float(norms[b])
                if na == 0.0 or nb == 0.0:
                    continue  # D4
                cos = float(gram[a, b]) / (na * nb)
                cos = min(1.0, max(-1.0, cos))
                r = 0.5 * (cos + 1.0)
                i, j = int(S_arr[a]), int(S_arr[b])
                N = self.N[i, j]
                val = (N / (N + 1)) * self.R[i, j] + (1.0 / (N + 1)) * r
                self.R[i, j] = self.R[j, i] = val
                self.N[i, j] = self.N[j, i] = N + 1

    # ------------------------------------------------------------ checkpoint
    def get_state(self) -> dict:
        return {"global": clone_state(self.global_model.state_dict()), "R": self.R.copy(),
                "N": self.N.copy(), "last_n": self.last_n.copy()}

    def set_state(self, state: dict) -> None:
        self.global_model.load_state_dict(state["global"])
        self.R, self.N, self.last_n = state["R"].copy(), state["N"].copy(), state["last_n"].copy()

    def final_log(self) -> dict:
        return {"R_final": np.round(self.R, 5).tolist(), "N_final": self.N.tolist()}
