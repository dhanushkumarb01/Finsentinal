"""FedIPW — Day 5 of the plan, and the single most important method in the
whole project: this is the existing correction whose own math (Mean
Ignorability, Corollary 6.2 in the source paper) predicts it should fail
specifically under MNAR, because participation there depends on a state
(drift) outside its observed-covariate set.

Two variants, per the spec (Part 7/11):

  - ORACLE  : reweight by 1 / p_true, where p_true is the ACTUAL generating
              participation probability (only knowable because we're the
              ones simulating it). Upper bound — isolates whether the
              ignorability violation itself causes bias, independent of any
              estimation error.
  - ESTIMATED: reweight by 1 / p_hat, where p_hat is estimated PURELY from
              each client's own observed participation history (empirical
              frequency, matching DFL-AA's own approach) — NEVER given
              access to drift D. This is the realistic version: it can
              only correct for what it can actually observe.

Under MCAR/MAR, p_hat should track p_true reasonably well (participation
doesn't depend on anything the estimator can't see). Under MNAR, p_hat is
expected to systematically mis-estimate p_true for high/low-drift clients,
since D is invisible to it — this is the predicted failure mode.
"""
from __future__ import annotations

import copy

import numpy as np

from src.mechanisms.dropout import POSITIVITY_HI, POSITIVITY_LO

# Participation propensities P(R=1) = 1 - P(miss). The dropout mechanisms clip
# P(miss) to [POSITIVITY_LO, POSITIVITY_HI], so every TRUE propensity a
# participant can have lies in this same band. FedIPW-estimated is clipped to
# the identical band, so oracle and estimated share one positivity floor and
# the max IPW weight ratio is bounded identically (0.95/0.05 = 19x) for both.
PROPENSITY_LO = 1.0 - POSITIVITY_HI  # 0.05
PROPENSITY_HI = 1.0 - POSITIVITY_LO  # 0.95


def clip_propensity(p: float) -> float:
    return float(min(max(p, PROPENSITY_LO), PROPENSITY_HI))


def ipw_aggregate(state_dicts: list[dict], sample_weights: list[float], propensities: list[float]) -> dict:
    """Inverse-probability-weighted FedAvg: each participant's update is
    weighted by (sample_count / propensity) instead of just sample_count —
    a client who was UNLIKELY to show up gets extra weight when it does,
    to compensate for similar clients who didn't.

    Self-normalized (Hajek) form: weights are renormalized over the
    participants, so a propensity shared by everyone cancels exactly (this
    is why oracle-IPW == FedAvg under MCAR).

    Propensities must be strictly positive: a silent epsilon floor used to
    hide positivity violations behind enormous weights, so they now raise.
    """
    if not (len(state_dicts) == len(sample_weights) == len(propensities)) or not state_dicts:
        raise ValueError("ipw_aggregate needs equal-length, non-empty inputs")
    if any(not (0.0 < p <= 1.0) for p in propensities):
        raise ValueError(f"propensities must lie in (0, 1], got {propensities}")
    ipw_weights = [w / p for w, p in zip(sample_weights, propensities)]
    total = sum(ipw_weights)
    avg = copy.deepcopy(state_dicts[0])
    for k in avg:
        avg[k] = sum(sd[k].float() * (w / total) for sd, w in zip(state_dicts, ipw_weights))
    return avg


class EstimatedPropensityTracker:
    """Tracks each client's empirical participation frequency across
    rounds — an exponential moving average, matching DFL-AA's own
    approach. This is the ONLY thing FedIPW-estimated is allowed to see:
    past participation record, nothing about current drift.

    The returned estimate is clipped to the same [0.05, 0.95] positivity band
    the oracle propensities live in. BUG FIX: the caller used to floor p_hat
    at 1e-3 while oracle propensities are floored at 0.05 — a client absent
    for ~20 rounds (common under MNAR) reached p_hat ~ 0.006 and got a weight
    ~170x a typical client's, so the oracle-vs-estimated comparison mixed
    "propensity estimation error" with an extra, estimator-only variance
    blow-up that the spec's positivity clip exists to rule out.
    """

    def __init__(self, decay: float = 0.8, prior: float = 0.5):
        self.decay = decay
        self.prior = prior
        self._freq: dict[int, float] = {}

    def observe_and_get(self, client_id: int, participated_this_round: bool) -> float:
        """Returns the propensity estimate to USE FOR THIS ROUND (based on
        history up to, not including, this round's outcome), then updates
        the running estimate with this round's outcome for next time."""
        p_hat = self._freq.get(client_id, self.prior)
        observed = 1.0 if participated_this_round else 0.0
        self._freq[client_id] = self.decay * p_hat + (1 - self.decay) * observed
        return clip_propensity(p_hat)

    def reset(self) -> None:
        self._freq.clear()


# ----------------------------------------------------------------------------
# Strategy wrappers (Step 8 refactor). Same arithmetic as ipw_aggregate, but
# streamed: weight_i = n_i / p_i, self-normalised over participants.
# ----------------------------------------------------------------------------
from src.methods.base import RoundContext  # noqa: E402
from src.methods.fedavg import GlobalModelStrategy  # noqa: E402


class FedIPWOracleStrategy(GlobalModelStrategy):
    """Weights by the TRUE generating propensity P(R=1) of each participant —
    the only method allowed to see it (spec Part 7)."""

    name = "fedipw_oracle"
    needs_oracle_propensity = True

    def begin_round(self, ctx: RoundContext) -> None:
        if ctx.oracle_propensity is None:
            raise ValueError("FedIPW-oracle requires ctx.oracle_propensity")
        props = {i: float(ctx.oracle_propensity[i]) for i in ctx.participants}
        if any(not (0.0 < p <= 1.0) for p in props.values()):
            raise ValueError(f"propensities must lie in (0, 1], got {props}")
        self._set_weights({i: max(ctx.n_samples[i], 1) / props[i] for i in ctx.participants})
        self._log = _propensity_log(props, self._weights)


class FedIPWEstimatedStrategy(GlobalModelStrategy):
    """Weights by p_hat from each client's own observed participation history
    (EstimatedPropensityTracker). Uses R_t for ALL clients — the server
    observes who reported — but never D."""

    name = "fedipw_estimated"

    def __init__(self, model, tracker: EstimatedPropensityTracker | None = None):
        super().__init__(model)
        self.tracker = tracker or EstimatedPropensityTracker()

    def begin_round(self, ctx: RoundContext) -> None:
        part = set(ctx.participants)
        props = {}
        for i in range(ctx.n_clients):  # every client's history advances, in index order
            p_hat = self.tracker.observe_and_get(i, i in part)
            if i in part:
                props[i] = p_hat
        self._set_weights({i: max(ctx.n_samples[i], 1) / props[i] for i in ctx.participants})
        self._log = _propensity_log(props, self._weights)

    def get_state(self) -> dict:
        return {**super().get_state(), "freq": dict(self.tracker._freq)}

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.tracker._freq = dict(state["freq"])


def _propensity_log(props: dict[int, float], weights: dict[int, float]) -> dict:
    if not props:
        return {"propensity": {}, "ipw_weight_max_ratio": None}
    w = np.array(list(weights.values()), dtype=float)
    return {
        "propensity": {int(i): round(float(p), 6) for i, p in props.items()},
        "propensity_mean": float(np.mean(list(props.values()))),
        "propensity_min": float(min(props.values())),
        "ipw_weight_max_ratio": float(w.max() / w.min()),
        "ipw_weight_ess": float(w.sum() ** 2 / (w ** 2).sum()),  # Kish effective sample size
    }
