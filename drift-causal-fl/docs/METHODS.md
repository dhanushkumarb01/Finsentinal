# Methods: implementation record (Steps 6–7)

This file records how each aggregation method is implemented, which source it
follows, and every point where the implementation had to decide something the
source leaves open or where the benchmark forces an adaptation. Code:
`src/methods/`. Tests: `tests/test_methods.py`.

All five method-variants share one round loop (`src/simulation/mechanism_runner.py`)
and one interface (`src/methods/base.py::Strategy`). Within a seed they see the
identical client pool, drift schedule, participation-mechanism RNG stream,
local-training routine (SGD, momentum 0.9, batch 10, 1 local epoch) and
evaluation sets. Only the aggregation logic differs.

## Information invariant (spec Part 1)

A strategy receives a `RoundContext` holding only the **participants** of the
round: their sample counts, their current local data (FedDrift only, for its
client-side loss evaluation) and their trained models. It never receives any
drift score (`D_model`, `D_indep`, `D_true`) or a non-participant's anything.
FedIPW-oracle additionally receives the true generating propensity of each
participant, which is its documented privilege. `test_strategies_only_see_participants`
wraps every strategy in a spy and checks this on every round.

FedIPW-estimated uses the participation indicator R_t of **all** clients. The
server observes who reported, so this is observable information.

## FedAvg (`fedavg.py`)

Sample-count-weighted average of the participants' local models; dropped
clients are skipped. Aggregation is streamed (one running sum) instead of
holding 70 state_dicts. The arithmetic is unchanged, and the result is
bit-identical to the pre-refactor `fedavg_aggregate` (regression fixture,
24/24 combinations).

## FedIPW (`fedipw.py`): unchanged from Step 5

Self-normalised (Hájek) weights `n_i / p_i` with the shared positivity band
[0.05, 0.95]. Oracle uses the true `P(R=1)`. Estimated uses each client's own
participation history (EMA, decay 0.8, prior 0.5). The Step-8 refactor only
wrapped these functions as strategies and added logging: per-participant
propensities, max weight ratio, and Kish effective sample size.

Note on the spec: Part 7 describes estimated-IPW as "fitting a propensity
model from observed (X_i, A_{i,r}) pairs". The Step-5 implementation, kept
here, uses the empirical-frequency (EMA) estimator, as in DFL-AA. This was a
Step-5 decision. It is recorded here because it is a deviation.

## FL-FDMS (`flfdms.py`): Wang & Xu, arXiv:2205.13222v3

Followed exactly:

| Paper | Implementation |
|---|---|
| Eq. 3: upload Δᵏ = w_local − w_t | `flatten_delta`, full parameter vector |
| Eq. 14: r = ½(cos(Δⁱ, Δʲ) + 1) for active pairs | `update_similarity` (one Gram matrix per round) |
| Eq. 15: running **arithmetic mean** over co-active rounds, unchanged otherwise | exact recurrence with co-activity counts N |
| φ_t(k) = argmax_{i ∈ S_t} R^{k,i} | `select_friends`, using R from earlier rounds only |
| Dropped k's slot gets the friend's **current-round** update (Eq. 5) | friend's model weight += n_k |
| All clients dropped ⇒ round skipped | yes |

The project spec (Part 7, Day 6) says "EMA score tracking". The paper has no
decay coefficient: Eq. 15 is an exact running mean. The paper is followed.

Decisions where the paper is silent or the benchmark requires consistency:

- **D1 weights.** The paper uses uniform 1/K on equal-size partitions. Here
  a_k ∝ n_k, matching FedAvg in the benchmark; a dropped client uses its last
  reported n_k. With equal n_k this is the paper's rule. With no dropout,
  FL-FDMS equals FedAvg exactly (tested).
- **D2 cold start.** Before k and i have ever been co-active, R^{k,i} is
  undefined, so the argmax runs only over active clients with N^{k,i} ≥ 1.
  If none exists, or k has never reported, k is not substituted (FedAvg
  "Dropout" treatment). The count is logged per round as `n_no_friend`.
  Appendix B (relaxed friend presence) was not accessible.
- **D3** Argmax ties go to the lowest client index.
- **D4** A zero-norm update gives an undefined cosine; that pair is skipped.
- **D5** The Section 5.3 complexity-reduction heuristic is not used. It is an
  efficiency device and the exact computation is affordable.

Diagnostics: friend map, number substituted or without a friend, mean R over
active pairs, and (from the evaluation side) `friend_same_cohort_frac`, the
fraction of substitutions where friend and dropped client are in the same
drift cohort. The last one is a god-view diagnostic and is never given to the
method.

## FedDrift (`feddrift.py`): Jothimurugesan et al., AISTATS 2023, Algorithm 4

The **paper** is the research definition. The official code (microsoft/FedDrift,
`FedAvgEnsAggregatorSoftCluster.py`) was read only to settle points the paper
leaves open. Where code and paper conflict, the paper is followed.

Followed from the paper:

- Client drift test (Eq. 4): `min_m l^τ > min_m l^{τ−1} + δ`, using the
  **loss**. The code uses accuracy; the paper uses loss.
- A flagged client gets a new model and a singleton cluster. Other clients join
  argmin-loss.
- `D(i,j) = max(L_ij − L_ii, L_ji − L_jj, 0)`.
- Complete-linkage agglomeration while `min D < δ`: merge the closest pair,
  `D(k,l) = max(D(i,l), D(j,l))`, merge tolerance re-uses δ.
- Merged model = training-data-size-weighted average of the two models.
- Per-cluster FedAvg weighted by sample count (Algorithm 1).
- Models with no clients are retained. Inference uses the client's cluster model.

Adaptations and decisions (F-codes match the module docstring):

- **F1** Time step = round (R = 1 round per time step). In this simulator a
  client's local data is re-drawn every round, which matches the paper's
  definition of a time step as "the granularity at which new data may arrive".
- **F2 partial participation** (the paper assumes full participation).
  - Only participants evaluate, test, get re-assigned and train.
  - Absent clients keep their cluster identity (paper footnote 2).
  - The drift-test reference is the client's own min-loss from its last
    participation. This is client-side state about its own data, so the "no D
    for absent rounds" invariant is respected.
  - At a client's first participation the reference is only recorded.
- **F3** L_ij comes from this round's participants assigned to cluster j:
  the sample-weighted mean of the losses they already computed. If a cluster
  has no participating data, its distances are undefined and it cannot merge
  that round. This replaces the paper's full-history subsample across all
  clients, which the server cannot query under dropout.
- **F4** Trains on the current round's data ("win-1"), like every other
  method. The official code also ships a `softclusterwin-1` variant.
- **F5 new-model initialisation.**
  - The paper leaves this open. The official code copies the client's best
    model (its random re-init is commented out), so a new model here is a
    copy of the client's argmin model.
  - The client trains that copy in the same round.
  - The new model enters distance computations from the next round ("added
    at τ+1").
- **F6** Mean cross-entropy loss, as in the paper.
- **F7** Merge weights are the cumulative training samples per model.
- **F8** `max_models` cap, default 10, as a resource guard. The paper has no
  cap; the code caps at `concept_num`. When the cap is reached, a flagged
  client joins its argmin model and the event is counted (`n_cap_blocked`).
- **F9** Ties go to the lowest model id.

Choosing δ: the paper grid-searches δ ∈ {0.02, …, 0.20} by test accuracy and
reports that a fixed δ = 0.04 changes accuracy by at most 1 pp. The value used
here and how it was chosen are in `docs/EXPERIMENTS.md` ("FedDrift δ").

Diagnostics logged every round:

- number of models and cluster sizes
- flagged clients and new models
- merges with their distances
- the full distance snapshot
- the assignment vector
- the raw drift-test statistic per tested participant

Per-cluster accuracy is derived at analysis time from per-client final
accuracy and the final assignment.

## Checkpointing (all methods)

`get_state`/`set_state` save the complete method state:

- models
- similarity matrices and counts (FL-FDMS)
- cluster bookkeeping (FedDrift)
- propensity EMA (FedIPW-estimated)

The runner checkpoint adds the NumPy and torch RNG states, both drift
trackers, the lagged model or snapshot, the MAR covariate, and all
accumulated logs. Resuming is bit-identical to an uninterrupted run, which is
tested for every method.
