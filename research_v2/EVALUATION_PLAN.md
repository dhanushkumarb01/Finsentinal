# Evaluation Plan (for Phases R2+)

*Pre-registration document. Written 2026-11-22, before any implementation or experiment. Thresholds are fixed here and must not be tuned after results are seen. Any deviation must be logged with its reason.*

**Cost basis**

| Scale | Measured cost | Source |
|---|---|---|
| Full MVP run (100 clients × 100 rounds) | **59,795 s ≈ 16.6 h** per run (mean of 14 finished runs; 4 workers sharing a 16 GB i5-12450H) | coordinator measurement; PROGRESS.md says "~110 s/round/worker" |
| day5-style run (25 clients × 25 rounds) | ~410–440 s | `results/day5_v3/*.json` `wall_clock_sec`, recorded with fewer concurrent workers |

The per-round cost does **not** scale linearly between these two points. The likely causes are contention, FL-FDMS memory use, and the model-relative forward pass over all data. **Measure the cost of the first pilot run before committing to a schedule.**

**Shared machine.** The MVP sweep is running or paused on the same machine. A pilot must run only while the sweep is paused, or with at most 1 worker alongside it. Never kill sweep processes.

---

## 1. Methods compared

**Baselines (existing code, unchanged):**
- FedAvg
- FedIPW-oracle (true p; an unbiased ceiling under drift-MNAR)
- FedIPW-estimated. This is an EMA of participation history, so it is really in the FedAU / DFL-AA family. Report it as **FedIPW-EMA** to avoid misattributing it to Morishita's covariate model.
- FL-FDMS
- FedDrift (δ = 0.04, max_models = 10)

**New:**
- DS-DR (lead)
- RCS (backup; only under the `+recall` mechanism; compared at matched total reporting budget)
- DTM as the DS-DR outcome model (optional, if memory allows)

**DS-DR ablations (mandatory).** Each one isolates one component.

| ID | Variant | What it tests |
|---|---|---|
| A1 | `dsdr_ipw`: shadow π̂, m ≡ 0 (Hájek) | value of the outcome model |
| A2 | `dsdr_outcome`: π̂ constant, tilted kernel imputation only | value of the propensity leg |
| A3 | `dsdr_lagmar` / PS-IPW: π̂(Z) from a logistic MLE of R on Z, no shadow | **does shadow identification beat the naive lag-MAR approximation?** |
| A4 | `dsdr_notilt`: ω ≡ 1 | value of the Lemma-2 tilt |
| A5 | `dsdr_nogate` | gate effect on variance and the negative control |
| A6 | `dsdr_oraclepi`: true p plus our m | outcome-model variance gain with the propensity error removed |
| A7 | reported S ∈ {D_indep}, {D_model} | Proposition 3(iii): selection on an unreported statistic |
| A8 | ρ_max = ∞ (no leverage cap) | stability |

## 2. Conditions

**Mechanisms**, matching `configs/mvp.yaml` (target miss rate 0.30, positivity band [0.05, 0.95]):
- MCAR
- MAR (γ = 2)
- MNAR-logistic, β ∈ {0.5, 1, 2, 4, 8}
- MNAR-threshold (exclude_frac = 0.2)
- Drift estimators driving MNAR: model_relative and model_independent.

**Drift modes**
1. `label_shift`, the existing writer substitution. **Primary for comparability with the MVP.** The current evidence suggests little accuracy headroom (GAP_ANALYSIS G5).
2. `label_swap`, a **new additive mode**:
   - Fix k = 10 of the 62 classes and draw one random permutation of them, **shared by the whole drift cohort** so that the cohort forms one coherent new concept.
   - A `magnitude` fraction of a drifting client's samples in those classes get permuted labels, using the same ramp (onset round 20, ramp 10 at MVP scale).
   - Test sets get the same transformation at the evaluated round.
   - This creates a P(Y|X) conflict between cohorts, so under-weighting the drift cohort costs it accuracy directly. It is the setting in which accuracy predictions P4 are testable.
   - D_indep is a function of x only, so it does **not** see a pure label permutation. Under `label_swap`, D_indep would carry no drift signal and the MNAR mechanism would have to run on D_model or on a label-aware statistic. **Decision:** in `label_swap`, the mechanism uses D_true⊕noise as a *declared* synthetic driver for the method-evaluation track. Alternatively, `label_swap` can be combined with the writer substitution so that x also shifts. The **combined mode is preferred**: half of the swapped samples come from the reserve writer, which gives an x-shift plus a concept conflict. Report this choice transparently.

**Recall track (RCS only):** mechanism `X+recall(q)`, with q ∈ {0.1}. Baselines are given q × (non-responders) extra *uniformly random* participants, so the total reporting budget matches.

## 3. Metrics

**Primary method metrics:**
- **M1. Participation bias of the aggregate (synthetic track):** ‖Δ̂_t − Δ̄_t‖ against the known full-participation update. Report bias, variance and RMSE over Monte-Carlo replications.
- **M2. Drift-cohort weight-share bias (FEMNIST, free and god-view):**
  - WS_t = Σ_{k∈S_t∩drift} c_k minus the full-participation share Σ_{i∈drift} w_i.
  - For FedAvg, c_k = n_k / Σ_S n. For IPW methods, the normalised weights. For DS-DR, the signed coefficients.
  - Average over rounds ≥ onset + ramp.
- **M3. Propensity recovery:** corr(π̂, p_true) among participants, and the calibration slope. Defined for FedIPW-EMA and the DS-DR variants.

**Accuracy metrics** (existing `evaluate_drift_aware`): final and last-3-eval-mean of:
- global accuracy
- drift-cohort accuracy (primary for H1a-type effects)
- stable-cohort accuracy
- worst-client accuracy
- across-client SD

**Diagnostics:** gate-open rate, λ (leverage-cap use), min/max c, signed ESS, Wald p, θ̂ trajectories, FL-FDMS same-cohort friend fraction, FedDrift model counts and flagged-in-drift-cohort counts, realised ρ, estimator fidelity corr(D, D_true).

## 4. Stages

### Stage 0: Synthetic known-truth testbed (numpy only, single thread, minutes)

**Purpose:** verify Proposition 1 numerically (bias identity), Lemma 1 (identification), and predictions P5/P6 where ground truth is exact. Write it as `research_v2`-adjacent code in the next phase, under `drift-causal-fl/analysis/` or a new `sim_synthetic/` (not this phase).

**Design**
- N = 100, d = 20. Client optima θ_i* = θ_0 + 1{i ∈ drift}·u·D_{i,t}, for a 30-client cohort.
- Quadratic local losses, and Δ_{i,t} = −η·∇L_i(θ_t) + noise, so Δ is known for **all** clients.
- Drift statistic S_{i,t}:
  - AR(1) persistence ρ_S ∈ {0.5, 0.9, 0.99} around D_{i,t}, plus measurement noise;
  - an onset ramp, as in the MVP;
  - optionally a "change-detector" statistic mimicking D_model.
- Mechanisms are identical in form to `src/mechanisms/dropout.py`, including α_t calibration and clipping.
- **Estimators:** FedAvg, HT-IPW and Hájek-IPW (oracle and EMA), FL-FDMS-style substitution (argmax similarity), lag-MAR, DS-DR and all ablations.
- 500 Monte-Carlo replications per cell, β ∈ {0, 1, 2, 4, 8}, plus threshold.

**Outputs**
- Bias, SD and RMSE of Δ̂_t per round.
- Final ‖θ_T − θ_T^{full}‖.
- Coverage of the DS-DR sandwich interval for θ̂.
- Empirical check that mean(Δ̂) − Δ̄ matches the Proposition-1 formula (relative error < 5% at 500 reps; otherwise there is a bug).

**Go / no-go**
- If P5 fails here, **stop**: DS-DR does not go to FEMNIST.
- If A3 (lag-MAR) matches DS-DR within Monte-Carlo error at ρ_S ≥ 0.9, report that honestly. In that case the shadow machinery matters only at low persistence or at onset.

### Stage 1: Cheap FEMNIST pilot (ranks methods within hours)

**Scale:** day5-style, 25 clients × 30 rounds, drift onset round 8, ramp 5, 8 drifting clients (fraction 0.3), lr 0.03, batch 10, eval every 3 rounds. Uses `run_mechanism_experiment` directly or a small `configs/pilot_dsdr.yaml`.

**Pilot P1-a (about 1 h on 4 free workers, about 4–5 h on 1 worker)**

| Factor | Levels |
|---|---|
| Conditions | MCAR, MNAR-logistic(model_independent, β = 4) |
| Methods | FedAvg, FedIPW-oracle, FedIPW-EMA, DS-DR, A1, A3 |
| Seeds | 3 |
| Drift | label_shift |

- 2 × 6 × 3 = 36 runs × ~500 s ≈ 5 worker-hours.
- **Decision rule:** continue only if DS-DR meets P1 (corr(π̂, p) ≥ 0.8) **and** the P3 gate rate (≤ 5% under MCAR). Accuracy is *not* used to rank at this scale, because of noise.

**Pilot P1-b (about 2 h on 4 workers)**
- Same as P1-a, plus MNAR-threshold, the label_swap+writer drift and FL-FDMS. 5 seeds, 4 methods (FedAvg, FedIPW-EMA, FL-FDMS, DS-DR).
- Measures the accuracy headroom: is drift-cohort accuracy under FedAvg in MNAR lower than in MCAR?
- If there is no headroom even in label_swap at pilot scale, the accuracy claims (P4) are suspended. The paper then reports aggregate-bias results (M1–M3) only.

### Stage 2: Mid-scale confirmation (only for methods that pass Stage 1)

**Scale:** 50 clients × 60 rounds, onset 15, ramp 8.

| Factor | Levels |
|---|---|
| Conditions | MCAR, MAR, MNAR β ∈ {1, 4, 8} (mind), MNAR β = 4 (mrel), MNAR-threshold (mind) — 7 |
| Methods | 5 baselines + DS-DR + A1 + A3 + A6 — 9 |
| Seeds | 6 (6 is the minimum at which an exact two-sided Wilcoxon signed-rank test can reach p < 0.05: 2/64 = 0.031) |
| Drift | label_swap+writer (accuracy track) and label_shift (comparability) |

- 7 × 9 × 6 × 2 = 756 runs. This is too many at an estimated 2–5 h per run.
- **Trimmed design:** run label_shift only for {FedAvg, FedIPW-EMA, DS-DR} × {MCAR, MNAR β = 4 mind} × 6 seeds = 36 runs, plus label_swap at full width = 378 runs.
- **Estimate:** 414 runs × ~3 h ≈ 1,240 worker-hours ≈ **13 days on 4 workers**. This is still heavy, so it can be cut further to β ∈ {4} for mid-scale: about 220 runs, ~7 days. **Decide after measuring Stage-1 per-round cost.**

### Stage 3: Full-scale integration (only the winner)

- Add DS-DR (and the best ablation) to the MVP matrix at 100 clients × 100 rounds, reusing the **existing MVP baseline runs**. The seeds are bit-identical and paired, so baselines are not rerun.
- **Minimum set:** model_independent × {MCAR, MAR, MNAR β = 2, 8} × DS-DR × 5 seeds = 20 runs × 16.6 h ≈ 332 worker-hours ≈ **3.5 days on 4 workers**.
- **Full set:** 7 conditions × 2 estimators × 5 seeds = 70 runs ≈ 12 days on 4 workers.
- MNAR-threshold needs the strong-paper-tier config.

## 5. Statistical analysis (pre-registered)

**Unit and pairing**
- The experimental unit is the **seed**. Within a seed, all methods share clients, the drift schedule and the mechanism RNG, so all comparisons are **paired by seed**.
- Per-round and per-client values are averaged to one number per (seed, cell) before testing. Rounds are never treated as independent samples.

**Primary hypotheses** (each method against each comparator; family-wise Holm correction over the listed set):

| ID | Hypothesis | Metric | Comparator | Test | Threshold |
|---|---|---|---|---|---|
| H-M1 | DS-DR reduces aggregate bias | M1 (synthetic) | FedAvg, FedIPW-EMA, lag-MAR | Monte-Carlo CI of the bias ratio | ratio ≤ 0.25 vs FedAvg, ≤ 0.5 vs lag-MAR (P5) |
| H-M2 | DS-DR reduces weight-share bias | M2 | FedAvg | paired t on seed means, plus a bootstrap CI | ≥ 50% reduction, Holm p < 0.05 (P2) |
| H-M3 | DS-DR recovers the propensity | M3 | FedIPW-EMA | paired difference in corr | ≥ +0.3 and absolute ≥ 0.8 (P1) |
| H-N1 | Negative control | global accuracy, MCAR and MAR | FedAvg | **TOST equivalence**, margin ±0.5 pp | equivalence at α = 0.05 (P3) |
| H-A1 | Drift-cohort accuracy gain (label_swap) | drift-cohort accuracy | FedAvg / FedIPW-EMA | paired t + exact Wilcoxon (n ≥ 6) | ≥ +2.0 / +1.0 pp (P4) |
| H-A2 | No global harm | global accuracy | FedAvg | one-sided non-inferiority | margin −0.5 pp |
| H-A3 | Shadow beats naive lag-MAR | M2, M3 | A3 | paired | DS-DR better with p < 0.05. **If not, report that the shadow adds nothing in this regime.** |

**Effect sizes and uncertainty**
- Report the mean paired difference with a 95% bootstrap CI (10,000 resamples over seeds; BCa), and Cohen's d_z.
- **Mixed model (secondary):** metric ~ method × mechanism + (1 | seed), reusing `analysis/` (it already handles boundary fits and the optimizer fallback).
- **Dose-response:** the slope of (DS-DR − FedAvg) against realised ρ_true across β levels. Prediction: a positive slope under logistic MNAR, and ≈ 0 under MCAR.
- **Multiplicity:** Holm across the primary hypotheses. Everything else is exploratory and labelled as such.
- **Power note.** With 5–6 seeds, only paired effects of about 1.3–1.5 SD can be detected reliably at 80% power. Accuracy differences below the seed-to-seed SD will be reported as inconclusive, not as null.

## 6. Controls and sanity checks

1. **Negative control.** MCAR and MAR must show no gain (H-N1), and the gate must stay closed (≤ 5% of rounds).
2. **Positive control.** FedIPW-oracle should reduce M2 under MNAR. If it does not, the metric or the mechanism is broken.
3. **Known-truth check.** Stage 0 Proposition-1 identity (relative error < 5%).
4. **Reproducibility.** MCAR and MAR duplicates across estimator passes must be bit-identical (existing check). DS-DR resume from checkpoint must be bit-identical (new test).
5. **Information-invariant test** extended to `reported_drift`.
6. **Boundary checks.** Under MNAR-threshold (P6) and mrel-MNAR (P7) the predicted *failures* must appear. If they do not, look for leakage.

## 7. Compute summary

| Stage | Runs | Estimated wall clock (4 free workers) | Gate |
|---|---|---|---|
| 0 synthetic | — | minutes (1 thread) | P5 |
| 1-a pilot | 36 | ~1–1.5 h | P1 + gate rate |
| 1-b pilot | ~80 | ~2–3 h | accuracy headroom |
| 2 mid-scale (trimmed) | ~220 | ~7 days | H-M2, H-A1, H-A3 |
| 3 full (minimum) | 20 | ~3.5 days | integration |

All Stage 1+ estimates assume the sweep is paused. If the sweep keeps running, use 1 worker and multiply the times by 4.
