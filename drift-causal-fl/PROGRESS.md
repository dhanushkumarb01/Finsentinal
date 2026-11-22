# PROGRESS — Drift-Causal Dropout Pipeline

*Single source of truth for project status. Read this first; update it every session.*

**Last updated:** 2026-11-19 (Steps 6–8 implemented and tested; smoke sweep + calibration pilot running; Step 9 sweep pending launch)
**Working directory:** `drift-causal-fl/` (Windows 11, `.venv`, Python 3.11.9, torch 2.9.1+cpu)
**Parent research docs (repo root):** `research_discovery.md`, `full_research_report.md`, `candidate1_adversarial_audit.md`, `candidate1_final_hostile_validation.md`, `candidate1_project_specification.md` — locked.
**Implementation records:** `docs/METHODS.md` (FL-FDMS, FedDrift — every decision/deviation), `docs/EXPERIMENTS.md` (config, logging, checkpointing, commands, resource measurements).

---

## Project identity (locked, do not re-litigate)

- **Title:** *When Absence Is Informative: Benchmarking Federated Learning Under Drift-Causal Client Participation*
- **Research question:** Does client dropout causally driven by a device's own current drift produce systematically different FL outcomes than exogenous dropout of equal marginal rate — and does FedIPW remain valid under this regime?
- Must-not-claim list, hypotheses, falsification criteria: `candidate1_project_specification.md`.

---

## Status checklist

| Step | Task | Status | Evidence |
|---|---|---|---|
| 1 | Full-participation FedAvg baseline | ✅ verified | Day-1 v2 run (below) |
| 2 | Drift injection + both estimators | ✅ verified | label-shift drift; corr(D_indep, D_true)≈0.75–0.80, corr(D_model, D_true)≈−0.06 at pilot scale |
| 3–4 | MCAR / MAR / MNAR-logistic / MNAR-threshold, α calibration on the clipped probability | ✅ verified | `tests/test_steps1_5.py` (74 tests) |
| 5 | FedIPW oracle + estimated, drift-aware evaluation | ✅ verified | Day-5 v3 sweeps COMPLETE (24/24 runs, `results/day5_v3/`) — see "Steps 1–5 verification" |
| 6 | FL-FDMS | ✅ implemented + unit/integration tested | `src/methods/flfdms.py`, `docs/METHODS.md` |
| 7 | FedDrift (multi-model) | ✅ implemented + unit/integration tested | `src/methods/feddrift.py`, `docs/METHODS.md` |
| 8 | Experiment infrastructure | ✅ implemented + tested | `src/experiments/`, `configs/`, `scripts/run_sweep.py`; `docs/EXPERIMENTS.md` |
| — | Smoke sweep (spec Day 7) | ✅ 100/100 done, 0 failed; analysis pipeline validated end to end | `results/smoke/`, `results/smoke/analysis/` |
| — | Calibration pilot (lr stability, FedDrift δ) | ✅ done — δ = 0.04, lr 0.03 stable (final acc 0.825) | `results/pilot_calibration/delta_calibration.json` |
| 9 | 350-run MVP sweep | ⏳ launched 2026-11-20 (4 workers × 1 thread, flfdms cap 1); measured ~110 s/round/worker → ~3 h/run → ETA ~11 days | `results/mvp/`, log `results/_logs/mvp_sweep.log` |
| 10 | Statistical analysis | pipeline built + unit-tested; awaits sweep data | `analysis/`, `scripts/run_analysis.py` |

---

## Steps 1–5 verification (2026-11-19)

- Full test suite at session start: **74/74 passed** (`tests/test_steps1_5.py`).
- Day-5 v3 sweeps (25 clients, 25 rounds, seed 0, β=4, both estimators, MCAR/MAR/MNAR/MNAR-threshold × FedAvg/oracle/estimated) **completed: 24/24 JSONs + 2 summaries** in `results/day5_v3/`. The `.log` files are empty (stdout was not flushed when redirected); the JSONs are complete.
- MCAR and MAR results are **bit-identical** between the two estimator passes (expected: neither mechanism reads D) — direct evidence the simulator is deterministic.
- Oracle-IPW == FedAvg exactly under MCAR (0.6113) and under MNAR-threshold (constant propensity p_base among participants) — correctness checks hold.
- corr(D_model, D_true) = −0.065, corr(D_indep, D_true) = 0.797: the model-relative estimator carries essentially no drift signal at this scale. Under model-relative MNAR, dropout is therefore informative about D_model but barely about true drift. This is a property of the estimator, to be re-measured at MVP scale (it is logged in every run).
- No Step 1–5 blocker for Steps 6–10 was found. `PROGRESS.md` had not been updated for the Step-4/5 fixes (drift-aware evaluation, clipped-α calibration, shared positivity band, num_classes fix, held-out drift-source test split); those are documented in the module docstrings and are now reflected here.

## FedDrift δ calibration (held-out seed 100, 2026-11-20)

The pilot ran FedDrift with δ = 1e6, which is exactly FedAvg. It used the MVP
operating point: 100 clients, 100 rounds, MCAR, model-relative estimator. It
took 35,758 s, sharing the CPU with the smoke sweep.

- **lr stability:** accuracy rises monotonically, without divergence: 0.154
  (round 10) → 0.665 (20) → 0.754 (30) → 0.797 (50) → 0.825 (100). Drift-cohort
  accuracy is 0.826 at the end. lr = 0.03 is kept.
- **δ:** the drift-test statistic min_m l^t − min_m l^{t_prev} was scored on the
  paper's grid {0.02, …, 0.20}:
  - Positives: 311 drift-cohort client-rounds in rounds 20–34.
  - Negatives: 3,947 stable client-rounds from round 20 on.
  - Maximum Youden J is **0.149 at δ = 0.04** (TPR 0.244, FPR 0.096), so
    **δ = 0.04** is selected by the pre-stated rule.
- **Caveat (reported, not tuned away):** the paper's loss-jump test is a weak
  detector under gradual label-shift drift with a 10-round ramp.
  - Median statistic: −0.073 in the drift window and −0.012 on stable clients.
    90th percentile: 0.116 in the drift window and 0.038 on stable clients.
  - At δ = 0.04 this means about 4.7 spurious flags per round. The
    `max_models = 10` cap therefore binds, and merging (also at δ) controls
    the model count.
  - This is a property of FedDrift in this regime, and it is logged per run
    (`feddrift_*` diagnostics, `n_cap_blocked`).

## Smoke sweep result (2026-11-20)

Setup: 20 clients, 15 rounds, 5 conditions × 5 methods × 2 seeds × 2
estimators.

- **Runs:** 100/100 completed, 0 failed, 0 interrupted. Mean run time 1,541 s;
  wall clock 10.75 h, shared with the calibration pilot.
- **Reproducibility:** all 20 MCAR/MAR duplicate cells are bit-identical
  across the two estimator passes (max |diff| = 0).
- **Analysis pipeline:** it ran end to end and produced 14 tables, 17 figures
  and `results.json`.
- **Fixes found by the smoke analysis:**
  1. *Mixed-model failures.* With 2 seeds the seed-variance component sits at
     its boundary (0). L-BFGS then hit a singular Hessian, and all 10 mixed
     models failed. Now each optimizer is tried in turn (lbfgs → powell → nm
     → bfgs), the optimizer used and any boundary fit are recorded, and a
     reported OLS-with-seed-fixed-effects fallback runs only if every
     optimizer fails. A likelihood-ratio test is computed only between models
     of the same family. A regression test was added; the mixed models now
     fit 10/10.
  2. *Hard-coded figure label.* Figures said "round 100"; they now read the
     final round from the data (new `final_round` column).
  3. *FedDrift figure layout.* The clipped y-label and the legend overlapping
     the bars were fixed.
- **Not a bug:** `*_lastk` (mean of the last 3 eval points) looks low in the
  smoke data because it averages rounds 5, 10 and 15. In the MVP it covers
  rounds 90, 95 and 100.
- **Smoke numbers are not evidence:** n = 2 seeds and 15 rounds, so the
  likelihood-ratio tests are anti-conservative. FedDrift reached about 4–8
  models within 15 rounds at δ = 0.04, consistent with the calibration's
  false-positive rate.
- **Tests:** full suite 172/172 passed (74 + 66 + 21 + 11); `ruff` clean.

## Steps 6–8 summary (details: docs/METHODS.md, docs/EXPERIMENTS.md)

- **Strategy refactor:** the round loop is method-agnostic; FedAvg/FedIPW are bit-identical to the pre-refactor loop (24/24 fixture combinations).
- **FL-FDMS:** paper Eqs. 3/5/14/15 exactly (running arithmetic mean — the spec's "EMA" wording is inaccurate); documented decisions D1–D5.
- **FedDrift:** paper Algorithm 4 (loss-based drift test, singleton clusters, complete-linkage merging with δ, size-weighted merge, per-cluster FedAvg), adapted to partial participation (F1–F9). Official code consulted only where the paper is silent.
- **Infrastructure:** YAML configs with validation; one directory per run (spec, incremental JSONL metrics, full per-round trace, result, attempts, error); checkpoint every 10 rounds with bit-identical resume; O_EXCL locks with stale-lock reclaim; failures recorded and never silently retried.
- **Tests:** 74 (Steps 1–5) + 66 (methods) + 21 (infrastructure) + 11 (analysis) = 172, all passing. `ruff` (F, E9) clean.
- **Performance:** `SGD(foreach=True)` — bit-identical, ~32% faster. Flush-to-zero for subnormals in sweep workers. Memory (not CPU) is the binding constraint on this shared 16 GB machine.

---

## Results so far

### Day 1, run v1 (`day1_femnist.json`) — too small, superseded
15 clients, 15 rounds, lr=0.01, batch=20. Accuracy stayed flat at ~4-5% (barely above random for 62 classes). **Diagnosis: not a bug — just too little data (2,826 train samples total) and too few rounds to learn 62-way classification.** Superseded by v2, kept only for reference.

### Day 1, run v2 (`day1_femnist_v2.json`) — ✅ the real proof-of-concept
40 clients, 40 rounds, local_epochs=2, lr=0.05, batch_size=10, eval_every=4. **7,153 train samples, 1,770 test samples, real writer-partitioned FEMNIST, 62 classes.**

| Round | Accuracy |
|---|---|
| 1 | 5.5% |
| 4 | 5.5% |
| 8 | 36.3% |
| 12 | 50.9% |
| 16 | 62.0% |
| 20 | 67.1% |
| 24 | 65.6% |
| 28 | 71.0% |
| 32 | **71.9%** |
| 36 | 72.8% (peak) |
| 40 | 56.2% ⚠️ |

**Verdict: the harness is proven correct** — accuracy climbs from near-random to ~72% peak, exactly the shape a real FedAvg run should have. **Open issue to investigate before trusting this fully:** the round-40 drop to 56% needs explanation. Leading hypotheses, cheapest-to-check first:
1. lr=0.05 may be too high for late-stage fine-tuning → try lr decay or a lower fixed lr (0.01–0.02) and rerun.
2. Small, non-IID per-client test sets → noisy accuracy estimate late in training; consider averaging last-k rounds or using a larger held-out eval set.
3. Real overfitting/divergence — less likely given the shape, but not ruled out.

**Do not treat round 40's number as the "final" baseline accuracy until this is resolved.** Use round 32-36 (~72%) as the working reference point for now.

### Day 2, drift estimator validation (`day2_drift_check.log`) — ✅ done, one estimator works well, one needs tuning

Setup: 15 FEMNIST clients, 6 of them scripted to start drifting (rotation, ramping up) at round 3, 10 rounds total. Compared both estimators' output against the known ground-truth injected drift (`D_true`).

| Estimator | Correlation with D_true | Verdict |
|---|---|---|
| **Model-independent** (frozen feature extractor + embedding distance) | **0.995** | ✅ Works essentially perfectly — clearly separates drifting (mean D=0.078) from stable (mean D=0.000) clients |
| **Model-relative** (loss under lagged global model − running EMA) | **0.020** | ⚠️ Not detecting drift reliably at this scale — essentially noise |

**Why the model-relative one is noisy (not a bug, a scale issue):** at only 15 clients / 10 rounds / small per-client batches, loss estimates are high-variance, and the EMA hasn't had enough rounds to stabilize. This is the exact risk the project spec flagged for this estimator (Part 3) — it's *supposed to* be compared against the model-independent one for exactly this reason. **Don't discard it** — retest at MVP scale (40 clients, more rounds) before deciding whether it needs a longer EMA warm-up, a lower decay rate, or is fundamentally too noisy to use as designed.

### Day 3/4, mechanism validation against real D (`day3_mechanism_check.log`) — ✅ works cleanly

40 FEMNIST clients, target dropout rate 30%. Result: **all mechanisms land near the 30% target** (MCAR 36.5%, MAR 30.5%, MNAR 29.5–33.0% across β=0–8), and **realized ρ(D, dropout) climbs monotonically with β**: 0.07 → 0.29 → 0.48 → 0.66 → 0.70 → 0.70 (saturating at high β — expected, a sigmoid can only push so hard). Threshold mechanism hit ρ=0.75, even stronger than logistic, as expected. **The core benchmark design (matched rate, controllable informativeness) works correctly on real data.**

### First real MCAR vs MAR vs MNAR head-to-head (`mvp_comparison.log`, `mvp_{mcar,mar,mnar}.json`) — ✅ ran, produced a genuinely interesting result

Single seed, reduced scale (25 clients, 25 rounds, target dropout rate 30%, drift = rotation on 10/25 clients starting round 2). **This is the actual assembled experiment for the first time** — drift injection + dropout mechanism + FedAvg all running together in `mechanism_runner.py`.

| Mechanism | Final accuracy (round 25) | Realized dropout rate | Realized ρ |
|---|---|---|---|
| MCAR | 52.1% | 0.360 | +0.24 (noise) |
| MAR | 50.7% | 0.280 | −0.14 (noise) |
| **MNAR** | **59.0%** | 0.360 | **+0.75 (real, by design)** |

**Important, honest finding: MNAR did *better* than both controls in this run, not worse.** This is not a bug — it's exactly the scenario the hostile-validation process (Part 7) flagged as a real possibility: our synthetic drift is a rotation, which makes a drifted client's data genuinely harder/noisier for the 62-way classification task. MNAR systematically *excludes* those noisier clients more often than MCAR/MAR do — so here it may be acting like an accidental denoising filter rather than a harmful blind spot. **This is a legitimate first data point for H1b (the genuinely open question — does informative dropout help, hurt, or do nothing to aggregate accuracy), not a confirmed result.** One run, one seed, reduced scale — needs the full multi-seed sweep (and a check with a *non-corrupting* drift definition, e.g. a label-distribution shift instead of image rotation, so "drift" isn't confounded with "objectively worse data") before treating this as a real finding either way.

**Action item before the full sweep:** reconsider whether rotation is the right drift-injection mechanism, since it conflates "different distribution" with "objectively corrupted/harder data" — a label-shift or class-imbalance-based drift injection would more cleanly test the intended question (informativeness of *absence*, not quality of the missing data itself). **Confirmed necessary by the Day 5 result below — do this before the full sweep.**

### Day 5, the actual theoretical test (`day5_fedipw_check.log`, `day5_*.json`) — ✅ ran, and it caught a real design flaw before the full sweep

Built `src/methods/fedipw.py` (`ipw_aggregate`, `EstimatedPropensityTracker` — the latter tracks ONLY each client's own participation history via EMA, never given access to D, exactly per the spec's no-god-view-leak rule). Ran FedAvg / FedIPW-oracle / FedIPW-estimated crossed with MCAR / MNAR (single seed, 25 clients, 25 rounds, same reduced-scale pilot as the MVP comparison).

| | FedAvg | FedIPW-oracle | FedIPW-estimated |
|---|---|---|---|
| **MCAR** | 0.5208 | **0.5208 (exactly equal)** | 0.5199 |
| **MNAR** | 0.5900 | 0.4949 (−0.095) | 0.4746 (−0.115) |

**Two real findings:**

1. **Correctness check passed:** under MCAR, FedIPW-oracle equals FedAvg *exactly*. This isn't a null result — under MCAR every client shares the same participation probability, so the IPW weight (1/p) is the same constant for everyone and mathematically collapses back to a plain average. Confirms `ipw_aggregate` is implemented correctly.

2. **Under MNAR, both FedIPW variants did *worse* than FedAvg — the opposite of the naive expectation, and directly explained by the same confound flagged above.** Since our rotation-based drift injection makes drifted clients' data objectively harder to classify (not just differently distributed), MNAR dropout is accidentally filtering out noisy data — and FedIPW's entire mechanism is to *undo* that filtering by up-weighting the rare drifted clients that do participate. Here, undoing the filtering hurts, because the filtered-out data really was worse for this task.

**This is genuinely useful, not a failure.** A pilot run at reduced scale caught a real experimental-design flaw — conflating "informativeness of dropout" with "data quality of the dropped client" — before it was baked into an expensive 350-run sweep. **This must be fixed before the full sweep or any of its results would be uninterpretable** (we wouldn't know whether an effect is about informativeness or about accidentally filtering bad data).

---

### The confound fix — ✅ implemented

Replaced the drift mechanism. `DriftSchedule` now has a `mode` field:
- **`"label_shift"` (new default):** a drifting client's data is progressively replaced with real, *undegraded* samples from a different real FEMNIST writer (drawn from a new `reserve_pool`, loaded via `load_femnist_reserve_pool()` in `loaders.py` — extra writer partitions disjoint from the main client set). No pixel is ever corrupted; only the client's local label/style distribution shifts. This is the fix.
- **`"rotation"` (legacy):** the original, confounded mechanism — kept in `injection.py` for reference only, not used going forward.

New pieces: `load_femnist_reserve_pool()` (data/loaders.py), `apply_label_shift_drift()` + `apply_drift()` dispatcher (drift/injection.py). All scripts (`run_day2_drift_check.py`, `run_day3_mechanism_check.py`, `run_mvp_comparison.py`, `run_day5_fedipw_check.py`) and `mechanism_runner.py` updated to load a reserve pool and use `mode="label_shift"`.

**Re-validated Day 2 under the fix:** model-independent estimator still detects drift cleanly — correlation with ground truth **0.752** (down from 0.995 under rotation, which makes sense: swapping in another real writer's samples is a gentler, more realistic shift than a dramatic 35° rotation, so the frozen-feature embedding distance is naturally smaller — but still clearly separates drifting from stable clients). Model-relative estimator still noisy (0.065), consistent with before — confirms that's a property of the estimator/scale, not an artifact of the old drift injection.

### Day 5 re-run under the fix (`day5_fedipw_check_v2_labelshift.log`) — ✅ confound confirmed fixed, cleaner signal emerging

| | FedAvg | FedIPW-oracle | FedIPW-estimated |
|---|---|---|---|
| **MCAR** | 0.6270 | **0.6270 (exactly equal)** | 0.5956 |
| **MNAR** | 0.6205 | 0.5873 (−0.033) | 0.6187 (−0.002) |

**Key confirmation: the old MNAR-beats-MCAR gap is gone.** FedAvg's accuracy under MCAR (0.627) and MNAR (0.621) are now nearly identical (before the fix: 0.52 vs 0.59, a huge artificial gap). This is strong evidence the earlier "MNAR does better" result really was the rotation/data-quality confound, not a genuine informativeness effect — exactly what fixing the drift mechanism should produce.

**A cleaner, theory-consistent (but still single-seed, NOT yet confirmed) signal is emerging:** oracle-IPW now costs accuracy specifically under MNAR (−3.3pp) while costing nothing under MCAR (still exactly equal — the correctness check holds up). That's the direction the project's central hypothesis predicts. Estimated-IPW doesn't show a clean version of this pattern yet (−3.1pp under MCAR, essentially 0 under MNAR — roughly flat rather than differential); this could be genuine (the EMA-based estimator has its own always-present lag/estimation-error regardless of mechanism) or could be single-seed noise. **Honest status: encouraging, not proven — this is exactly what the full multi-seed sweep exists to resolve, not a conclusion to report yet.**

---

---

## Exact next step

1. ~~Calibration pilot~~ done. ~~Smoke sweep + analysis validation~~ done.
2. MVP sweep **running**. Check progress with `python scripts/run_sweep.py --config configs/mvp.yaml --status`. If it is interrupted, rerun the same launch command (`--workers 4 --threads 1 --max-concurrent flfdms=1`); it resumes.
3. `python scripts/run_analysis.py --study mvp` once complete.

---

## How to rerun what already works (legacy day scripts)

```bash
cd drift-causal-fl

# Day-1 baseline, real FEMNIST, tuned settings (reproduces the ~72% result)
python3 scripts/run_day1_baseline.py --clients 40 --rounds 40 --local-epochs 2 --lr 0.05 --batch-size 10 --eval-every 4 --out results/day1_femnist_v3.json

# Quick smoke test (faster, smaller, for iterating on code changes)
python3 scripts/run_day1_baseline.py --clients 15 --rounds 15 --out results/smoke.json

# Day-2 drift estimator validation (reproduces corr=0.995 model-independent, corr=0.02 model-relative)
python3 scripts/run_day2_drift_check.py

# Day-3/4 mechanism validation against real drift scores (rate-matching + rho-vs-beta)
python3 scripts/run_day3_mechanism_check.py

# First real MCAR/MAR/MNAR head-to-head comparison (single seed, reduced scale, ~4-5 min)
python3 scripts/run_mvp_comparison.py
```

Runs are slow-ish on CPU (the v2 run took a while, with the last few rounds oddly slow — possibly machine sleep/throttling during a long background run; re-check wall-clock behavior on the next run and don't assume the same timing).
