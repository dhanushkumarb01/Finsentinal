# research_v2 — Phase R1 progress (resume file)

Last updated: 2026-11-22. Phase R1: literature review, gap analysis, candidate methods. No code was changed and no experiments were run.

## Deliverables

| File | Status |
|---|---|
| PROGRESS.md | this file |
| LITERATURE_REVIEW.md | DONE (about 100 entries; synthesis table; unverified list) |
| references.bib | DONE (verified entries only; the first names of a few authors (Canonaco, Casado, Khellaf, FACE authors) still need checking against their DOI pages) |
| GAP_ANALYSIS.md | DONE (G1–G8) |
| CANDIDATE_METHODS.md | DONE (lead DS-DR: full math, Lemmas 0–2, Props 1–3, pseudo-code, code mapping, predictions P1–P7, novelty verdict; backups RCS, DTM, SB-Agg, PS-IPW, DDU; ranking) |
| EVALUATION_PLAN.md | DONE (stages 0–3, pre-registered tests, compute) |
| PAPER_DRAFT.md | DONE (v0; Results section is a placeholder) |
| README.md | DONE |

## Facts already established (do not lose)

**Codebase**
- The strategy never sees D. It gets only participants' sample counts, models, and data (data for FedDrift only).
- FedIPW-oracle receives the true p(R=1). This makes it unbiased in expectation even under drift-MNAR, so it is the ceiling, not a straw man.
- FedIPW-estimated uses an EMA of each client's own participation history (decay 0.8) and is clipped to [0.05, 0.95].

**Mechanisms**
- MNAR-logistic: p_miss = clip(σ(α_t + β·std_t(D)), 0.05, 0.95), with α_t re-solved every round so that mean p_miss = 0.30.
- MNAR-threshold: the top 20% by D are excluded deterministically, which violates positivity.
- MAR: a fixed latent covariate X that the strategy cannot see.

**Drift**
- label_shift: writer substitution for a 30-client cohort, onset at round 20, 10-round ramp. D_true then stays at 1.

**Measured from the finished MVP traces** (s0 mrel mcar/mnar0.5 fedavg; trace.npz)
- D_indep: lag-1 autocorrelation 0.993. corr(D_t, last-reported D) = 0.992 for rounds ≥ 21. corr with D_true = 0.58.
- D_model: lag-1 autocorrelation 0.78–0.88. corr(D_t, last-reported D) = 0.44–0.58. corr with D_true ≈ 0.04.
- Conclusion: drift scores are highly persistent, so a client's last-reported drift score is a strong **shadow variable** for its current drift.

**MVP s0 results (14 finished runs)**
- Final global accuracy is about 0.81–0.82 in every condition and method.
- Drift-cohort accuracy (~0.82) is not below stable-cohort accuracy (~0.81).
- Under model-relative MNAR, participation_freq_std is about 0.042, the same as MCAR. Model-relative MNAR therefore behaves like MCAR with respect to true drift.
- Implication: the current label-shift benchmark leaves almost no accuracy headroom for any correction. A conflicting-concept (label-swap) drift mode is needed to evaluate methods.

**Cost**
- Full run: about 16.6 h (59,795 s mean).
- day5-style pilot run (25 clients × 25 rounds): about 410–440 s.

## Lead method (designed; write-up pending)

**DS-DR: Drift-Shadow Doubly Robust aggregation.**
- Each client reports a scalar, client-computable drift statistic with its update (D_indep and/or D_model).
- The server stores each client's last report, Z.
- The propensity π_θ(D) is identified by shadow-variable GMM: E[(R/π_θ(D) − 1)h(Z)] = 0, where h(Z) = (1, Z) (Miao & Tchetgen Tchetgen 2016; Wang, Shao & Kim 2014).
- Outcome model: a tilted kernel imputation, m_i = Σ_j K(Z_i, Z_j) O(D_j) Δ_j / Σ_j K O, with O = (1 − π)/π, computed leave-one-out.
- Aggregate: Σ_i w_i [m_i + R_i/π̂_i (Δ_i − m_i)]. This collapses to an affine combination Σ_k c_k Δ_k of participants' updates with scalar coefficients known before training (Σ c = 1), so it streams with O(1) model memory.
- Exact bias identity: Bias = Σ_i w_i (π_i/π̂_i − 1)(Δ_i − E m_i). This requires π̂ fitted on past rounds and m_i computed leave-one-out.
- Selection gate: a Wald test that the slope is 0. If it is not rejected, the method uses exact FedAvg weights (negative control).

**Backups**
- DTM: drift-tilted memory "ghost" updates.
- SB-Agg: sensitivity-bounded (marginal-sensitivity) minimax aggregation.
- RCS: recall-subsample (Hansen–Hurwitz double sampling) design.
- PS-IPW: post-stratified IPW, kept as an incremental ablation.

## Next (Phase R1 complete)
Phase R1 complete. Next phase: Stage 0 synthetic testbed, then implement DS-DR (additive, in a worktree), then Pilot P1-a. See EVALUATION_PLAN.md.
