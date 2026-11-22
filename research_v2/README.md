# research_v2: Phase R1 (literature, gaps, candidate methods)

*2026-11-22.* This phase changed no code, ran no experiments, and made no commits. Its only computation was reading `trace.npz` files from runs that had already finished.

## Files

| File | Contents |
|---|---|
| `LITERATURE_REVIEW.md` | About 100 verified works in 4 themes, a synthesis table, and a list of what could not be verified |
| `references.bib` | Verified entries only |
| `GAP_ANALYSIS.md` | Gaps G1–G8, each with literature evidence, repo evidence, and a falsifiable test |
| `CANDIDATE_METHODS.md` | Lead method DS-DR in full (math, proofs, pseudo-code, code mapping, predictions, novelty verdict), five alternatives, and a ranking |
| `EVALUATION_PLAN.md` | Pre-registered staged plan: synthetic, then pilot, then mid-scale, then full; tests, thresholds, compute |
| `PAPER_DRAFT.md` | NeurIPS-style draft. The Results section is a placeholder. |
| `PROGRESS.md` | Resume file |

## Summary

**The gap.**
- Every FL method that compensates for partial participation assumes one of these:
  - participation is independent of the client's data (memory, variance reduction, Markov availability, FedAU/FedAWE);
  - participation is ignorable given system covariates (FedIPW);
  - an absent client's stale contribution still represents it (MIFA, FedVARP, FL-FDMS).
- FLOSS is the only FL treatment of MNAR participation. It uses a static shadow covariate and IPW only.
- No verified work:
  - identifies a drift-dependent participation propensity;
  - aggregates model updates in a doubly robust way under non-random participation;
  - checks memory-based compensation against absence that is caused by drift.

**Key structural fact, measured in this repo.** Drift scores persist over rounds. For D_indep, the lag-1 autocorrelation is 0.993, and the correlation between a client's current score and its last reported score is 0.992. A client's stale drift report is therefore a valid shadow variable for its current drift, and that makes the propensity identifiable.

## Ranked recommendation

1. **DS-DR (Drift-Shadow Doubly Robust aggregation). Lead.**
   - **How it works:** propensity identified by temporal shadow GMM; an outcome model built from selection-tilted kernel imputation over current participants' updates; an AIPW aggregate.
   - **Implementation cost:** it reduces to an affine streaming rule, so it needs no extra model memory.
   - **Theory:** an exact bias identity, a product-rate bound, an identification lemma, and explicit failure regimes.
   - **Novelty:** *novel as an FL method, incremental as statistics.*
2. **RCS (recall-subsample, Hansen–Hurwitz double sampling).**
   - Unbiased by design, even when positivity is violated.
   - Changes the participation model, so it needs a recall lever.
   - Also serves to validate DS-DR's estimated propensity.
   - **Novelty:** *novel in FL, classical in statistics.*
3. **DTM (drift-tilted memory "ghost" updates).**
   - A difference-in-differences correction of stale updates, used as DS-DR's outcome model.
   - Memory-heavy.
   - **Novelty:** *incremental.*
4. SB-Agg: an analysis tool.
5. PS-IPW: a mandatory ablation, *already exists in spirit*.
6. DDU: future work.

**Implement first:** DS-DR together with ablations A1–A6, then RCS. Start with Stage 0 (the synthetic known-truth testbed, which takes minutes) and Pilot P1-a (25 clients × 30 rounds, about 1 h on 4 free workers). Each stage has a go/no-go gate. Do not start any FEMNIST work until P5 passes on the synthetic testbed.

## Risks to the project's existing claims

1. **Model-relative MNAR does not test drift-causal dropout (G6).**
   - D_model correlates −0.03 to −0.06 with D_true.
   - Under model-relative MNAR, participation_freq_std is 0.042, identical to MCAR.
   - So half of the 350-run MVP effectively tests loss-change-dependent dropout that is about MCAR with respect to true drift.
   - Reframe or caveat it. Report `realized_rho_true` for every cell.
2. **No accuracy headroom in the current drift (G5).**
   - In the seed-0 MVP runs, drift-cohort accuracy (about 0.82) is not below stable-cohort accuracy (about 0.81).
   - In the pilots, model-independent MNAR and MCAR FedAvg drift-cohort accuracy are 0.617 vs 0.619.
   - H1a ("the drifted subpopulation is necessarily worse under MNAR") may come out null at this drift severity.
   - A concept-conflict drift mode is needed to test it.
3. **FedIPW naming and precision.**
   - The repo's `fedipw_estimated` is an EMA of participation history, which is the FedAU/DFL-AA family. It is not Morishita's covariate propensity model. Call it FedIPW-EMA.
   - `fedipw_oracle` gets the true drift-dependent propensity and is therefore unbiased under drift-MNAR by construction.
   - So "FedIPW fails under MNAR" may be claimed only for the estimated variant.
   - The Corollary-6.2 argument applies to covariate-based IPW, not to the EMA. (METHODS.md already records the deviation; the paper must too.)
4. **New close prior art to cite and differentiate.**
   - FL-MalDrift (Sci. Rep. 2025): drift scores used to *exclude* updates on the server side.
   - FedUCA (NeurIPS 2026 per arXiv): endogenous exits driven by utility.
   - Okumus et al. (ICC 2026): clients self-select by a loss threshold.
   - AAggFF (ICML 2024): DR for unsampled clients, MCAR.
   - None of them studies passive, drift-caused self-selection, but the claim "first to study drift-dependent participation" must be scoped to *passive client self-selection driven by its own temporal drift*.
   - FLOSS still precedes any MNAR-in-FL claim. Its citation trail on Semantic Scholar shows one citing paper, FedIPW.
5. **Statistics.** With 5 seeds, an exact two-sided Wilcoxon test cannot reach p < 0.05; its minimum is 0.0625. The analysis must rely on paired t-tests and mixed models, as `analysis/` already does, or use 6 or more seeds.

## Open questions

- Has a lagged measurement already been used as a shadow variable in longitudinal MNAR biostatistics? This is unverified, and DS-DR's statistical novelty claim depends on it. Check Google Scholar for "shadow variable" with "longitudinal" and "previous outcome" before submission.
- Is clients reporting a scalar drift statistic acceptable in the threat model? If not, a DP-noised variant is needed and its cost must be measured.
- Which concept-conflict drift construction should be used? `label_swap+writer` is proposed; D_indep cannot see a pure label permutation.
- Is the measured per-round cost (16.6 h per 100-round run, against about 430 s for the 25-client pilot) dominated by worker contention? Measure it before scheduling Stage 2.
- **Unverified leads to read:** DriftGuard (IEEE IoT-J 2026), the "Lacuna" two-channel proposal, and Lian et al. (federated Heckman multiple imputation, medRxiv 2024).
