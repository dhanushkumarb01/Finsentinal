# Gap Analysis

*Phase R1, 2026-11-22.*

Each gap below has four parts:
- **Evidence** from the verified literature (keys refer to `references.bib`).
- **Repo evidence**, where the existing code or results bear on it.
- A **falsifiable test** that this codebase can run.
- The **candidate method** that targets it (see `CANDIDATE_METHODS.md`).

---

## G1. No FL method identifies a drift-dependent participation propensity from information the server actually has

- **Literature.**
  - Reweighting methods assume participation does not depend on the data:
    - fixed p_i: FedAU [wang2024fedau];
    - Markov, exogenous: [rodio2023cafed; sun2025debiasing];
    - non-stationary, exogenous: [xiang2024fedawe].
  - Others assume ignorability given system covariates [morishita2026whotrains].
  - FedIPW proves its own bias floor when a selection factor is omitted.
  - FLOSS [goetze2025floss] is the only FL paper that identifies an MNAR propensity, and it needs a **static** shadow covariate plus a satisfaction-mediation assumption.
  - No verified paper uses the **temporal persistence** of the client's own state for identification.
- **Repo evidence.**
  - FedIPW-estimated is an EMA of participation history, which is the FedAU/DFL-AA family [hewa2026dflaa].
  - D_indep has lag-1 autocorrelation 0.993, and corr(D_t, last-reported D) = 0.992 for rounds ≥ 21 (MVP seed-0 traces). The ingredient for temporal identification is therefore present.
- **Falsifiable test.** Under model-independent MNAR with β ≥ 2, compute corr(π̂, p_true) among participants. Prediction: history-EMA < 0.5, shadow-GMM ≥ 0.8 (P1).
- **Targeted by:** DS-DR.

## G2. Stale-memory and substitution compensation has never been checked against absence *caused by* change

- **Literature.**
  - These methods assume an absent client's past contribution still stands in for it: MIFA [gu2021mifa], FedVARP [jhunjhunwala2022fedvarp], FedStale [rodio2024fedstale], FedLaAvg [yan2020fedlaavg], MimiC [sun2024mimic], KPFL [lee2025kpfl].
  - FedSteer [zhang2026fedsteer] corrects staleness inside the subspace spanned by the active clients, which by construction excludes a drifted absent client.
  - FL-FDMS [wang2022flfdms] picks friends using pre-drop similarity.
- **Repo evidence.** FL-FDMS logs `friend_same_cohort_frac`. Under MNAR, a drifted client's friend should increasingly come from the stable cohort as drift ramps.
- **Falsifiable test.**
  - Compare `friend_same_cohort_frac` for drifted dropped clients under MCAR vs MNAR. Prediction: it is lower under MNAR after onset.
  - Synthetic Stage 0: the bias of substitution (Corollary 1(e)) grows with β.
- **Targeted by:** DTM (drift-corrected memory), and DS-DR's tilted outcome model.

## G3. No doubly robust aggregation of *model updates* under non-random participation

- **Literature.**
  - In FL, DR or AIPW appears only in:
    - federated treatment-effect estimation [xiong2023federated; han2025face; khellaf2025federated];
    - AAggFF [hahn2024aaggff], for the *responses* of unsampled clients under a **known uniform** sampling probability (MCAR), and only for fairness mixing weights.
  - In recommender systems, DR under MNAR [wang2019doubly] has no identification strategy.
- **Falsifiable test.** Synthetic Stage 0 and FEMNIST M2. DS-DR's bias should be second order (Corollary 1(b)). Ablation A1 (IPW only) and A2 (outcome only) should each be worse than the full method in at least one regime.
- **Targeted by:** DS-DR.

## G4. Drift-adaptation methods cannot see drifted clients that do not report

- **Literature.**
  - FedDrift assumes full participation and detects drift client-side, only for reporting clients [jothimurugesan2023feddrift].
  - Flash [panchal2023flash], FedCCFA [chen2024fedccfa], CFL [sattler2021cfl] and Fielding each infer drift only from submitted updates.
  - "Adaption via Selection" [thomas2025adaption] assumes high-loss clients can be selected.
  - FL-MalDrift [patel2025flmaldrift] *creates* drift-dependent exclusion by design.
- **Repo evidence.** The repo's FedDrift adapts to partial participation (F2/F3): absent clients keep their cluster, and distances are undefined for clusters with no participants.
- **Falsifiable test.** Count FedDrift drift flags within the drift cohort, normalised by drift-cohort participations, under MCAR vs MNAR (logged as `feddrift_*`). Prediction: fewer flags under MNAR, and new-model creation is delayed.
- **Targeted by:** DS-DR (orthogonal to clustering; a per-cluster DS-DR is future work).

## G5. Drift-MNAR may not harm accuracy in the current benchmark: an *evaluation* gap

- **Repo evidence.**
  - In the 14 finished MVP runs (seed 0, model-relative estimator), final global accuracy is 0.810–0.824 for every condition and method.
  - Drift-cohort accuracy (≈ 0.82) is not below stable-cohort accuracy (≈ 0.81).
  - In the day5_v3 pilots (25 clients, β = 4), FedAvg's drift-cohort accuracy is 0.617 under model-independent MNAR vs 0.619 under MCAR.
  - Label-shift drift (another real writer's samples) appears to be **in-distribution for a model trained on 100 writers**. Under-representing the drift cohort therefore costs it little.
- **Literature.** The hostile-validation document (Part 7) predicted that drift-MNAR need not hurt. Hashimoto et al. [hashimoto2018fairness] show harm requires a *conflict* between groups' risks.
- **Falsifiable test.** Pilot P1-b. The prediction is that headroom appears only with a conflicting concept (the `label_swap` mode). If FedAvg's drift-cohort accuracy is not lower under MNAR than MCAR even with `label_swap` at pilot scale, accuracy-based claims for *any* correction method are suspended.
- **Consequence.** Method evaluation must lead with aggregate-bias metrics (M1–M3), which are measurable without headroom. Accuracy is secondary.

## G6. The model-relative drift estimator measures *loss change*, not drift level: a construct-validity gap in the current benchmark

- **Repo evidence.**
  - D_model = loss under the lagged model minus the client's own loss EMA. It is a change detector: it decays back toward 0 once the EMA catches up.
  - corr(D_model, D_true) is −0.03 to −0.06 at pilot and MVP scale, against 0.58–0.80 for D_indep.
  - participation_freq_std under model-relative MNAR (β = 0.5) is 0.042, identical to MCAR's 0.042.
  - D_model's lag-1 autocorrelation is 0.78–0.88, and corr with its last-reported value is 0.44–0.58.
- **Implication.** The model-relative half of the MVP tests *loss-change-dependent* dropout, which is nearly independent of the true drift. It does **not** test drift-causal dropout as defined in the project title.
- **Falsifiable test.** Compute realised ρ_true (already logged as `realized_rho_true`) for model-relative MNAR. Prediction: it stays close to the MCAR value at all β.
- **Targeted by:** no method. This is a framing risk for the existing project (see README "Risks").

## G7. No design-based (double sampling) or sensitivity-analysis treatment of FL participation

- **Literature.**
  - Double sampling of nonrespondents [hansen1946nonresponse], follow-up subsamples [kim2011semiparametric], and the marginal sensitivity model [tan2006distributional; zhao2019sensitivity] are classical.
  - Incentive papers [luo2023incentive; wang2026incentive] reshape participation probabilities but do not subsample non-responders.
  - Su et al. [su2024adversarial] give only a worst-case bound.
  - No verified FL paper uses subsampling of non-responders or sensitivity bounds.
- **Falsifiable test.** Under the `+recall` mechanism, RCS is unbiased even under MNAR-threshold (Stage 0, then pilot).
- **Targeted by:** RCS and SB-Agg.

## G8. Positivity violations (deterministic exclusion of the most drifted clients) have no remedy

- **Literature.**
  - IPW and DR need π > 0 [bang2005doubly].
  - SAFARI [yang2024safari] shows that FL with never-participating clients is not PAC-learnable without side information.
- **Repo evidence.** MNAR-threshold sets p_miss = 1 for the top 20% by drift. In day5_v3, oracle IPW coincides exactly with FedAvg under threshold MNAR, because every participant shares the same constant propensity.
- **Falsifiable test.** P6: DS-DR's bias reduction under threshold MNAR is materially smaller than under logistic MNAR.
- **Targeted by:** RCS (design-based recall), and SB-Agg for reporting bounds.

---

## Summary matrix

| Gap | Type | Test available in the repo now? | Needs new code | Candidate |
|---|---|---|---|---|
| G1 | identification | partially (traces) | GMM in dsdr.py | DS-DR |
| G2 | memory validity | yes (FL-FDMS diagnostics) | — | DTM, DS-DR |
| G3 | estimator | synthetic | dsdr.py | DS-DR |
| G4 | drift blindness | yes (FedDrift logs) | — | (orthogonal) |
| G5 | evaluation headroom | yes (pilot) | label_swap mode | all |
| G6 | construct validity | yes (`realized_rho_true`) | — | framing |
| G7 | design | no | recall mechanism | RCS, SB-Agg |
| G8 | positivity | yes (threshold) | recall mechanism | RCS |
