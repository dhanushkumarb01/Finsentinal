# When the Missing Clients Are the Ones That Changed: Doubly Robust Aggregation under Drift-Dependent Client Dropout

*Draft v0, 2026-11-22. NeurIPS/ICML structure. **No experiments have been run.** The Results section is an explicit placeholder and must be filled only from real runs. Citation keys refer to `references.bib`. All cited works were verified on 2026-11-22.*

---

## Abstract

Federated learning (FL) systems lose clients every round. Almost all corrections for partial participation assume the loss is unrelated to the client's data:
- known or unknown fixed probabilities;
- Markov availability;
- ignorability given system covariates.

We study a regime that breaks this assumption: a client's chance of reporting depends on its own *current* data drift, which the server can see only in rounds when the client reports. This is missing-not-at-random (MNAR) participation.

We make three observations.
1. A drift statistic that the client already computes, together with the persistence of drift over time, makes the participation propensity **identifiable**. The client's last-reported drift statistic acts as a *shadow variable* for its current drift.
2. Building on this, we propose **DS-DR**, a doubly robust aggregation rule. It reduces to an affine combination of participants' models, with scalar coefficients known before local training, and needs no extra model memory.
3. We prove an exact conditional bias identity. It shows that DS-DR's bias is the product of propensity error and outcome-model error, whereas FedAvg, inverse-propensity-weighted (IPW) FedAvg and friend substitution are first-order biased. We also characterise when the method cannot work: positivity violations, drift onset, and selection on unreported state.

We pre-register falsifiable predictions for a synthetic known-truth testbed and for FEMNIST under matched-marginal-rate MCAR, MAR and MNAR dropout. These cover propensity recovery, participation bias of the aggregate, drift-cohort accuracy, and an MCAR negative control.

*[Results: placeholder; to be filled only after experiments.]*

## 1. Introduction

**Problem.** In cross-device FL a device may stop reporting *because of* what is happening to it: a sensor degrades, a user's behaviour shifts, a local anomaly begins. Its absence is then informative.

Existing corrections do not handle this case, for different reasons:
- **Memory-based compensation** reuses the absent client's last update [gu2021mifa; jhunjhunwala2022fedvarp; rodio2024fedstale; zhang2026fedsteer]. That update predates the change that caused the absence.
- **Reweighting** estimates participation rates from history [wang2024fedau; hewa2026dflaa] or from system covariates under ignorability [morishita2026whotrains]. The latter proves its own bias floor when a selection factor is omitted.
- **Drift-adaptation methods** detect drift only in clients that report [jothimurugesan2023feddrift; panchal2023flash; chen2024fedccfa].
- **The one FL paper on MNAR participation** [goetze2025floss] identifies opt-out with a *static* shadow covariate and corrects by IPW only.

**Contributions** (each is a precise, checkable claim):
- **C1 (identification).** Under drift-mediated selection (A1) and strict monotonicity of the stale report's regression on current drift (A4), a logistic participation propensity in the client's current drift statistic is identified. The identifying information is the client's last-reported statistic, used as a shadow variable (Lemma 1). Without it, the model is self-masking and not identified [mohan2021graphical].
- **C2 (estimator).** DS-DR is an augmented-IPW aggregate. Its outcome model is a selection-odds-tilted kernel regression on current participant updates (Lemma 2). It reduces to an affine combination of participants' models (Lemma 0), so it streams with O(1) extra model memory.
- **C3 (bias and variance).** An exact conditional bias identity (Proposition 1) gives:
  - a product-rate bound for DS-DR;
  - first-order bias expressions for FedAvg, FedIPW-type estimators and friend substitution (Corollary 1);
  - a variance expression showing gains for low-propensity clients (Proposition 2).
- **C4 (boundaries).** A characterisation of failure: positivity violation, drift onset, unreported selection drivers and weak persistence (Proposition 3). These are paired with pre-registered experiments designed to exhibit them.
- **C5 (protocol).** A pre-registered evaluation with matched marginal dropout rate, a synthetic known-truth testbed, an MCAR negative control, and ablations isolating each component.

**What we do not claim.**
- We do not claim to introduce MNAR participation in FL; FLOSS does [goetze2025floss].
- We do not claim that shadow-variable or doubly robust estimation is new; those methods are due to [miao2016varieties; bang2005doubly].
- We do not claim that drift-dependent dropout necessarily lowers global accuracy.

## 2. Related Work

**Partial participation with exogenous availability.**
- Arbitrary but data-independent unavailability, handled with memory: MIFA [gu2021mifa], FedLaAvg [yan2020fedlaavg].
- Variance reduction: FedVARP [jhunjhunwala2022fedvarp], FedStale [rodio2024fedstale].
- Correction variables: MimiC [sun2024mimic].
- Subspace steering: FedSteer [zhang2026fedsteer].
- Unknown participation statistics: FedAU [wang2024fedau], generalised FedAvg analysis [wang2022unified], FedAWE/FedSWE [xiang2024fedawe; xiang2026fedswe].
- Markov or correlated availability [rodio2023cafed; sun2025debiasing; ribero2023f3ast]; periodic or regularised participation [crawshaw2024periodic; malinovsky2023regularized]; anarchic participation [yang2022anarchic].
- Worst-case adversarial unavailability [su2024adversarial]. Impossibility without side information [yang2024safari].
- In all of these, participation is independent of the client's data.

**Selection bias and MNAR in FL.**
- FedIPW [morishita2026whotrains] corrects enrollment and participation bias by IPW under ignorability.
- FLOSS [goetze2025floss] models opt-out as MNAR mediated by satisfaction and identifies it with a static shadow variable. It is our closest prior work.
- *Difference:* our shadow is the client's own stale drift report, and identification rests on drift persistence. We add an outcome model and double robustness, and we analyse bias relative to FedAvg, IPW and substitution.
- DR for unsampled clients appears in AAggFF [hahn2024aaggff], under a known uniform sampling probability and only for fairness weights.
- Federated AIPW exists for treatment effects [xiong2023federated; han2025face; khellaf2025federated], and multiply robust weighting for records missing within sites [vazquez2026incomplete].

**Substitution.**
- FL-FDMS [wang2022flfdms] substitutes a similar client's current update.
- Our outcome model generalises it: a kernel instead of an argmax, plus a selection tilt plus a reweighting leg (Corollary 1(e)).

**Concept drift in FL.**
- Multi-model and clustered approaches [jothimurugesan2023feddrift; ghosh2020ifca; sattler2021cfl; guo2024fedrc; chen2024fedccfa] and adaptive optimisers [panchal2023flash; canonaco2021adaptive] assume full or random participation.
- Drift-based *selection* [thomas2025adaption] and drift-based *exclusion* [patel2025flmaldrift] are server-side choices, not passive self-selection.

**Endogenous participation.**
- Loss-dependent retention in centralised ML [hashimoto2018fairness; zhang2019group] and participation dynamics [dean2024emergent].
- In FL: utility-driven exits [yashwanth2026feduca; cho2022maxfl] and incentive designs [luo2023incentive; karimireddy2022mechanisms]. These *prevent* or *price* exits rather than correcting for them.

**Missing-data theory.**
- MCAR/MAR/MNAR [rubin1976inference]; selection models [heckman1979sample; diggle1994informative]; pattern mixture [little1993pattern].
- AIPW and DR [robins1994estimation; bang2005doubly; kang2007demystifying].
- Shadow variables and nonresponse instruments [miao2016varieties; miao2024identification; wang2014instrumental; tchetgen2017general; sun2018semiparametric].
- Graphical recoverability [mohan2021graphical]; tilting sensitivity analysis [scharfstein1999adjusting].

## 3. Problem Setup

**Clients, rounds and updates.**
- N clients, rounds t = 1..T, global model θ_t.
- Client i's would-be update is Δ_{i,t}, with weight w_i = n_i / Σ_k n_k.
- **Estimand:** Δ̄_t = Σ_i w_i Δ_{i,t}, the full-participation FedAvg update.

**Participation.**
- R_{i,t} ∈ {0, 1}, with propensity π_{i,t}.
- **Matched-marginal-rate design:** E_i[1 − π_{i,t}] = p = 0.30 in every round and every mechanism.
- The mechanisms are:

| Mechanism | Definition |
|---|---|
| MCAR | π_{i,t} = 1 − p |
| MAR | 1 − π = clip(σ(α + γ X_i)), where X_i is a fixed latent device covariate with X ⫫ data |
| MNAR-logistic | 1 − π = clip(σ(α_t + β·std_t(D_{i,t}))), where D is the client's current drift score |
| MNAR-threshold | π = 0 for the top 20% of clients by D; otherwise a constant calibrated to the target rate |

**Observation model.**
- Participants send Δ_{j,t} and a p-dimensional drift statistic S_{j,t}, computed on-device before training. For example, D_indep is a frozen-random-feature embedding distance to the client's own reference window.
- The server stores Z_{i,t}, the last report from each client.
- D (and hence π) is unobserved for absent clients. MNAR-logistic is therefore MNAR in Rubin's sense, and self-masking on S unless the shadow is used.

## 4. Method: DS-DR

**Propensity by temporal shadow GMM.** Over the past window 𝒲_t, solve

  Σ_{s∈𝒲_t} Σ_i ( R_{i,s} / σ(γ + η^⊤ S_{i,s}) − 1 ) (1, Z_{i,s}^⊤)^⊤ = 0.

Each term needs S only when R = 1. Then set π̂_j = clip(σ(γ̂ + η̂^⊤ S_{j,t}), 0.05, 0.95).

**Gate.** If the Wald test of η = 0 does not reject at level α_g, use exact FedAvg weights.

**Outcome model.** With selection odds ω_k = (1 − π̂_k)/π̂_k and a Gaussian kernel K on standardised Z, the leave-one-out tilted kernel mean is

  m_i = Σ_{k∈S_t∖i} K_{ik} ω_k Δ_k / Σ_{k∈S_t∖i} K_{ik} ω_k.

**Aggregate.**

  Δ̂_t = Σ_i w_i m_i + Σ_{j∈S_t} (w_j / π̂_j)(Δ_j − m_j)  =  Σ_{k∈S_t} c_k Δ_k,  with Σ_k c_k = 1.

Leverage is capped by shrinking c toward the FedAvg weights.

**Algorithm 1 (DS-DR, one round).**
1. Receive S_{j,t} from participants.
2. Fit θ̂ by GMM on the buffer and compute the gate.
3. Compute π̂, ω, K and c.
4. Participants train. The server streams Σ_k c_k θ_k^{local}.
5. Push (R_t, S_t, Z_t) to the buffer and update Z.

**Complexity.**
- Server: O(W N p) per Gauss–Newton step, plus O(N |S_t|) scalars for the kernel.
- Communication: +p floats per participant.
- Memory: O(W N p) scalars and no model copies, compared with N model copies for MIFA/FedVARP.

## 5. Theory

**Assumptions.**
- **A1. Drift-mediated selection.** R_{i,t} ⫫ (Δ_{i,t}, Z_{i,t}) | S_{i,t}, with conditional independence across clients.
- **A2. Positivity.** π ≥ c > 0.
- **A3. Logistic propensity, stable within a window.**
- **A4. Shadow relevance.** E[Z | S = s] is strictly monotone.

**Lemma 0 (affine form).** If |S_t| ≥ 2, then Σ_k c_k = 1.

**Lemma 1 (identification).** Under A1–A4 with scalar S, the moments E[(R/π_θ(S) − 1) h(Z)] = 0 with h = (1, Z) hold iff θ = θ*.
- *Sketch.* With g = π*/π_θ − 1, which crosses zero at most once for logistic families, the moments give E[g] = E[μ_Z(S) g] = 0. The product g·(μ_Z − μ_Z(s_0)) has constant sign, so g = 0.

**Lemma 2 (tilting).** E[Δ | Z, R = 0] = E[O(S) Δ | Z, R = 1] / E[O(S) | Z, R = 1], with O = (1 − π)/π.

**Proposition 1 (exact conditional bias).** Suppose θ̂ is fitted on past rounds and m is leave-one-out. Then

  E[Δ̂_t | 𝒢_t] − Δ̄_t = Σ_i w_i (π_i/π̂_i − 1)(Δ_i − E[m_i | 𝒢_t]).

- *Sketch.* R_i ⫫ m_i given 𝒢_t; the result follows by linearity.

**Corollary 1.**
- **(a)** π̂ = π implies unbiasedness for any m.
- **(b)** Product bound: ‖Bias‖ ≤ ‖π/π̂ − 1‖_{2,w} · ‖Δ − μ‖_{2,w}.
- **(c)** IPW with m = 0 has first-order bias Σ_i w_i (π_i/π̂_i − 1) Δ_i.
- **(d)** FedAvg has bias ≈ Cov_w(π, Δ)/π̄. This is zero under MCAR, and under MAR when X ⫫ Δ.
- **(e)** Friend substitution has bias Σ_{i∉S} w_i (Δ_i − Δ_{f(i)}).

**Proposition 2 (variance).**

  tr Var[Δ̂_t | 𝒢_t] = Σ_i w_i² · π_i(1 − π_i)/π̂_i² · ‖Δ_i − μ_i‖² + O(1/|S_t|).

**Proposition 3 (failure regimes).**
- (i) Positivity violation leaves the bias term Σ_{i∈ℰ} w_i (Δ_i − μ_i), which no reweighting can remove.
- (ii) At drift onset, A4 holds only weakly over the window.
- (iii) If selection depends on a statistic that is not reported, A1 fails.
- (iv) Weak persistence leads to weak identification.

## 6. Experimental Protocol (pre-registered; see EVALUATION_PLAN.md)

**Data and drift.**
- FEMNIST, writer-partitioned, 100 clients at full scale, with a drift cohort of 30 clients.
- Drift modes:
  - `label_shift`: real-writer substitution, already in the repository.
  - `label_swap+writer`: a cohort-shared class permutation combined with writer substitution, which creates a concept conflict.
- A synthetic known-truth testbed with quadratic losses and AR(1) drift statistics.

**Baselines.** FedAvg, FedIPW-oracle (true propensity; an unbiased ceiling), FedIPW-EMA (history-based, FedAU-type), FL-FDMS, FedDrift.

**Ablations.** IPW only; outcome only; lag-MAR or post-stratified IPW (no shadow); no tilt; no gate; oracle propensity with our outcome model; reported statistic restricted; no leverage cap.

**Mechanisms.** MCAR, MAR, MNAR-logistic with β ∈ {0.5, 1, 2, 4, 8}, MNAR-threshold. Model-relative and model-independent drivers. Matched marginal rate 0.30.

**Hypotheses and thresholds.**
- **P1.** corr(π̂, p_true) ≥ 0.8, and at least 0.3 above FedIPW-EMA.
- **P2.** Drift-cohort weight-share bias reduced by ≥ 50% relative to FedAvg.
- **P3.** MCAR/MAR global accuracy equivalent to FedAvg within ±0.5 pp (TOST), with the gate open in ≤ 5% of rounds.
- **P4.** Under `label_swap+writer`, drift-cohort accuracy ≥ FedAvg + 2 pp and ≥ FedIPW-EMA + 1 pp, with no global accuracy loss beyond 0.5 pp.
- **P5.** On the synthetic testbed, bias ≤ 25% of FedAvg's and RMSE ≤ oracle IPW's.
- **P6.** Under MNAR-threshold, the improvement is materially smaller.

**Statistics.**
- Seed-paired comparisons; at least 6 seeds where Wilcoxon tests are used.
- Paired t-tests and exact Wilcoxon tests.
- 95% BCa bootstrap confidence intervals and Cohen's d_z.
- Holm correction across the primary hypotheses.
- Mixed models as secondary analyses.
- No pseudo-replication across rounds.

**Stages.**
- Synthetic testbed: minutes.
- Pilot at 25 clients × 30 rounds: hours.
- Mid-scale run.
- Full scale at about 16.6 h per run, reusing the bit-identical MVP baseline runs.

## 7. Results

**PLACEHOLDER: no experiments have been run. Fill only from logged runs; every number must trace to a `results/` file.**

| Setting | Metric | FedAvg | FedIPW-oracle | FedIPW-EMA | FL-FDMS | FedDrift | DS-DR | DS-DR (lag-MAR) |
|---|---|---|---|---|---|---|---|---|
| Synthetic, β = 4, ρ_S = 0.99 | bias ratio to FedAvg | — | — | — | — | n/a | — | — |
| FEMNIST mind-MNAR β = 4 | corr(π̂, p) | n/a | 1 | — | n/a | n/a | — | — |
| FEMNIST mind-MNAR β = 4 | weight-share bias | — | — | — | — | — | — | — |
| FEMNIST MCAR | Δ global accuracy vs FedAvg | 0 | — | — | — | — | — | — |
| label_swap+writer mind-MNAR β = 4 | drift-cohort accuracy | — | — | — | — | — | — | — |
| MNAR-threshold | weight-share bias | — | — | — | — | — | — | — |

## 8. Limitations and Threats to Validity

- **Reporting assumption.** DS-DR needs participants to report a scalar drift statistic. This is a small privacy cost; a DP-noise ablation is planned.
- **A1 is untestable.** It assumes selection only through the reported statistic. The simulator satisfies it by construction, but real deployments may not. Proposition 3(iii) quantifies the consequence.
- **Persistence.** Identification relies on drift persistence. In the repository's traces this holds for D_indep (0.99) and only weakly for D_model (about 0.5).
- **Accuracy headroom may be absent.** Seed-0 MVP runs show no drift-cohort accuracy deficit under the current label-shift drift. Accuracy claims are therefore conditional on the concept-conflict drift mode, and aggregate-bias metrics are primary.
- **Synthetic drift.** FEMNIST has no natural drift, so all drift is injected. External validity to real devices is not established, and no public dataset with measured drift-dependent dropout is known.
- **Model-relative mechanism.** Its driver correlates about 0 with true drift, so MNAR-model-relative results speak to loss-change-dependent dropout, not drift-causal dropout.
- **Statistical power.** With 5–6 seeds, only paired effects larger than about 1.3 SD are detectable.
- **Novelty.** The statistical machinery is known. The contribution is its FL instantiation, the persistence-based identification condition, and the comparative bias analysis.
- **Sensitivity to the clip band and leverage cap.** These are pre-registered and ablated.
