# Candidate Methods for Mitigating Drift-Dependent (MNAR) Client Dropout

*Phase R1, 2026-11-22. Every novelty verdict cites verified entries in `LITERATURE_REVIEW.md` / `references.bib`. No experiment has been run, and no numbers below are results.*

Contents:
- Shared notation (§0)
- **Lead method: DS-DR** (§1), with full formulation, theory, pseudo-code, code mapping, predictions and a novelty verdict
- Backups and alternatives (§2–§6)
- Ranking (§7)

---

## 0. Setting and notation (shared by all candidates)

**Clients, rounds and the model.**
- Clients are i ∈ [N] and rounds are t = 1..T. The global model is θ_t.
- **Would-be update:** Δ_{i,t} = LocalTrain(θ_t; data_{i,t}) − θ_t. It is defined for every client and observed only if R_{i,t} = 1.
- **Footnote on local-SGD randomness.** In the simulator, local-SGD randomness comes from a shared torch stream, so a counterfactual Δ_{i,t} depends weakly on who else participates. In the theory, Δ_{i,t} denotes the update conditional on (θ_t, data_{i,t}), averaged over local-SGD noise.

**Aggregation weights.** w_i = n_i / Σ_k n_k, where n_i is the last reported sample count (fixed in this simulator).

**Estimand (full-participation FedAvg update):**

  Δ̄_t = Σ_{i=1}^N w_i Δ_{i,t}.

This is the estimand FedIPW also targets: the population-mean update [morishita2026whotrains].

**Participation.**
- R_{i,t} ∈ {0, 1}.
- **True propensity:** π_{i,t} = P(R_{i,t} = 1 | ·).
- In the simulator, MNAR-logistic is π_{i,t} = 1 − clip(σ(α_t + β·std_t(D_{i,t})), 0.05, 0.95), with α_t re-solved each round so that the mean miss rate is 0.30 (`src/mechanisms/dropout.py`).

**Client drift statistic S_{i,t} ∈ ℝ^p.** This is a vector the client can compute on-device *before* local training, from its own current data:
- D_indep: squared distance between the mean embedding of the current data under a frozen random CNN and the client's own reference window (`src/drift/model_independent.py`).
- Optionally D_model: the loss of the lagged model on the current data minus the client's own loss EMA (`src/drift/model_relative.py`).
- In the simulator, the D that drives MNAR is exactly one coordinate of S_{i,t}.

**Shadow (stale) report** Z_{i,t} = S_{i,τ_i(t)}, where τ_i(t) = max{s < t : R_{i,s} = 1}. The warm-up round 0 counts as a report for every client, with S = 0 by construction. The server stores Z for all clients, so **Z_{·,t} is fully observed at every round.**

**Measured persistence** (finished MVP traces, seed 0, rounds ≥ 21):

| Statistic | corr(S_t, Z_t) | lag-1 autocorrelation | corr with D_true |
|---|---|---|---|
| D_indep | 0.992 | 0.993 | 0.58 |
| D_model | 0.44–0.58 | 0.78–0.88 | ≈ 0.04 |

D_indep is highly persistent. D_model is a noisy *change* detector, not a level estimator.

---

## 1. LEAD METHOD — DS-DR: Drift-Shadow Doubly Robust Aggregation

### 1.1 Intuition

Under drift-MNAR, the server misses exactly the clients whose *current* drift is high. It never sees their current drift, which is what makes the problem MNAR.

Drift is **persistent**. The drift a client reported the last time it participated is highly predictive of its drift now. Given the current drift, though, that stale report carries no further information about whether the client participates *now*, because the mechanism acts on the current state.

That is exactly a **shadow variable** in the sense of Miao & Tchetgen Tchetgen [miao2016varieties; miao2024identification] and the nonresponse-instrument sense of Wang, Shao & Kim [wang2014instrumental]. It makes the drift-dependent propensity π(S) **identifiable** from data the server already holds, with no extra channel from absent clients.

On top of the identified propensity we build a **doubly robust (AIPW) aggregate** [robins1994estimation; bang2005doubly]. Its outcome model imputes an absent client's update from the *current* updates of participants with similar stale drift reports, tilted by the identified selection odds. This combines FL-FDMS-style substitution [wang2022flfdms] with IPW-style reweighting [morishita2026whotrains] in one estimator, with an exact bias identity.

The final rule is an **affine combination of participants' models with scalar coefficients known before local training starts**. It streams with O(1) extra model memory.

### 1.2 Assumptions

- **A1 (drift-mediated selection, cross-client conditional independence).** For each t, conditional on the past F_{t−1}:
  - R_{i,t} ⫫ (Δ_{i,t}, Z_{i,t}) | S_{i,t};
  - P(R_{i,t} = 1 | S_{i,t} = s, F_{t−1}) = π_t(s);
  - the R_{i,t} are independent across i given (S_{·,t}, F_{t−1}).

  In the simulator this holds *exactly* for MCAR, MNAR-logistic and MNAR-threshold, provided S contains the D that drives the mechanism. Independent uniforms are drawn per client, and α_t is a round-level constant. For MAR it holds with π_t(s) constant in s, because the latent covariate X is independent of the data.
- **A2 (positivity).** π_t(s) ≥ c > 0. In the simulator c = 0.05. **A2 fails for MNAR-threshold.**
- **A3 (parametric propensity, stable within a window).** π_t(s) = σ(γ + η^⊤ s) for t in a window of W rounds. For MNAR-logistic this is exact up to the [0.05, 0.95] clip. The clip is a bounded misspecification, and its effect enters Proposition 1 through π/π̂.
- **A4 (shadow relevance).** E[Z_{i,t} | S_{i,t} = s] is strictly monotone in s (scalar case), or the GMM Jacobian has full rank (vector case). It is satisfied empirically for D_indep and weakly for D_model (see §0).

### 1.3 Estimators

**(a) Propensity by temporal shadow-variable GMM.**
- Use the past window 𝒲_t = {t−W, …, t−1} and h(z) = (1, z^⊤)^⊤, which gives p+1 moments for p+1 parameters (just identified).
- Solve

  Ĝ_t(γ, η) = (1 / (N|𝒲_t|)) Σ_{s∈𝒲_t} Σ_{i=1}^N ( R_{i,s} / π_{γ,η}(S_{i,s}) − 1 ) h(Z_{i,s}) = 0.

- The summand needs S_{i,s} only when R_{i,s} = 1. When R_{i,s} = 0 it equals −h(Z_{i,s}). **The estimating equation is therefore computable from observed data alone.**
- The solution θ̂_t = (γ̂, η̂) is F_{t−1}-measurable (*temporal cross-fitting*). It is found by Gauss–Newton on p+1 parameters.
- Set π̂_{j,t} = clip(π_{θ̂_t}(S_{j,t}), 0.05, 0.95) for participants j.

**(b) Selection gate.**
- Compute the sandwich (just-identified GMM) Wald statistic for H0: η = 0.
- If H0 is not rejected at level α_g, *or* the Jacobian is ill-conditioned, *or* |𝒲_t| < W_min, the gate is **closed** and the round uses exact FedAvg weights.
- This makes DS-DR collapse to FedAvg under MCAR and MAR (the negative control).

**(c) Tilted-kernel outcome model, leave-one-out.**
- Selection odds: ω_k = (1 − π̂_k) / π̂_k.
- Kernel on standardised stale reports: K_{ik} = exp(−‖Z̃_i − Z̃_k‖² / (2h²)), with h = κ · median pairwise distance.
- For every client i, absent or participating:

  m_{i,t} = Σ_{k∈S_t∖{i}} κ_{ik} Δ_{k,t},  where κ_{ik} = K_{ik} ω_k / Σ_{l∈S_t∖{i}} K_{il} ω_l.

**(d) DS-DR aggregate:**

  Δ̂_t = Σ_{i=1}^N w_i m_{i,t} + Σ_{j∈S_t} (w_j / π̂_{j,t}) (Δ_{j,t} − m_{j,t}),  θ_{t+1} = θ_t + Δ̂_t.

**(e) Coefficient (streaming) form.** Substituting (c) into (d) gives Δ̂_t = Σ_{k∈S_t} c_k Δ_{k,t}, with

  c_k = Σ_{i=1}^N w_i κ_{ik} + w_k / π̂_k − Σ_{j∈S_t} (w_j / π̂_j) κ_{jk}.

**Lemma 0.** If |S_t| ≥ 2, then Σ_k c_k = Σ_i w_i = 1.
- *Proof.* Each row of κ sums to 1, and the second and third terms cancel after summing over k. ∎

Because the coefficients sum to 1, θ_{t+1} = Σ_k c_k θ^{local}_k exactly. The c_k depend only on (S_{j,t})_{j∈S_t}, Z_{·,t}, θ̂_t and w, all available in `begin_round` before any local training. They can be fed directly to the existing `StreamingWeightedAverage`, which accepts signed weights with a positive total.

**(f) Leverage control.**
- Signed coefficients can raise variance [kang2007demystifying].
- Let c^{FA} be the FedAvg weights. Use c^λ = (1 − λ) c + λ c^{FA}, with λ ∈ [0, 1] the smallest value such that max_k |c^λ_k| ≤ ρ_max / |S_t|.
- Any λ > 0 introduces a bias, which is logged and ablated.

### 1.4 Theory

**Lemma 1 (identification by the temporal shadow).** Assume A1–A4 with scalar S and π in the logistic family. Then the population moments E[(R/π_θ(S) − 1) h(Z)] = 0 hold iff θ = θ*.
- *Proof sketch.*
  1. By A1, E[R/π_θ(S) · h(Z)] = E[π*(S)/π_θ(S) · h(Z)]. So the moments read E[g(S)] = 0 and E[Z g(S)] = E[μ_Z(S) g(S)] = 0, where g = π*/π_θ − 1 and μ_Z(s) = E[Z | S = s].
  2. For two logistic functions, g has the sign of the difference of two affine functions of s. So g has at most one sign change, at some point s_0.
  3. Hence g(s)(μ_Z(s) − μ_Z(s_0)) has a **constant sign** in s, and it is non-zero wherever g ≠ 0, because μ_Z is strictly monotone (A4). If there is no sign change, g has a constant sign and E[g] = 0 already forces g = 0.
  4. Then E[g(S)(μ_Z(S) − μ_Z(s_0))] = E[Z g] − μ_Z(s_0) E[g] = 0 forces g = 0 a.s. ∎
  5. **Vector S:** the analogous statement needs a rank/completeness condition [miao2024identification]. We check it empirically through the condition number of the GMM Jacobian.
- **Without the shadow, identification fails.** Using only h ≡ 1 identifies a single parameter, which is the classic non-identifiability of self-masking [mohan2021graphical].
- **Persistence is the structural fact that buys identification.**

**Lemma 2 (tilting identity behind the outcome model).** Under A1,

  E[Δ | Z, R = 0] = E[O(S) Δ | Z, R = 1] / E[O(S) | Z, R = 1],  with O(s) = (1 − π(s)) / π(s).

- *Proof.* By Bayes' rule, p(Δ, S | Z, R = 0) = p(Δ, S | Z, R = 1) · O(S) · P(R = 1 | Z) / P(R = 0 | Z), using A1. ∎
- The kernel-weighted, ω-tilted mean in §1.3(c) is the Nadaraya–Watson plug-in of this identity.

**Proposition 1 (exact conditional bias).** Let 𝒢_t = (F_{t−1}, Δ_{·,t}, S_{·,t}, Z_{·,t}). Assume A1, that θ̂_t is F_{t−1}-measurable, and that m_{i,t} is leave-one-out. Write μ_i = E[m_{i,t} | 𝒢_t]. Then

  E[Δ̂_t | 𝒢_t] − Δ̄_t = Σ_{i=1}^N w_i (π_i / π̂_i − 1)(Δ_{i,t} − μ_i).

- *Proof.*
  1. m_i is a function of {R_k}_{k≠i}, so R_i ⫫ m_i | 𝒢_t by A1.
  2. Hence E[R_i/π̂_i · (Δ_i − m_i) | 𝒢_t] = (π_i/π̂_i)(Δ_i − μ_i).
  3. Adding Σ_i w_i μ_i from the first term gives Δ̄_t + Σ_i w_i (π_i/π̂_i − 1)(Δ_i − μ_i). ∎

**Corollary 1.**
- **(a) Unbiased if the propensity is correct.** If π̂ = π, DS-DR is conditionally unbiased for Δ̄_t for **any** outcome model. With m ≡ 0 this recovers oracle IPW in Horvitz–Thompson form.
- **(b) Product-rate double robustness.** ‖Bias‖ ≤ (Σ_i w_i (π_i/π̂_i − 1)²)^{1/2} · (Σ_i w_i ‖Δ_i − μ_i‖²)^{1/2}. The bias is second order: it vanishes if *either* nuisance is exact, and it is small if both are moderately good.
- **(c) FedIPW-estimated / FedAU (m ≡ 0).** Bias = Σ_i w_i (π_i/π̂_i − 1) Δ_i. This is first order in the propensity error times the *raw* update, not the residual.
- **(d) FedAvg** (Hájek with π̂ constant). To first order in 1/N, bias ≈ Σ_i w_i (π_i/π̄ − 1)(Δ_i − Δ̄) = Cov_w(π, Δ)/π̄_w. This is the formal statement of the drift-MNAR harm: it is non-zero iff the propensity covaries with the update. It is zero under MCAR, and it is zero under MAR when X ⫫ Δ.
- **(e) FL-FDMS-style substitution** is the case π̂ ≡ 1 with m_i = Δ_{friend(i)} for absent clients. Its bias is Σ_{i∉S} w_i (Δ_i − Δ_{friend(i)}). There is no reweighting leg, so the bias is first order in the substitution error.

**Proposition 2 (conditional variance, trace).** Under A1, treating m as fixed to leading order:

  tr Var[Δ̂_t | 𝒢_t] = Σ_i w_i² · π_i(1 − π_i)/π̂_i² · ‖Δ_i − μ_i‖² + O(1/|S_t|).

For comparison:
- Horvitz–Thompson IPW has ‖Δ_i‖² in place of the residual.
- Hájek IPW has approximately ‖Δ_i − Δ̄‖².

DS-DR reduces variance whenever the stale-drift-localised, tilted mean explains more of the update than the global mean does. The gain is largest for low-π (drifted) clients, whose factor (1 − π)/π reaches 19.

**Proposition 3 (when the method cannot work: an honest boundary).**
- **(i) Positivity failure (MNAR-threshold).** π_i = 0 on the excluded set ℰ. The term Σ_{i∈ℰ} w_i (Δ_i − μ_i) cannot be removed by any reweighting. Only the outcome model can reduce it, and that is pure extrapolation. Prediction: DS-DR helps at most partially under threshold MNAR.
- **(ii) Drift onset.** At an onset round, a client's stale report predates its change, so μ_Z(s) is locally flat for the newly drifting clients. Window pooling (A3) then mis-estimates η for about W rounds.
- **(iii) Selection on something S does not contain.** For example, MNAR driven by D_model while the client reports only D_indep. Then A1 fails, and the bias is Proposition 1 with a misspecified π. This is why S includes both statistics, and ablation (g) tests it.
- **(iv) Weak shadow.** If corr(Z, S) is low, as for D_model, the GMM is weakly identified and π̂ is noisy. The gate should then stay closed more often.

### 1.5 Pseudo-code

```
Server state: Z[1..N] = 0 (warm-up reports), buffer B = deque(maxlen=W) of (R_s[1..N], S_s[participants], Z_s[1..N])

for round t = 1..T:
    S_t <- participants' reported drift stats             # p floats each, sent with (or just before) the update
    theta_hat, gate_open <- shadow_gmm(B) if len(B) >= W_min else (None, False)
    if not gate_open or |S_t| < 2:
        c_k <- n_k / sum_{j in S_t} n_j                     # exact FedAvg
    else:
        pi_hat_j <- clip(sigmoid(gamma + eta . S_{j,t}), .05, .95);  omega_j <- (1 - pi_hat_j) / pi_hat_j
        K <- exp(-||Zs_i - Zs_k||^2 / 2h^2) for i in [N], k in S_t   # N x |S_t| scalars
        kappa_ik <- K_ik omega_k / sum_{l in S_t, l != i} K_il omega_l   (kappa_kk = 0)
        c_k <- sum_i w_i kappa_ik + w_k / pi_hat_k - sum_{j in S_t} (w_j / pi_hat_j) kappa_jk
        c <- leverage_cap(c, c_FedAvg, rho_max)
    broadcast theta_t; participants train; server accumulates sum_k c_k * theta_local_k  (streaming)
    B.append((R_t, S_t, Z_t)); Z[j] <- S_{j,t} for j in S_t
```

`shadow_gmm`: Gauss–Newton on Ĝ(θ) = 0 over the pooled buffer. It returns θ̂, the sandwich covariance, the Wald p-value for η = 0, and a condition number. The gate opens iff p < α_g and the condition number is below 1e6.

### 1.6 Cost

| Resource | Cost |
|---|---|
| Server compute per round | GMM: O(W·N·p) per Gauss–Newton step, ~10 steps. Kernel: O(N·\|S_t\|) scalars. Under 10 ms at N = 100, against about 600 s of local training per round. |
| Communication | +p floats per participant per round (p ≤ 2), against 6.6 M floats per model. Negligible. |
| Memory | O(W·N·p) scalars. **No extra model copies**, unlike MIFA/FedVARP (N copies) or FL-FDMS (\|S\| update vectors, ~1.8 GB here). |
| Client compute | D_indep needs one forward pass of a tiny frozen CNN over local data, which is already done in the simulator for every client. |

### 1.7 Expected advantages, limitations and failure modes

**Advantages**
- Targets the right estimand under drift-MNAR with a propensity identified from server-held data.
- Collapses to FedAvg when there is no selection on drift (gate).
- Lower variance than oracle IPW when the outcome model is informative.
- No model memory.

**Limitations**
- Needs a client-computable drift statistic to be *reported*. That is a privacy leak of one scalar; a DP-noise ablation is included.
- Needs drift persistence (A4).
- Needs A1: selection only through the reported statistic.
- Cannot fix positivity violations.
- Signed weights can be unstable (leverage cap).
- Gains cannot exceed the bias that FedAvg actually suffers. **The existing MVP evidence suggests that bias is small on the current label-shift benchmark** (see GAP_ANALYSIS G5). Accuracy gains may only appear with a conflicting-concept drift.

**Failure modes to monitor (all logged)**
- Gate-open rate under MCAR above 5% (false positives).
- Corr(π̂, true p) below 0.5 under mind-MNAR.
- Large λ (frequent capping).
- min_k c_k < −1/|S_t|.

### 1.8 Mapping onto `drift-causal-fl/src` (additive only)

**Hard requirement:** existing method code is not touched, and the paused sweep must resume bit-identically. All edits are additive and gated. Do them in a separate git worktree or branch, and merge only after the legacy regression fixtures pass.

| File | Change |
|---|---|
| `src/methods/dsdr.py` (**new**) | `DriftShadowDRStrategy(GlobalModelStrategy)`, `name = "dsdr"`, `needs_reported_drift = True`. `begin_round(ctx)`: GMM from its own buffer, gate, compute c, then `self._set_weights(c)`. `end_round`: `super().end_round(ctx)`, then push (R_t, S_t, Z_t) to the buffer and update Z. `round_log`: θ̂, Wald p-value, gate, π̂ per participant, λ, min/max c, signed ESS (Σc)²/Σc². `get_state` / `set_state`: Z, buffer, θ̂. Helper functions `shadow_gmm`, `dsdr_coefficients` are pure numpy and unit-testable. |
| `src/methods/base.py` | **Additive:** a `RoundContext` field `reported_drift: dict[int, np.ndarray] \| None = None` (default None) and the class attribute `Strategy.needs_reported_drift = False`. Existing strategies never read either. |
| `src/simulation/mechanism_runner.py` | **Additive, guarded:** when building `ctx`, set `reported_drift={i: np.array([D_indep[i], D_rel[i]]) for i in parts} if strategy.needs_reported_drift else None`. No RNG is consumed. In the evaluation-side diagnostics block (like `friend_same_cohort_frac`), add god-view `corr(pi_hat, 1 - p_miss)` and drift-cohort weight share Σ_{k∈drift} c_k. These are evaluation only and never fed back. `build_strategy`: new branches. |
| `src/constants.py` | Append `"dsdr"`, `"dsdr_ipw"` (m ≡ 0), `"dsdr_lagmar"` (π̂ from a logistic MLE of R on Z, no shadow), `"dsdr_nogate"`, `"dsdr_oraclepi"` (true p plus our m; evaluation-only ablation). The MVP config does not list them, so its expansion and config hash are unchanged. |
| `src/experiments/config.py` | Allow kwargs for `dsdr*` (W, W_min, alpha_gate, kappa, rho_max, stats). |
| `src/drift/injection.py` (+ the matching evaluation helper) | **New mode** `label_swap` (see EVALUATION_PLAN §2): a cohort-shared permutation of k classes applied to a `magnitude` fraction of samples. It creates P(Y\|X) conflict, which is needed for accuracy headroom. The existing modes are untouched. |
| `tests/test_dsdr.py` (**new**) | Lemma 0 (Σc = 1); gate closed ⇒ weights equal FedAvg exactly; Monte-Carlo check of Proposition 1 on a numpy toy; GMM recovers θ* on simulated data; the information invariant (only participants' S); checkpoint resume is bit-identical. |
| Regression | Run `tests/` including `legacy_runner_outputs.json` to prove FedAvg/FedIPW are bit-identical after the change. |

### 1.9 Hyperparameters (defaults, all pre-registered)

| Name | Default | Rationale | Ablation grid |
|---|---|---|---|
| W (GMM window) | 5 rounds | stability vs. drift onset | {3, 5, 10} |
| W_min | 3 | — | — |
| α_g (gate level) | 0.01 | false-positive rate under MCAR | {0.001, 0.01, 0.05, off} |
| κ (bandwidth multiplier) | 1.0 | median heuristic | {0.5, 1, 2, ∞ (no localisation)} |
| ρ_max (leverage cap) | 10 | — | {5, 10, ∞} |
| π clip | [0.05, 0.95] | same band as FedIPW in the repo | — |
| Reported S | (D_indep, D_model) | covers both MNAR drivers | {D_indep}, {D_model} |
| DP noise on S | 0 | — | σ ∈ {0, 0.1·sd(S)} |

### 1.10 Pre-registered, falsifiable predictions (decision thresholds, not claims)

| ID | Condition | Prediction / success threshold | Falsified if |
|---|---|---|---|
| P1 | mind-MNAR, β ≥ 2, rounds ≥ 30 | Seed-mean corr(π̂_DSDR, p_true) among participants ≥ 0.8, and it exceeds FedIPW-EMA's corr by ≥ 0.3 | either fails |
| P2 | mind-MNAR, β ≥ 2 | Drift-cohort weight-share bias \|WS − WS_full\| reduced by ≥ 50% vs FedAvg (seed-paired, Holm-adjusted p < 0.05) | reduction < 50% or not significant |
| P3 (negative control) | MCAR, MAR | Gate open in ≤ 5% of rounds. Global accuracy equivalent to FedAvg within ±0.5 pp (TOST, α = 0.05) | equivalence fails |
| P4 (accuracy) | label_swap drift, mind-MNAR β = 4 | Drift-cohort accuracy ≥ FedAvg + 2.0 pp and ≥ FedIPW-EMA + 1.0 pp. Global accuracy not below FedAvg − 0.5 pp | any of the three fails |
| P5 (synthetic, known truth) | AR(1) drift, ρ ≥ 0.9, β ≥ 2 | Aggregate bias norm ≤ 25% of FedAvg's and ≤ 50% of lag-MAR IPW's. RMSE ≤ oracle-IPW RMSE | any fails |
| P6 (boundary) | MNAR-threshold | DS-DR's bias reduction is materially smaller than under logistic MNAR (Proposition 3(i)) | DS-DR fully removes threshold bias, which would contradict the theory and point to a bug |
| P7 (boundary) | mrel-MNAR | Lower identification quality than mind (P1 corr lower). No accuracy effect expected, since mrel-MNAR ≈ MCAR w.r.t. true drift | — (descriptive) |

### 1.11 Novelty verdict (harsh)

**The three closest works and the exact difference:**

1. **FLOSS** [goetze2025floss]. MNAR participation in FL, identified with a **shadow variable**, then IPW.
   - FLOSS's shadow is a **static device covariate** (e.g. processing power), and selection is mediated by self-reported *satisfaction*.
   - It has **no outcome model and no DR**, no temporal structure, no drift, and a toy evaluation.
   - **Delta:** (i) the shadow is the client's **own stale drift report**, and identification is tied to **drift persistence** via the single-crossing Lemma 1; (ii) **DR aggregation of model updates** with the exact bias identity of Proposition 1; (iii) a streaming affine rule.
   - DS-DR is *not* the first use of shadow-variable IPW in FL. FLOSS is.

2. **FedIPW / Who Trains Matters** [morishita2026whotrains] and **FedAU** [wang2024fedau]. IPW aggregation under ignorability (FedIPW) or a fixed p_i (FedAU).
   - **Delta:** DS-DR drops ignorability in favour of drift-mediated selection plus a temporal shadow, and adds the outcome-model leg.
   - Corollary 1(c) gives the precise sense in which FedIPW-type estimators are first-order biased where DS-DR is second-order.

3. **DR with imputation for missing feedback** [wang2019doubly], **AAggFF** [hahn2024aaggff], and **shadow DR in statistics** [miao2016varieties].
   - wang2019doubly applies a DR loss to MNAR recommender data, but relies on correct propensity *or* imputation, with no identification strategy.
   - AAggFF uses DR for unsampled clients' *responses* under a **known uniform** sampling probability (MCAR), for fairness weights only.
   - Miao & Tchetgen Tchetgen supply the statistical machinery.
   - **Delta:** applying it to round-wise FL participation with a *lagged own-state* shadow, a tilted *kernel* outcome model over **current** participant updates, and the affine streaming reduction.

**Also related:** FL-FDMS [wang2022flfdms]. Its friend substitution is the special case of our outcome model with an argmax kernel and no tilt and no reweighting (Corollary 1(e)).

**Verdict: NOVEL as a federated-learning method, INCREMENTAL as statistics.**
- No verified FL work combines MNAR identification of participation with DR aggregation of updates, or uses a client's own persistent stale state as the shadow.
- Every statistical component is known: shadow-variable GMM, AIPW, tilting identities, Nadaraya–Watson.
- Using earlier measurements as shadow variables in longitudinal MNAR may already exist in biostatistics. A search snippet suggested this; it is unverified (LITERATURE_REVIEW, "Unverified"). The paper must therefore claim the *FL transplant, the persistence-based identification condition, and the bias/variance analysis relative to FedAvg/FedIPW/FL-FDMS*, never the estimator itself.
- **Not a breakthrough.** It is a principled, checkable synthesis with an honest boundary (Proposition 3).

**Why no stronger candidate is available.** The alternatives below are either more classical (§3 RCS: double sampling, 1946), more heuristic (§2 DTM), or diagnostic (§4 SB-Agg). DS-DR is the only one whose identification uses the *structure of the drift problem itself*, so it carries the highest novelty-to-risk ratio.

---

## 2. Backup 1 — DTM: Drift-Tilted Memory ("ghost" updates via difference-in-differences)

**Intuition.** MIFA/FedVARP/FedStale reuse an absent client's stale update [gu2021mifa; jhunjhunwala2022fedvarp; rodio2024fedstale], and that update is wrong in the direction of the drift. Correct it using how **drift-similar participants' updates changed** since the absent client last reported. This is a difference-in-differences transfer.

**Formulation.**
- The server stores the stale update Δ_i^{st} = Δ_{i,τ_i} for each client, in fp16 or as a rank-r sketch.
- For absent i:

  g_{i,t} = Δ_i^{st} + [ m̃_t(Z_i, Ŝ_i) − m̃_{τ_i}(Z_i) ],

  where m̃_t(z, s) is the tilted kernel mean of round-t participant updates around stale report z and imputed current drift s (Lemma 2 gives Ŝ_i = E[S | Z_i, R = 0]), and m̃_{τ_i}(Z_i) is the same quantity stored at round τ_i.
- **Use:** as the outcome model m_i inside DS-DR. DTM is an outcome-model upgrade, so DS-DR's propensity leg keeps Corollary 1(a).
- **Identifiability:** inherited from DS-DR.
- **Extra assumption (parallel trends):** absent client i's update would have moved like its drift-matched neighbours since τ_i.

**Pseudo-code.** Same as DS-DR, except the absent-client m_i uses g_i. The rule is no longer an affine combination of current models only, because stale vectors enter.

**Cost.** Memory N × d. In fp32 that is 2.6 GB at N = 100, d = 6.6 M; fp16 halves it; a rank-r sketch costs r × d. **This is the binding constraint on the 16 GB shared machine.**

**Advantages.** Uses the client's own idiosyncratic component, which the kernel mean discards.

**Failure modes.**
- Parallel trends fail at onset.
- Memory.
- Stale updates computed at an old θ (θ-staleness, which FedSteer addresses [zhang2026fedsteer]).

**Code.** `src/methods/dsdr.py` option `outcome="dtm"`, plus a memory store keyed by client.

**Hyperparameters.** Precision, rank r, max age a_max (beyond it, fall back to the kernel mean).

**Novelty verdict: INCREMENTAL / probably novel in detail.**
- Closest: MIFA, FedStale, FedSteer, MimiC [sun2024mimic]. All reuse or correct stale updates without drift conditioning.
- The DiD transfer conditioned on drift-matched neighbours was not found in any verified paper.
- The memory cost and the heuristic parallel-trends assumption make it a backup, not the lead.

---

## 3. Backup 2 — RCS: Recall-Subsample Aggregation (double sampling for FL)

**Intuition.** Instead of modelling selection, *design around it*. Each round the server "recalls" a uniformly random fraction q of non-responders and obtains their update, for example through an incentive, a deadline extension, or deferred upload once the device is back on Wi-Fi.

Among non-responders the recalled set is MCAR by design, so the Hansen–Hurwitz double-sampling estimator [hansen1946nonresponse] is unbiased **with no model of the selection mechanism at all**, even under positivity failure (threshold MNAR).

**Formulation.** With ℛ_t ⊂ [N]∖S_t, each non-responder recalled independently with probability q:

  Δ̂^{RCS}_t = Σ_{j∈S_t} w_j Δ_j + (1/q) Σ_{r∈ℛ_t} w_r Δ_r,  E[Δ̂ | 𝒢] = Δ̄_t exactly.

- **Variance** grows as (1 − q)/q times the non-responders' squared updates. It can be reduced by using DS-DR's m as a control variate:

  Δ̂ = Σ_S w Δ + Σ_{i∉S} w_i m_i + (1/q) Σ_{r∈ℛ} w_r (Δ_r − m_r).

- **Identifiability:** by design. The recalled set also gives an *external validation* of DS-DR's π̂, since the recalled clients' S is observed. This identifies an exponential tilting parameter in the style of Kim & Yu [kim2011semiparametric].

**Simulator change.**
- A new mechanism wrapper: after R_t is drawn, a recall mask with probability q over non-participants is drawn from a **separate** RNG stream, so existing streams stay intact.
- This is a *different participation model* (more information) and must be reported as such.
- Compare at matched *total* reporting rate (0.70 + 0.30·q) against baselines given the same budget of random extra participants.

**Cost.** Recalled clients train (+q × 30% clients per round of compute and communication). No server memory.

**Advantages.** Unbiased under any mechanism, including threshold. Also serves as a gold-standard validation tool for DS-DR.

**Limitations.** Requires a real-world recall lever: incentives [luo2023incentive; wang2026incentive], or deadline extension. It is a design, not a pure algorithm.

**Code.**
- `src/mechanisms/dropout.py`: new function `recall_subsample` (additive).
- Runner: a guarded branch for mechanism suffix `+recall`.
- New strategy `rcs` in `src/methods/rcs.py`.

**Hyperparameters.** q ∈ {0.05, 0.1, 0.2}.

**Novelty verdict: NOVEL in FL as an application, CLASSICAL in statistics.**
- Double sampling of nonrespondents dates to 1946.
- No verified FL paper recalls a random subsample of non-participants to debias aggregation.
- Incentive papers [luo2023incentive; wang2026incentive] reshape participation probabilities but do not use design-based subsampling.
- The unverified "probe channel" proposal (LITERATURE_REVIEW, Unverified) is adjacent.
- Being classical, RCS is a strong *baseline and validation device* but a weak headline.

---

## 4. Alternative — SB-Agg: Sensitivity-Bounded Aggregation (marginal sensitivity model for FL)

**Intuition.** When identification is doubtful (onset, weak shadow, threshold), do not trust a point estimate. Assume the true participation odds are within a factor Λ of a reference propensity π̃, the marginal sensitivity model [tan2006distributional]. Then:
- report sharp bounds on the aggregate's drift-cohort weight share, via linear-fractional programming as in Zhao, Small & Bhattacharya [zhao2019sensitivity];
- optionally aggregate with the minimax-regret weights inside the bounds.

**Formulation.**
- Admissible weights: v_j ∈ [ (1 + Λ^{-1}(1/π̃_j − 1)), (1 + Λ(1/π̃_j − 1)) ].
- Hájek aggregate Σ v_j w_j Δ_j / Σ v_j w_j.
- For a scalar functional (the drift-cohort weight share, or a projection of the update on the drift direction u = Σ_{drift-like} Δ − Σ_{stable} Δ), the extrema over v are attained at threshold-structured v. The extrema are found by sorting, in O(|S| log |S|).
- The aggregate uses the midpoint-in-odds weights, or the minimax-regret v for the projection.
- **Identifiability:** none claimed. This is partial identification.

**Advantages.** Honest under positivity failure. Gives the paper a *robustness certificate* figure.

**Limitations.** For vector updates, minimax is only defined per functional. As an aggregator it is heuristic.

**Code.** `src/methods/sbagg.py`. Its diagnostics can also be computed post hoc from logged π̂.

**Novelty verdict: NOVEL in FL as a diagnostic, INCREMENTAL as an aggregator.**
- Closest: FedeRage CVaR-DRO [rahimi2026federage] and Hashimoto DRO [hashimoto2018fairness]. Both are robust reweightings with no sensitivity model tied to propensity odds.
- No verified FL paper applies the marginal sensitivity model to participation.
- Recommended as an *analysis tool* for the paper, not as a method to beat baselines.

---

## 5. Ablation-level candidate — PS-IPW: Post-Stratified IPW on Stale Drift

- **Rule:** strata from quantiles of Z; Hájek weighting by within-stratum participation rate, π̂_b = |S_t ∩ b| / |b|.
- **Assumption:** MAR given Z. This is the "lag-MAR" approximation that drift persistence makes nearly true after onset.
- **Relation to DS-DR:** it is DS-DR with K = stratum indicator, no tilt, and π̂(Z) instead of π̂(S).
- **Verdict: INCREMENTAL / ALREADY EXISTS in spirit.** It is post-stratification plus FedIPW-type calibration [morishita2026whotrains]. It is kept *only* as the critical ablation: **if PS-IPW matches DS-DR, the shadow machinery does not earn its complexity.** This is a falsification test (EVALUATION_PLAN H-A3).

## 6. Design-level candidate — DDU: Deferred Drift Upload

- **Idea:** clients that would drop during drift buffer their update and upload it later (asynchronously), with staleness correction. This converts permanent missingness into delayed observation.
- **Closest work:** buffered asynchronous FL and staleness reweighting [ma2024fedstaleweight; askin2025fedrevive]. The behaviour-dependent-participation search found no verified FL paper proposing deferred upload *of drift-period data*.
- **Verdict: INCREMENTAL** (async FL plus scheduling).
- It requires a new mechanism and a model of asynchrony that the simulator lacks. **Out of scope for the next phase.** Listed for completeness.

---

## 7. Ranking (expected impact × novelty × feasibility; 1–5 each)

| Rank | Candidate | Impact | Novelty | Feasibility in repo | Product | Role |
|---|---|---|---|---|---|---|
| 1 | **DS-DR** | 4 | 4 (FL) / 2 (stats) | 5 (additive, no memory) | 80 | **Lead** |
| 2 | **RCS** | 4 (unbiased by design) | 3 | 4 (needs a mechanism wrapper) | 48 | Backup and validation device |
| 3 | **DTM** | 3 | 3 | 2 (memory) | 18 | Backup (outcome-model upgrade for DS-DR) |
| 4 | SB-Agg | 2 | 3 | 4 | 24 | Analysis and robustness certificate |
| 5 | PS-IPW | 2 | 1 | 5 | 10 | Mandatory ablation |
| 6 | DDU | 3 | 2 | 1 | 6 | Future work |

SB-Agg's product exceeds DTM's, but SB-Agg is ranked below it because it is not an aggregator to beat baselines with.

**Implement first:**
1. **DS-DR**, with its ablations: DS-IPW, outcome-only, no tilt, lag-MAR/PS-IPW, no gate, oracle-π.
2. **RCS**, as an independent, assumption-free comparator and validation of π̂.
3. Only if pilot memory allows, **DTM** as DS-DR's outcome model.

---

## 8. Stage-0 implementation findings (2026-11-22) — corrections to §1.3, made BEFORE any FEMNIST run

Found by the known-truth testbed (`sim_synthetic/`); each is a genuine methodological issue, not a tuning choice.

1. **Bandwidth must be F_{t−1}-measurable (affects Proposition 1).** §1.3(c) set h from the median pairwise
   distance among *participants*. Then m_i depends on R_i through h, breaking the leave-one-out independence
   that Proposition 1 needs; the Monte-Carlo identity check failed by 5.1%. **Fix:** h from the stale reports
   of *all* clients (fully observed). The identity then holds within Monte-Carlo error
   (`test_proposition1_exact_conditional_bias`).
2. **Round-varying intercept violates A3 in the repo's mechanism.** `dropout.py` re-solves α_t every round
   (mean miss = 0.30) on a per-round-standardised D, so pooling W rounds under one intercept is
   misspecified; the pooled GMM failed to converge in ~1/3 of windows and η diverged (|η| ~ 1e11).
   **Fix:** (i) the clipped-logistic link π = 0.05 + 0.9·σ(·) (the A3 family itself), (ii) round-specific
   intercepts γ_s with a common slope η — still just identified (W intercept moments + p shadow moments),
   (iii) the current round's intercept from the calibration equation Σ_{j∈S_t} 1/π̂_j = N. **Deviation:**
   (iii) uses round-t participation, so π̂ is not exactly F_{t−1}-measurable; the induced term in
   Proposition 1 is O(1/N). Stated, not hidden.
3. **Wald gate suffers the Hauck–Donner effect.** With strong selection the Wald SE explodes and the gate
   opened in only ~16% of strong-MNAR rounds. **Fix:** a score (LM) test of η = 0 built from the same shadow
   moments, evaluated under H0. Smoke result (20 reps): gate open 1.1% under MCAR, 92.5% under MNAR β=4.
4. **Observed, to be reported honestly:** without the gate, DS-DR's outcome model also lowers variance under
   MCAR (final distance 0.060 vs FedAvg 0.074 in the smoke run). This contradicts the literal form of P3
   ("no change under MCAR"); with the gate (the pre-registered configuration) DS-DR ≈ FedAvg under MCAR.
