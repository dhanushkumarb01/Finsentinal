# Literature Review: Client Dropout, Drift, and Informative (MNAR) Participation in Federated Learning

*Phase R1, compiled 2026-11-22.*

## How the entries were verified

Every entry below was checked against its arXiv abstract page or the arXiv API, a DOI record on Crossref, or a proceedings page (PMLR, NeurIPS, ICLR), fetched on 2026-11-22. Four parallel literature passes did this work, and I re-checked the items marked **[R]** myself.

Venue labels:
- **(per arXiv comment)** means the only source for the venue is the authors' own arXiv note.
- **"preprint"** means no peer-reviewed venue was confirmed.
- Items that could not be verified are listed separately under "Unverified" at the end. They are **not** used for any novelty argument and are **not** in `references.bib`.

**What "limitation w.r.t. drift-MNAR" means in each entry.** It is the precise reason the method fails, or is not guaranteed to work, when a client's probability of reporting depends on its own current, temporally evolving drift state. That state is visible only in rounds when the client reports.

---

## 1. Partial participation and dropout compensation

### 1a. Memory / stale-update compensation

**[gu2021mifa]** Gu, Huang, Zhang, Huang. *Fast Federated Learning in the Presence of Arbitrary Device Unavailability.* NeurIPS 2021. arXiv:2106.04159.
- **Idea (MIFA):** the server keeps every device's latest update and applies the average of all N stored updates each round.
- **Assumptions:** unavailability is arbitrary but bounded in duration and does not depend on the data.
- **Limitation w.r.t. drift-MNAR:** the stored update was computed on pre-drift data. If the client left *because* it drifted, the reused update is systematically wrong in the direction of the drift, and nothing corrects it.

**[jhunjhunwala2022fedvarp]** Jhunjhunwala, Sharma, Nagarkatti, Joshi. *FedVARP: Tackling the Variance Due to Partial Client Participation in Federated Learning.* UAI 2022 (per arXiv comment). arXiv:2207.14130.
- **Idea:** SAGA-style correction, Δ = ȳ + (1/|S|)Σ_{i∈S}(Δ_i − y_i), where y_i is client i's stored update.
- **Assumptions:** uniform random sampling.
- **Limitation:** the correction is unbiased only under uniform sampling. Under drift-dependent absence the stored y_i are stale, and the correction term is itself selected.

**[rodio2024fedstale]** Rodio, Neglia. *FedStale: Leveraging Stale Client Updates in Federated Learning.* Preprint 2024. arXiv:2405.04171.
- **Idea:** a convex mix, controlled by β, of fresh updates and stale updates from non-participants. It interpolates between FedAvg and FedVARP.
- **Limitation:** β is a fixed scalar that does not depend on how far an absent client has drifted. The paper itself finds that stale updates help less as participation heterogeneity grows.

**[yan2020fedlaavg]** Yan et al. *Distributed Non-Convex Optimization with Sublinear Speedup under Intermittent Client Availability.* INFORMS J. Computing (DOI 10.1287/ijoc.2022.0057). arXiv:2002.07399.
- **Idea (FedLaAvg):** average the latest gradient of every client.
- **Limitation:** same stale-memory failure as MIFA.

**[zhang2026fedsteer]** Zhang, Pereira, Siew, Liu, Joe-Wong, El-Azouzi. *FedSteer: Taming Extreme Gradient Staleness in Federated Learning with Corrective Projections and Caching.* UAI 2026 (PMLR v337). arXiv:2606.10124.
- **Idea:** stale updates are projected onto a subspace built from the active clients' gradients.
- **Limitation:** a drifted absent client lies *outside* the subspace spanned by the active clients, which is exactly the case the projection cannot represent.

**[askin2025fedrevive]** Askin, Roth, Sun, Joe-Wong, Joshi, Xu. *Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning.* arXiv:2511.00655 (NeurIPS 2026 per arXiv comment).
- **Idea:** data-free distillation of stale models in asynchronous FL.
- **Limitation:** it recovers what the stale model knew, never the client's new (drifted) data.

**[ma2024fedstaleweight]** Ma, Tu, Chen, Reddi. *FedStaleWeight: Buffered Asynchronous FL with Fair Aggregation via Staleness Reweighting.* Preprint. arXiv:2406.02877.
- **Idea:** weights depend only on staleness.
- **Limitation:** blind to drift.

**[sun2024mimic]** Sun, Mao, Zhang. *MimiC: Combating Client Dropouts in Federated Learning by Mimicking Central Updates.* IEEE TMC (DOI 10.1109/TMC.2023.3338021). arXiv:2306.12212.
- **Idea:** received updates are corrected with a variable built from previous updates so that the aggregate mimics full participation.
- **Limitation:** present drift of absent clients is not represented.

**[wang2022flfdms]** Wang, Xu. *Combating Client Dropout in Federated Learning via Friend Model Substitution* (earlier title "Friends to Help"). Preprint; no peer-reviewed venue confirmed. arXiv:2205.13222.
- **Idea:** the update of a dropped client is replaced by the current update of its most similar "friend", where similarity is a running mean of update cosine similarity.
- **Assumptions:** dropout is exogenous.
- **Limitation:** similarity is measured *before* the dropout. A client that drifted and then dropped keeps friends that matched its pre-drift data. This is the repo's FL-FDMS baseline.

**[stepka2025persistent]** Stępka, Gisolfi, Trębacz, Dubrawski. *Mitigating Persistent Client Dropout in Asynchronous Decentralized FL.* FedKDD @ KDD 2025 (workshop). arXiv:2508.01807.
- **Idea:** reconstruct lost clients by model/gradient inversion.
- **Limitation:** reconstructs pre-dropout data.

**[lee2025kpfl]** Lee et al. *Dynamic Participation in Federated Learning: Benchmarks and a Knowledge Pool Plugin.* Preprint. arXiv:2511.16523.
- **Idea:** participation generators (timed-random, Markov, programmed) that are explicitly data-agnostic, plus the KPFL knowledge pool.
- **Limitation:** the pool goes stale under drift. The paper names data-dependent participation as future work.

**[jiang2024fedar]** FedAR (approximation and rectification of unavailable clients' updates), ECML 2024. arXiv:2407.19103.
- Identified by the MNAR literature pass and checked only at the arXiv-ID level, so its title is not reproduced here.
- **Idea:** substitute the latest observed update, then rectify it.
- **Limitation:** no missingness model.

### 1b. Reweighting, unknown statistics, Markov and periodic availability

**[wang2024fedau]** Wang, Ji. *A Lightweight Method for Tackling Unknown Participation Statistics in Federated Averaging.* ICLR 2024 (per arXiv comment). arXiv:2306.03401.
- **Idea (FedAU):** online inverse-participation-rate weighting estimated from the gaps between participations.
- **Assumptions:** each client participates Bernoulli(p_i) with p_i *fixed* and independent of the data.
- **Limitation:** under drift-MNAR, p_i changes exactly when the client drifts, so a history-based estimate lags and is confounded. FedAU and the repo's FedIPW-estimated (EMA of history) are the same family.

**[wang2022unified]** Wang, Ji. *A Unified Analysis of Federated Learning with Arbitrary Client Participation.* NeurIPS 2022 (per arXiv comment). arXiv:2205.13648.
- **Idea:** generalized/amplified FedAvg analysed under arbitrary participation patterns.
- **Limitation:** the participation pattern is assumed independent of the data and of the optimisation trajectory.

**[xiang2024fedawe]** Xiang, Ioannidis, Yeh, Joe-Wong, Su. *Efficient Federated Learning against Heterogeneous and Non-stationary Client Unavailability.* NeurIPS 2024 (proceedings confirmed). arXiv:2409.17446.
- Called FedAWE in the proceedings and **FedAPM** in arXiv v2. They are the same paper and are cited once.
- **Idea:** compensates for missed rounds and uses implicit gossip.
- **Limitation:** availability is non-stationary but exogenous, and the compensation assumes the absent client's objective did not change while it was away.

**[xiang2026fedswe]** Xiang et al. *Resilience Beyond Stationary Client Unavailability: Unlocking Efficient and Unbiased Federated Learning.* JMLR 2026 (per arXiv comment). arXiv:2609.04763.
- Journal extension of FedAWE. Same limitation.

**[xiang2024fedpbc]** Xiang et al. *Empowering FL with Implicit Gossiping: Mitigating Connection Unreliability Amidst Unknown and Arbitrary Dynamics.* Preprint (extension of a CDC 2023 paper). arXiv:2404.10091.
- **Limitation:** link failures are assumed independent of the data.

**[rodio2023cafed]** Rodio, Faticanti, Marfoq, Neglia, Leonardi. *Federated Learning under Heterogeneous and Correlated Client Availability.* IEEE INFOCOM 2023. arXiv:2301.04632.
- **Idea:** Markov availability that is correlated in time and across clients. CA-Fed reweights clients and may *exclude* clients with low availability and high correlation.
- **Limitation:** availability is independent of the data, and excluding rarely available clients *amplifies* drift-MNAR bias.

**[sun2025debiasing]** Sun, Zhang, Xu, Joshi, Sharma, Wei. *Debiasing Federated Learning with Correlated Client Participation.* ICLR 2025. arXiv:2410.01209.
- **Idea:** Markov participation with a minimum gap between participations, and a debiased FedAvg using estimated probabilities.
- **Limitation:** the chain is exogenous and stationary.

**[ribero2023f3ast]** Ribero, Vikalo, de Veciana. *Federated Learning Under Intermittent Client Availability and Time-Varying Communication Constraints.* IEEE JSTSP (DOI 10.1109/JSTSP.2022.3224590). arXiv:2205.06730.
- **Idea (F3AST):** learns an availability-aware selection policy.
- **Limitation:** availability is independent of the data.

**[crawshaw2024periodic]** Crawshaw, Liu. *Federated Learning under Periodic Client Participation and Heterogeneous Data.* NeurIPS 2024. arXiv:2410.23131.
- **Limitation:** assumes equal participation probability within a window, which drift-MNAR directly violates.

**[malinovsky2023regularized]** Malinovsky, Horváth, Burlachenko, Richtárik. *Federated Learning with Regularized Client Participation.* Preprint. arXiv:2302.03662.
- **Limitation:** the participation schedule is enforced by the server, which is impossible when clients self-select out.

**[yang2022anarchic]** Yang, Zhang, Khanduri, Liu. *Anarchic Federated Learning.* ICML 2022 (per arXiv comment). arXiv:2108.09875.
- **Idea:** workers choose when to participate.
- **Limitation:** analysed with participation independent of the data.

**[ruan2021flexible]** Ruan, Zhang, Liang, Joe-Wong. *Towards Flexible Device Participation in Federated Learning.* AISTATS 2021. arXiv:2006.06954.
- **Limitation:** participation probabilities are assumed known or exogenous.

**[yang2024safari]** Yang, Qiu, Khanduri, Fang, Liu. *Understanding Server-Assisted Federated Learning in the Presence of Incomplete Client Participation.* ICML 2024 (per arXiv comment). arXiv:2405.02745.
- **Idea:** shows that FL with incomplete participation is not PAC-learnable in the worst case, and fixes it with a server-side representative dataset.
- **Relevance:** this is a key *negative* result. Some side information is necessary. A static server dataset cannot track drift.

**[rahimi2025agnostic]** Rahimi, Kalogerias. *Convergence of Agnostic Federated Averaging.* CAMSAP 2025 (per arXiv comment). arXiv:2507.10325.
- Plain FedAvg under unknown, possibly biased participation probabilities.
- The arXiv author field is malformed; check the names against the PDF before final citation.

**[rahimi2026federage]** Rahimi, Kalogerias. *FedeRage: Provably Convergent Agnostic Federated Learning under General Client Drift.* Preprint. arXiv:2609.21057.
- **Idea:** CVaR-DRO up-weights high-loss and rarely participating clients.
- **Limitation:** a robust hedge that never models the propensity. Here "client drift" means heterogeneity, not temporal drift.

**[su2024adversarial]** Su, Xiang, Xu, Yang. *Federated Learning in the Presence of Adversarial Client Unavailability.* Preprint. arXiv:2305.19971.
- **Result:** an adversary silencing an ε fraction of clients causes error Θ(ε(G² + σ²)).
- **Relevance:** worst-case ceiling for drift-MNAR (an adversary may silence exactly the drifted clients).

**[hewa2026dflaa]** Hewa Kaluannakkage, Buyya. *Asynchronous Decentralized Federated Learning over Lossy Wireless Links via Reception- and Age-Aware Aggregation.* Preprint. arXiv:2606.10774.
- **Idea:** IPW weighting with an empirical-frequency reception probability, multiplied by an age-of-information decay.
- **Limitation:** valid only because packet loss is channel-driven (independent of the data). The repo's FedIPW-estimated uses the same EMA primitive.

### 1c. Client sampling / selection (server-controlled)

**[cho2022poc]** Cho, Wang, Joshi. *Towards Understanding Biased Client Selection in Federated Learning.* AISTATS 2022. arXiv:2010.01243.
- **Idea:** Power-of-Choice, i.e. loss-biased selection.
- **Relevance:** gives the solution bias caused by *loss-dependent selection*, the mirror image of loss-dependent dropout.

**[fraboni2021clustered]** Fraboni, Vidal, Kameni, Lorenzi. *Clustered Sampling.* ICML 2021. arXiv:2105.05883.
- **Idea:** stratified unbiased sampling.
- **Limitation:** the server controls sampling, so a sampled client that refuses breaks it.

**[shen2025fedeve]** Shen et al. *FedEve: On Bridging the Client Drift and Period Drift for Cross-device FL.* Preprint. arXiv:2508.14539.
- "Period drift" (the sampled cohort deviating from the population) is treated as variance, not as systematic selection bias.

**[yuan2022generalization]** Yuan, Morningstar, Ning, Singhal. *What Do We Mean by Generalization in Federated Learning?* ICLR 2022. arXiv:2110.14216.
- Defines the "participation gap". Useful framing.

---

## 2. Missing-data theory: IPW, doubly robust estimation, MNAR identification

**[rubin1976inference]** Rubin. *Inference and Missing Data.* Biometrika 63(3):581–592, 1976. DOI 10.1093/biomet/63.3.581.
- Defines MCAR/MAR/MNAR. Drift-dependent dropout, where the client's drift is observed only when it reports, is MNAR.

**[heckman1979sample]** Heckman. *Sample Selection Bias as a Specification Error.* Econometrica 47(1):153–161, 1979. DOI 10.2307/1912352.
- Selection equation plus inverse-Mills-ratio correction. Needs normality or an exclusion restriction.

**[diggle1994informative]** Diggle, Kenward. *Informative Drop-out in Longitudinal Data Analysis.* JRSS-C 43(1):49–93, 1994. DOI 10.2307/2986113.
- Dropout hazard depends on the current, unobserved outcome. Identification is parametric and fragile. This is the closest classical analogue.

**[little1993pattern]** Little. *Pattern-Mixture Models for Multivariate Incomplete Data.* JASA 88(421):125–134, 1993. DOI 10.1080/01621459.1993.10594302.

**[robins1994estimation]** Robins, Rotnitzky, Zhao. *Estimation of Regression Coefficients When Some Regressors Are Not Always Observed.* JASA 89(427):846–866, 1994. DOI 10.1080/01621459.1994.10476818.
- Augmented IPW (AIPW) estimating equations.

**[bang2005doubly]** Bang, Robins. *Doubly Robust Estimation in Missing Data and Causal Inference Models.* Biometrics 61(4):962–973, 2005. DOI 10.1111/j.1541-0420.2005.00377.x.
- **Estimator:** μ̂ = n⁻¹Σ[RY/π̂ + (1 − R/π̂) m̂(X)].
- **Assumptions:** MAR.

**[kang2007demystifying]** Kang, Schafer. *Demystifying Double Robustness.* Statistical Science 22(4):523–539, 2007. DOI 10.1214/07-STS227.
- DR can do badly when both models are mildly wrong and propensities are extreme. This is a design constraint for us, since drifted clients can have π as low as 0.05.

**[rotnitzky1998semiparametric]** Rotnitzky, Robins, Scharfstein. *Semiparametric Regression for Repeated Outcomes with Nonignorable Nonresponse.* JASA 93(444):1321–1339, 1998. DOI 10.1080/01621459.1998.10473795.

**[scharfstein1999adjusting]** Scharfstein, Rotnitzky, Robins. *Adjusting for Nonignorable Drop-out Using Semiparametric Nonresponse Models.* JASA 94(448):1096–1120, 1999. DOI 10.1080/01621459.1999.10473862.
- Exponential-tilting sensitivity analysis.

**[miao2016varieties]** Miao, Tchetgen Tchetgen. *On Varieties of Doubly Robust Estimators under Missingness Not at Random with a Shadow Variable.* Biometrika 103(2):475–482, 2016. DOI 10.1093/biomet/asw016.
- A shadow variable Z satisfies Z ⫫ R | (X, Y) and is associated with Y. It identifies the selection odds ratio, and DR estimators follow. **This is the identification engine of our lead method.**

**[miao2024identification]** Miao, Liu, Li, Tchetgen Tchetgen, Geng. *Identification and Semiparametric Efficiency Theory of Nonignorable Missing Data with a Shadow Variable.* ACM/IMS J. Data Science 1(2), 2024. DOI 10.1145/3592389. arXiv:1509.02556.
- Completeness condition and efficient influence function.

**[wang2014instrumental]** Wang, Shao, Kim. *An Instrumental Variable Approach for Identification and Estimation with Nonignorable Nonresponse.* Statistica Sinica 24, 2014. DOI 10.5705/ss.2012.074.
- Parametric propensity identified by a "nonresponse instrument" (shadow), estimated by GMM.

**[tchetgen2017general]** Tchetgen Tchetgen, Wirth. *A General Instrumental Variable Framework for Regression Analysis with Outcome Missing Not at Random.* Biometrics 73(4):1123–1131, 2017. DOI 10.1111/biom.12670.

**[sun2018semiparametric]** Sun, Liu, Miao, Wirth, Robins, Tchetgen Tchetgen. *Semiparametric Estimation with Data Missing Not at Random Using an Instrumental Variable.* Statistica Sinica 28, 2018. DOI 10.5705/ss.202016.0324.

**[mohan2021graphical]** Mohan, Pearl. *Graphical Models for Processing Missing Data.* JASA 116(534):1023–1037, 2021. DOI 10.1080/01621459.2021.1874961.
- Self-masking (Y → R_Y) is not recoverable without extra structure.

**[li2023selfcensoring]** Li, Miao, Shpitser, Tchetgen Tchetgen. *A Self-Censoring Model for Multivariate Nonignorable Nonmonotone Missing Data.* Biometrics 79(4):3203–3214, 2023. DOI 10.1111/biom.13916.

**[malinsky2022semiparametric]** Malinsky, Shpitser, Tchetgen Tchetgen. *Semiparametric Inference for Nonmonotone MNAR Data: The No Self-Censoring Model.* JASA 117(539):1415–1423, 2022. DOI 10.1080/01621459.2020.1862669.

**[miao2016identifiability]** Miao, Ding, Geng. *Identifiability of Normal and Normal Mixture Models with Nonignorable Missing Data.* JASA 111(516):1673–1683, 2016. DOI 10.1080/01621459.2015.1105808.

**[kim2011semiparametric]** Kim, Yu. *A Semiparametric Estimation of Mean Functionals with Nonignorable Missing Data.* JASA 106(493):157–165, 2011. DOI 10.1198/jasa.2011.tm10104.
- The tilting parameter is identified from a *follow-up subsample* of nonrespondents.

**[hansen1946nonresponse] [R]** Hansen, Hurwitz. *The Problem of Non-Response in Sample Surveys.* JASA 41(236):517–529, 1946. DOI 10.1080/01621459.1946.10501894.
- Double sampling: a random subsample of nonrespondents is followed up. It gives unbiased estimation under arbitrary nonresponse.

**[tan2006distributional] [R]** Tan. *A Distributional Approach for Causal Inference Using Propensity Scores.* JASA 101(476):1619–1637, 2006. DOI 10.1198/016214506000000023.
- Source of the marginal sensitivity model, which bounds the propensity odds ratio by Λ.

**[zhao2019sensitivity] [R]** Zhao, Small, Bhattacharya. *Sensitivity Analysis for Inverse Probability Weighting Estimators via the Percentile Bootstrap.* JRSS-B 81(4):735–761, 2019. DOI 10.1111/rssb.12327.
- Sharp bounds on IPW under the marginal sensitivity model, computed by linear-fractional programming.

**[nabi2026causal]** Nabi, Bhattacharya, Shpitser, Robins. *Causal and Counterfactual Views of Missing Data Models.* Statistica Sinica (discussion paper), 2026. DOI 10.5705/ss.202023.0382.

**[robins2000marginal]** Robins, Hernán, Brumback. *Marginal Structural Models and Causal Inference in Epidemiology.* Epidemiology 11(5):550–560, 2000. DOI 10.1097/00001648-200009000-00011.
- Time-varying stabilised weights (sequential ignorability, MAR-type).

**[robins2000correcting]** Robins, Finkelstein. *Correcting for Noncompliance and Dependent Censoring in an AIDS Clinical Trial with IPCW Log-Rank Tests.* Biometrics 56(3):779–788, 2000. DOI 10.1111/j.0006-341X.2000.00779.x.

### 2b. ML transplants of IPW and DR under MNAR

**[schnabel2016recommendations]** Schnabel, Swaminathan, Singh, Chandak, Joachims. *Recommendations as Treatments: Debiasing Learning and Evaluation.* ICML 2016. arXiv:1602.05352.
- IPS-weighted ERM.

**[wang2019doubly]** Wang, Zhang, Sun, Qi. *Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random.* ICML 2019, PMLR 97:6638–6647.
- DR loss with a jointly learned imputation model. **This is the closest structural template for DR aggregation of gradients**, but its unbiasedness needs a correct propensity *or* a correct imputation. It has no identification strategy under MNAR.

**[ipsen2021notmiwae]** Ipsen, Mattei, Frellsen. *not-MIWAE: Deep Generative Modelling with Missing Not at Random Data.* ICLR 2021. arXiv:2006.12871.

### 2c. Federated learning and federated causal inference

**[morishita2026whotrains]** Morishita. *Who Trains Matters: Federated Learning under Enrollment and Participation Selection Biases.* Preprint 2026. arXiv:2604.26604. **(FedIPW)**
- **Estimator:** Δ̄_r = (1/N)Σ_{A=1} Δ_{i,r}/p̂_{i,r}, with p̂ = π̂^enroll(Z) · π̂^part(Z, X_r).
- **Assumptions:** round ignorability E[Δ | Z, X_r, A=1] = E[Δ | Z, X_r], plus positivity. The covariates X_r are pre-round system covariates (battery, connectivity, time, history).
- **Theory:** shows a non-vanishing bias floor under residual weighting error (Corollary 6.2).
- **Experiments:** synthetic logistic regression only.
- **Limitation:** ignorability fails when participation depends on the current update or drift.

**[goetze2025floss]** Goetze, Felten, Albrecht, Bhattacharya. *FLOSS: Federated Learning with Opt-Out and Straggler Support.* arXiv:2507.23115 (v2 Apr 2026; PACMI '25 workshop per the earlier project audit).
- **Mechanism:** MNAR opt-out mediated by user satisfaction S: R depends on (X, Y) only through S.
- **Correction:** IPW sampling with π = p(R=1 | D′, S). π is identified with a **static shadow variable** Z ⊂ D (e.g. device processing power), with Z ⫫ R | S, D′, by solving the moments E[(R/p_β − 1)f(D′, Z)] = 0.
- **What it does not have:** no DR component, no outcome model, no temporal dynamics or drift, and a toy evaluation.
- **Why it matters:** FLOSS is the only FL paper with formal MNAR shadow identification and **the single closest prior work to our lead method.**

**[vazquez2026incomplete]** Vazquez et al. *Federated Learning with Incomplete Data: When to Use Complete Cases and When to Weight.* Preprint. arXiv:2605.20125.
- Multiply robust weights under MAR for records missing *within* sites. A different unit of missingness.

**[xiong2023federated]** Xiong, Koenecke, Powell, Shen, Vogelstein, Athey. *Federated Causal Inference in Heterogeneous Observational Data.* Statistics in Medicine 42(24):4418–4439, 2023. DOI 10.1002/sim.9868.
- Federated AIPW for treatment effects.

**[han2025face]** Han, Hou, Cho, Duan, Cai. *Federated Adaptive Causal Estimation of Target Treatment Effects.* JASA 120(551):1503–1516, 2025. DOI 10.1080/01621459.2025.2453249.

**[khellaf2025federated]** Khellaf, Bellet, Josse. *Federated Causal Inference from Multi-Site Observational Data via Propensity Score Aggregation.* Preprint. arXiv:2505.17961.

**[vo2022bayesian]** Vo, Lee, Hoang, Leong. *Bayesian Federated Estimation of Causal Effects from Observational Data.* UAI 2022, PMLR 180. arXiv:2106.00456.

**[hahn2024aaggff] [R]** Hahn, Kim, Lee. *Pursuing Overall Welfare in Federated Learning through Sequential Decision Making.* ICML 2024. arXiv:2405.20821.
- **Verified in the full text (Lemma 4.3, Eq. 10):** a DR estimator for the responses of *unsampled* clients, r̆_i = (1 − 1{i∈S}/C) r̄ + (1{i∈S}/C) r_i, with C a **known uniform** sampling probability.
- It is used only to update fairness mixing coefficients, never the model update.
- **The only "DR for unobserved clients in FL" hit. It is MCAR by design.**

---

## 3. Concept and data drift in FL

**[jothimurugesan2023feddrift]** Jothimurugesan, Hsieh, Wang, Joshi, Gibbons. *Federated Learning under Distributed Concept Drift.* AISTATS 2023. arXiv:2206.00799.
- **Idea:** multi-model clustering with a client-side loss-jump drift test.
- **Assumptions:** full participation; the paper states clients are "stateful and participating at each round".
- **Limitation:** drift is detected only when the client reports, so absent drifted clients are invisible. The repo's FedDrift baseline adapts it to partial participation (F2–F3).

**[panchal2023flash]** Panchal et al. *Flash: Concept Drift Adaptation in Federated Learning.* ICML 2023, PMLR 202:26931–26962.
- Drift is inferred from the updates that arrive, so silent drifted clients produce no signal.

**[canonaco2021adaptive]** Canonaco, Bergamasco, Mongelluzzo, Roveri. *Adaptive Federated Learning in Presence of Concept Drift.* IJCNN 2021. DOI 10.1109/IJCNN52387.2021.9533710.

**[casado2022cdafedavg]** Casado et al. *Concept Drift Detection and Adaptation for Federated and Continual Learning.* Multimedia Tools and Applications 81(3):3397–3419, 2022. DOI 10.1007/s11042-021-11219-x.
- In its experiments every client participates every round.

**[chen2024fedccfa]** Chen, Xue, Wang, Liu, Huang. *Classifier Clustering and Feature Alignment for Federated Learning under Distributed Concept Drift.* NeurIPS 2024. arXiv:2410.18478.
- Targets real P(Y|X) drift. Sampling is uniform.

**[ganguly2022masterfl]** Ganguly, Aggarwal. *Online Federated Learning via Non-Stationary Detection and Adaptation amidst Concept Drift.* Preprint. arXiv:2211.12578.

**[guo2021timeevolving]** Guo, Lin, Tang. *Towards Federated Learning on Time-Evolving Heterogeneous Data.* Preprint. arXiv:2112.13246.
- Clients may join or leave, but exogenously.

**[peng2025feddaa]** Peng, Tang. *FedDAA: Dynamic Client Clustering for Concept Drift Adaptation in FL.* Preprint. arXiv:2506.21054.

**[li2024fielding]** Li et al. *Fielding: Clustered Federated Learning with Data Drift.* Preprint. arXiv:2411.01580.

**[thomas2025adaption]** Thomas, Saile, Fischer, Kaaser, Schulte. *Adaption via Selection: On Client Selection to Counter Concept Drift in FL.* ESOCC 2025, LNCS. DOI 10.1007/978-3-031-84617-5_1.
- Server-side selection biased toward high-loss (drifted) clients. It assumes those clients are *reachable*, the opposite of our setting.

**[fang2025cafe]** Fang, Wu, Yang, Wu, Hu. *Drift-Aware Federated Learning: A Causal Perspective.* Preprint. arXiv:2503.09116.
- **Reversed causal arrow:** participation frequency causes update drift.

**[patel2025flmaldrift]** Patel, Tomar, Pateriya, Haripriya. *FL-MalDrift.* Scientific Reports 16:1821, 2025. DOI 10.1038/s41598-025-31592-z.
- Drift scores are used to *exclude* updates, which creates drift-dependent exclusion by design. No correction.
- Method details come from the abstract only.

**[garg2026felix]** Garg et al. *Robust Federated Learning Under Real-World Client Churn (FeLiX).* Preprint. arXiv:2607.06979.
- Names churn and drift but treats them as separate problems.

**[li2026feroma]** Li, Fenoglio, Gjoreski, Langheinrich. *Federated Learning with Profile Mapping under Distribution Shifts and Drifts.* ICLR 2026. arXiv:2602.07671.

**Clustered and multi-model FL (static clusters, random participation):**
- **[ghosh2020ifca]** IFCA, NeurIPS 2020, arXiv:2006.04088.
- **[marfoq2021fedem]** FedEM, NeurIPS 2021, arXiv:2108.10252.
- **[sattler2021cfl]** CFL, IEEE TNNLS 32(8), 2021, DOI 10.1109/TNNLS.2020.3015958.
- **[guo2024fedrc]** FedRC, ICML 2024, arXiv:2301.12379.

**Continual FL:**
- **[yoon2021fedweit]** FedWeIT, ICML 2021, arXiv:2003.03196.
- **[hamedi2025fclsurvey]** Hamedi, Razavi-Far, Hallaji, Neurocomputing 651:130844, 2025.

**Shift detection and label shift:**
- **[rabanser2019failing]** Failing Loudly, NeurIPS 2019, arXiv:1810.11953.
- **[gretton2012mmd]** Kernel two-sample test, JMLR 13:723–773, 2012.
- **[lipton2018bbse]** BBSE, ICML 2018, arXiv:1802.03916.
- **[wu2025vrls]** VRLS, ICLR 2025, arXiv:2502.02544.
- **[listozec2024fedpals]** FedPALS, preprint, arXiv:2411.03799.
- All of these need samples from the shifted distribution, which an absent client never sends.

**Distinction:** **[karimireddy2020scaffold]** SCAFFOLD (ICML 2020, arXiv:1910.06378) concerns optimisation "client drift". That is not temporal data drift.

---

## 4. Behaviour-, loss- and state-dependent participation

**[hashimoto2018fairness]** Hashimoto, Srivastava, Namkoong, Liang. *Fairness Without Demographics in Repeated Loss Minimization.* ICML 2018. arXiv:1806.08010.
- User retention is a function of loss, which causes disparity amplification. The fix is DRO. **This is the canonical centralised template for drift → loss → departure.**

**[zhang2019group]** Zhang, Khalili, Tekin, Liu. *Group Retention when Using Machine Learning in Sequential Decision Making.* NeurIPS 2019. arXiv:1905.00569.

**[perdomo2020performative]** Perdomo, Zrnic, Mendler-Dünner, Hardt. *Performative Prediction.* ICML 2020. arXiv:2002.06673.

**[dean2024emergent]** Dean, Curmei, Ratliff, Morgenstern, Fazel. *Emergent Specialization from Participation Dynamics and Multi-Learner Retraining.* AISTATS 2024. arXiv:2206.02667.

**[ginart2021competing]** Ginart, Zhang, Kwon, Zou. *Competing AI.* AISTATS 2021, PMLR 130:1693–1701.

**[jin2024performative]** Jin et al. *Performative Federated Learning.* arXiv:2305.05090 (AAAI 2024 per DOI found in search; the AAAI page itself was not fetched).
- The data responds to the model, but participation is random.

**[yashwanth2026feduca]** Yashwanth, Singh, Nayak, Bulusu, Chakraborty. *Federated Learning by Utility-Constrained Stochastic Aggregation for Improving Rational Participation.* arXiv:2605.18020 (NeurIPS 2026 per arXiv comment).
- Clients exit when the global model's utility falls below their local utility, so participation is endogenous. The paper *prevents* exits through aggregation rather than *correcting* for them. **This is the closest endogenous-participation work.**

**[cho2022maxfl]** Cho, Jhunjhunwala, Li, Smith, Joshi. *Maximizing Global Model Appeal in Federated Learning.* Preprint. arXiv:2205.14840.

**[donahue2021model]** Donahue, Kleinberg. *Model-Sharing Games.* AAAI 2021. arXiv:2010.00753.

**[blum2021one]** Blum, Haghtalab, Phillips, Shao. *One for One, or All for All.* ICML 2021. arXiv:2103.03228.

**[karimireddy2022mechanisms]** Karimireddy, Guo, Jordan. *Mechanisms that Incentivize Data Sharing in FL.* Preprint. arXiv:2207.04557.

**[luo2023incentive]** Luo, Feng, Wang, Huang, Tassiulas. *Incentive Mechanism Design for Unbiased FL with Randomized Client Participation.* ICDCS 2023. arXiv:2304.07981.
- Pricing shapes participation probabilities.

**[wang2026incentive]** Wang, Luo, Tang. *An Incentive Mechanism for FL with Time-Varying Client Availability.* IEEE TMC 25(1):284–299, 2026. DOI 10.1109/TMC.2025.3592202.

**[li2020fair]** Li, Sanjabi, Beirami, Smith. q-FFL. ICLR 2020. arXiv:1905.10497.

**[okumus2026typebased]** Okumus et al. *Type-Based Unsourced FL with Client Self-Selection.* IEEE ICC 2026 (per arXiv comment). arXiv:2602.06601.
- Loss-threshold self-selection by design. No debiasing.

**[behfar2026cumulative]** Behfar, Mortier. *Cumulative Utility Parity for Fair FL under Intermittent Participation.* Preprint. arXiv:2602.13651.

---

## 5. Synthesis table

Column meanings:
- **P depends on data/state?** Does the participation model allow dependence on the client's data or state?
- **Uses absent-client info?** Does the method use any information about absent clients?
- **Valid under drift-MNAR?** Is the correction valid when dropout depends on the client's current drift?

| Work | Year/venue | P depends on data/state? | Correction type | Uses absent-client info? | Valid under drift-MNAR? | Exact gap left |
|---|---|---|---|---|---|---|
| MIFA / FedLaAvg | NeurIPS'21 / IJOC | No | stale memory | own stale update | No: stale update is drift-biased | drift-corrected memory |
| FedVARP / FedStale | UAI'22 / '24 | No | variance reduction | stale updates | No | selection-aware control variate |
| FedSteer | UAI'26 | No | subspace projection | cached gradients | No: drifted client is off-subspace | — |
| MimiC | TMC | No | correction variable | past updates | No | — |
| FL-FDMS | preprint | No (exogenous) | friend substitution | pre-drop similarity | No: stale friends | drift-aware friend choice |
| FedAU / FedIPW-est (repo) | ICLR'24 | No (fixed p_i) | IPW from history | participation history | No: lagged, confounded | current-state propensity |
| FedAWE / FedSWE | NeurIPS'24 / JMLR'26 | Non-stationary, exogenous | missed-round compensation | round gaps | No | — |
| CA-Fed / Sun'25 | INFOCOM'23 / ICLR'25 | Markov, exogenous | reweighting | availability estimates | No: may amplify bias | — |
| FedIPW (Morishita) | preprint'26 | MAR on system covariates | IPW (+ calibration) | covariates of all clients | No: ignorability violated (bias floor) | MNAR identification |
| **FLOSS** | preprint / workshop'25 | **Yes, MNAR via satisfaction** | IPW, static shadow variable | static covariate Z | Only if a static shadow exists and selection is satisfaction-mediated | temporal shadow, DR, outcome model, drift |
| AAggFF | ICML'24 | No (known uniform C) | DR for responses of unsampled clients | mean response | MCAR only; fairness weights only | DR for model updates under MNAR |
| SAFARI | ICML'24 | Some never participate | server dataset | server data | Static data cannot track drift | — |
| FedDrift / Flash / FedCCFA | AISTATS'23 / ICML'23 / NeurIPS'24 | Full or random participation | clustering / adaptive LR | — | Drift unseen for absent clients | — |
| Adaption via Selection | ESOCC'25 | Server selects high-loss clients | selection | — | Assumes drifted clients are reachable | — |
| FL-MalDrift | Sci. Rep.'25 | Drift-based *exclusion* by design | — | — | Creates the bias, does not correct it | — |
| FedUCA / MaxFL | NeurIPS'26 / preprint | Utility-driven exits | prevent exits | — | No bias correction | — |
| Hashimoto'18 / Zhang'19 | ICML / NeurIPS | Loss-driven retention (centralised) | DRO | — | Not FL; no propensity | — |
| Su et al. | preprint | Adversarial ε | bound | — | Worst-case only | structure-exploiting correction |
| DFL-AA | preprint'26 | No (channel) | EMA-IPW × AoI decay | reception history | No | — |

## 6. Synthesis: what the literature does and does not do

1. **Participation models.** Every participation model in the dropout and partial-participation literature is either exogenous (random, Markov, periodic, adversarial-but-data-blind, non-stationary) or MAR on observed system covariates. FedIPW is the strongest of these. **FLOSS** is the only exception that models MNAR and identifies it, and it does so with a *static* shadow variable and IPW only.
2. **Memory-based compensation.** MIFA, FedVARP, FedStale, FedLaAvg, FedSteer, MimiC and KPFL all assume that an absent client's past contribution is still a good stand-in. That assumption fails exactly when absence is *caused by* change.
3. **Drift methods.** FedDrift, Flash, FedCCFA, CFL and Fielding detect drift only from clients that report. A drifted, absent client is invisible to them, which is the blind spot drift-MNAR exploits.
4. **Doubly robust estimation in FL.** It exists only for (a) treatment-effect estimation across sites and (b) AAggFF's MCAR fairness weights. No verified work applies AIPW/DR to *model-update aggregation* under non-random participation. None exploits the *temporal persistence* of a client's own previously reported state as a shadow variable for its current participation.
5. **Statistics.** All the ingredients exist: shadow-variable identification, DR estimators, tilting sensitivity analysis, and double sampling. Their combination for longitudinal, round-wise client participation with drift has not been done in FL. **Any novelty claim must therefore be stated as a transplant plus a new structural observation, not as new statistics.**

---

## Unverified (not cited as evidence, not in references.bib)

- "Lacuna" web page *Two-Channel FL for Outcome-Dependent Client Dropout*. It describes an unpublished research proposal for a cheap probe channel (a noised label histogram plus loss) that would model participation for IPW reweighting. It was seen by a sub-agent and **not independently verified, nor peer-reviewed**.
  - Its probe-channel idea overlaps with candidate 4 (RCS) and with the reporting assumption of the lead method. The lead method differs because it needs no extra channel from absent clients.
- Lian, Jiang, Long (2024), medRxiv 10.1101/2024.09.15.24313479: a federated Heckman multiple imputation for MNAR *variables within sites*. Crossref metadata only; full text returned 403.
- DriftGuard (Han, Wu, Varghese, IEEE IoT-J 2026, DOI 10.1109/JIOT.2026.3749783): metadata only. **Should be read before submission.**
- FedCausal-Dyn (arXiv:2607.09695): the author list looks suspicious. Excluded.
- Wang, Xu, Lu, Song, *Off-Policy Evaluation Under Nonignorable Missing Data* (arXiv:2507.06961): the abstract was verified, but it does **not** state that lagged measurements are used as shadow variables. A search snippet suggested they are. Whether earlier measurements have been used as shadow variables in longitudinal MNAR work is therefore **open**, and the novelty verdict treats it as possibly known.
