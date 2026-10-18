# Research Protocol V12 (PRT-2026-006)

> Post-audit implementation note: repairs do not add a new search. The historical scalar grid contained 30 configurations against a limit of 24; this deviation remains disclosed. Corrected replay retains prior selection and is not a new confirmation. [Current status](results-v12.md) and [replay instructions](v12-repair-replay.md).
## Craving Engine: Diagnosing Personalization Weaknesses, Bounded Compact Method Search, and Confirmatory Testing

**Version:** 1.0 (Pre-registered prior to experimental execution)  
**Registration Date:** 2026-10-17  
**Governance Protocol:** [PRT-2026-006](https://gov.nutrifit.health/documents/6ad2c396556a2264e7a1339e)  
**Governance Task:** [TASK-2026-0796](https://gov.nutrifit.health/tasks/6ad2c46b556a2264e7a133a8)  
**Repository:** `/Users/gurutom/Develop/NutriFit/apps/craving-engine`

---

## 1. Introduction and Objectives

In the V11 research iteration (verified in `runs/v11-closure-20261016`):
- Hypothesis H1 (superiority of linear contextual personalization over scalar baseline) was not supported: $\Delta \text{Brier} = -0.000123$, 95% CI $[-0.000639, +0.000428]$, $p = 0.3337$.
- Hypothesis H2 (superiority of anatomical Kenyon Cell projection over rewired control) was not supported: $\Delta \text{Brier} = -0.000026$, 95% CI crossed zero, $p = 0.4791$.
- The V11 holdout dataset (`seed: 20261104`) is considered burned and restricted strictly to retrospective diagnostics.

### Primary Goals for V12:
1. Quantitatively and cleanly diagnose why V11 failed to beat the scalar baseline.
2. Conduct a strictly bounded search across up to 3 compact method families ($\le 24$ configurations per family) against a properly tuned scalar baseline (Tuned Scalar V12) across three independent development seeds.
3. Conduct robustness and sensitivity analysis across delays, missingness, label noise, concept drift, and MNAR reporting dependencies.
4. If a qualified candidate passes the pre-registered admission gate, execute a strictly **single-shot confirmatory test on 200 users across 5 reserved seeds**. If no candidate qualifies, seal a justified negative conclusion without generating or peeking at confirmatory seeds.

---

## 2. Research Questions (Stage 1 Diagnostics)

- **Question A (Protective Mechanisms):** How much potential contextual gain was attenuated by the closed safety gate ($\gamma$), the familiarity dampening factor, and logit residual clipping?
- **Question B (Feature Informativeness):** Do available contextual features at decision time carry individual sensitivity information beyond the user intercept?
- **Question C (Feedback Limits):** How does personalization utility scale with feedback volume, response delay (20h vs 72h), missing observations, and parameter drift?
- **Question D (Representation Properties):** Does PCA preserve residual-relevant signal for $y - p_0$ or predominantly population variance already captured by the GBDT?
- **Question E (Locus of Error):** Is performance bottlenecked by the context representation, the online update rule, data volume, or the synthetic generator itself?

---

## 3. Control Benchmarks and Oracles

1. **Benchmark: Intercept-Only (Negative Contextual Control):**
   - Synthetic population with individual baseline willpower intercepts $b_u$, but zero individual contextual sensitivity variation ($\mathbf{w}_u = \mathbf{0}$).
   - *Requirement:* Contextual methods must not show spurious persistent gains over scalar.
2. **Benchmark: Known Contextual Effect (Positive Contextual Control):**
   - Synthetic environment with an injected individual linear slope of fixed amplitude on key triggers (stress, sleep debt, fasting).
   - *Requirement:* A valid contextual adaptation algorithm must reliably recover the individual slopes and significantly outperform scalar given sufficient feedback.
3. **Diagnostic Oracles (Upper bounds only, forbidden for model selection):**
   - **Simulator Ground-Truth Oracle:** True simulator probability $P_{\text{true}}$.
   - **Privileged User-Profile Oracle:** Model accessing hidden profile parameters `latent_sensitivities` and `willpower`.
   - **Retrospective Full-History Fit:** Post-hoc regression fit across all 180 days.

---

## 4. Cohort Architecture and Seed Reservation

Disjoint data roles:
- **`training`** (Seed: `20261201`): 60 users $\times$ 180 days = 10,800 decisions.
  - Used for: training population model $p_0$ (GBDT), shared projectors (PCA, supervised bottleneck, autoencoder), and population experts.
- **`tuning`** (Seed: `20261202`): 30 users $\times$ 180 days = 5,400 decisions.
  - Used for: tuning the scalar baseline (Tuned Scalar V12: $\eta^*, \lambda^*$) and general grid boundaries.
- **`development` (3 seeds for search and ablations):**
  - Dev Seed 1: `20261203` (40 users $\times$ 180 days)
  - Dev Seed 2: `20261204` (40 users $\times$ 180 days)
  - Dev Seed 3: `20261205` (40 users $\times$ 180 days)
  - Used for: V11 diagnostic ablations, bounded candidate search, stress testing.
- **`confirmation` (5 strictly reserved seeds — NOT generated until passing Stage 4 admission):**
  - Conf Seed 1: `20261211` (40 users)
  - Conf Seed 2: `20261212` (40 users)
  - Conf Seed 3: `20261213` (40 users)
  - Conf Seed 4: `20261214` (40 users)
  - Conf Seed 5: `20261215` (40 users)
  - Total: 200 users $\times$ 180 days = 36,000 decisions.
- **`bootstrap`**: Seed `20261220` (20,000 resamples for stratified paired bootstrap).

All synthetic profiles are validated for unique parameter hashes to prevent duplicate users across cohorts.

---

## 5. Candidate Families and Baselines

### 5.1. Baselines
1. **Population Model ($p_0$):** GBDT (HistGradientBoostingClassifier) trained strictly on `training`.
2. **Frozen Scalar V11:** Scalar memory with frozen parameters from V11 ($\eta_{\text{fast}} = 0.1, \eta_{\text{slow}} = 0.025, \lambda = 0.01$).
3. **Tuned Scalar V12:** Scalar memory optimized for Macro Brier on the `tuning` cohort under a matched budget (primary comparison baseline).

### 5.2. Compact Contextual Candidate Families
1. **Family 1: Hierarchical Regularized Logistic Residual (Hierarchical Ridge):**
   - Personal vector $\mathbf{w}_u \in \mathbb{R}^d$ ($d \in \{2, 4, 8\}$) over normalized contextual features.
   - Prior distribution $\mathbf{w}_u \sim \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I})$ estimated on `training`.
   - Online update with empirical Bayes shrinkage, confidence weighting, and recency decay.
2. **Family 2: Supervised Bottleneck Residual:**
   - Linear or low-rank projection $X \to \mathbb{R}^d$ trained on `training` targeting the residual risk $r = y - p_0$.
   - Personal weights adapted directly in the supervised residual coordinate space.
3. **Family 3: Mixture of Population Experts:**
   - $K \in \{2, 3, 4\}$ specialized population models trained on behavioral user clusters from `training`.
   - Personal adaptation represented compactly as user responsibility weights $\boldsymbol{\alpha}_u \in \Delta^{K-1}$.

---

## 6. Search Budget and Protocol

- Maximum **24 configurations** per family ($\le 72$ candidate configs total) and up to 24 configs for Tuned Scalar.
- Fixed hyperparameter grids specified prior to running evaluation across dev seeds.
- Evaluation across all 3 development seeds (`20261203`, `20261204`, `20261205`).
- Shared $p_0$ and representation caching to respect the 12-hour local compute budget.
- Maximum of 2 algorithmic revision cycles; all modifications recorded in the run registry.

---

## 7. Robustness and Sensitivity Matrix (Stage 3)

Evaluation of top configurations on development cohorts under:
1. **Default feedback:** 20h delay, $\sim 70-80\%$ reporting rate.
2. **Reduced observability:** 50% and 10% logging rates.
3. **Extended delays:** 72h fixed delay; mixed exponential delay (mean 36h).
4. **Label noise:** 15% random flip of reported feedback.
5. **Abrupt concept drift:** 3-phase sensitivity shifts.
6. **MNAR reporting dependence:** $P(\text{report}|y=1) \neq P(\text{report}|y=0)$ in both directions.
7. **History horizon:** Serialized state size measured at 180 and 365 days.

---

## 8. Candidate Selection and Confirmation Admission Gate (Stage 4)

### Selection Rule:
1. Minimum average Macro Brier across the 3 development seeds in the primary scenario.
2. Ties/tolerances within $0.0002$ broken by smaller state size, then lower latency.
3. Verification of strict causality, finite valid probabilities, and serialization integrity.

### Admission Gate to Confirmation:
- A candidate is **admitted to confirmation** if and only if:
  $$\Delta \text{Brier} = \text{Brier}_{\text{candidate}} - \text{Brier}_{\text{TunedScalar}} \le -0.0020 \quad (\ge 0.0020 \text{ gain})$$
  on average across the 3 development seeds, **AND**
  $$\Delta \text{Brier} < 0 \quad \text{on at least 2 of the 3 development seeds}.$$
- **Stop Rule:** If no candidate satisfies the admission gate, the confirmatory seeds `20261211`–`20261215` **MUST NOT be generated or accessed**. The program concludes with a documented negative finding.

---

## 9. Confirmatory Evaluation Criteria (Confirmation)

If admitted, single-shot evaluation is executed on 200 users across 5 seeds (`20261211`–`20261215`).

### Primary Criterion:
- Contrast: $\Delta \text{Brier} = \text{Brier}_{\text{candidate}} - \text{Brier}_{\text{TunedScalar}}$.
- Analysis: Stratified paired bootstrap by user (20,000 resamples, seed `20261220`).
- **Success Requirements:**
  1. Mean $\Delta \text{Brier} \le -0.0020$.
  2. Upper bound of two-sided 95% CI $< 0$.
  3. Pre-specified one-sided test at $\alpha = 0.05$ rejected ($p < 0.05$).
  4. $\Delta \text{Brier} < 0$ on at least 4 of the 5 confirmation seeds.

### Secondary Engineering Criteria:
- Macro Log-Loss within $0.0020$ of Tuned Scalar.
- Proportion of degraded users ($\Delta \text{AUC} < -0.01$ vs population) no more than 5 percentage points worse than scalar.
- Robustness degradation $\le 0.0020$ across stress scenarios.
- Peak serialized JSON state $\le 16$ KiB per user at 180 days.
- p95 inference latency $\le 2\times$ Tuned Scalar.

---

## 10. Machine-Readable Run Registry

All experimental executions are tracked in `runs/v12-registry/run_registry.json` recording:
- `run_id`, `stage`, `family`, `status`, `config`, `inputs` (paths & SHA-256 digests), `metrics`, `execution_time_seconds`, `stop_reason`, and UTC timestamps.
