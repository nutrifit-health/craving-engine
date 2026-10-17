# Research V11: Compact Contextual Personalization vs. Scalar and Anatomical Control (Technical Audit & Reproducible Report)

[Русский](results-v11.ru.md) · [README](../README.md) · [Protocol](protocol-v11.md)

Date: October 16, 2026. Governance task: [TASK-2026-0794](https://gov.nutrifit.health/tasks/6ad2b10a2858f3e61a4f22f5). Previous iterations: [V9](results-v9.md), [V10](results-v10.md).

> [!IMPORTANT]
> **Holdout Cohort Status:** The holdout cohort (seed 20261104, 40 users, 7,200 decisions) was already exposed and evaluated in the initial run. This report documents the findings of a technical audit, resolution of defects in statistical and analytical code, expansion of invariant tests, and comprehensive archival of artifacts for independent reproduction. This recomputation represents a **technical reproduction**, not a new blind confirmation.

---

## Main Empirical Conclusions

1. **Compact contextual personalization ($d=8$) demonstrated no confirmed advantage over 1D scalar intercept:** point estimate difference in Macro Brier was only **−0.000123** with a 95% bootstrap CI of **[−0.000639, +0.000428]** ($p = 0.3337$, statistically non-significant after corrected Holm step-down procedure; engineering threshold $\le -0.0020$ not achieved).
2. **Anatomical connectivity of Drosophila MaleCNS v1.0 showed no confirmed advantage over a matched random graph under the tested conditions:** the Brier score difference between anatomical and rewired representations was **−0.000026** with a 95% bootstrap CI of **[−0.000442, +0.000470]** ($p = 0.4791$, non-significant).
3. **Exploratory comparison:** scalar baseline adaptation ($b_u$) had a lower Brier score than population GBDT ($p_0$): Macro Brier reduction is **−0.003947** on holdout. Adding 8 contextual weights $w_u$ yields negligible incremental benefit over scalar intercept adaptation.

> [!WARNING]
> **Scope and Limitations:**
> - All conclusions are established on a synthetic longitudinal simulator and cannot be generalized directly to human clinical behavior.
> - Absence of statistically significant difference establishes failure to confirm an advantage, but does not prove mathematical equivalence of models.
> - Partitioning cohorts with fixed seeds reduces data dredging risks, but does not inherently prove absence of temporal information leakage during feature engineering.

---

## 1. Technical Audit and Defect Resolution

During review of the initial run (`runs/v11-run-20261016`), four specific defects were identified and resolved:

1. **Holm-Bonferroni Procedure Fix ([research_v11/stats.py](../research_v11/stats.py)):**
   * *Issue:* The original code tested hypotheses independently against rank thresholds without stopping after the first non-rejected test. In boundary conditions (e.g., $p=[0.03, 0.04]$), subsequent hypotheses could be falsely declared significant.
   * *Fix:* Implemented canonical step-down Holm procedure: testing halts immediately upon the first non-rejected hypothesis, and all subsequent hypotheses are declared non-significant.
   * *Impact:* Both V11 p-values ($p_1 = 0.3337$, $p_2 = 0.4791$) exceed 0.05; substantive conclusions remain unchanged (neither contrast is significant).
2. **Few-Shot Comparison Partitioning Fix ([research_v11/runner.py](../research_v11/runner.py)):**
   * *Issue:* The `feedbacks_processed` counter only incremented inside `if memory is not None:`. Consequently, `population` received all 7,200 decisions in bin $K=0$, leaving higher bins empty.
   * *Fix:* The counter of matured feedbacks prior to $t_{\text{decision}}$ is now computed strictly chronologically per user regardless of model family.
   * *Impact:* All five model families now share identical decision counts in every bin ($K=0$: 49, $K \in [1,2]$: 108, $K \in [3,5]$: 156, $K \in [6,10]$: 265, $K \ge 11$: 6,622). This allows fair comparison of $p_0$ baseline (Brier 0.1843 at $K \ge 11$) with personalized models (Brier 0.1796–0.1800).
   * *Causal Timestamping:* Causal gating uses `feedback_at < decision_at`. In the generated cohorts, minimum delay is exactly 20.00h; no events have identical timestamps.
3. **Artifact Completeness & Phase Decoupling ([research_v11/experiment.py](../research_v11/experiment.py)):**
   * *Issue:* The original pipeline did not serialize trained PCA encoders (`encoders.pickle`), omitted gating parameters in `frozen_config.json`, and did not export event predictions.
   * *Fix:* Decoupled into `fit_and_freeze_v11` (training/tuning without holdout access) and `evaluate_frozen_v11` (evaluating holdout strictly from frozen artifacts). Exported `models/encoders.pickle`, `models/gbdt.pickle`, complete `frozen_config.json`, `holdout_scores.json.gz` (36,000 decision rows), and `user_metrics.json`.
4. **Test Suite Expansion & Real Invariant Verification:**
   * Earlier tests covered only 50 steps with fixed 20h delay and empty pending queues.
   * Added comprehensive tests ([tests/test_v11_memory.py](../tests/test_v11_memory.py), [tests/test_v11_pipeline.py](../tests/test_v11_pipeline.py)): 60 steps with irregular delays (2–48h, exceeding decision step), serialization with active pending items and trajectory continuation verification, gate locking ($\gamma=0$ when $N_{\text{eff}} < 12$), amplitude gradient linearity, and training isolation.

---

## 2. Holdout Cohort Results (40 users, 7,200 decisions)

Technical reproduction under `.venv` (Python 3.12.14, scikit-learn 1.9.1) matched original results exactly to 6 decimal places:

| Model Family | Macro Brier ↓ | Macro AUC ↑ | Pooled AUC ↑ | Macro Log-Loss ↓ | AUC Degradation Fraction | Peak JSON State (bytes) |
|---|---|---|---|---|---|---|
| **Population ($p_0$)** | 0.186667 | 0.7240 | 0.7694 | 0.5596 | 0.0% | 0 |
| **Scalar ($d=0$)** | **0.182720** | 0.7229 | 0.7760 | 0.5492 | 5.0% | 13,595 |
| **Compact Linear ($d=8$)** | **0.182597** | 0.7224 | 0.7764 | 0.5487 | 2.5% | 14,096 |
| **Compact KC Anatomical ($d=8$)** | **0.182360** | 0.7230 | 0.7768 | 0.5478 | 2.5% | 14,080 |
| **Compact KC Rewired ($d=8$)** | **0.182386** | 0.7230 | 0.7768 | 0.5480 | 2.5% | 14,080 |

*Denominators:* Total users: 40, decisions: 7,200 (180 per user). Evaluable AUC users (both classes observed): 40 of 40 (100%). Degradation fraction: percentage of users with $\text{AUC} < \text{AUC}_{\text{pop}} - 0.01$.

---

## 3. Confirmatory Statistical Analysis (20,000 bootstrap resamples)

Evaluated with corrected step-down Holm-Bonferroni procedure ($\alpha = 0.05$):

### Contrast 1 (Primary): `compact_linear` vs. `scalar`
- **Hypothesis:** $\mu_\Delta = \text{Brier}(\text{compact\_linear}) - \text{Brier}(\text{scalar}) < 0$.
- **Point Estimate $\hat{\Delta}$:** **−0.000123**
- **95% Percentile CI:** **[−0.000639, +0.000428]** (covers zero).
- **One-sided p-value under $H_0$:** **0.3337**
- **Holm Threshold (Rank 1):** $\alpha / 2 = 0.025$
- **Statistical Significance:** **Rejected (False)**
- **Engineering Threshold ($\le -0.0020$):** **Not achieved**
- **Conclusion:** **Not confirmed.**

### Contrast 2 (Anatomical): `compact_kc_anatomical` vs. `compact_kc_rewired`
- **Hypothesis:** $\mu_\Delta = \text{Brier}(\text{anatomical}) - \text{Brier}(\text{rewired}) < 0$.
- **Point Estimate $\hat{\Delta}$:** **−0.000026**
- **95% Percentile CI:** **[−0.000442, +0.000470]** (covers zero).
- **One-sided p-value under $H_0$:** **0.4791**
- **Holm Threshold (Rank 2):** $\alpha / 1 = 0.050$ (halted at Rank 1)
- **Statistical Significance:** **Rejected (False)**
- **Conclusion:** **Not confirmed.** Difference between MaleCNS and degree-matched random graph is indistinguishable from noise.

---

## 4. Few-Shot Analysis: Dynamics by Matured Feedbacks

With corrected event partitioning, all models are evaluated on identical decision sets:

| Feedback Bin ($K$) | Decisions | Users | Brier: Pop ($p_0$) | Brier: Scalar | Brier: Linear | Brier: KC Anat | Brier: KC Rewired | Gate Open % |
|---|---|---|---|---|---|---|---|---|
| **$K = 0$** | 49 | 40 | 0.2523 | 0.2523 | 0.2523 | 0.2523 | 0.2523 | 0.0% |
| **$K \in [1, 2]$** | 108 | 40 | 0.2450 | 0.2450 | 0.2450 | 0.2450 | 0.2450 | 0.0% |
| **$K \in [3, 5]$** | 156 | 40 | 0.2118 | 0.2118 | 0.2118 | 0.2118 | 0.2118 | 0.0% |
| **$K \in [6, 10]$** | 265 | 40 | 0.1963 | 0.1963 | 0.1963 | 0.1963 | 0.1963 | 0.0% |
| **$K \ge 11$** | 6,622 | 40 | **0.1843** | **0.17996** | **0.17983** | **0.17957** | **0.17960** | 12.0–14.6% |

---

## 5. Memory Footprint and State Serializability

| Model Family | Weight Size (numeric bytes) | Peak JSON State (max bytes) | Mean Peak JSON State (bytes) |
|---|---|---|---|
| **Scalar** | 16 bytes (2 × float64) | 13,595 | 12,393 |
| **Compact Linear ($d=8$)** | 144 bytes (18 × float64) | 14,096 | 12,881 |
| **Compact KC Anat ($d=8$)** | 144 bytes (18 × float64) | 14,080 | 12,883 |
| **Compact KC Rewired ($d=8$)** | 144 bytes (18 × float64) | 14,080 | 12,884 |

All models strictly satisfy the 16 KiB state budget.

---

## 6. Repository Test Suite Execution

Executed command:
```bash
NUTRIFIT_BACKEND_ROOT=/absolute/path/to/NutriFit/apps/backend PYTHONPATH=. .venv/bin/pytest tests/ research_v8/ research_v9/ research_v10/ -ra
```
- **Result:** 88 passed, 7 skipped, 0 failed.
- **Skipped test details:**
  - `test_local_backend_adapter` passed with explicit `NUTRIFIT_BACKEND_ROOT`; no token is required. It is skipped when that checkout is unavailable.
  - 7 tests in `research_v10/test_author_audit.py`: author listing audit tests (third-party Habr listing not redistributed).

---

## 7. Artifacts & Reproduction

1. **Initial Run Archive (Immutable):**
   - Path: `runs/v11-initial-run-20261016-archive/`
   - Manifest with SHA-256: `runs/v11-initial-run-20261016-archive/manifest.json`.
2. **Reproducible Audit Run:**
   - Path: `runs/v11-reproducible-audit-20261016/`
   - Trained models: `models/gbdt.pickle`, `models/encoders.pickle`
   - Configuration: `frozen_config.json`, `freeze_manifest.json`
   - Decision records: `holdout_scores.json.gz` (36,000 rows)
   - User metrics: `user_metrics.json`
3. **Independent Verification Command:**
   ```bash
   .venv/bin/python -c "
   from pathlib import Path
   from research_v11.experiment import evaluate_frozen_v11
   evaluate_frozen_v11(Path('runs/v11-data'), Path('runs/v11-closure-20261016'), eval_output_dir=Path('runs/v11-independent-reproduction'))
   "
   ```

## 8. Independent audit closure

Follow-up: [V11 corrections and next research protocol](https://gov.nutrifit.health/tasks/6ad2bbb2556a2264e7a13394).

The new bundle is `runs/v11-closure-20261016/`. All three previous run/archive directories remain unchanged; all 36 archive file hashes were verified. The historical manifest incorrectly described user-level Brier scores as present in the initial summary. They were absent; `provenance_correction.json` records the correction without modifying the archive.

Manifest v2 binds computational dependency versions, training/tuning/calibration inputs, schema, copied connectome, current sources (including uncommitted changes) and model artifacts. Evaluation verifies integrity before deserialization and again after replay; `evaluation_manifest.json` binds holdout input hashes to outputs. This is explicitly a **retrospective** seal after holdout exposure, not proof of the original freeze chronology or an externally signed timestamp.

Replay from the saved models exactly reproduced all 36,000 decision records and the summary (`reproduction_check.json`), without fitting or tuning. Autonomous tests now cover the real fit/transform/replay path, complete object immutability, no evaluation-time fitting, future-feedback causality, equal-time ordering, tamper rejection and partial-output protection without relying on local run data. Ruff passes for V11 and its tests; the wheel includes V11 and imports outside the checkout. Logs are in the bundle's `verification/` directory.

The commands require the local data/model bundles; `runs/` is ignored by Git, so publishing source code alone does not publish those bundles. Use a fresh evaluation output directory on each reproduction. The 16 KiB state budget applies to the measured 180-day trajectories, not an unbounded operational lifetime.
