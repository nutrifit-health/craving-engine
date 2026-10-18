# V12: exploratory search and technical audit repairs

[Русский](results-v12.ru.md) · [Protocol](protocol-v12.md)

**Status: corrected replay and verification completed on October 17, 2026; exploratory search closed without admission to confirmation.** Confirmation seeds `20261211`–`20261215` remain unopened. No deployment or collection of real user data is authorized.

## Verified corrected results

Saved models were replayed without fit or new selection. The true frozen V11 artifact uses rates 0.2/0.05; its SHA-256 is `40d1030847c626a05f908a8c1159b6714ac7e1afdb0f1834bcc461993d6e24cc`.

| Method | dev_1 Brier | dev_2 Brier | dev_3 Brier | Mean Macro Brier |
|---|---:|---:|---:|---:|
| Population | 0.194613 | 0.175314 | 0.197007 | 0.188978 |
| True Frozen Scalar V11 | 0.184132 | 0.171273 | 0.189215 | **0.181540** |
| Tuned Scalar V12 | 0.183169 | 0.171037 | 0.188288 | **0.180831** |
| Regularized contextual residual | 0.183890 | 0.171445 | 0.189394 | 0.181576 |
| Supervised bottleneck | 0.183464 | 0.171228 | 0.188675 | 0.181123 |
| Mixture of experts | 0.186593 | 0.172932 | 0.188558 | 0.182694 |
| Fixed mixture of the same experts, diagnostic | 0.189673 | 0.174062 | 0.192394 | 0.185376 |

The corrected Tuned V12 − Frozen V11 difference is **−0.000708828**, replacing the withdrawn −0.002449. The selected contextual candidate remains worse than tuned scalar by **+0.000291491**, losing on all three seeds. No confirmation admission. All results remain exploratory.

All 12 stress scenarios were replayed. Actual abrupt-drift scalar/candidate Brier is 0.192351/0.192574; temporary drift is 0.191069/0.191483; default is 0.183169/0.183464. Reporting scenarios realized 49.51% and 10.46%. With 10% reporting the personal methods matched the population model, not an established improvement under scarce feedback.

Complete JSON peaks after every event: scalar 9,607 bytes at 180 days and **10,304** at 365 days; candidate 11,684 and **12,381** bytes. Both fit 16,384 bytes on this specific load trajectory, with up to eight pending snapshots. This is not a guarantee for arbitrary event rates or delays.

The autonomous bundle verified 113 hashed files, regenerated 151,200 individual forecasts, metrics, stress scenarios and memory trajectories. Full suite with the local backend adapter and replay bundle: **112 passed, 7 skipped** (unavailable external V10 author listing). Lint, affected V12/V13 formatting, compilation, dependency checks, wheel/sdist build and installed-wheel imports passed. Historical V11 formatting was unchanged; setuptools reports a non-blocking future license-metadata warning.

Artifacts: `runs/v12-repair-20261017/{stage2,stage3,closure}/`. Logs: `runs/v12-v13-verification-20261017/`. The immutable bundle retains its packaging-time pending-verification status; subsequent success is recorded separately in `autonomous-replay.log` and `pytest-full.log`. All 192 prehashed historical input/archive files were unchanged. Warning usefulness is evaluated separately in [V13](results-v13.md).

## Useful outcome and limits

V11 did not establish a practical advantage of the tested compact contextual adaptation over scalar memory. In Gemini's saved V12 search, none of three candidate families passed the admission rule: mean Brier difference versus tuned scalar ≤ −0.002 and a negative difference on at least two of three development seeds. Keeping confirmation closed remains appropriate. This is not proof that contextual personalization cannot work.

Synthetic Brier measures probability accuracy within the generator. It does not demonstrate prevention of lapses, useful alerts, or effectiveness on real users.

## Historical numbers, before these repairs

Source: `runs/v12-stage2/search_summary.json`. These are saved exploratory values, not independently verified results of corrected replay.

| Method | Mean Macro Brier | Interpretation |
|---|---:|---|
| Population | 0.188978 | Saved estimate |
| V11 scalar with defaults | 0.183280 | Incorrectly called frozen V11: rates 0.1/0.025 instead of frozen 0.2/0.05 |
| Tuned scalar V12 | 0.180831 | Exploratory search result |
| Regularized contextual residual | 0.181576 | Exploratory search result |
| Supervised bottleneck | 0.181123 | Δ versus tuned scalar ≈ +0.000291; no admission |
| Mixture of experts | 0.182694 | Changes population models as well as personal mixture weights |

The claimed **−0.002449 gain over frozen V11 is withdrawn** and replaced by the verified −0.000708828 above. Historical abrupt/temporary drift results are withdrawn: a missing day field made both scenarios identical to default. The 5939/6060-byte values were two endpoint measurements, not peaks of complete state; the associated memory guarantee is withdrawn.

Imputing zero/one worsened Brier in two saved scenarios; this does not establish that all weak supervision is harmful. Low linear residual R² does not prove PCA destroyed individual signal. Retrospective fitting evaluated on the same full trajectory is an in-sample diagnostic, not an attainable online target. Mechanistic explanations remain hypotheses.

## Implementation repairs

- Frozen V11 settings are loaded explicitly from its artifact, with a source SHA-256; missing fields cannot silently use defaults.
- Full versioned checkpoints include configuration, clock, weights, history and pending contexts/expert probabilities. Restoration is implemented and backward clocks are rejected.
- Drift uses evaluation `day_index`: abrupt at day ≥ 90, temporary at 60 ≤ day < 90. The probability becomes p′ = p + 0.4(1−p); coupled zero outcomes switch to one with probability 0.4. Truth and reported labels remain consistent. Features and reporting masks stay fixed: this is a conditional-risk sensitivity experiment, not a new physiological simulator.
- Reporting 50%/10% denotes a Bernoulli probability, not an exact finite-sample quota; realized fractions are recorded.
- Memory is measured after every predict/observe/expire over 365 days, with 20/72/120-hour feedback delays and seven-day pending expiry for missing reports. Every measurement, both horizon peaks and the drained-stream peak are saved for scalar and candidate. This is a state load benchmark, not a quality evaluation.
- Saved models and previously selected configurations are replayed without fit or further search. All baseline and family-winner scores are saved. A fixed mixture of the same experts isolates the effect of personal MoE weights.
- New bundles refuse existing output directories and include data, models, complete configurations, dependency source snapshots, environment and hashes. Verification checks hashes before pickle, regenerates individual predictions and metrics, and repeats memory measurement.
- Regression tests covering frozen settings, actual drift windows, pending-state restoration, peaks, hidden-label isolation and individual prediction comparison passed.

## Deviations and next evaluation

The historical scalar grid contained 30 configurations (5 × 3 × 2), exceeding the protocol's limit of 24. This is disclosed rather than repaired by selectively deleting trials. The correction keeps the previously selected configuration and remains exploratory. Additional CLI search is disabled.

Historical `runs/v12-stage*`, `runs/v12-closure-20261017` and `runs/v12-initial-run-20261017-archive` remain unchanged. They are historical evidence with known defects. Corrected results are in the new `runs/v12-repair-20261017/` directory; see [replay instructions](v12-repair-replay.md). The exploratory search is closed with a bounded negative result. No automatic confirmation generation, relaxed threshold or new search is allowed.

## Future real-data requirements

This is a draft, not permission to collect data. Craving and actual food lapses require separate operational definitions; define the prediction horizon before collection. The simulator's same-evening daily target does not establish arbitrary 24-hour performance.

Record decision/event/feedback timestamps, feature availability, timezone, label source and confidence, missing versus explicit negative replies, and interventions. Predefine delayed-feedback handling. Estimate sample size from user-level paired-difference variability and MPID with explicit missingness/attrition assumptions; the previous claim that 40–50 users suffice is not an established power calculation.
