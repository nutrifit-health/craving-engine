# Public V9 protocol snapshot

[Русская версия](protocol-v9.ru.md)

This document describes the original V9 experiment and the intended standalone replay of its selected inputs. It is a public methodological description, not proof that the extracted runner has reproduced the historical result. Packaging changes are recorded in `extraction.json`.

## Question and fixed comparison

The broader research goal is useful warning of diet lapses through a shared model and per-user memory, without fitting a full model for every user. Approximately 94% event recall is a desired sensitivity target, not accuracy and not an achieved product result. V9 admission instead uses the constrained FPR20 comparison below; the high-recall operating point is diagnostic and cannot override failed admission.

Does a population-trained mushroom body representation with personal online memory improve synthetic lapse detection enough to justify a new blind test, relative to a fixed GBDT model and a scalar memory control?

The designated candidate was anatomical MB with personal memory. The six variants were GBDT population, GBDT + scalar memory, anatomical MB population, anatomical MB + personal memory, rewired MB population and rewired MB + personal memory. Rewired variants are controls, not alternative candidates selected after viewing development outcomes.

## Data

All person histories are synthetic. There are 60 training users (10,800 decisions), 20 calibration users (3,600 decisions), and 40 development users (7,200 decisions), with 180 daily decisions per user. Users do not cross splits. The dataset does not establish the prevalence or predictive behavior of real users.

Each row has event and user IDs, decision and availability times, observed feedback and its arrival time, evaluator outcome, and historical fields retained for provenance. `X` is the frozen 94-field observed context; `schema.json` lists names, units, availability indicators and semantics. The selected bundle contains matrices rather than a generator or the original backend encoder. Changing raw records does not automatically regenerate feature matrices.

The simulation uses outcome-dependent reporting (0.8 for positive outcomes, 0.7 for negative outcomes) and a 20-hour feedback delay. These are simulation assumptions, not observed NutriFit reporting rates. Missing feedback is not treated as a negative outcome. Of 10,800 training decisions, 8,136 have observed feedback; observed prevalence is approximately 58.12%.

Evaluator labels are available for scoring all calibration and development decisions, including unreported events. This idealizes what would be measurable in a deployed product. Evaluator labels are excluded from MB fitting and personal memory updates. Calibration uses complete synthetic evaluator labels for threshold selection only; it is not a training input to the population model.

## Representation

The anatomical artifact is an ALPN → KC submatrix from MaleCNS v1.0: 164 connected ALPN, 2,019 left-side Kenyon cells, 11,301 weighted edges, and 192,391 synapses. Of these KCs, 124 have no ALPN input in the selected submatrix. It excludes other inputs, APL, dopamine neurons, MBON and recurrent dynamics. Synapse counts supply nonnegative structural weights; they are not experimentally established conductances or a physiological learning rule.

For every observed feature, the fitted transform computes masked asinh contrast relative to training statistics, followed by ON and OFF channels. Missing values give zero channels while their availability indicators remain separate inputs. The mapping from these human context channels onto ALPN is artificial; mapping seed is fixed at 11.

The representation uses row-normalized anatomical weights, positive top-k activity with a 5% winner fraction, a training-derived RMS scale, and an anchor coordinate. The encoded vector is L2-normalized. The anchor is a context coordinate whose relative effect depends on context norm; it is not a separately learned global head.

The rewired control uses degree-preserving double edge swaps with seed 20260923. It preserves binary KC in-degree, ALPN out-degree and the multiset of weights within each KC. It does not preserve ALPN output strength or all higher-order motifs. Both graph variants receive the same engineering adapter and fit procedure.

## Population fitting

MB is a standalone population predictor. It does not receive GBDT probabilities as a logit offset. The readout is binary LogisticRegression (`C=10`, `max_iter=2000`, `tol=1e-6`; complete parameters are in `research_v9/types.py`). A three-fold user-disjoint ensemble uses sigmoid calibration. Each fold fits its normalization, representation and readout on its own fit rows; the sigmoid uses the corresponding excluded users. The final probability is the mean of three calibrated probabilities.

Only rows with observed training outcomes enter the readout fit, with one weight per row and no class reweighting. Frozen fold indices and training event IDs match the historical GBDT comparison. Same inputs and folds do not equate model capacities or optimization budgets.

The final ensemble's training predictions are not fully out-of-fold: other ensemble members may have fit a given training user. The online representation is fitted separately on all training `X`, including unlabelled rows, without evaluator outcomes. It is fixed before calibration and development replay.

The public runner **does not refit GBDT**. It consumes the preserved aligned calibration and development predictions. It permits reproduction of the MB fitting and comparison against this fixed reference, not reproduction of the full synthetic generation and GBDT training pipeline.

## Personal memory and chronology

Each user receives an independent memory state. The population model and encoding remain fixed during replay. Memory contains fast and slow residual weights, exposure state, a familiarity and error-based gate, pending event snapshots and observation history. Defaults and formulas are specified in `research_v7/types.py`, `research_v8/types.py` and `research_v8/memory.py`.

Replay predicts before seeing the current outcome. An observed feedback event updates memory only when its arrival time is strictly earlier than the next decision time. Prediction snapshots preserve the context used at the original decision. Unobserved feedback events do not update the model. Each split and model family uses its own memory. Calibration user experience is not transferred to development users.

The final probability is a gated correction in logit space over the corresponding family's population prediction. A gate can suppress personalization when available evidence does not support it; that mechanism does not prove an absence of individual harm. The complete state is larger than one vector of readout weights and must not be described as a measured 16 KiB per-user product budget.

## Metrics

The primary operating point is `calibration_macro_fpr20`: choose a model-specific threshold on calibration that maximizes macro recall subject to macro FPR ≤ 0.20. Freeze all six thresholds and the simple baseline before development scoring. The stronger simple control is selected between population and scalar using calibration recall, then Brier as a tie-breaker.

Report macro recall, FPR, AUC, Brier, Youden J, paired user comparisons, pooled confusion counts and notification statistics. Macro averages give users equal weight; metric-specific eligibility matters when a user has only one outcome class. Youden J = recall − FPR; it is dimensionless and is not accuracy. Confidence intervals are resampled at user level under the implementation's fixed bootstrap procedure.

Warning limits are evaluated separately from classifier thresholds: at most 2, 3 or 7 warnings in a rolling seven-day window. Classifier recall is not warning coverage. Detecting an event is not evidence that a warning would prevent it.

## Admission

Against both GBDT population and the calibration-selected simple baseline, the designated candidate must satisfy all of the following:

- recall gain at least 5 percentage points and paired 95% CI above zero;
- Youden J gain at least 0.02 and paired 95% CI above zero;
- macro FPR at most 0.20, and increase no more than 0.01;
- no worse macro Brier, and positive pooled Brier skill relative to training prevalence;
- fraction of users with AUC degradation over 0.01 no more than 0.05;
- improved warning recall under the two-per-seven-days policy without worse false alarms per caught event.

The original candidate failed. Scalar was selected as the simple baseline on calibration. No blind test followed. Existing development scores are known, so future changes judged on them are exploratory and require a new untouched user cohort for confirmation. The [roadmap](research-roadmap.md) describes proposed follow-up questions; it does not replace this historical protocol or preregister a successor experiment.

## Preservation and standalone changes

The original experiment remains in NutriFit's private research archive. This repository preserves selected original inputs and outputs, with original input hashes and source artifact names in `data/v9-frozen/manifest.json`. That selected manifest is not the manifest of the entire private archive.

The standalone runner uses the curated bundle in place of the private parent run, snapshots public sources without private Git metadata, and writes a fresh directory for each attempt. It checks artifact consistency when executed. No new experiment, installation, test or source-to-result verification was performed during extraction.

The selected result set also includes `high-recall-excerpt.json`, containing fields copied from the original `report.json` without recomputation. Its provenance is recorded in `extraction.json`; it is separate from the selected input manifest. See [release preparation](release-preparation.md) for the extraction's publication status.
