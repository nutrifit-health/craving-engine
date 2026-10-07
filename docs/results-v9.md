# Recorded V9 result — September 24, 2026

[Русская версия](results-v9.ru.md)

Source run: `run-v9-20260924T001842Z`. Original complete artifact manifest SHA256: `7405d213c86267deb1bb450bb156482b7712f75ae0fe31144877102632e8fd36`. Selected original files are preserved in `results/v9-2026-09-24/`. This snapshot has not been rerun.

Thresholds: chosen on 20 calibration users, frozen, then applied to 40 development users (7,200 decisions). The calibration constraint was macro FPR ≤ 20%; it is not a guarantee of the same bound on future data.

| Model | Recall | FPR | Youden J | AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| GBDT population | 39.647% | 15.398% | 0.242488 | 0.739105 | 0.173208 |
| GBDT + scalar memory | 39.515% | 15.116% | 0.243990 | 0.736194 | 0.170901 |
| Anatomical MB population | 43.236% | 19.695% | 0.235405 | 0.730732 | 0.182020 |
| Anatomical MB + personal memory | 42.770% | 19.592% | 0.231780 | 0.727969 | 0.177963 |
| Rewired MB population | 38.014% | 14.135% | 0.238796 | 0.733002 | 0.177562 |
| Rewired MB + personal memory | 37.944% | 14.292% | 0.236519 | 0.730612 | 0.175061 |

Values are macro averages across development users using the recorded metric definitions. Recall and FPR are rates with different denominators; neither is accuracy. Brier is probability error (lower is better); AUC measures ranking. Exact values and pooled counts are in [summary.json](../results/v9-2026-09-24/summary.json); decision and paired comparisons are in [admission.json](../results/v9-2026-09-24/admission.json) and [comparisons.json](../results/v9-2026-09-24/comparisons.json). The [calibration selection](../results/v9-2026-09-24/calibration_selection.json) chose scalar as the simple baseline; the fixed candidate had to pass against both scalar and GBDT population.

## What the result establishes

The designated candidate did not meet the fixed admission rules. Against GBDT it increased recall by 3.123 percentage points and FPR by 4.193 points; Youden J fell by 0.010708, AUC fell by 0.011136, and Brier increased by 0.004755. Twenty of forty users had AUC degradation exceeding 0.01.

At two warnings per rolling seven days, the recorded candidate caught 992 events with 175 false warnings, compared with 968 events and 148 false warnings for GBDT. An extra 24 caught events came with 27 extra false warnings. This does not establish whether any warning prevents an event.

Personal memory reduced anatomical MB Brier from 0.182020 to 0.177963, a delta of −0.004057 with paired user-bootstrap 95% CI [−0.008260, −0.000916]. At the same time macro AUC fell by 0.002764 and macro recall fell by 0.466 percentage points. This establishes a probability-error improvement relative to that MB's own population forecast on these synthetic users, not better overall detection or superiority to GBDT. The paired values are in `comparisons.json`, under `anatomy_personalization`. Changing thresholds can alter the recall comparison, so a new comparison should fix a common operating constraint in advance and report both detection and calibration.

Pooled AUC for the same population-to-personal comparison increased from 0.802483 to 0.809885, as preserved in `high-recall-excerpt.json`. Pooled AUC compares all users' rows together; macro AUC averages within-user ranking. This does not contradict the macro decrease, and neither aggregation alone establishes useful warning behavior. The primary protocol uses macro AUC; statements about a ranking decrease refer to that metric.

At the exploratory `calibration_macro_recall94` point, the anatomical personalized model had 93.957% macro recall and 85.488% macro FPR on development. The threshold was 0.28769973629539086. With two warnings per rolling seven days it caught 1,253 of 4,191 events (29.897% pooled warning coverage) with 785 false warnings. This is not 94% accuracy or a useful-warning guarantee. The [high-recall excerpt](../results/v9-2026-09-24/high-recall-excerpt.json) copies the original report's macro metrics, pooled counts and warning statistics for this operating point; it is not a new run or recomputation. The primary summary above covers a different, FPR20 operating point. The aspirational 94% target has not been reached together with acceptable warning burden.

## What remains open

This does not prove that every sparse personal memory or biologically inspired representation is ineffective. It rejects this candidate under this protocol on this synthetic development dataset. There is no established clinical accuracy, real-user generalization, causal prevention effect or million-user operational benchmark.

The anatomical and rewired classifiers operate at different frozen thresholds and false alarm rates. The scores do not by themselves establish a practical advantage caused by anatomy. That hypothesis needs matched-capacity controls, multiple prespecified seeds, comparable operating points and new data.

The original source archive contains prior checks from the historical experiment. Those checks do not validate the modified standalone loader, dependency ranges or packaging in this repository. This extraction's validation status is **not run**.

For the next proposed experiments see the [roadmap](research-roadmap.md). For publication and execution status see [release preparation](release-preparation.md).
