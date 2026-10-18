# V13: fewer false alarms; consistent recall improvement not established

[Russian full report](results-v13.ru.md) · [Protocol (Russian)](protocol-v13.ru.md)

October 17, 2026. The fixed exploratory experiment is complete. **No admission to a new confirmatory study or production deployment.** V12 confirmation seeds remain unopened.

The evaluation covers 80 synthetic users, 180 daily decisions each: 14,400 decisions, 8,479 lapse outcomes and 5,921 non-lapse outcomes. These development cohorts were previously used in V12; this is not independent confirmation or evidence of real-user efficacy.

| Pooled metric | Static | Budget-aware |
|---|---:|---:|
| True alerts | 2,441 | 2,449 |
| False alerts | 818 | 753 |
| Missed lapses | 6,038 | 6,030 |
| True negatives | 5,103 | 5,168 |
| Total alerts | 3,259 | 3,202 |
| Precision | 74.90% | 76.48% |
| Recall | 28.79% | 28.88% |
| FPR | 13.82% | 12.72% |
| Alerts per user-week | 1.5842 | 1.5565 |

The candidate produced 8 additional true alerts and 65 fewer false alerts (−7.95%). However, the primary user-weighted mean recall changed from 28.131% to 28.054% (−0.0765 percentage points), missing the required +5 points.

| Cohort | Macro recall delta, pp [95% CI] | Macro FPR delta, pp [95% CI] | Candidate macro FPR |
|---|---|---|---:|
| dev_2 | −0.480 [−0.956, −0.054] | −0.740 [−1.390, −0.131] | 15.680% |
| dev_3 | +0.327 [−0.377, +0.967] | −1.609 [−2.573, −0.673] | 15.697% |

Intervals use 10,000 paired user bootstrap draws within each seed, seed 20261301. They are exploratory and do not account for selecting the underlying model on V12 development. Recall did not increase on both cohorts; macro FPR exceeded the 15% cap on both. Pooled FPR cannot substitute for the preregistered macro constraint.

Both policies use the same frozen tuned-scalar forecasts and a maximum of two alerts per rolling 168 hours. The complete fixed grid was 22 static and 66 budget-aware configurations on dev_1. Selected static base=0.55; candidate base=0.55 and reserve_penalty=0.05. Evaluation used the saved settings on dev_2/dev_3, with no retuning. The penalty increases the threshold when one slot remains and fades as the next slot approaches release.

Corrected V12 was replayed from saved models, including 151,200 forecasts, metrics, 12 stress scenarios and full state-memory trajectories. Full suite including both the replay bundle and local TypeScript encoder: **112 passed, 7 skipped**. Lint, affected-file formatting, compilation, dependency checks, wheel/sdist build and installed-wheel imports passed. Seven tests requiring an unavailable external V10 author listing remain skipped. Setuptools emits a non-blocking future license-metadata warning. Historical V11 formatting was left unchanged.

An independent recount of all 28,800 policy decisions matched event outcomes, original probabilities, confusion matrices, macro recall/FPR and the rolling cap. Cap violations: zero. All 192 prehashed historical source/archive files were unchanged.

Artifacts are local and gitignored: `runs/v12-repair-20261017/`, `runs/v13-policy-20261017/`, and `runs/v12-v13-verification-20261017/`. The latter includes test logs and `independent_decision_audit.json`. Forecast SHA-256: `2709218f70e12907e1b983324337a43f48a8a9aa06b312dc55d14a6fffde04c6`; frozen-policy SHA-256: `660dc73976941322d7609321c34b1a2e6a3432816ba76fdeff9c88506f0136bd`.

An additional post-hoc arithmetic bound: two alerts per rolling seven days allow at most 52 alerts per person over 180 daily decisions. Across 80 users, 4,160 possible alerts cannot cover more than 49.1% of the 8,479 lapse outcomes even with zero false alerts. This is an upper bound, not an achievable oracle result. A 94% recall target is incompatible with this event frequency and notification budget.

The result supports reduced noise in this simulation, not consistent improved detection. The next research question concerns timely informative inputs and realistic outcome frequencies, not post-hoc threshold changes. Intervention benefit requires a separate real-data design because alerts may change outcomes.
