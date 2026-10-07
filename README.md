# NutriFit Craving Engine

[English](README.md) · [Русский](README.ru.md)

Can one shared model learn from each person's experience without retraining a full model per user? Craving Engine investigates this architecture for predicting diet lapse events from nutrition, sleep, activity and behavioral context. A fixed shared representation produces a population forecast; separate personal memory updates when that user's feedback arrives.

**Current stage: V9 completed on synthetic development data; the anatomical personalized candidate failed admission to a new blind test.** This repository preserves the negative result and the questions it leaves open. The standalone extraction has not yet been executed.

## Goal and success criteria

The long-term goal is useful advance warning of diet lapses with manageable false warnings and modest per-user state. The desired recall is about **94% of lapse events**, not 94% accuracy. This target has not been achieved together with an acceptable false warning burden. It is a research aspiration, not a validated product specification.

V9 asked a narrower question: can a population-trained mushroom body (MB) representation plus online personal memory outperform GBDT and a simple scalar personal correction enough to justify a fresh blind test? Admission required improvements in recall and Youden J, controlled FPR, no worse Brier, limited individual AUC degradation and better warning coverage at two warnings per rolling seven days. The [protocol](docs/protocol-v9.md) gives the fixed criteria.

The anatomical prior is a partial ALPN → Kenyon cell graph from [MaleCNS v1.0](https://male-cns.janelia.org/): 164 connected ALPN, 2,019 KCs and 11,301 weighted edges. Human context is mapped onto it through an artificial adapter. This is a sparse representation experiment, not a functional simulation of the fly brain. Anatomy's value is itself a hypothesis to test.

## Where the project stands

| Stage | Current status |
| --- | --- |
| Shared model and causal personal memory | Implemented and exercised in the original V9 experiment |
| Population fitting and synthetic comparison | One fixed V9 run completed on September 24, 2026; six variants reported |
| Admission to independent confirmation | Failed: `eligible=false`; no new blind cohort was opened |
| Standalone research snapshot | Inputs, selected results and code extracted; installation and execution not yet verified |
| Real-user validation, prevention and scale | Not established; no clinical study or million-user benchmark |

Historical checks of the original run do not establish reproducibility of this modified extraction. See [release preparation](docs/release-preparation.md) for the remaining publication steps.

## What we tried

All histories are synthetic: 60 training, 20 calibration and 40 development users, with 180 daily decisions per user and 94 observed features. Training used 8,136 reported outcomes out of 10,800 decisions. Feedback is selectively missing and delayed by 20 hours under simulated reporting rules. Missing feedback is not a negative label.

We fitted anatomical and degree-preserving rewired MB predictors with logistic readouts and user-disjoint sigmoid calibration, then compared their forecasts with and without personal memory. The fixed reference was GBDT, also evaluated with a scalar personal correction. Models received the same observed training rows and folds, but their capacities were not matched. MB predicts independently of GBDT; it does not use GBDT as an offset.

Personal memory updates chronologically from received feedback. Complete synthetic evaluator labels are used for scoring and calibration threshold selection, not population fitting or memory updates. Thresholds were frozen before development scoring. These development users had already been explored in earlier research; they are not a blind holdout.

## Recorded result

Values below come from the original run, not a rerun of this extraction. Each model's threshold maximized calibration macro recall subject to macro FPR ≤ 20%. The selected simple baseline was **scalar**; the designated candidate was **anatomical_personal**. The candidate had to pass against both scalar and GBDT population.

| Model | Macro recall ↑ | Macro FPR ↓ | Macro AUC ↑ | Macro Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| GBDT population | 39.65% | 15.40% | 0.7391 | 0.1732 |
| GBDT + scalar memory | 39.51% | 15.12% | 0.7362 | 0.1709 |
| Anatomical MB population | 43.24% | 19.70% | 0.7307 | 0.1820 |
| Anatomical MB + personal memory | 42.77% | 19.59% | 0.7280 | 0.1780 |
| Rewired MB population | 38.01% | 14.13% | 0.7330 | 0.1776 |
| Rewired MB + personal memory | 37.94% | 14.29% | 0.7306 | 0.1751 |

Recall measures the fraction of lapse events detected. FPR measures the fraction of non-lapse decisions falsely flagged. Macro metrics give each eligible user equal weight. Brier measures probability error; Youden J = recall − FPR. Neither recall nor J is accuracy. Thresholds are model-specific; rows do not compare models at identical development FPR.

- **Against GBDT:** the candidate gained 3.12 percentage points of recall at 4.19 points more FPR; J and AUC fell and Brier worsened. Twenty of forty users had AUC degradation greater than 0.01.
- **Within anatomical MB:** personal memory improved macro Brier from 0.1820 to 0.1780, but macro recall and macro AUC fell. Pooled AUC increased; these aggregations answer different ranking questions. A better probability error did not establish better lapse detection or useful warnings.
- **With two warnings per rolling seven days:** the candidate caught 992 events with 175 false warnings; GBDT caught 968 with 148. An extra 24 caught events came with 27 extra false warnings. These are pooled counts, not macro percentages.
- **At the high-recall point:** the candidate reached 93.96% macro recall with 85.49% macro FPR. Warning coverage under the two-warning limit was only 29.90% of events. Raising sensitivity alone did not solve the product problem.

Exact primary values, paired intervals and the high-recall excerpt are linked in [results](docs/results-v9.md).

## Established findings and open hypotheses

The historical experiment demonstrates that this implementation can fit an MB population predictor independently of GBDT and update personal state from delayed observed feedback. It also records a lower Brier error after anatomical MB personalization relative to that MB's own population forecast. These are findings on this synthetic dataset.

It does **not** establish an overall advantage of the candidate, an advantage caused by biological anatomy, generalization to real people, prevention of lapses, or a measured memory/throughput budget for a million users. It does not reject every possible personal memory or sparse representation. The next task is to locate the components and assumptions limiting this candidate before seeking independent confirmation.

**Personal-state budget:** the current MB memory has three 2,020-value float64 arrays, totaling 48,480 bytes (47.34 KiB) of numerical buffers, plus pending snapshots, gate history, event IDs and metadata. This source-derived calculation is not a measured checkpoint or process size. A 16 KiB complete state has not been achieved. See [state accounting](docs/personal-state-budget.md).

## Start here

1. Read the [results](docs/results-v9.md) and [protocol](docs/protocol-v9.md). For discussion alone, open a research question with the relevant metric and a proposed control.
2. For reproduction, follow the route below and report installation or execution failures. Keep a fresh output directory and all attempts. Reproduction is the first unresolved technical step for this extraction.
3. Pick one [roadmap question](docs/research-roadmap.md): first personal memory versus simpler corrections, then anatomy versus matched sparse controls, then reporting and warning policy. State the comparison and success criteria before implementation.
4. Treat improvements on the included development data as exploratory. Independent confirmation needs new untouched users and a frozen protocol. The repository currently provides neither a fresh cohort nor its generator.

See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting and contribution procedure. A focused negative experiment, a reproducibility report or a methodological critique is useful.

## Reproduction

These commands document the intended route from a source checkout; they have not been run on this extraction. The input bundle is in the repository, not in a Python wheel. Standalone reproduction and the contributor-rights workflow remain unverified.

```bash
git clone https://github.com/nutrifit-health/craving-engine.git
cd craving-engine
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m research_v9.experiment --output-dir runs/v9-first --workers 1
```

Use a new output directory for every attempt. The runner retrains two MB families and compares them with preserved GBDT predictions. It does not regenerate data, retrain GBDT or open a fresh holdout. `--workers 2` enables two isolated workers. Outputs include models, checkpoints and prediction logs, so disk usage exceeds the input bundle.

The [original environment](original-environment.json) records historical versions. Dependency ranges in `pyproject.toml` are proposed installation constraints, not established compatibility or bitwise reproducibility. Installation, training, tests and builds have not been run during public preparation. Copied historical tests are available for a separately authorized verification pass.

The runner loads pickle files only from its own fit stage. Hashes establish artifact consistency, not trust in a third-party pickle; do not substitute downloaded models.

Browse the [documentation index](docs/README.md) for all research and author materials.

## Repository map

| Path | Purpose |
| --- | --- |
| `research_v9/models.py`, `research_v9/types.py` | MB population fitting and fixed parameters |
| `research_v9/experiment.py`, `research_v9/bundle.py` | Standalone run and frozen input loading |
| `research_v8/context_representation.py`, `research_v8/memory.py` | Shared representation and personal memory |
| `research_v8/replay.py`, `research_v8/evaluation.py` | Chronological feedback, scoring and warning policy |
| `data/v9-frozen/` | Synthetic inputs, schema, folds, graph and archived GBDT predictions |
| `results/v9-2026-09-24/` | Selected original results; keep new runs separate |
| `docs/` | Protocol, interpretation, roadmap, release status and author notes |
| `extraction.json` | Source-run provenance and standalone changes |

There is no NutriFit service integration or real patient dataset. The synthetic generator, original backend encoder, GBDT training pipeline and full original archive are outside this snapshot. Editing raw JSONL does not regenerate the frozen feature matrices. Some helpers retain their original V6–V8 directory names; the public entrypoint is V9.

---

© 2026 NUTRIFIT LLC. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
