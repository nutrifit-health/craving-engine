# Research roadmap

[Русская версия](research-roadmap.ru.md)

October 7 update: standalone V9 was reproduced; [V10](results-v10.md) tested LIF/STDP, limited reporting scenarios and JSON state size. Remaining items below are research directions; new parameters must not be selected using already examined evaluation outcomes.

These are proposed next steps, not completed experiments or a preregistered successor to V9. The [recorded V9 candidate failed admission](results-v9.md). The immediate task is to understand that outcome before attempting independent confirmation. The included development cohort is already known.

## 0. Reproduce the standalone snapshot

Start from the [README route](../README.md#reproduction), recording source revision, environment, input bundle and a fresh output directory. First resolve installation or execution failures; do not change model settings to make scores match. Compare the six primary rows, calibration-selected scalar baseline, frozen thresholds, admission decision and warning counts with the preserved files. Document numerical differences rather than assuming bitwise identity across library versions.

Useful contribution: a complete reproduction report or a minimal, documented packaging fix. Standalone V9 was reproduced on October 7; principal metrics matched exactly. [Evidence](../results/v10/verification/v9-reproduction-comparison.json). No full generator or GBDT fit can be reproduced from this snapshot.

## 1. Separate the value of memory from the value of the shared model

**Question:** why does anatomical MB personalization lower macro Brier and macro AUC while increasing pooled AUC, with lower macro recall at the selected operating points? Per-user and pooled ranking are different evaluation questions.

Keep one population forecast and representation fixed. Compare no personal correction, a scalar bias, a direct linear personal residual, and the existing sparse fast/slow memory. Do not attribute a comparison between MB and GBDT entirely to personalization. Inspect memory behavior in `research_v8/memory.py`, causal replay in `research_v8/replay.py`, and population fitting in `research_v9/models.py`.

Choose the training/calibration tuning budget and primary operating constraint in advance. Use the same calibration FPR constraint or warning burden across variants; freeze thresholds before development. Report actual development FPR because a calibration bound does not force identical FPR on new users. Include per-user AUC degradation, Brier, recall, J, warning coverage and false warnings per caught event. Separate cold-start from later experience with predeclared time windows.

A useful result identifies whether personal information helps, whether only calibration improves, or whether memory harms ranking. A negative result should remain visible. New ablations of learning rate, decay, familiarity, anchor and gating must be reported separately rather than combined into a best-scoring configuration after seeing development.

## 2. Test the anatomical prior against matched controls

**Question:** does the ALPN → KC graph add value beyond a generic sparse representation?

The existing rewiring preserves binary degrees and each KC's weight multiset, but not ALPN output strength or all graph motifs. The anatomical and rewired variants also have different achieved FPR. That comparison alone cannot establish anatomy's practical benefit.

Add random sparse projections and a simpler feature-based representation. Match dimension, sparsity, scale, readout regularization and fitting budget where possible; disclose remaining differences. Prespecify multiple mapping and rewiring seeds and report all of them. Keep the population-only comparison separate from the personal-memory comparison. Inspect `research_v7/connectome.py` and `research_v8/context_representation.py`.

A useful result shows a stable benefit at comparable operating constraints, or shows that a simpler representation is sufficient. Do not select the best seed on known development users and call it confirmation.

## 3. Examine reporting assumptions and warning usefulness

**Question:** can performance survive incomplete, selectively reported and delayed outcomes, while respecting a warning limit?

The current reporting probabilities (0.8 positive / 0.7 negative) and 20-hour delay are simulation assumptions. Full evaluator labels idealize calibration and scoring. Study the distinction between event risk and reporting likelihood; specify assumptions behind observation modeling or weighting. Missing feedback must remain unknown, not negative.

Evaluate warning coverage, false warnings per caught event and timing alongside classifier metrics. A lower threshold can increase detection while the rolling limit blocks most warnings. The current implementation is in `research_v8/evaluation.py` and `research_v8/replay.py`.

The frozen bundle cannot generate alternative reporting mechanisms or new causal feature matrices. Such stress experiments first need an explicit generator/encoder and a separate protocol. Do not relabel the current JSONL or insert future information into `X` and call the result a new realistic cohort. Offline warning coverage does not measure prevention.

## 4. Seek independent confirmation only after exploration

Freeze a candidate, baselines, outcome definition, available-at-decision features, tuning rules, thresholds, seeds and admission rules before opening a new cohort's outcomes. Use users absent from all prior tuning and evaluation. A new synthetic cohort tests generalization under its declared generator; it still cannot establish human effectiveness.

Real-user validation requires a separately designed collection and outcome-validation process. No real health data should be submitted here. Prevention requires a prospective intervention study, separate from predicting events.

## 5. Measure the architecture's operational cost

Once a candidate has useful predictive behavior, measure serialized personal state, pending snapshots, history, concurrent updates, inference/update latency and throughput. Current source-derived accounting gives 48,480 bytes for three float64 vectors per MB user, before auxiliary state; see [personal-state budget](personal-state-budget.md). A hypothetical 4,096-float32 vector is 16 KiB; that is not the measured size of a V9 user checkpoint. No million-user benchmark has been performed. A proposed 16 KiB optimization must define what it includes and evaluate quality separately; it is not an implemented feature.

## How to propose a first contribution

Open a research question stating the hypothesis, fixed baseline, one change, data needed, success/failure evidence and known limitations. Link the relevant preserved result. For executions, include every attempt and a new run directory; leave `results/v9-2026-09-24/` unchanged. Follow [CONTRIBUTING.md](../CONTRIBUTING.md). The broader [open questions](open-questions.md) provide additional context.
