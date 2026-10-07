# Open research questions

[Русская версия](open-questions.ru.md)

The [roadmap](research-roadmap.md) orders these questions into proposed next steps. Start by reproducing the standalone snapshot, then separate the effects of memory and representation. None of the proposed experiments below has been completed in this public extraction.

## Does personal memory add value beyond a scalar correction?

The anatomical personal model performed worse than its population-only variant on several primary metrics. Compare a scalar bias, a direct linear personal residual and sparse memory at a predeclared common FPR or warning burden. Separate the value of personalization from the value of the population classifier. Include per-user degradation, not just average recall.

## Is anatomical connectivity useful?

The partial ALPN → KC graph is a structural prior, and the human feature mapping is artificial. Add random sparse projections and controls matched for dimension, sparsity, scale, regularization and fit budget. Prespecify multiple mapping and rewiring seeds. Use confidence intervals and avoid choosing the best seed on known development users.

## Can calibration handle selective reporting?

Positive and negative outcomes have different simulated reporting probabilities. A model trained on observed feedback may learn reporting bias. Study inverse probability correction or explicit observation modeling under stated assumptions, and measure calibration drift when those assumptions are wrong. Do not treat absent reports as confirmed negatives.

## Which learning and gating choices matter?

Separate representation geometry, residual learning rate, fast/slow decay, familiarity and gating. The anchor coordinate changes with context norm. Use ablations and fixed tuning budgets on training/calibration data to locate failures. A gate that lowers average loss is not a guarantee of no harm for every user.

## What does a useful warning mean?

The product would limit warning frequency, so classifier recall alone is insufficient. Prespecify warning coverage, false warnings per caught event, time before an event, and consequences of missed or excessive warnings. Evaluate prevention separately with an appropriate prospective design; offline detection cannot prove it.

## How should a new dataset be designed?

Current users and data-generating rules are synthetic and already explored. First make assumptions, reporting mechanics and outcome definition explicit. A new confirmation cohort must be untouched, with user separation, causal feature availability and threshold selection fixed before opening its outcomes. Real-user data requires a separately designed consent, privacy and outcome-validation process; this repository does not contain or authorize that process.

## What would million-user support cost?

The experiment has a shared representation and per-user state, but no demonstrated million-user throughput or storage budget. Measure the complete checkpoint, pending snapshots, serialization, concurrency and update frequency. One hypothetical 4,096-element float32 vector is 16 KiB. The current V9 MB has three 2,020-element float64 arrays: 48,480 bytes of numerical buffers alone, plus variable auxiliary state. See [code-derived accounting](personal-state-budget.md); compressed storage size has not been measured.

Start with a reproducibility report or a proposed protocol. Keep negative results. Refer to [contribution terms](../CONTRIBUTING.md) before submitting code and distinguish an exploratory improvement from an independently confirmed result.
