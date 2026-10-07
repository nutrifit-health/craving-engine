[Русский](protocol-v10.ru.md) · [Documentation](README.md)

# Craving Engine V10: LIF and local plasticity

Protocol dated 2026-10-07, fixed before V10 evaluation. [Research task](https://gov.nutrifit.health/tasks/6ac6b3a156965bd66f60880f), [canonical protocol](https://gov.nutrifit.health/documents/6ac6b48e56965bd66f608816), [previous V9](protocol-v9.md), [ToxaBes example](https://habr.com/ru/articles/1045696/).

## Question and scope

Do LIF dynamics and error-modulated pair-based STDP improve personal adaptation compared with the same memory without spikes? Audit the author's published implementation separately. This is a finite synthetic experiment, not a test of every SNN or clinical validation. A weight change after two observations is not proof of useful prediction.

## Sequential stages

1. Execute diagnostics against the exact reviewed article listing: gradients, parameter changes, repeated evaluation, traces between batches and actual connectivity. Preserve URL and SHA256; do not redistribute the full listing.
2. Validate the frozen V9 bundle, user separation, causal timestamps and observed versus evaluation labels. Reproduce V9 tests and benchmark in a separate run directory.
3. Implement an original mechanism ablation: common fixed encoder, personal readout and 32 LIF simulation ticks per event. Membrane and synaptic dynamics reset between events. Simulation-tick leakage is separate from forgetting in calendar days. This is not a reproduction of the author's cortical network.
4. Select configurations on the first 10 calibration users; select thresholds on the other 10. Freeze both before development evaluation. V9 development is previously examined synthetic data, not a new blind test.
5. Evaluate new controlled scenarios with fixed seeds and known changes. The generator supplies an exact shared population probability; individual coefficients are hidden from models. These are mechanistic personalization tasks, not realistic nutrition simulations.
6. Preserve all outcomes, costs, memory measurements, reproducibility evidence and bilingual reports. Do not retune after evaluation.

## Finite comparison matrix

The principal ablations share the frozen GBDT p0. Baselines: population, scalar memory and original KC direct memory. A separate reproduction retains standalone MB V9 results.

LIF profiles `(tau_mem, gain)`: `(5,0.1)`, `(20,0.03)`, `(20,0.1)`, `(20,0.3)`, `(80,0.1)`; `tau_syn=5`, threshold `2.5`, 32 ticks. Input scale is the median positive training KC activation, excluding the anchor.

Rules: direct prediction-error × rate; unmodulated pair-based STDP control; STDP × prediction error, an engineering three-factor hypothesis. Pre/post traces use tau20 ticks, Aplus1 and Aminus1.2. Traces decay and contribute before the current spikes are added, excluding simultaneous pairs. A deterministic accumulator of population probability produces post spikes. Signed eligibility is L2-normalized and saved until delayed feedback; it is not reconstructed using later weights.

Fast rates: 0.2 and0.8; slow rate0.15×fast. Other V9 constraints and forgetting remain identical. The original gate and familiarity define the primary policy. Ungated `p_learning` is a separate diagnostic policy. Per-family selection minimizes macro Brier on tuning users, ties broken by variant name.

Additional matched direct and unmodulated-STDP controls reuse exactly the selected rSTDP LIF profile and learning rate. Matched contrasts isolate the learning rule; independently selected families answer the practical model-choice question.

## New controlled scenarios

40 users ×120 daily decisions, change at day60. Calibration seed6101; evaluation seeds7101,7102,7103 with disjoint user IDs. Sixteen observed Gaussian contexts, ON/OFF representation plus anchor (33 coordinates), shared logistic p0; personal coefficients and offsets are hidden. Scenarios: stable, abrupt, gradual, transient, 15% feedback label noise, 72-hour feedback delay versus20, 50% and10% reporting versus80%. Reuse selected V9 configurations without tuning on new evaluation seeds. Freeze scenario thresholds using the separate calibration cohort. Report transfer across different representation dimensions as a limitation.

Noise, delay and reporting interventions are applied to abrupt, which is their reference scenario. V9's strict timestamp ordering is preserved: prediction precedes feedback with an equal timestamp. A72-hour response therefore first affects the96-hour daily prediction. Every model follows the same ordering.

## Metrics and decisions

Primary: macro Brier. Secondary: AUC, recall, FPR, Youden J, precision, individual regressions and notifications capped at2 per rolling7 days. Paired bootstrap resamples users, 2000 draws, seed20260923 inherited from V9; adaptation diagnostics use20261007. Report95% intervals; days are not independent samples. Primary mechanistic contrast: matched LIF+rSTDP versus LIF+direct. Practical contrasts: tuned non-spiking direct and scalar. Other comparisons are exploratory, without confirmatory claims from multiple comparisons.

A positive recommendation requires Brier reduction of at least0.002 with its difference CI wholly below0, FPR increase at most0.01, nondecreasing J and stable-scenario Brier degradation at most0.002. The improvement direction must recur on at least2 of3 new seeds. Failure means the recommendation is not supported in this implementation; it does not establish universal uselessness.

Report adaptation errors after0/1/2/5/10/20 actually received post-change outcomes, separately gated and ungated, with user/decision counts. Improvement after one or two observations alone does not establish adequate quality.

Controls: no feedback, shuffled feedback, timing boundaries, user isolation, serialization, deterministic repeats, STDP ordering/signs and exact-listing gradients. Costs: wall time with environment, numeric buffers and complete serialized state separately. Preserve failed attempts. No automatic optimization after negative evaluation; a new hypothesis needs a new protocol.
