# Research V10: did LIF and local plasticity help?

[Русский](results-v10.ru.md) · [README](../README.md) · [Protocol](protocol-v10.md)

Date: October 7, 2026. **The tested LIF + STDP implementation did not establish an advantage.** A scalar personal correction had lower probability error on the original task. Small effects in individual new scenarios did not meet the preregistered conditions. This finite experiment does not establish universal uselessness of spiking networks.

A useful positive result: ordinary personal memory also showed initial improvements after one or two new responses in the controlled task. Spikes were not necessary for that effect. Complete adaptation after two responses and real-user effectiveness were not established.

## Completed work

- Exact [ToxaBes listing](https://habr.com/ru/articles/1045696/) audited on CPU/MPS with seeds 7/17/27. We tested executable mechanisms, not the full 40-epoch MNIST training or reported accuracy.
- Standalone V9 reproduced: five principal macro metrics for all six models matched the archive exactly; admission remained negative.
- 35 fixed configurations screened on 10 tuning users, thresholds selected on 10 separate calibration users, 8 frozen variants including 2 matched controls evaluated on 40 previously examined V9 development users.
- Eight new controlled scenarios × 3 seeds × 40 users × 120 decisions: 115200 scenario decisions per variant. The same 120 synthetic users recur across scenarios; they are not 960 independent people.
- Main series: 239.68s. Independent replay reproduced all 8 × 7200 development rows exactly; 284 artifacts verified. Checks passed: 38 V8/V9 tests +31 subtests, 14 V10 mechanism tests, 7 author-listing tests, Ruff, sdist/wheel build, wheel imports and pip check.

## What was transferred

LIF implements current integration, leakage, threshold and reset. Five time/gain profiles and two learning rates were tested. Learning rules: error×rate, pair-based STDP, and STDP×prediction error. The latter is our engineering third-factor hypothesis, not a ready-made rule copied from the article.

Principal ablations share GBDT p0 and the original KC input representation. Personal memory, delays, correction bounds and gating are shared; matched controls use exactly the same LIF profile and rate. Standalone MB V9 was reproduced separately; its historical results are not redefined as a GBDT hybrid.

V10 differs from the author's network: fixed KC inputs, 32 deterministic ticks per event, a personal readout and stored eligibility for delayed responses. The article uses cortical modules, Poisson encoding and inter-module plasticity. Our post spikes derive from population p0. We do not implement recurrent spiking state between daily events or encoder training with surrogate gradients. The global model is not retrained as an SNN.

## Exact-listing audit

| Check | Observation | Implication |
| --- | --- | --- |
| Gradients / AdamW | No gradient for input_scale or column weights; readout/interweights update | Compatible with a fixed reservoir, not end-to-end training of every named parameter |
| Additional rate loss | requires_grad=False | It contributes no regularizing gradient |
| Repeated evaluation | Logits changed about 0.0015–0.0028; restoring EMA removed the change | spike_rate_avg mutates during evaluation; tested batches had no argmax changes |
| Short STDP run | Weight changes about 3–5×10⁻⁸ across 3 batches; no float32 logit difference | Does not attribute reported accuracy to STDP or rule out a long-run effect |
| Connectivity intervention | Interweights affect output features, not column inputs/spikes/mem/syn | No feedback of this modulation into LIF dynamics |
| Connection formation | Maximum correlation 0.63501, default threshold 0.8 | Default formation threshold is unreachable; 1000 maximal updates checked |

Exact measurements, listing SHA256 and limitations are in the [audit](../results/v10-author-audit/summary.json). The full third-party listing is not redistributed.

## Original task: probabilities and warnings

| Model | Brier ↓ | AUC ↑ | Recall | FPR | J | Caught / false, 2 per 7d |
| --- | --- | --- | --- | --- | --- | --- |
| Population GBDT | 0.173208 | 0.739105 | 42.48% | 17.57% | 0.249087 | 993 / 170 |
| Scalar | 0.169199 | 0.735216 | 43.22% | 19.29% | 0.239319 | 1004 / 162 |
| KC direct | 0.169841 | 0.735187 | 43.19% | 19.52% | 0.236731 | 1002 / 170 |
| LIF + direct | 0.170744 | 0.735160 | 43.54% | 19.52% | 0.240252 | 1010 / 168 |
| LIF + STDP | 0.169340 | 0.736832 | 43.11% | 19.28% | 0.238356 | 1003 / 165 |
| LIF + error-modulated STDP | 0.171681 | 0.738213 | 42.08% | 18.33% | 0.237487 | 982 / 169 |

Brier is primary here; lower is better. Scalar memory reduces Brier but lowers AUC/J relative to population. Better probability error does not establish better warnings. V10 selects thresholds on 10 users versus 20 in V9, so cross-version recall/FPR differences cannot be attributed solely to models.

LIF+rSTDP versus KC direct: ΔBrier **+0.001840 [+0.000439; +0.003532]**; versus scalar: **+0.002482 [+0.000630; +0.004846]**; versus matched LIF direct: **+0.000937 [+0.000098; +0.001952]**. Positive differences indicate deterioration.

![Development probability error](../results/v10/plots/development-brier.png)

## New controlled scenarios

| Scenario | ΔBrier vs KC direct, 95% CI | ΔBrier vs matched LIF direct |
| --- | --- | --- |
| stable | -0.000075 [-0.000252; +0.000093] | -0.000194 [-0.000398; +0.000009] |
| abrupt | -0.000011 [-0.000369; +0.000346] | -0.000300 [-0.000651; +0.000071] |
| gradual | -0.000192 [-0.000505; +0.000118] | -0.000467 [-0.000789; -0.000146] |
| transient | -0.000096 [-0.000297; +0.000095] | -0.000228 [-0.000490; -0.000001] |
| noise15 | -0.000471 [-0.001017; -0.000023] | -0.000738 [-0.001354; -0.000227] |
| delay72 | +0.000229 [-0.000027; +0.000499] | -0.000316 [-0.000592; -0.000029] |
| report50 | +0.000889 [+0.000522; +0.001269] | -0.000138 [-0.000493; +0.000195] |
| report10 | +0.000000 [+0.000000; +0.000000] | +0.000000 [+0.000000; +0.000000] |

Differences use gated Brier with paired 95% user-bootstrap intervals. Noise/delay/reporting interventions modify abrupt, their reference scenario. Under V9 timestamp ordering, a 72-hour response first influences the 96-hour daily prediction.

The noise15 improvement against KC direct is small, below the 0.002 threshold and exploratory. Report50 is worse. In report10 the gate does not allow a personal correction and gated forecasts coincide. Selecting only the favourable scenario would not support an overall claim.

![Scenario paired differences](../results/v10/plots/scenario-deltas.png)

## One or two new responses

Models already have 60 days of experience at the abrupt change. Comparing only with population does not isolate new responses. A post-evaluation diagnostic, without retuning, therefore compares each model with its own copy receiving no post-change responses. Pre-change predictions match exactly.

| Model | After 1 response: ΔBrier, 95% CI | After 2 responses: ΔBrier, 95% CI |
| --- | --- | --- |
| Scalar | -0.000438 [-0.001879; +0.001636] | -0.002445 [-0.004448; -0.000649] |
| KC direct | -0.002873 [-0.004805; -0.001106] | -0.005116 [-0.008713; -0.001837] |
| LIF + direct (matched) | -0.002773 [-0.004570; -0.001232] | -0.005580 [-0.009141; -0.002305] |
| LIF + error-modulated STDP | -0.002293 [-0.004066; -0.000744] | -0.004515 [-0.007318; -0.001882] |

Negative differences favour continued learning over its own control. Several methods benefit after 1–2 responses in this synthetic task. The effect is **not specific to spikes**. Direct rSTDP-versus-KC comparison after two responses: ΔBrier **+0.000742 [-0.000748; +0.002322]**; no advantage established.

For entirely new personal memory, gated forecasts after 1–2 responses equal population: at least 12 effective observations and other admission conditions are required. Ungated scores are diagnostics; their configurations were selected by gated Brier. Better performance than a frozen memory does not establish complete adaptation; no adequate-quality threshold for such adaptation was prespecified.

![Adaptation against generator probability](../results/v10/plots/adaptation.png)

## Mechanistic diagnosis

Post-evaluation geometry, without retuning: mean eligibility/rate cosine=-0.210; directions oppose in 75.6% of events. Pair-based STDP need not align with the logistic readout's error-reducing direction. Multiplying an arbitrary signed trace by prediction error does not guarantee alignment. This motivates a better-founded third factor; it is not an explanation of every SNN.

The selected LIF code activates 53.5 units on average versus 101 original KC units. LIF also changes feature geometry; fewer spikes alone do not demonstrate an advantage.

All variants cap corrections at ±0.7 logit. Evaluator-only oracle diagnostics quantify error unreachable under this bound. Models never receive the hidden probability. Relaxing the bound after viewing results would constitute a new experiment.

## Cost and state

| Model | Numeric bytes / user | JSON checkpoint bytes, min–max | Replay µs / decision | Encoding µs / decision |
| --- | --- | --- | --- | --- |
| Scalar | 24 | 14,388–17,125 | 77.2 | 0.0 |
| KC direct | 48480 | 73,271–103,740 | 135.1 | 0.0 |
| LIF + direct | 48480 | 67,758–93,452 | 120.5 | 355.0 |
| LIF + STDP | 48480 | 73,262–104,396 | 136.2 | 376.0 |
| LIF + error-modulated STDP | 48480 | 68,108–93,397 | 121.2 | 321.0 |

Measured locally on macOS arm64, one NumPy CPU thread, 180 V9 days. Replay includes Python orchestration and final state serialization but excludes fitting and computing shared GBDT p0. Encoding is additional LIF work; the common KC transformation is excluded from that column. This is not an SLA or a million-user load test.

KC/LIF retain three 2020-value float64 arrays: 48480 bytes (47.34 KiB). Complete JSON is larger and depends on history/pending/seen. V10 also stores pending eligibility. Process heap, compressed checkpoints, quantization and production storage were not measured. Complete 16 KiB state was not achieved.

## Reproducibility and limitations

The first independent checker required exact `sigmoid(logit(p0)) == p0`. Measured maximum round-trip error was 1.11×10⁻¹⁶; gated forecasts matched exactly. Only the checker changed to 1e−14 tolerance for the ungated round-trip. [Original failure](../results/v10/verification/recheck.log) and [successful retry](../results/v10/verification/recheck-retry1.log) are preserved. Models, selected settings and the main archive were unchanged. The corrected checker has a separate SHA256; the main snapshot retains its earlier version.

All data are synthetic. V9 development is previously examined. New seeds test a specified generator; settings transfer from 2020 to 33 coordinates without retuning. Complete calibration labels are idealized. Secondary comparisons have no multiplicity correction. Conclusions cover the selected LIF range and readout only. Full MNIST training, neuromorphic hardware, clinical benefit and prevention were not tested.

## Decision

`recommendation_supported=false`: conditions failed against matched LIF direct, KC direct and scalar. This SNN variant is not justified as an improvement to the predictor. The recommendation led to a useful experiment and exposed learning-signal design issues. A separate future study could prespecify a better-founded eligibility/modulator and temporal state across events against ordinary memory. That requires a new protocol and evaluation cohort, not further tuning on V10 results.

## Reproduction

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v10.txt
python -m pip install --no-deps -e .
python -m pytest research_v8 research_v9 research_v10/test_mechanisms.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m research_v10.experiment --output-dir runs/v10-new
python -m research_v10.verify --run runs/v10-new --output-dir runs/v10-new-check
python -m research_v10.followup --run runs/v10-new --output runs/v10-new-counterfactual.json
```


For the author audit, save article HTML temporarily, read the extracted listing and inspect `python -m research_v10.author_audit --help`. Execution is restricted to the reviewed SHA256 and requires `--execute-reviewed-snapshot`. Listing tests read `V10_AUTHOR_SOURCE` and skip if the temporary file is missing. V10 runtime needs no Torch; Torch is used for the separate author audit.

Use fresh output directories. [Exact dependencies](../requirements-v10.txt) describe the tested platform, not universal compatibility.

## Artifacts

- [Frozen configurations and thresholds](../results/v10/run-20261007/frozen.json)
- [All tuning attempts](../results/v10/run-20261007/tuning.json)
- [Main summary](../results/v10/run-20261007/summary.json)
- [Paired development comparisons](../results/v10/run-20261007/development_comparisons.json)
- [Independent verification and admission](../results/v10/recheck-20261007-retry1/verification.json)
- [Paired few-shot comparisons](../results/v10/recheck-20261007-retry1/few-shot-paired.json)
- [Counterfactual / cold-start control](../results/v10/verification/counterfactual.json)
- [Representation diagnosis](../results/v10/verification/representation-diagnostic.json)
- [Author audit and provenance](../results/v10-author-audit/summary.json)
- [Complete artifact hashes](../results/v10/run-20261007/artifact_manifest.json)
