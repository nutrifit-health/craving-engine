# Personal state: code-derived size, not a measured storage benchmark

[Русская версия](personal-state-budget.ru.md)

The current V9 implementation does **not** establish a 16 KiB complete per-user state. The often-used example of 4,096 float32 values describes a hypothetical single vector. V9 has a different dimension, dtype and state structure.

## Numerical buffers in the current anatomical and rewired variants

`FixedContextRepresentation.encode()` in `research_v8/context_representation.py` appends an anchor coordinate to the 2,019-KC graph representation, yielding 2,020 coordinates. `score_records()` in `research_v8/replay.py` passes that encoded dimension into `EncodedMemory`.

`EncodedMemory.__init__()` in `research_v8/memory.py` creates three arrays:

```python
self.fast = np.zeros(dimension)
self.slow = np.zeros(dimension)
self.exposure = np.zeros(dimension)
```

No dtype is supplied; [NumPy specifies float64 as the default](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html). Restore also uses `np.asarray(..., dtype=float)`. The three numerical buffers therefore require:

```text
3 arrays × 2,020 values × 8 bytes = 48,480 bytes = 47.34375 KiB
```

This is a calculation from the source and the frozen graph dimensions, not a measured process allocation or serialized checkpoint size. It excludes array/Python object overhead, transient computation, and other user state. It concerns one MB personal variant; research runs evaluate variants separately.

## Other state is necessary

The current checkpoint also includes:

- `pending`: sparse decision-time snapshots needed when delayed feedback arrives;
- `observations`: recent paired forecast/outcome history used by the per-user gate, with a configured 60-day window;
- `seen`: event identifiers for duplicate prevention; the current implementation retains these without a size bound;
- `clock`, representation identity, configuration and format metadata.

V9 uses `head_mode=off`, so it does not copy a global readout into each user's memory. The user's three arrays start at zero; the learned population probability is computed separately and supplied to `predict()`.

`export_state()` converts vectors to lists and writes JSON. The archive stores gzip-compressed checkpoints. Neither JSON length nor compressed size equals the sum of numeric buffer bytes; they depend on values and accumulated history. No complete memory or storage benchmark has been performed during public preparation.

## What a 16 KiB target would require

The following are numerical-buffer budgets for hypothetical formats, not implemented optimizations or quality results:

| Three 2,020-value arrays | Numeric bytes | Numeric KiB |
| --- | ---: | ---: |
| Current float64 | 48,480 | 47.34375 |
| Proposed float32 | 24,240 | 23.671875 |
| Proposed float16 | 12,120 | 11.8359375 |

Switching only to float32 would still exceed 16 KiB before other state. A float16 format could place the numerical portion below that target, but would leave 4,264 bytes of a 16,384-byte budget for all other state if the target covers the complete user record. The current unbounded `seen` collection prevents a fixed lifetime guarantee.

Define the target first: numeric parameters, compressed durable record, or complete resident state. A proposed experiment can then compare reduced dimension, quantized storage or arithmetic, and bounded event/history storage. Precision changes need evaluation of decay, updates, probabilities, thresholds and admission; smaller data types are not evidence of preserved quality. Do not silently change the preserved V9 implementation to satisfy the article's number.

For the article, the supportable statement is: **the current MB variant has about 47.34 KiB of personal numerical buffers, plus variable auxiliary state; a 16 KiB personal-state budget is an open optimization target.** Million-user throughput and total storage remain unmeasured.
