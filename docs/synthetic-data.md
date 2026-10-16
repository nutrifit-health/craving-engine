# New synthetic datasets and active development

[English](synthetic-data.md) · [Русский](synthetic-data.ru.md)

This standalone repository is the only active Craving Engine research workspace.
`research_v9/` and `research_v10/` retain the existing experiments. `synthetic_data/`
contains selected tools for new datasets extracted from the private NutriFit archive.
The migration was verified on October 16, 2026 with the old archive absent.

## Retained capabilities

- `population.py`: new grid-sampled profiles, without the previously explored persona catalog.
- `simulator.py`: 180-day trajectories with missing observations, latent outcomes and regime changes.
- `contexts.py`: record availability, observation windows and user history.
- `features.py`: observed features, 7/28-day history and feedback received by decision time.
- `quality.py`: built-in synthetic observation consistency diagnostics.
- `generate.py`: separate datasets with seeds, schemas, source snapshots and a manifest.
- `nutrifit_encoder.py`: optional explicit access to a local NutriFit encoder.

Historical models, HTTP service, runners, known diagnostic personas, logs and
checkpoints were not imported. Full history remains in private Git;
[selected source provenance](../research-migration.json) is recorded separately.

## Standalone generation

Install this Python package as described in the main README first. The command
below is a reference example, not an executed experiment. Freeze your own protocol,
cohort sizes and seed before research.

```sh
python -m synthetic_data.generate --output runs/new-corpus --seed 2026101601 --users 60 20 40
```

This creates new training/calibration/development cohorts with 180 decisions per
user. The output directory must not exist; `data/` and `results/` are protected.
An interrupted run retains an `.incomplete` marker.

Each cohort produces:

- `*.X.npy`: current observed features;
- `*.history.X.npy`: current features, prior history and available feedback;
- `*.records.jsonl`: row identifiers, decision times and feedback availability times;
- `*.evaluation.jsonl`: complete synthetic outcomes for the evaluator only;
- `*.contexts.jsonl`: observed contexts in the same row order.

Shared `schema.json` and `manifest.json` record feature order, units, seeds, cohort
sizes, checksums and environment versions. `sources/` preserves generation source.
An outcome in records must not be used before feedback_at. Missing feedback stays
missing. Hidden outcomes and profile parameters are excluded from X. Reporting
rates 0.8/0.7 and the 20-hour delay are scenario assumptions, not real-user estimates.

## Optional NutriFit encoder features

Owners of a separate local NutriFit checkout may use:

```sh
python -m synthetic_data.generate --output runs/new-corpus-with-pn --seed 2026101601 --users 60 20 40 --backend-root /absolute/path/to/NutriFit/apps/backend
```

This requires installed backend dependencies, local `tsx` and the
`scripts/craving-research/encoder.ts` adapter. The client installs nothing, does not
search for neighboring repositories and does not access network services or databases.
The TypeScript adapter stays in private NutriFit beside its contracts. This mode
adds 16 PN features; the standalone mode contains observed features only. The mode
and schema are recorded in the manifest. Integration metadata may contain local
paths; do not automatically publish newly generated runs.

## Limits

This prepares data for future research. It is not a trained model, blind evaluation
or V9 reproduction. New identifiers and seeds change trajectories. Do not replace
`data/v9-frozen` with this output: model comparisons require a separately frozen
protocol and runner. V10 keeps its own controlled-scenario generator. Synthetic
formulas establish neither clinical realism nor benefit to people.

The removed archive remains available to the private NutriFit owner at revision
`27d3124cb403765ae50c75d43ad1c7b5fd9b2170`, path
`apps/backend/research/craving-engine-archive`. Recovering historical materials
does not require restoring a second active research workspace.

## Migration checks

On October 16, 2026, the Python suite passed 65 tests and 31 subtests, including
13 migration cases. Seven unrelated author-listing audit tests were skipped
because their external reviewed source snapshot was absent. The checks cover
repeatable artifacts, cohort separation, delayed feedback, unavailable records,
hidden outcomes, output protection and the real optional TS encoder. The backend
encoder's existing 29 tests also passed. Python lint and formatting, backend and
adapter typechecks, and scoped adapter lint passed. The full backend lint also ran and reported
1,117 pre-existing errors across 133 unchanged files; those unrelated files were
not modified by this migration.

Both source and wheel distributions were built. The installed wheel generated
540 rows from a temporary directory outside the checkout. The old archive was
absent during all these runs. This verifies the engineering migration; no model
was retrained and no new predictive or clinical result is claimed.

From this repository, with test dependencies installed:

```sh
python -m pytest tests research_v8 research_v9 research_v10 -q -rs
ruff check synthetic_data tests
ruff format --check synthetic_data tests
python -m build --no-isolation
```

Set `NUTRIFIT_BACKEND_ROOT` to your local `apps/backend` directory to include the
optional integration test; otherwise that case is explicitly skipped. In the
private checkout, run these commands from `apps/backend/scripts/craving-research`:

```sh
../../node_modules/.bin/tsc -p tsconfig.json
node ../../../../scripts/run-eslint.mjs encoder.ts types.ts --max-warnings=0
```
