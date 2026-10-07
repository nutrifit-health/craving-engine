---
name: Reproducibility report
about: Report an attempted reproduction
title: ""
labels: ""
assignees: ""
---

[Русский шаблон](https://github.com/nutrifit-health/craving-engine/blob/main/.github/ISSUE_TEMPLATE/reproduction.ru.md)

## Revision and environment

Repository commit, Python version, dependency versions, operating system and worker count.

## Inputs and run

Input bundle revision, command used, new output directory, and complete or failed attempt status.

## Expected and actual result

Describe the difference. Include primary metrics with FPR and calibration where available.

Compare all six primary result rows, the scalar baseline selection, frozen thresholds, admission decision and two-warning counts. Report numerical differences and failed attempts; do not tune settings to match the archived scores. The public snapshot reproduces MB fitting and replay against preserved GBDT predictions, not the generator or GBDT training.

## Logs or artifacts

Attach only synthetic or sanitized information. Do not upload signed contributor agreements, tokens or real user data. A matching hash alone does not make a downloaded pickle trusted.
