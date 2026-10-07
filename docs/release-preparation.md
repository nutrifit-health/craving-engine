# Publication preparation status

[Русская версия](release-preparation.ru.md)

This initial source publication dated October 7, 2026 contains the standalone research snapshot, final English terms and enabled electronic-acceptance configuration. The initial publication was unverified. During V10 on October 7, installation, tests, build and standalone V9 reproduction passed; contributor-rights workflow execution remains unverified. Publishing the source does not establish runtime readiness.

## Prepared locally

- Independent repository on main with origin nutrifit-health/craving-engine; its nested directory is ignored by the NutriFit parent.
- Selected synthetic inputs, graph, folds, frozen feature matrices and GBDT predictions; no real patient dataset or company formation PDFs.
- V9 MB fitting/replay, standalone loader, dependency metadata and selected historical results, including a copied high-recall excerpt.
- English and Russian documentation explaining goals, negative admission, state accounting and next steps.
- Final English noncommercial license and contributor assignment; active=true in rights-config.json, with both texts retained in acceptance receipts.
- English and Russian issue/PR templates. Article fragments are excluded from this publication.

## Remaining deployment and verification

1. The initial source publication was authorized by the owner. Its revision is recorded in the repository Git history.
2. After publishing to main, GitHub Actions must be available. Configure NutriFit / contributor rights as a required merge status and restrict bypass/direct pushes. The local YAML does not set those repository rules; they have not been configured remotely.
3. Installation, applicable tests, build and standalone reproduction were explicitly requested and completed. [Principal V9 metrics matched the archive exactly](../results/v10/verification/v9-reproduction-comparison.json). Actual contributor-rights workflow verification remains separate.
4. Record the released revision and actual execution evidence when available. Resolve observed portability or numerical differences without overwriting historical results. The original V9 generator, backend encoder and GBDT training remain outside this snapshot; V10 includes its own controlled adaptation-scenario generator.

The original V9 run has historical checks and a negative result. Publishing that research does not require a successful candidate. Calling the extraction or electronic process verified requires evidence for those specific components.
