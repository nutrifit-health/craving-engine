# Participation

[Русская версия](CONTRIBUTING.ru.md)

Start with the [README](README.md), a [roadmap question](docs/research-roadmap.md), a reproducibility report or a proposed comparison. Explain expected evidence before making a substantial change. A negative experiment or a methodological critique is useful.

## Experiment reports

Record source revision, environment, input origin, splits, thresholds, training inputs, observed versus evaluator labels, warning policy and every attempt. Report recall with FPR, macro Brier/AUC, individual degradation and warning burden. Keep new runs in fresh directories; do not overwrite the historical V9 results.

The included development cohort is already known. Improvements found using its scores are exploratory and require new untouched users for confirmation. Do not choose the best seed or threshold on development and call it a blind result. The snapshot includes MB fitting and comparison with archived GBDT predictions, not the generator, original encoder or GBDT training.

Submit only synthetic or sanitized information, never real health records, company formation papers, credentials or payment details.

## Code contributions

The repository process is configured for one express electronic acceptance of the English [Contributor Copyright Assignment Agreement](legal/CONTRIBUTOR-ASSIGNMENT.md) and its [grant-back license](legal/NUTRIFIT-NONCOMMERCIAL-RESEARCH-LICENSE.md). Upon publication with Actions enabled, a first PR receives the exact agreement links and acceptance command from the bot. Read both documents and post that command yourself as your electronic signature. No printing or scans are required. Later covered PRs reuse the recorded acceptance while the terms remain unchanged; changed agreement or license text requires fresh acceptance.

The agreement transfers transferable economic copyright in identified intentional contributions to NUTRIFIT LLC, which may commercialize them. You retain actual authorship and the stated noncommercial grant back; mandatory rights remain. A buried link, ordinary commit, issue or silence alone is not consent. Reading and discussion do not assign your rights.

After initial acceptance, covered submissions are assigned by default. To exclude a submission, notify maintainers before or with submission using `Rights-Assignment: excluded` and full commit SHA(s). Excluded work requires separate rights clearance before merge and does not reverse an already effective transfer.

Identify preexisting and third-party material and rights held by an employer or university. Unidentified authors, co-authored changes and organizational ownership require separate consideration. Public receipts record account identity, accepted texts, PR and commits. Confidential authority and identity evidence is handled privately when required. The automated status records the defined process; it is not proof of legal ownership.

See [electronic acceptance](legal/electronic-acceptance.md) for details. The local configuration is enabled; remote execution and merge protection are separate deployment steps documented in [release preparation](docs/release-preparation.md).

## Recognition

After evaluating a contribution, NUTRIFIT LLC may offer a voluntary payment or other recognition. The company decides whether to offer a reward, its amount and terms. Submission, assignment, merge or commercialization does not guarantee payment, royalties or revenue share. An approved reward is communicated separately in writing; payment details are handled privately afterward. Mandatory compensation rights remain unaffected. See agreement section 4.1.
