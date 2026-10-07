# Electronic contributor acceptance

[Русская версия](electronic-acceptance.ru.md)

Local configuration is enabled for agreement version 1.0. On publication, the workflow uses trusted `main` code and the ordinary `GITHUB_TOKEN`; it does not execute external PR code or require a contributor's separate token. No remote execution has been performed during preparation.

## Ordinary first contribution

1. Open a PR. The bot links the English assignment and grant-back license at an immutable repository revision and explains company ownership, commercialization and the noncommercial grant back.
2. Read both documents. If you own the rights or have authority to sign for their owner and agree, post the exact `/nutrifit-agree VERSION HASH_PREFIX` command supplied by the bot from your account. This intentional authenticated statement is the electronic signature. Do not print or scan anything.
3. The receipt records the public account ID, acceptance comment/time, exact agreement and license text, individual SHA256 hashes and a combined acceptance hash. The combined hash is SHA256 of JSON.stringify({agreement: agreementText, license: licenseText}) using the UTF-8 texts loaded from the immutable revision. Its first twelve hexadecimal characters appear in the command.
4. The coverage receipt records PR URL, current head and full commit SHA(s), author accounts, immutable commit links and time. Those commits identify the included files and patches. Confidential identity or authority evidence is handled separately when required.

Later covered submissions reuse that acceptance. A change to either accepted text changes the combined hash and requires fresh acceptance. Signature records and current commit coverage are distinct. A bot does not accept on behalf of a contributor or impersonate a company signatory.

## Exclusions and cases requiring separate clearance

After initial express acceptance, covered intentional submissions are assigned by default. To exclude a submission, notify maintainers before or with submission using `Rights-Assignment: excluded` and full commit SHA(s). This blocks automatic coverage; it does not reverse an effective earlier assignment or license company code commercially.

Silence, a fork, a local commit, issue, ordinary commit author field or GitHub Verified badge alone is not acceptance. Unidentified authors, bots, Co-authored-by, organizational rights and mandatory additional formalities require separate clearance before merge. Automated success records the defined steps; it does not independently establish copyright ownership.

## Optional reward

The company may offer a discretionary reward after evaluating the contribution. No payment, royalty, employment or revenue share is guaranteed by submission or merge. Approved terms are communicated separately in writing; payment details are requested privately afterward. Mandatory remuneration rights remain unaffected.

## Deployment and evidence

`.github/rights-config.json` sets active=true and points to the final English documents. Publication to main and Actions availability are necessary for the workflow to run. Configure `NutriFit / contributor rights` as a required merge status and restrict bypass/direct pushes; this is a GitHub setting, not something the YAML alone enforces. Remote merge protection remains a separate deployment step.

The company must retain agreement/license versions, receipts and contribution scopes in a durable archive; GitHub comments alone do not promise permanent retention. Search indexing can delay reuse of an existing receipt; rerun via a new comment without signing again. Workflow execution has not been tested during this preparation.

## Legal framework

[17 U.S.C. § 204(a)](https://www.copyright.gov/title17/92chap2.html) requires signed writing for the relevant copyright transfer. The [E-SIGN Act, §§ 7001, 7006](https://www.govinfo.gov/content/pkg/USCODE-2024-title15/html/USCODE-2024-title15-chap96.htm) recognizes electronic processes adopted with signing intent; it does not turn an arbitrary commit into consent. [37 CFR § 201.4](https://www.copyright.gov/title37/201/37cfr201-4.html) addresses electronic-signature evidence in recordation.

Mandatory rights, owner authority and jurisdiction-specific requirements remain applicable. Separate required instruments must be obtained before merging affected contributions. English legal texts govern; this document explains the procedure.
