const fs = require('node:fs');
const crypto = require('node:crypto');

const ACCEPTANCE_MARKER = 'nutrifit-rights-acceptance-v1';
const COVERAGE_MARKER = 'nutrifit-rights-coverage-v1';
const NOTICE_MARKER = 'nutrifit-rights-notice-v1';
const EXCLUSION = /^Rights-Assignment:\s*excluded\s*$/im;

function receipt(body, marker) {
  const match = body.match(/```json\n([\s\S]*?)\n```/);
  if (!body.startsWith(`<!-- ${marker} `) || !match) return null;
  try {
    return JSON.parse(match[1]);
  } catch {
    return null;
  }
}

function botComment(comment) {
  return comment.user?.type === 'Bot' && comment.user?.login === 'github-actions[bot]';
}

module.exports = async function run({ github, context, core }) {
  const config = JSON.parse(fs.readFileSync('.github/rights-config.json', 'utf8'));
  const repo = context.repo;
  const number = context.payload.pull_request?.number ?? context.payload.issue?.number;
  if (!number) return;
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: number });
  if (pr.state !== 'open' || pr.base.ref !== 'main') return;
  const sha = pr.head.sha;
  const detailsUrl = `${context.serverUrl}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`;
  const status = async (state, description) => github.rest.repos.createCommitStatus({
    ...repo, sha, context: config.statusContext, state, description, target_url: detailsUrl,
  });
  await status('pending', 'Recording contributor agreement coverage');
  try {
    if (!config.active) {
      await status('failure', 'Company adoption of final agreement is pending');
      core.notice('Юридические проекты не активированы; согласие не запрашивается.');
      return;
    }
    if (config.agreementPath.includes('.draft.') || config.licensePath?.includes('.draft.')) {
      throw new Error('Нельзя активировать передачу прав по проекту документа.');
    }
    const text = fs.readFileSync(config.agreementPath, 'utf8');
    const licenseText = fs.readFileSync(config.licensePath, 'utf8');
    const agreementHash = crypto.createHash('sha256').update(text).digest('hex');
    const licenseHash = crypto.createHash('sha256').update(licenseText).digest('hex');
    // Согласие покрывает одновременно передачу прав и точные условия grant-back.
    const acceptanceHash = crypto.createHash('sha256')
      .update(JSON.stringify({ agreement: text, license: licenseText })).digest('hex');
    const { data: reference } = await github.rest.git.getRef({ ...repo, ref: 'heads/main' });
    // Текст читается по тому же immutable commit, что использован при checkout.
    const checkoutSha = require('node:child_process').execFileSync('git', ['rev-parse', 'HEAD'],
      { encoding: 'utf8' }).trim();
    if (reference.object.sha !== checkoutSha) {
      throw new Error('Main изменилась; требуется новый запуск на актуальной версии.');
    }
    const agreementUrl = `${context.serverUrl}/${repo.owner}/${repo.repo}/blob/${checkoutSha}/${config.agreementPath}`;
    const licenseUrl = `${context.serverUrl}/${repo.owner}/${repo.repo}/blob/${checkoutSha}/${config.licensePath}`;
    const commits = await github.paginate(github.rest.pulls.listCommits, { ...repo, pull_number: number, per_page: 100 });
    if (!commits.length || commits.length >= 250) {
      throw new Error('Состав PR требует ручного оформления: пустой или достигнут API-лимит.');
    }
    const comments = await github.paginate(github.rest.issues.listComments, { ...repo, issue_number: number, per_page: 100 });
    const submittingAuthors = new Set([pr.user.id, ...commits.map(item => item.author?.id)
      .filter(id => id != null)]);
    if (EXCLUSION.test(pr.body ?? '') || commits.some(item => EXCLUSION.test(item.commit.message))
        || comments.some(item => submittingAuthors.has(item.user?.id) && EXCLUSION.test(item.body ?? ''))) {
      await status('failure', 'Contribution excluded from assignment; separate clearance required');
      return;
    }
    if (commits.some(item => /^Co-authored-by:/im.test(item.commit.message))) {
      await status('failure', 'Co-authored work requires separate rights-owner clearance');
      return;
    }
    const people = new Map([[pr.user.id, pr.user]]);
    for (const commit of commits) {
      if (!commit.author || commit.author.type !== 'User') {
        await status('failure', 'Unidentified author or bot contribution requires separate clearance');
        return;
      }
      people.set(commit.author.id, commit.author);
    }
    if (pr.user.type !== 'User') {
      await status('failure', 'An actual rights owner must accept the agreement');
      return;
    }
    const command = `/nutrifit-agree ${config.version} ${acceptanceHash.slice(0, 12)}`;
    const acceptanceKey = actorId => `${ACCEPTANCE_MARKER} actor-${actorId} ${acceptanceHash}`;
    const accepted = new Map();
    for (const person of people.values()) {
      const local = comments.find(item => botComment(item) && item.body?.startsWith(`<!-- ${acceptanceKey(person.id)} -->`));
      let recorded = local && receipt(local.body, ACCEPTANCE_MARKER);
      let recordedUrl = local?.html_url;
      if (!recorded) {
        // Поиск находит сохранённое согласие в предыдущих PR; индекс может обновляться с задержкой.
        const { data: found } = await github.rest.search.issuesAndPullRequests({
          q: `repo:${repo.owner}/${repo.repo} is:pr in:comments "${acceptanceKey(person.id)}"`, per_page: 100,
        });
        if (found.incomplete_results || found.total_count > 100) {
          throw new Error('Поиск согласий неполон; права требуют ручного рассмотрения.');
        }
        for (const issue of found.items) {
          const history = await github.paginate(github.rest.issues.listComments,
            { ...repo, issue_number: issue.number, per_page: 100 });
          const previous = history.find(item => botComment(item)
            && item.body?.startsWith(`<!-- ${acceptanceKey(person.id)} -->`));
          if (previous) {
            recorded = receipt(previous.body, ACCEPTANCE_MARKER);
            recordedUrl = previous.html_url;
            break;
          }
        }
      }
      if (recorded?.actor?.id === person.id && recorded.acceptance_sha256 === acceptanceHash
          && recorded.repository === `${repo.owner}/${repo.repo}`) {
        accepted.set(person.id, { actor: person, receipt: recordedUrl });
        continue;
      }
      const signed = comments.find(item => item.user?.id === person.id
        && item.user.type === 'User' && item.body?.trim() === command);
      const notice = comments.find(item => botComment(item)
        && item.body?.startsWith(`<!-- ${NOTICE_MARKER} ${acceptanceHash} -->`)
        && (!signed || Date.parse(item.created_at) <= Date.parse(signed.updated_at)));
      if (!signed || !notice) continue;
      const evidence = {
        repository: `${repo.owner}/${repo.repo}`, actor: { id: person.id, login: person.login },
        agreement_version: config.version, agreement_sha256: agreementHash,
        agreement_url: agreementUrl, agreement_text: text,
        license_sha256: licenseHash, license_url: licenseUrl, license_text: licenseText,
        acceptance_sha256: acceptanceHash,
        signature: { comment_id: signed.id, url: signed.html_url, text: signed.body,
          accepted_at: signed.updated_at, recorded_at: new Date().toISOString() },
        initial_pull_request: number,
      };
      const { data: saved } = await github.rest.issues.createComment({ ...repo, issue_number: number,
        body: `<!-- ${acceptanceKey(person.id)} -->\nAgreement recorded for @${person.login}.\n\n<details><summary>Electronic acceptance record</summary>\n\n\`\`\`json\n${JSON.stringify(evidence, null, 2)}\n\`\`\`\n</details>`,
      });
      accepted.set(person.id, { actor: person, receipt: saved.html_url });
    }
    const missing = [...people.values()].filter(person => !accepted.has(person.id));
    if (missing.length) {
      const marker = `<!-- ${NOTICE_MARKER} ${acceptanceHash} -->`;
      if (!comments.some(item => botComment(item) && item.body?.startsWith(marker))) {
        await github.rest.issues.createComment({ ...repo, issue_number: number,
          body: `${marker}\n${missing.map(person => `@${person.login}`).join(' ')}: your first code contribution needs one electronic acceptance. NUTRIFIT LLC receives transferable economic copyright in covered contributions and may commercialize them. You retain authorship and receive the specified noncommercial grant back. A discretionary reward is possible, not guaranteed.\n\nRead [the agreement](${agreementUrl}) and [the grant-back license](${licenseUrl}). If you accept and own the rights or are authorized to sign, post this exact line as your electronic signature:\n\n\`\`\`text\n${command}\n\`\`\`\n\nNo printing or scans. Covered later submissions use this acceptance automatically while these terms remain unchanged. To exclude this contribution, use \`Rights-Assignment: excluded\`; excluded changes need separate clearance before merge. Do not accept on behalf of a rights owner without authority.`,
        });
      }
      await status('failure', 'Initial electronic acceptance is required for an identified author');
      return;
    }
    const coverage = {
      repository: `${repo.owner}/${repo.repo}`, pull_request: number,
      pull_request_url: pr.html_url, head_sha: sha,
      commits: commits.map(item => ({ sha: item.sha, author_id: item.author.id,
        url: item.html_url, committed_at: item.commit.committer?.date ?? null })),
      scope: 'Files and patches contained in the identified immutable commits',
      agreement_sha256: agreementHash, agreement_url: agreementUrl,
      license_sha256: licenseHash, license_url: licenseUrl, acceptance_sha256: acceptanceHash,
      acceptance_records: [...accepted.values()].map(item => ({ actor_id: item.actor.id, receipt: item.receipt })),
      recorded_at: new Date().toISOString(),
    };
    const coverageKey = `<!-- ${COVERAGE_MARKER} ${sha} ${acceptanceHash} -->`;
    if (!comments.some(item => botComment(item) && item.body?.startsWith(coverageKey))) {
      await github.rest.issues.createComment({ ...repo, issue_number: number,
        body: `${coverageKey}\n<details><summary>Contribution coverage recorded</summary>\n\n\`\`\`json\n${JSON.stringify(coverage, null, 2)}\n\`\`\`\n</details>`,
      });
    }
    const { data: current } = await github.rest.pulls.get({ ...repo, pull_number: number });
    if (current.head.sha !== sha) throw new Error('Состав PR изменился во время фиксации.');
    await status('success', 'Acceptance and current contribution identifiers recorded');
  } catch (error) {
    await status('failure', 'Rights records incomplete; maintainer review required');
    core.setFailed(error instanceof Error ? error.message : String(error));
  }
};
