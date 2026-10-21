import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { exec, spawn } from 'node:child_process';
import util from 'node:util';
import type { CravingMetricsData, CravingRoundSummary, CravingStepInfo } from '@/types';

const execPromise = util.promisify(exec);

export function getCravingEngineDir(): string {
  // apps/craving-engine/app -> apps/craving-engine
  const current = process.cwd();
  if (current.endsWith('app')) {
    return path.resolve(current, '..');
  }
  if (current.endsWith('craving-engine')) {
    return current;
  }
  return path.resolve(current, 'apps', 'craving-engine');
}

export function getCampaignDir(campaign = 'gemini'): string {
  const engineDir = getCravingEngineDir();
  return path.join(engineDir, '.research', 'runs', campaign);
}

export function isProcessAlive(pid: number | null | undefined): boolean {
  if (!pid || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function getCravingState(campaign = 'gemini') {
  const campaignDir = getCampaignDir(campaign);
  const autoDir = path.join(campaignDir, 'autonomous');
  const stateFile = path.join(autoDir, 'state.json');
  const goalFile = path.join(autoDir, 'goal.json');
  const stopFile = path.join(campaignDir, 'STOP');

  let state: Record<string, unknown> | null = null;
  let goal: Record<string, unknown> | null = null;
  let hasStop = false;

  if (fs.existsSync(stateFile)) {
    try {
      state = JSON.parse(fs.readFileSync(stateFile, 'utf-8'));
    } catch {
      state = null;
    }
  }

  if (fs.existsSync(goalFile)) {
    try {
      goal = JSON.parse(fs.readFileSync(goalFile, 'utf-8'));
    } catch {
      goal = null;
    }
  }

  if (fs.existsSync(stopFile)) {
    hasStop = true;
  }

  const pid = (state?.pid as number) ?? null;
  const alive = isProcessAlive(pid);
  const rawStatus = (state?.status as string) ?? 'ready';
  const effectiveStatus = hasStop ? 'stop_requested' : alive ? rawStatus : (rawStatus === 'running' ? 'stopped' : rawStatus);
  const bestMetrics = getCampaignBestMetrics(campaign);

  // Read latest 20 events
  const eventsDir = path.join(autoDir, 'events');
  const recentEvents: unknown[] = [];
  if (fs.existsSync(eventsDir)) {
    try {
      const eventFiles = fs.readdirSync(eventsDir)
        .filter(f => f.endsWith('.json'))
        .sort()
        .slice(-20);
      for (const ef of eventFiles) {
        try {
          const content = JSON.parse(fs.readFileSync(path.join(eventsDir, ef), 'utf-8'));
          recentEvents.push(content);
        } catch {
          // ignore corrupted files
        }
      }
    } catch {
      // ignore readdir errors
    }
  }

  return {
    campaign,
    active: alive,
    status: effectiveStatus,
    phase: state?.phase ?? 'qualification',
    epoch: state?.epoch ?? 1,
    round: state?.round ?? 1,
    taskIndex: state?.task_index ?? 0,
    failures: state?.failures ?? 0,
    lastFailure: state?.last_failure ?? null,
    failedRound: state?.failed_round ?? null,
    cumulative: state?.cumulative ?? {},
    nextRetryAt: state?.next_retry_at ?? 0,
    updatedAt: state?.updated_at ?? Date.now() / 1000,
    pid,
    hasStop,
    goal,
    bestMetrics,
    recentEvents: recentEvents.reverse(),
  };
}

export function getEpochsAndRounds(campaign = 'gemini') {
  const campaignDir = getCampaignDir(campaign);
  const epochsDir = path.join(campaignDir, 'autonomous', 'epochs');
  const state = getCravingState(campaign);

  if (!fs.existsSync(epochsDir)) {
    return { epochs: [] };
  }

  const epochDirs = fs.readdirSync(epochsDir)
    .filter(name => fs.statSync(path.join(epochsDir, name)).isDirectory())
    .sort()
    .reverse();

  const epochsResult = [];

  for (const epName of epochDirs) {
    const epPath = path.join(epochsDir, epName);
    const preregPath = path.join(epPath, 'preregistration.json');
    let prereg = null;
    if (fs.existsSync(preregPath)) {
      try {
        prereg = JSON.parse(fs.readFileSync(preregPath, 'utf-8'));
      } catch {
        prereg = null;
      }
    }

    const tasksDir = path.join(epPath, 'tasks');
    const tasksList: { taskId: string; rounds: CravingRoundSummary[] }[] = [];

    if (fs.existsSync(tasksDir)) {
      const taskNames = fs.readdirSync(tasksDir).filter(t => fs.statSync(path.join(tasksDir, t)).isDirectory());

      for (const tName of taskNames) {
        const tPath = path.join(tasksDir, tName);
        const roundNames = fs.readdirSync(tPath)
          .filter(r => r.startsWith('round-') && fs.statSync(path.join(tPath, r)).isDirectory())
          .sort()
          .reverse();

        const roundsList: CravingRoundSummary[] = [];

        for (const rName of roundNames) {
          const rPath = path.join(tPath, rName);
          const roundNum = parseInt(rName.replace('round-', ''), 10) || 1;
          const epochNum = parseInt(epName, 10) || 1;

          // Look for attempt-report
          const rFiles = fs.readdirSync(rPath);
          const reportFile = rFiles.find(f => f.startsWith('attempt-report-') && f.endsWith('.json'));
          let reportData: Record<string, unknown> | null = null;

          if (reportFile) {
            try {
              reportData = JSON.parse(fs.readFileSync(path.join(rPath, reportFile), 'utf-8'));
            } catch {
              reportData = null;
            }
          }

          // Check if this round is the active one
          const isCurrentlyActive = Boolean(state?.epoch === epochNum && state?.round === roundNum && state?.active);
          let roundStatus: CravingRoundSummary['status'] = isCurrentlyActive ? 'running' : 'waiting';
          let activeSeconds = 0;
          let calls = 0;
          let errObj: CravingRoundSummary['error'] | undefined;

          if (reportData) {
            const rawStatus = String(reportData.status || '').toLowerCase();
            const err = reportData.error as Record<string, unknown> | undefined;

            if (rawStatus === 'passed' || rawStatus === 'success') {
              roundStatus = 'passed';
            } else if (err || rawStatus === 'failed' || rawStatus === 'base-failed' || rawStatus === 'inconclusive') {
              roundStatus = 'failed';
            } else if (isCurrentlyActive) {
              roundStatus = 'running';
            } else {
              roundStatus = 'waiting';
            }

            const usage = (reportData.usage as Record<string, unknown>) || {};
            activeSeconds = Number(usage.active_seconds) || 0;
            calls = Number(usage.calls) || 0;

            if (err) {
              errObj = {
                message: String(err.message || 'Ошибка выполнения'),
                type: String(err.type || 'Error'),
                traceback: err.traceback ? String(err.traceback) : undefined,
              };
            }
          } else if (isCurrentlyActive) {
            roundStatus = 'running';
          }

          // Check cli-workspaces
          const wsDir = path.join(rPath, 'cli-workspaces');
          let hasStreamEvents = false;
          let hasReport = false;
          let hasHypothesis = false;
          let hasExperiment = false;
          let workspaceDirName: string | undefined;

          if (fs.existsSync(wsDir)) {
            const wsEntries = fs.readdirSync(wsDir).filter(w => {
              const full = path.join(wsDir, w);
              return fs.statSync(full).isDirectory();
            });

            if (wsEntries.length > 0) {
              workspaceDirName = wsEntries[0];
              const singleWs = path.join(wsDir, workspaceDirName);
              hasStreamEvents = fs.existsSync(path.join(singleWs, 'stream-events.jsonl'));
              hasReport = fs.existsSync(path.join(singleWs, 'report.md'));
              hasHypothesis = fs.existsSync(path.join(singleWs, 'hypothesis.json'));
              hasExperiment = fs.existsSync(path.join(singleWs, 'experiment', 'experiment.py'));
            }
          }

          roundsList.push({
            id: `${epName}/${tName}/${rName}`,
            epochStr: epName,
            epochNum,
            task: tName,
            roundStr: rName,
            roundNum,
            status: roundStatus,
            activeSeconds,
            calls,
            error: errObj,
            hasStreamEvents,
            hasReport,
            hasHypothesis,
            hasExperiment,
            workspaceDirName,
          });
        }

        tasksList.push({
          taskId: tName,
          rounds: roundsList,
        });
      }
    }

    epochsResult.push({
      epochStr: epName,
      epochNum: parseInt(epName, 10) || 1,
      preregistration: prereg,
      tasks: tasksList,
    });
  }

  return { epochs: epochsResult };
}

const CONFIG_NAMES_RU: Record<string, { name: string; category: string }> = {
  cfg_base_population: { name: 'Популяционный базовый уровень (Baseline)', category: 'До запуска' },
  cfg_world_heterogeneity: { name: 'Гетерогенный мир (разные профили людей)', category: 'После обучения' },
  cfg_world_interactions: { name: 'Мир сложных взаимодействий привычек', category: 'После обучения' },
  cfg_world_additive: { name: 'Аддитивный мир (линейные эффекты)', category: 'После обучения' },
  cfg_world_drift_slow: { name: 'Медленный дрейф привычек', category: 'Стресс-тест' },
  cfg_world_drift_abrupt: { name: 'Резкий слом режима (отпуск/командировка)', category: 'Стресс-тест' },
  cfg_world_disruption: { name: 'Внешние нарушения (болезнь/стресс)', category: 'Стресс-тест' },
  cfg_world_selective_feedback: { name: 'Выборочная обратная связь', category: 'Стресс-тест' },
  cfg_world_null_signal: { name: 'Нулевой мир (шум без сигнала, с Gate)', category: 'Контроль' },
  cfg_null_nogate: { name: 'Нулевой мир без гейта', category: 'Контроль' },
  cfg_world_naive_missingness: { name: 'Мир наивных пропусков данных', category: 'Стресс-тест' },
  cfg_scalar_lr005: { name: 'Скалярная память (скорость lr=0.05)', category: 'Гиперпараметры' },
  cfg_scalar_lr010: { name: 'Скалярная память (скорость lr=0.10)', category: 'Гиперпараметры' },
  cfg_scalar_lr020: { name: 'Скалярная память (скорость lr=0.20)', category: 'Гиперпараметры' },
  cfg_scalar_lr040: { name: 'Скалярная память (скорость lr=0.40)', category: 'Гиперпараметры' },
  cfg_scalar_nogate: { name: 'Скалярная память без гейта (No Gate)', category: 'Контроль' },
  cfg_scalar_halflife_fast7: { name: 'Период полураспада 7 дней', category: 'Физиология' },
  cfg_scalar_halflife_fast28: { name: 'Период полураспада 28 дней', category: 'Физиология' },
};

export function parseMetricsJson(
  raw: Record<string, unknown>,
  sourceLabel = 'Измерения OCI; научный допуск проверяется отдельно',
  sourceRoundId?: string,
  isBest = false
): CravingMetricsData {
  const configs = (raw.configurations || {}) as Record<string, Record<string, unknown>>;
  const summary = (raw.summary || {}) as Record<string, unknown>;
  const measured = (value: unknown, digits = 5) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  const percent = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
  const evidence = (raw.stage_evidence || {}) as Record<string, unknown>;
  const worlds = (Array.isArray(evidence.worlds) ? evidence.worlds : []) as Record<string, unknown>[];
  const splits = (evidence.splits || {}) as Record<string, unknown>;
  const analysis = (raw.validated_analysis || {}) as Record<string, unknown>;
  const aggregates = (Array.isArray(analysis.recomputed_metrics) ? analysis.recomputed_metrics : []) as Record<string, unknown>[];
  const evidenceRows: CravingMetricsData['simulationWorlds'] = aggregates.length ? aggregates.map((row, index) => ({
    id: `measured-${index}`, name: `${row.world} · ${row.model}`, category: String(row.feature_set),
    brier: measured(row.macro_brier), deltaBrier: measured(row.delta_macro_brier), auc: measured(row.macro_auc, 4),
    gateOpen: '—', degraded: '—', memory: '—', statusColor: 'default',
    status: `n=${row.n_users}; ДИ95% ${Array.isArray(row.ci95_user_bootstrap) ? row.ci95_user_bootstrap.map(v => measured(v)).join(' … ') : '—'}; p Holm=${measured(row.p_holm_within_packet)}`,
  })) : worlds.map((world) => ({
    id: String(world.family), name: String(world.family), category: `H2 · ${world.n_users} пользователей · ${world.n_decisions} решений`,
    brier: '—', deltaBrier: '—', auc: '—', gateOpen: '—', degraded: '—', memory: '—', statusColor: 'default',
    status: `Обратная связь ${percent(world.observed_feedback_fraction)}; синтетические исходы ${percent(world.latent_event_fraction)}`,
  }));
  return {
    sourceRoundId, sourceLabel, isBest,
    nUsers: typeof summary.n_users === 'number' ? summary.n_users : Array.isArray(splits.evaluation_users) ? splits.evaluation_users.length : undefined,
    totalDecisions: typeof summary.total_decisions_per_config === 'number' ? summary.total_decisions_per_config : worlds.length ? worlds.reduce((sum, world) => sum + (typeof world.n_decisions === 'number' ? world.n_decisions : 0), 0) : undefined,
    kpis: [],
    simulationWorlds: evidenceRows.length ? evidenceRows : Object.entries(configs).map(([id, cfg]) => ({
      id, name: CONFIG_NAMES_RU[id]?.name || id, category: 'Синтетический опыт',
      brier: measured(cfg.macro_brier), deltaBrier: measured(cfg.delta_brier_mean),
      auc: measured(cfg.macro_auc, 4), gateOpen: percent(cfg.gate_open_fraction),
      degraded: percent(cfg.degraded_users_fraction),
      memory: typeof cfg.max_state_size_bytes === 'number' ? `${cfg.max_state_size_bytes} Б` : '—',
      status: 'Измерение; требуется оценка', statusColor: 'default',
    })),
    rawJson: raw,
  };
}

export function findMetricsInDir(dirPath: string): Record<string, unknown> | null {
  // Только выбранный завершённый OCI probe, связанный с неизменяемым receipt.
  // Текст модели и её рабочая папка не являются измерениями.
  try {
    const receipt = JSON.parse(fs.readFileSync(path.join(dirPath, 'round-complete.json'), 'utf8'));
    if (receipt.execution_complete !== true) return null;
    const body = fs.readFileSync(path.join(dirPath, 'probe.json'), 'utf8');
    if (createHash('sha256').update(body).digest('hex') !== receipt.files?.['probe.json']) return null;
    const probe = JSON.parse(body);
    if (probe.execution_complete !== true) return null;
    const metrics = JSON.parse(probe.files['artifacts/metrics.json']);
    delete metrics.validated_analysis;
    if (receipt.files?.['scientific-analysis.json']) {
      const analysis = fs.readFileSync(path.join(dirPath, 'scientific-analysis.json'), 'utf8');
      if (createHash('sha256').update(analysis).digest('hex') !== receipt.files['scientific-analysis.json']) return null;
      metrics.validated_analysis = JSON.parse(analysis);
    }
    return metrics;
  } catch { return null; }
}

export function getCampaignBestMetrics(campaign = 'gemini'): CravingMetricsData | null {
  // Имя сохранено для совместимости API. Возвращается последний расчёт,
  // а не максимум качества из несопоставимых миров и адаптивных эпох.
  const epochsDir = path.join(getCampaignDir(campaign), 'autonomous', 'epochs');
  if (!fs.existsSync(epochsDir)) return null;
  for (const ep of fs.readdirSync(epochsDir, { withFileTypes: true }).filter(entry => entry.isDirectory()).map(entry => entry.name).sort().reverse()) {
    const tasksDir = path.join(epochsDir, ep, 'tasks');
    if (!fs.existsSync(tasksDir)) continue;
    for (const task of fs.readdirSync(tasksDir, { withFileTypes: true }).filter(entry => entry.isDirectory()).map(entry => entry.name).sort().reverse()) {
      const taskDir = path.join(tasksDir, task);
      for (const round of fs.readdirSync(taskDir, { withFileTypes: true }).filter(entry => entry.isDirectory() && entry.name.startsWith('round-')).map(entry => entry.name).sort().reverse()) {
        const raw = findMetricsInDir(path.join(taskDir, round));
        if (raw) return parseMetricsJson(raw, `Эпоха ${Number(ep)} · ${task} · раунд ${Number(round.replace('round-', ''))}; исследовательские измерения`, `${ep}/${task}/${round}`);
      }
    }
  }
  return null;
}

export function getRoundDetails(epoch: string, task: string, round: string, campaign = 'gemini') {
  const campaignDir = getCampaignDir(campaign);
  const roundPath = path.join(campaignDir, 'autonomous', 'epochs', epoch, 'tasks', task, round);

  if (!fs.existsSync(roundPath)) {
    return null;
  }

  const rFiles = fs.readdirSync(roundPath);
  const reportFile = rFiles.find(f => f.startsWith('attempt-report-') && f.endsWith('.json'));
  let attemptReport: Record<string, unknown> | null = null;

  if (reportFile) {
    try {
      attemptReport = JSON.parse(fs.readFileSync(path.join(roundPath, reportFile), 'utf-8'));
    } catch {
      attemptReport = null;
    }
  }

  // Find workspace
  const wsDir = path.join(roundPath, 'cli-workspaces');
  let steps: CravingStepInfo[] = [];
  let modelName = 'gemini-3.8-flash-high';
  let hypothesisText = '';
  let reportMarkdown = '';
  let experimentCode = '';

  if (fs.existsSync(wsDir)) {
    const wsEntries = fs.readdirSync(wsDir)
      .filter(w => fs.statSync(path.join(wsDir, w)).isDirectory())
      .sort((a, b) => {
        const mtimeA = fs.statSync(path.join(wsDir, a)).mtimeMs;
        const mtimeB = fs.statSync(path.join(wsDir, b)).mtimeMs;
        return mtimeB - mtimeA;
      });

    if (wsEntries.length > 0) {
      const activeWs = path.join(wsDir, wsEntries[0]);

      // Read stream-events.jsonl from the most active workspace
      const streamFile = path.join(activeWs, 'stream-events.jsonl');
      if (fs.existsSync(streamFile)) {
        try {
          const lines = fs.readFileSync(streamFile, 'utf-8').split('\n').filter(Boolean);
          const stepsMap = new Map<number, CravingStepInfo>();

          for (const line of lines) {
            try {
              const row = JSON.parse(line);
              if (row.event === 'init') {
                modelName = row.model || modelName;
              } else if (row.event === 'step_update' && row.step_update) {
                const su = row.step_update;
                const idx = typeof su.step_index === 'number' ? su.step_index : stepsMap.size;
                const existing: CravingStepInfo = stepsMap.get(idx) || {
                  index: idx,
                  type: su.step_type || 'status',
                  state: su.state || 'DONE',
                };

                if (su.step_type) {
                  existing.type = su.step_type;
                }
                if (su.state) {
                  existing.state = su.state;
                }
                if (typeof su.duration_seconds === 'number') {
                  existing.durationSeconds = su.duration_seconds;
                }
                if (su.usage) {
                  existing.usage = {
                    inputTokens: su.usage.input_tokens,
                    outputTokens: su.usage.output_tokens,
                    thinkingTokens: su.usage.thinking_tokens,
                    totalTokens: su.usage.total_tokens,
                  };
                }

                if (su.step_type === 'user_input') {
                  existing.content = 'Получены входные директивы этапа и контекст от автономного контроллера';
                } else if (su.step_type === 'agent_response') {
                  existing.content = 'Размышления и формирование решения (Thinking & Plan generation)';
                } else if (su.step_type === 'system_message') {
                  existing.content = 'Системное уведомление и директивы безопасной изолированной среды';
                } else if (su.step_type === 'tool') {
                  const toolInfo = su.tool_info || {};
                  if (toolInfo.name) existing.toolName = toolInfo.name;
                  if (toolInfo.parameters) existing.toolParameters = toolInfo.parameters;
                  if (toolInfo.output) existing.toolOutput = toolInfo.output;
                }

                stepsMap.set(idx, existing);
              }
            } catch {
              // skip
            }
          }

          steps = Array.from(stepsMap.values()).sort((a, b) => a.index - b.index);
        } catch {
          // ignore
        }
      }

      // Check all workspaces in this round for artifacts (prioritizing most recent)
      for (const ws of wsEntries) {
        const curWs = path.join(wsDir, ws);

        if (!hypothesisText) {
          const hypFile = path.join(curWs, 'hypothesis.json');
          if (fs.existsSync(hypFile)) {
            try {
              hypothesisText = fs.readFileSync(hypFile, 'utf-8');
            } catch {}
          }
        }

        if (!reportMarkdown) {
          const repFile = path.join(curWs, 'report.md');
          if (fs.existsSync(repFile)) {
            try {
              reportMarkdown = fs.readFileSync(repFile, 'utf-8');
            } catch {}
          }
        }

        if (!experimentCode) {
          const expFile = path.join(curWs, 'experiment', 'experiment.py');
          if (fs.existsSync(expFile)) {
            try {
              experimentCode = fs.readFileSync(expFile, 'utf-8');
            } catch {}
          }
        }
      }
    }
  }

  // Fallback for experimentCode from chosen or rollouts
  if (!experimentCode) {
    const chosenExp = path.join(roundPath, 'chosen', 'experiment', 'experiment.py');
    if (fs.existsSync(chosenExp)) {
      try {
        experimentCode = fs.readFileSync(chosenExp, 'utf-8');
      } catch {}
    } else {
      const rolloutsDir = path.join(roundPath, 'rollouts');
      if (fs.existsSync(rolloutsDir)) {
        try {
          const rDirs = fs.readdirSync(rolloutsDir);
          for (const rd of rDirs) {
            const expF = path.join(rolloutsDir, rd, 'experiment', 'experiment.py');
            if (fs.existsSync(expF)) {
              try {
                experimentCode = fs.readFileSync(expF, 'utf-8');
                break;
              } catch {}
            }
          }
        } catch {}
      }
    }
  }

  // Fallback for report if not found in active workspace
  if (!reportMarkdown || reportMarkdown.trim().length === 0) {
    const chosenReport = path.join(roundPath, 'chosen', 'report.md');
    const researchReport = path.join(roundPath, 'research-report.md');
    if (fs.existsSync(chosenReport)) {
      try {
        reportMarkdown = fs.readFileSync(chosenReport, 'utf-8');
      } catch {
        // ignore
      }
    } else if (fs.existsSync(researchReport)) {
      try {
        reportMarkdown = fs.readFileSync(researchReport, 'utf-8');
      } catch {
        // ignore
      }
    }
  }

  // Deduplicate and group steps
  const compactSteps: CravingStepInfo[] = [];
  const seenStepIndices = new Map<number, number>();

  for (const st of steps) {
    if (st.type === 'init') {
      compactSteps.push(st);
      continue;
    }
    const existingIndex = seenStepIndices.get(st.index);
    if (existingIndex !== undefined) {
      compactSteps[existingIndex] = {
        ...compactSteps[existingIndex],
        ...st,
        toolOutput: st.toolOutput || compactSteps[existingIndex].toolOutput,
        usage: st.usage || compactSteps[existingIndex].usage,
      };
    } else {
      seenStepIndices.set(st.index, compactSteps.length);
      compactSteps.push(st);
    }
  }

  const roundMetricsRaw = findMetricsInDir(roundPath);
  const roundMetrics = roundMetricsRaw
    ? parseMetricsJson(
        roundMetricsRaw,
        `Эпоха #${parseInt(epoch, 10) || 1} • Раунд #${parseInt(round.replace('round-', ''), 10) || 1}`,
        `${epoch}/${task}/${round}`
      )
    : null;
  const bestMetrics = getCampaignBestMetrics(campaign);

  return {
    epoch,
    task,
    round,
    modelName,
    attemptReport,
    steps: compactSteps,
    hypothesis: hypothesisText,
    report: reportMarkdown,
    experimentCode,
    metrics: roundMetrics,
    bestMetrics,
  };
}

export function getStdoutLogs(campaign = 'gemini', maxLines = 250): string[] {
  const campaignDir = getCampaignDir(campaign);
  const logFile = path.join(campaignDir, 'autonomous', 'stdout.log');

  if (fs.existsSync(logFile)) {
    try {
      const content = fs.readFileSync(logFile, 'utf-8').trim();
      if (content.length > 0) {
        const lines = content.split('\n');
        return lines.slice(-maxLines);
      }
    } catch {
      // fallback
    }
  }

  // Fallback: build live system logs from events
  const eventsDir = path.join(campaignDir, 'autonomous', 'events');
  if (fs.existsSync(eventsDir)) {
    try {
      const eventFiles = fs.readdirSync(eventsDir)
        .filter(f => f.endsWith('.json'))
        .sort((a, b) => {
          const numA = parseInt(a.replace('.json', ''), 10) || 0;
          const numB = parseInt(b.replace('.json', ''), 10) || 0;
          return numA - numB;
        });

      const logLines: string[] = [];
      for (const ef of eventFiles) {
        try {
          const ev = JSON.parse(fs.readFileSync(path.join(eventsDir, ef), 'utf-8'));
          const timeStr = new Date(ev.at * 1000).toLocaleTimeString('ru-RU');
          const epochStr = ev.epoch ? ` [Эпоха #${ev.epoch}]` : '';
          const roundStr = ev.round ? ` [Раунд #${ev.round}]` : '';
          const reasonStr = ev.reason ? ` — ${ev.reason}` : '';
          const failureStr = ev.failure ? ` (Причина доработки: ${ev.failure.message})` : '';
          logLines.push(`[${timeStr}] [${ev.kind}]${epochStr}${roundStr}${reasonStr}${failureStr}`);
        } catch {
          // ignore
        }
      }

      if (logLines.length > 0) {
        return logLines.slice(-maxLines);
      }
    } catch {
      // ignore
    }
  }

  return ['[Контроллер] Исследовательский процесс инициализирован, ожидание потока событий...'];
}

export async function executeCravingAction(action: 'start' | 'stop' | 'status', campaign = 'gemini') {
  const engineDir = getCravingEngineDir();
  const scriptPath = path.join(engineDir, 'scripts', 'craving-research');

  if (action === 'start') {
    const state = getCravingState(campaign);
    if (state.active) {
      return { success: false, message: 'Исследование уже запущено и активно в фоне', pid: state.pid };
    }

    const child = spawn(
      scriptPath,
      ['start'],
      {
        cwd: engineDir,
        env: {
          ...process.env,
          CRAVING_CAMPAIGN: campaign,
        },
        detached: true,
        stdio: 'ignore',
      }
    );
    child.unref();

    return {
      success: true,
      message: `Исследовательская кампания ${campaign} запущена в фоне`,
      pid: child.pid,
    };
  }

  if (action === 'stop') {
    try {
      const { stdout } = await execPromise(`${scriptPath} stop`, {
        cwd: engineDir,
        env: {
          ...process.env,
          CRAVING_CAMPAIGN: campaign,
        },
      });
      return { success: true, message: 'Запрос на остановку отправлен (файл STOP записан)', stdout };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, message: `Ошибка отправки остановки: ${msg}` };
    }
  }

  if (action === 'status') {
    try {
      const { stdout } = await execPromise(`${scriptPath} status`, {
        cwd: engineDir,
        env: {
          ...process.env,
          CRAVING_CAMPAIGN: campaign,
        },
      });
      return { success: true, stdout };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return { success: false, message: `Ошибка получения статуса: ${msg}` };
    }
  }

  return { success: false, message: `Неизвестное действие: ${action}` };
}
