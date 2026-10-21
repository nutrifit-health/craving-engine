'use client';

import { useMemo, useRef, useState, useEffect } from 'react';
import { Button, Spinner, Tooltip } from '@heroui/react';
import { SvgIcon } from '@nutrifit/web-ui/SvgIcon';
import { extractReportSummary, parseHypothesisHuman } from '@/utils/craving-insights';
import { PythonCodeViewer } from './PythonCodeViewer';
import { MarkdownViewer } from './MarkdownViewer';
import { CravingMetricsComparison } from './CravingMetricsComparison';
import { CravingHierarchyGuide } from './CravingHierarchyGuide';
import { CravingStageTransitionGate } from './CravingStageTransitionGate';
import type { CravingDashboardState, CravingRoundDetails, CravingRoundSummary } from '@/types';

interface CravingTerminalProps {
  state?: CravingDashboardState | null;
  selectedRound: CravingRoundSummary | null;
  roundDetails: CravingRoundDetails | null;
  terminalLogs: string[];
  activeTab: 'steps' | 'stdout' | 'hypothesis' | 'experiment' | 'report' | 'diagnostics' | 'metrics' | 'guide';
  onTabChange: (tab: 'steps' | 'stdout' | 'hypothesis' | 'experiment' | 'report' | 'diagnostics' | 'metrics' | 'guide') => void;
  isLoading: boolean;
  onSelectActiveRound?: () => void;
  liveRoundSeconds?: number;
  selectedTaskFilter?: string;
}

export function CravingTerminal({
  state,
  selectedRound,
  roundDetails,
  terminalLogs,
  activeTab,
  onTabChange,
  isLoading,
  onSelectActiveRound,
  liveRoundSeconds,
  selectedTaskFilter,
}: CravingTerminalProps) {
  const [copied, setCopied] = useState(false);
  const [isTracebackOpen, setIsTracebackOpen] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [hypothesisMode, setHypothesisMode] = useState<'human' | 'json'>('human');
  const [readReportRoundIds, setReadReportRoundIds] = useState<Set<string>>(new Set());
  const stepsEndRef = useRef<HTMLDivElement>(null);
  const stdoutEndRef = useRef<HTMLDivElement>(null);

  // Mark report as read once the user visits the 'report' tab for this round
  useEffect(() => {
    if (activeTab === 'report' && roundDetails?.report && selectedRound?.id) {
      setReadReportRoundIds((prev) => {
        if (prev.has(selectedRound.id)) return prev;
        const next = new Set(prev);
        next.add(selectedRound.id);
        return next;
      });
    }
  }, [activeTab, roundDetails?.report, selectedRound?.id]);

  const hasUnreadReport = Boolean(roundDetails?.report && selectedRound?.id && !readReportRoundIds.has(selectedRound.id));

  // Auto-scroll logic
  useEffect(() => {
    if (!autoScroll) return;
    if (activeTab === 'steps') {
      stepsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    } else if (activeTab === 'stdout') {
      stdoutEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [roundDetails?.steps?.length, terminalLogs.length, activeTab, autoScroll]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const stepsList = roundDetails?.steps || [];
  const attemptReport = roundDetails?.attemptReport;
  const errorObj = selectedRound?.error || (attemptReport?.error as Record<string, string> | undefined);

  // Human report extraction
  const reportSummary = useMemo(() => {
    return extractReportSummary(
      roundDetails?.report,
      selectedRound?.status,
      selectedRound?.task || 'H2',
      String(selectedRound?.roundNum || '1'),
      selectedRound?.error
    );
  }, [roundDetails?.report, selectedRound]);

  // Human hypothesis extraction
  const hypothesisSummary = useMemo(() => {
    return parseHypothesisHuman(roundDetails?.hypothesis);
  }, [roundDetails?.hypothesis]);

  // Calculate total tokens & durations
  const totalTokensSummary = useMemo(() => {
    let input = 0;
    let output = 0;
    let thinking = 0;
    for (const s of stepsList) {
      if (s.usage) {
        input += s.usage.inputTokens || 0;
        output += s.usage.outputTokens || 0;
        thinking += s.usage.thinkingTokens || 0;
      }
    }
    return { input, output, thinking, total: input + output };
  }, [stepsList]);

  if (!selectedRound && activeTab !== 'metrics' && activeTab !== 'guide') {
    if (selectedTaskFilter && selectedTaskFilter !== 'ALL' && selectedTaskFilter !== 'H2') {
      return (
        <CravingStageTransitionGate
          targetStage={selectedTaskFilter}
          state={state}
          onSelectActiveRound={onSelectActiveRound}
        />
      );
    }

    return (
      <div className='flex flex-col items-center justify-center h-[calc(100vh-74px)] p-8 text-center bg-background space-y-4'>
        <div className='flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary mb-1'>
          <SvgIcon aria-hidden='true' iconName='solar:document-text-linear' width={34} />
        </div>
        <div className='space-y-1.5 max-w-md'>
          <h3 className='text-base font-bold text-foreground'>Выберите раунд для изучения</h3>
          <p className='text-xs text-default-500 leading-relaxed'>
            В списке слева выберите интересующий раунд или этап. Вы сможете изучить выводы отчета, гипотезу, сгенерированный код нейросети и логи терминала.
          </p>
        </div>

        {onSelectActiveRound && (
          <div className='pt-2'>
            <Button
              variant='secondary'
              size='sm'
              onPress={onSelectActiveRound}
              className='font-semibold text-xs'>
                <SvgIcon aria-hidden='true' iconName='solar:play-circle-linear' width={16} />
              Открыть активный раунд (Этап H2)
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className='flex flex-col h-[calc(100vh-74px)] bg-background overflow-hidden'>
      {/* Top Bar with round metadata */}
      <div className='p-3.5 border-b border-default-200 bg-default-50/50 space-y-2.5'>
        <div className='flex flex-wrap items-center justify-between gap-3'>
          <div className='flex items-center gap-2.5 min-w-0'>
            <div>
              <div className='flex items-center gap-2 flex-wrap'>
                <span className='font-bold text-foreground text-sm'>
                  {selectedRound
                    ? `Эпоха #${selectedRound.epochNum} • Этап ${selectedRound.task} • Раунд #${selectedRound?.roundNum}`
                    : activeTab === 'metrics'
                    ? 'Сравнение метрик (До vs После)'
                    : 'Иерархия циклов и устройство'}
                </span>

                {selectedRound ? (
                  selectedRound?.status === 'running' ? (
                    <span className='inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-emerald-600 dark:text-emerald-400 text-[11px] font-semibold'>
                      <span className='relative flex h-1.5 w-1.5 shrink-0'>
                        <span className='animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75' />
                        <span className='relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500' />
                      </span>
                      <span>Выполняется</span>
                    </span>
                  ) : selectedRound?.status === 'failed' || selectedRound.error ? (
                    <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-foreground text-[11px] font-medium'>
                      Доработка
                    </span>
                  ) : selectedRound?.status === 'passed' ? (
                    <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-foreground text-[11px] font-medium'>
                      Подтвержден
                    </span>
                  ) : (
                    <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-default-500 text-[11px] font-normal'>
                      В процессе проверки
                    </span>
                  )
                ) : (
                  <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-default-100 text-foreground text-[11px] font-semibold'>
                    Справка
                  </span>
                )}
              </div>

              {/* Execution time & tokens with clear tooltips */}
              {selectedRound ? (
                <div className='text-xs text-default-500 flex items-center gap-2.5 mt-0.5 font-mono'>
                  <span className='flex items-center gap-1.5'>
                    {selectedRound?.status === 'running' && (
                      <span className='inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse' />
                    )}
                    <span>
                      Время: {Math.round(selectedRound?.status === 'running' ? (liveRoundSeconds || selectedRound.activeSeconds) : selectedRound.activeSeconds)}с
                    </span>
                    {selectedRound?.status === 'running' && (
                      <span className='text-[11px] text-emerald-800 dark:text-emerald-300 font-semibold font-sans'>(выполняется...)</span>
                    )}
                  </span>
                  {totalTokensSummary.total > 0 && (
                    <>
                      <span>•</span>
                      <Tooltip>
                        <span className='cursor-help underline decoration-dotted decoration-default-400 underline-offset-2'>
                          Токены AI: {totalTokensSummary.total.toLocaleString()}
                        </span>
                      <Tooltip.Content placement='bottom'>Суммарный объем входящих и сгенерированных токенов модели, затраченных на этот раунд</Tooltip.Content></Tooltip>
                      {totalTokensSummary.thinking > 0 && (
                        <Tooltip>
                          <span className='text-primary-600 dark:text-primary-400 cursor-help underline decoration-dotted decoration-primary-400 underline-offset-2'>
                            (мышление: {totalTokensSummary.thinking.toLocaleString()})
                          </span>
                        <Tooltip.Content placement='bottom'>Внутренние токены глубокого мышления (CoT reasoning), когда нейросеть проектирует биологическую архитектуру и ищет решение</Tooltip.Content></Tooltip>
                      )}
                    </>
                  )}
                  <span>•</span>
                  <Tooltip>
                    <span className='text-default-400 cursor-help underline decoration-dotted decoration-default-400 underline-offset-2'>
                      Действий AI: {stepsList.length}
                    </span>
                  <Tooltip.Content placement='bottom'>Количество атомарных действий автономного исследователя: поиск файлов, редактирование кода, запуск тестов и оценка метрик</Tooltip.Content></Tooltip>
                </div>
              ) : (
                <div className='text-xs text-default-500 mt-0.5'>
                  Официальные данные научных бенчмарков NutriFit Craving Engine
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions (Copy, Autoscroll icon toggle) */}
          <div className='flex items-center gap-2'>
            <Button
              size='sm'
              variant='secondary'
              onPress={() => {
                const fullText = stepsList
                  .map((s) => `[Шаг ${s.index}] ${s.type}: ${s.toolName || s.content || ''}\n${s.toolOutput || ''}`)
                  .join('\n\n');
                copyToClipboard(fullText);
              }}
              className='h-8 text-xs'
            >
              <SvgIcon aria-hidden='true' iconName={copied ? 'solar:check-circle-bold' : 'solar:copy-linear'} width={15} />
              <span>{copied ? 'Скопировано!' : 'Копировать'}</span>
            </Button>

            {/* Icon-only autoscroll button with tooltip */}
            <Tooltip>
              <Button
                isIconOnly
                size='sm'
                variant={autoScroll ? 'primary' : 'secondary'}
                onPress={() => setAutoScroll(!autoScroll)}
                className='h-8 w-8 shadow-none'
                aria-label='Автопрокрутка терминала'
              >
                <SvgIcon
                  aria-hidden='true'
                  iconName={autoScroll ? 'mdi:pin-outline' : 'solar:bookmark-linear'}
                  width={15}
                />
              </Button>
            <Tooltip.Content placement='bottom'>{autoScroll ? 'Автопрокрутка включена (нажмите для паузы)' : 'Автопрокрутка выключена (нажмите для включения)'}</Tooltip.Content></Tooltip>
          </div>
        </div>

        {/* Historical inspection banner */}
        {selectedRound && selectedRound?.status !== 'running' && (
          <div className='flex items-center justify-between gap-3 px-3 py-2 rounded-xl bg-default-100/80 text-xs'>
            <div className='flex items-center gap-2 min-w-0'>
              <SvgIcon aria-hidden='true' iconName='solar:history-bold' width={15} className='text-default-500 shrink-0' />
              <span className='text-default-700 dark:text-default-300 truncate'>
                Архивный раунд #{selectedRound?.roundNum} (Эпоха #{selectedRound.epochNum}, Этап {selectedRound.task}) — вы просматриваете сохраненные данные из истории
              </span>
            </div>
            {onSelectActiveRound && (
              <button
                onClick={onSelectActiveRound}
                className='flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-900 dark:text-emerald-200 text-xs font-bold transition-all shrink-0'
                title='Вернуться к выполняемому в данный момент раунду'
              >
                <span className='relative flex h-2 w-2'>
                  <span className='animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75' />
                  <span className='relative inline-flex rounded-full h-2 w-2 bg-emerald-600 dark:bg-emerald-400' />
                </span>
                <span>К текущему (Live)</span>
              </button>
            )}
          </div>
        )}

        {/* Error notification card if error exists */}
        {errorObj && (
          <div className='rounded-2xl bg-default-100 p-3.5 space-y-2'>
            <div className='flex items-center justify-between text-foreground'>
              <div className='flex items-center gap-2 font-medium text-xs'>
                <span className='text-[10px] font-semibold bg-default-200 text-foreground px-1.5 py-0.5 rounded'>
                  Доработка
                </span>
                <span>{errorObj.message}</span>
              </div>
              {errorObj.traceback && (
                <button
                  onClick={() => setIsTracebackOpen(!isTracebackOpen)}
                  className='text-[11px] underline hover:text-foreground text-default-500 transition-colors'
                >
                  {isTracebackOpen ? 'Свернуть детали' : 'Технические детали'}
                </button>
              )}
            </div>

            {errorObj.traceback && isTracebackOpen && (
              <pre className='rounded-xl bg-zinc-950 p-3 text-[11px] font-mono text-zinc-300 overflow-x-auto max-h-40 overflow-y-auto leading-relaxed whitespace-pre-wrap'>
                {errorObj.traceback}
              </pre>
            )}
          </div>
        )}

        {/* Tabs Bar: Clean, smooth scroll, no ugly scrollbar, plain names */}
        <div className='flex items-center border-b border-default-200 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden gap-1 flex-nowrap pt-1'>
          {/* TAB: REPORT (FIRST OR HIGHLIGHTED) */}
          <button
            onClick={() => onTabChange('report')}
            className={`px-3 py-2 text-xs font-bold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'report'
                ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:document-text-linear' width={15} />
            <span>Отчет</span>
            {/* Зеленая точка ТОЛЬКО когда сформирован новый отчет, готовый к чтению */}
            {hasUnreadReport && (
              <span
                className='h-2 w-2 rounded-full bg-emerald-500 animate-pulse inline-block shrink-0'
                title='Сформирован новый отчет для ознакомления'
              />
            )}
          </button>

          {/* TAB: METRICS (ДО VS ПОСЛЕ) */}
          <button
            onClick={() => onTabChange('metrics')}
            className={`px-3 py-2 text-xs font-bold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'metrics'
                ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:chart-2-bold' width={15} />
            <span>Сравнение метрик (До vs После)</span>
            <span className='text-[10px] font-bold px-1.5 py-0.2 rounded-full bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'>
              {roundDetails?.metrics?.kpis?.[0]?.delta || roundDetails?.bestMetrics?.kpis?.[0]?.delta || state?.bestMetrics?.kpis?.[0]?.delta || '—'}
            </span>
          </button>

          {/* TAB: HIERARCHY GUIDE (СПРАВКА) */}
          <button
            onClick={() => onTabChange('guide')}
            className={`px-3 py-2 text-xs font-bold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'guide'
                ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:layers-linear' width={15} />
            <span>Иерархия циклов (Справка)</span>
          </button>

          {/* TAB: HYPOTHESIS */}
          <button
            onClick={() => onTabChange('hypothesis')}
            className={`px-3 py-2 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'hypothesis'
                ? 'border-primary text-primary'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:lightbulb-bolt-bold' width={15} />
            <span>Гипотеза</span>
          </button>

          {/* TAB: AI STEPS */}
          <button
            onClick={() => onTabChange('steps')}
            className={`px-3 py-2 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'steps'
                ? 'border-primary text-primary'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:code-square-linear' width={15} />
            <span>Ход рассуждений AI ({stepsList.length})</span>
          </button>

          {/* TAB: STDOUT */}
          <button
            onClick={() => onTabChange('stdout')}
            className={`px-3 py-2 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'stdout'
                ? 'border-primary text-primary'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:code-linear' width={15} />
            <span>Логи процесса</span>
          </button>

          {/* TAB: EXPERIMENT CODE */}
          <button
            onClick={() => onTabChange('experiment')}
            className={`px-3 py-2 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'experiment'
                ? 'border-primary text-primary'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:code-linear' width={15} />
            <span>Код эксперимента</span>
          </button>

          {/* TAB: DIAGNOSTICS */}
          <button
            onClick={() => onTabChange('diagnostics')}
            className={`px-3 py-2 text-xs font-semibold border-b-2 transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === 'diagnostics'
                ? 'border-primary text-primary'
                : 'border-transparent text-default-600 hover:text-foreground'
            }`}
          >
            <SvgIcon aria-hidden='true' iconName='solar:cpu-bolt-linear' width={15} />
            <span>Диагностика</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className='flex-1 overflow-y-auto p-4 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden'>
        {isLoading && stepsList.length === 0 ? (
          <div className='flex items-center justify-center h-48'>
            <Spinner aria-label='Загрузка данных раунда...' size='md' />
          </div>
        ) : null}

        {/* TAB: METRICS COMPARISON (ДО VS ПОСЛЕ) */}
        {activeTab === 'metrics' && (
          <CravingMetricsComparison
            metrics={roundDetails?.metrics}
            bestMetrics={roundDetails?.bestMetrics || state?.bestMetrics}
            selectedRound={selectedRound}
            isActiveRound={Boolean(selectedRound?.status === 'running')}
          />
        )}

        {/* TAB: HIERARCHY GUIDE (СПРАВКА И ИЕРАРХИЯ) */}
        {activeTab === 'guide' && <CravingHierarchyGuide />}

        {/* TAB 1: REPORT (С ЧЕЛОВЕЧЕСКИМ ВЫВОДОМ В САМОМ ВЕРХУ ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ!) */}
        {activeTab === 'report' && (
          <div className='space-y-5 max-w-4xl'>
            {/* Карточка-резюме отчета в самом верху */}
            <div className='rounded-2xl bg-emerald-500/10 dark:bg-emerald-950/30 p-5 space-y-4'>
              {/* Верхняя плашка статуса и даты */}
              <div className='flex flex-wrap items-center justify-between gap-3 bg-emerald-500/10 dark:bg-emerald-950/40 px-3.5 py-2.5 rounded-xl'>
                <div className='flex items-center gap-2'>
                  <SvgIcon aria-hidden='true' iconName='solar:document-text-bold' width={16} className='text-emerald-600 dark:text-emerald-400' />
                  <span className='text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-300'>
                    {reportSummary.taskTitle}
                  </span>
                </div>
                <div className='flex items-center gap-3 text-xs text-default-500'>
                  <span>Дата: <strong className='text-foreground'>{reportSummary.date}</strong></span>
                  <span>•</span>
                  <span>Раунд: <strong className='text-foreground'>#{reportSummary.roundNum}</strong></span>
                  <span className='font-semibold px-2.5 py-0.5 rounded-md bg-default-100 text-foreground text-xs whitespace-nowrap'>
                    {reportSummary.statusLabel}
                  </span>
                </div>
              </div>

              {/* ГЛАВНЫЙ ВЫВОД И РЕЗУЛЬТАТ ОТЧЕТА ПРОСТЫМ ЧЕЛОВЕЧЕСКИМ ЯЗЫКОМ ДЛЯ ОБЫВАТЕЛЯ */}
              <div className='space-y-2'>
                <div className='flex items-center gap-2 text-sm font-bold text-foreground'>
                  <SvgIcon aria-hidden='true' iconName='solar:star-bold' width={18} className='text-emerald-500' />
                  <span>Главный вывод эксперимента:</span>
                </div>
                <p className='text-sm text-foreground/90 font-medium leading-relaxed bg-background/90 dark:bg-zinc-900/90 p-4 rounded-2xl'>
                  {reportSummary.keyTakeaway}
                </p>
              </div>

              {/* Практическая польза для продукта NutriFit */}
              <div className='space-y-1.5 pt-1'>
                <div className='text-xs font-semibold text-default-500 uppercase tracking-wider flex items-center gap-1.5'>
                  <SvgIcon aria-hidden='true' iconName='solar:shield-check-linear' width={14} className='text-primary' />
                  <span>Что это значит для NutriFit и пользователя:</span>
                </div>
                <p className='text-xs text-default-600 dark:text-default-400 leading-relaxed'>
                  {reportSummary.practicalImplication}
                </p>
              </div>
            </div>

            {/* Ниже — Технические подробности отчета */}
            <div className='space-y-3 pt-2'>
              <div className='flex items-center justify-between pb-1 px-1'>
                <h4 className='text-xs font-bold uppercase tracking-wider text-default-500 flex items-center gap-1.5'>
                  <SvgIcon aria-hidden='true' iconName='solar:document-text-linear' width={15} />
                  <span>Полный технический текст отчета</span>
                </h4>
                <span className='text-[11px] text-default-400 font-mono'>
                  Формат Markdown • Окружение: {reportSummary.environmentInfo}
                </span>
              </div>

              {roundDetails?.report ? (
                <div className='rounded-2xl bg-default-50/70 dark:bg-zinc-900/60 p-6'>
                  <MarkdownViewer content={roundDetails.report} />
                </div>
              ) : (
                <p className='text-default-400 italic text-sm py-4'>
                  Полный текст отчета еще формируется моделью в этом раунде.
                </p>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: HYPOTHESIS (ДВОЙНОЙ РЕЖИМ: ПОНЯТНЫЙ ЧЕЛОВЕЧЕСКИЙ РАЗБОР VS СЫРОЙ JSON) */}
        {activeTab === 'hypothesis' && (
          <div className='space-y-4 max-w-4xl'>
            {/* Mode Switcher */}
            <div className='flex items-center justify-between bg-default-100/80 p-2 rounded-2xl'>
              <div className='flex items-center gap-2'>
                <button
                  onClick={() => setHypothesisMode('human')}
                  className={`text-xs px-3 py-1.5 rounded-xl font-semibold transition-all ${
                    hypothesisMode === 'human'
                      ? 'bg-primary text-primary-foreground'
                      : 'text-default-600 hover:text-foreground'
                  }`}
                >
                  Понятное объяснение для продукта
                </button>
                <button
                  onClick={() => setHypothesisMode('json')}
                  className={`text-xs px-3 py-1.5 rounded-xl font-semibold transition-all ${
                    hypothesisMode === 'json'
                      ? 'bg-foreground text-background'
                      : 'text-default-600 hover:text-foreground'
                  }`}
                >
                  Технический JSON
                </button>
              </div>

              <span className='text-[11px] text-default-400 font-mono hidden sm:inline'>
                Проверка ложных тревог и лимитов памяти
              </span>
            </div>

            {hypothesisMode === 'human' ? (
              <div className='space-y-4'>
                {/* Hypothesis Human Cards */}
                <div className='rounded-2xl bg-primary/10 p-5 space-y-2.5'>
                  <div className='flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-primary'>
                    <SvgIcon aria-hidden='true' iconName='solar:target-bold' width={16} />
                    <span>Какую идею проверяем в этом раунде:</span>
                  </div>
                  <p className='text-sm font-medium text-foreground leading-relaxed'>
                    {hypothesisSummary.goal}
                  </p>
                </div>

                <div className='grid grid-cols-1 md:grid-cols-2 gap-3.5'>
                  {/* Why it matters */}
                  <div className='rounded-2xl bg-default-100/70 p-4 space-y-1.5'>
                    <div className='text-xs font-bold text-default-700 flex items-center gap-1.5'>
                      <SvgIcon aria-hidden='true' iconName='solar:shield-warning-linear' width={16} className='text-amber-500' />
                      <span>Защита от ложных тревог:</span>
                    </div>
                    <p className='text-xs text-default-600 leading-relaxed'>
                      {hypothesisSummary.safetyCheck}
                    </p>
                  </div>

                  {/* Memory constraint */}
                  <div className='rounded-2xl bg-default-100/70 p-4 space-y-1.5'>
                    <div className='text-xs font-bold text-default-700 flex items-center gap-1.5'>
                      <SvgIcon aria-hidden='true' iconName='solar:cpu-bolt-linear' width={16} className='text-emerald-500' />
                      <span>Память на телефоне (до 16 КБ):</span>
                    </div>
                    <p className='text-xs text-default-600 leading-relaxed'>
                      {hypothesisSummary.memoryConstraint}
                    </p>
                  </div>
                </div>

                {/* Practical Example (Sport experience 6 years vs 1 week) */}
                <div className='rounded-2xl bg-default-100/70 p-4 space-y-1.5'>
                  <div className='text-xs font-bold text-default-700 flex items-center gap-1.5'>
                    <SvgIcon aria-hidden='true' iconName='solar:user-id-linear' width={16} className='text-primary' />
                    <span>Пример работы на практике:</span>
                  </div>
                  <p className='text-xs text-default-600 leading-relaxed'>
                    Человек, который регулярно тренируется 6 лет, имеет сформированную привычку. Если он пропустил тренировку из-за поездки, вероятность полного срыва у него минимальна. Новичок же после первой недели может бросить спорт при малейшей сложности. Персональная память учитывает этот контекст и отправляет предупреждение только новичку.
                  </p>
                </div>
              </div>
            ) : (
              <div className='rounded-2xl bg-zinc-950 p-4 font-mono text-xs text-amber-200 overflow-x-auto'>
                {roundDetails?.hypothesis ? (
                  <pre className='whitespace-pre-wrap leading-relaxed'>
                    {roundDetails.hypothesis}
                  </pre>
                ) : (
                  <p className='text-zinc-500 italic'>hypothesis.json не сформирован.</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: AI STEPS TERMINAL */}
        {activeTab === 'steps' && (
          <div className='space-y-4 max-w-full font-mono text-xs'>
            {stepsList.length === 0 && !isLoading ? (
              <div className='text-center py-12 text-default-400 font-sans'>
                <p>Шаги рассуждений нейросети для этого раунда пока формируются.</p>
              </div>
            ) : (
              <>
                {stepsList.map((step, idx) => {
                  const getStepTypeLabel = () => {
                    if (step.type === 'tool') return `ИНСТРУМЕНТ: ${step.toolName || 'system'}`;
                    if (step.type === 'agent_response') return 'РАЗМЫШЛЕНИЕ И РЕШЕНИЕ (Thinking & CoT)';
                    if (step.type === 'system_message') return 'СИСТЕМНОЕ СОБЫТИЕ ОКРУЖЕНИЯ';
                    if (step.type === 'user_input') return 'ВХОДНЫЕ ДИРЕКТИВЫ ЭТАПА';
                    return (step.type as string).toUpperCase();
                  };

                  return (
                    <div
                      key={`${step.index}-${idx}`}
                      className='rounded-2xl bg-zinc-950 text-zinc-100 overflow-hidden'
                    >
                      {/* Step Header */}
                      <div className='flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 bg-zinc-900 text-[11px]'>
                        <div className='flex items-center gap-2'>
                          <span className='rounded bg-zinc-800 px-2 py-0.5 text-zinc-200 font-bold'>
                            Шаг {idx + 1} из {stepsList.length} <span className='text-zinc-400 font-normal'>(#{step.index})</span>
                          </span>
                          <span className='font-semibold text-primary-300 uppercase tracking-wider text-[11px]'>
                            {getStepTypeLabel()}
                          </span>
                          {step.state && (
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                              step.state === 'DONE' ? 'bg-emerald-950 text-emerald-400' : 'bg-amber-950 text-amber-400 animate-pulse'
                            }`}>
                              {step.state}
                            </span>
                          )}
                        </div>

                        <div className='flex items-center gap-3 text-zinc-400 text-[11px]'>
                          {step.durationSeconds ? (
                            <span>⏱ {step.durationSeconds.toFixed(2)}с</span>
                          ) : null}
                          {step.usage?.totalTokens ? (
                            <span>
                              {step.usage.totalTokens.toLocaleString()} ток.
                              {step.usage.thinkingTokens ? ` (think: ${step.usage.thinkingTokens.toLocaleString()})` : ''}
                            </span>
                          ) : null}
                        </div>
                      </div>

                      {/* Step Body */}
                      <div className='p-3.5 space-y-2 text-zinc-200'>
                        {step.content && (
                          <p className='text-zinc-300 font-sans leading-relaxed text-xs'>{step.content}</p>
                        )}

                        {/* Tool parameters */}
                        {step.toolParameters && Object.keys(step.toolParameters).length > 0 && (
                          <div className='space-y-1'>
                            <span className='text-[10px] uppercase tracking-wider text-zinc-500 font-sans'>
                              Параметры вызова:
                            </span>
                            <pre className='p-3 rounded-xl bg-zinc-900/90 text-zinc-300 text-[11px] overflow-x-auto whitespace-pre-wrap font-mono'>
                              {JSON.stringify(step.toolParameters, null, 2)}
                            </pre>
                          </div>
                        )}

                        {/* Tool Output (Terminal style) */}
                        {step.toolOutput && (
                          <div className='space-y-1 pt-1'>
                            <span className='text-[10px] uppercase tracking-wider text-emerald-500 font-sans flex items-center gap-1'>
                              <span className='h-1.5 w-1.5 rounded-full bg-emerald-500' />
                              Вывод терминала:
                            </span>
                            <pre className='p-3.5 rounded-xl bg-black text-emerald-400 text-[11px] overflow-x-auto max-h-72 overflow-y-auto whitespace-pre-wrap font-mono leading-normal'>
                              {step.toolOutput}
                            </pre>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}

                {/* Active live-generating block when round is running */}
                {selectedRound?.status === 'running' && (
                  <div className='rounded-2xl bg-default-100/80 p-4 space-y-2 animate-fade-in'>
                    <div className='flex items-center justify-between gap-2 flex-wrap'>
                      <div className='flex items-center gap-2.5'>
                        <span className='relative flex h-2 w-2 shrink-0'>
                          <span className='animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75' />
                          <span className='relative inline-flex rounded-full h-2 w-2 bg-emerald-500' />
                        </span>
                        <span className='font-bold text-xs text-foreground font-sans tracking-tight'>
                          Нейросеть Gemini сейчас активно работает
                        </span>
                      </div>
                      <span className='text-[11px] font-mono font-medium text-default-500 bg-default-200/70 px-2 py-0.5 rounded-md'>
                        Действие #{stepsList.length + 1}
                      </span>
                    </div>
                    <p className='text-xs text-default-600 dark:text-default-400 font-sans font-normal pl-4 leading-relaxed'>
                      Генерирует программный код эксперимента, формулирует гипотезу или выполняет инструмент...
                    </p>
                  </div>
                )}
              </>
            )}
            <div ref={stepsEndRef} />
          </div>
        )}

        {/* TAB 4: STDOUT LOG */}
        {activeTab === 'stdout' && (
          <div className='rounded-2xl bg-zinc-950 p-5 font-mono text-xs text-zinc-300 space-y-1 overflow-x-auto max-h-[calc(100vh-280px)] overflow-y-auto'>
            {terminalLogs.map((line, idx) => (
              <div key={idx} className='leading-relaxed whitespace-pre-wrap hover:bg-zinc-900/60 px-1 rounded'>
                {line}
              </div>
            ))}
            <div ref={stdoutEndRef} />
          </div>
        )}

        {/* TAB 5: EXPERIMENT CODE (SYNTAX HIGHLIGHTED) */}
        {activeTab === 'experiment' && (
          <div>
            {roundDetails?.experimentCode ? (
              <PythonCodeViewer
                code={roundDetails.experimentCode}
                filename={`experiment_r${selectedRound?.roundNum}.py`}
              />
            ) : (
              <div className='rounded-2xl bg-zinc-950 p-8 text-center space-y-2'>
                <p className='text-zinc-400 font-sans text-sm'>Код эксперимента (experiment.py) еще формируется автономным исследователем.</p>
                <p className='text-zinc-600 font-sans text-xs'>После генерации скрипта агентом здесь появится полный код с подсветкой синтаксиса Python.</p>
              </div>
            )}
          </div>
        )}

        {/* TAB 6: DIAGNOSTICS HUB */}
        {activeTab === 'diagnostics' && (
          <div className='space-y-5 max-w-4xl text-foreground pb-8'>
            {/* 1. Header Banner */}
            <div className='rounded-2xl bg-gradient-to-r from-default-100 to-default-200/50 p-5 space-y-2 border border-default-200/50'>
              <div className='flex flex-wrap items-center justify-between gap-3'>
                <div className='flex items-center gap-2.5'>
                  <div className='flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary'>
                    <SvgIcon aria-hidden='true' iconName='solar:shield-check-linear' width={20} />
                  </div>
                  <div>
                    <h3 className='text-sm font-bold text-foreground'>
                      Диагностический центр: Раунд #{selectedRound?.roundNum || 1}
                    </h3>
                    <p className='text-xs text-default-500'>
                      Анализ здоровья вычислительной среды, причин доработок и статуса самоисцеления
                    </p>
                  </div>
                </div>

                <div className='flex items-center gap-2'>
                  <span className={`text-xs font-semibold px-2.5 py-1 rounded-lg ${
                    selectedRound?.status === 'passed'
                      ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                      : selectedRound?.status === 'running'
                      ? 'bg-primary/15 text-primary'
                      : 'bg-amber-500/15 text-amber-600 dark:text-amber-400'
                  }`}>
                    {selectedRound?.status === 'passed'
                      ? 'Статус: Подтверждено'
                      : selectedRound?.status === 'running'
                      ? 'Статус: В процессе исполнения'
                      : 'Статус: Требуется доработка'}
                  </span>
                </div>
              </div>
            </div>

            {/* 2. Error Analysis & Auto-heal Card */}
            {errorObj ? (
              <div className='rounded-2xl bg-amber-500/10 dark:bg-amber-950/20 border border-amber-500/20 p-5 space-y-3'>
                <div className='flex items-center gap-2 text-amber-800 dark:text-amber-300 font-bold text-sm'>
                  <SvgIcon aria-hidden='true' iconName='solar:danger-triangle-bold' width={18} className='text-amber-500' />
                  <span>Анализ сбоя и контур самоисцеления</span>
                </div>

                <div className='grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs'>
                  <div className='p-3 rounded-xl bg-background/80 dark:bg-zinc-900/80 space-y-1'>
                    <span className='text-default-400 text-[11px] uppercase tracking-wider font-semibold'>Тип ошибки:</span>
                    <div className='font-mono font-bold text-amber-600 dark:text-amber-400'>{errorObj.type || 'Error'}</div>
                  </div>
                  <div className='p-3 rounded-xl bg-background/80 dark:bg-zinc-900/80 space-y-1'>
                    <span className='text-default-400 text-[11px] uppercase tracking-wider font-semibold'>Реакция супервизора:</span>
                    <div className='font-semibold text-emerald-600 dark:text-emerald-400'>Автоматический переход к доработке (Self-heal)</div>
                  </div>
                </div>

                <div className='p-3.5 rounded-xl bg-background/90 dark:bg-zinc-900/90 text-xs space-y-1.5'>
                  <span className='text-default-400 text-[11px] uppercase tracking-wider font-semibold'>Сообщение системы:</span>
                  <p className='font-mono text-foreground leading-relaxed'>{errorObj.message}</p>
                </div>

                {errorObj.traceback && (
                  <div className='space-y-1.5 pt-1'>
                    <button
                      type='button'
                      onClick={() => setIsTracebackOpen(!isTracebackOpen)}
                      className='text-xs font-semibold text-primary hover:underline flex items-center gap-1'
                    >
                      <span>{isTracebackOpen ? 'Скрыть трассировку стека (Traceback)' : 'Показать трассировку стека (Traceback)'}</span>
                      <SvgIcon aria-hidden='true' iconName={isTracebackOpen ? 'solar:alt-arrow-up-linear' : 'solar:alt-arrow-down-linear'} width={14} />
                    </button>
                    {isTracebackOpen && (
                      <pre className='p-3 rounded-xl bg-zinc-950 font-mono text-[11px] text-zinc-300 overflow-x-auto whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto'>
                        {errorObj.traceback}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            ) : selectedRound?.status === 'running' ? (
              <div className='rounded-2xl bg-primary/10 border border-primary/20 p-5 space-y-2'>
                <div className='flex items-center gap-2 text-primary font-bold text-sm'>
                  <Spinner size='sm' />
                  <span>Раунд активно исполняется в фоновом режиме</span>
                </div>
                <p className='text-xs text-default-600 dark:text-default-400 leading-relaxed'>
                  Контроллер сообщает об активном раунде. Выполняемый шаг и возможные ошибки отражаются в журнале событий.
                </p>
              </div>
            ) : (
              <div className='rounded-2xl bg-emerald-500/10 border border-emerald-500/20 p-5 space-y-2'>
                <div className='flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-bold text-sm'>
                  <SvgIcon aria-hidden='true' iconName='solar:check-circle-bold' width={18} />
                  <span>Состояние раунда — в сохранённом отчёте</span>
                </div>
                <p className='text-xs text-default-600 dark:text-default-400 leading-relaxed'>
                  Отсутствие сообщения об ошибке не подтверждает завершение стадий или научный допуск.
                </p>
              </div>
            )}

            {/* 3. Environment & Resources Card */}
            <div className='rounded-2xl bg-default-100/70 p-5 space-y-3'>
              <h4 className='text-xs font-bold uppercase tracking-wider text-default-600'>
                Параметры изолированного окружения
              </h4>
              <div className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs'>
                <div className='p-3 rounded-xl bg-background/80 dark:bg-zinc-900/80 space-y-1'>
                  <div className='text-default-400 text-[10px] uppercase font-semibold'>Исполнение</div>
                  <div className='font-bold text-foreground'>OCI Sandbox (Lima VM)</div>
                  <div className='text-[10px] text-default-500 font-mono'>Сеть: отключена (--network none)</div>
                </div>

                <div className='p-3 rounded-xl bg-background/80 dark:bg-zinc-900/80 space-y-1'>
                  <div className='text-default-400 text-[10px] uppercase font-semibold'>Модель AI</div>
                  <div className='font-bold text-foreground'>{roundDetails?.modelName || 'gemini-3.8-flash-high'}</div>
                  <div className='text-[10px] text-emerald-600 dark:text-emerald-400 font-mono'>Personal Cached Auth</div>
                </div>

                <div className='p-3 rounded-xl bg-background/80 dark:bg-zinc-900/80 space-y-1'>
                  <div className='text-default-400 text-[10px] uppercase font-semibold'>Целевой лимит памяти H4</div>
                  <div className='font-bold text-foreground'>16 384 байт (16 КиБ)</div>
                  <div className='text-[10px] text-emerald-600 dark:text-emerald-400 font-mono'>Фактический размер — в измерениях раунда</div>
                </div>

                <div className='p-3 rounded-xl bg-background/80 dark:bg-zinc-900/80 space-y-1'>
                  <div className='text-default-400 text-[10px] uppercase font-semibold'>Токены раунда</div>
                  <div className='font-bold text-foreground'>{totalTokensSummary.total.toLocaleString('ru-RU')}</div>
                  <div className='text-[10px] text-default-500 font-mono'>Рассуждения: {totalTokensSummary.thinking.toLocaleString('ru-RU')}</div>
                </div>
              </div>
            </div>

            {/* 4. Full Developer JSON (Accordion) */}
            {attemptReport && (
              <div className='rounded-2xl bg-default-100/50 p-4 space-y-2'>
                <div className='flex items-center justify-between'>
                  <span className='text-xs font-semibold text-default-500'>Сырой отчёт попытки (Attempt Report Payload):</span>
                  <Button
                    size='sm'
                    variant='secondary'
                    onClick={() => copyToClipboard(JSON.stringify(attemptReport, null, 2))}
                    className='text-xs h-7'
                  >
                    {copied ? 'Скопировано!' : 'Копировать JSON'}
                  </Button>
                </div>
                <div className='p-3 rounded-xl bg-zinc-950 font-mono text-[11px] text-zinc-400 overflow-x-auto max-h-60 overflow-y-auto leading-relaxed'>
                  <pre className='whitespace-pre-wrap'>{JSON.stringify(attemptReport, null, 2)}</pre>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
