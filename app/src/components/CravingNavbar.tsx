'use client';

import { Button, Tooltip } from '@heroui/react';
import { SvgIcon } from '@nutrifit/web-ui/SvgIcon';
import { NutriFitBrandLogo } from './NutriFitBrandLogo';
import type { CravingDashboardState } from '@/types';

interface CravingNavbarProps {
  state: CravingDashboardState | null;
  campaign: string;
  onCampaignChange: (campaign: string) => void;
  onStart: () => void;
  onStop: () => void;
  isActionPending: boolean;
  actionMessage: string | null;
  liveActiveSeconds?: number;
  isSidebarOpen?: boolean;
  onToggleSidebar?: () => void;
}

export function CravingNavbar({
  state,
  onStart,
  onStop,
  isActionPending,
  actionMessage,
  liveActiveSeconds,
  isSidebarOpen = true,
  onToggleSidebar,
}: CravingNavbarProps) {
  const isRunning = state?.active && ['ready', 'running'].includes(state.status);
  const isWaitingQuota = state?.status === 'waiting_quota';
  const isWaitingEnv = state?.status === 'waiting_environment' || state?.status === 'repair_pending';
  const isStopped = !state?.active || state?.status === 'stopped' || state?.status === 'interrupted';

  const formatSeconds = (sec: number | undefined) => {
    if (!sec || sec < 0) return '0с';
    const totalSec = Math.floor(sec);
    const mins = Math.floor(totalSec / 60);
    const hours = Math.floor(mins / 60);
    const remMins = mins % 60;
    const remSecs = totalSec % 60;
    if (hours > 0) return `${hours}ч ${remMins}м ${remSecs}с`;
    if (mins > 0) return `${mins}м ${remSecs}с`;
    return `${remSecs}с`;
  };

  // Человекопонятное описание фазы
  const getPhaseDescription = (phase: string | undefined) => {
    switch (phase?.toLowerCase()) {
      case 'research':
        return { label: 'Генерация гипотезы и кода', hint: 'Gemini анализирует прошлый опыт и пишет код нового эксперимента' };
      case 'qualification':
        return { label: 'Квалификация и проверка', hint: 'Проверка структуры кода и гипотезы перед запуском' };
      case 'replay':
        return { label: 'Изолированные вычисления', hint: 'Запуск вычислений в закрытом OCI-контейнере' };
      case 'critic':
        return { label: 'Оценка независимым критиком', hint: 'Критик беспристрастно проверяет точность и отсутствие ложных тревог' };
      case 'completed':
        return { label: 'Цикл завершен', hint: 'Все запланированные этапы успешно выполнены' };
      default:
        return { label: phase || 'Ожидание задачи', hint: 'Система ожидает следующего шага' };
    }
  };

  const phaseInfo = getPhaseDescription(state?.phase);

  return (
    <header className='fixed top-0 left-0 right-0 z-50 w-full border-b border-default-200 bg-background/95 backdrop-blur-md px-4 py-2.5 transition-all shadow-none'>
      <div className='flex flex-wrap items-center justify-between gap-3'>
        {/* Left: Burger Menu Toggle, NutriFit Brand Logo & Craving Engine Title */}
        <div className='flex items-center gap-2.5 min-w-0'>
          {onToggleSidebar && (
            <button
              onClick={onToggleSidebar}
              className='p-1.5 -ml-1 rounded-xl text-default-600 hover:text-foreground hover:bg-default-100 dark:hover:bg-default-100/40 transition-colors focus:outline-hidden'
              title={isSidebarOpen ? 'Схлопнуть боковую панель задач (Roadmap)' : 'Развернуть боковую панель задач (Roadmap)'}
              aria-label='Переключить боковую панель'
            >
              <SvgIcon aria-hidden='true' iconName='solar:hamburger-menu-linear' width={22} />
            </button>
          )}

          <NutriFitBrandLogo size={34} />

          <div>
            <div className='flex items-center gap-2 flex-wrap'>
              <h1 className='text-base font-bold tracking-tight text-foreground flex items-center gap-1.5'>
                <span>Craving Engine</span>
                <span className='text-xs font-normal text-default-400'>• Исследование срывов</span>
              </h1>

              {/* Status Badge: neutral background with green dot for active status */}
              {isRunning && (
                <span className='inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-default-100 dark:bg-default-100/50 text-emerald-600 dark:text-emerald-400 text-xs font-semibold whitespace-nowrap ml-2 mr-1'>
                  <span className='relative flex h-2 w-2 shrink-0'>
                    <span className='animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75' />
                    <span className='relative inline-flex rounded-full h-2 w-2 bg-emerald-500' />
                  </span>
                  <span>В работе (Активен)</span>
                </span>
              )}
              {isWaitingQuota && (
                <span className='inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-default-100 text-foreground text-xs font-medium whitespace-nowrap ml-2 mr-1'>
                  <SvgIcon aria-hidden='true' iconName='solar:clock-circle-bold' width={14} className='text-default-500' />
                  <span>Ожидание лимитов Gemini</span>
                </span>
              )}
              {isWaitingEnv && (
                <span className='inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-default-100 text-foreground text-xs font-medium whitespace-nowrap ml-2 mr-1'>
                  <SvgIcon aria-hidden='true' iconName='solar:restart-linear' width={14} className='text-default-500' />
                  <span>Восстановление среды</span>
                </span>
              )}
              {isStopped && (
                <span className='inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-default-100 text-default-500 text-xs font-medium whitespace-nowrap ml-2 mr-1'>
                  <SvgIcon aria-hidden='true' iconName='solar:pause-circle-linear' width={14} />
                  <span>Исследование приостановлено</span>
                </span>
              )}
            </div>

            {/* Model & Secondary metadata (NO "компания Gemini", PID made small/subtle) */}
            <div className='text-xs text-default-500 mt-0.5 flex items-center gap-2 flex-wrap'>
              <span className='flex items-center gap-1'>
                <span>Модель AI:</span>
                <span className='text-primary font-medium bg-primary/10 px-1.5 py-0.2 rounded text-[11px]'>
                  gemini-3.8-flash-high
                </span>
              </span>
              <span>•</span>
              <span className='text-default-400 text-[11px]'>
                Контейнер: <span className='text-default-600 font-mono'>Lima VM / Isolated</span>
              </span>
              {state?.pid && (
                <>
                  <span>•</span>
                  <span className='text-default-400 text-[11px] font-mono'>
                    PID: {state.pid}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Center: Key Metrics Summary with Clear Unified Terminology & Tooltips */}
        <div className='hidden lg:flex items-center gap-3 bg-default-100/80 rounded-2xl px-3.5 py-1.5 text-xs shadow-none !shadow-none border-none'>
          {/* Current Experiment: Epoch & Round */}
          <Tooltip>
            <div tabIndex={0} className='cursor-help'>
              <div className='text-[10px] text-default-400 uppercase tracking-wider font-semibold'>
                Текущий эксперимент
              </div>
              <div className='font-bold text-foreground text-xs'>
                Эпоха {state?.epoch ?? 1} <span className='text-default-400'>•</span> Раунд {state?.round ?? 1}
              </div>
            </div>
          <Tooltip.Content placement='bottom'>Эпоха — глобальный цикл научного поиска. Раунд — законченный эксперимент по формулированию гипотезы, генерации кода и оценке критиком.</Tooltip.Content></Tooltip>

          <div className='h-6 w-px bg-default-200' />

          {/* Phase / Action */}
          <Tooltip>
            <div tabIndex={0} className='cursor-help max-w-[180px]'>
              <div className='text-[10px] text-default-400 uppercase tracking-wider font-semibold'>
                Текущее действие
              </div>
              <div className='font-medium text-foreground text-xs truncate text-primary'>
                {phaseInfo.label}
              </div>
            </div>
          <Tooltip.Content placement='bottom'>{phaseInfo.hint}</Tooltip.Content></Tooltip>

          <div className='h-6 w-px bg-default-200' />

          {/* Runtime */}
          <Tooltip>
            <div tabIndex={0} className='cursor-help'>
              <div className='text-[10px] text-default-400 uppercase tracking-wider font-semibold flex items-center gap-1'>
                {isRunning && (
                  <span className='inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse' />
                )}
                <span>Время работы</span>
              </div>
              <div className='font-semibold text-foreground text-xs font-mono flex items-center gap-1.5'>
                <span>{formatSeconds(liveActiveSeconds ?? state?.cumulative?.active_seconds)}</span>
                {isRunning && (
                  <span className='text-[10px] text-emerald-800 dark:text-emerald-300 font-sans font-semibold'>• тикает</span>
                )}
              </div>
            </div>
          <Tooltip.Content placement='bottom'>Суммарное время активных вычислений автономного процесса (учитывается каждая секунда в реальном времени)</Tooltip.Content></Tooltip>

          <div className='h-6 w-px bg-default-200' />

          {/* Code Runs (Process Attempts) */}
          <Tooltip>
            <div tabIndex={0} className='cursor-help'>
              <div className='text-[10px] text-default-400 uppercase tracking-wider font-semibold'>
                Запусков кода
              </div>
              <div className='font-semibold text-foreground text-xs font-mono'>
                {state?.cumulative?.process_attempts ?? 0}
              </div>
            </div>
          <Tooltip.Content placement='bottom'>Суммарное число запусков скриптов и тестов в изолированном контейнере за всю кампанию</Tooltip.Content></Tooltip>

          <div className='h-6 w-px bg-default-200' />

          {/* Corrections / Repairs (Failures) */}
          <Tooltip>
            <div tabIndex={0} className='cursor-help'>
              <div className='text-[10px] text-default-400 uppercase tracking-wider font-semibold'>
                Коррекций
              </div>
              <div className={`font-semibold text-xs font-mono ${(state?.failures ?? 0) > 0 ? 'text-amber-500 font-bold' : 'text-default-600'}`}>
                {state?.failures ?? 0}
              </div>
            </div>
          <Tooltip.Content placement='bottom'>Количество раундов, где результат не удовлетворил строгим критериям критика и потребовал корректировки</Tooltip.Content></Tooltip>
        </div>

        {/* Right: Controls & Single Toggle Action Button (Play / Pause) */}
        <div className='flex items-center gap-2.5 flex-wrap'>
          {/* Action notification message */}
          {actionMessage && (
            <div className='text-xs font-medium text-primary bg-primary/15 px-3 py-1 rounded-xl animate-fade-in'>
              {actionMessage}
            </div>
          )}

          {/* Single Unified Play / Pause Toggle Button */}
          {isRunning ? (
            <Button
              isPending={isActionPending}
              onPress={onStop}
              size='sm'
              variant='secondary'
              className='font-semibold shadow-none !shadow-none border-none'
            >
              <SvgIcon aria-hidden='true' iconName='solar:pause-bold' width={16} />
              <span>Приостановить исследование</span>
            </Button>
          ) : (
            <Button
              isPending={isActionPending}
              onPress={onStart}
              size='sm'
              variant='primary'
              className='font-semibold shadow-none !shadow-none border-none'
            >
              <SvgIcon aria-hidden='true' iconName='solar:play-bold' width={16} />
              <span>Запустить исследование</span>
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
