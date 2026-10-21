'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Input } from '@heroui/react';
import { SvgIcon } from '@nutrifit/web-ui/SvgIcon';
import type { CravingEpochGroup, CravingRoundSummary } from '@/types';

interface CravingRoundsListProps {
  epochs: CravingEpochGroup[];
  selectedRound: CravingRoundSummary | null;
  onSelectRound: (round: CravingRoundSummary) => void;
  selectedTaskFilter: string;
  statusFilter: 'ALL' | 'ERRORS' | 'SUCCESS';
  onStatusFilterChange: (status: 'ALL' | 'ERRORS' | 'SUCCESS') => void;
  onResetTaskFilter?: () => void;
}

export function CravingRoundsList({
  epochs,
  selectedRound,
  onSelectRound,
  selectedTaskFilter,
  statusFilter,
  onStatusFilterChange,
  onResetTaskFilter,
}: CravingRoundsListProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const selectedRoundElementRef = useRef<HTMLDivElement | null>(null);

  // Автоматический скролл к выбранному раунду при выборе или переходе к прямому эфиру
  useEffect(() => {
    if (selectedRound && selectedRoundElementRef.current) {
      selectedRoundElementRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [selectedRound?.id]);

  // Human-friendly error translation
  const translateErrorHuman = (type: string | undefined, message: string | undefined) => {
    if (!message) return 'Неизвестный сбой при выполнении';
    if (message.includes('step limit 50 reached')) {
      return 'Исчерпан лимит шагов рассуждения — модель автоматически оптимизирует план для нового раунда.';
    }
    if (message.includes('timeout') || message.includes('timed out')) {
      return 'Время ожидания расчета истекло — запланирован безопасный перезапуск.';
    }
    if (message.includes('quota') || message.includes('rate limit')) {
      return 'Временное исчерпание лимитов вызовов AI — алгоритм ожидает окна восстановления.';
    }
    return message;
  };

  // Filter epochs, tasks, and rounds
  const filteredEpochs = useMemo(() => {
    return epochs
      .map((epoch) => {
        const filteredTasks = epoch.tasks
          .filter((task) => selectedTaskFilter === 'ALL' || task.taskId === selectedTaskFilter)
          .map((task) => {
            const filteredRounds = task.rounds.filter((round) => {
              // Status filter
              if (statusFilter === 'ERRORS' && !round.error && round.status !== 'failed' && round.status !== 'base-failed') {
                return false;
              }
              if (statusFilter === 'SUCCESS' && round.status !== 'passed') {
                return false;
              }
              // Search query
              if (searchQuery) {
                const q = searchQuery.toLowerCase();
                const matchesId = round.id.toLowerCase().includes(q);
                const matchesError = round.error?.message.toLowerCase().includes(q);
                const matchesTask = round.task.toLowerCase().includes(q);
                return matchesId || matchesError || matchesTask;
              }
              return true;
            });
            return {
              ...task,
              rounds: filteredRounds,
            };
          })
          .filter((task) => task.rounds.length > 0);

        return {
          ...epoch,
          tasks: filteredTasks,
        };
      })
      .filter((epoch) => epoch.tasks.length > 0);
  }, [epochs, selectedTaskFilter, statusFilter, searchQuery]);

  const totalRoundsCount = useMemo(() => {
    return filteredEpochs.reduce((acc, ep) => acc + ep.tasks.reduce((tAcc, t) => tAcc + t.rounds.length, 0), 0);
  }, [filteredEpochs]);

  // Чистый человекопонятный статус (без кричащих цветов и без оранжевого фона)
  const renderStatusBadge = (status: CravingRoundSummary['status'], hasError: boolean) => {
    if (status === 'running') {
      return (
        <span className='inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-emerald-600 dark:text-emerald-400 text-[10px] font-semibold whitespace-nowrap'>
          <span className='relative flex h-1.5 w-1.5 shrink-0'>
            <span className='animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75' />
            <span className='relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500' />
          </span>
          <span>Выполняется</span>
        </span>
      );
    }
    if (status === 'failed' || status === 'base-failed' || hasError) {
      return (
        <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-foreground text-[10px] font-medium whitespace-nowrap'>
          Доработка
        </span>
      );
    }
    if (status === 'passed') {
      return (
        <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-foreground text-[10px] font-medium whitespace-nowrap'>
          Подтвержден
        </span>
      );
    }
    // Вместо сырого 'unknown'
    return (
      <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-default-100 dark:bg-default-100/50 text-default-500 text-[10px] font-normal whitespace-nowrap'>
        В процессе проверки
      </span>
    );
  };

  return (
    <div className='flex flex-col h-[calc(100vh-74px)] border-r border-default-200 bg-background/50 overflow-hidden'>
      {/* Header & Full-width Search with comfortable top padding & generous margin */}
      <div className='p-3.5 pt-4 border-b border-default-200 space-y-2.5 bg-default-50/50'>
        <div className='flex items-center justify-between px-0.5 gap-2 flex-wrap'>
          <h2 className='text-xs font-bold uppercase tracking-wider text-default-600 flex items-center gap-1.5'>
            <SvgIcon aria-hidden='true' iconName='solar:history-linear' width={15} className='text-primary' />
            <span>Хронология раундов</span>
          </h2>

          <div className='flex items-center gap-2'>
            {onResetTaskFilter && (
              <button
                onClick={onResetTaskFilter}
                className='flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-900 dark:text-emerald-200 text-xs font-bold transition-all'
                title='Быстро вернуться к текущему выполняемому раунду в прямом эфире'
              >
                <span className='relative flex h-2 w-2'>
                  <span className='animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75' />
                  <span className='relative inline-flex rounded-full h-2 w-2 bg-emerald-600 dark:bg-emerald-400' />
                </span>
                <span>К текущему (Live)</span>
              </button>
            )}
            <span className='text-[11px] font-semibold text-default-500 bg-default-100 px-2 py-0.5 rounded-full'>
              {totalRoundsCount}
            </span>
          </div>
        </div>

        {/* Search input: Full width (100%) */}
        <div className='relative w-full'>
          <span className='absolute left-3 top-1/2 -translate-y-1/2 text-default-400 pointer-events-none flex items-center'>
            <SvgIcon aria-hidden='true' iconName='solar:magnifer-linear' width={14} />
          </span>
          <Input
            placeholder='Поиск по номеру раунда, этапу или ошибке...'
            aria-label='Поиск раундов'
            variant='primary'
            className='w-full pl-7'
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className='absolute right-2.5 top-1/2 -translate-y-1/2 text-default-400 hover:text-default-600'
            >
              <SvgIcon aria-hidden='true' iconName='solar:close-circle-bold' width={13} />
            </button>
          )}
        </div>

        {/* Filter chips: compact */}
        <div className='flex items-center gap-1.5 pt-0.5'>
          <button
            onClick={() => onStatusFilterChange('ALL')}
            className={`text-[11px] px-2.5 py-0.5 rounded-lg transition-colors ${
              statusFilter === 'ALL'
                ? 'bg-foreground text-background font-semibold'
                : 'bg-default-100 text-default-600 hover:bg-default-200'
            }`}
          >
            Все
          </button>
          <button
            onClick={() => onStatusFilterChange('SUCCESS')}
            className={`text-[11px] px-2.5 py-0.5 rounded-lg transition-colors flex items-center gap-1 ${
              statusFilter === 'SUCCESS'
                ? 'bg-success text-success-foreground font-semibold'
                : 'bg-success/10 text-success hover:bg-success/20'
            }`}
          >
            <span>Подтвержденные</span>
          </button>
          <button
            onClick={() => onStatusFilterChange('ERRORS')}
            className={`text-[11px] px-2.5 py-0.5 rounded-lg transition-colors flex items-center gap-1 ${
              statusFilter === 'ERRORS'
                ? 'bg-amber-600 text-white font-semibold'
                : 'bg-amber-500/10 text-amber-700 dark:text-amber-400 hover:bg-amber-500/20'
            }`}
          >
            <span>Доработки</span>
          </button>
        </div>
      </div>

      {/* Rounds Scroll Area: Compact and dense padding */}
      <div className='flex-1 overflow-y-auto p-2 space-y-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden'>
        {filteredEpochs.length === 0 ? (
          searchQuery ? (
            <div className='text-center py-10 px-4'>
              <SvgIcon aria-hidden='true' iconName='solar:inbox-linear' width={28} className='mx-auto text-default-300 mb-1.5' />
              <p className='text-xs text-default-500 font-medium'>Ничего не найдено</p>
              <p className='text-[11px] text-default-400 mt-0.5'>Попробуйте изменить поисковый запрос</p>
            </div>
          ) : selectedTaskFilter !== 'ALL' ? (
            <div className='rounded-2xl bg-default-100/70 p-4 text-center space-y-3 my-3 mx-1 border border-default-200/50'>
              <div className='flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary mx-auto'>
                <SvgIcon aria-hidden='true' iconName='solar:clock-circle-linear' width={22} />
              </div>
              <div className='space-y-1.5'>
                <h4 className='text-xs font-bold text-foreground'>
                  Этап {selectedTaskFilter} ожидает своей очереди
                </h4>
                <p className='text-[11px] text-default-500 leading-relaxed max-w-xs mx-auto'>
                  Кампания решает задачи последовательно. Текущий этап и состояние показаны в панели управления.
                </p>

                <p className='text-[11px] text-default-500'>
                  Допуск определяется сохранёнными измерениями и решением критика. Процент готовности и время завершения не рассчитаны.
                </p>
              </div>
              {onResetTaskFilter && (
                <button
                  onClick={onResetTaskFilter}
                  className='inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:bg-primary-600 transition-colors shadow-2xs'
                >
                  <SvgIcon aria-hidden='true' iconName='solar:play-circle-linear' width={14} />
                  <span>Показать активный этап</span>
                </button>
              )}
            </div>
          ) : (
            <div className='text-center py-10 px-4'>
              <SvgIcon aria-hidden='true' iconName='solar:inbox-linear' width={28} className='mx-auto text-default-300 mb-1.5' />
              <p className='text-xs text-default-500 font-medium'>Раунды формируются</p>
              <p className='text-[11px] text-default-400 mt-0.5'>Ожидание данных от автономного контроллера</p>
            </div>
          )
        ) : (
          filteredEpochs.map((epoch) => (
            <div key={epoch.epochStr} className='space-y-1.5'>
              {/* Epoch header: строго по левому краю */}
              <div className='flex items-center justify-between px-3 pt-2 pb-0.5'>
                <span className='text-[11px] font-bold text-foreground uppercase tracking-wider'>
                  Эпоха #{epoch.epochNum}
                </span>
                <span className='text-[10px] text-default-400 font-normal'>
                  Цикл поиска
                </span>
              </div>

              {/* Tasks & Rounds */}
              {epoch.tasks.map((task) => (
                <div key={task.taskId} className='space-y-1'>
                  {/* Task/Stage header: строго по левому краю */}
                  <div className='text-[11px] font-semibold text-default-500 flex items-center justify-between px-3 py-0.5'>
                    <span>Этап {task.taskId}</span>
                    <span className='text-[10px] text-default-400 font-mono'>({task.rounds.length} раундов)</span>
                  </div>

                  <div className='space-y-1'>
                    {task.rounds.map((round) => {
                      const isSelected = selectedRound?.id === round.id;
                      const hasError = Boolean(round.error) || round.status === 'failed' || round.status === 'base-failed';

                      return (
                        <div
                          key={round.id}
                          ref={isSelected ? selectedRoundElementRef : undefined}
                          role='button'
                          tabIndex={0}
                          onClick={() => onSelectRound(round)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              onSelectRound(round);
                            }
                          }}
                          className={`w-full rounded-2xl transition-all text-left cursor-pointer p-3 space-y-1.5 focus:outline-hidden focus:ring-2 focus:ring-default-400/40 ${
                            isSelected
                              ? 'bg-default-200/90 dark:bg-default-100/60 text-foreground font-medium'
                              : 'bg-default-100/70 hover:bg-default-200/60 text-foreground'
                          }`}
                        >
                          {/* Top row: Round number + duration + status badge */}
                          <div className='flex items-center justify-between gap-2'>
                            <div className='flex items-center gap-1.5'>
                              <span className='text-xs font-bold text-foreground'>
                                Раунд #{round.roundNum}
                              </span>
                              <span className='text-[10px] text-default-400 font-mono'>
                                {round.activeSeconds > 0 ? `${Math.round(round.activeSeconds)}с` : '—'}
                              </span>
                            </div>
                            {renderStatusBadge(round.status, hasError)}
                          </div>

                          {/* Human-friendly Error translation: строго по левой линии без отступов и без лишних рамок */}
                          {round.error && (
                            <div className='pt-0.5 space-y-0.5 text-left'>
                              <div className='text-[10px] font-semibold text-foreground'>
                                Доработка: <span className='font-normal text-default-500'>{round.error.type || 'Автоматическая корректировка'}</span>
                              </div>
                              <div className='text-default-700 dark:text-default-300 text-[11px] leading-snug'>
                                {translateErrorHuman(round.error.type, round.error.message)}
                              </div>
                            </div>
                          )}

                          {/* Report ready indicator: строго по левой линии */}
                          {round.hasReport && (
                            <div className='text-[11px] text-default-500 font-normal pt-0.5'>
                              <span>Сформирован отчет раунда</span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
