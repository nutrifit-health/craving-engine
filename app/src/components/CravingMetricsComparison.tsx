'use client';

import { useState, useMemo } from 'react';
import { Button, Tooltip } from '@heroui/react';
import { SvgIcon } from '@nutrifit/web-ui/SvgIcon';
import type { CravingMetricsData, CravingRoundSummary } from '@/types';

interface CravingMetricsComparisonProps {
  metrics?: CravingMetricsData | null;
  bestMetrics?: CravingMetricsData | null;
  selectedRound?: CravingRoundSummary | null;
  isActiveRound?: boolean;
}

export function CravingMetricsComparison({
  metrics,
  bestMetrics,
  selectedRound,
  isActiveRound,
}: CravingMetricsComparisonProps) {
  // Режим просмотра: 'best' (рекорд кампании) или 'round' (выбранный раунд)
  const hasRoundMetrics = Boolean(metrics?.rawJson);
  const [viewMode, setViewMode] = useState<'best' | 'round'>(hasRoundMetrics ? 'round' : 'best');

  // Активные данные: либо выбранный раунд, либо лучший результат
  const activeData: CravingMetricsData | null = useMemo(() => {
    if (viewMode === 'round') return metrics || null;
    return bestMetrics || null;
  }, [viewMode, hasRoundMetrics, metrics, bestMetrics]);

  const [selectedWorld, setSelectedWorld] = useState<string>(
    activeData?.bestConfigId || ''
  );
  const [activeKpiInfo, setActiveKpiInfo] = useState<string | null>(null);

  const kpis = activeData?.kpis || [];
  const simulationWorlds = activeData?.simulationWorlds || [];

  return (
    <div className='space-y-6 w-full min-w-0 pb-10 text-foreground'>
      {/* Selector: Best result vs Selected round metrics */}
      <div className='flex flex-wrap items-center justify-between gap-3 bg-default-100/70 p-2.5 px-4 rounded-2xl'>
        <div className='flex items-center gap-2'>
          <span className='text-xs font-semibold text-default-500'>Отображение:</span>
          <div className='flex items-center gap-1.5 bg-background p-1 rounded-xl shadow-2xs'>
            <button
              type='button'
              onClick={() => setViewMode('best')}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
                viewMode === 'best'
                  ? 'bg-primary text-primary-foreground shadow-xs'
                  : 'text-default-500 hover:text-foreground hover:bg-default-100'
              }`}
            >
              <SvgIcon aria-hidden='true' iconName='solar:cup-star-bold' width={14} />
              <span>Последний завершённый расчёт</span>
            </button>

            <button
              type='button'
              onClick={() => setViewMode('round')}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
                viewMode === 'round'
                  ? 'bg-primary text-primary-foreground shadow-xs'
                  : 'text-default-500 hover:text-foreground hover:bg-default-100'
              }`}
            >
              <SvgIcon aria-hidden='true' iconName='solar:test-tube-bold' width={14} />
              <span>
                {selectedRound ? `Раунд #${selectedRound?.roundNum}` : 'Выбранный раунд'}
                {hasRoundMetrics ? ' (Есть данные)' : ''}
              </span>
            </button>
          </div>
        </div>

        <div className='text-xs text-default-500 flex items-center gap-2'>
          <span className='inline-block h-2 w-2 rounded-full bg-emerald-500' />
          <span>Динамическое обновление после каждого цикла</span>
        </div>
      </div>

      {/* Notice if selected round has no metrics yet */}
      {viewMode === 'round' && !hasRoundMetrics && (
        <div className='p-4 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-xs flex items-start gap-3'>
          <SvgIcon aria-hidden='true' iconName='solar:info-circle-bold' width={20} className='text-amber-500 shrink-0 mt-0.5' />
          <div className='space-y-1'>
            <div className='font-bold text-amber-800 dark:text-amber-300'>
              Метрики для Раунда #{selectedRound?.roundNum || 1} ещё рассчитываются
            </div>
            <p className='leading-relaxed text-default-600 dark:text-default-400'>
              {isActiveRound
                ? 'В активном раунде прямо сейчас модель генерирует код гипотезы и готовит симуляцию. Свежие метрики появятся здесь сразу после выполнения скрипта в изолированном OCI-контейнере. Измерений этого раунда пока нет.'
                : 'В этом архивном раунде эксперимент завершился до расчета финальных метрик (на этапе ранней валидации или системного сбоя). Измерения не подменяются результатами другого раунда.'}
            </p>
          </div>
        </div>
      )}

      {/* Header Banner */}
      <div className='rounded-2xl bg-gradient-to-r from-primary/15 via-primary/5 to-emerald-500/10 p-6 space-y-3'>
        <div className='flex flex-wrap items-center justify-between gap-3'>
          <div className='flex items-center gap-2.5'>
            <div className='flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground'>
              <SvgIcon aria-hidden='true' iconName='solar:chart-square-bold' width={22} />
            </div>
            <div>
              <h2 className='text-base font-bold text-foreground'>
                Измерения синтетического эксперимента
              </h2>
              <p className='text-xs text-default-500'>
                Исследовательские результаты. Научный допуск и внедрение проверяются отдельно.
              </p>
            </div>
          </div>

          <div className='flex items-center gap-2'>
            <span className='font-semibold text-xs text-primary-700 dark:text-primary-300 bg-primary-500/15 px-2.5 py-1 rounded-lg'>
              {activeData?.sourceLabel || 'Нет измерений'}
            </span>
            <span className='text-[11px] font-mono text-default-400 bg-background/80 px-2.5 py-1 rounded-lg'>
              {activeData?.totalDecisions ? `${activeData.totalDecisions.toLocaleString('ru-RU')} решений` : 'Число решений неизвестно'} • {activeData?.nUsers ?? '—'} пользователей
            </span>
          </div>
        </div>

        <p className='text-xs text-default-600 dark:text-default-400 leading-relaxed max-w-3xl pt-1'>
          В ходе кампании автономный агент тестирует альтернативные конфигурации и наборы признаков в изолированной виртуальной машине. Сравнивать разные миры и выборки по одному максимуму AUC нельзя. Гейт памяти не измеряет число уведомлений; для скорости и батареи нужны отдельные замеры на устройствах.
        </p>
      </div>

      {/* SECTION 1: 4 КЛЮЧЕВЫХ KPI КАРТОЧКИ БЕЗ ИКОНОК, С РАСКРЫТИЕМ ОПИСАНИЯ ПО КЛИКУ */}
      <div className='space-y-3'>
        <div className='flex items-center gap-2 px-1'>
          <h3 className='text-xs font-bold uppercase tracking-wider text-default-600'>
            Ключевые показатели эффективности (KPI)
          </h3>
        </div>

        <div className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5'>
          {kpis.map((kpi) => {
            const isInfoOpen = activeKpiInfo === kpi.id;
            return (
              <div
                key={kpi.id}
                className={`rounded-2xl ${kpi.bg} p-4 space-y-3 transition-transform hover:-translate-y-0.5 flex flex-col justify-between`}
              >
                <div className='space-y-2'>
                  <div className='flex items-start justify-between gap-2'>
                    <div className='text-sm font-bold text-foreground leading-snug flex-1'>
                      {kpi.title}
                    </div>

                    <button
                      type='button'
                      onClick={() => setActiveKpiInfo(isInfoOpen ? null : kpi.id)}
                      className={`p-1.5 rounded-xl transition-all shrink-0 flex items-center justify-center ${
                        isInfoOpen
                          ? 'bg-primary text-primary-foreground font-bold shadow-xs'
                          : 'text-default-400 hover:text-foreground hover:bg-default-200/60'
                      }`}
                      title={isInfoOpen ? 'Скрыть справку' : 'Нажмите, чтобы прочитать подробное описание метрики'}
                      aria-label='Справка по метрике'
                    >
                      <SvgIcon aria-hidden='true' iconName='solar:question-circle-linear' width={18} />
                    </button>
                  </div>

                  {isInfoOpen && (
                    <div className='p-3 rounded-xl bg-background/95 border border-default-200 text-xs text-foreground space-y-1 mt-1 shadow-sm'>
                      <div className='font-bold text-primary text-[11px] flex items-center justify-between'>
                        <span>Справка по метрике</span>
                        <button
                          type='button'
                          onClick={(e) => {
                            e.stopPropagation();
                            setActiveKpiInfo(null);
                          }}
                          className='text-default-400 hover:text-foreground text-[10px]'
                        >
                          ✕
                        </button>
                      </div>
                      <p className='text-[11px] text-default-600 dark:text-default-300 leading-relaxed'>
                        {kpi.tooltip}
                      </p>
                    </div>
                  )}
                </div>

                {/* Before vs After Numbers in dedicated clean container */}
                <div className='space-y-2 pt-1'>
                  <div className='p-2.5 rounded-xl bg-background/70 dark:bg-background/40 space-y-1.5'>
                    <div className='flex items-baseline justify-between gap-2 text-[11px] text-default-500 font-mono'>
                      <span className='whitespace-nowrap shrink-0'>До запуска:</span>
                      <span className='line-through text-default-400 text-right font-medium'>{kpi.before}</span>
                    </div>
                    <div className='flex items-baseline justify-between gap-2 text-xs font-mono font-bold text-foreground'>
                      <span className='whitespace-nowrap shrink-0'>После обучения:</span>
                      <span className='text-emerald-600 dark:text-emerald-400 text-sm text-right'>{kpi.after}</span>
                    </div>
                  </div>

                  <div>
                    <span className='inline-flex items-center gap-1 text-[11px] font-bold px-2.5 py-1 rounded-lg bg-background/90 text-emerald-600 dark:text-emerald-400 shadow-2xs'>
                      <span>Разность: {kpi.delta}</span>
                    </span>
                  </div>

                  <p className='text-[11px] text-default-600 dark:text-default-400 leading-relaxed pt-0.5'>
                    {kpi.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* SECTION 4: СРАВНИТЕЛЬНАЯ ТАБЛИЦА ВСЕХ СИМУЛЯЦИЙ */}
      <div className='space-y-3 pt-2'>
        <div className='flex flex-wrap items-center justify-between gap-2 px-1'>
          <h3 className='text-xs font-bold uppercase tracking-wider text-default-600'>
            Конфигурации и сценарии
          </h3>
          <span className='text-[11px] text-default-400 font-mono'>
            Данные выбранного OCI-расчёта; — означает отсутствие измерения
          </span>
        </div>

        <div className='rounded-2xl bg-default-100/60 overflow-hidden'>
          <div className='overflow-x-auto'>
            <table className='w-full text-left text-xs'>
              <thead className='bg-default-200/50 text-[11px] uppercase tracking-wider text-default-600 font-semibold'>
                <tr>
                  <th className='py-3 px-4'>Симуляционный мир / Конфигурация</th>
                  <th className='py-3 px-3 whitespace-nowrap'>Категория</th>
                  <th className='py-3 px-3 font-mono whitespace-nowrap'>Brier Score</th>
                  <th className='py-3 px-3 font-mono whitespace-nowrap'>Δ Brier</th>
                  <th className='py-3 px-3 font-mono whitespace-nowrap'>Macro AUC</th>
                  <th className='py-3 px-3 font-mono whitespace-nowrap'>Вентиль</th>
                  <th className='py-3 px-3 font-mono whitespace-nowrap'>Память</th>
                  <th className='py-3 px-4 text-right whitespace-nowrap'>Статус</th>
                </tr>
              </thead>
              <tbody className='divide-y divide-default-200/40 text-foreground'>
                {simulationWorlds.map((row) => {
                  const isCurrent = row.id === selectedWorld;
                  const isNegativeDelta = row.deltaBrier.startsWith('-');
                  const isZeroDelta = row.deltaBrier === '0.0000';

                  return (
                    <tr
                      key={row.id}
                      onClick={() => setSelectedWorld(row.id)}
                      className={`cursor-pointer transition-colors ${
                        isCurrent ? 'bg-primary/10 font-medium' : 'hover:bg-default-200/30'
                      }`}
                    >
                      <td className='py-3 px-4'>
                        <div className='font-semibold text-foreground'>{row.name}</div>
                        <div className='text-[10px] text-default-400 font-mono'>{row.id}</div>
                      </td>
                      <td className='py-3 px-3 text-[11px] text-default-500 whitespace-nowrap'>{row.category}</td>
                      <td className='py-3 px-3 font-mono font-medium whitespace-nowrap'>{row.brier}</td>
                      <td className='py-3 px-3 font-mono whitespace-nowrap'>
                        <span
                          className={`font-semibold ${
                            isNegativeDelta
                              ? 'text-emerald-600 dark:text-emerald-400'
                              : isZeroDelta
                              ? 'text-default-500'
                              : 'text-amber-600 dark:text-amber-400'
                          }`}
                        >
                          {row.deltaBrier}
                        </span>
                      </td>
                      <td className='py-3 px-3 font-mono font-medium whitespace-nowrap'>{row.auc}</td>
                      <td className='py-3 px-3 font-mono text-default-600 whitespace-nowrap'>{row.gateOpen}</td>
                      <td className='py-3 px-3 font-mono text-[11px] text-default-600 whitespace-nowrap font-medium'>
                        {row.memory}
                      </td>
                      <td className='py-3 px-4 text-right whitespace-nowrap font-medium text-[11px]'>
                        {row.status === 'Лучший результат' ? (
                          <span className='text-emerald-600 dark:text-emerald-400 font-bold'>
                            Лучший результат
                          </span>
                        ) : row.status === 'Деградация (спам)' ? (
                          <span className='text-rose-600 dark:text-rose-400 font-bold'>
                            Деградация (спам)
                          </span>
                        ) : row.status === 'Подтверждено' ? (
                          <span className='text-emerald-600 dark:text-emerald-400 font-semibold'>
                            Подтверждено
                          </span>
                        ) : row.status === 'Шум заблокирован' ? (
                          <span className='text-default-600 dark:text-default-400 font-medium'>
                            Шум заблокирован
                          </span>
                        ) : row.status === 'Стабильно' || row.status === 'Устойчиво' ? (
                          <span className='text-foreground font-semibold'>
                            {row.status}
                          </span>
                        ) : (
                          <span className='text-default-500'>
                            {row.status}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
