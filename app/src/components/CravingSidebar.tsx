'use client';

import { useMemo } from 'react';
import { Chip, Tooltip } from '@heroui/react';
import { SvgIcon, type IconsType } from '@nutrifit/web-ui/SvgIcon';
import type { CravingDashboardState, CravingEpochGroup } from '@/types';

interface CravingSidebarProps {
  state: CravingDashboardState | null;
  epochs?: CravingEpochGroup[];
  selectedTaskFilter: string;
  onSelectTaskFilter: (task: string) => void;
  onOpenEvents: () => void;
  onSelectTab?: (tab: 'metrics' | 'guide') => void;
}

export function CravingSidebar({
  epochs = [],
  selectedTaskFilter,
  onSelectTaskFilter,
  onOpenEvents,
  onSelectTab,
}: CravingSidebarProps) {
  // Вычисляем количество раундов для каждого этапа
  const roundCounts = useMemo(() => {
    const counts: Record<string, number> = { ALL: 0, H2: 0, H3: 0, H4: 0, H5: 0 };
    for (const ep of epochs) {
      for (const t of ep.tasks) {
        const count = t.rounds?.length || 0;
        counts[t.taskId] = (counts[t.taskId] || 0) + count;
        counts.ALL += count;
      }
    }
    return counts;
  }, [epochs]);

  // Активная в данный момент цель исследования
  const activeMilestoneId = useMemo(() => {
    for (const ep of epochs) {
      for (const t of ep.tasks) {
        if (t.rounds.some((r) => r.status === 'running')) {
          return t.taskId;
        }
      }
    }
    return 'H2';
  }, [epochs]);

  const milestones: { id: string; title: string; icon: IconsType; summary: string; count: number }[] = [
    {
      id: 'ALL',
      title: 'Все цели проекта (H1–H5)',
      icon: 'solar:layers-linear',
      summary: 'Стратегический roadmap: 5 задач исследования и хронология всех раундов.',
      count: roundCounts.ALL,
    },
    {
      id: 'H2',
      title: 'H2: Фундамент и симуляторы',
      icon: 'solar:test-tube-linear',
      summary: 'Генераторы поведения людей для безопасной проверки до работы с живыми данными.',
      count: roundCounts.H2,
    },
    {
      id: 'H3',
      title: 'H3: Факторы риска срывов',
      icon: 'solar:graph-up-linear',
      summary: 'Отбор ключевых сигналов (стаж спорта, сон, пропуски) для отсечения ложных тревог.',
      count: roundCounts.H3,
    },
    {
      id: 'H4',
      title: 'H4: Память человека (≤ 16 КиБ)',
      icon: 'solar:cpu-bolt-linear',
      summary: 'Архитектура Mushroom Body: учет стажа тренировок прямо на смартфоне.',
      count: roundCounts.H4,
    },
    {
      id: 'H5',
      title: 'H5: Умные предупреждения',
      icon: 'solar:bell-linear',
      summary: 'Своевременные персональные интервенции в момент риска без спама (строго ≤ 2 в неделю).',
      count: roundCounts.H5,
    },
  ];

  return (
    <aside className='w-80 shrink-0 border-r border-default-200 bg-default-50/40 p-4 flex flex-col justify-between hidden md:flex h-[calc(100vh-74px)] sticky top-[74px] overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden'>
      <div className='space-y-5'>
        {/* Project Mission Card */}
        <div className='rounded-2xl bg-primary/10 p-3.5 space-y-1.5'>
          <div className='flex items-center gap-2 text-xs font-bold text-primary'>
            <SvgIcon aria-hidden='true' iconName='solar:shield-check-linear' width={16} />
            <span>Главная цель исследования</span>
          </div>
          <p className='text-xs text-default-600 leading-relaxed'>
            Снижение ложных тревог и точное предсказание срывов в питании и тренировках за счет компактной персональной памяти.
          </p>
        </div>

        {/* Milestones Navigation with Round Counts and Statuses */}
        <div>
          <div className='flex items-center justify-between px-1 mb-2'>
            <span className='text-[11px] font-bold uppercase tracking-wider text-default-400'>
              Задачи исследования (Roadmap)
            </span>
            <span className='text-[10px] text-default-400 font-medium'>H1–H5</span>
          </div>

          <nav className='space-y-2'>
            {milestones.map((item) => {
              const isSelected = selectedTaskFilter === item.id;
              const hasRounds = item.count > 0;
              const isCurrentActive = item.id === activeMilestoneId;

              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTaskFilter(item.id)}
                  className={`w-full p-3 rounded-2xl text-left transition-all ${
                    isSelected
                      ? 'bg-primary text-primary-foreground font-semibold'
                      : 'text-default-700 hover:bg-default-100 hover:text-foreground'
                  }`}
                >
                  {/* 1. Заголовок во всю ширину карточки без обрезания */}
                  <div className='flex items-start justify-between gap-2 w-full'>
                    <span className='text-xs font-bold leading-snug break-words flex-1'>
                      {item.title}
                    </span>
                    {isCurrentActive && (
                      <span className='relative flex h-2 w-2 shrink-0 mt-1' title='Текущая активная цель исследования в реальном времени'>
                        <span className='animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75' />
                        <span className='relative inline-flex rounded-full h-2 w-2 bg-emerald-500' />
                      </span>
                    )}
                  </div>

                  {/* 2. Под заголовком — две колонки: пиктограмма + статус, справа количество в одну строку */}
                  <div className='flex items-center justify-between gap-2 mt-2 pt-2 border-t border-current/10 text-xs w-full'>
                    {/* Колонка 1: Пиктограмма и статус задачи */}
                    <div className='flex items-center gap-1.5 text-[11px] font-medium min-w-0'>
                      <div
                        className={`p-1 rounded-md shrink-0 ${
                          isSelected ? 'bg-primary-foreground/20 text-primary-foreground' : 'bg-default-200/60 text-default-600'
                        }`}
                      >
                        <SvgIcon aria-hidden='true' iconName={item.icon} width={13} />
                      </div>
                      <span className='text-[10px] whitespace-nowrap opacity-90 font-medium'>
                        {item.id === 'ALL'
                          ? 'Roadmap'
                          : isCurrentActive
                          ? 'В работе сейчас'
                          : hasRounds
                          ? 'Завершена'
                          : 'В очереди'}
                      </span>
                    </div>

                    {/* Колонка 2: Количество раундов в ОДНУ СТРОКУ (whitespace-nowrap) */}
                    <div className='shrink-0'>
                      {item.id === 'ALL' ? (
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold whitespace-nowrap ${
                            isSelected ? 'bg-primary-foreground/20 text-primary-foreground' : 'bg-default-200/80 text-default-700'
                          }`}
                        >
                          {item.count} раундов
                        </span>
                      ) : hasRounds ? (
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-bold whitespace-nowrap ${
                            isSelected ? 'bg-primary-foreground/20 text-primary-foreground' : 'bg-default-200/80 text-foreground'
                          }`}
                        >
                          {item.count} {item.count === 1 ? 'раунд' : item.count < 5 ? 'раунда' : 'раундов'}
                        </span>
                      ) : (
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full whitespace-nowrap font-medium ${
                            isSelected ? 'bg-primary-foreground/15 text-primary-foreground/85' : 'bg-default-200/50 text-default-500'
                          }`}
                        >
                          В очереди
                        </span>
                      )}
                    </div>
                  </div>

                  {/* 3. Снизу во всю ширину — описание */}
                  <p
                    className={`text-[11px] leading-relaxed mt-2 w-full ${
                      isSelected ? 'text-primary-foreground/90' : 'text-default-500'
                    }`}
                  >
                    {item.summary}
                  </p>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Governance & Reference Links */}
        <div>
          <span className='text-[11px] font-bold uppercase tracking-wider text-default-400 px-2 block mb-2'>
            Справка и аналитика
          </span>
          <div className='space-y-1 text-xs'>
            {onSelectTab && (
              <>
                <button
                  onClick={() => onSelectTab('metrics')}
                  className='w-full flex items-center gap-2 p-2 rounded-xl text-left bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 font-semibold transition-colors'
                >
                  <SvgIcon aria-hidden='true' iconName='solar:chart-2-bold' width={15} />
                  <span>Сравнение метрик (До vs После)</span>
                </button>
                <button
                  onClick={() => onSelectTab('guide')}
                  className='w-full flex items-center gap-2 p-2 rounded-xl text-left bg-purple-500/10 hover:bg-purple-500/20 text-purple-700 dark:text-purple-300 font-semibold transition-colors'
                >
                  <SvgIcon aria-hidden='true' iconName='solar:layers-linear' width={15} />
                  <span>Иерархия циклов (Справка)</span>
                </button>
              </>
            )}

            <div className='pt-1.5 space-y-1 px-1'>
              <a
                href='https://gov.nutrifit.health/documents/6ad4f7dbf282b7ed30b9a9f7/versions/12'
                target='_blank'
                rel='noopener noreferrer'
                className='flex items-center gap-2 text-primary hover:underline py-0.5'
              >
                <SvgIcon aria-hidden='true' iconName='solar:document-text-linear' width={14} />
                <span>Научный протокол исследования</span>
              </a>
              <a
                href='https://gov.nutrifit.health/documents/6ad4fb8bf282b7ed30b9abe9/versions/10'
                target='_blank'
                rel='noopener noreferrer'
                className='flex items-center gap-2 text-primary hover:underline py-0.5'
              >
                <SvgIcon aria-hidden='true' iconName='solar:document-text-linear' width={14} />
                <span>Регламент валидации (Runbook)</span>
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Footer: Human-friendly Environment Summary & Memory Budget Explanation */}
      <div className='mt-6 pt-3 border-t border-default-200/60 space-y-2 text-xs'>
        <div className='flex items-center justify-between text-default-500'>
          <span>Среда вычислений:</span>
          <span className='font-medium text-foreground'>Изолированная VM</span>
        </div>

        {/* Clear Memory Limit Explanation */}
        <Tooltip>
          <div tabIndex={0} className='flex items-center justify-between text-default-500 cursor-help pb-0.5'>
            <span className='flex items-center gap-1'>
              <span>Память на телефоне:</span>
              <SvgIcon aria-hidden='true' iconName='solar:info-circle-linear' width={12} className='text-default-400' />
            </span>
            <span className='font-semibold text-emerald-600 dark:text-emerald-400'>строго ≤ 16 КиБ</span>
          </div>
        <Tooltip.Content placement='top'>Лимит оперативной памяти на профиль пользователя на смартфоне (Mushroom Body). Позволяет хранить личную историю (например, стаж спорта 6 лет) прямо на телефоне без передачи приватных данных на сервер и с отдельной проверкой энергопотребления. Является жестким критерием приемки алгоритма.</Tooltip.Content></Tooltip>

        <button
          onClick={onOpenEvents}
          className='w-full mt-2.5 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-default-100 hover:bg-default-200 text-default-700 transition-colors text-xs font-medium'
        >
          <SvgIcon aria-hidden='true' iconName='solar:history-linear' width={14} />
          <span>Журнал действий исследователя</span>
        </button>
      </div>
    </aside>
  );
}
