'use client';

import { Button, Chip } from '@heroui/react';
import { SvgIcon, type IconsType } from '@nutrifit/web-ui/SvgIcon';
import type { CravingDashboardState } from '@/types';

interface CravingEventsModalProps {
  isOpen: boolean;
  onClose: () => void;
  events: CravingDashboardState['recentEvents'];
}

export function CravingEventsModal({ isOpen, onClose, events }: CravingEventsModalProps) {
  if (!isOpen) return null;

  // Человекопонятный перевод событий автономного исследователя
  const formatEventHuman = (kind: string, reason?: string): { title: string; description: string; icon: IconsType; color: 'default' | 'success' | 'warning' | 'danger' | 'primary' } => {
    switch (kind) {
      case 'AUTONOMY_BOOT':
        return {
          title: 'Старт автономного контроллера',
          description: 'Инициализация исследовательской кампании, проверка песочницы и очереди гипотез.',
          icon: 'solar:play-circle-linear',
          color: 'default' as const,
        };
      case 'PHASE_TRANSITION':
        return {
          title: 'Смена фазы исследования',
          description: reason || 'Контроллер переключил стадию выполнения цикла (генерация, расчет или критика).',
          icon: 'solar:restart-linear',
          color: 'primary' as const,
        };
      case 'PREREGISTER_START':
      case 'PREREGISTRATION':
        return {
          title: 'Регистрация научной гипотезы',
          description: 'Зафиксированы условия фальсификации, базовые модели и проверяемые конфигурации памяти.',
          icon: 'solar:document-text-bold',
          color: 'primary' as const,
        };
      case 'DISPATCH_SOLVER':
      case 'SOLVER':
        return {
          title: 'Генерация кода модели (Gemini)',
          description: 'Нейросеть сформировала файл гипотезы, код эксперимента и предварительный план проверки.',
          icon: 'solar:cpu-bolt-linear',
          color: 'primary' as const,
        };
      case 'RUN_PROBE':
      case 'RUN_CONTAINER':
      case 'REPLAY':
        return {
          title: 'Изолированные вычисления в песочнице',
          description: 'Программа эксперимента запущена в закрытом контейнере без доступа в интернет.',
          icon: 'solar:code-square-linear',
          color: 'primary' as const,
        };
      case 'CRITIC_EVALUATION':
      case 'CRITIC':
        return {
          title: 'Оценка независимым критиком',
          description: 'Беспристрастная сверка контрольных сумм, проверка ложных тревог и бюджета памяти (до 16 КБ).',
          icon: 'solar:shield-check-linear',
          color: 'success' as const,
        };
      case 'INCONCLUSIVE_RETRY':
      case 'FAILURE':
        return {
          title: 'Автоматическая доработка плана',
          description: reason || 'Зафиксировано превышение лимита шагов или шум. Алгоритм перезапускает раунд с оптимизацией.',
          icon: 'solar:info-circle-linear',
          color: 'warning' as const,
        };
      case 'SUCCESS':
      case 'EPOCH_ADVANCE':
        return {
          title: 'Эксперимент успешно подтвержден',
          description: 'Результаты проверены критиком и приняты. Переход к следующему исследовательскому этапу.',
          icon: 'solar:check-circle-bold',
          color: 'success' as const,
        };
      default:
        return {
          title: kind,
          description: reason || 'Системное действие контроллера автономных исследований.',
          icon: 'solar:history-linear',
          color: 'default' as const,
        };
    }
  };

  return (
    <div className='fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fade-in'>
      <div className='w-full max-w-3xl rounded-3xl bg-background shadow-2xl overflow-hidden flex flex-col max-h-[85vh]'>
        {/* Header */}
        <div className='flex items-center justify-between p-4 border-b border-default-200 bg-default-50/70'>
          <div className='flex items-center gap-2.5'>
            <div className='p-2 rounded-lg bg-primary/10 text-primary'>
              <SvgIcon aria-hidden='true' iconName='solar:history-linear' width={20} />
            </div>
            <div>
              <h3 className='text-sm font-bold text-foreground'>Журнал действий исследователя</h3>
              <p className='text-xs text-default-500'>
                Хронология решений, запусков кода и контрольных проверок автономного агента
              </p>
            </div>
          </div>
          <Button isIconOnly size='sm' variant='ghost' onPress={onClose}>
            <SvgIcon aria-hidden='true' iconName='solar:close-circle-bold' width={18} />
          </Button>
        </div>

        {/* Content: Human-friendly timeline */}
        <div className='flex-1 overflow-y-auto p-4 space-y-3 text-xs'>
          {events.length === 0 ? (
            <p className='text-center text-default-400 py-8'>События пока отсутствуют</p>
          ) : (
            events.map((ev) => {
              const meta = formatEventHuman(ev.kind, ev.reason);

              return (
                <div
                  key={ev.index}
                  className='rounded-2xl bg-default-100/70 p-4 space-y-2 transition-all hover:bg-default-200/60'
                >
                  <div className='flex items-center justify-between gap-2 flex-wrap'>
                    <div className='flex items-center gap-2'>
                      <div className='p-1 rounded bg-default-200/60 text-default-700'>
                        <SvgIcon aria-hidden='true' iconName={meta.icon} width={14} />
                      </div>
                      <span className='font-bold text-foreground text-xs font-sans'>
                        {meta.title}
                      </span>
                      {ev.epoch && (
                        <Chip size='sm' variant='soft' color='accent' className='h-5 text-[10px]'>
                          Эпоха #{ev.epoch}
                        </Chip>
                      )}
                      {ev.round && (
                        <Chip size='sm' variant='soft' color='default' className='h-5 text-[10px]'>
                          Раунд #{ev.round}
                        </Chip>
                      )}
                    </div>
                    <span className='text-[11px] text-default-400 font-mono'>
                      {new Date(ev.at * 1000).toLocaleTimeString('ru-RU', {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}
                    </span>
                  </div>

                  {/* Human-friendly description */}
                  <p className='text-default-600 text-xs leading-relaxed'>
                    {meta.description}
                  </p>

                  {/* Error if present */}
                  {ev.failure && (
                    <div className='rounded-xl bg-amber-500/15 text-amber-800 dark:text-amber-300 p-3 text-xs'>
                      <strong>Причина доработки:</strong> {ev.failure.message}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className='p-3 border-t border-default-200 bg-default-50/50 flex justify-end'>
          <Button size='sm' variant='secondary' onPress={onClose}>
            Закрыть
          </Button>
        </div>
      </div>
    </div>
  );
}
