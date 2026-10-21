'use client';

import { Button } from '@heroui/react';
import type { CravingDashboardState } from '@/types';

interface CravingStageTransitionGateProps {
  targetStage: string;
  state?: CravingDashboardState | null;
  onSelectActiveRound?: () => void;
}

export function CravingStageTransitionGate({ targetStage, state, onSelectActiveRound }: CravingStageTransitionGateProps) {
  const predecessor = { H3: 'H2', H4: 'H3', H5: 'H4' }[targetStage as 'H3' | 'H4' | 'H5'];
  const criteria = predecessor ? Object.entries(state?.goal?.criteria?.[predecessor] || {}) : [];
  const activeStage = state?.goal?.tasks?.[state.taskIndex]?.id;
  return (
    <div className='w-full min-w-0 p-4 md:p-6 space-y-6 text-foreground'>
      <div className='rounded-2xl bg-default-100/70 p-5 space-y-3'>
        <h2 className='text-lg font-bold'>Переход к этапу {targetStage}</h2>
        <p className='text-sm text-default-600'>
          Следующий этап открывается после воспроизведения расчёта, проверки измерений и независимой оценки требований предыдущего этапа.
          Завершение программы и размер файла памяти сами по себе не дают научного допуска.
        </p>
        <p className='text-xs text-default-500'>
          Текущий этап: {activeStage || 'неизвестен'} · фаза: {state?.phase || 'неизвестна'} · состояние: {state?.status || 'неизвестно'}.
          Эпоха {state?.epoch ?? '—'}, раунд {state?.round ?? '—'}.
        </p>
        {onSelectActiveRound && <Button variant='secondary' size='sm' onPress={onSelectActiveRound}>Открыть активный раунд</Button>}
      </div>
      <div className='space-y-3'>
        <h3 className='text-sm font-semibold'>Зафиксированные требования {predecessor || 'предыдущего этапа'}</h3>
        <p className='text-xs text-default-500'>Это требования из сохранённой цели, а не отметки об их выполнении. Результат оценки находится в machine-admission и отчёте критика.</p>
        {criteria.length ? criteria.map(([id, requirement]) => (
          <div key={id} className='rounded-xl bg-default-100/70 p-4 text-sm'>
            <span className='text-xs text-default-500'>{id}</span>
            <p className='mt-1'>{requirement}</p>
          </div>
        )) : <p className='text-sm text-default-500'>Зафиксированные требования пока недоступны.</p>}
      </div>
    </div>
  );
}
