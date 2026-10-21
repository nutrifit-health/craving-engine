'use client';

import { SvgIcon } from '@nutrifit/web-ui/SvgIcon';

export function CravingHierarchyGuide() {
  const hierarchyLevels = [
    {
      level: 1,
      name: 'Этап исследования (Milestone / Task)',
      badge: 'Уровень 1 • Стратегический',
      color: 'text-purple-700 dark:text-purple-300',
      accentColor: 'text-purple-600 dark:text-purple-400',
      bg: 'bg-purple-500/10 dark:bg-purple-950/20',
      numberBg: 'bg-purple-600 text-white',
      badgeStyle: 'bg-purple-500/20 text-purple-700 dark:text-purple-300',
      question: 'Что решаем глобально?',
      description:
        'Верхнеуровневый этап научного проекта NutriFit. Проект состоит из 5 последовательных этапов (H1–H5). Переход на следующий этап происходит только тогда, когда независимый критик подтвердил достижение целевых метрик текущего этапа.',
      cycleRelation: 'Содержит в себе серию Эпох (Epochs). Выполняется строго последовательно.',
      example: 'Сейчас активен этап H2: "Фундамент и симуляторы". Следующие этапы (H3, H4, H5) стоят в очереди.',
    },
    {
      level: 2,
      name: 'Эпоха (Epoch)',
      badge: 'Уровень 2 • Тактический',
      color: 'text-blue-700 dark:text-blue-300',
      accentColor: 'text-blue-600 dark:text-blue-400',
      bg: 'bg-blue-500/10 dark:bg-blue-950/20',
      numberBg: 'bg-blue-600 text-white',
      badgeStyle: 'bg-blue-500/20 text-blue-700 dark:text-blue-300',
      question: 'Какой научный подход проверяем?',
      description:
        'Глобальная итерация научного поиска. Когда нейросеть Gemini кардинально меняет вектор исследования (например, переходит от базовых статистических моделей к биологической памяти Mushroom Body или меняет стратегию отбора признаков), она начинает новую Эпоху.',
      cycleRelation: 'Выполняется внутри Этапа. Содержит в себе серию Раундов (Rounds).',
      example: 'Эпоха #6: глобальный цикл тестирования скалярной памяти в различных синтетических мирах.',
    },
    {
      level: 3,
      name: 'Раунд (Round)',
      badge: 'Уровень 3 • Законченный эксперимент',
      color: 'text-emerald-700 dark:text-emerald-300',
      accentColor: 'text-emerald-600 dark:text-emerald-400',
      bg: 'bg-emerald-500/10 dark:bg-emerald-950/20',
      numberBg: 'bg-emerald-600 text-white',
      badgeStyle: 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300',
      question: 'Какую конкретную гипотезу проверяем прямо сейчас?',
      description:
        'Главный рабочий такт исследования. Ровно в рамках одного раунда формулируется проверяемая гипотеза (hypothesis.json), пишется код эксперимента (experiment.py), производятся вычисления и формируется финальный отчет раунда (report.md).',
      cycleRelation: 'Выполняется внутри Эпохи. Содержит в себе цепочку Шагов AI (AI Steps) и Запусков кода.',
      example: 'Раунд #1 в Эпохе 6: проверка устойчивости порога срабатывания памяти при дрейфе привычек.',
    },
    {
      level: 4,
      name: 'Ход рассуждений AI (AI Steps & Actions)',
      badge: 'Уровень 4 • Мысли и действия модели',
      color: 'text-amber-800 dark:text-amber-200',
      accentColor: 'text-amber-600 dark:text-amber-400',
      bg: 'bg-amber-500/10 dark:bg-amber-950/20',
      numberBg: 'bg-amber-600 text-white',
      badgeStyle: 'bg-amber-500/20 text-amber-800 dark:text-amber-200',
      question: 'Что нейросеть делает в каждую секунду?',
      description:
        'Атомарные шаги модели Gemini внутри раунда: глубокое размышление (CoT reasoning / thinking), чтение протокола, написание кода скрипта, запуск системных утилит и анализ логов.',
      cycleRelation: 'Происходят последовательно внутри Раунда. Число действий отображается в счетчике.',
      example: 'Действие #169: размышление над формулой Brier score и написание итогового отчета раунда.',
    },
    {
      level: 5,
      name: 'Запуск кода (Process Attempts / Executions)',
      badge: 'Уровень 5 • Вычисления в песочнице',
      color: 'text-cyan-800 dark:text-cyan-200',
      accentColor: 'text-cyan-600 dark:text-cyan-400',
      bg: 'bg-cyan-500/10 dark:bg-cyan-950/20',
      numberBg: 'bg-cyan-600 text-white',
      badgeStyle: 'bg-cyan-500/20 text-cyan-800 dark:text-cyan-200',
      question: 'Как исполняется программа?',
      description:
        'Попытки включают модельные сессии и контейнерные расчёты. Размер синтетической выборки задаётся конкретным опытом и читается из его измерений.',
      cycleRelation: 'Вызывается моделью как инструмент внутри раунда. В одном раунде может быть несколько запусков.',
      example: 'Количество запусков и число решений являются разными величинами.',
    },
    {
      level: 6,
      name: 'Коррекции и самоисцеление (Repairs / Auto-heal)',
      badge: 'Уровень 6 • Защитный контур',
      color: 'text-rose-800 dark:text-rose-200',
      accentColor: 'text-rose-600 dark:text-rose-400',
      bg: 'bg-rose-500/10 dark:bg-rose-950/20',
      numberBg: 'bg-rose-600 text-white',
      badgeStyle: 'bg-rose-500/20 text-rose-800 dark:text-rose-200',
      question: 'Что происходит при ошибках?',
      description:
        'Автономный контур самоисцеления. Если код завершился с ошибкой, возник таймаут или критик отклонил результат, система не зависает и не беспокоит человека. Ошибка фиксируется в счетчике «Коррекции», передается обратно модели, и модель автоматически перестраивает код в следующем раунде.',
      cycleRelation: 'Срабатывает автоматически при обнаружении дефектов, замыкая цикл обучения.',
      example: '51 коррекция — это 51 успешное преодоление синтаксических сбоев и отклонений критика без падения системы.',
    },
  ];

  const quickTable = [
    { name: 'Этап (Milestone H1..H5)', scope: 'Проект целиком', who: 'NutriFit научный совет', trigger: 'Подтверждение критика' },
    { name: 'Эпоха (Epoch)', scope: 'Внутри Этапа', who: 'Автономный контроллер', trigger: 'Смена научной парадигмы' },
    { name: 'Раунд (Round)', scope: 'Внутри Эпохи', who: 'Модель Gemini', trigger: 'Новая проверяемая гипотеза' },
    { name: 'Шаги AI (Steps)', scope: 'Внутри Раунда', who: 'Мышление модели CoT', trigger: 'Атомарное действие' },
    { name: 'Запуск кода', scope: 'Внутри Шага', who: 'Изолированная Lima VM', trigger: 'Инструмент запуска тестов' },
    { name: 'Коррекция (Repair)', scope: 'Межраундовый контур', who: 'Auto-heal супервизор', trigger: 'Сбой или замечание критика' },
  ];

  const faqs = [
    {
      q: 'Почему время работы в шапке тикает (например, 4ч 44м), а номер раунда не увеличивается?',
      a: 'Раунд включает генерацию кода, улучшение инструкций, ограниченные расчёты и оценку результатов. Время модельных сессий и время OCI-вычислений учитываются отдельно; длительная сессия не означает много часов численных расчётов.',
    },
    {
      q: 'Что означает счетчик «Действий AI: 169»?',
      a: 'Это количество атомарных шагов нейросети внутри активного раунда: сколько раз модель обратилась к файлам, сколько шагов размышляла над структурой памяти и сколько раз запускала тесты. Накопленный счётчик не подтверждает, что процесс активен сейчас.',
    },
    {
      q: 'Что означает число «Коррекций: 51» — это плохо или хорошо?',
      a: 'Это счётчик зарегистрированных ошибок и повторов, а не доказательство их устранения. Повторяющаяся причина требует анализа; смотрите последнюю ошибку, срок повтора и текущий статус.',
    },
    {
      q: 'Где увидеть, как исследование улучшило предсказания?',
      a: 'На вкладке измерений показан выбранный завершённый расчёт. Проверяйте выборку, macro-метрики и интервалы; отсутствие данных обозначается прочерком. Гейт памяти не равен уведомлению.',
    },
  ];

  return (
    <div className='space-y-6 max-w-5xl pb-10 text-foreground'>
      {/* Header Banner */}
      <div className='rounded-2xl bg-gradient-to-r from-purple-500/15 via-primary/5 to-cyan-500/10 p-6 space-y-3'>
        <div className='flex items-center gap-3'>
          <div className='flex h-10 w-10 items-center justify-center rounded-xl bg-purple-600 text-white'>
            <SvgIcon aria-hidden='true' iconName='solar:layers-linear' width={22} />
          </div>
          <div>
            <h2 className='text-base font-bold text-foreground'>
              Иерархия циклов и устройство Craving Engine
            </h2>
            <p className='text-xs text-default-500'>
              Наглядное объяснение вложенности процессов: кто над кем главнее и в каком порядке выполняются этапы
            </p>
          </div>
        </div>

        <p className='text-xs text-default-600 dark:text-default-400 leading-relaxed max-w-3xl pt-1'>
          Автономная исследовательская система NutriFit работает по строгой многоуровневой схеме. Ниже визуализирована вложенность всех циклов — от стратегических этапов проекта до атомарных запусков кода и циклов самоисцеления.
        </p>
      </div>

      {/* ВИЗУАЛИЗАЦИЯ ВЛОЖЕННОСТИ: ВЛОЖЕННЫЕ БЛОКИ (МАТРЕШКА) БЕЗ БОРДЮРОВ */}
      <div className='space-y-3'>
        <div className='flex items-center gap-2 px-1'>
          <SvgIcon aria-hidden='true' iconName='solar:tuning-4-bold' width={18} className='text-primary' />
          <h3 className='text-xs font-bold uppercase tracking-wider text-default-600'>
            Визуальная вложенность циклов (Кто в чем находится)
          </h3>
        </div>

        {/* Наглядная интерактивная каскадная схема */}
        <div className='rounded-3xl bg-purple-500/10 p-5 space-y-4'>
          {/* Level 1: Milestone */}
          <div className='flex items-center justify-between flex-wrap gap-2'>
            <div className='flex items-center gap-2'>
              <span className='px-2.5 py-1 rounded-lg bg-purple-500 text-white font-bold text-xs uppercase tracking-wider'>
                1. Этап (H1..H5)
              </span>
              <span className='text-xs font-bold text-foreground'>Глобальная стратегическая цель проекта</span>
            </div>
            <span className='text-[11px] text-purple-700 dark:text-purple-300 font-medium'>
              Внутри этапа выполняются Эпохи
            </span>
          </div>

          {/* Level 2: Epoch */}
          <div className='rounded-2xl bg-blue-500/10 p-4 space-y-3'>
            <div className='flex items-center justify-between flex-wrap gap-2'>
              <div className='flex items-center gap-2'>
                <span className='px-2 py-0.5 rounded-lg bg-blue-600 text-white font-bold text-xs uppercase tracking-wider'>
                  2. Эпоха (Epoch)
                </span>
                <span className='text-xs font-semibold text-foreground'>Глобальная научная итерация поиска</span>
              </div>
              <span className='text-[11px] text-blue-700 dark:text-blue-300 font-medium'>
                Внутри эпохи проводятся Раунды
              </span>
            </div>

            {/* Level 3: Round */}
            <div className='rounded-xl bg-emerald-500/10 p-4 space-y-3'>
              <div className='flex items-center justify-between flex-wrap gap-2'>
                <div className='flex items-center gap-2'>
                  <span className='px-2 py-0.5 rounded-lg bg-emerald-600 text-white font-bold text-xs uppercase tracking-wider'>
                    3. Раунд (Round)
                  </span>
                  <span className='text-xs font-semibold text-foreground'>Законченный эксперимент (Гипотеза → Код → Отчет)</span>
                </div>
                <span className='text-[11px] text-emerald-700 dark:text-emerald-300 font-medium'>
                  Внутри раунда идут Шаги AI
                </span>
              </div>

              {/* Level 4: AI Steps */}
              <div className='rounded-lg bg-amber-500/15 p-3.5 space-y-2.5'>
                <div className='flex items-center justify-between flex-wrap gap-2'>
                  <div className='flex items-center gap-2'>
                    <span className='px-2 py-0.5 rounded-md bg-amber-600 text-white font-bold text-[11px] uppercase tracking-wider'>
                      4. Шаги AI (Steps)
                    </span>
                    <span className='text-xs font-medium text-foreground'>Размышления (CoT), правка файлов, вызов команд</span>
                  </div>
                  <span className='text-[10px] text-amber-800 dark:text-amber-200 font-medium'>
                    Инициируют запуски кода в контейнере
                  </span>
                </div>

                {/* Level 5 & 6 side by side */}
                <div className='grid grid-cols-1 md:grid-cols-2 gap-2.5 pt-1'>
                  <div className='rounded-lg bg-cyan-500/20 p-2.5 text-xs space-y-1'>
                    <div className='font-bold text-cyan-800 dark:text-cyan-200 flex items-center gap-1.5'>
                      <SvgIcon aria-hidden='true' iconName='solar:code-linear' width={14} />
                      <span>5. Запуск кода (Lima VM)</span>
                    </div>
                    <p className='text-[11px] text-cyan-900 dark:text-cyan-300 leading-snug'>
                      Вычисления изолированного Python-скрипта на тысячах профилей.
                    </p>
                  </div>

                  <div className='rounded-lg bg-rose-500/20 p-2.5 text-xs space-y-1'>
                    <div className='font-bold text-rose-800 dark:text-rose-200 flex items-center gap-1.5'>
                      <SvgIcon aria-hidden='true' iconName='solar:restart-linear' width={14} />
                      <span>6. Коррекции (Self-healing)</span>
                    </div>
                    <p className='text-[11px] text-rose-900 dark:text-rose-300 leading-snug'>
                      Авто-исправление ошибок без падения и остановки таймера.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ПОДРОБНЫЕ КАРТОЧКИ ПО КАЖДОМУ УРОВНЮ */}
      <div className='space-y-3 pt-2'>
        <div className='flex items-center gap-2 px-1'>
          <SvgIcon aria-hidden='true' iconName='solar:book-2-bold' width={18} className='text-primary' />
          <h3 className='text-xs font-bold uppercase tracking-wider text-default-600'>
            Подробное описание каждого уровня и понятные примеры
          </h3>
        </div>

        <div className='space-y-3.5'>
          {hierarchyLevels.map((lvl) => (
            <div key={lvl.level} className={`rounded-2xl ${lvl.bg} p-5 space-y-3`}>
              <div className='flex flex-wrap items-center justify-between gap-2'>
                <div className='flex items-center gap-2.5'>
                  {/* Номерной бейдж в цвет схемы выше: решает проблему белой точки */}
                  <span className={`flex h-7 w-7 items-center justify-center rounded-lg ${lvl.numberBg} font-bold text-xs shrink-0 shadow-xs`}>
                    {lvl.level}
                  </span>
                  <div>
                    <h4 className='text-sm font-bold text-foreground'>{lvl.name}</h4>
                    <span className='text-[11px] text-default-500 font-medium'>{lvl.question}</span>
                  </div>
                </div>

                <span className={`text-[10px] font-semibold px-2.5 py-1 rounded-md whitespace-nowrap ${lvl.badgeStyle}`}>
                  {lvl.badge}
                </span>
              </div>

              <p className='text-xs text-default-700 dark:text-default-300 leading-relaxed'>
                {lvl.description}
              </p>

              <div className='grid grid-cols-1 md:grid-cols-2 gap-3 pt-1 text-xs'>
                <div className='rounded-xl bg-background/70 p-3 space-y-1'>
                  <span className='text-[10px] font-bold uppercase tracking-wider text-default-400'>
                    Связь с другими циклами:
                  </span>
                  <p className='text-[11px] text-default-600 dark:text-default-400 leading-snug'>
                    {lvl.cycleRelation}
                  </p>
                </div>

                <div className='rounded-xl bg-background/70 p-3 space-y-1'>
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${lvl.accentColor}`}>
                    Пример на текущем экране:
                  </span>
                  <p className='text-[11px] text-foreground font-medium leading-snug'>
                    {lvl.example}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* СВОДНАЯ ТАБЛИЦА ШПАРГАЛКА */}
      <div className='space-y-3 pt-2'>
        <div className='flex items-center gap-2 px-1'>
          <SvgIcon aria-hidden='true' iconName='solar:list-bold' width={18} className='text-primary' />
          <h3 className='text-xs font-bold uppercase tracking-wider text-default-600'>
            Шпаргалка: Быстрое сопоставление терминов
          </h3>
        </div>

        <div className='rounded-2xl bg-default-100/70 overflow-hidden'>
          <div className='overflow-x-auto'>
            <table className='w-full text-left text-xs'>
              <thead className='bg-default-200/50 text-[11px] uppercase tracking-wider text-default-600 font-semibold'>
                <tr>
                  <th className='py-3 px-4'>Термин</th>
                  <th className='py-3 px-3'>Масштаб</th>
                  <th className='py-3 px-3'>Кто выполняет</th>
                  <th className='py-3 px-4'>Триггер перехода</th>
                </tr>
              </thead>
              <tbody className='divide-y divide-default-200/40 text-foreground'>
                {quickTable.map((row, idx) => (
                  <tr key={idx} className='hover:bg-default-200/30 transition-colors'>
                    <td className='py-3 px-4 font-bold text-foreground'>{row.name}</td>
                    <td className='py-3 px-3 text-default-500'>{row.scope}</td>
                    <td className='py-3 px-3 font-medium'>{row.who}</td>
                    <td className='py-3 px-4 text-default-600'>{row.trigger}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* FAQ: ОТВЕТЫ НА ЧАСТЫЕ ВОПРОСЫ */}
      <div className='space-y-3 pt-2'>
        <div className='flex items-center gap-2 px-1'>
          <SvgIcon aria-hidden='true' iconName='solar:info-circle-bold' width={18} className='text-primary' />
          <h3 className='text-xs font-bold uppercase tracking-wider text-default-600'>
            Часто задаваемые вопросы (FAQ)
          </h3>
        </div>

        <div className='grid grid-cols-1 md:grid-cols-2 gap-3.5'>
          {faqs.map((faq, fIdx) => (
            <div key={fIdx} className='rounded-2xl bg-default-100/70 p-4 space-y-2'>
              <h4 className='text-xs font-bold text-foreground leading-snug flex items-start gap-1.5'>
                <span className='text-primary font-mono font-bold'>Q:</span>
                <span>{faq.q}</span>
              </h4>
              <p className='text-xs text-default-600 dark:text-default-400 leading-relaxed pl-4'>
                {faq.a}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
