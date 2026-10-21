/**
 * Вспомогательный модуль для перевода сложной технической терминологии
 * Craving Engine в ясный и понятный язык для руководителя и пользователей.
 */

export interface HumanReportSummary {
  status: 'passed' | 'failed' | 'in_progress' | 'planned';
  statusLabel: string;
  statusColor: 'success' | 'danger' | 'warning' | 'primary';
  date: string;
  taskId: string;
  taskTitle: string;
  roundNum: string;
  keyTakeaway: string; // Главный вывод простым языком
  practicalImplication: string; // Что это значит для продукта и пользователя
  environmentInfo: string;
}

export function extractReportSummary(
  reportMd: string | undefined | null,
  roundStatus: string | undefined,
  taskId: string,
  roundNum: string,
  error?: { type: string; message: string }
): HumanReportSummary {
  const isFailed = roundStatus === 'failed' || roundStatus === 'base-failed' || Boolean(error);
  const isPassed = roundStatus === 'passed';
  const isRunning = roundStatus === 'running';

  // Извлечение даты из текста отчета (Date: YYYY-MM-DD)
  let date = 'Дата не указана';
  if (reportMd) {
    const dateMatch = reportMd.match(/\*\*Date\*\*:\s*([^\n\r]+)/i);
    if (dateMatch?.[1]) {
      date = dateMatch[1].trim();
    }
  }

  // Определение названия задачи
  const taskTitles: Record<string, string> = {
    H2: 'Научные предпосылки и симуляторы поведения',
    H3: 'Анализ факторов риска и информативность признаков',
    H4: 'Персональная память человека (до 16 КБ, Mushroom Body)',
    H5: 'Умная политика персональных предупреждений',
  };
  const taskTitle = taskTitles[taskId] || `Исследовательский этап ${taskId}`;

  if (isFailed) {
    let explanation = 'В этом раунде не получен полный проверяемый результат. Причина указана в журнале.';
    if (error?.message) {
      if (error.message.includes('step limit')) {
        explanation = 'Модель превысила лимит шагов в одной сессии. Автономный контроллер автоматически перезапускает попытку с оптимизацией плана.';
      } else if (error.message.includes('timeout')) {
        explanation = 'Превышено время ожидания ответа вычислительной среды. Запланирован автоматический повтор.';
      } else {
        explanation = `Сбой на этапе проверки: ${error.message}. Система самовосстановления ищет корректное решение.`;
      }
    }

    return {
      status: 'failed',
      statusLabel: 'Требуется доработка (Inconclusive)',
      statusColor: 'danger',
      date,
      taskId,
      taskTitle,
      roundNum,
      keyTakeaway: explanation,
      practicalImplication: 'Результаты этого раунда не приняты в основную модель. Автономный агент вносит исправления в алгоритм и готовит следующий эксперимент.',
      environmentInfo: 'Изолированный контейнер без доступа в интернет',
    };
  }

  if (isRunning) {
    return {
      status: 'in_progress',
      statusLabel: 'Эксперимент выполняется прямо сейчас',
      statusColor: 'primary',
      date,
      taskId,
      taskTitle,
      roundNum,
      keyTakeaway: 'Нейросеть генерирует и верифицирует программный код эксперимента в изолированной среде.',
      practicalImplication: 'После расчётов выполняются независимое воспроизведение и оценка критериев текущего этапа.',
      environmentInfo: 'Безопасная изолированная песочница',
    };
  }

  if (isPassed) {
    // Успешный раунд
    return {
      status: 'passed',
      statusLabel: 'Вычисление завершено',
      statusColor: 'success',
      date,
      taskId,
      taskTitle,
      roundNum,
      keyTakeaway: 'Программа завершилась и сохранила результаты синтетического эксперимента. Научный допуск этапа требует отдельной записи приёмки и воспроизведения.',
      practicalImplication: 'Успех вычисления не подтверждает преимущество модели, пользу для людей или готовность к внедрению. Проверяйте macro-метрики, интервалы и сравниваемую базовую модель в измерениях.',
      environmentInfo: 'Автономная среда Gemini',
    };
  }

  // not_executed / запланирован
  return {
    status: 'planned',
    statusLabel: 'План эксперимента зарегистрирован',
    statusColor: 'warning',
    date,
    taskId,
    taskTitle,
    roundNum,
    keyTakeaway:
      taskId === 'H4'
        ? 'Сформирован план проверки компактной памяти (архитектура Mushroom Body). Цель — учитывать личную историю человека (стаж спорта, срывы в прошлом), не раздувая размер приложения (строго ≤ 16 КБ).'
        : 'Сформирована гипотеза для проверки факторов риска. Алгоритм готовится к запуску изолированных вычислений.',
    practicalImplication: 'Код подготовлен к изолированному запуску. Измерения будут проведены независимо от генерации текста модели.',
    environmentInfo: 'Среда готова к запуску',
  };
}

export interface HumanHypothesisSummary {
  goal: string;
  whyItMatters: string;
  safetyCheck: string;
  memoryConstraint: string;
  targetMetrics: string;
}

export function parseHypothesisHuman(hypothesisRaw: string | Record<string, unknown> | null | undefined): HumanHypothesisSummary {
  if (!hypothesisRaw) {
    return {
      goal: 'Гипотеза еще не сформирована для выбранного раунда.',
      whyItMatters: 'Ожидается запуск автономного исследователя.',
      safetyCheck: 'Автоматическая проверка ложных тревог',
      memoryConstraint: 'Лимит до 16 КиБ',
      targetMetrics: 'Снижение числа ложных тревог',
    };
  }

  let data: Record<string, unknown> = {};
  if (typeof hypothesisRaw === 'string') {
    try {
      data = JSON.parse(hypothesisRaw);
    } catch {
      data = { hypothesis: hypothesisRaw };
    }
  } else {
    data = hypothesisRaw;
  }

  const hypothesisText = String(data.hypothesis || '');
  const falsificationText = String(data.falsification || '');

  // Человекопонятная цель
  let goal = 'Проверить, позволяет ли персональная адаптация точнее предсказывать пищевые срывы и пропуски тренировок.';
  if (hypothesisText.toLowerCase().includes('mushroom') || hypothesisText.toLowerCase().includes('ridge') || hypothesisText.toLowerCase().includes('memory')) {
    goal = 'Оценить компактную персональную память: может ли модель учитывать индивидуальные особенности человека (например, 6 лет стажа vs новичок) для более точного предупреждения срывов.';
  } else if (hypothesisText.toLowerCase().includes('missingness') || hypothesisText.toLowerCase().includes('discard')) {
    goal = 'Научить модель отличать забывчивость пользователя (пропустил лог) от реального срыва питания, чтобы избежать ложных тревог.';
  }

  // Почему это важно для бизнеса и пользователей
  const whyItMatters =
    'Исследование проверяет, даёт ли персональная память пользу сверх простой базовой модели. Польза для реальных пользователей пока не установлена.';

  // Защитный механизм
  let safetyCheck = 'Защитный гейт: если персональные данные создают шум, модель автоматически возвращается к надежной базовой оценке.';
  if (falsificationText.includes('safety') || falsificationText.includes('gate') || falsificationText.includes('null-signal')) {
    safetyCheck = 'Защитный барьер: в ситуациях, когда у пользователя нет индивидуальных аномалий, алгоритм отключает адаптацию, чтобы не поддаваться случайным шумам.';
  }

  return {
    goal,
    whyItMatters,
    safetyCheck,
    memoryConstraint: 'Строго до 16 КиБ на пользователя (работает локально на смартфоне без передачи на сервер)',
    targetMetrics: 'Критерии текущего этапа из зафиксированного плана; Brier, macro AUC и неопределённость при сравнении моделей',
  };
}
