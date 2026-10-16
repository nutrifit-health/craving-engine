"""Причинные ограничения и описательная диагностика синтетического корпуса."""

from datetime import timedelta

import numpy as np

from synthetic_data.types import TrueDailyState, UserHealthContext

DATASET_VERSION = "synthetic-causal-v2"


def context_issues(context: UserHealthContext) -> list[str]:
    """Проверяет данные источника до их свёртки в признаки модели."""
    issues = []
    at = context.evaluationTime
    for record in [
        *context.meals,
        *context.workouts,
        context.sleep,
        context.bodyMeasurement,
        context.wellbeing,
    ]:
        if record is None:
            continue
        if record.availableAt is None or record.availableAt > at:
            issues.append("Запись не имеет подтверждённой доступности до решения")
    for meal in context.meals:
        if meal.eatenAt > at or (
            meal.availableAt is not None and meal.availableAt < meal.eatenAt
        ):
            issues.append("Еда находится в будущем или опубликована до события")
        if (
            meal.sugars > meal.carbs + 1e-8
            or (meal.addedSugar or 0) > meal.sugars + 1e-8
        ):
            issues.append("Нарушена вложенность сахаров и углеводов")
        expected = 4 * meal.proteins + 9 * meal.fats + 4 * meal.carbs + 7 * meal.alcohol
        if abs(meal.calories - expected) > 1.0:
            issues.append("Калории не согласованы с моделью макронутриентов симулятора")
    for workout in context.workouts:
        end = workout.startedAt + timedelta(minutes=workout.durationMinutes)
        if end > at or (workout.availableAt is not None and workout.availableAt < end):
            issues.append("Данные завершённой тренировки поступили до её завершения")
    if context.sleep is not None:
        sleep = context.sleep
        if sleep.wakeTime > at or (
            sleep.availableAt is not None and sleep.availableAt < sleep.wakeTime
        ):
            issues.append("Сон ещё не завершился")
        hours = (sleep.wakeTime - sleep.bedtime).total_seconds() / 3600
        if abs(hours - sleep.sleepHours) > 0.05:
            issues.append("Продолжительность сна расходится с временными отметками")
    if context.bodyMeasurement is not None and context.bodyMeasurement.measuredAt > at:
        issues.append("Замер тела находится в будущем")
    if context.wellbeing is not None and context.wellbeing.recordedAt > at:
        issues.append("Чек-ин находится в будущем")
    return issues


def summarize_dataset(
    histories: list[list[tuple[TrueDailyState, UserHealthContext]]],
) -> dict[str, object]:
    """Не выдаёт инженерную согласованность за медицинскую реалистичность."""
    rows = [row for history in histories for row in history]
    failures = [
        f"{context.userId}/{truth.day_index}: {issue}"
        for truth, context in rows
        for issue in context_issues(context)
    ]
    if failures:
        raise ValueError("Некачественные данные: " + "; ".join(failures[:8]))
    identities = [(context.userId, context.evaluationTime) for _, context in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Повторные пользователь/момент решения")
    per_user = []
    for history in histories:
        if not history:
            raise ValueError("Пустая траектория")
        labels = [int(truth.relapse_occurred) for truth, _ in history]
        probabilities = [truth.true_relapse_prob for truth, _ in history]
        per_user.append(
            {
                "user_id": history[0][1].userId,
                "days": len(history),
                "event_rate": float(np.mean(labels)),
                "expected_event_rate": float(np.mean(probabilities)),
                "bernoulli_variance": float(
                    np.mean([p * (1 - p) for p in probabilities])
                ),
                "meal_coverage": sum(bool(context.meals) for _, context in history)
                / len(history),
                "sleep_coverage": sum(
                    context.sleep is not None for _, context in history
                )
                / len(history),
                "wellbeing_coverage": sum(
                    context.wellbeing is not None for _, context in history
                )
                / len(history),
            }
        )
    return {
        "dataset_version": DATASET_VERSION,
        "users": len(histories),
        "rows": len(rows),
        "context_violations": 0,
        "per_user": per_user,
        "qualification": "Сценарный инженерный корпус, без доказанной клинической репрезентативности",
    }
