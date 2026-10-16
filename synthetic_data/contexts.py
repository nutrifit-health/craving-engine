"""Причинное преобразование синтетической истории в наблюдаемый контекст.

Не подключается к серверу и базе данных; скрытые состояния не читает.
"""

import math
from datetime import UTC, date, datetime, timedelta

from synthetic_data.types import (
    BodyMeasurementRecord,
    MealRecord,
    SleepRecord,
    TrueDailyState,
    UserHealthContext,
    WellbeingRecord,
    WorkoutRecord,
)

CONTEXT_WINDOWS = {
    "mealsHours": 48,
    "workoutsHours": 24,
    "wellbeingHours": 36,
    "bodyDays": 30,
    "engagementDays": 30,
}


def _utc(value: datetime) -> datetime:
    """Наивное время корпуса обозначает условный UTC, без вывода географии."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _available(
    available_at: datetime | None,
    event_at: datetime,
    now: datetime,
    excluded: dict[str, int],
) -> bool:
    if available_at is None:
        excluded["records_without_available_at"] += 1
        return False
    available = _utc(available_at)
    event = _utc(event_at)
    if available < event:
        raise ValueError("Запись объявлена доступной до завершения события")
    if available > now or event > now:
        excluded["not_yet_available_records"] += 1
        return False
    return True


def _engagement(
    event_times: list[datetime],
    now: datetime,
    checkin_today: bool,
    meals_today: int,
    water_ml: float,
) -> dict[str, object]:
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    log_days = {event.date() for event in event_times}
    cursor = now.date() if now.date() in log_days else now.date() - timedelta(days=1)
    streak = 0
    for _ in range(CONTEXT_WINDOWS["engagementDays"]):
        if cursor not in log_days:
            break
        streak += 1
        cursor -= timedelta(days=1)
    latest = max(event_times, default=None)
    days_since = (now.date() - latest.date()).days if latest is not None else None
    completeness = min(
        1.0,
        (0.34 if checkin_today else 0.0)
        + min(meals_today, 3) / 3 * 0.5
        + (0.16 if water_ml > 0 else 0.0),
    )
    if days_since is None:
        dropoff = 0.5
    else:
        dropoff = (
            0.05
            if days_since <= 0
            else 0.25
            if days_since == 1
            else 0.45
            if days_since == 2
            else 0.7
        )
    if streak >= 7:
        dropoff -= 0.15
    elif streak >= 3:
        dropoff -= 0.05
    hour = (now - day_start).total_seconds() / 3600
    if hour >= 18 and completeness < 0.34:
        dropoff += 0.25
    elif hour >= 12 and completeness == 0:
        dropoff += 0.1
    dropoff = min(1.0, max(0.0, math.floor(dropoff * 100 + 0.5) / 100))
    risk_level = (
        "critical"
        if dropoff >= 0.7
        else "high"
        if dropoff >= 0.45
        else "medium"
        if dropoff >= 0.25
        else "low"
    )
    return {
        "loggingStreakDays": streak,
        "daysSinceLastLog": days_since,
        "todayCompleteness": math.floor(completeness * 100 + 0.5) / 100,
        "dropoffRisk": dropoff,
        "riskLevel": risk_level,
    }


def history_contexts(
    history: list[tuple[TrueDailyState, UserHealthContext]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    meals: dict[tuple[datetime, str], MealRecord] = {}
    workouts: dict[tuple[datetime, str], WorkoutRecord] = {}
    sleeps: dict[datetime, SleepRecord] = {}
    checkins: dict[datetime, WellbeingRecord] = {}
    bodies: dict[datetime, BodyMeasurementRecord] = {}
    water_snapshots: dict[date, tuple[datetime, float]] = {}
    contexts: list[dict[str, object]] = []
    excluded = {
        "records_without_available_at": 0,
        "not_yet_available_records": 0,
        "habits_without_current_day": 0,
    }
    previous_at: datetime | None = None
    expected_user = history[0][1].userId if history else None

    for _, observed in history:
        now = _utc(observed.evaluationTime)
        if observed.userId != expected_user or (
            previous_at is not None and now <= previous_at
        ):
            raise ValueError(
                "История должна содержать одного пользователя и строго возрастающие моменты решения"
            )
        previous_at = now
        for item in observed.meals:
            if _available(item.availableAt, item.eatenAt, now, excluded):
                meals[(_utc(item.eatenAt), item.mealType.value)] = item
        for item in observed.workouts:
            end = item.startedAt + timedelta(minutes=item.durationMinutes)
            if _available(item.availableAt, end, now, excluded):
                if not 0 <= item.fatigueScore <= 10:
                    raise ValueError(
                        "fatigueScore должен иметь явную шкалу 0..10 до вызова production-моста"
                    )
                workouts[(_utc(item.startedAt), item.sportKey)] = item
        for item in [
            *observed.recentSleepHistory,
            *([observed.sleep] if observed.sleep else []),
        ]:
            if _available(item.availableAt, item.wakeTime, now, excluded):
                sleeps[_utc(item.wakeTime)] = item
        if observed.wellbeing is not None:
            item = observed.wellbeing
            if _available(item.availableAt, item.recordedAt, now, excluded):
                checkins[_utc(item.recordedAt)] = item
        if observed.bodyMeasurement is not None:
            item = observed.bodyMeasurement
            if _available(item.availableAt, item.measuredAt, now, excluded):
                bodies[_utc(item.measuredAt)] = item
        if observed.habits is not None:
            if observed.habits.date == now.date():
                # Это накопительный снимок дня, не отдельная порция воды.
                # availableAt у типа отсутствует: доступность — момент этого контекста.
                water_snapshots[now.date()] = (now, observed.habits.waterMl)
            else:
                excluded["habits_without_current_day"] += 1

        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        engagement_from = day_start - timedelta(days=CONTEXT_WINDOWS["engagementDays"])
        context_from = now - timedelta(hours=CONTEXT_WINDOWS["mealsHours"])
        workouts_from = now - timedelta(hours=CONTEXT_WINDOWS["workoutsHours"])
        recent_meals = sorted(
            (item for (at, _), item in meals.items() if context_from <= at <= now),
            key=lambda item: _utc(item.eatenAt),
        )[:60]
        recent_workouts = sorted(
            (item for (at, _), item in workouts.items() if workouts_from <= at <= now),
            key=lambda item: _utc(item.startedAt),
        )[:20]
        fresh_from = now - timedelta(hours=CONTEXT_WINDOWS["wellbeingHours"])
        sleep_at = max((at for at in sleeps if fresh_from <= at <= now), default=None)
        checkin_at = max(
            (at for at in checkins if fresh_from <= at <= now), default=None
        )
        body_from = now - timedelta(days=CONTEXT_WINDOWS["bodyDays"])
        body_at = max((at for at in bodies if body_from <= at <= now), default=None)
        latest_sleep = sleeps[sleep_at] if sleep_at is not None else None
        latest_checkin = checkins[checkin_at] if checkin_at is not None else None
        latest_body = bodies[body_at] if body_at is not None else None
        weight = latest_body.weightKg if latest_body else None
        body_fat = latest_body.bodyFatPercent if latest_body else None
        water = water_snapshots.get(now.date(), (now, 0.0))[1]
        water_target = (
            math.floor(weight * 30 / 50 + 0.5) * 50 if weight is not None else 2000
        )

        event_times = [at for at, _ in meals] + [at for at, _ in workouts]
        event_times += list(sleeps) + list(checkins) + list(bodies)
        event_times += [at for at, _ in water_snapshots.values()]
        event_times = [at for at in event_times if engagement_from <= at <= now]
        checkin_today = any(day_start <= at <= now for at in [*sleeps, *checkins])
        meals_today = sum(_utc(item.eatenAt) >= day_start for item in recent_meals)
        populated_sections = sum(
            [
                bool(recent_meals),
                bool(recent_workouts),
                latest_sleep is not None,
                latest_checkin is not None,
                weight is not None or body_fat is not None,
                False,
                water > 0,
            ]
        )
        contexts.append(
            {
                "evaluationTime": now.isoformat(),
                "timeZone": "UTC",
                "localHourFraction": now.hour + now.minute / 60 + now.second / 3600,
                "contextFrom": context_from.isoformat(),
                "meals": [
                    {
                        "eatenAt": _utc(item.eatenAt).isoformat(),
                        "mealType": item.mealType.value,
                        "calories": item.calories,
                        "protein": item.proteins,
                        "fat": item.fats,
                        "carbs": item.carbs,
                        "fiber": item.fiber,
                        "sugars": item.sugars,
                        "addedSugars": item.addedSugar,
                        "starch": 0.0,
                        "alcohol": item.alcohol,
                    }
                    for item in recent_meals
                ],
                "workouts": [
                    {
                        "startedAt": _utc(item.startedAt).isoformat(),
                        "durationMinutes": item.durationMinutes,
                        "caloriesBurned": item.caloriesBurned,
                        "sessionRpe": item.sessionRpe,
                        "fatigueScore": item.fatigueScore,
                        "volumeLoadKg": item.volumeLoadKg,
                        "intensity": item.intensity.value,
                    }
                    for item in recent_workouts
                ],
                "sleep": {
                    "observedAt": sleep_at.isoformat(),
                    "sleepHours": latest_sleep.sleepHours,
                    "sleepQuality": latest_sleep.sleepQuality,
                }
                if latest_sleep is not None and sleep_at is not None
                else None,
                "wellbeing": {
                    "observedAt": checkin_at.isoformat(),
                    "energyScore": latest_checkin.energyScore,
                    "moodScore": latest_checkin.moodScore,
                    "stressLevel": latest_checkin.stressLevel,
                }
                if latest_checkin is not None and checkin_at is not None
                else None,
                "body": {
                    "weightKg": weight,
                    "bodyFatPercent": body_fat,
                    "bodyFatSource": "measured" if body_fat is not None else None,
                    "bmrKcal": latest_body.bmrKcal if latest_body else None,
                    "gender": latest_body.gender if latest_body else None,
                    "heightCm": None,
                    "ageYears": None,
                },
                "vitals": None,
                "habits": {
                    "waterMl": water,
                    "targetWaterMl": water_target,
                    "alcoholUnits24h": 0.0,
                },
                "engagement": _engagement(
                    event_times, now, checkin_today, meals_today, water
                ),
                "populatedSections": populated_sections,
            }
        )
    return contexts, excluded
