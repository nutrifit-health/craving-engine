"""Наблюдаемые признаки и причинная история без скрытых исходов генератора."""

import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import numpy as np

from synthetic_data.types import FeedbackRecord, InformationFeature

HISTORY_DAYS = (7, 28)
FEEDBACK_DELAY_HOURS = 20
NUTRIENTS = (
    ("calories", "kcal"),
    ("protein", "g"),
    ("fat", "g"),
    ("carbs", "g"),
    ("fiber", "g"),
    ("sugars", "g"),
    ("addedSugars", "g"),
    ("starch", "g"),
    ("alcohol", "g"),
)
OPTIONAL_FIELDS = (
    ("sleep", "sleepHours", "h"),
    ("sleep", "sleepQuality", "0..100"),
    ("wellbeing", "stressLevel", "0..100"),
    ("wellbeing", "energyScore", "0..100"),
    ("wellbeing", "moodScore", "0..100"),
    ("body", "weightKg", "kg"),
    ("body", "bodyFatPercent", "%"),
    ("body", "bmrKcal", "kcal/day"),
    ("body", "ageYears", "years"),
)


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{label}: ожидался объект")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: ожидалось конечное число")
    return float(value)


def _at(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Время payload должно быть строкой с часовым поясом")
    at = datetime.fromisoformat(value)
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("Payload требует явно обозначенного UTC")
    return at.astimezone(UTC)


def _rows(value: object, label: str) -> list[dict]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: ожидался список")
    return [_object(row, label) for row in value]


def observed_schema() -> tuple[InformationFeature, ...]:
    """Порядок и смысл whitelist фиксированы до чтения исходов или данных."""
    result = [
        InformationFeature("local_hour", "h", "Доступное локальное время решения")
    ]
    for hours in (6, 24):
        result.append(
            InformationFeature(
                f"meals_{hours}h_count",
                "count",
                f"Записанные приёмы за последние {hours} часов, включая границы",
            )
        )
        for nutrient, unit in NUTRIENTS:
            result.append(
                InformationFeature(
                    f"meals_{hours}h_{nutrient}",
                    unit,
                    f"Сумма {nutrient} только в доступных приёмах за {hours} часов",
                )
            )
    result += [
        InformationFeature(
            "last_meal_present", "0/1", "Есть доступный приём в 48-часовом payload"
        ),
        InformationFeature(
            "last_meal_age_hours",
            "h",
            "Возраст последнего доступного приёма",
            "last_meal_present",
        ),
        InformationFeature(
            "previous_meal_gap_present",
            "0/1",
            "Есть два разных времени доступных приёмов",
        ),
        InformationFeature(
            "previous_meal_gap_hours",
            "h",
            "Интервал между двумя последними разными временами приёмов",
            "previous_meal_gap_present",
        ),
    ]
    for nutrient, unit in NUTRIENTS:
        result.append(
            InformationFeature(
                f"last_meal_{nutrient}",
                unit,
                f"{nutrient} последнего доступного приёма",
                "last_meal_present",
            )
        )
    for section, field, unit in OPTIONAL_FIELDS:
        name = f"{section}_{field}"
        result += [
            InformationFeature(
                name + "_present",
                "0/1",
                f"Поле {section}.{field} присутствует в payload",
            ),
            InformationFeature(
                name,
                unit,
                f"Наблюдаемое {section}.{field}; ноль с отдельной маской при пропуске",
                name + "_present",
            ),
        ]
    for section in ("sleep", "wellbeing"):
        result += [
            InformationFeature(
                f"{section}_present", "0/1", f"Секция {section} присутствует"
            ),
            InformationFeature(
                f"{section}_age_hours",
                "h",
                f"Возраст наблюдения {section}",
                f"{section}_present",
            ),
        ]
    result += [
        InformationFeature(
            "body_gender_present",
            "0/1",
            "Измерение содержит поддерживаемый пол male/female",
        ),
        InformationFeature(
            "body_gender_female", "0/1", "Наблюдаемый пол female", "body_gender_present"
        ),
        InformationFeature(
            "workouts_count",
            "count",
            "Число доступных завершённых тренировок в payload",
        ),
        InformationFeature(
            "workouts_duration_minutes",
            "min",
            "Сумма длительностей доступных тренировок",
        ),
        InformationFeature(
            "workouts_calories", "kcal", "Сумма калорий доступных тренировок"
        ),
        InformationFeature(
            "workouts_present", "0/1", "Есть завершённая доступная тренировка"
        ),
        InformationFeature(
            "last_workout_age_hours",
            "h",
            "Возраст завершения последней тренировки",
            "workouts_present",
        ),
    ]
    for field, unit, operation in (
        ("sessionRpe", "1..10", "max"),
        ("fatigueScore", "0..10", "max"),
        ("volumeLoadKg", "kg", "sum"),
    ):
        name = f"workouts_{field}"
        result += [
            InformationFeature(
                name + "_present", "0/1", f"Есть доступное поле тренировок {field}"
            ),
            InformationFeature(
                name, unit, f"{operation} наблюдаемого {field}", name + "_present"
            ),
        ]
    for field, unit in (
        ("waterMl", "ml"),
        ("targetWaterMl", "ml"),
        ("alcoholUnits24h", "10g units"),
    ):
        result.append(
            InformationFeature(
                f"habits_{field}", unit, f"Наблюдаемое поле habits.{field} из payload"
            )
        )
    for field, unit in (
        ("loggingStreakDays", "days"),
        ("todayCompleteness", "0..1"),
        ("dropoffRisk", "0..1"),
    ):
        result.append(
            InformationFeature(
                f"engagement_{field}",
                unit,
                f"Доступная агрегированная история логирования {field}",
            )
        )
    result += [
        InformationFeature(
            "engagement_daysSinceLastLog_present",
            "0/1",
            "Известна давность последней записи",
        ),
        InformationFeature(
            "engagement_daysSinceLastLog",
            "days",
            "Давность последней доступной записи",
            "engagement_daysSinceLastLog_present",
        ),
        InformationFeature(
            "populated_sections", "count", "Число заполненных секций production payload"
        ),
    ]
    return tuple(result)


def observed_summary(context: dict[str, object]) -> dict[str, float]:
    """Извлекает только whitelist payload; идентификаторы и truth не читаются."""
    now = _at(context.get("evaluationTime"))
    if context.get("vitals") is not None or context.get("timeZone") != "UTC":
        raise ValueError("Диагностика ограничена существующим мостом UTC без vitals")
    meals = _rows(context.get("meals"), "meals")
    meal_times = [_at(row.get("eatenAt")) for row in meals]
    if any(at > now for at in meal_times):
        raise ValueError("Будущая еда не может стать признаком")
    result = {"local_hour": _number(context.get("localHourFraction"), "local hour")}
    for hours in (6, 24):
        recent = [
            row
            for row, at in zip(meals, meal_times, strict=True)
            if now - timedelta(hours=hours) <= at <= now
        ]
        result[f"meals_{hours}h_count"] = float(len(recent))
        for nutrient, _ in NUTRIENTS:
            result[f"meals_{hours}h_{nutrient}"] = sum(
                _number(row.get(nutrient), nutrient) for row in recent
            )
    order = sorted(range(len(meals)), key=lambda index: meal_times[index])
    latest = meals[order[-1]] if order else None
    distinct_times = sorted(set(meal_times))
    result["last_meal_present"] = float(latest is not None)
    result["last_meal_age_hours"] = (
        (now - distinct_times[-1]).total_seconds() / 3600 if distinct_times else 0.0
    )
    result["previous_meal_gap_present"] = float(len(distinct_times) >= 2)
    result["previous_meal_gap_hours"] = (
        (distinct_times[-1] - distinct_times[-2]).total_seconds() / 3600
        if len(distinct_times) >= 2
        else 0.0
    )
    for nutrient, _ in NUTRIENTS:
        result[f"last_meal_{nutrient}"] = (
            _number(latest.get(nutrient), nutrient) if latest is not None else 0.0
        )
    for section, field, _ in OPTIONAL_FIELDS:
        container = context.get(section)
        value = (
            _object(container, section).get(field) if container is not None else None
        )
        name = f"{section}_{field}"
        result[name + "_present"] = float(value is not None)
        result[name] = _number(value, name) if value is not None else 0.0
    for section in ("sleep", "wellbeing"):
        container = context.get(section)
        result[f"{section}_present"] = float(container is not None)
        age = (
            0.0
            if container is None
            else (
                now - _at(_object(container, section).get("observedAt"))
            ).total_seconds()
            / 3600
        )
        if age < 0:
            raise ValueError("Будущее наблюдение не может стать признаком")
        result[f"{section}_age_hours"] = age
    gender = _object(context.get("body"), "body").get("gender")
    result["body_gender_present"] = float(gender in ("male", "female"))
    result["body_gender_female"] = float(gender == "female")
    workouts = _rows(context.get("workouts"), "workouts")
    ends = [
        _at(row.get("startedAt"))
        + timedelta(minutes=_number(row.get("durationMinutes"), "durationMinutes"))
        for row in workouts
    ]
    if any(end > now for end in ends):
        raise ValueError("Незавершённая тренировка не может стать признаком")
    result["workouts_count"] = float(len(workouts))
    result["workouts_duration_minutes"] = sum(
        _number(row.get("durationMinutes"), "durationMinutes") for row in workouts
    )
    result["workouts_calories"] = sum(
        _number(row.get("caloriesBurned"), "caloriesBurned") for row in workouts
    )
    result["workouts_present"] = float(bool(workouts))
    result["last_workout_age_hours"] = (
        (now - max(ends)).total_seconds() / 3600 if ends else 0.0
    )
    for field in ("sessionRpe", "fatigueScore", "volumeLoadKg"):
        values = [
            _number(row[field], field) for row in workouts if row.get(field) is not None
        ]
        result[f"workouts_{field}_present"] = float(bool(values))
        result[f"workouts_{field}"] = (
            sum(values) if field == "volumeLoadKg" else max(values, default=0.0)
        )
    habits = _object(context.get("habits"), "habits")
    for field in ("waterMl", "targetWaterMl", "alcoholUnits24h"):
        result[f"habits_{field}"] = _number(habits.get(field), field)
    engagement = _object(context.get("engagement"), "engagement")
    for field in ("loggingStreakDays", "todayCompleteness", "dropoffRisk"):
        result[f"engagement_{field}"] = _number(engagement.get(field), field)
    last_log = engagement.get("daysSinceLastLog")
    result["engagement_daysSinceLastLog_present"] = float(last_log is not None)
    result["engagement_daysSinceLastLog"] = (
        _number(last_log, "daysSinceLastLog") if last_log is not None else 0.0
    )
    result["populated_sections"] = _number(
        context.get("populatedSections"), "populatedSections"
    )
    if set(result) != {feature.name for feature in observed_schema()}:
        raise RuntimeError("Whitelist признаков разошёлся со схемой")
    return result


def build_history_features(
    records: list[FeedbackRecord],
    current: np.ndarray,
    schema: tuple[InformationFeature, ...],
) -> tuple[np.ndarray, tuple[InformationFeature, ...]]:
    """Предыдущие 7/28 суток и доступный reported feedback; порядок строк сохранён.

    Окно признаков: [now-window, now), только уже доступные решения того же
    пользователя. Окно feedback также привязано к прежнему decision_at, а
    исход читается только при feedback_at <= now. В знаменатель reporting
    входят прежние решения с истёкшей фиксированной задержкой 20 часов.
    """
    values = np.asarray(current, dtype=np.float64)
    names = [feature.name for feature in schema]
    if (
        values.shape != (len(records), len(schema))
        or not np.isfinite(values).all()
        or len(set(names)) != len(names)
    ):
        raise ValueError("Неверная форма, значения или имена текущих признаков")
    columns = {name: index for index, name in enumerate(names)}
    if any(
        feature.available_if is not None and feature.available_if not in columns
        for feature in schema
    ):
        raise ValueError("Индикатор наличия отсутствует в схеме")
    availability = np.ones(values.shape, dtype=bool)
    for column, feature in enumerate(schema):
        if feature.available_if is not None:
            mask = values[:, columns[feature.available_if]]
            if np.any((mask != 0) & (mask != 1)):
                raise ValueError("Индикатор наличия должен быть бинарным")
            availability[:, column] = mask == 1
    extended = list(schema)
    for days in HISTORY_DAYS:
        for statistic in ("mean", "delta", "count"):
            for feature in schema:
                unit = "count" if statistic == "count" else feature.unit
                extended.append(
                    InformationFeature(
                        f"history_{days}d_{statistic}__{feature.name}",
                        unit,
                        f"{statistic}: прежние доступные значения {feature.name} за {days} суток; count=0 означает отсутствие истории, delta=0 при отсутствии текущего значения или истории",
                    )
                )
    for window in ("7d", "28d", "all"):
        for field, unit, description in (
            ("due_count", "count", "Прежние решения, для которых истекли 20 часов"),
            ("reported_count", "count", "Доступные reported исходы прежних решений"),
            (
                "positive_count",
                "count",
                "Положительные среди доступных reported исходов",
            ),
            (
                "smoothed_rate",
                "0..1",
                "(1+positive_count)/(2+reported_count), без подстановки пропусков",
            ),
            (
                "reporting_fraction",
                "0..1",
                "Доля доступных reported среди due решений; 0 при due_count=0",
            ),
        ):
            extended.append(
                InformationFeature(f"feedback_{window}_{field}", unit, description)
            )
    output = np.empty((len(records), len(extended)), dtype=np.float64)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    events: set[tuple[str, str]] = set()
    for index, row in enumerate(records):
        if row.available_at > row.decision_at or (row.split, row.event_id) in events:
            raise ValueError("Будущая доступность или повтор события в истории")
        groups[(row.split, row.user_id)].append(index)
        events.add((row.split, row.event_id))
    for indices in groups.values():
        indices.sort(
            key=lambda index: (records[index].decision_at, records[index].event_id)
        )
        prior: list[int] = []
        for index in indices:
            row, now = records[index], records[index].decision_at
            earlier = [
                j
                for j in prior
                if records[j].decision_at < now and records[j].available_at <= now
            ]
            pieces = [values[index]]
            for days in HISTORY_DAYS:
                selected = [
                    j
                    for j in earlier
                    if records[j].decision_at >= now - timedelta(days=days)
                ]
                count = (
                    availability[selected].sum(axis=0)
                    if selected
                    else np.zeros(len(schema))
                )
                total = (
                    np.where(availability[selected], values[selected], 0.0).sum(axis=0)
                    if selected
                    else np.zeros(len(schema))
                )
                mean = np.divide(
                    total, count, out=np.zeros(len(schema)), where=count > 0
                )
                delta = np.where(
                    availability[index] & (count > 0), values[index] - mean, 0.0
                )
                pieces.extend((mean, delta, count))
            for days in (7, 28, None):
                candidates = [
                    j
                    for j in earlier
                    if days is None
                    or records[j].decision_at >= now - timedelta(days=days)
                ]
                due = [
                    j
                    for j in candidates
                    if records[j].decision_at + timedelta(hours=FEEDBACK_DELAY_HOURS)
                    <= now
                ]
                reported = [
                    j
                    for j in candidates
                    if records[j].feedback_at is not None
                    and records[j].feedback_at <= now
                ]
                if any(records[j].outcome is None for j in reported):
                    raise ValueError("Feedback timestamp без наблюдаемого исхода")
                # Фиксированный корпус запрещает feedback раньше договорённого срока.
                if any(
                    records[j].feedback_at
                    < records[j].decision_at + timedelta(hours=FEEDBACK_DELAY_HOURS)
                    for j in reported
                ):
                    raise ValueError(
                        "Feedback поступил раньше фиксированной задержки корпуса"
                    )
                positive = sum(records[j].outcome for j in reported)
                pieces.append(
                    np.asarray(
                        [
                            len(due),
                            len(reported),
                            positive,
                            (1 + positive) / (2 + len(reported)),
                            len(reported) / len(due) if due else 0.0,
                        ]
                    )
                )
            output[index] = np.concatenate(pieces)
            prior.append(index)
    output.setflags(write=False)
    return output, tuple(extended)
