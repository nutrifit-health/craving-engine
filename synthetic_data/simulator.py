"""Синтетические 180-дневные истории с раздельными исходами и наблюдениями.

Сохранены формулы, пропуски, временные отметки и порядок случайных выборов
исторического симулятора. Медицинская репрезентативность не установлена.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import date, datetime, time, timedelta

import numpy as np

from synthetic_data.types import (
    BodyMeasurementRecord,
    HealthHabitsRecord,
    MealRecord,
    MealType,
    SleepRecord,
    SyntheticUser,
    TrueDailyState,
    UserHealthContext,
    WellbeingRecord,
    WorkoutIntensity,
    WorkoutRecord,
)


class IndependentGroundTruthEngine:
    """Независимый генератор истинных событий срыва и тяги (Ground Truth)."""

    @staticmethod
    def evaluate_ground_truth(
        user: SyntheticUser,
        day_index: int,
        day_state: dict[str, float | bool],
        rng: random.Random,
        drift_active_trigger: str | None = None,
    ) -> tuple[float, bool, int, float]:
        """Вычисляет истинную вероятность срыва и факт наступления события.

        Возвращает:
            (prob [0.0..1.0], occurred [bool], intensity [1..10], peak_hour [float])
        """
        sens = dict(user.latent_sensitivities)

        # Обработка Concept Drift: изменение чувствительностей во времени
        if user.concept_drift_type == "slow":
            # Плавный дрейф за 180 дней
            drift_progress = day_index / 180.0
            sens["stress"] = sens.get("stress", 0.5) * (1.0 - 0.4 * drift_progress)
            sens["fasting"] = sens.get("fasting", 0.5) * (1.0 + 0.4 * drift_progress)
        elif user.concept_drift_type == "moderate":
            # Смена паттерна после 90-го дня
            if day_index > 90:
                sens["stress"] = sens.get("stress", 0.5) * 0.4
                sens["glycogen"] = 0.85
                sens["hypoglycemia"] = 0.80
        elif user.concept_drift_type == "abrupt":
            # Резкая 3-фазная смена: дни 1-60 -> 61-120 -> 121-180
            if day_index <= 60:
                sens = {
                    "stress": 0.90,
                    "sleep_debt": 0.85,
                    "noise": sens.get("noise", 0.12),
                }
            elif day_index <= 120:
                sens = {
                    "glycogen": 0.90,
                    "hypoglycemia": 0.80,
                    "noise": sens.get("noise", 0.12),
                }
            else:
                sens = {
                    "fasting": 0.95,
                    "circadian": 0.85,
                    "noise": sens.get("noise", 0.12),
                }

        # Извлечение факторов дня
        sleep_debt = max(0.0, 7.8 - day_state["sleep_hours"])
        sleep_debt_norm = min(1.0, sleep_debt / 3.5)
        fasting_norm = min(1.0, max(0.0, (day_state["fasting_hours"] - 3.5) / 5.0))
        glycogen_norm = day_state["glycogen_deficit"]
        stress_norm = day_state["stress_level"] / 100.0
        sugar_norm = min(1.0, day_state["sugar_intake_g"] / 40.0)
        alcohol_norm = min(1.0, day_state["alcohol_units"] / 4.0)

        # Нелинейные межфакторные синергии (в биологии эффекты умножаются)
        synergy_sleep_stress = sleep_debt_norm * stress_norm
        synergy_fasting_night = fasting_norm * (1.0 if day_state["is_evening"] else 0.2)
        synergy_glycogen_hypo = glycogen_norm * sugar_norm

        # Взвешенная сумма стимулов
        stimulus = (
            sens.get("sleep_debt", 0.3) * sleep_debt_norm * 1.8
            + sens.get("stress", 0.3) * stress_norm * 2.0
            + sens.get("fasting", 0.3) * fasting_norm * 2.2
            + sens.get("glycogen", 0.2) * glycogen_norm * 2.0
            + sens.get("sugar_priming", 0.3) * sugar_norm * 1.7
            + sens.get("alcohol", 0.1) * alcohol_norm * 1.5
            + sens.get("circadian", 0.2)
            * (0.8 if day_state["is_circadian_dip"] else 0.1)
            + 0.9 * synergy_sleep_stress
            + 1.1 * synergy_fasting_night
            + 0.8 * synergy_glycogen_hypo
        )

        # Индивидуальный базовый самоконтроль (willpower threshold)
        # Усталость и алкоголь истощают самоконтроль. Откалибровано для базовой частоты срывов 20-35%.
        willpower = 2.55 - (
            0.7 * sleep_debt_norm + 0.9 * alcohol_norm + 0.6 * stress_norm
        )
        willpower = max(0.5, willpower)

        # Стохастический шум среды и настроения
        noise_sigma = sens.get("noise", 0.15)
        stochastic_noise = rng.gauss(0, noise_sigma)

        # Истинная логистическая функция уязвимости
        latent_score = stimulus - willpower + stochastic_noise
        prob = 1.0 / (1.0 + math.exp(-2.0 * latent_score))
        prob = round(float(np.clip(prob, 0.01, 0.99)), 3)

        # Событие срыва
        occurred = rng.random() < prob

        # Интенсивность тяги (1..10)
        base_intensity = 1.0 + 8.5 * prob + rng.gauss(0, 0.8)
        intensity = int(np.clip(round(base_intensity), 1, 10))

        # Час пика уязвимости
        if day_state["is_evening"]:
            peak_hour = 21.5 + rng.gauss(0, 0.8)
        elif day_state["is_circadian_dip"]:
            peak_hour = 17.0 + rng.gauss(0, 0.6)
        else:
            peak_hour = 14.0 + rng.gauss(0, 1.5)
        peak_hour = float(np.clip(peak_hour, 8.0, 23.9))

        return prob, occurred, intensity, peak_hour


class LongitudinalSimulator:
    """Симулирует 180-дневную траекторию пользователя и формирует наблюдаемый контекст."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.base_rng = random.Random(seed)

    def simulate_user_180_days(
        self,
        user: SyntheticUser,
        user_seed: int | None = None,
        override_wearable_mode: str | None = None,
        force_missingness_level: float | None = None,
        inject_noise_at_day_100: bool = False,
    ) -> list[tuple[TrueDailyState, UserHealthContext]]:
        """Генерирует полную 180-дневную последовательность (истинное состояние + наблюдаемый контекст).

        Параметры:
            user: синтетический профиль пользователя;
            override_wearable_mode: для matched pairs (e.g. "none" vs "continuous_24_7");
            force_missingness_level: принудительный уровень доступности данных [0.25..1.0];
            inject_noise_at_day_100: инъекция аномальной ошибки на 100-й день (тест устойчивости).
        """
        sim_seed = (
            user_seed
            if user_seed is not None
            else int.from_bytes(
                hashlib.sha256(f"{self.seed}:{user.user_id}".encode()).digest(),
                "big",
            )
        )
        # Истина не зависит от количества случайных решений о доступности логов.
        rng = random.Random(f"{sim_seed}:truth")
        observation_rng = random.Random(f"{sim_seed}:observations")

        wearable_mode = override_wearable_mode or user.wearable_mode
        history: list[tuple[TrueDailyState, UserHealthContext]] = []

        start_date = date(2026, 1, 1)
        current_weight = user.weight_kg
        accumulated_fatigue = 0.0

        # Период отпуска (дни 80-90)
        vacation_start = 80
        vacation_end = 91

        yesterday_relapse = False

        for day_idx in range(1, 181):
            sim_date = start_date + timedelta(days=day_idx - 1)
            is_weekend = sim_date.weekday() >= 5
            is_vacation = vacation_start <= day_idx <= vacation_end

            # ── 1. Тренировочный микроцикл и физическая активность ─────────────
            cycle_week = ((day_idx - 1) // 7) % 4  # 0, 1, 2 = загрузка, 3 = разгрузка
            is_deload_week = cycle_week == 3

            # Вероятность тренировки сегодня
            workout_prob = user.workouts_per_week / 7.0
            if is_deload_week:
                workout_prob *= 0.6
            if is_vacation:
                workout_prob *= 0.3
            if is_weekend and user.activity_type != "student_athlete":
                workout_prob *= 0.8

            has_workout = rng.random() < workout_prob
            workout_dur = 0
            workout_cals = 0.0
            glycogen_def = 0.0

            if has_workout:
                workout_dur = int(
                    user.avg_workout_duration_min * rng.uniform(0.85, 1.15)
                )
                intensity = (
                    "hard"
                    if user.activity_type in ("crossfit", "strength")
                    else "moderate"
                )
                cal_rate = 10.0 if intensity == "hard" else 7.0
                workout_cals = workout_dur * cal_rate
                accumulated_fatigue += 0.35
                # Восполнение углеводами (случайное)
                replenished = rng.uniform(0.3, 0.9)
                glycogen_def = float(np.clip(1.0 - replenished, 0.05, 0.95))
            else:
                accumulated_fatigue = max(0.0, accumulated_fatigue - 0.25)
                glycogen_def = max(0.0, accumulated_fatigue * 0.2)

            # ── 2. Истинный сон и циркадные параметры ──────────────────────────
            base_sleep = user.sleep_hours
            if is_weekend:
                base_sleep += 0.8 if user.sleep_pattern == "social_jetlag" else 0.3
            if is_vacation:
                base_sleep += 0.7
            if accumulated_fatigue > 0.8:
                base_sleep -= 0.5  # Перетренированность ухудшает сон

            true_sleep_h = float(np.clip(base_sleep + rng.gauss(0, 0.4), 3.5, 10.5))
            true_sleep_q = float(
                np.clip(
                    user.sleep_quality
                    + rng.gauss(0, 5.0)
                    - (accumulated_fatigue * 8.0),
                    30.0,
                    98.0,
                )
            )

            # ── 3. Стресс и витальные показатели (HR/HRV) ──────────────────────
            base_stress = 30.0
            if is_weekend:
                base_stress -= 12.0
            if is_vacation:
                base_stress -= 18.0
            if true_sleep_h < 6.0:
                base_stress += 25.0
            if accumulated_fatigue > 0.6:
                base_stress += 15.0

            true_stress = float(np.clip(base_stress + rng.gauss(0, 10.0), 5.0, 98.0))
            true_energy = float(
                np.clip(
                    95.0 - true_stress * 0.5 - (accumulated_fatigue * 20.0), 10.0, 100.0
                )
            )

            # Пульс покоя и HRV реагируют на стресс и недосып
            hr_shift = int((true_stress - 30.0) * 0.15) + (
                3 if true_sleep_h < 6.0 else 0
            )
            true_rhr = int(
                np.clip(user.resting_hr + hr_shift + rng.gauss(0, 1.5), 40, 105)
            )
            hrv_ratio = math.exp(
                -((true_stress - 30.0) * 0.012) - ((7.5 - true_sleep_h) * 0.05)
            )
            true_hrv = float(
                np.clip(
                    user.hrv_rmssd * hrv_ratio * rng.uniform(0.92, 1.08), 12.0, 135.0
                )
            )

            # ── 4. Питание и интервалы голодания ──────────────────────────────
            fasting_hours = rng.uniform(3.0, 6.5)
            if user.nutrition_logging_mode == "chaotic":
                fasting_hours = rng.uniform(2.5, 9.0)
            fasting_hours = round(fasting_hours, 1)
            # Время решения фиксировано заранее и не зависит от будущего пика тяги.
            is_evening = False
            is_circadian_dip = True

            sugar_intake = rng.uniform(5.0, 35.0)
            if is_weekend:
                sugar_intake += rng.uniform(10.0, 40.0)
            alcohol_units = 0.0
            if is_weekend and rng.random() < 0.45:
                alcohol_units = rng.uniform(1.5, 4.5)

            # Сценарий смены сахарного триггера; исход остаётся вероятностным.
            is_decay_persona = (
                user.user_id == "USR-P098" or user.persona_id == "Persona 098"
            )
            if is_decay_persona:
                sugar_intake = 45.0 if day_idx <= 60 else 0.0
            sugar_intake = round(sugar_intake, 1)

            # Плавный дрейф массы тела
            current_weight += rng.gauss(0, 0.03)

            # ── 5. Независимый расчет Ground Truth ─────────────────────────────
            day_state_dict = {
                "sleep_hours": true_sleep_h,
                "sleep_quality": true_sleep_q,
                "fasting_hours": fasting_hours,
                "glycogen_deficit": glycogen_def,
                "stress_level": true_stress,
                "sugar_intake_g": sugar_intake,
                "alcohol_units": alcohol_units,
                "is_evening": is_evening,
                "is_circadian_dip": is_circadian_dip,
            }

            prob, occurred, intensity, peak_hour = (
                IndependentGroundTruthEngine.evaluate_ground_truth(
                    user, day_idx, day_state_dict, rng
                )
            )

            # Заданные вероятности сценария не означают принудительные метки.
            if is_decay_persona:
                prob = 0.85 if day_idx <= 60 else 0.05
                occurred = rng.random() < prob
                intensity = int(
                    np.clip(round(1.0 + 8.5 * prob + rng.gauss(0, 0.8)), 1, 10)
                )

            true_state = TrueDailyState(
                day_index=day_idx,
                date=sim_date,
                is_weekend=is_weekend,
                is_vacation=is_vacation,
                true_sleep_hours=round(true_sleep_h, 1),
                true_sleep_quality=round(true_sleep_q, 1),
                true_weight_kg=round(current_weight, 1),
                true_body_fat_pct=user.body_fat_pct,
                true_resting_hr=true_rhr,
                true_hrv_rmssd=round(true_hrv, 1),
                true_stress_level=round(true_stress, 1),
                true_energy_level=round(true_energy, 1),
                true_fasting_hours_at_eval=round(fasting_hours, 1),
                true_glycogen_deficit=round(glycogen_def, 2),
                true_sugar_intake_g=round(sugar_intake, 1),
                true_alcohol_units=round(alcohol_units, 1),
                true_water_deficit_ml=round(rng.uniform(100, 600)),
                workout_performed=has_workout,
                workout_intensity="hard"
                if has_workout and user.activity_type in ("crossfit", "strength")
                else "moderate",
                workout_calories=round(workout_cals, 1),
                true_relapse_prob=round(prob, 3),
                relapse_occurred=occurred,
                craving_intensity=intensity,
                peak_risk_hour=round(peak_hour, 1),
            )

            # ── 6. Маскирование данных: формирование наблюдаемого UserHealthContext ─
            observed_context = self._build_observed_context(
                user,
                true_state,
                wearable_mode,
                force_missingness_level,
                observation_rng,
                yesterday_relapse=yesterday_relapse,
                inject_noise=(inject_noise_at_day_100 and day_idx == 100),
            )

            history.append((true_state, observed_context))
            yesterday_relapse = true_state.relapse_occurred

        return history

    def _build_observed_context(
        self,
        user: SyntheticUser,
        state: TrueDailyState,
        wearable_mode: str,
        force_missingness: float | None,
        rng: random.Random,
        yesterday_relapse: bool = False,
        inject_noise: bool = False,
    ) -> UserHealthContext:
        """Превращает объективную реальность в то, что видит приложение NutriFit."""
        eval_dt = datetime.combine(state.date, time(16, 30))

        # Определение базовой вероятности логирования еды
        if force_missingness is not None:
            meal_prob = force_missingness
        elif user.nutrition_logging_mode == "main_meals_only":
            meal_prob = 0.80
        elif user.nutrition_logging_mode == "forgets_snacks":
            meal_prob = 0.85
        elif user.nutrition_logging_mode == "chaotic":
            meal_prob = 0.40
        elif user.nutrition_logging_mode == "drop_off":
            meal_prob = (
                0.95
                if state.day_index <= 14
                else (0.45 if state.day_index <= 30 else 0.15)
            )
        elif user.nutrition_logging_mode == "none":
            meal_prob = 0.05
        else:
            meal_prob = 0.95

        # Поведенческий пропуск: срыв НАКАНУНЕ (стыд за вчерашний срыв) или выходные.
        # ВАЖНО: отсутствие утечки (no target leakage) — пропуск в день t зависит от исхода t-1, а не t!
        if user.behavioral_missingness_prone:
            if yesterday_relapse:
                meal_prob *= 0.50  # Стыдно вносить еду на следующий день после срыва
            if state.is_weekend:
                meal_prob *= 0.70  # Отдых от трекинга

        # Последняя реальная еда предшествует решению на истинный интервал голодания.
        last_meal_at = eval_dt - timedelta(hours=state.true_fasting_hours_at_eval)
        breakfast_at = datetime.combine(state.date, time(8, 0))
        meal_specs = [
            (last_meal_at, MealType.BREAKFAST, 1.0, 22.0, 15.0, 55.0, 6.0),
        ]
        if last_meal_at >= breakfast_at + timedelta(hours=2):
            meal_specs = [
                (breakfast_at, MealType.BREAKFAST, 0.25, 22.0, 15.0, 55.0, 6.0),
                (
                    last_meal_at - timedelta(hours=1),
                    MealType.LUNCH,
                    0.50,
                    35.0,
                    20.0,
                    70.0,
                    5.0,
                ),
                (last_meal_at, MealType.SNACK, 0.25, 4.0, 6.0, 28.0, 1.5),
            ]

        # Пропуски скрывают реальные события, но не меняют истину дня.
        meals: list[MealRecord] = []
        if rng.random() < meal_prob:
            for (
                eaten_at,
                meal_type,
                sugar_share,
                proteins,
                fats,
                carbs,
                fiber,
            ) in meal_specs:
                if meal_type == MealType.SNACK and user.nutrition_logging_mode in (
                    "forgets_snacks",
                    "main_meals_only",
                ):
                    continue
                sugars = state.true_sugar_intake_g * sugar_share
                carbs = max(carbs, sugars + fiber)
                added_sugar = max(0.0, sugars - 8.0)
                # Внутренняя единица алкоголя этого симулятора равна 10 граммам.
                alcohol = (
                    state.true_alcohol_units * 10.0 if eaten_at == last_meal_at else 0.0
                )
                gi = 65.0 if sugars > 25.0 else 48.0
                if inject_noise and meal_type == MealType.LUNCH:
                    # Ошибка порции согласованно уменьшает все нутриенты в три раза.
                    proteins /= 3.0
                    fats /= 3.0
                    carbs /= 3.0
                    fiber /= 3.0
                    sugars /= 3.0
                    added_sugar /= 3.0
                    alcohol /= 3.0
                meals.append(
                    MealRecord(
                        eatenAt=eaten_at,
                        availableAt=eaten_at,
                        mealType=meal_type,
                        calories=4.0 * proteins
                        + 9.0 * fats
                        + 4.0 * carbs
                        + 7.0 * alcohol,
                        proteins=proteins,
                        fats=fats,
                        carbs=carbs,
                        fiber=fiber,
                        sugars=sugars,
                        addedSugar=added_sugar,
                        alcohol=alcohol,
                        gi=gi,
                        gl=round(gi * carbs / 100.0, 2),
                    )
                )

        # Тренировка
        workouts: list[WorkoutRecord] = []
        if state.workout_performed:
            log_workout = True
            if force_missingness is not None and rng.random() > force_missingness:
                log_workout = False
            if log_workout:
                calorie_rate = 10.0 if state.workout_intensity == "hard" else 7.0
                workout_duration = state.workout_calories / calorie_rate
                workout_end = datetime.combine(state.date, time(12, 0))
                workouts.append(
                    WorkoutRecord(
                        startedAt=workout_end - timedelta(minutes=workout_duration),
                        availableAt=workout_end,
                        durationMinutes=workout_duration,
                        sportKey=user.activity_type,
                        intensity=WorkoutIntensity.HARD
                        if state.workout_intensity == "hard"
                        else WorkoutIntensity.MODERATE,
                        caloriesBurned=state.workout_calories,
                        sessionRpe=8.0 if state.workout_intensity == "hard" else 6.0,
                        fatigueScore=6.5,
                        volumeLoadKg=5000.0
                        if "strength" in user.activity_type
                        or "crossfit" in user.activity_type
                        else 0.0,
                    )
                )

        # Сон
        sleep_rec: SleepRecord | None = None
        has_sleep_data = wearable_mode not in ("none", "day_only", "workout_only")
        if force_missingness is not None and rng.random() > force_missingness:
            has_sleep_data = False

        if has_sleep_data:
            wake_time = datetime.combine(state.date, time(7, 15))
            sleep_rec = SleepRecord(
                bedtime=wake_time - timedelta(hours=state.true_sleep_hours),
                wakeTime=wake_time,
                availableAt=wake_time,
                sleepHours=state.true_sleep_hours,
                sleepQuality=state.true_sleep_quality,
                deepSleepMinutes=round(state.true_sleep_hours * 12.0),
            )

        # Тело (замеры раз в 14 дней или при инициализации)
        body_rec = BodyMeasurementRecord(
            measuredAt=eval_dt,
            availableAt=eval_dt,
            weightKg=state.true_weight_kg,
            bodyFatPercent=state.true_body_fat_pct,
            muscleMassKg=user.muscle_mass_kg,
            bmrKcal=user.bmr_kcal,
            bmi=round(state.true_weight_kg / ((user.height_cm / 100.0) ** 2), 1),
            gender=user.gender,
        )

        # Субъективное самочувствие (чек-ин)
        wellbeing: WellbeingRecord | None = None
        has_wellbeing = user.nutrition_logging_mode != "none"
        if force_missingness is not None and rng.random() > force_missingness:
            has_wellbeing = False

        if has_wellbeing:
            rep_stress = state.true_stress_level
            if inject_noise:
                rep_stress = 95.0  # Ложный стресс-выброс
            wellbeing = WellbeingRecord(
                recordedAt=eval_dt,
                availableAt=eval_dt,
                energyScore=state.true_energy_level,
                moodScore=round(max(10.0, 100.0 - rep_stress * 0.6)),
                stressLevel=rep_stress,
            )

        # Привычки
        habits = HealthHabitsRecord(
            date=state.date,
            waterMl=round(max(0.0, 2500.0 - state.true_water_deficit_ml)),
            targetWaterMl=2500.0,
            habitsCompleted=3 if state.workout_performed else 2,
            habitsTotal=5,
            currentStreakDays=min(30, state.day_index),
        )

        return UserHealthContext(
            userId=user.user_id,
            evaluationTime=eval_dt,
            meals=meals,
            workouts=workouts,
            sleep=sleep_rec,
            bodyMeasurement=body_rec,
            wellbeing=wellbeing,
            habits=habits,
        )
