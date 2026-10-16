"""Генератор новых синтетических профилей для исследовательских когорт.

Перенесена только сеточная ветвь, использовавшаяся для V7–V9.
Заданные распределения не являются оценкой реальной популяции.
"""

import random

import numpy as np

from synthetic_data.types import SyntheticUser


def _calc_bmr(weight: float, height: float, age: int, gender: str) -> float:
    """Сохраняет формулу основного обмена исторического симулятора."""
    base = 10.0 * weight + 6.25 * height - 5.0 * age
    return round(base + (5.0 if gender == "male" else -161.0), 1)


def calculate_deurenberg_fat(
    bmi: float, age: int, gender: str, shift: float = 0.0
) -> float:
    """Формула Deurenberg et al. (1991) для процента жира."""
    gender_code = 1 if gender == "male" else 0
    raw = 1.20 * bmi + 0.23 * age - 10.8 * gender_code - 5.4 + shift
    min_fat = 5.0 if gender == "male" else 12.0
    max_fat = 52.0
    return float(np.clip(raw, min_fat, max_fat))


def estimate_hrv_rmssd(age: int, rhr: int, athletic: bool = False) -> float:
    """Оценка популяционной нормы HRV RMSSD с учетом возраста и пульса покоя."""
    # Возрастное падение RMSSD: у молодых ~65-90 мс, у пожилых ~20-35 мс
    base = 90.0 - (age * 0.75)
    # Поправка на ЧСС покоя: чем ниже пульс, тем выше парасимпатический тонус
    rhr_bonus = (70 - rhr) * 0.8
    athletic_bonus = 20.0 if athletic else 0.0
    val = base + rhr_bonus + athletic_bonus
    return float(np.clip(val, 15.0, 130.0))


class SyntheticPopulationGenerator:
    """Создаёт ровно запрошенное число новых сеточных профилей."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(self, count: int, start_idx: int = 1) -> list[SyntheticUser]:
        """Генерирует систематически распределенных пользователей для равномерного покрытия пространства."""
        users: list[SyntheticUser] = []

        age_brackets = [
            ("young_adult", 18, 24),
            ("adult_25_40", 25, 40),
            ("middle_40_60", 41, 60),
            ("senior_60_75", 61, 75),
            ("elderly_75_plus", 76, 84),
        ]
        body_fat_tiers = [
            ("lean", -6.0),
            ("normal", 0.0),
            ("overweight", 5.0),
            ("obese", 11.0),
        ]
        activity_tiers = [
            ("sedentary", 0, 0),
            ("walking", 4, 40),
            ("sports_regular", 3, 50),
            ("crossfit", 5, 80),
            ("endurance", 5, 90),
        ]
        wearable_options = [
            "continuous_24_7",
            "continuous_24_7",
            "irregular",
            "none",
            "day_only",
            "noisy",
        ]
        logging_options = [
            "perfect",
            "main_meals_only",
            "forgets_snacks",
            "retroactive",
            "chaotic",
            "underreporting",
        ]
        drift_options = ["none", "none", "slow", "moderate", "abrupt"]

        for i in range(count):
            uid = f"USR-G{start_idx + i:03d}"
            gender = self.rng.choice(["male", "female"])
            age_tier, min_age, max_age = self.rng.choice(age_brackets)
            age = self.rng.randint(min_age, max_age)

            height = round(self.rng.gauss(178 if gender == "male" else 165, 7.0), 1)
            height = float(np.clip(height, 150.0, 198.0))

            _fat_tier, fat_shift = self.rng.choice(body_fat_tiers)
            base_bmi = self.rng.uniform(19.5, 36.0)
            weight = round(base_bmi * ((height / 100.0) ** 2), 1)
            bmi = round(weight / ((height / 100.0) ** 2), 1)
            body_fat = round(calculate_deurenberg_fat(bmi, age, gender, fat_shift), 1)
            muscle = round(
                weight
                * (1.0 - body_fat / 100.0)
                * (0.57 if gender == "male" else 0.51),
                1,
            )
            bmr = _calc_bmr(weight, height, age, gender)

            act_type, w_per_wk, w_dur = self.rng.choice(activity_tiers)
            is_athletic = act_type in ("crossfit", "endurance")

            rhr_base = 50 if is_athletic else (66 if gender == "male" else 70)
            rhr = int(np.clip(rhr_base + self.rng.gauss(0, 4.5), 42, 90))
            hrv = round(
                estimate_hrv_rmssd(age, rhr, is_athletic)
                * (1.0 + self.rng.gauss(0, 0.08)),
                1,
            )
            hrv_sd = round(hrv * 0.18, 1)

            wearable = self.rng.choice(wearable_options)
            nutr_log = self.rng.choice(logging_options)
            drift = self.rng.choice(drift_options)

            # Формируем скрытые триггеры
            # Назначаем индивидуальный профиль (User A..J стили)
            profile_type = self.rng.choice(
                ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            )
            noise_level = 0.12
            triggers: dict[str, float] = {}

            if profile_type == "A":  # Недосып + стресс
                triggers = {
                    "sleep_debt": 0.88,
                    "stress": 0.85,
                    "circadian": 0.65,
                    "noise": 0.10,
                }
            elif profile_type == "B":  # Голодание + вечер
                triggers = {
                    "fasting": 0.92,
                    "circadian": 0.85,
                    "hypoglycemia": 0.70,
                    "noise": 0.08,
                }
            elif profile_type == "C":  # Тренировка + недовосстановление
                triggers = {
                    "glycogen": 0.90,
                    "hypoglycemia": 0.75,
                    "sleep_debt": 0.60,
                    "noise": 0.10,
                }
            elif profile_type == "D":  # Случайный шумный пользователь
                triggers = {
                    "stress": 0.40,
                    "fasting": 0.40,
                    "sugar_priming": 0.40,
                    "noise": 0.45,
                }
                noise_level = 0.45
            elif profile_type == "E":  # Устойчивый сильный триггер, мало шума
                triggers = {"sugar_priming": 0.95, "hypoglycemia": 0.90, "noise": 0.05}
                noise_level = 0.05
            elif profile_type == "F":  # Сильные триггеры, плохое логирование
                triggers = {"fasting": 0.85, "stress": 0.80, "noise": 0.12}
                nutr_log = "chaotic"
            elif profile_type == "G":  # Физиология качественная, чек-инов нет
                triggers = {"stress": 0.75, "glycogen": 0.70, "noise": 0.10}
                wearable = "continuous_24_7"
            elif profile_type == "H":  # Без часов, но дневник идеален
                triggers = {"fasting": 0.80, "sugar_priming": 0.75, "noise": 0.09}
                wearable = "none"
                nutr_log = "perfect"
            elif profile_type == "I":  # Много данных, низкая предсказуемость
                triggers = {
                    "stress": 0.35,
                    "sleep_debt": 0.35,
                    "fasting": 0.35,
                    "noise": 0.60,
                }
                wearable = "continuous_24_7"
                nutr_log = "perfect"
                noise_level = 0.60
            elif profile_type == "J":  # Мало данных, высокая предсказуемость
                triggers = {"fasting": 0.96, "circadian": 0.90, "noise": 0.04}
                wearable = "none"
                nutr_log = "main_meals_only"
                noise_level = 0.04

            # Difficulty level
            if noise_level >= 0.40 or drift == "abrupt":
                diff = 5 if noise_level >= 0.50 else 4
            elif nutr_log in ("chaotic", "drop_off") or wearable == "none":
                diff = 3
            elif len(triggers) > 2:
                diff = 2
            else:
                diff = 1

            completeness = 0.95
            if nutr_log == "main_meals_only":
                completeness = 0.75
            elif nutr_log == "forgets_snacks":
                completeness = 0.80
            elif nutr_log in ("retroactive", "underreporting"):
                completeness = 0.70
            elif nutr_log == "chaotic":
                completeness = 0.40
            if wearable == "none":
                completeness *= 0.65

            sleep_h = round(self.rng.uniform(5.5, 8.5), 1)
            sleep_q = round(self.rng.uniform(55.0, 90.0), 1)
            sleep_pat = (
                "stable"
                if sleep_h >= 7.0
                else ("short" if sleep_h < 6.0 else "social_jetlag")
            )

            users.append(
                SyntheticUser(
                    user_id=uid,
                    persona_id=None,
                    persona_name=f"Grid User {age_tier} ({profile_type})",
                    persona_category=f"Grid Coverage: {age_tier}",
                    age=age,
                    gender=gender,
                    height_cm=height,
                    weight_kg=weight,
                    bmi=bmi,
                    body_fat_pct=body_fat,
                    muscle_mass_kg=muscle,
                    bmr_kcal=bmr,
                    resting_hr=rhr,
                    hrv_rmssd=hrv,
                    hrv_baseline_sd=hrv_sd,
                    activity_type=act_type,
                    workouts_per_week=w_per_wk,
                    avg_workout_duration_min=w_dur,
                    wearable_mode=wearable,
                    nutrition_logging_mode=nutr_log,
                    sleep_hours=sleep_h,
                    sleep_quality=sleep_q,
                    sleep_pattern=sleep_pat,
                    difficulty_level=diff,
                    is_out_of_distribution=False,
                    concept_drift_type=drift,
                    data_completeness_nominal=round(completeness, 2),
                    behavioral_missingness_prone=nutr_log
                    in ("chaotic", "forgets_snacks"),
                    latent_sensitivities=triggers,
                    description=f"Grid user {uid}: profile {profile_type}, {act_type}, {nutr_log}, wearable={wearable}.",
                )
            )

        return users
