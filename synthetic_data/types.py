"""Контракты синтетических наблюдений, скрытого состояния и доступного feedback.

Скрытые параметры и истинные исходы предназначены только генератору и оценщику.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    LATE_SNACK = "late_snack"
    PRE_WORKOUT = "pre_workout"
    POST_WORKOUT = "post_workout"


class WorkoutIntensity(str, Enum):
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    EXTREME = "extreme"


class MealRecord(BaseModel):
    """Наблюдаемая запись приёма пищи в симуляторе."""

    eatenAt: dt_datetime = Field(..., description="Время приёма пищи")
    availableAt: dt_datetime | None = Field(
        default=None, description="Время доступности записи приложению"
    )
    mealType: MealType = Field(default=MealType.LUNCH, description="Тип приёма пищи")
    calories: float = Field(..., ge=0, description="Энергия, ккал")
    proteins: float = Field(..., ge=0, description="Белки, г")
    fats: float = Field(..., ge=0, description="Жиры, г")
    carbs: float = Field(..., ge=0, description="Углеводы, г")
    fiber: float = Field(default=0.0, ge=0, description="Клетчатка, г")
    sugars: float = Field(default=0.0, ge=0, description="Сахара, г")
    addedSugar: float = Field(default=0.0, ge=0, description="Добавленные сахара, г")
    alcohol: float = Field(default=0.0, ge=0, description="Алкоголь, г")
    gi: float | None = Field(
        default=None, ge=0, le=100, description="Гликемический индекс, 0–100"
    )
    gl: float | None = Field(default=None, ge=0, description="Гликемическая нагрузка")

    @field_validator("gl", mode="before")
    @classmethod
    def calculate_gl_if_missing(
        cls, v: float | None, info: ValidationInfo
    ) -> float | None:
        """Рассчитывает гликемическую нагрузку, если она не задана."""
        if v is not None:
            return v
        data = info.data
        carbs = data.get("carbs", 0.0)
        gi = data.get("gi")
        if gi is not None and carbs is not None:
            return round((gi * carbs) / 100.0, 2)
        return None


class WorkoutRecord(BaseModel):
    """Наблюдаемая запись тренировки в симуляторе."""

    startedAt: dt_datetime = Field(..., description="Начало тренировки")
    availableAt: dt_datetime | None = Field(
        default=None, description="Время доступности завершённой тренировки"
    )
    durationMinutes: float = Field(..., gt=0, description="Длительность, мин")
    sportKey: str = Field(
        default="strength_training", description="Идентификатор дисциплины"
    )
    intensity: WorkoutIntensity = Field(default=WorkoutIntensity.MODERATE)
    caloriesBurned: float = Field(..., ge=0, description="Оценка расхода энергии, ккал")
    sessionRpe: float = Field(
        ..., ge=1, le=10, description="Субъективная нагрузка RPE, 1–10"
    )
    fatigueScore: float = Field(
        default=5.0, ge=0, le=100, description="Субъективная усталость"
    )
    volumeLoadKg: float | None = Field(
        default=0.0, ge=0, description="Объём силовой нагрузки, кг"
    )


class SleepRecord(BaseModel):
    """Наблюдаемая запись сна в симуляторе."""

    bedtime: dt_datetime = Field(..., description="Время отхода ко сну")
    wakeTime: dt_datetime = Field(..., description="Время пробуждения")
    availableAt: dt_datetime | None = Field(
        default=None, description="Время доступности записи сна"
    )
    sleepHours: float = Field(..., ge=0, le=24, description="Продолжительность сна, ч")
    sleepQuality: float = Field(..., ge=0, le=100, description="Качество сна, 0–100")
    deepSleepMinutes: float | None = Field(
        default=None, ge=0, description="Глубокий сон, мин"
    )


class BodyMeasurementRecord(BaseModel):
    """Наблюдаемый замер состава тела в симуляторе."""

    measuredAt: dt_datetime = Field(default_factory=dt_datetime.now)
    availableAt: dt_datetime | None = Field(
        default=None, description="Время доступности замера"
    )
    weightKg: float = Field(..., gt=20, le=300, description="Масса тела, кг")
    bodyFatPercent: float | None = Field(
        default=None, ge=3, le=60, description="Жировая масса, %"
    )
    muscleMassKg: float | None = Field(
        default=None, gt=10, le=150, description="Мышечная масса, кг"
    )
    bmrKcal: float | None = Field(
        default=None, ge=800, le=4000, description="Основной обмен, ккал"
    )
    bmi: float | None = Field(
        default=None, ge=10, le=60, description="Индекс массы тела"
    )
    gender: str = Field(default="male", description="'male' or 'female'")


class WellbeingRecord(BaseModel):
    """Наблюдаемый чек-ин самочувствия в симуляторе."""

    recordedAt: dt_datetime = Field(default_factory=dt_datetime.now)
    availableAt: dt_datetime | None = Field(
        default=None, description="Время доступности чек-ина"
    )
    energyScore: float = Field(..., ge=0, le=100, description="Энергия, 0–100")
    moodScore: float = Field(..., ge=0, le=100, description="Настроение, 0–100")
    stressLevel: float = Field(..., ge=0, le=100, description="Стресс, 0–100")


class HealthHabitsRecord(BaseModel):
    """Наблюдаемый снимок привычек в симуляторе."""

    date: dt_date = Field(default_factory=dt_date.today)
    waterMl: float = Field(default=0.0, ge=0, description="Выпитая вода, мл")
    targetWaterMl: float = Field(default=2500.0, gt=0, description="Цель воды, мл")
    habitsCompleted: int = Field(
        default=0, ge=0, description="Число выполненных привычек"
    )
    habitsTotal: int = Field(
        default=5, ge=1, description="Число запланированных привычек"
    )
    currentStreakDays: int = Field(default=0, ge=0, description="Серия выполнения, дни")


class UserHealthContext(BaseModel):
    """Наблюдаемый контекст истории; это внутренний формат симулятора, не API NutriFit."""

    userId: str = Field(..., description="Идентификатор синтетического пользователя")
    evaluationTime: dt_datetime = Field(
        default_factory=dt_datetime.now, description="Время решения"
    )
    meals: list[MealRecord] = Field(
        default_factory=list, description="Наблюдаемые приёмы пищи"
    )
    workouts: list[WorkoutRecord] = Field(
        default_factory=list, description="Наблюдаемые тренировки"
    )
    sleep: SleepRecord | None = Field(default=None, description="Последняя запись сна")
    recentSleepHistory: list[SleepRecord] = Field(
        default_factory=list, description="История сна"
    )
    bodyMeasurement: BodyMeasurementRecord | None = Field(
        default=None, description="Последний замер тела"
    )
    wellbeing: WellbeingRecord | None = Field(
        default=None, description="Последний чек-ин"
    )
    habits: HealthHabitsRecord | None = Field(
        default=None, description="Снимок привычек за день"
    )


@dataclass
class SyntheticUser:
    """Полный профиль синтетического пользователя с явным и скрытым (latent) состоянием."""

    user_id: str
    persona_id: str | None
    persona_name: str
    persona_category: str
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    bmi: float
    body_fat_pct: float
    muscle_mass_kg: float
    bmr_kcal: float
    resting_hr: int
    hrv_rmssd: float
    hrv_baseline_sd: float
    activity_type: str
    workouts_per_week: int
    avg_workout_duration_min: int
    wearable_mode: str
    nutrition_logging_mode: str
    sleep_hours: float
    sleep_quality: float
    sleep_pattern: str
    difficulty_level: int
    is_out_of_distribution: bool
    concept_drift_type: str
    data_completeness_nominal: float
    behavioral_missingness_prone: bool

    latent_sensitivities: dict[str, float] = field(default_factory=dict)

    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrueDailyState:
    """Объективное истинное состояние физиологии и поведения человека в день t."""

    day_index: int  # 1 .. 180
    date: date
    is_weekend: bool
    is_vacation: bool
    # Истинные физиологические параметры
    true_sleep_hours: float
    true_sleep_quality: float
    true_weight_kg: float
    true_body_fat_pct: float
    true_resting_hr: int
    true_hrv_rmssd: float
    true_stress_level: float  # 0 .. 100
    true_energy_level: float  # 0 .. 100
    true_fasting_hours_at_eval: float
    true_glycogen_deficit: float  # 0.0 .. 1.0
    true_sugar_intake_g: float
    true_alcohol_units: float
    true_water_deficit_ml: float
    workout_performed: bool
    workout_intensity: str
    workout_calories: float
    # Истинный результат Ground Truth
    true_relapse_prob: float  # 0.0 .. 1.0
    relapse_occurred: bool  # True / False
    craving_intensity: int  # 1 .. 10
    peak_risk_hour: float  # 0.0 .. 24.0


@dataclass(frozen=True)
class InformationFeature:
    """Отсутствующее значение равно нулю и имеет отдельный индикатор наличия.

    available_if ссылается на столбец текущего представления. Исторические
    средние исключают значения, для которых этот индикатор равен нулю.
    """

    name: str
    unit: str
    semantics: str
    available_if: str | None = None


@dataclass(frozen=True)
class FeedbackRecord:
    """Прежний исход становится доступен признакам только после feedback_at."""

    split: str
    user_id: str
    event_id: str
    decision_at: datetime
    available_at: datetime
    outcome: int | None
    feedback_at: datetime | None
