"""Контролируемые синтетические задачи; не симуляция пищевого поведения.

Общий p0 намеренно известен точно. Скрытые персональные коэффициенты создают
возможность адаптации, но не передаются модели. Одинаковые случайные числа
между сценариями позволяют изучать только изменение feedback или режима.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
from scipy.special import expit

from research_v10.spiking import normalize
from research_v10.types import SCENARIOS


def generate(
    seed: int,
    scenario: str,
    users: int = 40,
    days: int = 120,
    split: str = "development",
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    if scenario not in SCENARIOS:
        raise ValueError("Неизвестный сценарий")
    rng = np.random.default_rng(seed)
    contexts = rng.normal(size=(users, days, 16))
    # Умеренная временная зависимость без использования будущих контекстов.
    for day in range(1, days):
        contexts[:, day] = 0.4 * contexts[:, day - 1] + np.sqrt(0.84) * contexts[:, day]
    offsets = rng.normal(0, 0.6, size=users)
    personal = rng.normal(size=(users, 16))
    personal /= np.linalg.norm(personal, axis=1, keepdims=True)
    uniforms = rng.random((users, days))
    reporting = rng.random((users, days))
    noise = rng.random((users, days))
    global_weights = np.linspace(-0.35, 0.35, 16)
    base_logits = contexts @ global_weights - 0.4
    change_day = 60
    shift = np.zeros(days)
    if scenario != "stable":
        shift[change_day:] = 1
    if scenario == "gradual":
        shift[change_day:] = np.minimum(1, np.arange(days - change_day) / 30)
    if scenario == "transient":
        shift[change_day + 3 :] = 0
    direction = np.where(offsets >= 0, 1, -1)
    correction = offsets[:, None] + 0.4 * np.einsum("udi,ui->ud", contexts, personal)
    correction += shift[None, :] * (
        1.2 * direction[:, None] - 1.2 * np.einsum("udi,ui->ud", contexts, personal)
    )
    true_p = expit(base_logits + correction)
    y = uniforms < true_p
    observed = reporting < {"report50": 0.5, "report10": 0.1}.get(scenario, 0.8)
    observed_y = np.logical_xor(y, noise < (0.15 if scenario == "noise15" else 0))
    records = []
    for user in range(users):
        for day in range(days):
            at = datetime(2026, 1, 1, 16, 30, tzinfo=UTC) + timedelta(days=day)
            records.append(
                {
                    "user_id": f"V10-{seed}-{user:03d}",
                    "event_id": f"{seed}:{scenario}:{user}:{day}",
                    "decision_at": at,
                    "available_at": at,
                    "split": split,
                    "change_day": change_day,
                    "feedback_at": at
                    + timedelta(hours=72 if scenario == "delay72" else 20)
                    if observed[user, day]
                    else None,
                    "outcome": int(observed_y[user, day])
                    if observed[user, day]
                    else None,
                    "evaluation_outcome": int(y[user, day]),
                    "evaluation_probability": float(true_p[user, day]),
                    "confidence": 1.0,
                }
            )
    z = contexts.reshape(-1, 16)
    codes = normalize(
        np.column_stack((np.ones(len(z)), np.maximum(z, 0), np.maximum(-z, 0)))
    )
    return records, codes, expit(base_logits).ravel()
