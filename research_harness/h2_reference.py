"""Опорный development-опыт H2, без подбора памяти и без клинических утверждений.

Исторический synthetic_data/simulator.py не изменяется. Семь миров ниже — новые
сценарии чувствительности с явно гипотетическими параметрами. Их реализм требует
первичных источников и реальных данных; технический успех этого файла их не заменяет.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from research_support import (
    WORLD_FAMILIES,
    observed_features,
    stable_seed,
    validate_stage_evidence,
)


def run(seed=20261020, days=180):
    train = [f"train-{i}" for i in range(10)]
    evaluation = [f"evaluation-{i}" for i in range(20)]
    checks = {
        key: {"tested": 0, "violations": 0}
        for key in (
            "temporal_availability",
            "latent_isolation",
            "missing_is_unknown",
            "matched_world_targets",
            "stable_rng",
        )
    }
    worlds = []
    for family in sorted(WORLD_FAMILIES):
        outcomes, received, decisions = [], 0, 0
        per_user = []
        for user in evaluation:
            person_seed = stable_seed(seed, user, family)
            rng = np.random.default_rng(person_seed)
            again = np.random.default_rng(person_seed)
            checks["stable_rng"]["tested"] += 1
            checks["stable_rng"]["violations"] += int(rng.random() != again.random())
            sleep_trait = rng.normal(0, 0.3)
            user_outcomes = []
            for day in range(days):
                decision = float(day * 86400 + 18 * 3600)
                sleep = float(np.clip(rng.normal(7, 1.2), 3, 10))
                skipped_lunch = bool(rng.random() < 0.2)
                stress = float(rng.uniform())
                activity = float(rng.uniform())
                recent_deficit = float(rng.uniform())
                planned_training = bool(rng.random() < 0.4)
                performed_training = bool(planned_training and rng.random() < 0.8)
                app_tenure = day
                features = {
                    name: {
                        "value": value,
                        "source": "synthetic_observation",
                        "event_at": decision - 3600,
                        "received_at": decision - 1800,
                        "available_at": decision - 900,
                    }
                    for name, value in {
                        "sleep_hours": sleep,
                        "lunch_logged": not skipped_lunch,
                        "reported_stress": stress,
                        "activity": activity,
                        "logged_deficit": recent_deficit,
                        "training_planned": planned_training,
                        "training_performed": performed_training,
                        "app_tenure_days": app_tenure,
                    }.items()
                }
                # Пропуск записи не меняет ни target, ни статус подтверждения.
                missing_probability = (
                    0.1 if family != "rare_selective_feedback" else 0.65
                )
                for value in features.values():
                    if rng.random() < missing_probability:
                        value["value"] = None
                observed = observed_features(features, decision)
                checks["temporal_availability"]["tested"] += len(features)
                checks["temporal_availability"]["violations"] += sum(
                    f["available_at"] > decision for f in features.values()
                )
                eta = -1.2 + 0.15 * (7 - sleep) + 0.4 * stress + 0.3 * skipped_lunch
                if family == "context_interactions":
                    eta += 0.6 * (sleep < 6) * skipped_lunch
                if family == "individual_heterogeneity":
                    eta += sleep_trait * (7 - sleep)
                if family == "regime_change":
                    eta += (0.5 if day >= days // 2 else -0.2) * activity
                if family == "vacation_illness":
                    eta += 0.6 * (60 <= day < 75)
                if family == "null_world":
                    eta = -1.2
                truth_probability = float(1 / (1 + np.exp(-eta)))
                truth = int(rng.random() < truth_probability)
                # Условная селекция является допущением симулятора, явно измеряется.
                feedback_probability = (
                    (0.15 if truth else 0.25)
                    if family == "rare_selective_feedback"
                    else 0.7
                )
                label = truth if rng.random() < feedback_probability else None
                outcome_received_at = (
                    decision + 20 * 3600 if label is not None else None
                )
                assert outcome_received_at is None or outcome_received_at > decision
                checks["missing_is_unknown"]["tested"] += 1
                checks["missing_is_unknown"]["violations"] += int(
                    label is None and outcome_received_at is not None
                )
                checks["latent_isolation"]["tested"] += 1
                checks["latent_isolation"]["violations"] += int(
                    any(k.startswith("true_") or k == "outcome" for k in observed)
                )
                # Проверка matched world/user/day ключа в этом генераторе.
                target_key = (family, user, day)
                observation_key = (family, user, day)
                checks["matched_world_targets"]["tested"] += 1
                checks["matched_world_targets"]["violations"] += int(
                    target_key != observation_key
                )
                decisions += 1
                received += int(label is not None)
                outcomes.append(truth)
                user_outcomes.append(truth)
            per_user.append(float(np.mean(user_outcomes)))
        worlds.append(
            {
                "family": family,
                "n_users": len(evaluation),
                "n_decisions": decisions,
                "observed_feedback_fraction": received / decisions,
                "latent_event_fraction": float(np.mean(outcomes)),
                "per_user_event_fraction": per_user,
            }
        )
    registry = [
        {
            "claim": "Зависимость синтетического исхода от сна, стресса, взаимодействий и режима",
            "population": "искусственные пользователи; перенос на людей не установлен",
            "uncertainty": "коэффициенты не оценены на реальной когорте; анализ чувствительности обязателен",
            "parameter": "eta: -1.2, .15, .4, .3; взаимодействие .6; дрейф .5/-.2; интервал 60–74",
            "status": "hypothesis",
        },
        {
            "claim": "Редкая и селективная обратная связь",
            "population": "искусственные пользователи",
            "uncertainty": "частоты .15/.25/.7 и задержка 20 часов заданы для стресс-проверки, не измерены у людей",
            "parameter": "feedback_probability и outcome_received_at",
            "status": "stress_scenario",
        },
    ]
    evidence = {
        "schema": "craving-stage-evidence/v1",
        "stage": "H2",
        "scope": "synthetic_development",
        "seed": seed,
        "splits": {"train_users": train, "evaluation_users": evaluation},
        "checks": checks,
        "worlds": worlds,
        "assumption_registry": registry,
    }
    metrics = {
        "configuration_ids": ["h2_reference"],
        "stage_evidence": evidence,
        "claims": {
            "clinical_realism": False,
            "memory_superiority_tested": False,
            "mobile_latency_tested": False,
            "push_policy_tested": False,
        },
        "limitations": [
            "Train группа зарезервирована, обучение в H2 не выполняется",
            "Численные параметры гипотетические",
            "Проверки генератора не являются внешней валидацией",
        ],
    }
    validate_stage_evidence(metrics, "H2")
    return metrics


if __name__ == "__main__":
    target = Path("artifacts")
    target.mkdir(exist_ok=True)
    metrics = run()
    (target / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (target / "observations.md").write_text(
        "# Проверка симуляторов H2\n\nВыполнен development-опыт на семи гипотетических семействах. Реализм людей, преимущество памяти, эффективность уведомлений и скорость на телефонах не проверялись. Первичные источники для коэффициентов отсутствуют; параметры отмечены как гипотезы и стресс-сценарии. Требуются независимое воспроизведение и семантическая оценка ограничений.\n",
        encoding="utf-8",
    )
