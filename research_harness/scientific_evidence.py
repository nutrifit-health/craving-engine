"""Измерения для H2–H4: парные исходы, macro-оценка, интервалы по людям.

Это исследовательский контракт, а не доказательство клинической пользы.
AUC ранжирует пары исходов; Brier не имеет универсального порога значимости.
Открытие гейта не является уведомлением. Новые seeds не являются новой когортой.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

WORLD_FAMILIES = {
    "additive",
    "context_interactions",
    "individual_heterogeneity",
    "regime_change",
    "vacation_illness",
    "rare_selective_feedback",
    "null_world",
}


def stable_seed(seed, user, world):
    """В отличие от hash(), seed одинаков в разных процессах и контейнерах."""
    return int.from_bytes(
        hashlib.sha256(json.dumps([seed, user, world]).encode()).digest()[:8], "big"
    )


def observed_features(features, decision_at):
    """Изолирует значения от латентной истины; проверяет три времени каждой записи."""
    result = {}
    for name, record in features.items():
        if name.startswith("true_") or record.get("source") == "latent":
            raise ValueError("Латентный признак: " + name)
        times = [record[k] for k in ("event_at", "received_at", "available_at")]
        if any(not isinstance(t, (float, int)) or not math.isfinite(t) for t in times):
            raise ValueError("Нужны конечные времена доступности")
        if not times[0] <= times[1] <= times[2] <= decision_at:
            raise ValueError("Признак ещё недоступен: " + name)
        result[name] = record["value"]
    return result


def paired_metrics(rows, training_user_ids, *, seed=1729, bootstrap=2000):
    """Одна строка: world,user,event,model,feature_set,y,p,p0,feedback_id.

    Внутри каждого world/feature_set все модели получают ровно те же события,
    исходы, frozen p0 и feedback. Тестовые люди не входят в train. Повторная
    выборка делается по людям, сохраняя зависимость всех их наблюдений.
    """
    groups = defaultdict(lambda: defaultdict(dict))
    training = set(training_user_ids)
    for row in rows:
        if row["user"] in training:
            raise ValueError("Пересечение train/evaluation пользователей")
        if row["y"] not in (0, 1) or any(
            isinstance(row[k], bool)
            or not isinstance(row[k], (float, int))
            or not math.isfinite(row[k])
            or not 0 <= row[k] <= 1
            for k in ("p", "p0")
        ):
            raise ValueError("Некорректные исходы/вероятности")
        key = (row["user"], row["event"])
        target = groups[(row["world"], row["feature_set"])][row["model"]]
        if key in target:
            raise ValueError("Повторное событие модели")
        target[key] = row
    if not groups:
        raise ValueError("Нет наблюдений")
    result = []
    for (world, features), models in sorted(groups.items()):
        reference = next(iter(models.values()))
        for model, events in sorted(models.items()):
            if events.keys() != reference.keys() or any(
                any(events[k][f] != reference[k][f] for f in ("y", "p0", "feedback_id"))
                for k in events
            ):
                raise ValueError(
                    "Модели сравниваются на разных событиях, исходах, p0 или feedback"
                )
            users = defaultdict(list)
            for row in events.values():
                users[row["user"]].append(row)
            deltas, aucs, briers, bases = [], [], [], []
            for user in sorted(users):
                data = users[user]
                y, p, p0 = (np.array([r[k] for r in data]) for k in ("y", "p", "p0"))
                brier, base = (
                    float(np.mean((p - y) ** 2)),
                    float(np.mean((p0 - y) ** 2)),
                )
                briers.append(brier)
                bases.append(base)
                deltas.append(brier - base)
                if len(set(y)) == 2:
                    aucs.append(float(roc_auc_score(y, p)))
            if len(users) < 2:
                raise ValueError("Недостаточно независимых людей для интервала")
            rng = np.random.default_rng(stable_seed(seed, features, world))
            draws = np.mean(
                rng.choice(deltas, (bootstrap, len(users)), replace=True), axis=1
            )
            y, p = ([r[k] for r in events.values()] for k in ("y", "p"))
            null = np.mean(
                np.asarray(deltas) * rng.choice([-1, 1], (bootstrap, len(users))),
                axis=1,
            )
            p_value = float(
                (1 + np.count_nonzero(np.abs(null) >= abs(np.mean(deltas))))
                / (bootstrap + 1)
            )
            result.append(
                {
                    "world": world,
                    "feature_set": features,
                    "model": model,
                    "n_users": len(users),
                    "n_events": len(events),
                    "macro_brier": float(np.mean(briers)),
                    "macro_baseline_brier": float(np.mean(bases)),
                    "delta_macro_brier": float(np.mean(deltas)),
                    "ci95_user_bootstrap": np.quantile(draws, [0.025, 0.975]).tolist(),
                    "macro_auc": float(np.mean(aucs)) if aucs else None,
                    "auc_eligible_users": len(aucs),
                    "pooled_auc": float(roc_auc_score(y, p))
                    if len(set(y)) == 2
                    else None,
                    "per_user_delta_brier": deltas,
                    "p_user_signflip": p_value,
                    "inference": "exploratory; интервалы не учитывают адаптивный выбор из всех эпох",
                }
            )
    # Поправка Holm внутри данного конечного пакета; не исправляет повторный поиск эпох.
    order = sorted(range(len(result)), key=lambda i: result[i]["p_user_signflip"])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = max(
            adjusted, min(1.0, result[index]["p_user_signflip"] * (len(result) - rank))
        )
        result[index]["p_holm_within_packet"] = adjusted
    return result


def validate_stage_evidence(metrics, stage):
    """Детерминированная полнота измерений; семантику кода проверяет отдельный критик."""
    evidence = metrics.get("stage_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != "craving-stage-evidence/v1"
    ):
        raise ValueError(
            "Нет измеренного stage_evidence по контракту research_support.py"
        )
    if (
        evidence.get("stage") != stage
        or evidence.get("scope") != "synthetic_development"
    ):
        raise ValueError("Неверный этап или scope измерений")
    if not isinstance(evidence.get("seed"), int) or isinstance(evidence["seed"], bool):
        raise TypeError("Нужен воспроизводимый seed")
    splits = evidence.get("splits", {})
    train, evaluation = (
        splits.get("train_users", []),
        splits.get("evaluation_users", []),
    )
    if (
        not evaluation
        or len(set(evaluation)) != len(evaluation)
        or set(train) & set(evaluation)
    ):
        raise ValueError("Невалидное разделение пользователей")
    checks = evidence.get("checks", {})
    required = {
        "temporal_availability",
        "latent_isolation",
        "missing_is_unknown",
        "matched_world_targets",
        "stable_rng",
    }
    if stage == "H3":
        required |= {
            "same_features",
            "same_feedback",
            "frozen_population",
            "cold_start_reported",
            "rare_feedback_reported",
        }
    for key in required:
        value = checks.get(key)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("tested"), int)
            or isinstance(value["tested"], bool)
            or value["tested"] <= 0
            or value.get("violations") != 0
        ):
            raise ValueError("Отсутствующая/нарушенная измеренная проверка: " + key)
    worlds = evidence.get("worlds", [])
    if not WORLD_FAMILIES <= {w.get("family") for w in worlds}:
        raise ValueError("Не покрыты семь обязательных семейств")
    for world in worlds:
        if (
            world.get("n_users", 0) < 2
            or world.get("n_decisions", 0) < world["n_users"]
            or not 0 <= world.get("observed_feedback_fraction", -1) <= 1
        ):
            raise ValueError("Неполные измерения мира")
    if stage == "H2":
        registry = evidence.get("assumption_registry", [])
        if not registry:
            raise ValueError("Нет реестра допущений")
        for entry in registry:
            if not all(
                entry.get(k)
                for k in ("claim", "population", "uncertainty", "parameter", "status")
            ):
                raise ValueError("Неполное происхождение параметра")
            if entry["status"] not in {
                "retrieved_primary",
                "hypothesis",
                "stress_scenario",
            }:
                raise ValueError("Неверный статус источника")
            if entry["status"] == "retrieved_primary" and not all(
                entry.get(k) for k in ("url", "excerpt", "retrieved_at")
            ):
                raise ValueError("Ссылка без прочитанного текста не является evidence")
    if stage == "H3":
        rows = evidence.get("prediction_rows", [])
        features = evidence.get("feature_set_ids", [])
        if not 1 <= len(features) <= 6 or set(features) != {
            r.get("feature_set") for r in rows
        }:
            raise ValueError("Наборы признаков не покрыты измерениями")
        if set(evaluation) != {r.get("user") for r in rows} or set(
            metrics.get("configuration_ids", [])
        ) != {r.get("model") for r in rows}:
            raise ValueError("Не покрыты заявленные пользователи/конфигурации")
        if not WORLD_FAMILIES <= {r.get("world") for r in rows}:
            raise ValueError("Нет прогнозов обязательных миров")
        expected_models = set(metrics["configuration_ids"])
        expected_worlds = {world["family"] for world in worlds}
        grouped = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            key = (row["world"], row["feature_set"])
            event = (row["user"], row["event"])
            if event in grouped[key][row["model"]]:
                raise ValueError("Повтор события H3")
            grouped[key][row["model"]][event] = row
        if set(grouped) != {
            (world, features_id)
            for world in expected_worlds
            for features_id in features
        }:
            raise ValueError("Не покрыты комбинации мир × признаки")
        for world in expected_worlds:
            reference_events = None
            for features_id in features:
                models = grouped[(world, features_id)]
                if set(models) != expected_models:
                    raise ValueError(
                        "Не покрыты модели внутри каждого мира/набора признаков"
                    )
                events = next(iter(models.values()))
                if {user for user, _ in events} != set(evaluation):
                    raise ValueError("Разные пользователи между группами H3")
                if reference_events is not None and (
                    events.keys() != reference_events.keys()
                    or any(
                        any(
                            events[k][f] != reference_events[k][f]
                            for f in ("y", "feedback_id")
                        )
                        for k in events
                    )
                ):
                    raise ValueError(
                        "Наборы признаков сравниваются на разных событиях/исходах/feedback"
                    )
                reference_events = events
        return {
            "stage": stage,
            "recomputed_metrics": paired_metrics(rows, train, seed=evidence["seed"]),
            "semantic_review_required": True,
        }
    return {
        "stage": stage,
        "measured_checks": checks,
        "world_count": len(worlds),
        "semantic_review_required": True,
    }
