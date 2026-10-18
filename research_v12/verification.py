"""Повторное получение прогнозов, метрик и пиков памяти из комплекта V12."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from research_v11.gbdt import PopulationGBDTTrainer
from research_v12.memory_benchmark import measure_memory_stream
from research_v12.models import HierarchicalRidgeMemory, TunedScalarMemory
from research_v12.registry import file_sha256
from research_v12.replay import evaluate_saved, read_jsonl
from research_v12.robustness import run_scenario_evaluation, transform_feedback_stream


def verify_manifest(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("format_version") != 2:
        raise ValueError("Для replay нужен комплект исправленной версии 2")
    for item in manifest["files"]:
        path = (bundle / item["path"]).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("Путь вне комплекта")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Изменён файл: {item['path']}")


def compare_scores(expected: dict, actual: dict) -> None:
    if expected.keys() != actual.keys():
        raise AssertionError("Не совпадает набор моделей")
    for family, cohorts in expected.items():
        if cohorts.keys() != actual[family].keys():
            raise AssertionError("Не совпадает набор когорт")
        for split, rows in cohorts.items():

            def indexed(values):
                result = {(r["user_id"], r["event_id"]): r for r in values}
                if len(result) != len(values):
                    raise AssertionError("Повтор event_id")
                return result

            left, right = indexed(rows), indexed(actual[family][split])
            if left.keys() != right.keys():
                raise AssertionError("Не совпадает набор решений")
            for key, row in left.items():
                other = right[key]
                for field in (
                    "evaluation_outcome",
                    "feedbacks_processed",
                    "decision_at",
                ):
                    if row.get(field) != other.get(field):
                        raise AssertionError(f"Изменён {field}: {family}/{split}/{key}")
                for field in ("p_final", "p_population", "p_learning", "gamma"):
                    if field in row or field in other:
                        np.testing.assert_allclose(
                            row[field], other[field], atol=1e-12, rtol=0
                        )


def reproduce_bundle(bundle: Path) -> dict:
    # Проверка до загрузки любых pickle.
    verify_manifest(bundle)
    expected = json.loads((bundle / "search_summary.json").read_text())
    summary, scores = evaluate_saved(
        bundle / "data", bundle / "models", expected, bundle / "frozen_v11_config.json"
    )
    compare_scores(json.loads((bundle / "dev_scores_all.json").read_text()), scores)
    if summary["replay_metrics"] != expected["replay_metrics"]:
        raise AssertionError("Повторные метрики не совпали")
    measurements = json.loads((bundle / "memory_measurements.json").read_text())
    for name, cls in (
        ("tuned_scalar", TunedScalarMemory),
        ("candidate", HierarchicalRidgeMemory),
    ):
        old = measurements[name]
        result = measure_memory_stream(
            cls(**old["config"]), days=old["days"], seed=old["seed"]
        )
        if result != old:
            raise AssertionError(f"Не совпала траектория измерения памяти: {name}")
    # Повторение стресс-сценариев теми же сохранёнными моделями.
    robustness = json.loads((bundle / "robustness_summary.json").read_text())
    x = np.load(bundle / "data/dev_1.X.npy")
    records = read_jsonl(bundle / "data/dev_1.records.jsonl")
    evaluations = read_jsonl(bundle / "data/dev_1.evaluation.jsonl")
    gbdt = PopulationGBDTTrainer.load(bundle / "models/gbdt_population.pickle")
    with (bundle / "models/supervised_bottleneck_encoders.pickle").open("rb") as f:
        encoders = pickle.load(f)
    cfg = expected["best_candidate"]["config"]
    z = encoders[cfg["dimension"]].transform(x, d=cfg["dimension"])
    p0 = gbdt.predict_proba(x)
    scalar_cfg = json.loads((bundle / "models/tuned_scalar_config.json").read_text())
    for scenario, old in robustness["scenarios"].items():
        rows, truth = transform_feedback_stream(
            records, evaluations, scenario, seed=20261203
        )
        new = run_scenario_evaluation(rows, truth, p0, z, scalar_cfg, cfg)
        if new != old:
            raise AssertionError(f"Не совпал стресс-сценарий: {scenario}")
    return {
        "scores_replayed": True,
        "metrics_recomputed": True,
        "memory_replayed": True,
        "stress_replayed": True,
        "confirmation_accessed": False,
    }
