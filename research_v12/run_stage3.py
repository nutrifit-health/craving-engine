"""Скрипт выполнения Этапа 3: матрица устойчивости, MNAR и реальное измерение роста состояния."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from research_v12.memory_benchmark import measure_memory_stream
from research_v12.models import HierarchicalRidgeMemory, TunedScalarMemory
from research_v12.robustness import (
    run_scenario_evaluation,
    transform_feedback_stream,
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(*, data_dir: Path, stage2_dir: Path, output_dir: Path) -> None:
    print(
        "=== ЭТАП 3: АНАЛИЗ УСТОЙЧИВОСТИ И ГРАНИЦЫ РЕЗУЛЬТАТА (STRESS MATRIX) ===",
        flush=True,
    )
    t0 = time.time()

    output_dir.mkdir(parents=True, exist_ok=False)
    v12_dir = data_dir
    stage2_models_dir = stage2_dir / "models"

    # Загружаем непосредственно обученные модели из Этапа 2
    from research_v11.gbdt import PopulationGBDTTrainer

    gbdt = PopulationGBDTTrainer.load(stage2_models_dir / "gbdt_population.pickle")
    with (stage2_models_dir / "supervised_bottleneck_encoders.pickle").open("rb") as f:
        sb_encoders = pickle.load(f)
    with (stage2_models_dir / "tuned_scalar_config.json").open(
        "r", encoding="utf-8"
    ) as f:
        scalar_cfg = json.load(f)

    selection = json.loads((stage2_dir / "search_summary.json").read_text())
    candidate = selection["best_candidate"]
    candidate_cfg = candidate["config"]
    if candidate["family"] != "family_2_supervised_bottleneck":
        raise ValueError(
            "Этот стресс-протокол зафиксирован для выбранного supervised-кандидата V12"
        )
    dimension = candidate_cfg["dimension"]
    sb_encoder = sb_encoders[dimension]

    # Загружаем данные dev_1 для стресс-тестирования
    X_dev1 = np.load(v12_dir / "dev_1.X.npy")
    dev1_records = _read_jsonl(v12_dir / "dev_1.records.jsonl")
    dev1_eval = _read_jsonl(v12_dir / "dev_1.evaluation.jsonl")
    p0_dev1 = gbdt.predict_proba(X_dev1)
    Z_dev1 = sb_encoder.transform(X_dev1, d=dimension)

    scenarios = [
        "default",
        "observability_50",
        "observability_10",
        "delay_72h",
        "delay_mixed",
        "label_noise_15",
        "abrupt_drift",
        "temporary_drift",
        "mnar_forward",
        "mnar_reverse",
        "pseudolabel_impute_zero",
        "pseudolabel_impute_one",
    ]

    scenario_results: dict[str, Any] = {}
    print("\n--- Запуск матрицы сценариев устойчивости ---", flush=True)
    for sc in scenarios:
        rec_sc, eval_sc = transform_feedback_stream(
            dev1_records, dev1_eval, scenario=sc, seed=20261203
        )
        res = run_scenario_evaluation(
            rec_sc, eval_sc, p0_dev1, Z_dev1, scalar_cfg, candidate_cfg
        )
        scenario_results[sc] = res
        print(
            f"Scenario [{sc:24s}]: Pop={res['pop_brier']:.6f} | "
            f"Scalar={res['scalar_brier']:.6f} | Cand={res['cand_brier']:.6f} | "
            f"Δ(Cand - Scal)={res['diff_cand_minus_scalar']:+.6f} | "
            f"ScalarAUC={res['scalar_auc']:.4f} | CandAUC={res['cand_auc']:.4f}",
            flush=True,
        )

    memory_results = {
        "tuned_scalar": measure_memory_stream(TunedScalarMemory(**scalar_cfg)),
        "candidate": measure_memory_stream(HierarchicalRidgeMemory(**candidate_cfg)),
    }
    # Полные траектории измерений сохраняются; максимум не подменяется последним днём.
    (output_dir / "memory_measurements.json").write_text(
        json.dumps(memory_results, indent=2)
    )

    brier_clean = scenario_results["default"]["scalar_brier"]
    brier_impute0 = scenario_results["pseudolabel_impute_zero"]["scalar_brier"]
    brier_impute1 = scenario_results["pseudolabel_impute_one"]["scalar_brier"]
    diff_impute0 = brier_impute0 - brier_clean
    diff_impute1 = brier_impute1 - brier_clean

    print("\n--- Эмпирическое влияние псевдоразметки ---")
    print(f"Brier чистый (игнорирование пропусков): {brier_clean:.6f}")
    print(
        f"Brier при подстановке 0 (нет ответа = успех): {brier_impute0:.6f} (Δ={diff_impute0:+.6f})"
    )
    print(
        f"Brier при подстановке 1 (нет ответа = срыв): {brier_impute1:.6f} (Δ={diff_impute1:+.6f})"
    )

    # Математический анализ MNAR
    mnar_analysis = {
        "identifiability_statement": (
            "Под режимом MNAR (Missing Not At Random), когда факт наличия отчета R зависит от ненаблюдаемого "
            "исхода Y (P(R=1 | X, Y) != P(R=1 | X)), маргинальное распределение P(Y=1 | X) является строго "
            "неидентифицируемым без внешних неверифицируемых допущений. "
            "Точные границы Манского (Manski bounds): "
            "[P(Y=1, R=1 | X), P(Y=1, R=1 | X) + P(R=0 | X)]. "
            "Ширина границ вероятности не сравнивается напрямую с разностью Brier. "
            "Использование оракульной пропенсити P(R=1 | X, Y) в рабочей модели невозможно, так как истинный Y "
            "неизвестен в момент предсказания и пропущенных ответов. "
            f"Разность Brier на данном сценарии: подстановка 0 {diff_impute0:+.4f}, "
            f"подстановка 1 {diff_impute1:+.4f}. Эти результаты не доказывают вред любой псевдоразметки."
        ),
        "pseudolabel_comparison": {
            "brier_clean_protocol": brier_clean,
            "brier_impute_zero": brier_impute0,
            "delta_impute_zero": diff_impute0,
            "brier_impute_one": brier_impute1,
            "delta_impute_one": diff_impute1,
        },
        "empirical_findings": {
            "mnar_forward_scalar_brier": scenario_results["mnar_forward"][
                "scalar_brier"
            ],
            "mnar_forward_cand_brier": scenario_results["mnar_forward"]["cand_brier"],
            "mnar_reverse_scalar_brier": scenario_results["mnar_reverse"][
                "scalar_brier"
            ],
            "mnar_reverse_cand_brier": scenario_results["mnar_reverse"]["cand_brier"],
        },
    }

    summary = {
        "created_at": time.time(),
        "elapsed_seconds": time.time() - t0,
        "scenarios": scenario_results,
        "state_size_evaluation": {
            name: {k: v for k, v in result.items() if k != "measurements"}
            for name, result in memory_results.items()
        },
        "mnar_analysis": mnar_analysis,
    }

    with (output_dir / "robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\nЭтап 3 завершен. Результаты: {output_dir / 'robustness_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(data_dir=args.data_dir, stage2_dir=args.stage2_dir, output_dir=args.output_dir)
