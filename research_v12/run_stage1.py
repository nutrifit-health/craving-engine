"""Скрипт выполнения Этапа 1: диагностические абляции, бенчмарки и оракулы."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from research_v11.encoders import CompactLinearEncoder
from research_v11.gbdt import PopulationGBDTTrainer
from research_v11.metrics import compute_comprehensive_metrics
from research_v11.runner import run_v11_replay
from research_v11.types import MemoryConfigV11
from research_v12.diagnostics import (
    generate_intercept_only_benchmark,
    generate_known_contextual_benchmark,
    run_ablation_experiment,
    run_question_d_pca_analysis,
)
from research_v12.registry import RunRegistry

OUTPUT_DIR = Path("runs/v12-stage1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
registry = RunRegistry()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    print("=== ЗАПУСК ЭТАПА 1: ДИАГНОСТИЧЕСКИЕ АБЛЯЦИИ И БЕНЧМАРКИ ===", flush=True)
    t_start = time.time()
    registry.register(
        run_id="run-stage1-diagnostics",
        stage=1,
        family="diagnostics_and_benchmarks",
        config={
            "ablations": [
                "baseline",
                "no_gate",
                "no_familiarity",
                "no_gate_no_familiarity",
                "no_clipping",
            ],
            "benchmarks": ["intercept_only", "known_contextual_effect"],
        },
        notes="Stage 1: Full diagnostic ablations, Question A-E quantitative analysis, benchmarks and oracles",
    )
    registry.mark_running("run-stage1-diagnostics")

    # 1. Загрузка данных V11 holdout и V12 training / dev_1
    print("\n--- 1. Диагностика на историческом корпусе V11 holdout ---", flush=True)
    v11_dir = Path("runs/v11-data")
    v11_records = _read_jsonl(v11_dir / "holdout.records.jsonl")
    v11_eval = _read_jsonl(v11_dir / "holdout.evaluation.jsonl")
    v11_X = np.load(v11_dir / "holdout.X.npy")
    v11_p0 = np.load("runs/v11-closure-20261016/p0_holdout.npy")

    # Получение z для V11 holdout (linear encoder из closure)
    import pickle

    with open("runs/v11-closure-20261016/models/encoders.pickle", "rb") as f:
        v11_encoders = pickle.load(f)
    v11_z8 = v11_encoders["compact_linear"].transform(v11_X, d=8)

    frozen_v11_cfg = MemoryConfigV11(
        fast_rate=0.2,
        slow_rate=0.05,
        fast_half_life_days=14.0,
        slow_half_life_days=90.0,
        exposure_half_life_days=30.0,
        l2=0.01,
        max_logit_residual=0.7,
        gate_window_days=60.0,
        gate_min_effective_samples=12.0,
        gate_min_class_weight=2.0,
        minimum_gain=0.005,
        uncertainty_penalty=1.64,
        gate_temperature=0.008,
    )

    ablations = [
        "baseline",
        "no_gate",
        "no_familiarity",
        "no_gate_no_familiarity",
        "no_clipping",
    ]
    v11_ablation_results = {}

    for abl in ablations:
        res_scalar = run_ablation_experiment(
            v11_records,
            v11_eval,
            v11_p0,
            None,
            dimension=0,
            base_config=frozen_v11_cfg,
            ablation_type=abl,
        )
        res_linear = run_ablation_experiment(
            v11_records,
            v11_eval,
            v11_p0,
            v11_z8,
            dimension=8,
            base_config=frozen_v11_cfg,
            ablation_type=abl,
        )
        v11_ablation_results[abl] = {
            "scalar_brier": res_scalar["metrics"]["macro_brier"],
            "linear_brier": res_linear["metrics"]["macro_brier"],
            "diff_linear_minus_scalar": res_linear["metrics"]["macro_brier"]
            - res_scalar["metrics"]["macro_brier"],
            "scalar_zero_gamma_pct": res_scalar["zero_gamma_pct"],
            "linear_zero_gamma_pct": res_linear["zero_gamma_pct"],
            "linear_clipped_pct": res_linear["clipped_pct"],
        }
        print(
            f"V11 Holdout [{abl}]: Scalar Brier={res_scalar['metrics']['macro_brier']:.6f} | "
            f"Linear Brier={res_linear['metrics']['macro_brier']:.6f} | "
            f"Δ={res_linear['metrics']['macro_brier'] - res_scalar['metrics']['macro_brier']:+.6f} | "
            f"ZeroGamma={res_linear['zero_gamma_pct']:.1f}% | Clipped={res_linear['clipped_pct']:.1f}%",
            flush=True,
        )

    # 2. Обучение базовой GBDT на V12 training и диагностика на V12 dev_1
    print("\n--- 2. Диагностика на когорте V12 dev_1 ---", flush=True)
    v12_dir = Path("runs/v12-data")
    X_train = np.load(v12_dir / "training.X.npy")
    train_records = _read_jsonl(v12_dir / "training.records.jsonl")
    train_eval = _read_jsonl(v12_dir / "training.evaluation.jsonl")
    observed_mask = np.asarray(
        [r["outcome"] is not None for r in train_records], dtype=bool
    )
    y_train = np.asarray(
        [r["outcome"] if r["outcome"] is not None else 0 for r in train_records],
        dtype=int,
    )
    y_eval_train = np.asarray([r["evaluation_outcome"] for r in train_eval], dtype=int)

    gbdt = PopulationGBDTTrainer(random_state=20261201).fit(
        X_train, y_train, observed_mask
    )
    p0_train = gbdt.predict_proba(X_train)

    # Обучение CompactLinearEncoder на training.X
    linear_enc = CompactLinearEncoder(max_components=8, random_state=20261201).fit(
        X_train
    )

    X_dev1 = np.load(v12_dir / "dev_1.X.npy")
    dev1_records = _read_jsonl(v12_dir / "dev_1.records.jsonl")
    dev1_eval = _read_jsonl(v12_dir / "dev_1.evaluation.jsonl")
    p0_dev1 = gbdt.predict_proba(X_dev1)
    z8_dev1 = linear_enc.transform(X_dev1, d=8)

    v12_ablation_results = {}
    for abl in ablations:
        res_scalar = run_ablation_experiment(
            dev1_records,
            dev1_eval,
            p0_dev1,
            None,
            dimension=0,
            base_config=frozen_v11_cfg,
            ablation_type=abl,
        )
        res_linear = run_ablation_experiment(
            dev1_records,
            dev1_eval,
            p0_dev1,
            z8_dev1,
            dimension=8,
            base_config=frozen_v11_cfg,
            ablation_type=abl,
        )
        v12_ablation_results[abl] = {
            "scalar_brier": res_scalar["metrics"]["macro_brier"],
            "linear_brier": res_linear["metrics"]["macro_brier"],
            "diff_linear_minus_scalar": res_linear["metrics"]["macro_brier"]
            - res_scalar["metrics"]["macro_brier"],
            "scalar_zero_gamma_pct": res_scalar["zero_gamma_pct"],
            "linear_zero_gamma_pct": res_linear["zero_gamma_pct"],
            "linear_clipped_pct": res_linear["clipped_pct"],
        }
        print(
            f"V12 Dev_1 [{abl}]: Scalar Brier={res_scalar['metrics']['macro_brier']:.6f} | "
            f"Linear Brier={res_linear['metrics']['macro_brier']:.6f} | "
            f"Δ={res_linear['metrics']['macro_brier'] - res_scalar['metrics']['macro_brier']:+.6f} | "
            f"ZeroGamma={res_linear['zero_gamma_pct']:.1f}% | Clipped={res_linear['clipped_pct']:.1f}%",
            flush=True,
        )

    # 3. Анализ Вопроса D: PCA vs Residual Signal
    print(
        "\n--- 3. Анализ Вопроса D: Информативность PCA vs Residual Signal ---",
        flush=True,
    )
    question_d_res = run_question_d_pca_analysis(X_train, y_eval_train, p0_train)
    print(
        f"Общая дисперсия X, объясненная 8 компонентами PCA: {question_d_res['total_pca_variance_8']:.4f}"
    )
    print(
        f"R^2 остатка y - p0 на 8 компонентах PCA: {question_d_res['residual_r2_pca']:.6f}"
    )
    print(
        f"R^2 остатка y - p0 на исходных нормализованных признаках: {question_d_res['residual_r2_raw_features']:.6f}"
    )
    print(
        f"Корреляции компонент PCA с остатком: {[round(c, 4) for c in question_d_res['pca_component_residual_correlations']]}"
    )

    # 4. Бенчмарк 1: Intercept-only (Отрицательный контекстный контроль)
    print("\n--- 4. Бенчмарк 1: Intercept-Only Control ---", flush=True)
    X_into, rec_into, eval_into, p0_into = generate_intercept_only_benchmark(
        n_users=40, seed=20261208
    )
    into_scalar, _ = run_v11_replay(
        rec_into, eval_into, p0_into, None, family="scalar", config=frozen_v11_cfg
    )
    m_into_scalar = compute_comprehensive_metrics(into_scalar)

    # Линейный адаптер на 8 признаках X_into
    into_linear, _ = run_v11_replay(
        rec_into,
        eval_into,
        p0_into,
        X_into,
        family="compact_linear",
        dimension=8,
        config=frozen_v11_cfg,
    )
    m_into_linear = compute_comprehensive_metrics(into_linear)
    pop_brier_into = float(
        np.mean(
            [
                (float(p0_into[i]) - int(eval_into[i]["evaluation_outcome"])) ** 2
                for i in range(len(p0_into))
            ]
        )
    )

    diff_into = m_into_linear["macro_brier"] - m_into_scalar["macro_brier"]
    print(f"Intercept-Only: Population Brier={pop_brier_into:.6f}")
    print(f"Intercept-Only: Scalar Brier={m_into_scalar['macro_brier']:.6f}")
    print(f"Intercept-Only: Linear Brier={m_into_linear['macro_brier']:.6f}")
    print(
        f"Intercept-Only: Linear - Scalar = {diff_into:+.6f} (должно быть >= 0 или ~ 0, без ложного выигрыша)"
    )

    # 5. Бенчмарк 2: Known Contextual Effect (Положительный контекстный контроль)
    print("\n--- 5. Бенчмарк 2: Known Contextual Effect Control ---", flush=True)
    X_know, rec_know, eval_know, p0_know = generate_known_contextual_benchmark(
        n_users=40, seed=20261209, effect_magnitude=0.8
    )
    # Тестируем открытую линейную адаптацию (no_gate_no_familiarity) vs scalar на положительном контроле
    know_cfg = MemoryConfigV11(
        fast_rate=0.1, slow_rate=0.025, l2=0.01, max_logit_residual=2.0
    )
    res_know_scalar = run_ablation_experiment(
        rec_know,
        eval_know,
        p0_know,
        None,
        dimension=0,
        base_config=know_cfg,
        ablation_type="no_gate",
    )
    res_know_linear = run_ablation_experiment(
        rec_know,
        eval_know,
        p0_know,
        X_know,
        dimension=8,
        base_config=know_cfg,
        ablation_type="no_gate",
    )
    pop_brier_know = float(
        np.mean(
            [
                (float(p0_know[i]) - int(eval_know[i]["evaluation_outcome"])) ** 2
                for i in range(len(p0_know))
            ]
        )
    )
    diff_know = (
        res_know_linear["metrics"]["macro_brier"]
        - res_know_scalar["metrics"]["macro_brier"]
    )

    print(f"Known Effect: Population Brier={pop_brier_know:.6f}")
    print(f"Known Effect: Scalar Brier={res_know_scalar['metrics']['macro_brier']:.6f}")
    print(f"Known Effect: Linear Brier={res_know_linear['metrics']['macro_brier']:.6f}")
    print(
        f"Known Effect: Linear - Scalar = {diff_know:+.6f} (должно быть отрицательным, т.е. выигрыш контекста)"
    )

    # 6. Оракулы
    print("\n--- 6. Диагностика Оракулов на V12 dev_1 ---", flush=True)
    true_probs_dev1 = np.asarray([e["evaluation_probability"] for e in dev1_eval])
    y_dev1 = np.asarray([e["evaluation_outcome"] for e in dev1_eval])
    oracle_true_brier = float(np.mean((true_probs_dev1 - y_dev1) ** 2))
    pop_brier_dev1 = float(np.mean((p0_dev1 - y_dev1) ** 2))

    # Retrospective Ridge на dev_1
    by_user_dev1 = {}
    for i, r in enumerate(dev1_records):
        by_user_dev1.setdefault(r["user_id"], []).append(i)

    retro_briers = []
    for idxs in by_user_dev1.values():
        yu = y_dev1[idxs]
        Xu = X_dev1[idxs]
        Xu_norm = (Xu - np.mean(Xu, axis=0)) / (np.std(Xu, axis=0) + 1e-4)
        pred = np.clip(Ridge(alpha=10.0).fit(Xu_norm, yu).predict(Xu_norm), 0.01, 0.99)
        retro_briers.append(float(np.mean((pred - yu) ** 2)))
    oracle_retro_brier = float(np.mean(retro_briers))

    print(f"Population Brier on Dev_1: {pop_brier_dev1:.6f}")
    print(f"Oracle True Prob Brier:    {oracle_true_brier:.6f}")
    print(f"Oracle Retrospective Brier:{oracle_retro_brier:.6f}")

    # Сохранение полного резюме диагностики
    diagnostic_summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "v11_holdout_ablations": v11_ablation_results,
        "v12_dev1_ablations": v12_ablation_results,
        "question_d_pca": question_d_res,
        "benchmark_intercept_only": {
            "pop_brier": pop_brier_into,
            "scalar_brier": m_into_scalar["macro_brier"],
            "linear_brier": m_into_linear["macro_brier"],
            "diff_linear_minus_scalar": diff_into,
        },
        "benchmark_known_contextual_effect": {
            "pop_brier": pop_brier_know,
            "scalar_brier": res_know_scalar["metrics"]["macro_brier"],
            "linear_brier": res_know_linear["metrics"]["macro_brier"],
            "diff_linear_minus_scalar": diff_know,
        },
        "oracles_dev1": {
            "population_brier": pop_brier_dev1,
            "oracle_true_prob_brier": oracle_true_brier,
            "oracle_retrospective_fit_brier": oracle_retro_brier,
        },
    }

    summary_file = OUTPUT_DIR / "diagnostic_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(diagnostic_summary, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    registry.mark_completed(
        "run-stage1-diagnostics",
        metrics={
            "v11_baseline_diff": v11_ablation_results["baseline"][
                "diff_linear_minus_scalar"
            ],
            "v11_no_gate_diff": v11_ablation_results["no_gate"][
                "diff_linear_minus_scalar"
            ],
            "pca_residual_r2": question_d_res["residual_r2_pca"],
            "intercept_only_diff": diff_into,
            "known_effect_diff": diff_know,
            "oracle_true_prob_brier": oracle_true_brier,
        },
        execution_time_seconds=elapsed,
        stop_reason="success",
    )
    print(
        f"\nЭтап 1 успешно завершен за {elapsed:.1f}с. Результаты сохранены в {summary_file}"
    )


if __name__ == "__main__":
    main()
