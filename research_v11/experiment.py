"""Основной пайплайн эксперимента V11: раздельные этапы обучения, заморозки и оценки на holdout."""

from __future__ import annotations

import gzip
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from research_v11.encoders import build_v11_encoders
from research_v11.gbdt import PopulationGBDTTrainer
from research_v11.metrics import compute_comprehensive_metrics
from research_v11.provenance import (
    cohort_hashes,
    digest,
    seal_artifacts,
    verify_artifacts,
)
from research_v11.runner import run_v11_replay
from research_v11.stats import holm_bonferroni_correction, paired_bootstrap_test
from research_v11.types import MemoryConfigV11


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fit_and_freeze_v11(
    cohorts_dir: Path,
    connectome_path: Path,
    output_dir: Path,
    allow_overwrite: bool = False,
) -> dict[str, object]:
    """Этапы 1-3: обучение GBDT и проекторов на training, тюнинг d/eta и сохранение полной заморозки.
    
    Не касается holdout-когорты. Сохраняет обученные модели, манифест и полный frozen_config.
    """
    cohorts_dir = Path(cohorts_dir)
    connectome_path = Path(connectome_path)
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Каталог {output_dir} уже содержит файлы. Для защиты от случайной перезаписи укажите новый каталог или allow_overwrite=True."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== ЭТАП 1: Обучение GBDT и проекторов строго на training ===", flush=True)
    X_train = np.load(cohorts_dir / "training.X.npy")
    train_records = _read_jsonl(cohorts_dir / "training.records.jsonl")
    schema = json.loads((cohorts_dir / "schema.json").read_text(encoding="utf-8"))

    observed_mask = np.asarray([r["outcome"] is not None for r in train_records], dtype=bool)
    y_train = np.asarray([r["outcome"] if r["outcome"] is not None else 0 for r in train_records], dtype=int)

    print(f"Обучение GBDT на {observed_mask.sum()} наблюдаемых строках из {len(train_records)}...", flush=True)
    gbdt_trainer = PopulationGBDTTrainer(random_state=20261101).fit(X_train, y_train, observed_mask)
    gbdt_trainer.save(models_dir / "gbdt.pickle")

    # Предвычисление p_0 для training, tuning, calibration
    p0_splits = {}
    for split in ("training", "tuning", "calibration"):
        X_split = np.load(cohorts_dir / f"{split}.X.npy")
        p0_splits[split] = gbdt_trainer.predict_proba(X_split)
        np.save(output_dir / f"p0_{split}.npy", p0_splits[split])

    # Обучение 3 компактных семейств проекторов строго на training.X
    encoders, encoder_meta = build_v11_encoders(X_train, schema, connectome_path)
    with (models_dir / "encoders.pickle").open("wb") as f:
        pickle.dump(encoders, f)
    _write_json(output_dir / "encoders_manifest.json", encoder_meta)

    print("\n=== ЭТАП 2: Подбор параметров (Tuning) на когорте tuning ===", flush=True)
    X_tune = np.load(cohorts_dir / "tuning.X.npy")
    tune_records = _read_jsonl(cohorts_dir / "tuning.records.jsonl")
    tune_eval = _read_jsonl(cohorts_dir / "tuning.evaluation.jsonl")
    p0_tune = p0_splits["tuning"]

    grid_d = [2, 4, 8]
    grid_eta = [0.05, 0.1, 0.2]

    # 1. Тюнинг scalar
    scalar_tuning = {}
    for eta in grid_eta:
        cfg = MemoryConfigV11(fast_rate=eta, slow_rate=eta / 4.0, l2=0.01)
        scores, _ = run_v11_replay(tune_records, tune_eval, p0_tune, None, family="scalar", config=cfg)
        metrics = compute_comprehensive_metrics(scores)
        scalar_tuning[str(eta)] = metrics["macro_brier"]
        print(f"Tuning scalar: eta={eta} -> Brier={metrics['macro_brier']:.6f}", flush=True)

    best_scalar_eta = float(min(scalar_tuning, key=scalar_tuning.get))
    print(f"Выбранный scalar eta*: {best_scalar_eta} (Brier: {scalar_tuning[str(best_scalar_eta)]:.6f})", flush=True)

    # 2. Тюнинг компактных семейств
    compact_families = ["compact_linear", "compact_kc_anatomical", "compact_kc_rewired"]
    pair_scores: dict[tuple[int, float], dict[str, float]] = {}
    compact_grid_results = []

    for d in grid_d:
        contexts_tune = {name: enc.transform(X_tune, d=d) for name, enc in encoders.items()}
        for eta in grid_eta:
            cfg = MemoryConfigV11(fast_rate=eta, slow_rate=eta / 4.0, l2=0.01)
            briers = {}
            for fam in compact_families:
                scores, _ = run_v11_replay(tune_records, tune_eval, p0_tune, contexts_tune[fam], family=fam, dimension=d, config=cfg)
                m = compute_comprehensive_metrics(scores)
                briers[fam] = m["macro_brier"]
            avg_brier = float(np.mean(list(briers.values())))
            pair_scores[(d, eta)] = {"average_brier": avg_brier, **briers}
            compact_grid_results.append({"d": d, "eta_fast": eta, "average_brier": avg_brier, **briers})
            print(f"Tuning compact: d={d}, eta={eta} -> Avg Brier={avg_brier:.6f} {briers}", flush=True)

    # Правило выбора (d*, eta*):
    # 1. Минимум среднего Brier
    # 2. В пределах 0.0002 от минимума: меньшее d, затем меньшая eta
    min_brier = min(v["average_brier"] for v in pair_scores.values())
    candidates = [
        (d, eta, v["average_brier"])
        for (d, eta), v in pair_scores.items()
        if v["average_brier"] <= min_brier + 0.0002
    ]
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_d, best_eta, chosen_brier = candidates[0]
    print(f"\nЗаморозка конфигурации: d*={best_d}, eta*={best_eta} (Avg Brier: {chosen_brier:.6f})", flush=True)

    cfg_compact_obj = MemoryConfigV11(fast_rate=best_eta, slow_rate=best_eta / 4.0, l2=0.01)
    cfg_scalar_obj = MemoryConfigV11(fast_rate=best_scalar_eta, slow_rate=best_scalar_eta / 4.0, l2=0.01)

    frozen_config = {
        "compact": {
            "d": best_d,
            **cfg_compact_obj.model_dump(),
        },
        "scalar": {
            "d": 0,
            **cfg_scalar_obj.model_dump(),
        },
        "seeds": {
            "training": 20261101,
            "tuning": 20261102,
            "calibration": 20261103,
            "holdout": 20261104,
            "bootstrap": 20261105,
        },
        "selection_rule": "minimum average Brier across compact families on tuning cohort; ties/tolerance within 0.0002 broken by lower d then lower eta",
        "pre_holdout_freeze_timestamp": datetime.now(UTC).isoformat(),
    }
    _write_json(output_dir / "frozen_config.json", frozen_config)
    _write_json(output_dir / "tuning_results.json", {
        "scalar_tuning": scalar_tuning,
        "compact_grid": compact_grid_results,
        "selected_compact": {"d": best_d, "eta_fast": best_eta, "average_brier": chosen_brier},
        "selected_scalar": {"eta_fast": best_scalar_eta, "brier": scalar_tuning[str(best_scalar_eta)]},
    })

    print("\n=== ЭТАП 3: Диагностический replay на calibration (без подстройки) ===", flush=True)
    X_cal = np.load(cohorts_dir / "calibration.X.npy")
    cal_records = _read_jsonl(cohorts_dir / "calibration.records.jsonl")
    cal_eval = _read_jsonl(cohorts_dir / "calibration.evaluation.jsonl")
    p0_cal = p0_splits["calibration"]

    contexts_cal = {name: enc.transform(X_cal, d=best_d) for name, enc in encoders.items()}

    cal_metrics = {}
    for fam in ("population", "scalar", "compact_linear", "compact_kc_anatomical", "compact_kc_rewired"):
        cfg = cfg_scalar_obj if fam == "scalar" else cfg_compact_obj
        ctx = None if fam in ("population", "scalar") else contexts_cal[fam]
        dim = 0 if fam in ("population", "scalar") else best_d
        scores, diag = run_v11_replay(cal_records, cal_eval, p0_cal, ctx, family=fam, dimension=dim, config=cfg)
        m = compute_comprehensive_metrics(scores)
        cal_metrics[fam] = {
            "macro_brier": m["macro_brier"],
            "macro_logloss": m["macro_logloss"],
            "macro_auc": m["macro_auc"],
            "pooled_auc": m["pooled_auc"],
            "diagnostics": diag,
        }
        print(f"Calibration {fam}: Brier={m['macro_brier']:.6f}, AUC={m['macro_auc']:.4f}", flush=True)
    _write_json(output_dir / "calibration_summary.json", cal_metrics)

    seal_artifacts(output_dir, cohorts_dir, connectome_path, status="fit_and_freeze")
    print("Заморозка завершена и сохранена в", output_dir, flush=True)
    return frozen_config


def evaluate_frozen_v11(
    cohorts_dir: Path,
    artifacts_dir: Path,
    eval_output_dir: Path | None = None,
    allow_overwrite: bool = False,
) -> dict[str, object]:
    """Этапы 4-5: строго воспроизводимая оценка на holdout с использованием уже замороженных артефактов."""
    cohorts_dir = Path(cohorts_dir)
    artifacts_dir = Path(artifacts_dir)
    eval_output_dir = Path(eval_output_dir) if eval_output_dir is not None else artifacts_dir

    summary_path = eval_output_dir / "evaluation_summary.json"
    existing = ("evaluation_summary.json", "holdout_scores.json.gz", "user_metrics.json", "p0_holdout.npy", "evaluation_manifest.json")
    if any((eval_output_dir / name).exists() for name in existing) and not allow_overwrite:
        raise FileExistsError(
            f"Файл {summary_path} уже существует. Для защиты от случайной перезаписи укажите новый каталог или allow_overwrite=True."
        )

    manifest = verify_artifacts(artifacts_dir, cohorts_dir)
    evaluation_inputs = cohort_hashes(cohorts_dir, ("holdout",))
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    frozen_config = json.loads((artifacts_dir / "frozen_config.json").read_text(encoding="utf-8"))
    best_d = frozen_config["compact"]["d"]
    cfg_compact = MemoryConfigV11(**{k: v for k, v in frozen_config["compact"].items() if k != "d"})
    cfg_scalar = MemoryConfigV11(**{k: v for k, v in frozen_config["scalar"].items() if k != "d"})

    # Загрузка обученных моделей
    gbdt_path = artifacts_dir / "models" / "gbdt.pickle"
    with gbdt_path.open("rb") as f:
        gbdt_model = pickle.load(f)

    encoders_path = artifacts_dir / "models" / "encoders.pickle"
    with encoders_path.open("rb") as f:
        encoders = pickle.load(f)

    print("\n=== ЭТАП 4: Воспроизводимый replay на Holdout Cohort ===", flush=True)
    X_hold = np.load(cohorts_dir / "holdout.X.npy")
    hold_records = _read_jsonl(cohorts_dir / "holdout.records.jsonl")
    hold_eval = _read_jsonl(cohorts_dir / "holdout.evaluation.jsonl")

    # Предсказание p0 для holdout
    p0_hold_raw = gbdt_model.predict_proba(X_hold)[:, 1]
    p0_hold = np.clip(p0_hold_raw, 1e-4, 1.0 - 1e-4)
    np.save(eval_output_dir / "p0_holdout.npy", p0_hold)

    contexts_hold = {name: enc.transform(X_hold, d=best_d) for name, enc in encoders.items()}

    holdout_scores = {}
    holdout_diagnostics = {}
    holdout_metrics = {}

    for fam in ("population", "scalar", "compact_linear", "compact_kc_anatomical", "compact_kc_rewired"):
        cfg = cfg_scalar if fam == "scalar" else cfg_compact
        ctx = None if fam in ("population", "scalar") else contexts_hold[fam]
        dim = 0 if fam in ("population", "scalar") else best_d

        scores, diag = run_v11_replay(hold_records, hold_eval, p0_hold, ctx, family=fam, dimension=dim, config=cfg)
        holdout_scores[fam] = scores
        holdout_diagnostics[fam] = diag

    # Сохраняем полный массив решений holdout в сжатом JSON
    all_scores_flat = []
    for fam, s_list in holdout_scores.items():
        all_scores_flat.extend(s_list)
    with gzip.open(eval_output_dir / "holdout_scores.json.gz", "wt", encoding="utf-8") as f_gz:
        json.dump(all_scores_flat, f_gz)

    # Метрики
    pop_scores = holdout_scores["population"]
    user_metrics_export = {}
    for fam, scores in holdout_scores.items():
        m = compute_comprehensive_metrics(scores, baseline_scores=pop_scores)
        holdout_metrics[fam] = m
        user_metrics_export[fam] = {
            "user_briers": m["user_briers"],
            "user_aucs": m["user_aucs"],
            "user_auc_deltas": m["user_auc_deltas"],
            "user_logloss": m["user_logloss"],
            "evaluable_auc_users": m["evaluable_auc_users"],
            "total_users": m["total_users"],
            "feedbacks_mean": holdout_diagnostics[fam]["feedbacks_mean"],
        }
        print(
            f"Holdout {fam}: Macro Brier={m['macro_brier']:.6f}, "
            f"Macro AUC={m['macro_auc']:.4f}, Pooled AUC={m['pooled_auc']:.4f}, "
            f"Peak JSON bytes={holdout_diagnostics[fam]['peak_state_bytes_max']}",
            flush=True,
        )

    _write_json(eval_output_dir / "user_metrics.json", user_metrics_export)

    print("\n=== ЭТАП 5: Подтверждающий статистический bootstrap-анализ (20 000 реплик, исправленный Холм) ===", flush=True)
    users = sorted(holdout_metrics["population"]["user_briers"].keys())

    # Контраст 1: compact_linear - scalar
    diffs_linear_vs_scalar = np.array([
        holdout_metrics["compact_linear"]["user_briers"][u] - holdout_metrics["scalar"]["user_briers"][u]
        for u in users
    ])
    test_1 = paired_bootstrap_test(diffs_linear_vs_scalar, n_resamples=20_000, seed=frozen_config["seeds"]["bootstrap"])

    # Контраст 2: compact_kc_anatomical - compact_kc_rewired
    diffs_anat_vs_rewired = np.array([
        holdout_metrics["compact_kc_anatomical"]["user_briers"][u] - holdout_metrics["compact_kc_rewired"]["user_briers"][u]
        for u in users
    ])
    test_2 = paired_bootstrap_test(diffs_anat_vs_rewired, n_resamples=20_000, seed=frozen_config["seeds"]["bootstrap"])

    # Поправка Холма с правильной ранней остановкой
    p_values = [
        ("compact_linear_vs_scalar", test_1["p_value"]),
        ("anatomical_vs_rewired", test_2["p_value"]),
    ]
    holm_results = holm_bonferroni_correction(p_values, alpha=0.05)

    print(f"Контраст 1 (compact_linear - scalar): mean={test_1['mean_diff']:.6f}, 95% CI=[{test_1['ci_lower']:.6f}; {test_1['ci_upper']:.6f}], p={test_1['p_value']:.5f}", flush=True)
    print(f"Контраст 2 (anat - rewired): mean={test_2['mean_diff']:.6f}, 95% CI=[{test_2['ci_lower']:.6f}; {test_2['ci_upper']:.6f}], p={test_2['p_value']:.5f}", flush=True)
    print(f"Поправка Холма (step-down): {holm_results}", flush=True)

    c1_holm_sig = holm_results["compact_linear_vs_scalar"]["significant"]
    c1_success = bool(test_1["mean_diff"] <= -0.0020 and test_1["ci_upper"] < 0.0 and c1_holm_sig)

    c2_holm_sig = holm_results["anatomical_vs_rewired"]["significant"]
    c2_success = bool(test_2["mean_diff"] < 0.0 and c2_holm_sig)

    evaluation_report = {
        "status": "reproduced_audit_run",
        "notes": "Holdout cohort seed 20261104 had already been evaluated in initial run. This is a technical reproduction with corrected Holm procedure and aligned few-shot partitioning.",
        "frozen_config": frozen_config,
        "holdout_diagnostics": holdout_diagnostics,
        "primary_contrast": {
            "hypothesis": "compact_linear - scalar < 0",
            "stats": test_1,
            "holm": holm_results["compact_linear_vs_scalar"],
            "engineering_threshold_met": bool(test_1["mean_diff"] <= -0.0020),
            "ci_excludes_zero": bool(test_1["ci_upper"] < 0.0),
            "overall_success": c1_success,
        },
        "secondary_contrast": {
            "hypothesis": "compact_kc_anatomical - compact_kc_rewired < 0",
            "stats": test_2,
            "holm": holm_results["anatomical_vs_rewired"],
            "overall_success": c2_success,
        },
        "families": {
            fam: {
                "macro_brier": holdout_metrics[fam]["macro_brier"],
                "macro_logloss": holdout_metrics[fam]["macro_logloss"],
                "macro_auc": holdout_metrics[fam]["macro_auc"],
                "pooled_auc": holdout_metrics[fam]["pooled_auc"],
                "evaluable_auc_users": holdout_metrics[fam]["evaluable_auc_users"],
                "total_users": holdout_metrics[fam]["total_users"],
                "degraded_fraction": holdout_metrics[fam]["degraded_auc_users_fraction"],
                "few_shot": holdout_metrics[fam]["few_shot"],
                "peak_state_bytes_max": holdout_diagnostics[fam]["peak_state_bytes_max"],
                "peak_state_bytes_mean": holdout_diagnostics[fam]["peak_state_bytes_mean"],
                "feedbacks_mean": holdout_diagnostics[fam]["feedbacks_mean"],
            }
            for fam in holdout_metrics
        },
    }
    verify_artifacts(artifacts_dir, cohorts_dir)
    if evaluation_inputs != cohort_hashes(cohorts_dir, ("holdout",)):
        raise ValueError("Holdout изменился во время оценки")
    _write_json(eval_output_dir / "evaluation_summary.json", evaluation_report)
    _write_json(eval_output_dir / "evaluation_manifest.json", {
        "freeze_sha256": digest(artifacts_dir / "freeze_manifest.json"),
        "freeze_status": manifest["status"],
        "holdout_inputs": evaluation_inputs,
        "outputs": {name: digest(eval_output_dir / name) for name in existing if name != "evaluation_manifest.json"},
    })
    return evaluation_report


def run_full_v11_experiment(
    cohorts_dir: Path,
    connectome_path: Path,
    output_dir: Path,
    allow_overwrite: bool = False,
) -> dict[str, object]:
    """Сквозной запуск двух этапов (заморозка + оценка)."""
    fit_and_freeze_v11(cohorts_dir, connectome_path, output_dir, allow_overwrite=allow_overwrite)
    return evaluate_frozen_v11(cohorts_dir, output_dir, allow_overwrite=allow_overwrite)
