"""Поисковый скрипт Этапа 2: тюнинг скаляра и оценка 3 компактных семейств на 3 dev-сидах.

Исправления аудита:
1. Полноценный Frozen Scalar V11 через run_v11_replay и MemoryConfigV11.
2. Обучение SupervisedBottleneckEncoder и MixtureOfExpertsModel строго на наблюдаемых строках (observed_mask).
3. Прямое сохранение использованных моделей и конфигураций в runs/v12-stage2/models/.
4. Сохранение пошаговых прогнозов на dev-сидах для воспроизведения без переобучения.
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from research_v11.gbdt import PopulationGBDTTrainer
from research_v11.metrics import compute_comprehensive_metrics
from research_v11.runner import run_v11_replay
from research_v12.baseline import load_frozen_scalar
from research_v12.models import (
    HierarchicalRidgeMemory,
    MixtureOfExpertsModel,
    MoEMemory,
    SupervisedBottleneckEncoder,
    TunedScalarMemory,
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_scalar_pipeline(
    records: list[dict],
    evaluations: list[dict],
    p0: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    eval_by_event = {e["event_id"]: e for e in evaluations}
    by_user: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        by_user.setdefault(r["user_id"], []).append((i, r))

    scores = []
    for user_rows in by_user.values():
        user_rows.sort(key=lambda item: (item[1]["decision_at"], item[1]["event_id"]))
        memory = TunedScalarMemory(**config)
        pending = []
        feedbacks = 0
        for idx, row in user_rows:
            dec_at = row["decision_at"]
            t_dec = datetime.fromisoformat(dec_at)
            event_id = row["event_id"]
            p_pop = float(p0[idx])
            eval_info = eval_by_event[event_id]
            true_outcome = int(eval_info["evaluation_outcome"])

            pending.sort(key=lambda x: x[0])
            while pending and pending[0][0] < t_dec:
                fb_at, fb_ev_id, fb_out, fb_conf = pending.pop(0)
                memory.observe(fb_ev_id, fb_at, fb_out, fb_conf)
                feedbacks += 1

            snap = memory.predict(event_id, t_dec, p_pop)
            scores.append(
                {
                    "user_id": row["user_id"],
                    "event_id": event_id,
                    "decision_at": dec_at,
                    "p_population": p_pop,
                    "p_final": snap["p_final"],
                    "p_learning": snap["p_learning"],
                    "gamma": snap.get("gamma", 1.0),
                    "evaluation_outcome": true_outcome,
                    "feedbacks_processed": feedbacks,
                }
            )

            if row.get("feedback_at") and row.get("outcome") is not None:
                fb_time = datetime.fromisoformat(row["feedback_at"])
                pending.append(
                    (
                        fb_time,
                        event_id,
                        int(row["outcome"]),
                        float(row.get("confidence", 1.0)),
                    )
                )
            else:
                memory.discard(event_id)

    metrics = compute_comprehensive_metrics(scores)
    return {"metrics": metrics, "scores": scores}


def evaluate_hierarchical_ridge_pipeline(
    records: list[dict],
    evaluations: list[dict],
    p0: np.ndarray,
    contexts_z: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    eval_by_event = {e["event_id"]: e for e in evaluations}
    by_user: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        by_user.setdefault(r["user_id"], []).append((i, r))

    scores = []
    for user_rows in by_user.values():
        user_rows.sort(key=lambda item: (item[1]["decision_at"], item[1]["event_id"]))
        memory = HierarchicalRidgeMemory(**config)
        pending = []
        feedbacks = 0
        for idx, row in user_rows:
            dec_at = row["decision_at"]
            t_dec = datetime.fromisoformat(dec_at)
            event_id = row["event_id"]
            p_pop = float(p0[idx])
            z_row = contexts_z[idx]
            eval_info = eval_by_event[event_id]
            true_outcome = int(eval_info["evaluation_outcome"])

            pending.sort(key=lambda x: x[0])
            while pending and pending[0][0] < t_dec:
                fb_at, fb_ev_id, fb_out, fb_conf = pending.pop(0)
                memory.observe(fb_ev_id, fb_at, fb_out, fb_conf)
                feedbacks += 1

            snap = memory.predict(event_id, t_dec, z_row, p_pop)
            scores.append(
                {
                    "user_id": row["user_id"],
                    "event_id": event_id,
                    "decision_at": dec_at,
                    "p_population": p_pop,
                    "p_final": snap["p_final"],
                    "p_learning": snap["p_learning"],
                    "gamma": snap.get("gamma", 1.0),
                    "evaluation_outcome": true_outcome,
                    "feedbacks_processed": feedbacks,
                }
            )

            if row.get("feedback_at") and row.get("outcome") is not None:
                fb_time = datetime.fromisoformat(row["feedback_at"])
                pending.append(
                    (
                        fb_time,
                        event_id,
                        int(row["outcome"]),
                        float(row.get("confidence", 1.0)),
                    )
                )
            else:
                memory.discard(event_id)

    metrics = compute_comprehensive_metrics(scores)
    return {"metrics": metrics, "scores": scores}


def evaluate_moe_pipeline(
    records: list[dict],
    evaluations: list[dict],
    expert_probs: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    eval_by_event = {e["event_id"]: e for e in evaluations}
    by_user: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        by_user.setdefault(r["user_id"], []).append((i, r))

    scores = []
    for user_rows in by_user.values():
        user_rows.sort(key=lambda item: (item[1]["decision_at"], item[1]["event_id"]))
        memory = MoEMemory(**config)
        pending = []
        feedbacks = 0
        for idx, row in user_rows:
            dec_at = row["decision_at"]
            t_dec = datetime.fromisoformat(dec_at)
            event_id = row["event_id"]
            p_experts = expert_probs[idx]
            eval_info = eval_by_event[event_id]
            true_outcome = int(eval_info["evaluation_outcome"])

            pending.sort(key=lambda x: x[0])
            while pending and pending[0][0] < t_dec:
                fb_at, fb_ev_id, fb_out, fb_conf = pending.pop(0)
                memory.observe(fb_ev_id, fb_at, fb_out, fb_conf)
                feedbacks += 1

            snap = memory.predict(event_id, t_dec, p_experts)
            scores.append(
                {
                    "user_id": row["user_id"],
                    "event_id": event_id,
                    "decision_at": dec_at,
                    "p_population": float(np.mean(p_experts)),
                    "p_final": snap["p_final"],
                    "p_learning": snap["p_final"],
                    "gamma": 1.0,
                    "evaluation_outcome": true_outcome,
                    "feedbacks_processed": feedbacks,
                }
            )

            if row.get("feedback_at") and row.get("outcome") is not None:
                fb_time = datetime.fromisoformat(row["feedback_at"])
                pending.append(
                    (
                        fb_time,
                        event_id,
                        int(row["outcome"]),
                        float(row.get("confidence", 1.0)),
                    )
                )
            else:
                memory.discard(event_id)

    metrics = compute_comprehensive_metrics(scores)
    return {"metrics": metrics, "scores": scores}


def main(*, data_dir: Path, output_dir: Path, frozen_config_path: Path) -> None:
    print(
        "=== ЭТАП 2: ТЮНИНГ СКАЛЯРА И ПОИСК КОМПАКТНЫХ МЕТОДОВ (ИСПРАВЛЕННЫЙ) ===",
        flush=True,
    )
    t0 = time.time()

    frozen_v11_cfg, frozen_provenance = load_frozen_scalar(frozen_config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    models_dir = output_dir / "models"
    models_dir.mkdir()
    v12_dir = data_dir
    X_train = np.load(v12_dir / "training.X.npy")
    train_records = _read_jsonl(v12_dir / "training.records.jsonl")
    y_train = np.asarray(
        [r["outcome"] if r["outcome"] is not None else 0 for r in train_records],
        dtype=int,
    )
    obs_mask = np.asarray([r["outcome"] is not None for r in train_records], dtype=bool)
    user_ids_train = [r["user_id"] for r in train_records]

    # Обучение популяционной GBDT строго на training (только по наблюдаемым исходам)
    print("Обучение популяционного GBDT...", flush=True)
    gbdt = PopulationGBDTTrainer(random_state=20261201).fit(X_train, y_train, obs_mask)
    gbdt.save(models_dir / "gbdt_population.pickle")
    p0_train = gbdt.predict_proba(X_train)

    # 1. ТЮНИНГ СКАЛЯРА на когорте tuning
    print("\n--- 1. Тюнинг скалярного бейзлайна на когорте tuning ---", flush=True)
    X_tune = np.load(v12_dir / "tuning.X.npy")
    tune_records = _read_jsonl(v12_dir / "tuning.records.jsonl")
    tune_eval = _read_jsonl(v12_dir / "tuning.evaluation.jsonl")
    p0_tune = gbdt.predict_proba(X_tune)

    scalar_configs = []
    for fr in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for sr in [0.1, 0.25, 0.5]:
            for l2 in [0.005, 0.01]:
                scalar_configs.append(
                    {"fast_rate": fr, "slow_ratio": sr, "l2": l2, "use_gate": True}
                )

    best_scalar_cfg = None
    best_scalar_brier = float("inf")
    for cfg in scalar_configs:
        res = evaluate_scalar_pipeline(tune_records, tune_eval, p0_tune, cfg)
        brier = res["metrics"]["macro_brier"]
        if brier < best_scalar_brier:
            best_scalar_brier = brier
            best_scalar_cfg = cfg

    print(
        f"Лучший конфиг Tuned Scalar: {best_scalar_cfg} с Brier на tuning = {best_scalar_brier:.6f}",
        flush=True,
    )
    best_scalar_cfg = TunedScalarMemory(**best_scalar_cfg).get_config()
    with (models_dir / "tuned_scalar_config.json").open("w", encoding="utf-8") as f:
        json.dump(best_scalar_cfg, f, indent=2, ensure_ascii=False)

    # Загрузка dev когорт
    dev_data = {}
    for d_name in ["dev_1", "dev_2", "dev_3"]:
        X_d = np.load(v12_dir / f"{d_name}.X.npy")
        rec_d = _read_jsonl(v12_dir / f"{d_name}.records.jsonl")
        eval_d = _read_jsonl(v12_dir / f"{d_name}.evaluation.jsonl")
        p0_d = gbdt.predict_proba(X_d)
        dev_data[d_name] = {"X": X_d, "records": rec_d, "eval": eval_d, "p0": p0_d}

    # Оценка Population, Frozen Scalar V11 и Tuned Scalar V12 на 3 dev-сидах
    pop_briers = {}
    frozen_scalar_briers = {}
    tuned_scalar_briers = {}
    tuned_scalar_scores = {}
    frozen_scalar_scores = {}

    for d_name, d_dict in dev_data.items():
        y_d = np.asarray([e["evaluation_outcome"] for e in d_dict["eval"]])
        pop_briers[d_name] = float(np.mean((d_dict["p0"] - y_d) ** 2))

        # Настоящий Frozen Scalar V11 через канонический v11 runner
        frozen_scores, _ = run_v11_replay(
            d_dict["records"],
            d_dict["eval"],
            d_dict["p0"],
            None,
            family="scalar",
            config=frozen_v11_cfg,
        )
        frozen_scalar_scores[d_name] = frozen_scores
        m_frozen = compute_comprehensive_metrics(frozen_scores)
        frozen_scalar_briers[d_name] = m_frozen["macro_brier"]

        # Tuned Scalar V12
        res_tuned = evaluate_scalar_pipeline(
            d_dict["records"], d_dict["eval"], d_dict["p0"], best_scalar_cfg
        )
        tuned_scalar_briers[d_name] = res_tuned["metrics"]["macro_brier"]
        tuned_scalar_scores[d_name] = res_tuned["scores"]

    print("\n--- Базовые линии на 3 dev-сидах ---", flush=True)
    print(
        "Population Macro Briers:",
        pop_briers,
        f"Mean={np.mean(list(pop_briers.values())):.6f}",
    )
    print(
        "True Frozen Scalar V11 Macro Briers:",
        frozen_scalar_briers,
        f"Mean={np.mean(list(frozen_scalar_briers.values())):.6f}",
    )
    print(
        "Tuned Scalar V12 Macro Briers:",
        tuned_scalar_briers,
        f"Mean={np.mean(list(tuned_scalar_briers.values())):.6f}",
    )
    mean_tuned_scalar_brier = float(np.mean(list(tuned_scalar_briers.values())))

    # 2. СЕМЕЙСТВО 1: Иерархическая регуляризованная логистическая поправка (Hierarchical Ridge)
    print("\n--- 2. Поиск: Семейство 1 (Hierarchical Ridge Residual) ---", flush=True)
    mean_X = np.mean(X_train, axis=0)
    std_X = np.where(np.std(X_train, axis=0) > 1e-4, np.std(X_train, axis=0), 1.0)
    np.savez(models_dir / "linear_scaler.npz", mean=mean_X, std=std_X)

    f1_configs = []
    for d in [2, 4, 8]:
        for kappa in [5.0, 15.0, 30.0]:
            for fr in [0.05, 0.1]:
                f1_configs.append(
                    {
                        "dimension": d,
                        "shrinkage_kappa": kappa,
                        "fast_rate": fr,
                        "slow_ratio": 0.25,
                        "l2": 0.05,
                        "use_gate": True,
                    }
                )
    f1_configs = f1_configs[:18]

    f1_results = []
    for cfg in f1_configs:
        dim = cfg["dimension"]
        seed_briers = {}
        for d_name, d_dict in dev_data.items():
            Z_norm = (d_dict["X"] - mean_X) / std_X
            res = evaluate_hierarchical_ridge_pipeline(
                d_dict["records"], d_dict["eval"], d_dict["p0"], Z_norm[:, :dim], cfg
            )
            seed_briers[d_name] = res["metrics"]["macro_brier"]
        avg_brier = float(np.mean(list(seed_briers.values())))
        diff_vs_scalar = avg_brier - mean_tuned_scalar_brier
        f1_results.append(
            {
                "config": cfg,
                "avg_brier": avg_brier,
                "seed_briers": seed_briers,
                "diff": diff_vs_scalar,
            }
        )
        print(
            f"F1 [d={dim}, k={cfg['shrinkage_kappa']}, fr={cfg['fast_rate']}]: Avg Brier={avg_brier:.6f} (Δ={diff_vs_scalar:+.6f})",
            flush=True,
        )

    # 3. СЕМЕЙСТВО 2: Контролируемый остаточный проектор (Supervised Bottleneck)
    # Обучаем проекторы на training строго по observed_mask (БЕЗ превращения None в 0)
    print(
        "\n--- 3. Поиск: Семейство 2 (Supervised Bottleneck, строго на observed исходах) ---",
        flush=True,
    )
    sb_encoders = {}
    for d in [2, 4, 8]:
        sb_encoders[d] = SupervisedBottleneckEncoder(
            n_components=d, alpha=10.0, seed=20261201
        ).fit(X_train, y_train, p0_train, observed_mask=obs_mask)
    with (models_dir / "supervised_bottleneck_encoders.pickle").open("wb") as f:
        pickle.dump(sb_encoders, f)

    f2_configs = []
    for d in [2, 4, 8]:
        for kappa in [5.0, 15.0]:
            for fr in [0.05, 0.1, 0.2]:
                f2_configs.append(
                    {
                        "dimension": d,
                        "shrinkage_kappa": kappa,
                        "fast_rate": fr,
                        "slow_ratio": 0.25,
                        "l2": 0.02,
                        "use_gate": True,
                    }
                )
    f2_configs = f2_configs[:18]

    f2_results = []
    for cfg in f2_configs:
        dim = cfg["dimension"]
        encoder = sb_encoders[dim]
        seed_briers = {}
        curr_scores = {}
        for d_name, d_dict in dev_data.items():
            Z_proj = encoder.transform(d_dict["X"], d=dim)
            res = evaluate_hierarchical_ridge_pipeline(
                d_dict["records"], d_dict["eval"], d_dict["p0"], Z_proj, cfg
            )
            seed_briers[d_name] = res["metrics"]["macro_brier"]
            curr_scores[d_name] = res["scores"]
        avg_brier = float(np.mean(list(seed_briers.values())))
        diff_vs_scalar = avg_brier - mean_tuned_scalar_brier
        f2_results.append(
            {
                "config": cfg,
                "avg_brier": avg_brier,
                "seed_briers": seed_briers,
                "diff": diff_vs_scalar,
                "_scores": curr_scores,
            }
        )
        print(
            f"F2 [d={dim}, k={cfg['shrinkage_kappa']}, fr={cfg['fast_rate']}]: Avg Brier={avg_brier:.6f} (Δ={diff_vs_scalar:+.6f})",
            flush=True,
        )

    # 4. СЕМЕЙСТВО 3: Смесь популяционных экспертов (Mixture of Experts)
    # Обучение строго на observed_mask
    print(
        "\n--- 4. Поиск: Семейство 3 (Mixture of Experts, строго на observed исходах) ---",
        flush=True,
    )
    moe_models = {}
    for k in [2, 3, 4]:
        moe_models[k] = MixtureOfExpertsModel(k_experts=k, seed=20261201).fit(
            X_train, y_train, user_ids_train, observed_mask=obs_mask
        )
    with (models_dir / "moe_models.pickle").open("wb") as f:
        pickle.dump(moe_models, f)

    f3_configs = []
    for k in [2, 3, 4]:
        for eta in [0.2, 0.5]:
            for hl in [30.0, 60.0]:
                f3_configs.append(
                    {"k_experts": k, "eta_expert": eta, "half_life_days": hl}
                )
    f3_configs = f3_configs[:12]

    f3_results = []
    for cfg in f3_configs:
        k = cfg["k_experts"]
        moe = moe_models[k]
        seed_briers = {}
        for d_name, d_dict in dev_data.items():
            exp_probs = moe.predict_experts(d_dict["X"])
            res = evaluate_moe_pipeline(
                d_dict["records"], d_dict["eval"], exp_probs, cfg
            )
            seed_briers[d_name] = res["metrics"]["macro_brier"]
        avg_brier = float(np.mean(list(seed_briers.values())))
        diff_vs_scalar = avg_brier - mean_tuned_scalar_brier
        f3_results.append(
            {
                "config": cfg,
                "avg_brier": avg_brier,
                "seed_briers": seed_briers,
                "diff": diff_vs_scalar,
            }
        )
        print(
            f"F3 [k={k}, eta={cfg['eta_expert']}, hl={cfg['half_life_days']}]: Avg Brier={avg_brier:.6f} (Δ={diff_vs_scalar:+.6f})",
            flush=True,
        )

    # Выбор лучших в семействах
    best_f1 = min(f1_results, key=lambda x: x["avg_brier"])
    best_f2 = min(f2_results, key=lambda x: x["avg_brier"])
    best_f3 = min(f3_results, key=lambda x: x["avg_brier"])

    all_candidates = [
        {"family": "family_1_hierarchical_ridge", **best_f1},
        {"family": "family_2_supervised_bottleneck", **best_f2},
        {"family": "family_3_mixture_of_experts", **best_f3},
    ]
    best_cand = min(all_candidates, key=lambda x: x["avg_brier"])
    candidate_scores = {}
    for name, data in dev_data.items():
        cfg = best_cand["config"]
        if best_cand["family"] == "family_3_mixture_of_experts":
            experts = moe_models[cfg["k_experts"]].predict_experts(data["X"])
            result = evaluate_moe_pipeline(data["records"], data["eval"], experts, cfg)
        else:
            dim = cfg["dimension"]
            z = (
                ((data["X"] - mean_X) / std_X)[:, :dim]
                if best_cand["family"] == "family_1_hierarchical_ridge"
                else sb_encoders[dim].transform(data["X"], d=dim)
            )
            result = evaluate_hierarchical_ridge_pipeline(
                data["records"], data["eval"], data["p0"], z, cfg
            )
        candidate_scores[name] = result["scores"]
    for row in f2_results:
        row.pop("_scores", None)
    best_cand.pop("_scores", None)
    for filename, scores in (
        ("dev_scores_tuned_scalar.json", tuned_scalar_scores),
        ("dev_scores_best_candidate.json", candidate_scores),
        ("dev_scores_frozen_scalar.json", frozen_scalar_scores),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(scores, f)

    elapsed = time.time() - t0
    summary = {
        "created_at": time.time(),
        "elapsed_seconds": elapsed,
        "frozen_scalar_provenance": frozen_provenance,
        "tuning_scalar": {
            "best_config": best_scalar_cfg,
            "best_tuning_brier": best_scalar_brier,
        },
        "baselines_dev": {
            "population": pop_briers,
            "mean_population_brier": float(np.mean(list(pop_briers.values()))),
            "frozen_scalar_v11": frozen_scalar_briers,
            "mean_frozen_scalar_brier": float(
                np.mean(list(frozen_scalar_briers.values()))
            ),
            "tuned_scalar_v12": tuned_scalar_briers,
            "mean_tuned_scalar_brier": mean_tuned_scalar_brier,
        },
        "best_candidate_per_family": {
            "family_1_hierarchical_ridge": {
                "config": best_f1["config"],
                "avg_brier": best_f1["avg_brier"],
                "seed_briers": best_f1["seed_briers"],
                "diff": best_f1["diff"],
            },
            "family_2_supervised_bottleneck": {
                "config": best_f2["config"],
                "avg_brier": best_f2["avg_brier"],
                "seed_briers": best_f2["seed_briers"],
                "diff": best_f2["diff"],
            },
            "family_3_mixture_of_experts": {
                "config": best_f3["config"],
                "avg_brier": best_f3["avg_brier"],
                "seed_briers": best_f3["seed_briers"],
                "diff": best_f3["diff"],
            },
        },
        "best_candidate": {
            "family": best_cand["family"],
            "config": best_cand["config"],
            "avg_brier": best_cand["avg_brier"],
            "seed_briers": best_cand["seed_briers"],
            "diff": best_cand["diff"],
        },
        "all_f1_results": f1_results,
        "all_f2_results": f2_results,
        "all_f3_results": f3_results,
    }

    with (output_dir / "search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== ИТОГИ ПОИСКА ЭТАПА 2 ===", flush=True)
    print(
        f"Population Mean Brier: {summary['baselines_dev']['mean_population_brier']:.6f}"
    )
    print(
        f"Frozen Scalar V11 Mean Brier: {summary['baselines_dev']['mean_frozen_scalar_brier']:.6f}"
    )
    print(f"Tuned Scalar V12 Mean Brier: {mean_tuned_scalar_brier:.6f}")
    print(
        f"Лучший кандидат: {best_cand['family']} с конфигурацией {best_cand['config']}"
    )
    print(
        f"Средний Brier: {best_cand['avg_brier']:.6f} vs Tuned Scalar {mean_tuned_scalar_brier:.6f}"
    )
    print(f"Разность (ΔBrier): {best_cand['diff']:+.6f}")
    print("Разности по сидам:")
    for d_name in ("dev_1", "dev_2", "dev_3"):
        d_diff = best_cand["seed_briers"][d_name] - tuned_scalar_briers[d_name]
        print(
            f"  {d_name}: Cand={best_cand['seed_briers'][d_name]:.6f}, Scalar={tuned_scalar_briers[d_name]:.6f}, Δ={d_diff:+.6f}"
        )

    passes_delta = best_cand["diff"] <= -0.0020
    neg_seeds = sum(
        1
        for d_name in ("dev_1", "dev_2", "dev_3")
        if (best_cand["seed_briers"][d_name] - tuned_scalar_briers[d_name]) < 0
    )
    passes_seeds = neg_seeds >= 2
    print(
        f"\nКритерий допуска: Δ <= -0.0020 ({passes_delta}, факт {best_cand['diff']:+.6f}), "
        f"знак на >=2 сидах ({passes_seeds}, факт {neg_seeds}/3)"
    )
    print(f"Допуск к подтверждающему испытанию: {passes_delta and passes_seeds}")


if __name__ == "__main__":
    raise SystemExit(
        "Поисковый бюджет V12 исчерпан; используйте research_v12.replay для исправления без нового подбора"
    )
