"""Повторная оценка уже выбранных V12-моделей без fit, tuning и открытия confirmation."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from research_v11.gbdt import PopulationGBDTTrainer
from research_v11.metrics import compute_comprehensive_metrics
from research_v11.runner import run_v11_replay
from research_v12.baseline import load_frozen_scalar
from research_v12.models import HierarchicalRidgeMemory, MoEMemory, TunedScalarMemory
from research_v12.registry import file_sha256
from research_v12.search import (
    evaluate_hierarchical_ridge_pipeline,
    evaluate_moe_pipeline,
    evaluate_scalar_pipeline,
)

DEV_SPLITS = ("dev_1", "dev_2", "dev_3")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate_saved(
    data_dir: Path, models_dir: Path, selection: dict, frozen_config_path: Path
) -> tuple[dict, dict]:
    """Входы — доверенные локальные pickle; при воспроизведении сначала проверяется manifest."""
    frozen, provenance = load_frozen_scalar(frozen_config_path)
    gbdt = PopulationGBDTTrainer.load(models_dir / "gbdt_population.pickle")
    scalar_cfg = json.loads((models_dir / "tuned_scalar_config.json").read_text())
    candidates = selection["best_candidate_per_family"]
    with (models_dir / "supervised_bottleneck_encoders.pickle").open("rb") as f:
        encoders = pickle.load(f)
    with (models_dir / "moe_models.pickle").open("rb") as f:
        experts = pickle.load(f)
    # Scaler однозначно восстанавливается только из training, без меток/подбора.
    train_x = np.load(data_dir / "training.X.npy")
    mean = train_x.mean(axis=0)
    std = np.where(train_x.std(axis=0) > 1e-4, train_x.std(axis=0), 1.0)
    scores = {
        name: {}
        for name in (
            "population",
            "frozen_scalar_v11",
            "tuned_scalar_v12",
            "fixed_expert_mixture",
            *candidates,
        )
    }
    for split in DEV_SPLITS:
        x = np.load(data_dir / f"{split}.X.npy")
        records = read_jsonl(data_dir / f"{split}.records.jsonl")
        evaluations = read_jsonl(data_dir / f"{split}.evaluation.jsonl")
        p0 = gbdt.predict_proba(x)
        for name, family in (
            ("population", "population"),
            ("frozen_scalar_v11", "scalar"),
        ):
            rows, _ = run_v11_replay(
                records, evaluations, p0, None, family=family, config=frozen
            )
            scores[name][split] = rows
        scores["tuned_scalar_v12"][split] = evaluate_scalar_pipeline(
            records, evaluations, p0, scalar_cfg
        )["scores"]
        for family, candidate in candidates.items():
            cfg = candidate["config"]
            if family == "family_1_hierarchical_ridge":
                z = ((x - mean) / std)[:, : cfg["dimension"]]
                result = evaluate_hierarchical_ridge_pipeline(
                    records, evaluations, p0, z, cfg
                )
            elif family == "family_2_supervised_bottleneck":
                z = encoders[cfg["dimension"]].transform(x, d=cfg["dimension"])
                result = evaluate_hierarchical_ridge_pipeline(
                    records, evaluations, p0, z, cfg
                )
            elif family == "family_3_mixture_of_experts":
                expert_probs = experts[cfg["k_experts"]].predict_experts(x)
                result = evaluate_moe_pipeline(records, evaluations, expert_probs, cfg)
                fixed_cfg = {**cfg, "eta_expert": 0.0}
                scores["fixed_expert_mixture"][split] = evaluate_moe_pipeline(
                    records, evaluations, expert_probs, fixed_cfg
                )["scores"]
            else:
                raise ValueError(f"Неизвестное семейство: {family}")
            scores[family][split] = result["scores"]
    summary = copy.deepcopy(selection)
    summary["frozen_scalar_provenance"] = provenance
    summary["status"] = "exploratory_replay_no_new_selection"
    summary["replay_metrics"] = {
        family: {
            split: compute_comprehensive_metrics(rows, scores["population"][split])
            for split, rows in per_split.items()
        }
        for family, per_split in scores.items()
    }
    baselines = summary["baselines_dev"]
    for family, mean_key in (
        ("population", "mean_population_brier"),
        ("frozen_scalar_v11", "mean_frozen_scalar_brier"),
        ("tuned_scalar_v12", "mean_tuned_scalar_brier"),
    ):
        values = {
            s: summary["replay_metrics"][family][s]["macro_brier"] for s in DEV_SPLITS
        }
        baselines[family] = values
        baselines[mean_key] = float(np.mean(list(values.values())))
    for family, candidate in summary["best_candidate_per_family"].items():
        cls = (
            MoEMemory
            if family == "family_3_mixture_of_experts"
            else HierarchicalRidgeMemory
        )
        candidate["config"] = cls(**candidate["config"]).get_config()
        candidate["seed_briers"] = {
            s: summary["replay_metrics"][family][s]["macro_brier"] for s in DEV_SPLITS
        }
        candidate["avg_brier"] = float(np.mean(list(candidate["seed_briers"].values())))
        candidate["diff"] = (
            candidate["avg_brier"] - baselines["mean_tuned_scalar_brier"]
        )
    # Исправление не выбирает нового победителя после просмотра повторных метрик.
    family = selection["best_candidate"]["family"]
    summary["best_candidate"] = {
        "family": family,
        **summary["best_candidate_per_family"][family],
    }
    summary["confirmation"] = {"generated": False, "opened": False, "authorized": False}
    return summary, scores


def replay_to_directory(
    data_dir: Path, stage2_dir: Path, frozen_config_path: Path, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = json.loads((stage2_dir / "search_summary.json").read_text())
    shutil.copytree(stage2_dir / "models", output_dir / "models")
    shutil.copy2(frozen_config_path, output_dir / "frozen_v11_config.json")
    shutil.copy2(
        stage2_dir / "search_summary.json", output_dir / "selection_summary.json"
    )
    config_path = output_dir / "models/tuned_scalar_config.json"
    config = TunedScalarMemory(**json.loads(config_path.read_text())).get_config()
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    inputs = [
        data_dir / "training.X.npy",
        frozen_config_path,
        stage2_dir / "search_summary.json",
    ]
    for split in DEV_SPLITS:
        inputs.extend(
            data_dir / f"{split}.{suffix}"
            for suffix in ("X.npy", "records.jsonl", "evaluation.jsonl")
        )
    inputs.extend(p for p in (stage2_dir / "models").iterdir() if p.is_file())
    (output_dir / "input_manifest.json").write_text(
        json.dumps({str(p.resolve()): file_sha256(p) for p in inputs}, indent=2)
    )
    summary, scores = evaluate_saved(
        data_dir,
        output_dir / "models",
        selection,
        output_dir / "frozen_v11_config.json",
    )
    (output_dir / "search_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "dev_scores_all.json").write_text(json.dumps(scores))
    for filename, family in (
        ("dev_scores_tuned_scalar.json", "tuned_scalar_v12"),
        ("dev_scores_best_candidate.json", summary["best_candidate"]["family"]),
    ):
        (output_dir / filename).write_text(json.dumps(scores[family]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--frozen-v11-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    replay_to_directory(
        args.data_dir, args.stage2_dir, args.frozen_v11_config, args.output_dir
    )
