"""Генератор непересекающихся когорт V12: training, tuning, dev_1, dev_2, dev_3 и confirmation.

Строго изолирует подтверждающие сиды (confirmation) до получения допуска на Этапе 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

from synthetic_data.contexts import history_contexts
from synthetic_data.features import (
    observed_schema,
    observed_summary,
)
from synthetic_data.nutrifit_encoder import encode_histories
from synthetic_data.population import SyntheticPopulationGenerator
from synthetic_data.quality import summarize_dataset
from synthetic_data.simulator import LongitudinalSimulator
from synthetic_data.types import FeedbackRecord, InformationFeature

DevSplitName = Literal["training", "tuning", "dev_1", "dev_2", "dev_3"]
ConfSplitName = Literal["conf_1", "conf_2", "conf_3", "conf_4", "conf_5"]
SplitName = Literal[DevSplitName, ConfSplitName]

DEV_COHORT_SPECS: list[tuple[DevSplitName, int, int]] = [
    ("training", 60, 20261201),
    ("tuning", 30, 20261202),
    ("dev_1", 40, 20261203),
    ("dev_2", 40, 20261204),
    ("dev_3", 40, 20261205),
]

CONFIRMATION_COHORT_SPECS: list[tuple[ConfSplitName, int, int]] = [
    ("conf_1", 40, 20261211),
    ("conf_2", 40, 20261212),
    ("conf_3", 40, 20261213),
    ("conf_4", 40, 20261214),
    ("conf_5", 40, 20261215),
]


def _json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _array(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def generate_v12_cohorts(
    output_dir: Path,
    backend_root: Path | None = Path("/Users/gurutom/Develop/NutriFit/apps/backend"),
    include_confirmation: bool = False,
    stage_4_authorized: bool = False,
) -> dict[str, object]:
    """Генерирует когорты V12 с проверкой уникальности профилей и защитой confirmation."""
    if include_confirmation and not stage_4_authorized:
        raise PermissionError(
            "Генерация подтверждающей когорты (confirmation) заблокирована до выполнения условий допуска на Этапе 4."
        )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = list(DEV_COHORT_SPECS)
    if include_confirmation and stage_4_authorized:
        specs.extend(CONFIRMATION_COHORT_SPECS)

    seen_profiles: dict[str, str] = {}
    summaries: dict[str, object] = {}
    schemas: dict[str, list[dict]] = {}

    for split, count, split_seed in specs:
        population = SyntheticPopulationGenerator(split_seed).generate(count)
        simulator = LongitudinalSimulator(split_seed)
        histories = []
        profile_hashes = {}

        prefix = split.replace("_", "").upper()[:4]
        for index, profile in enumerate(population, start=1):
            user_id = f"V12-{prefix}-{index:03d}"
            user = replace(profile, user_id=user_id, persona_id=None)

            params = {
                k: v
                for k, v in user.to_dict().items()
                if k
                not in {
                    "user_id",
                    "persona_id",
                    "persona_name",
                    "persona_category",
                    "description",
                }
            }
            digest = hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()

            if digest in seen_profiles:
                prev_user = seen_profiles[digest]
                raise ValueError(f"Повтор профиля: {user_id} совпадает с {prev_user}")

            seen_profiles[digest] = user_id
            profile_hashes[user_id] = digest
            histories.append(simulator.simulate_user_180_days(user))

        quality = summarize_dataset(histories)
        contexts = [ctx for hist in histories for ctx in history_contexts(hist)[0]]
        base_schema = observed_schema()
        summaries_raw = [observed_summary(ctx) for ctx in contexts]
        current_x = np.asarray(
            [[row[f.name] for f in base_schema] for row in summaries_raw],
            dtype=np.float64,
        )

        encoder_info = None
        full_schema = base_schema
        if backend_root is not None and backend_root.exists():
            vectors, encoder_info = encode_histories(
                histories, backend_root=backend_root
            )
            axes = encoder_info["runtime"]["axes"]
            pn_schema = tuple(
                InformationFeature(
                    f"pn_{axis}", "0..1", f"Локальный энкодер NutriFit: {axis}"
                )
                for axis in axes
            )
            full_schema = pn_schema + base_schema
            pn_features = np.concatenate(vectors).astype(np.float64)
            current_x = np.concatenate((pn_features, current_x), axis=1)

        records = []
        evaluators = []
        feedback_rows = []

        for history in histories:
            user_id = history[0][1].userId
            fb_seed = int.from_bytes(
                hashlib.sha256(f"{split_seed}:{user_id}:feedback".encode()).digest()[
                    :8
                ],
                "big",
            )
            rng = np.random.default_rng(fb_seed)

            for index, (truth, context) in enumerate(history):
                at = context.evaluationTime.replace(tzinfo=UTC)
                actual = int(truth.relapse_occurred)
                # Положительный исход сообщается с 0.8, отрицательный с 0.7 при наличии логов
                reported = bool(context.meals or context.wellbeing) and rng.random() < (
                    0.8 if actual else 0.7
                )
                feedback_at = at + timedelta(hours=20) if reported else None
                event_id = f"{user_id}:{index}"

                row = FeedbackRecord(
                    split=split,
                    user_id=user_id,
                    event_id=event_id,
                    decision_at=at,
                    available_at=at,
                    outcome=actual if reported else None,
                    feedback_at=feedback_at,
                )
                feedback_rows.append(row)
                records.append(
                    {
                        **asdict(row),
                        "decision_at": at.isoformat(),
                        "available_at": at.isoformat(),
                        "feedback_at": feedback_at.isoformat() if feedback_at else None,
                    }
                )
                evaluators.append(
                    {
                        "event_id": event_id,
                        "user_id": user_id,
                        "day_index": index,
                        "decision_at": at.isoformat(),
                        "evaluation_outcome": actual,
                        "evaluation_probability": truth.true_relapse_prob,
                    }
                )

        _array(output_dir / f"{split}.X.npy", current_x)
        _jsonl(output_dir / f"{split}.records.jsonl", records)
        _jsonl(output_dir / f"{split}.evaluation.jsonl", evaluators)

        rows_meta = [
            {
                "row_index": i,
                "user_id": r["user_id"],
                "event_id": r["event_id"],
                "decision_at": r["decision_at"],
            }
            for i, r in enumerate(records)
        ]
        _jsonl(output_dir / f"{split}.rows.jsonl", rows_meta)

        schemas[split] = [asdict(f) for f in full_schema]
        summaries[split] = {
            "seed": split_seed,
            "users": count,
            "rows": len(records),
            "profile_sha256": profile_hashes,
            "quality": quality,
            "encoder": encoder_info,
        }

    # Проверка единообразия схемы
    first_schema = next(iter(schemas.values()))
    for s_name, s_val in schemas.items():
        if s_val != first_schema:
            raise ValueError(f"Схема {s_name} отличается от базовой")
    _json(output_dir / "schema.json", first_schema)

    manifest = {
        "format": "craving-synthetic-v12",
        "created_at": "2026-10-17",
        "generator_version": "12.0.0",
        "cohorts": {
            split: {
                "users": count,
                "seed": seed,
                "rows": count * 180,
                "file_x": f"{split}.X.npy",
                "file_records": f"{split}.records.jsonl",
                "file_evaluation": f"{split}.evaluation.jsonl",
            }
            for split, count, seed in specs
        },
        "features_count": len(first_schema),
        "confirmation_locked": not (include_confirmation and stage_4_authorized),
        "summaries": summaries,
    }
    _json(output_dir / "manifest.json", manifest)
    return manifest
