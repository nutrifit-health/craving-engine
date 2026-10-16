"""Создаёт новый синтетический корпус без обучения и без изменения V9/V10.

Истинные исходы записываются отдельно от наблюдаемых строк и признаков.
Доступный feedback имеет явное время; пропуск ответа не становится нулём.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from datetime import UTC, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np

from synthetic_data.contexts import history_contexts
from synthetic_data.features import (
    build_history_features,
    observed_schema,
    observed_summary,
)
from synthetic_data.population import SyntheticPopulationGenerator
from synthetic_data.quality import summarize_dataset
from synthetic_data.simulator import LongitudinalSimulator
from synthetic_data.types import FeedbackRecord, InformationFeature


def _json(path: Path, value: object) -> None:
    """Создаёт новый файл и запрещает перезапись существующего результата."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _array(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def generate_dataset(
    output: Path,
    seed: int,
    user_counts: tuple[int, int, int],
    backend_root: Path | None = None,
) -> dict:
    """Фиксирует отдельный новый запуск; не объявляет его слепой проверкой."""
    if seed < 0 or len(user_counts) != 3 or any(count < 1 for count in user_counts):
        raise ValueError(
            "Нужны неотрицательный seed и три положительных размера когорт"
        )
    output = Path(output).expanduser().resolve()
    project = Path(__file__).resolve().parents[1]
    if any(output.is_relative_to(project / name) for name in ("data", "results")):
        raise ValueError(
            "Замороженные data/ и results/ не предназначены для новых корпусов; используйте runs/"
        )
    output.mkdir(parents=True, exist_ok=False)
    marker = output / ".incomplete"
    marker.write_text(
        "Корпус не завершён; использовать его нельзя.\n", encoding="utf-8"
    )
    source_dir = output / "sources"
    source_dir.mkdir()
    source_hashes = {}
    for source in sorted(Path(__file__).parent.glob("*.py")):
        content = source.read_bytes()
        (source_dir / source.name).write_bytes(content)
        source_hashes[source.name] = hashlib.sha256(content).hexdigest()

    summaries = {}
    seen_profiles: set[str] = set()
    schemas = {}
    for offset, (split, count) in enumerate(
        zip(("training", "calibration", "development"), user_counts, strict=True)
    ):
        split_seed = seed + offset
        population = SyntheticPopulationGenerator(split_seed).generate(count)
        simulator = LongitudinalSimulator(split_seed)
        histories = []
        profile_hashes = {}
        for index, profile in enumerate(population, start=1):
            user = replace(
                profile, user_id=f"S{seed}-{split}-{index:03d}", persona_id=None
            )
            parameters = {
                key: value
                for key, value in user.to_dict().items()
                if key
                not in {
                    "user_id",
                    "persona_id",
                    "persona_name",
                    "persona_category",
                    "description",
                }
            }
            digest = hashlib.sha256(
                json.dumps(parameters, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if digest in seen_profiles:
                raise ValueError(
                    "Повтор полного профиля: не заменяйте пользователя после просмотра исходов"
                )
            seen_profiles.add(digest)
            profile_hashes[user.user_id] = digest
            histories.append(simulator.simulate_user_180_days(user))
        quality = summarize_dataset(histories)
        contexts = [
            context for history in histories for context in history_contexts(history)[0]
        ]
        schema = observed_schema()
        summaries_raw = [observed_summary(context) for context in contexts]
        current = np.asarray(
            [[row[field.name] for field in schema] for row in summaries_raw],
            dtype=np.float64,
        )
        encoder = None
        if backend_root is not None:
            from synthetic_data.nutrifit_encoder import encode_histories

            vectors, encoder = encode_histories(histories, backend_root=backend_root)
            axes = encoder["runtime"]["axes"]
            schema = (
                tuple(
                    InformationFeature(
                        f"pn_{axis}", "0..1", f"Локальный энкодер NutriFit: {axis}"
                    )
                    for axis in axes
                )
                + schema
            )
            current = np.concatenate(
                (np.concatenate(vectors).astype(np.float64), current), axis=1
            )

        records = []
        evaluators = []
        feedback_rows = []
        for history in histories:
            user_id = history[0][1].userId
            feedback_seed = int.from_bytes(
                hashlib.sha256(f"{split_seed}:{user_id}:feedback".encode()).digest()[
                    :8
                ],
                "big",
            )
            rng = np.random.default_rng(feedback_seed)
            for index, (truth, context) in enumerate(history):
                at = context.evaluationTime.replace(tzinfo=UTC)
                actual = int(truth.relapse_occurred)
                reported = bool(context.meals or context.wellbeing) and rng.random() < (
                    0.8 if actual else 0.7
                )
                feedback_at = at + timedelta(hours=20) if reported else None
                event_id = f"{user_id}:{index}"
                row = FeedbackRecord(
                    split,
                    user_id,
                    event_id,
                    at,
                    at,
                    actual if reported else None,
                    feedback_at,
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
                        "evaluation_outcome": actual,
                        "evaluation_probability": truth.true_relapse_prob,
                    }
                )
        history_values, history_schema = build_history_features(
            feedback_rows, current, schema
        )
        for name, value in (("current", schema), ("history", history_schema)):
            serialized = [asdict(field) for field in value]
            if name in schemas and schemas[name] != serialized:
                raise ValueError("Схема признаков не должна зависеть от когорты")
            schemas[name] = serialized
        _array(output / f"{split}.X.npy", current)
        _array(output / f"{split}.history.X.npy", history_values)
        _jsonl(output / f"{split}.records.jsonl", records)
        _jsonl(output / f"{split}.evaluation.jsonl", evaluators)
        _jsonl(output / f"{split}.contexts.jsonl", contexts)
        summaries[split] = {
            "seed": split_seed,
            "users": count,
            "rows": len(records),
            "profile_sha256": profile_hashes,
            "quality": quality,
            "encoder": encoder,
        }
    _json(output / "schema.json", schemas)
    artifacts = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != marker
    }
    manifest = {
        "format": "craving-synthetic-observed-v1",
        "synthetic_only": True,
        "historical_reproduction": False,
        "blind_holdout": False,
        "seed": seed,
        "days_per_user": 180,
        "splits": summaries,
        "features": "observed_with_local_nutrifit_pn"
        if backend_root is not None
        else "observed_without_pn",
        "feedback": {
            "positive_reporting": 0.8,
            "negative_reporting": 0.7,
            "delay_hours": 20,
        },
        "source_sha256": source_hashes,
        "artifacts": artifacts,
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pydantic": version("pydantic"),
        },
        "limitations": [
            "Новый seed не доказывает перенос на реальных людей.",
            "evaluation.jsonl и скрытые параметры не являются входами обучения.",
            "records.outcome доступен только начиная с feedback_at; отсутствие ответа не является отрицательным исходом.",
            "Формат не заменяет замороженный bundle V9; старые результаты не пересчитываются.",
        ],
    }
    _json(output / "manifest.json", manifest)
    marker.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Новый синтетический корпус Craving Engine"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--users",
        type=int,
        nargs=3,
        default=(60, 20, 40),
        metavar=("TRAIN", "CAL", "DEV"),
    )
    parser.add_argument(
        "--backend-root",
        type=Path,
        help="Необязательный явный путь к локальному apps/backend NutriFit",
    )
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output, args.seed, tuple(args.users), args.backend_root
    )
    print(
        json.dumps(
            {"output": str(args.output), "features": manifest["features"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
