"""Проверки переноса: воспроизводимость, причинность и автономность корпуса."""

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from synthetic_data import generate
from synthetic_data.contexts import history_contexts
from synthetic_data.features import build_history_features
from synthetic_data.generate import generate_dataset
from synthetic_data.nutrifit_encoder import encode_histories
from synthetic_data.population import SyntheticPopulationGenerator
from synthetic_data.simulator import LongitudinalSimulator
from synthetic_data.types import FeedbackRecord, InformationFeature


@pytest.fixture(scope="module")
def history():
    user = SyntheticPopulationGenerator(42).generate(1)[0]
    return LongitudinalSimulator(42).simulate_user_180_days(user)


def test_reproducible_complete_corpus(tmp_path):
    """Одинаковый seed даёт одинаковые артефакты; пропуски не становятся нулями."""
    first = generate_dataset(tmp_path / "first", 20261016, (1, 1, 1))
    second = generate_dataset(tmp_path / "second", 20261016, (1, 1, 1))
    assert first == second
    assert not (tmp_path / "first/.incomplete").exists()
    for name, digest in first["artifacts"].items():
        assert (
            hashlib.sha256((tmp_path / "first" / name).read_bytes()).hexdigest()
            == digest
        )
    schema = json.loads((tmp_path / "first/schema.json").read_text())
    users = set()
    for split in ("training", "calibration", "development"):
        root = tmp_path / "first"
        current = np.load(root / f"{split}.X.npy", allow_pickle=False)
        historical = np.load(root / f"{split}.history.X.npy", allow_pickle=False)
        assert current.shape == (180, len(schema["current"]))
        assert historical.shape == (180, len(schema["history"]))
        assert np.isfinite(historical).all()
        rows = [
            json.loads(line)
            for line in (root / f"{split}.records.jsonl").read_text().splitlines()
        ]
        assert any(row["outcome"] is None for row in rows)
        assert all(
            (row["outcome"] is None) == (row["feedback_at"] is None) for row in rows
        )
        identities = {row["user_id"] for row in rows}
        assert not users & identities
        users.update(identities)
        assert first["splits"][split]["quality"]["context_violations"] == 0
    different = generate_dataset(tmp_path / "different", 20261017, (1, 1, 1))
    assert (
        first["artifacts"]["training.X.npy"] != different["artifacts"]["training.X.npy"]
    )


def test_existing_output_is_preserved(tmp_path):
    """Повтор запуска не перезаписывает чужие результаты."""
    protected = tmp_path / "keep.txt"
    protected.write_text("keep")
    with pytest.raises(FileExistsError):
        generate_dataset(tmp_path, 42, (1, 1, 1))
    assert protected.read_text() == "keep"


@pytest.mark.parametrize("directory", ["data", "results"])
def test_frozen_directories_are_protected(directory):
    """Новый корпус нельзя записать внутрь замороженных данных и результатов."""
    output = (
        Path(generate.__file__).resolve().parents[1]
        / directory
        / "forbidden-new-corpus"
    )
    with pytest.raises(ValueError, match="Замороженные"):
        generate_dataset(output, 42, (1, 1, 1))
    assert not output.exists()


def test_failed_integration_keeps_incomplete_marker(tmp_path):
    """Недоступный адаптер не оставляет корпус, выглядящий завершённым."""
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="tsx"):
        generate_dataset(
            output, 42, (1, 1, 1), backend_root=tmp_path / "absent-backend"
        )
    assert (output / ".incomplete").exists()
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("seed,counts", [(-1, (1, 1, 1)), (1, (0, 1, 1)), (1, (1, 1))])
def test_invalid_cohorts_do_not_create_output(tmp_path, seed, counts):
    """Ошибочные параметры отклоняются до создания файлов."""
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        generate_dataset(output, seed, counts)
    assert not output.exists()


def test_hidden_outcomes_do_not_change_observed_contexts(history):
    """Скрытые исходы оценщика не влияют на вход энкодера."""
    changed = [
        (
            replace(
                truth,
                relapse_occurred=not truth.relapse_occurred,
                true_relapse_prob=1 - truth.true_relapse_prob,
            ),
            context,
        )
        for truth, context in history
    ]
    assert history_contexts(changed) == history_contexts(history)


def test_contexts_exclude_unavailable_records(history):
    """Поздняя публикация еды не доступна в момент решения."""
    truth, context = next(row for row in history if row[1].meals)
    late = context.model_copy(deep=True)
    for meal in late.meals:
        meal.availableAt = late.evaluationTime + timedelta(days=1)
    prepared, exclusions = history_contexts([(truth, late)])
    assert prepared[0]["meals"] == []
    assert exclusions["not_yet_available_records"] >= len(late.meals)


def test_future_feedback_and_other_users_do_not_leak():
    """Обратная связь появляется через 20 часов и не переносится между людьми."""
    at = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        FeedbackRecord("training", "a", "a0", at, at, 1, at + timedelta(hours=20)),
        FeedbackRecord(
            "training",
            "a",
            "a1",
            at + timedelta(hours=10),
            at + timedelta(hours=10),
            None,
            None,
        ),
        FeedbackRecord(
            "training",
            "a",
            "a2",
            at + timedelta(days=1),
            at + timedelta(days=1),
            None,
            None,
        ),
        FeedbackRecord(
            "training",
            "b",
            "b0",
            at + timedelta(days=1),
            at + timedelta(days=1),
            None,
            None,
        ),
    ]
    schema = (InformationFeature("value", "unit", "Наблюдаемое значение"),)
    current = np.array([[2.0], [4.0], [8.0], [99.0]])
    values, extended = build_history_features(records, current, schema)
    columns = {field.name: index for index, field in enumerate(extended)}
    count = columns["feedback_all_reported_count"]
    assert values[:, count].tolist() == [0, 0, 1, 0]
    assert values[2, columns["history_7d_mean__value"]] == 3
    assert values[3, columns["history_7d_count__value"]] == 0
    changed = [replace(records[0], outcome=0), *records[1:]]
    other, _ = build_history_features(changed, current, schema)
    np.testing.assert_array_equal(values[:2], other[:2])
    assert values[2, columns["feedback_all_positive_count"]] == 1
    assert other[2, columns["feedback_all_positive_count"]] == 0


def test_missing_backend_is_explicit(tmp_path, history):
    """Отсутствие локального checkout не запускает установку или поиск архива."""
    with pytest.raises(RuntimeError, match="tsx"):
        encode_histories([history], backend_root=tmp_path)


def test_local_backend_adapter(history, tmp_path):
    """Реальный TS-энкодер работает без архива и возвращает provenance и готовность."""
    backend = os.environ.get("NUTRIFIT_BACKEND_ROOT")
    if backend is None:
        pytest.skip("Для интеграции нужен явно заданный локальный NutriFit checkout")
    root = Path(backend)
    assert not (root / "research/craving-engine-archive").exists()
    vectors, metadata = encode_histories([history], backend_root=root)
    repeated, repeated_metadata = encode_histories([history], backend_root=root)
    assert vectors[0].shape == (180, 16)
    assert np.isfinite(vectors[0]).all()
    assert np.all((vectors[0] >= 0) & (vectors[0] <= 1))
    np.testing.assert_array_equal(vectors[0], repeated[0])
    assert metadata == repeated_metadata
    assert metadata["readiness"]["rows"] == 180
    assert all(
        "craving-engine-archive" not in name for name in metadata["source_sha256"]
    )
    assert metadata["runtime"]["apiContractSha256"]
    with pytest.raises(ValueError, match="повторяющиеся"):
        encode_histories([history, history], backend_root=root)
    empty, empty_metadata = encode_histories([], backend_root=root)
    assert empty == []
    assert empty_metadata["readiness"]["rows"] == 0
    plain = tmp_path / "plain"
    integrated = tmp_path / "integrated"
    generate_dataset(plain, 20261016, (1, 1, 1))
    manifest = generate_dataset(integrated, 20261016, (1, 1, 1), backend_root=root)
    assert manifest["features"] == "observed_with_local_nutrifit_pn"
    for split in ("training", "calibration", "development"):
        observed = np.load(plain / f"{split}.X.npy", allow_pickle=False)
        with_pn = np.load(integrated / f"{split}.X.npy", allow_pickle=False)
        assert with_pn.shape == (180, observed.shape[1] + 16)
        np.testing.assert_array_equal(observed, with_pn[:, 16:])
