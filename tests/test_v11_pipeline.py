"""Автономные проверки реального пайплайна, причинности и защиты заморозки."""

import gzip
import json
import pickle
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from research_v11 import experiment, provenance
from research_v11.encoders import CompactKCEncoder, CompactLinearEncoder
from research_v11.gbdt import PopulationGBDTTrainer
from research_v11.runner import run_v11_replay

FAMILIES = (
    "population",
    "scalar",
    "compact_linear",
    "compact_kc_anatomical",
    "compact_kc_rewired",
)


def records(split, count=32):
    """Создаёт наблюдения с задержками, пропусками и обоими классами."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows, outcomes = [], []
    for i in range(count):
        at = start + timedelta(days=i)
        event = f"{split}-{i}"
        rows.append(
            {
                "user_id": split,
                "event_id": event,
                "decision_at": at.isoformat(),
                "outcome": i % 2 if i % 5 else None,
                "confidence": 1.0,
                "feedback_at": (at + timedelta(hours=20 if i % 3 else 48)).isoformat()
                if i % 5
                else None,
            }
        )
        outcomes.append(
            {"user_id": split, "event_id": event, "evaluation_outcome": i % 2}
        )
    return rows, outcomes


@pytest.fixture
def corpus(tmp_path):
    """Малый корпус не требует runs/, сети или backend."""
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(731)
    schema = [
        {
            "name": f"x{i}",
            "unit": "1",
            "semantics": "Тестовый признак",
            "available_if": None,
        }
        for i in range(10)
    ]
    (data / "schema.json").write_text(json.dumps(schema))
    for index, split in enumerate(("training", "tuning", "calibration", "holdout")):
        np.save(data / f"{split}.X.npy", rng.normal(loc=index, size=(96, 10)))
        rows, outcomes = [], []
        for user in range(3):
            user_rows, user_outcomes = records(f"{split}-{user}")
            rows.extend(user_rows)
            outcomes.extend(user_outcomes)
        for suffix, values in (("records", rows), ("evaluation", outcomes)):
            (data / f"{split}.{suffix}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in values)
            )
    graph = tmp_path / "graph.npz"
    rows = np.repeat(np.arange(128), 4)
    cols = np.concatenate([rng.choice(16, 4, replace=False) for _ in range(128)])
    np.savez(
        graph,
        rows=rows,
        columns=cols,
        weights=np.ones(len(rows)),
        kc_ids=np.arange(128),
        pn_ids=np.arange(16),
        metadata=json.dumps({"purpose": "test"}),
    )
    return data, graph


def test_real_pipeline_training_isolation_and_immutability(
    corpus, tmp_path, monkeypatch
):
    """Реальные fit вызываются только на training; оценка не переобучает и не меняет объекты."""
    data, graph = corpus
    X_train = np.load(data / "training.X.npy")
    calls = []
    for cls in (PopulationGBDTTrainer, CompactLinearEncoder, CompactKCEncoder):
        original = cls.fit

        def guarded_fit(self, X, *args, _original=original, **kwargs):
            np.testing.assert_array_equal(X, X_train)
            calls.append(type(self).__name__)
            return _original(self, X, *args, **kwargs)

        monkeypatch.setattr(cls, "fit", guarded_fit)
    original_load = np.load
    original_read = experiment._read_jsonl

    def no_holdout_load(path, *args, **kwargs):
        assert not Path(path).name.startswith("holdout")
        return original_load(path, *args, **kwargs)

    def no_holdout_read(path):
        assert not path.name.startswith("holdout")
        return original_read(path)

    monkeypatch.setattr(np, "load", no_holdout_load)
    monkeypatch.setattr(experiment, "_read_jsonl", no_holdout_read)
    artifacts = tmp_path / "frozen"
    experiment.fit_and_freeze_v11(data, graph, artifacts)
    assert sorted(calls) == [
        "CompactKCEncoder",
        "CompactKCEncoder",
        "CompactLinearEncoder",
        "PopulationGBDTTrainer",
    ]
    monkeypatch.setattr(np, "load", original_load)
    monkeypatch.setattr(experiment, "_read_jsonl", original_read)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Оценка не должна вызывать fit")

    for cls in (PopulationGBDTTrainer, CompactLinearEncoder, CompactKCEncoder):
        monkeypatch.setattr(cls, "fit", forbidden_fit)
    loaded = []
    original_pickle_load = pickle.load

    def tracked_load(stream):
        obj = original_pickle_load(stream)
        loaded.append((obj, pickle.dumps(obj)))
        return obj

    monkeypatch.setattr(pickle, "load", tracked_load)
    output = tmp_path / "evaluation"
    experiment.evaluate_frozen_v11(data, artifacts, output)
    assert len(loaded) == 2
    for obj, before in loaded:
        assert pickle.dumps(obj) == before
    with gzip.open(output / "holdout_scores.json.gz", "rt") as stream:
        scores = json.load(stream)
    assert len(scores) == 96 * 5
    aligned = {}
    for row in scores:
        aligned.setdefault(row["event_id"], set()).add(row["feedbacks_processed"])
    assert all(len(values) == 1 for values in aligned.values())
    assert (output / "evaluation_manifest.json").is_file()


@pytest.mark.parametrize("family", FAMILIES)
def test_runner_future_outcomes_and_equal_time_boundary(family):
    """Изменение будущего ответа/оценочной метки не влияет на прошлые решения runner."""
    rows, outcomes = records("causality", count=20)
    # Первый ответ поступит точно в момент третьего решения: сначала решение, затем ответ.
    rows[0]["outcome"] = 0
    rows[0]["feedback_at"] = rows[2]["decision_at"]
    altered = deepcopy(rows)
    altered[0]["outcome"] = 1
    truth = deepcopy(outcomes)
    for row in truth:
        row["evaluation_outcome"] = 1 - row["evaluation_outcome"]
    dim = 0 if family in ("population", "scalar") else 2
    z = np.ones((len(rows), dim)) if dim else None
    before, _ = run_v11_replay(rows, outcomes, np.full(len(rows), 0.3), z, family, dim)
    after, _ = run_v11_replay(altered, truth, np.full(len(rows), 0.3), z, family, dim)
    for first, second in zip(before[:3], after[:3], strict=True):
        assert {k: v for k, v in first.items() if k != "evaluation_outcome"} == {
            k: v for k, v in second.items() if k != "evaluation_outcome"
        }
    if family != "population":
        assert before[3]["p_learning"] != after[3]["p_learning"]
    else:
        assert all(r["p_final"] == 0.3 for r in after)
    assert before[2]["feedbacks_processed"] == 1
    assert before[3]["feedbacks_processed"] == 3


@pytest.fixture
def sealed(corpus, tmp_path):
    """Манифест тестируется до десериализации; содержимое модели здесь намеренно не pickle."""
    data, graph = corpus
    artifacts = tmp_path / "sealed"
    for name in provenance.ARTIFACTS:
        path = artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed")
    provenance.seal_artifacts(artifacts, data, graph, status="test")
    return artifacts, data


@pytest.mark.parametrize(
    "target",
    ["model", "config", "data", "snapshot", "source", "environment", "manifest"],
)
def test_integrity_rejected_before_deserialization(
    sealed, tmp_path, monkeypatch, target
):
    """Изменённые входы, окружение и исходники блокируются до загрузки моделей."""
    artifacts, data = sealed
    if target in ("model", "config", "data", "snapshot"):
        paths = {
            "model": artifacts / "models/gbdt.pickle",
            "config": artifacts / "frozen_config.json",
            "data": data / "training.X.npy",
            "snapshot": artifacts / "source_snapshot/research_v11/memory.py",
        }
        paths[target].write_bytes(b"changed")
    elif target == "source":
        monkeypatch.setattr(provenance, "source_hashes", dict)
    elif target == "environment":
        monkeypatch.setattr(provenance, "environment", dict)
    else:
        (artifacts / "freeze_manifest.json").write_text('{"format_version":1}')
    monkeypatch.setattr(
        pickle,
        "load",
        lambda _: pytest.fail("Повреждённый комплект дошёл до десериализации"),
    )
    with pytest.raises(ValueError):
        experiment.evaluate_frozen_v11(data, artifacts, tmp_path / "out")


def test_partial_outputs_are_not_overwritten(sealed, tmp_path):
    """Частичный запуск защищён даже при отсутствии итогового summary."""
    artifacts, data = sealed
    out = tmp_path / "out"
    out.mkdir()
    (out / "holdout_scores.json.gz").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        experiment.evaluate_frozen_v11(data, artifacts, out)
    with pytest.raises(FileExistsError):
        experiment.fit_and_freeze_v11(data, tmp_path / "graph.npz", out)
    assert (out / "holdout_scores.json.gz").read_bytes() == b"original"
