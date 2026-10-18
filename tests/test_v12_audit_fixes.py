"""Регрессии конкретных дефектов аудита V12; локальные runs не нужны."""

import copy
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from research_v11.types import MemoryConfigV11
from research_v12.baseline import load_frozen_scalar
from research_v12.memory_benchmark import measure_memory_stream
from research_v12.models import HierarchicalRidgeMemory, MoEMemory, TunedScalarMemory
from research_v12.robustness import transform_feedback_stream
from research_v12.verification import compare_scores


def test_frozen_scalar_loads_artifact_instead_of_defaults(tmp_path):
    """Замороженные скорости 0.2/0.05 нельзя заменить defaults 0.1/0.025."""
    config = MemoryConfigV11(fast_rate=0.2, slow_rate=0.05)
    path = tmp_path / "frozen_config.json"
    path.write_text(json.dumps({"scalar": {"d": 0, **config.model_dump()}}))
    loaded, provenance = load_frozen_scalar(path)
    assert loaded == config
    assert loaded != MemoryConfigV11()
    assert provenance["config"] == config.model_dump()
    incomplete = {"d": 0, **config.model_dump()}
    incomplete.pop("fast_rate")
    path.write_text(json.dumps({"scalar": incomplete}))
    with pytest.raises(ValueError, match="все параметры"):
        load_frozen_scalar(path)


def _drift_stream():
    records, evaluations = [], []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for day in range(180):
        at = start + timedelta(days=day)
        reported = day % 3 != 0
        records.append(
            {
                "user_id": "u",
                "event_id": f"u:{day}",
                "decision_at": at.isoformat(),
                "outcome": 0 if reported else None,
                "feedback_at": (at + timedelta(hours=20)).isoformat()
                if reported
                else None,
            }
        )
        evaluations.append(
            {
                "event_id": f"u:{day}",
                "day_index": day,
                "evaluation_outcome": 0,
                "evaluation_probability": 0.0,
            }
        )
    return records, evaluations


def test_drift_changes_truth_and_report_only_inside_declared_window():
    """Номер дня берётся из evaluation; окна дрифта реально меняют истину и ответы."""
    records, evaluations = _drift_stream()
    original = copy.deepcopy((records, evaluations))
    outputs = {}
    for scenario in ("abrupt_drift", "temporary_drift"):
        rows, truth = transform_feedback_stream(records, evaluations, scenario)
        outputs[scenario] = truth
        active_days = range(90, 180) if scenario == "abrupt_drift" else range(60, 90)
        for day, (row, event) in enumerate(zip(rows, truth, strict=True)):
            assert event["evaluation_probability"] == (
                0.4 if day in active_days else 0.0
            )
            if day not in active_days:
                assert event == evaluations[day]
                assert row == records[day]
            if records[day]["outcome"] is not None:
                assert row["outcome"] == event["evaluation_outcome"]
            else:
                assert row["outcome"] is None
        assert any(e["evaluation_outcome"] == 1 for e in truth)
    assert (records, evaluations) == original
    broken = copy.deepcopy(evaluations)
    broken[0].pop("day_index")
    with pytest.raises(KeyError):
        transform_feedback_stream(records, broken, "abrupt_drift")


@pytest.mark.parametrize("cls", [TunedScalarMemory, HierarchicalRidgeMemory, MoEMemory])
def test_checkpoint_preserves_pending_and_future_learning(cls):
    """После восстановления с несколькими pending дальнейшее обучение тождественно."""
    memory = cls()
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def predict(model, event, at):
        if cls is HierarchicalRidgeMemory:
            return model.predict(event, at, np.array([1.0, -2.0, 0.5, 3.0]), 0.2)
        if cls is MoEMemory:
            return model.predict(event, at, np.array([0.1, 0.25, 0.7]))
        return model.predict(event, at, 0.2)

    for day in range(3):
        predict(memory, f"e{day}", start + timedelta(days=day))
    restored = cls.from_state(memory.export_state())
    assert restored.export_state() == memory.export_state()
    for day in range(3, 23):
        at = start + timedelta(days=day)
        for model in (memory, restored):
            model.observe(f"e{day - 3}", at, day % 2, confidence=0.8)
            predict(model, f"e{day}", at + timedelta(hours=1))
        assert restored.export_state() == memory.export_state()
        assert restored.state_size_bytes() == len(
            restored.export_state().encode("utf-8")
        )
    before = memory.export_state()
    with pytest.raises(ValueError):
        memory.observe("e22", start, 1)
    assert memory.export_state() == before


@pytest.mark.parametrize(
    "memory", [TunedScalarMemory(), HierarchicalRidgeMemory(dimension=8)]
)
def test_peak_measurement_includes_pending_feedback_and_every_event(memory):
    """Пик вычисляется по событиям, а не по последней точке 180/365 дней."""
    result = measure_memory_stream(memory)
    assert result["max_pending"] >= 3
    rows = result["measurements"]
    assert {r["phase"] for r in rows} == {"predict", "observe", "expire"}
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    for days in (180, 365):
        end = start + timedelta(days=days)
        peak = max(r["bytes"] for r in rows if datetime.fromisoformat(r["at"]) < end)
        assert result[f"peak_bytes_{days}_days"] == peak
    assert result["peak_including_drain"] == max(r["bytes"] for r in rows)
    assert result["within_budget"] == (result["peak_including_drain"] <= 16384)
    assert result["within_budget"]


def test_score_verification_rejects_changed_predictions_and_duplicate_events():
    """Совпадения агрегата недостаточно: сверяются каждое решение и его исход."""
    scores = {
        "scalar": {
            "dev_1": [
                {
                    "user_id": "u",
                    "event_id": "e",
                    "p_final": 0.2,
                    "evaluation_outcome": 1,
                    "feedbacks_processed": 0,
                }
            ]
        }
    }
    compare_scores(scores, copy.deepcopy(scores))
    modified = copy.deepcopy(scores)
    modified["scalar"]["dev_1"][0]["p_final"] = 0.3
    with pytest.raises(AssertionError):
        compare_scores(scores, modified)
    modified["scalar"]["dev_1"].append(modified["scalar"]["dev_1"][0])
    with pytest.raises(AssertionError):
        compare_scores(scores, modified)


def test_incomplete_pending_context_cannot_restore():
    """Checkpoint без z не считается полным и не принимается для продолжения обучения."""
    memory = HierarchicalRidgeMemory(dimension=2)
    memory.predict(
        "pending", datetime(2026, 1, 1, tzinfo=UTC), np.array([1.0, -1.0]), 0.3
    )
    payload = json.loads(memory.export_state())
    payload["state"]["pending"]["pending"].pop("z")
    with pytest.raises(ValueError, match="контекст"):
        HierarchicalRidgeMemory.from_state(json.dumps(payload))
