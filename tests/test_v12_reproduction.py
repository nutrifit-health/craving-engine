"""Воспроизведение нового комплекта по явному пути; исторический архив не выдаётся за проверку исправлений."""

import os
from pathlib import Path

import pytest

from research_v12.verification import reproduce_bundle


def test_v12_bundle_replays_models_metrics_and_memory():
    """Опциональный интеграционный replay не зависит от артефактов в свежем клоне."""
    value = os.environ.get("V12_REPLAY_BUNDLE")
    if not value:
        pytest.skip("V12_REPLAY_BUNDLE не задан: исправленный комплект ещё не оценён")
    result = reproduce_bundle(Path(value))
    assert result == {
        "scores_replayed": True,
        "metrics_recomputed": True,
        "memory_replayed": True,
        "stress_replayed": True,
        "confirmation_accessed": False,
    }
