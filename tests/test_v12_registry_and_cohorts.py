"""Тесты реестра запусков и генератора когорт V12."""

import pytest

from research_v12.cohorts import (
    CONFIRMATION_COHORT_SPECS,
    DEV_COHORT_SPECS,
    generate_v12_cohorts,
)
from research_v12.registry import RunRegistry


def test_registry_lifecycle(tmp_path):
    registry_file = tmp_path / "v12_registry.json"
    registry = RunRegistry(registry_file)

    run = registry.register(
        run_id="run-test-001",
        stage=0,
        family="test_family",
        config={"param": 42},
        inputs={"file1": "hash123"},
        notes="Test run note",
    )
    assert run["status"] == "planned"
    assert run["stage"] == 0

    # Запрет повторного добавления одинакового run_id
    with pytest.raises(ValueError, match="уже существует"):
        registry.register(
            run_id="run-test-001",
            stage=0,
            family="test_family",
            config={},
        )

    # Перевод в running
    registry.mark_running("run-test-001")
    assert registry.get_run("run-test-001")["status"] == "running"
    assert registry.get_run("run-test-001")["started_at"] is not None

    # Завершение
    registry.mark_completed(
        run_id="run-test-001",
        metrics={"brier": 0.123},
        execution_time_seconds=1.5,
        stop_reason="target_achieved",
    )
    completed = registry.get_run("run-test-001")
    assert completed["status"] == "completed"
    assert completed["metrics"]["brier"] == 0.123
    assert completed["execution_time_seconds"] == 1.5

    # Запрет повторного running для завершенного запуска
    with pytest.raises(RuntimeError, match="уже завершен"):
        registry.mark_running("run-test-001")

    # Регистрация и сбой второго запуска
    registry.register("run-test-002", stage=1, family="baseline", config={})
    registry.mark_running("run-test-002")
    registry.mark_failed(
        "run-test-002",
        error_message="Memory limit exceeded",
        execution_time_seconds=0.8,
    )
    failed = registry.get_run("run-test-002")
    assert failed["status"] == "failed"
    assert failed["stop_reason"] == "Memory limit exceeded"

    # Фильтрация по этапу
    stage0_runs = registry.list_runs(stage=0)
    assert len(stage0_runs) == 1
    assert stage0_runs[0]["run_id"] == "run-test-001"

    stage1_runs = registry.list_runs(stage=1)
    assert len(stage1_runs) == 1
    assert stage1_runs[0]["run_id"] == "run-test-002"


def test_cohorts_specs_and_confirmation_lock(tmp_path):
    # Проверка спецификаций сидов и размеров
    dev_dict = {name: (count, seed) for name, count, seed in DEV_COHORT_SPECS}
    assert dev_dict["training"] == (60, 20261201)
    assert dev_dict["tuning"] == (30, 20261202)
    assert dev_dict["dev_1"] == (40, 20261203)
    assert dev_dict["dev_2"] == (40, 20261204)
    assert dev_dict["dev_3"] == (40, 20261205)

    conf_seeds = [seed for _, _, seed in CONFIRMATION_COHORT_SPECS]
    assert conf_seeds == [20261211, 20261212, 20261213, 20261214, 20261215]

    # Попытка сгенерировать confirmation без авторизации должна падать с PermissionError
    with pytest.raises(PermissionError, match="заблокирована"):
        generate_v12_cohorts(
            tmp_path / "cohorts", include_confirmation=True, stage_4_authorized=False
        )
