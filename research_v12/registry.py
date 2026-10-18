"""Машинно-читаемый реестр запусков V12 (run_registry.json).

Управляет жизненным циклом запусков: planned -> running -> completed / failed.
Защищает от перезаписи завершенных прогонов и логирует входные хеши, конфигурации и метрики.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

RunStatus = Literal["planned", "running", "completed", "failed"]
DEFAULT_REGISTRY_PATH = Path(
    "/Users/gurutom/Develop/NutriFit/apps/craving-engine/runs/v12-registry/run_registry.json"
)


def file_sha256(path: Path | str) -> str:
    """Вычисляет SHA-256 хеш файла."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Файл не найден: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class RunRegistry:
    """Управление реестром запусков V12."""

    def __init__(self, path: Path | str = DEFAULT_REGISTRY_PATH) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save(
                {
                    "format": "craving-engine-v12-registry",
                    "version": "1.0.0",
                    "created_at": datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "runs": [],
                }
            )

    def _load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = datetime.now(UTC).isoformat()
        temp_path = self.path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        temp_path.replace(self.path)

    def register(
        self,
        run_id: str,
        stage: int,
        family: str,
        config: dict[str, Any],
        inputs: dict[str, str] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Регистрирует новый запланированный запуск."""
        data = self._load()
        for r in data["runs"]:
            if r["run_id"] == run_id:
                raise ValueError(
                    f"Запуск с run_id='{run_id}' уже существует в реестре. Повторные попытки сохраняйте отдельно."
                )

        entry: dict[str, Any] = {
            "run_id": run_id,
            "stage": stage,
            "family": family,
            "status": "planned",
            "config": config,
            "inputs": inputs or {},
            "metrics": {},
            "created_at": datetime.now(UTC).isoformat(),
            "started_at": None,
            "completed_at": None,
            "execution_time_seconds": None,
            "stop_reason": None,
            "notes": notes,
        }
        data["runs"].append(entry)
        self._save(data)
        return entry

    def mark_running(self, run_id: str) -> None:
        """Переводит запуск в статус running."""
        data = self._load()
        for r in data["runs"]:
            if r["run_id"] == run_id:
                if r["status"] == "completed":
                    raise RuntimeError(
                        f"Запуск {run_id} уже завершен, повторный запуск запрещен"
                    )
                r["status"] = "running"
                r["started_at"] = datetime.now(UTC).isoformat()
                self._save(data)
                return
        raise KeyError(f"Запуск {run_id} не найден в реестре")

    def mark_completed(
        self,
        run_id: str,
        metrics: dict[str, Any],
        execution_time_seconds: float,
        stop_reason: str | None = None,
    ) -> None:
        """Завершает запуск с сохранением метрик."""
        data = self._load()
        for r in data["runs"]:
            if r["run_id"] == run_id:
                r["status"] = "completed"
                r["completed_at"] = datetime.now(UTC).isoformat()
                r["metrics"] = metrics
                r["execution_time_seconds"] = execution_time_seconds
                r["stop_reason"] = stop_reason
                self._save(data)
                return
        raise KeyError(f"Запуск {run_id} не найден в реестре")

    def mark_failed(
        self,
        run_id: str,
        error_message: str,
        execution_time_seconds: float | None = None,
    ) -> None:
        """Помечает запуск как неудачный с фиксацией ошибки."""
        data = self._load()
        for r in data["runs"]:
            if r["run_id"] == run_id:
                r["status"] = "failed"
                r["completed_at"] = datetime.now(UTC).isoformat()
                r["stop_reason"] = error_message
                r["execution_time_seconds"] = execution_time_seconds
                self._save(data)
                return
        raise KeyError(f"Запуск {run_id} не найден в реестре")

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Возвращает запись запуска."""
        data = self._load()
        for r in data["runs"]:
            if r["run_id"] == run_id:
                return r
        raise KeyError(f"Запуск {run_id} не найден в реестре")

    def list_runs(self, stage: int | None = None) -> list[dict[str, Any]]:
        """Возвращает список запусков (с возможностью фильтрации по этапу)."""
        data = self._load()
        runs = data["runs"]
        if stage is not None:
            runs = [r for r in runs if r["stage"] == stage]
        return runs
