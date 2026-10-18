"""Полный версионированный checkpoint исследовательской памяти V12."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

COMMON_CONFIG = (
    "fast_rate",
    "slow_ratio",
    "l2",
    "fast_half_life_days",
    "slow_half_life_days",
    "exposure_half_life_days",
    "max_logit_residual",
    "gate_window_days",
    "gate_min_effective_samples",
    "gate_min_class_weight",
    "minimum_gain",
    "uncertainty_penalty",
    "gate_temperature",
    "use_gate",
)
STATE_FIELDS = (
    "b_fast",
    "b_slow",
    "w_fast",
    "w_slow",
    "exposure",
    "clock",
    "pending",
    "observations",
    "log_alpha",
)


def _encode(value):
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist()}
    raise TypeError(f"Неподдерживаемый тип состояния: {type(value).__name__}")


def _decode(value):
    if set(value) == {"__datetime__"}:
        return datetime.fromisoformat(value["__datetime__"])
    if set(value) == {"__array__"}:
        return np.asarray(value["__array__"], dtype=np.float64)
    return value


class CheckpointMemory:
    """Сохраняет параметры и все данные, необходимые для продолжения обучения."""

    config_fields: tuple[str, ...] = ()

    def get_config(self) -> dict:
        return {name: getattr(self, name) for name in self.config_fields}

    def export_state(self) -> str:
        payload = {
            "version": 1,
            "model": type(self).__name__,
            "config": self.get_config(),
            "state": {
                name: getattr(self, name)
                for name in STATE_FIELDS
                if hasattr(self, name)
            },
        }
        return json.dumps(payload, default=_encode, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_state(cls, serialized: str):
        payload = json.loads(serialized, object_hook=_decode)
        if payload["version"] != 1 or payload["model"] != cls.__name__:
            raise ValueError("Неверная версия или тип checkpoint V12")
        memory = cls(**payload["config"])
        required = {name for name in STATE_FIELDS if hasattr(memory, name)}
        if set(payload["state"]) != required:
            raise ValueError("Checkpoint содержит неполное состояние")
        for snapshot in payload["state"]["pending"].values():
            required_snapshot = {"decision_at", "p_final"}
            if hasattr(memory, "b_fast"):
                required_snapshot |= {"p0", "p_learning", "p_candidate"}
            if hasattr(memory, "w_fast"):
                required_snapshot.add("z")
                if np.shape(snapshot.get("z")) != (memory.dimension,):
                    raise ValueError("В checkpoint отсутствует полный контекст pending")
            if hasattr(memory, "log_alpha"):
                required_snapshot.add("expert_probs")
                if np.shape(snapshot.get("expert_probs")) != (memory.k_experts,):
                    raise ValueError("В checkpoint отсутствуют вероятности экспертов")
            if not required_snapshot.issubset(snapshot):
                raise ValueError("Неполный снимок pending")
        for name, value in payload["state"].items():
            setattr(memory, name, value)
        return memory

    def state_size_bytes(self) -> int:
        return len(self.export_state().encode("utf-8"))
