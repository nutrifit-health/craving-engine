"""Загрузка фактически замороженного scalar V11 без defaults и переобучения."""

from __future__ import annotations

import json
from pathlib import Path

from research_v11.types import MemoryConfigV11
from research_v12.registry import file_sha256


def load_frozen_scalar(path: Path) -> tuple[MemoryConfigV11, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scalar = dict(raw["scalar"])
    if scalar.pop("d", None) != 0:
        raise ValueError("Ожидается замороженная конфигурация scalar с d=0")
    required = set(MemoryConfigV11.model_fields)
    if set(scalar) != required:
        raise ValueError("Frozen config должен явно содержать все параметры памяти V11")
    config = MemoryConfigV11.model_validate(scalar)
    return config, {
        "source": str(path.resolve()),
        "sha256": file_sha256(path),
        "config": config.model_dump(),
    }
