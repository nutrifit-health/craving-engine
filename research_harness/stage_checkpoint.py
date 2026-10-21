"""Атомарные контрольные точки данных: JSON/NumPy, без pickle и default=str.

Файл artifacts/checkpoint.json экспортируется OCI как обычный именованный
артефакт. Возобновление разрешено только при совпадении полного контекста;
независимый replay начинается без resume-файла.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

MAX_BYTES = 1024 * 1024


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise TypeError("Разрешены только числовые массивы")
        return {
            "__ndarray__": value.tolist(),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Неподдерживаемый тип checkpoint: " + type(value).__name__)


def decode(value):
    if "__ndarray__" in value:
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Нечисловой dtype")
        result = np.asarray(value["__ndarray__"], dtype=dtype)
        if list(result.shape) != value["shape"]:
            raise ValueError("Форма checkpoint изменилась")
        return result
    return value


class StageCheckpoint:
    def __init__(self, context, path="artifacts/checkpoint.json", resume=None):
        if not isinstance(context, str) or not re.fullmatch("[0-9a-f]{64}", context):
            raise ValueError("Нужен SHA256 полного входного контекста")
        self.context, self.path = context, Path(path)
        self.values = {}
        if resume is not None:
            source = Path(resume)
            if source.is_symlink() or source.stat().st_size > MAX_BYTES:
                raise ValueError("Недопустимый checkpoint")
            data = json.loads(source.read_text(), object_hook=decode)
            if (
                data.get("schema") != "craving-stage-checkpoint/v1"
                or data.get("context") != context
            ):
                raise ValueError("Checkpoint относится к другой программе/данным")
            self.values = data["values"]
        self._write()

    def _write(self):
        data = json.dumps(
            {
                "schema": "craving-stage-checkpoint/v1",
                "context": self.context,
                "values": self.values,
            },
            default=encode,
            allow_nan=False,
            ensure_ascii=False,
        ).encode()
        if len(data) > MAX_BYTES:
            raise ValueError("Checkpoint превышает 1 МиБ")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            with tmp.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def has(self, stage_id):
        return stage_id in self.values

    def load(self, stage_id):
        return self.values[stage_id]

    def save(self, stage_id, data):
        if not isinstance(stage_id, str) or not re.fullmatch(
            "[a-zA-Z0-9_-]{1,80}", stage_id
        ):
            raise ValueError("Неверный stage_id")
        previous = dict(self.values)
        self.values[stage_id] = data
        try:
            self._write()
        except (TypeError, ValueError, OSError):
            self.values = previous
            raise
        return self.path

    def load_or_compute(self, stage_id, compute_fn):
        if self.has(stage_id):
            return self.load(stage_id)
        result = compute_fn()
        self.save(stage_id, result)
        return result
