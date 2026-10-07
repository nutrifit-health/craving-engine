"""Загружает отобранные синтетические входы V9 без приватного архива NutriFit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_v8.archive import RunArchive, digest
from research_v8.data_types import V8Record
from research_v8.reuse import _path, _verify_files


SOURCE_PARENT_SHA256 = "56ca00b6a8e31cfce9fb315a55e1b60571a412515f561d0d4951a99e3527528d"
SOURCE_RUN_SHA256 = "7405d213c86267deb1bb450bb156482b7712f75ae0fe31144877102632e8fd36"
SPLIT_ROWS = {"training": 10800, "calibration": 3600, "development": 7200}
ARTIFACTS = {"manifest.json", "schema.json", "folds.json", "connectome.npz", "training.rows.jsonl",
             *(f"{split}.{suffix}" for split in SPLIT_ROWS for suffix in ("jsonl", "X.npy")),
             "calibration.gbdt.npy", "development.gbdt.npy"}


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(source.read_bytes())


def _array(archive: RunArchive, name: str, values: np.ndarray) -> Path:
    path = archive._path(name)
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return path


def _records(archive: RunArchive, name: str, rows: list[V8Record]) -> Path:
    path = archive._path(name)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")
    return path


def load_bundle(bundle: Path, archive: RunArchive) -> tuple[dict, dict, dict, list[dict]]:
    """Фиксирует bytes входов и их порядок; оценочные метки не входят в X."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version") != 1 or manifest.get("synthetic_only") is not True
            or manifest.get("source_manifest_sha256") != SOURCE_RUN_SHA256
            or set(manifest["artifacts"]) != ARTIFACTS - {"manifest.json"}):
        raise ValueError("Ожидается выделенный синтетический набор V9")
    _verify_files(bundle, manifest["artifacts"])
    for name in sorted(ARTIFACTS):
        _copy(_path(bundle, name), archive.path / "inputs" / name)
    copied = archive.path / "inputs"
    _verify_files(copied, manifest["artifacts"])
    schema = json.loads((copied / "schema.json").read_text(encoding="utf-8"))
    if len(schema) != 94:
        raise ValueError("V9 требует 94 закреплённых признака")
    corpus, matrices, probabilities, cohorts = {}, {}, {}, {}
    all_events = set()
    for split, count in SPLIT_ROWS.items():
        payloads = [json.loads(line) for line in (copied / f"{split}.jsonl").read_text().splitlines()]
        # Историческое поле p_population training не является входом MB-fit.
        rows = [V8Record.model_validate({**payload, "p_population": None}) for payload in payloads]
        matrix = np.load(copied / f"{split}.X.npy", allow_pickle=False)
        if len(rows) != count or matrix.shape != (count, 94) or not np.isfinite(matrix).all():
            raise ValueError("Форма входов не соответствует синтетической серии V9")
        if any(row.split != split or row.evaluation_outcome is None for row in rows):
            raise ValueError("Неверная когорта либо отсутствует оценочная разметка")
        events = {row.event_id for row in rows}
        if len(events) != count or all_events & events:
            raise ValueError("Повтор event_id между строками или когортами")
        all_events.update(events)
        cohorts[split] = {row.user_id for row in rows}
        matrix.setflags(write=False)
        corpus[split], matrices[split] = rows, matrix
        if split != "training":
            p0 = np.load(copied / f"{split}.gbdt.npy", allow_pickle=False)
            if p0.shape != (count,) or not np.isfinite(p0).all() or np.any((p0 <= 0) | (p0 >= 1)):
                raise ValueError("Некорректные сохранённые прогнозы GBDT")
            p0.setflags(write=False)
            probabilities[split] = p0
    if any(cohorts[left] & cohorts[right] for left, right in
           (("training", "calibration"), ("training", "development"), ("calibration", "development"))):
        raise ValueError("Пользователи пересекаются между когортами")
    alignment = [json.loads(line) for line in (copied / "training.rows.jsonl").read_text().splitlines()]
    if len(alignment) != len(corpus["training"]):
        raise ValueError("Training row index имеет другую длину")
    for index, (item, row) in enumerate(zip(alignment, corpus["training"], strict=True)):
        if item != {"row_index": index, "user_id": row.user_id, "event_id": row.event_id,
                    "decision_at": row.decision_at.isoformat()}:
            raise ValueError("Нарушен порядок training-признаков")
    archive.write_json("bundle_provenance.json", {"manifest_sha256": digest(copied / "manifest.json"),
        "source_manifest_sha256": SOURCE_RUN_SHA256, "source_parent_manifest_sha256": SOURCE_PARENT_SHA256,
        "rows": SPLIT_ROWS, "users": {split: len(users) for split, users in cohorts.items()},
        "gbdt_refitted": False, "generator_included": False, "synthetic_only": True,
        "evaluation_labels_used_for_fit": False})
    return corpus, matrices, probabilities, schema
