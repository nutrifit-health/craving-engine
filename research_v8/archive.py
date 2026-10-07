"""Новый каталог на каждый запуск; завершённые файлы никогда не перезаписываются."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunArchive:
    """Фиксирует входной протокол до fit и сохраняет незавершённые попытки."""

    def __init__(self, path: Path, protocol: dict) -> None:
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.write_json("protocol.json", protocol)
        self.event("created", protocol_sha256=digest(self.path / "protocol.json"))

    def _path(self, relative: str) -> Path:
        result = (self.path / relative).resolve()
        if not result.is_relative_to(self.path) or result == self.path:
            raise ValueError("Артефакт должен находиться внутри каталога прогона")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def write_json(self, relative: str, payload: object) -> Path:
        path = self._path(relative)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return path

    def write_checkpoint(self, relative: str, payload: str) -> Path:
        path = self._path(relative)
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(payload.encode("utf-8"))
        return path

    def event(self, status: str, **details: object) -> None:
        """Журнал событий append-only; итоговый отчёт остаётся неизменяемым."""
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "status": status, **details}, ensure_ascii=False, allow_nan=False) + "\n")

    def snapshot_sources(self, research_root: Path) -> dict[str, str]:
        """Сохраняет исходники, включая untracked, а не только их контрольные суммы."""
        paths = [research_root / "hybrid_engine.py", research_root / "mushroom_body.py", research_root / "synthetic_benchmark/metrics.py"]
        paths += [path for folder in ("research_v6", "research_v7", "research_v8", "research_v9") for path in sorted((research_root / folder).glob("*.py"))]
        paths += [research_root / "pyproject.toml", research_root / "docs/protocol-v9.md"]
        hashes = {}
        for source in paths:
            relative = str(source.relative_to(research_root))
            data = source.read_bytes()
            with self._path("sources/" + relative).open("xb") as stream:
                stream.write(data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
        self.write_json("source_manifest.json", {"sha256": hashes})
        self.write_json("environment.json", {"python": platform.python_version(), "platform": platform.platform(), "packages": {name: version(name) for name in ("numpy", "scipy", "pydantic", "scikit-learn", "joblib", "threadpoolctl")}})
        return hashes

    def finish(self) -> None:
        """Контрольные суммы фиксируют весь опубликованный результат, кроме журнала."""
        manifest = {str(path.relative_to(self.path)): {"sha256": digest(path), "bytes": path.stat().st_size} for path in sorted(self.path.rglob("*")) if path.is_file() and path.name != "events.jsonl"}
        self.write_json("artifact_manifest.json", manifest)
        self.event("completed", artifact_count=len(manifest))
