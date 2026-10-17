"""Проверяемая фиксация артефактов, исходников и данных без чтения holdout при обучении."""

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "models/gbdt.pickle", "models/encoders.pickle", "frozen_config.json",
    "tuning_results.json", "encoders_manifest.json", "calibration_summary.json",
    "p0_training.npy", "p0_tuning.npy", "p0_calibration.npy",
)
DEPENDENCIES = ("numpy", "scipy", "scikit-learn", "pydantic", "joblib", "threadpoolctl", "pyarrow")


def digest(path: Path) -> str:
    """Вычисляет SHA-256 потоково."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment() -> dict:
    """Фиксирует версию Python и вычислительных зависимостей."""
    packages = {}
    for name in DEPENDENCIES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def source_hashes() -> dict[str, str]:
    """Учитывает незакоммиченные Python-файлы исследовательских зависимостей."""
    paths = [ROOT / "pyproject.toml", ROOT / "hybrid_engine.py", ROOT / "mushroom_body.py"]
    for directory in sorted(ROOT.iterdir()):
        if directory.is_dir() and directory.name.startswith(("research_v", "synthetic_")):
            paths.extend(sorted(directory.rglob("*.py")))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def cohort_hashes(cohorts: Path, splits: tuple[str, ...]) -> dict[str, str]:
    """Не читает общие манифесты с возможными сводками будущих исходов."""
    names = ["schema.json"]
    for split in splits:
        names.extend(f"{split}.{suffix}" for suffix in ("X.npy", "records.jsonl", "evaluation.jsonl"))
    return {name: digest(cohorts / name) for name in names}


def seal_artifacts(artifacts: Path, cohorts: Path, connectome: Path, *, status: str) -> dict:
    """Фиксирует текущий комплект; retrospective означает только фиксацию задним числом."""
    source = source_hashes()
    for name in source:
        destination = artifacts / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(connectome, artifacts / "connectome.npz")
    manifest = {
        "format_version": 2,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "environment": environment(),
        "files": {name: digest(artifacts / name) for name in (*ARTIFACTS, "connectome.npz")},
        "sources": source,
        "inputs": cohort_hashes(cohorts, ("training", "tuning", "calibration")),
    }
    (artifacts / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _check_files(root: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Нарушена целостность: {path}")


def verify_artifacts(artifacts: Path, cohorts: Path) -> dict:
    """Проверяет комплект до десериализации моделей; старый формат требует отдельной фиксации."""
    manifest = json.loads((artifacts / "freeze_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2:
        raise ValueError("Нужен манифест версии 2; исторический прогон следует скопировать и явно зафиксировать как retrospective")
    if set(manifest["files"]) != {*ARTIFACTS, "connectome.npz"}:
        raise ValueError("Неполный комплект замороженных артефактов")
    if manifest["environment"] != environment():
        raise ValueError("Окружение отличается от зафиксированного")
    if manifest["sources"] != source_hashes():
        raise ValueError("Исходники отличаются от зафиксированных")
    if manifest["inputs"] != cohort_hashes(cohorts, ("training", "tuning", "calibration")):
        raise ValueError("Входные когорты отличаются от зафиксированных")
    _check_files(artifacts, manifest["files"])
    _check_files(artifacts / "source_snapshot", manifest["sources"])
    return manifest
