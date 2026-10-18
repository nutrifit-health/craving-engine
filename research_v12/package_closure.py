"""Упаковка исправленного replay V12 без fit и без перезаписи исторических комплектов."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

from research_v12.registry import file_sha256


def _sha256(path: Path) -> str:
    return file_sha256(path)


def package_closure(
    *, data_dir: Path, stage2_dir: Path, stage3_dir: Path, closure_dir: Path
) -> dict:
    closure_dir.mkdir(parents=True, exist_ok=False)
    summary = json.loads((stage2_dir / "search_summary.json").read_text())
    if summary.get("status") != "exploratory_replay_no_new_selection":
        raise ValueError("Сначала нужен исправленный replay из сохранённых моделей")
    shutil.copytree(stage2_dir / "models", closure_dir / "models")
    for name in (
        "search_summary.json",
        "selection_summary.json",
        "input_manifest.json",
        "dev_scores_all.json",
        "frozen_v11_config.json",
    ):
        shutil.copy2(stage2_dir / name, closure_dir / name)
    for name in ("robustness_summary.json", "memory_measurements.json"):
        shutil.copy2(stage3_dir / name, closure_dir / name)
    data = closure_dir / "data"
    data.mkdir()
    shutil.copy2(data_dir / "training.X.npy", data / "training.X.npy")
    for split in ("dev_1", "dev_2", "dev_3"):
        for suffix in ("X.npy", "records.jsonl", "evaluation.jsonl"):
            filename = f"{split}.{suffix}"
            shutil.copy2(data_dir / filename, data / filename)
    root = Path(__file__).resolve().parents[1]
    source = closure_dir / "source_snapshot"
    source.mkdir()
    for directory in (
        *root.glob("research_v*"),
        root / "synthetic_data",
        root / "synthetic_benchmark",
    ):
        if directory.is_dir():
            for path in directory.rglob("*.py"):
                if any(
                    p in {"runs", "results", "__pycache__"}
                    for p in path.relative_to(directory).parts
                ):
                    continue
                target = source / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ("hybrid_engine.py", "mushroom_body.py", "pyproject.toml"):
        shutil.copy2(root / name, source / name)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "pydantic", "joblib")
        },
    }
    (closure_dir / "environment.json").write_text(json.dumps(env, indent=2))
    candidate = summary["best_candidate"]
    scalar = summary["baselines_dev"]["tuned_scalar_v12"]
    negative = sum(candidate["seed_briers"][s] < scalar[s] for s in scalar)
    passes = candidate["diff"] <= -0.002 and negative >= 2
    decision = {
        "status": "exploratory_replay_pending_verification",
        "statistical_admission_rule_met": passes,
        "candidate_minus_scalar": candidate["diff"],
        "negative_seeds": negative,
        "confirmation_authorized": False,
        "confirmation_accessed_by_repair": False,
        "note": "Исправительный replay не разрешает новую генерацию или переотбор кандидата.",
    }
    (closure_dir / "decision_and_closure.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False)
    )
    verification = closure_dir / "verification"
    verification.mkdir()
    # Проверяем snapshot до его импорта, затем выполняем код именно сохранённой версии.
    (
        verification / "reproduce_v12.py"
    ).write_text('''"""Воспроизведение доверенного локального комплекта после сверки SHA-256."""
import hashlib
import json
import sys
from pathlib import Path

bundle = Path(__file__).resolve().parents[1]
manifest = json.loads((bundle / "manifest.json").read_text())
for item in manifest["files"]:
    path = (bundle / item["path"]).resolve()
    if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Нарушена целостность: {item['path']}")
sys.path.insert(0, str(bundle / "source_snapshot"))
from research_v12.verification import reproduce_bundle
print(json.dumps(reproduce_bundle(bundle), indent=2))
''')
    files = [
        {
            "path": p.relative_to(closure_dir).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": file_sha256(p),
        }
        for p in sorted(closure_dir.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "format_version": 2,
        "cycle": "V12",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "exploratory_replay_pending_verification",
        "files": files,
    }
    (closure_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data-dir", "stage2-dir", "stage3-dir", "closure-dir"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    package_closure(
        data_dir=args.data_dir,
        stage2_dir=args.stage2_dir,
        stage3_dir=args.stage3_dir,
        closure_dir=args.closure_dir,
    )
