"""Конечный H1 scope, curated sources и ограниченные файловые артефакты."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

UPSTREAM_COMMIT = "1cfe254e8d25b15f37048c3730134686abe60335"
RESEARCH_COMMIT = "e5d1accef15fa38b11580bbf7b529c2c0295f979"
PROTOCOL = "https://gov.nutrifit.health/documents/6ad4f7dbf282b7ed30b9a9f7/versions/12"
TASK = "checkpoint_restore"
CANDIDATE = "candidate/checkpoint.py"
SOURCE_FILES = (
    "hybrid_engine.py",
    "research_v12/__init__.py",
    "research_v12/models.py",
    "research_v12/state.py",
)
FIXTURE_SOURCE = "tests/test_v12_audit_fixes.py"
MAX_FILE = 128 * 1024
MAX_TREE = 2 * 1024 * 1024
PACKAGE = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    """Hash potentially long evidence streams without loading them into RAM."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()


def relative(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Недопустимый artifact path: {name!r}")
    return name


def output_directory(path: Path, repo: Path, upstream: Path | None = None) -> Path:
    """Only .research/runs/<name> may contain outputs inside the source checkout."""
    repo = repo.resolve()
    lexical = Path(path).absolute()
    root = lexical.resolve()
    local = repo / ".research" / "runs"
    if lexical.is_relative_to(repo):
        if not lexical.is_relative_to(local) or lexical == local:
            raise ValueError(
                "Project outputs must be named directories under .research/runs/"
            )
        for component in (repo / ".research", local, *lexical.parents, lexical):
            if component.is_relative_to(repo) and component.is_symlink():
                raise ValueError(
                    "Symlink components are forbidden in project run paths"
                )
        if not root.is_relative_to(local) or root == local:
            raise ValueError("Project run path escaped its dedicated boundary")
    elif root == repo or repo in root.parents or root in repo.parents:
        raise ValueError("Output overlaps the source repository")
    if upstream is not None:
        source = upstream.resolve()
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("Output overlaps the pinned upstream repository")
    return root


def read_regular(path: Path, limit: int = MAX_FILE) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Необычный или слишком большой файл: {path}")
    return path.read_text(encoding="utf-8")


def read_tree(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink запрещён в curated tree")
        if path.is_file():
            result[relative(path.relative_to(root).as_posix())] = read_regular(path)
    if len(json_bytes(result)) > MAX_TREE:
        raise ValueError("Curated tree превышает лимит")
    return result


def write_tree(root: Path, files: dict[str, str]) -> None:
    if len(json_bytes(files)) > MAX_TREE:
        raise ValueError("Artifact tree превышает лимит")
    for name, body in files.items():
        relative(name)
        if not isinstance(body, str) or len(body.encode()) > MAX_FILE:
            raise ValueError("Artifact не является ограниченным UTF-8 текстом")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def source_packet(repo: Path) -> dict[str, str]:
    return {name: read_regular(repo / name) for name in SOURCE_FILES}


def source_manifest(repo: Path) -> dict:
    paths = (*SOURCE_FILES, FIXTURE_SOURCE)
    return {name: digest((repo / name).read_bytes()) for name in paths}


def fixture() -> dict:
    """Буквальный stream существующего test_checkpoint_preserves_pending_and_future_learning.

    Никаких RNG, seed, generation, fit или чтения cohorts. UTC и p0=0.2
    сохранены из development fixture; scalar/ridge — две существующие модели.
    """
    return {
        "source": FIXTURE_SOURCE
        + "::test_checkpoint_preserves_pending_and_future_learning",
        "start": "2026-01-01T00:00:00+00:00",
        "initial_pending": 3,
        "days": list(range(3, 23)),
        "p0": 0.2,
        "confidence": 0.8,
        "z": [1.0, -2.0, 0.5, 3.0],
        "models": ["TunedScalarMemory", "HierarchicalRidgeMemory"],
    }


def plan() -> dict:
    return {
        "protocol": PROTOCOL,
        "task": TASK,
        "upstream_commit": UPSTREAM_COMMIT,
        "research_baseline": RESEARCH_COMMIT,
        "mode": "agg",
        "k": 1,
        "library": None,
        "solver_executions": 4,
        "candidate_allowlist": [CANDIDATE],
        "fixture": fixture(),
        "status": "implementation_only_not_experimentally_verified",
        "required_before_run": [
            "explicit validation and subscription-run authorization",
            "digest-pinned worker image",
            "rootless Docker in craving-h1",
            "active Gemini CLI subscription login (oauth-personal)",
            "exact executor/improver/selector model IDs (gemini-3.8-flash-high)",
            "finite wall/call caps and bounded quota backoff (no dollar budget)",
            "trusted immutable source and evaluator manifests",
            "independent isolation review",
        ],
    }
