"""Read-only saved H1 inventory, independent of scientific queue admission."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .contracts import digest, json_bytes, read_regular


def load(path):
    value = json.loads(read_regular(path, 4 * 1024 * 1024))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path.name}")
    return value


def supervisor_active(root):
    """Observe the existing lock without creating it or changing journal state."""
    path = root / ".supervisor.lock"
    if not path.is_file() or path.is_symlink():
        return False
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def h1_run(root):
    from .cli import evaluator_manifest, require_complete_run

    row = {"name": root.name, "run": str(root), "scientific_admission": False}
    try:
        frozen = load(root / "manifest.json")
        checkpoint = load(root / "cycle_checkpoint.json")
        journal_path = root / "run-state.json"
        journal = load(journal_path) if journal_path.exists() else {}
        active = supervisor_active(root)
        reports = sorted(root.glob("run-report-*.json"))
        report = load(reports[-1]) if reports else {}
        state = report.get("status", checkpoint.get("status", "unknown"))
        if active:
            state = "running"
        elif state in {"in_progress", "running"}:
            state = "interrupted"
        evidence = [
            str(p)
            for p in (
                root / "manifest.json",
                root / "cycle_checkpoint.json",
                journal_path,
            )
            if p.exists()
        ]
        if reports:
            evidence.append(str(reports[-1]))
        judges = sorted(root.glob("judge-report-*.json"))
        judge = load(judges[-1]) if judges else None
        if (
            (root / "run-complete.json").exists()
            or state in {"completed", "selected_not_judged", "graded"}
            or (judge and judge.get("status") == "graded")
        ):
            marker = load(root / "run-complete.json")
            load(root / "selection.json")
            report_name = marker.get("report")
            if (
                not isinstance(report_name, str)
                or Path(report_name).name != report_name
            ):
                raise ValueError("Invalid completion report path")
            load(root / report_name)
            require_complete_run(root, frozen)
            if not active:
                state = "selected_not_judged"
            evidence += [str(root / "run-complete.json"), str(root / "selection.json")]
        if judge:
            evidence.append(str(judges[-1]))
            if judge.get("manifest_sha256") != digest(json_bytes(frozen)):
                raise ValueError("Judge manifest mismatch")
            if not active:
                state = (
                    "graded"
                    if judge.get("status") == "graded" and not judge.get("error")
                    else "judge_inconclusive"
                )
        operations = journal.get("operations", {})
        if not isinstance(operations, dict) or any(
            not isinstance(v, dict) for v in operations.values()
        ):
            raise ValueError(
                "Journal operations must be an object of operation objects"
            )
        row.update(
            status=state,
            method=checkpoint.get("method"),
            supervisor_active=active,
            next_step=journal.get("next_step"),
            attempts=journal.get("attempts", 0),
            operations={
                k: {key: v.get(key) for key in ("status", "attempts", "error")}
                for k, v in operations.items()
            },
            completed_executions=checkpoint.get("completed_executions", []),
            error=(judge or report).get("error"),
            judge=judge.get("result") if judge else None,
            manifest_sha256=digest(json_bytes(frozen)),
            evaluator_revision_current=frozen.get("evaluator") == evaluator_manifest(),
            evidence=evidence,
        )
        if state == "interrupted":
            row["reason"] = (
                "Супервизор не удерживает lock; сохранённый незавершённый исход не считается успехом."
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(status="evidence_incomplete", error=f"{type(exc).__name__}: {exc}")
    return row


def h1_inventory(runs_root):
    return [
        h1_run(root)
        for root in sorted(Path(runs_root).glob("*-h1"))
        if root.is_dir() and not root.is_symlink()
    ]


def operator_status(root, queue_path):
    from .queue import status

    h1 = h1_inventory(root.parent)
    if not (root / "queue-state.json").exists():
        snapshot = {
            "status": "not_initialized",
            "output": str(root),
            "message": "Научная очередь ещё не создана; состояние инженерных H1 запусков показано отдельно.",
            "next_step": "./scripts/craving-research agent",
        }
    else:
        try:
            frozen = load(root / "manifest.json")
            plan = load(queue_path)
            snapshot = status(root, plan, digest(json_bytes(frozen)))
            snapshot["status"] = "initialized"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            snapshot = {
                "status": "evidence_incomplete",
                "error": f"{type(exc).__name__}: {exc}",
            }
    snapshot["queue_status"] = snapshot["status"]
    snapshot["h1"] = h1
    return snapshot
