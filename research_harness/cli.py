"""Тонкий внешний CLI: plan / run / judge. Ничего не запускается при import."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from .contracts import (
    CANDIDATE,
    PACKAGE,
    PROTOCOL,
    RESEARCH_COMMIT,
    TASK,
    UPSTREAM_COMMIT,
    digest,
    fixture,
    json_bytes,
    output_directory,
    plan,
    read_regular,
    source_manifest,
    source_packet,
    write_tree,
)
from .state import RunStopped, atomic_json
from .subscription import QuotaExhaustedError


def git(root: Path, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()


def upstream(root: Path):
    if git(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("Нужен закреплённый upstream commit")
    status = git(root, "status", "--porcelain", "--untracked-files=normal")
    unclean = [
        line
        for line in status.splitlines()
        if "__pycache__" not in line and line.strip()
    ]
    if unclean:
        raise ValueError("Upstream checkout должен быть чистым")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root.resolve()))
    from .upstream import load_core_config

    load_core_config()


def evaluator_manifest() -> dict:
    return {
        p.relative_to(PACKAGE).as_posix(): digest(p.read_bytes())
        for p in sorted(PACKAGE.rglob("*"))
        if p.is_file()
        and not p.is_symlink()
        and "__pycache__" not in p.parts
        and p.suffix in (".py", ".md", ".json", ".Dockerfile")
    }


def manifest(repo: Path, upstream_root: Path, profile: dict) -> dict:
    return {
        "schema": "craving-harness-manifest/v1",
        "protocol": PROTOCOL,
        "research_baseline": RESEARCH_COMMIT,
        "research_head": git(repo, "log", "-1", "--format=%H", "--", "."),
        "upstream": UPSTREAM_COMMIT,
        "sources": source_manifest(repo),
        "evaluator": evaluator_manifest(),
        "fixture_sha256": digest(json_bytes(fixture())),
        "profile_sha256": digest(json_bytes(profile)),
        "python_version": sys.version,
        "upstream_sources": {
            p.relative_to(upstream_root).as_posix(): digest(p.read_bytes())
            for p in sorted((upstream_root / "harness2").rglob("*.py"))
        },
    }


def write_json(path: Path, data):
    atomic_json(path, data)


def load_json(path: Path):
    return json.loads(read_regular(path, 4 * 1024 * 1024))


def require_complete_run(root: Path, frozen: dict) -> dict:
    complete = load_json(root / "run-complete.json")
    committed = load_json(root / "selection.json")
    name = complete.get("report")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.startswith("run-report-")
    ):
        raise ValueError("Неверный run completion report")
    report = load_json(root / name)
    if (
        complete.get("schema") != "craving-harness-complete/v1"
        or complete.get("manifest_sha256") != digest(json_bytes(frozen))
        or complete.get("selection_sha256") != digest(json_bytes(committed))
        or complete.get("report_sha256") != digest(json_bytes(report))
        or report.get("status") != "selected_not_judged"
        or report.get("error")
        or report.get("command") != "run"
        or report.get("manifest_sha256") != digest(json_bytes(frozen))
        or not isinstance(report.get("usage"), dict)
        or report["usage"].get("closed") is not True
        or report["usage"].get("enforced") is not True
    ):
        raise ValueError("Исходный run не имеет успешного завершённого статуса")
    return committed


def require_harness_stages(result, adapter):
    mode = result.modes.get("agg", {})
    stages = mode.get("stages")
    if (
        result.status != "ok"
        or result.errors
        or not mode.get("ok")
        or mode.get("errors")
        or mode.get("fell_back")
        or not isinstance(stages, list)
        or len(stages) != 3
        or {stage.get("label") for stage in stages} != {"inv1", "dom1", "agg1"}
        or any(
            stage.get("fell_back") or stage.get("error") or not stage.get("attempts")
            for stage in stages
        )
    ):
        raise ValueError(
            "Обязательная стадия Harness не завершилась без ошибки/fallback"
        )
    executions = adapter.completed_executions
    if (
        len(executions) != 4
        or len({entry["run_id"] for entry in executions}) != 4
        or {entry["run_id"].rsplit("/", 1)[-1] for entry in executions}
        != {"base1", "inv1", "dom1", "agg1"}
        or any(not adapter.delivered(Path(entry["run_dir"])) for entry in executions)
    ):
        raise ValueError("Есть неуспешное или отсутствующее обязательное execution")


def restore_execution_inventory(result, adapter, runs_root, *, tag):
    """Core may resume its summary; validate all exact rollout inputs before admission."""
    if result.status not in {"ok", "skipped"}:
        return
    resumed_summary = result.status == "skipped"
    from harness2.core.schedule import Namespace

    from .adapter import safe_id

    namespace = Namespace(
        adapter.spec.name, adapter.substrate.name, "agg", 1, None, tag
    )
    work = namespace.task_dir(runs_root, result.task_id)
    for label in ("base1", "inv1", "dom1", "agg1"):
        run_id = namespace.run_id(result.task_id, label)
        if not any(entry["run_id"] == run_id for entry in adapter.completed_executions):
            harness = (
                adapter.base_harness()
                if label == "base1"
                else work / "harnesses" / label
            )
            if resumed_summary:
                existing = adapter.root / "rollouts" / safe_id(run_id)
                if not adapter.delivered(existing):
                    raise ValueError(
                        "Cached core summary is missing an exact rollout receipt"
                    )
            adapter.execute(result.task_id, harness, run_id)
    if resumed_summary:
        result.status = "ok"
        try:
            require_harness_stages(result, adapter)
        except BaseException:
            result.status = "skipped"
            raise


def execute(args):
    from .runtime import Runtime, profile_from

    profile = profile_from(args.profile)
    repo, up = args.repo.resolve(), args.upstream.resolve()
    upstream(up)
    from harness2.core.schedule import run_task

    from .adapter import CravingAdapter, IsolatedProposer

    is_quota = False
    quota_retry_after = 30
    if args.command == "run":
        root = output_directory(args.output, repo, up)
        is_resume = getattr(args, "resume", False)
        if is_resume:
            checkpoint = load_json(root / "cycle_checkpoint.json")
            if checkpoint.get("method") != args.method:
                raise ValueError("Resume method mismatch")
            frozen = load_json(root / "manifest.json")
            if manifest(repo, up, profile) != frozen:
                raise ValueError(
                    "Source/evaluator/profile manifest изменён; возобновление запрещено"
                )
            sources = source_packet(repo)
            if checkpoint.get("status") == "completed":
                require_complete_run(root, frozen)
                print(
                    json.dumps(
                        {
                            "run": str(root),
                            "status": "selected_not_judged",
                            "cached": True,
                        }
                    )
                )
                return
            if not (root / "run-state.json").is_file():
                raise ValueError(
                    "Legacy checkpoint has no durable role journal; start a new run"
                )
        else:
            root.mkdir(parents=True, exist_ok=False)
            frozen = manifest(repo, up, profile)
            write_json(root / "manifest.json", frozen)
            sources = source_packet(repo)
            write_tree(
                root / "task_source",
                dict(sources, **{CANDIDATE: read_regular(PACKAGE / CANDIDATE)}),
            )
            write_json(
                root / "cycle_checkpoint.json",
                {
                    "schema": "craving-harness-checkpoint/v1",
                    "status": "in_progress",
                    "method": args.method,
                    "created_at": time.time(),
                },
            )
    else:
        root = output_directory(args.run, repo, up)
        frozen = load_json(root / "manifest.json")
        if manifest(repo, up, profile) != frozen:
            raise ValueError(
                "Source/evaluator/profile manifest изменён; judge запрещён"
            )
        sources = source_packet(repo)

    original_selection = (
        require_complete_run(root, frozen) if args.command == "judge" else None
    )
    runtime = Runtime(profile, root, frozen, phase=args.command)
    adapter = CravingAdapter(root, runtime, sources)
    error = None
    interrupted = False
    result = None
    pending_selection = None

    try:
        runtime.prepare(model_access=args.command == "run")
        if args.command == "run":
            if args.method == "harness":
                result = run_task(
                    TASK,
                    1,
                    adapter=adapter,
                    prop=IsolatedProposer(adapter, "proposer"),
                    selector_prop=IsolatedProposer(adapter, "selector"),
                    runs_root=root / "core",
                    modes=("agg",),
                    seed=None,
                    library=None,
                    tag="h1",
                )
                restore_execution_inventory(result, adapter, root / "core", tag="h1")
                require_harness_stages(result, adapter)
                selection = result.selections.get("agg")
                if result.status != "ok" or not selection or not selection.get("ok"):
                    raise ValueError("Core не завершил selection; допуска нет")
            else:
                chosen_run = adapter.execute(
                    TASK, None, "plain/checkpoint_restore/base"
                )
                selection = {
                    "ok": True,
                    "chosen_run_dir": str(chosen_run),
                    "chosen_label": "plain",
                    "chosen_run_id": "plain/checkpoint_restore/base",
                }
                result = {"method": "plain", "selection": selection}

            chosen = Path(selection["chosen_run_dir"]).resolve()
            if chosen.parent != (root / "rollouts").resolve() or not adapter.delivered(
                chosen
            ):
                raise ValueError("Selection не указывает на supervisor-owned rollout")
            expected_calls = 4 if args.method == "harness" else 1
            if (
                runtime.solver_calls != expected_calls
                and len(adapter.completed_executions) != expected_calls
            ):
                raise ValueError("Неполный 4-execution pilot")
            if manifest(repo, up, profile) != frozen:
                raise ValueError("Источники изменились во время run")
            committed = {
                "task_id": TASK,
                "mode": "agg",
                "k": 1,
                "method": args.method,
                "chosen_run_dir": str(chosen),
                "chosen_label": selection["chosen_label"],
                "chosen_run_id": selection["chosen_run_id"],
                "candidate_sha256": digest(read_regular(chosen / CANDIDATE).encode()),
                "manifest_sha256": digest(json_bytes(frozen)),
                "selection": selection,
            }
            pending_selection = committed
        else:
            from harness2.core.judge import judge_one

            if original_selection is None:
                raise ValueError("Missing original selection")
            committed = original_selection
            chosen = Path(committed["chosen_run_dir"]).resolve()
            if (
                committed.get("manifest_sha256") != digest(json_bytes(frozen))
                or chosen.parent != (root / "rollouts").resolve()
                or committed.get("task_id") != TASK
                or committed.get("mode") != "agg"
                or committed.get("k") != 1
            ):
                raise ValueError("Неверный committed selection")
            adapter.committed = committed
            scored = judge_one(
                adapter,
                TASK,
                "agg",
                1,
                committed["chosen_label"],
                committed["chosen_run_id"],
                chosen,
            )
            if manifest(repo, up, profile) != frozen:
                raise ValueError("Manifest изменён во время judge")
            result = scored.__dict__
            if not scored.graded:
                raise ValueError("Judge inconclusive: " + scored.error)
    except QuotaExhaustedError as q_exc:
        is_quota = True
        quota_retry_after = q_exc.retry_after
        error = f"QuotaExhaustedError: {q_exc}"
        write_json(
            root / "cycle_checkpoint.json",
            {
                "schema": "craving-harness-checkpoint/v1",
                "status": "paused_quota",
                "method": args.method if args.command == "run" else None,
                "retry_after": quota_retry_after,
                "paused_at": time.time(),
                "completed_executions": [
                    entry["run_id"] for entry in adapter.completed_executions
                ],
                "reason": str(q_exc),
                "next_step": runtime.journal.data["next_step"],
                "manifest_sha256": digest(json_bytes(frozen)),
                "journal": "run-state.json",
                "not_before": max(
                    (
                        r.get("not_before", 0)
                        for r in runtime.journal.data["operations"].values()
                    ),
                    default=0,
                ),
            },
        )
    except KeyboardInterrupt:
        interrupted = True
        error = "Interrupted: supervisor received an operator interrupt"
    except (RunStopped, Exception) as exc:  # noqa: BLE001 - persist failure at supervisor boundary
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            runtime.finish(
                status="paused_quota"
                if is_quota
                else "interrupted"
                if interrupted
                else "inconclusive"
                if error
                else "completed"
            )
        except Exception as exc:  # noqa: BLE001
            error = error or f"Runtime finish incomplete: {exc}"
        status_val = (
            "paused_quota"
            if is_quota
            else "interrupted"
            if interrupted
            else (
                "inconclusive"
                if error
                else ("selected_not_judged" if args.command == "run" else "graded")
            )
        )
        report = {
            "schema": "craving-harness-report/v1",
            "protocol": PROTOCOL,
            "command": args.command,
            "status": status_val,
            "error": error,
            "manifest_sha256": digest(json_bytes(frozen)),
            "result": result.to_dict()
            if result is not None and hasattr(result, "to_dict")
            else result,
            "usage": runtime.usage,
            "calls": runtime.audit,
            "durable_operations": runtime.journal.data["operations"],
            "upstream_cost_fields": "numeric compatibility placeholders; billing and credits unknown",
            "wall_seconds": time.monotonic() - runtime.started,
            "experimentally_verified": False,
            "reason": "A checkpoint report does not establish isolation proof or matched plain-agent benefit.",
        }
        report_name = f"{args.command}-report-{time.time_ns()}.json"
        write_json(root / report_name, report)
        if args.command == "run" and error and not is_quota:
            write_json(
                root / "cycle_checkpoint.json",
                {
                    "schema": "craving-harness-checkpoint/v1",
                    "status": status_val,
                    "method": args.method,
                    "error": error,
                    "report": report_name,
                    "completed_executions": [
                        entry["run_id"] for entry in adapter.completed_executions
                    ],
                },
            )
        if args.command == "run" and not error and pending_selection is not None:
            write_json(root / "selection.json", pending_selection)
            write_json(
                root / "cycle_checkpoint.json",
                {
                    "schema": "craving-harness-checkpoint/v1",
                    "status": "completed",
                    "method": args.method,
                    "completed_at": time.time(),
                    "completed_executions": [
                        entry["run_id"] for entry in adapter.completed_executions
                    ],
                },
            )
            write_json(
                root / "run-complete.json",
                {
                    "schema": "craving-harness-complete/v1",
                    "report": report_name,
                    "manifest_sha256": digest(json_bytes(frozen)),
                    "selection_sha256": digest(json_bytes(pending_selection)),
                    "report_sha256": digest(json_bytes(report)),
                },
            )

    if is_quota:
        print(
            json.dumps(
                {
                    "run": str(root),
                    "status": "paused_quota",
                    "message": (
                        f"Запуск приостановлен из-за исчерпания квоты Gemini CLI. "
                        f"Состояние сохранено в cycle_checkpoint.json. "
                        f"Возобновите запуск с ключом --resume через ~{quota_retry_after}s."
                    ),
                    "retry_after": quota_retry_after,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if error:
        raise RuntimeError(error)
    print(json.dumps({"run": str(root), "status": report["status"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    preview = subs.add_parser("plan")
    preview.add_argument("--task", choices=[TASK], default=TASK)
    for name in ("run", "judge"):
        command = subs.add_parser(name)
        command.add_argument("--task", choices=[TASK], default=TASK)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--upstream", type=Path, required=True)
        if name == "run":
            command.add_argument("--k", type=int, choices=[1], default=1)
            command.add_argument(
                "--method", choices=["harness", "plain"], default="harness"
            )
            command.add_argument("--output", type=Path, required=True)
            command.add_argument(
                "--resume",
                action="store_true",
                default=False,
                help="Возобновить прерванный или приостановленный по квоте запуск",
            )
        else:
            command.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
    else:

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        execute(args)


if __name__ == "__main__":
    main()
