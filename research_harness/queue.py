"""Bounded research campaigns over pinned Harness²; never starts at import."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from .cli import (
    evaluator_manifest,
    git,
    load_json,
    require_harness_stages,
    restore_execution_inventory,
    upstream,
)
from .contracts import (
    MAX_TREE,
    PACKAGE,
    PROTOCOL,
    UPSTREAM_COMMIT,
    digest,
    json_bytes,
    output_directory,
    read_regular,
    read_tree,
    relative,
    write_tree,
)
from .runtime import Runtime, profile_from
from .state import BudgetExhaustedError, ResumeMismatchError, RunStopped, atomic_json
from .subscription import QuotaExhaustedError, redact


def bounded_int(value, name, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(name + " must be a finite bounded positive integer")
    return value


def load_plan(path, repo):
    plan = load_json(path)
    if plan.get("schema") != "craving-research-queue/v1":
        raise ValueError("Unknown research queue schema")
    bounded_int(plan.get("max_total_rounds"), "max_total_rounds", 128)
    bounded_int(plan.get("max_process_attempts"), "max_process_attempts", 4096)
    bounded_int(plan.get("active_seconds"), "active_seconds", 7 * 24 * 3600)
    bounded_int(plan.get("max_quota_resumes"), "max_quota_resumes", 128)
    bounded_int(
        plan.get("max_quota_wait_seconds"), "max_quota_wait_seconds", 30 * 24 * 3600
    )
    if not isinstance(plan.get("mandate"), str) or not plan["mandate"].strip():
        raise ValueError("Canonical research mandate must be inline model context")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 16:
        raise ValueError("Finite research task list required")
    known = {"H1"}
    inputs = {}
    for task in tasks:
        task_id = task.get("id")
        if task_id not in {"H2", "H3", "H4", "H5", "H6", "H7"} or task_id in known:
            raise ValueError("Unknown/duplicate research stage")
        if not set(task.get("dependencies", [])) <= known:
            raise ValueError("Dependencies must precede this stage")
        known.add(task_id)
        if task_id in {"H6", "H7"}:
            if task.get("enabled") is not False:
                raise ValueError(
                    "Confirmation/production are always manual and blocked"
                )
            continue
        bounded_int(task.get("max_rounds"), "max_rounds", 32)
        bounded_int(task.get("max_configurations"), "max_configurations", 24)
        bounded_int(task.get("max_feature_sets"), "max_feature_sets", 6)
        if task_id == "H4" and task["max_rounds"] > 6:
            raise ValueError("H4 has at most 24 program proposals (six Harness rounds)")
        if not isinstance(task.get("mission"), str) or not task["mission"].strip():
            raise ValueError("Explicit research mission required")
        names = task.get("sources")
        if (
            not isinstance(names, list)
            or not names
            or len(names) > 80
            or len(set(names)) != len(names)
        ):
            raise ValueError("Exact finite curated source list required")
        files = {}
        for name in names:
            relative(name)
            source = repo / name
            if source.resolve().is_relative_to(repo) is False or any(
                part.startswith(".") for part in Path(name).parts
            ):
                raise ValueError("Source outside the curated repository boundary")
            if source.suffix != ".py" or source.is_symlink():
                raise ValueError(
                    "Default research source packet accepts Python source only"
                )
            files[name] = read_regular(source)
        if len(json_bytes(files)) > MAX_TREE:
            raise ValueError("Curated source packet exceeds cap")
        outputs = task.get("experiment_outputs")
        if (
            not isinstance(outputs, list)
            or not 1 <= len(outputs) <= 8
            or len(set(outputs)) != len(outputs)
        ):
            raise ValueError("Exact finite output list required")
        for name in outputs:
            relative(name)
            if not name.startswith("artifacts/") or Path(name).suffix not in {
                ".json",
                ".md",
                ".txt",
            }:
                raise ValueError("Named UTF-8 outputs must be inside artifacts/")
        inputs[task_id] = files
    return plan, inputs


def frozen_manifest(repo, up, plan, inputs, profile):
    return {
        "schema": "craving-research-manifest/v1",
        "protocol": PROTOCOL,
        "queue": plan,
        "profile": profile,
        "upstream": UPSTREAM_COMMIT,
        "upstream_sources": {
            p.relative_to(up).as_posix(): digest(p.read_bytes())
            for p in sorted((up / "harness2").rglob("*.py"))
        },
        "research_head": git(repo, "log", "-1", "--format=%H", "--", "."),
        "python_version": sys.version,
        "evaluator": evaluator_manifest(),
        "sources": {
            stage: {name: digest(body.encode()) for name, body in files.items()}
            for stage, files in inputs.items()
        },
    }


def aggregate_budget(root):
    attempts, active, rounds = 0, 0.0, 0
    for path in sorted(root.glob("tasks/H*/round-*/run-state.json")):
        state = load_json(path)
        attempts += state["attempts"]
        active += state["active_seconds"]
        rounds += 1
    return {
        "process_attempts": attempts,
        "active_seconds": active,
        "rounds_started": rounds,
        "internal_model_calls": None,
        "cost_usd": None,
        "credits": None,
    }


def verify_round(root):
    receipt = load_json(root / "round-complete.json")
    if receipt.get("schema") != "craving-research-round/v1":
        raise ValueError("Unknown completed research round")
    for name, expected in receipt["files"].items():
        relative(name)
        if digest(read_regular(root / name, 4 * 1024 * 1024).encode()) != expected:
            raise ResumeMismatchError("Completed round artifact changed: " + name)
    if receipt.get("usage", {}).get("closed") is not True:
        raise ValueError("Round accounting was not closed")
    return receipt


def stage_fingerprint(root):
    rounds = [
        verify_round(path)
        for path in sorted(root.glob("round-*"))
        if (path / "round-complete.json").exists()
    ]
    declarations = (
        load_json(root / "declarations.json")
        if (root / "declarations.json").exists()
        else {}
    )
    return digest(json_bytes({"rounds": rounds, "declarations": declarations}))


def acceptance(root, task_id, manifest_hash):
    path = root / "acceptance" / (task_id + ".json")
    if not path.exists():
        return False
    record = load_json(path)
    if (
        record.get("schema") != "craving-research-acceptance/v1"
        or record.get("task") != task_id
        or record.get("manifest_sha256") != manifest_hash
        or record.get("decision") != "accepted"
        or not record.get("operator")
        or not record.get("evidence")
    ):
        raise ValueError("Invalid explicit operator acceptance receipt")
    if task_id != "H1":
        stage = load_json(root / "tasks" / task_id / "stage.json")
        if (
            stage.get("status") != "awaiting_scientific_review"
            or record.get("artifact_sha256") != stage.get("artifact_sha256")
            or stage_fingerprint(root / "tasks" / task_id)
            != stage.get("artifact_sha256")
        ):
            raise ResumeMismatchError(
                "Acceptance does not cover the frozen stage artifacts"
            )
    return True


def dependency_packet(root, task, manifest_hash):
    """Freeze accepted predecessor evidence as read-only model data, never host code."""
    packet = {}
    for predecessor in task["dependencies"]:
        if not acceptance(root, predecessor, manifest_hash):
            raise ValueError("Missing predecessor acceptance: " + predecessor)
        receipt_path = root / "acceptance" / (predecessor + ".json")
        entry = {
            "acceptance": load_json(receipt_path),
            "acceptance_sha256": digest(receipt_path.read_bytes()),
            "rounds": [],
        }
        if predecessor != "H1":
            for prior in sorted((root / "tasks" / predecessor).glob("round-*")):
                if not (prior / "round-complete.json").exists():
                    continue
                receipt = verify_round(prior)
                names = [
                    "research-report.md",
                    "hypothesis.json",
                    "claims.json",
                    "probe.json",
                    "chosen/experiment/experiment.py",
                    "chosen/hypothesis.json",
                ]
                evidence = {
                    name: read_regular(prior / name, 2 * 1024 * 1024) for name in names
                }
                entry["rounds"].append(
                    {
                        "round": prior.name,
                        "receipt": receipt,
                        "receipt_sha256": digest(json_bytes(receipt)),
                        "files": evidence,
                        "sha256": {
                            name: digest(body.encode())
                            for name, body in evidence.items()
                        },
                    }
                )
        packet[predecessor] = entry
    if len(json_bytes(packet)) > MAX_TREE:
        raise ValueError("Accepted dependency packet exceeds model-context bound")
    return packet


def status(root, plan, manifest_hash):
    result = {
        "schema": "craving-research-status/v1",
        "manifest_sha256": manifest_hash,
        "budget": aggregate_budget(root),
        "tasks": [],
    }
    for task in plan["tasks"]:
        task_id = task["id"]
        path = root / "tasks" / task_id / "stage.json"
        stage = load_json(path) if path.exists() else {}
        if task_id in {"H6", "H7"} or task.get("enabled") is False:
            state, reason = (
                "blocked_manual",
                "This stage is outside automatic research admission",
            )
        elif acceptance(root, task_id, manifest_hash):
            state, reason = "accepted", "Explicit operator scientific review receipt"
        elif any(
            not acceptance(root, dep, manifest_hash) for dep in task["dependencies"]
        ):
            state, reason = (
                "blocked_dependency",
                "Predecessor requires explicit evidence-bound acceptance",
            )
        else:
            state, reason = stage.get("status", "ready"), stage.get("reason")
        result["tasks"].append(
            {
                "id": task_id,
                "status": state,
                "reason": reason,
                "round": stage.get("round"),
                "next_step": stage.get("next_step"),
                "artifact_sha256": stage.get("artifact_sha256"),
                "not_before": stage.get("not_before", 0),
            }
        )
    return result


def run_round(root, task, sources, profile, manifest, *, prior=None):
    from .upstream import load_core_config

    load_core_config()

    from harness2.core.library import Library
    from harness2.core.schedule import Namespace, run_task
    from harness2.substrate.opencode import OpencodeSubstrate

    from .adapter import IsolatedProposer
    from .research import ResearchAdapter

    root.mkdir(parents=True, exist_ok=True)
    if (root / "round-complete.json").exists():
        return verify_round(root)
    inputs = dict(sources)
    history = []
    previous_rounds = sorted(root.parent.glob("round-*"))
    if task.get("autonomous"):
        previous_rounds = [p for p in previous_rounds if p.name < root.name][-2:]
    for previous in previous_rounds:
        if (
            previous.name >= root.name
            or not (previous / "round-complete.json").exists()
        ):
            continue
        previous_receipt = verify_round(previous)
        summary = load_json(previous / "hypothesis.json")
        probes = []
        for probe_path in sorted(previous.glob("rollouts/*/probe.json")):
            probe = load_json(probe_path)
            probes.append(
                {
                    "receipt_sha256": digest(
                        (probe_path.parent / "receipt.json").read_bytes()
                    ),
                    "execution_complete": probe.get("execution_complete"),
                    "errors": probe.get("errors"),
                    "artifacts": {
                        name: digest(text.encode())
                        for name, text in probe.get("files", {}).items()
                    },
                }
            )
        history.append(
            {
                "round": previous.name,
                "reflection": summary,
                "probes_including_failures": probes,
                "round_sha256": digest(json_bytes(previous_receipt)),
            }
        )
    inputs["campaign/hypothesis-and-negative-ledger.json"] = json_bytes(
        history
    ).decode()
    declarations = (
        Path(task.get("declarations_root", root.parent)) / "declarations.json"
    )
    # Always freeze a snapshot, including the initially empty ledger, before any turn.
    snapshot = root / "declarations-input.json"
    if not snapshot.exists():
        atomic_json(snapshot, load_json(declarations) if declarations.exists() else {})
    inputs["campaign/prior-declarations.json"] = read_regular(snapshot, 2 * 1024 * 1024)
    base = None
    previous_library = None
    if prior:
        verify_round(prior)
        base = prior / "final-harness"
        previous_library = prior / "library-output"
        # Every previous final report/negative result stays immutable and is explicit evidence.
        inputs["campaign/previous-report.md"] = read_regular(
            prior / "research-report.md"
        )
        inputs["campaign/previous-hypothesis.json"] = read_regular(
            prior / "hypothesis.json"
        )
        inputs["campaign/previous-program.py"] = read_regular(
            prior / "chosen/experiment/experiment.py"
        )
        inputs["campaign/previous-observations.json"] = read_regular(
            prior / "probe.json", 2 * 1024 * 1024
        )
        if task.get("autonomous"):
            # A bounded retrieval view contains complete recent entries. The immutable
            # older libraries and all their raw experience remain in prior rounds.
            entries = previous_library / task["id"] / "harnesses"
            keep = (
                set(sorted(p.name for p in entries.iterdir() if p.is_dir())[-2:])
                if entries.exists()
                else set()
            )
            view = root / "library-context"
            files = {
                name: body
                for name, body in read_tree(previous_library).items()
                if len(Path(name).parts) > 2
                and Path(name).parts[1] == "harnesses"
                and Path(name).parts[2] in keep
            }
            write_tree(view, files)
            # The INDEX belongs to the frozen input, including on quota resume.
            # Build it before the manifest/Journal binding, never afterwards.
            Library(view, task["id"], OpencodeSubstrate().surface()).rebuild_index()
            previous_library = view
    task_source = root / "task_source"
    if task_source.exists() and read_tree(task_source) != inputs:
        raise ResumeMismatchError("Round inputs changed")
    write_tree(task_source, inputs)
    round_manifest = dict(
        manifest,
        task=task,
        round=root.name,
        round_inputs={k: digest(v.encode()) for k, v in inputs.items()},
        base_harness={
            k: digest(v.encode())
            for k, v in read_tree(base or PACKAGE / "research_base_harness").items()
        },
        library_input={
            k: digest(v.encode()) for k, v in read_tree(previous_library).items()
        }
        if previous_library
        else {},
    )
    runtime = Runtime(profile, root, round_manifest)
    # Исходный manifest остаётся идентичностью опыта. Новое исполнение контроллера
    # отдельно записывается в execution-manifests и receipt восстановления.
    round_manifest = runtime.manifest
    adapter = ResearchAdapter(root, runtime, inputs, task, base=base)
    # Read a frozen prior library, write new experience into this round only.
    frozen_library = (
        Library(previous_library, task["id"], adapter.substrate.surface())
        if previous_library
        else None
    )

    class RoundLibrary(Library):
        def context_dir(self):
            return frozen_library.context_dir() if frozen_library else None

        def add(self, task_id, harness, **kwargs):
            return super().add(task_id + "-" + root.name, harness, **kwargs)

    if previous_library and not (root / "library-output").exists():
        shutil.copytree(previous_library, root / "library-output")
    library = RoundLibrary(
        root / "library-output", task["id"], adapter.substrate.surface()
    )
    result = None
    stopped = None
    receipt = None
    try:
        runtime.prepare(model_access=True)
        # Производные контексты upstream пересоздаются; предыдущие проходы
        # сохраняются. Durable операции и rollouts остаются в том же round.
        passes = root / "core-passes"
        passes.mkdir(exist_ok=True)
        core_root = passes / f"{len(list(passes.iterdir())) + 1:06d}"
        core_root.mkdir()
        atomic_json(root / "core-current.json", {"path": str(core_root.relative_to(root))})
        result = run_task(
            task["id"],
            1,
            adapter=adapter,
            prop=IsolatedProposer(adapter, "proposer"),
            selector_prop=IsolatedProposer(adapter, "selector"),
            runs_root=core_root,
            modes=("agg",),
            seed=None,
            library=library,
            tag="campaign",
        )
        restore_execution_inventory(result, adapter, core_root, tag="campaign")
        require_harness_stages(result, adapter)
        selection = result.selections["agg"]
        chosen = Path(selection["chosen_run_dir"]).resolve()
        if chosen.parent != (root / "rollouts").resolve() or not adapter.delivered(
            chosen
        ):
            raise ValueError("Research selection outside supervisor rollouts")
        probe = load_json(chosen / "probe.json")
        atomic_json(
            root / "selection.json",
            {
                "schema": "craving-selected-operation/v1",
                "rollout": chosen.relative_to(root).as_posix(),
                "receipt_sha256": digest((chosen / "receipt.json").read_bytes()),
                "operation": "probe/" + load_json(chosen / "receipt.json")["run_id"],
            },
        )
        analysis = {"status": "not_required"}
        if task.get("measured_evidence_required"):
            from .critic import machine_assessment

            try:
                analysis = {
                    "status": "complete_for_review",
                    **machine_assessment(probe, task["id"], "inconclusive"),
                }
            except (ValueError, KeyError, TypeError) as error:
                analysis = {"status": "evidence_incomplete", "reason": str(error)}
        atomic_json(root / "scientific-analysis.json", analysis)
        report_input = dict(
            inputs,
            **{
                "selected/" + k: v for k, v in adapter.deliverable_files(chosen).items()
            },
        )
        report_input["selected/probe.json"] = json_bytes(probe).decode()
        report_input["scientific-analysis.json"] = json_bytes(analysis).decode()
        report_operation = None
        for report_attempt in range(1, profile.get("max_role_attempts", 4) + 1):
            operation = (
                "campaign/report"
                if report_attempt == 1
                else f"campaign/report/retry-{report_attempt}"
            )
            record = runtime.journal.data["operations"].get(operation, {})
            rejected = (
                root / "report-rejections" / (digest(operation.encode()) + ".json")
            )
            if record.get("status") != "failed" and not rejected.exists():
                report_operation = operation
                break
        if report_operation is None:
            raise BudgetExhaustedError(
                "Исчерпаны попытки отчёта; готовые расчёты сохранены"
            )
        reflection = runtime.execute(
            report_input,
            {
                "kind": "agent",
                "role": "proposer",
                "operation_id": report_operation,
                "mission": "Write a grounded research report in Russian using only the supplied measured probe and program. Все отчеты, гипотезы, претензии и формулировки выводов должны быть написаны исключительно на русском языке для пользователя. In answer return a JSON object directly (a JSON-encoded string is also accepted) with hypothesis (string in Russian), next_hypothesis (string in Russian), decision (continue|request_review|blocked), finding_status (executed|evidence_incomplete|inconclusive|refuted|provisional_candidate), claims (list of {id,text,evidence:[relative paths]} covering every empirical claim in report, all claim texts must be in Russian), report (Markdown string fully written in Russian). Allowed evidence paths: probe.json, scientific-analysis.json, chosen/experiment/experiment.py and chosen/hypothesis.json. files must be {}. Preserve failed/negative findings and uncertainties. Never claim scientific acceptance, live-user benefit, new citations, or inspect confirmation. request_review means the independent evidence critic should assess criteria; it does not request a human interaction in autonomous mode. Otherwise propose the next bounded within-stage experiment.",
            },
            model_role="proposer",
        )
        try:
            hypothesis = json.loads(reflection["answer"])
            if (
                not isinstance(hypothesis, dict)
                or set(hypothesis)
                != {
                    "hypothesis",
                    "next_hypothesis",
                    "decision",
                    "finding_status",
                    "claims",
                    "report",
                }
                or hypothesis["decision"]
                not in {"continue", "request_review", "blocked"}
                or hypothesis["finding_status"]
                not in {
                    "executed",
                    "evidence_incomplete",
                    "inconclusive",
                    "refuted",
                    "provisional_candidate",
                }
                or any(
                    not isinstance(hypothesis[k], str) or not hypothesis[k].strip()
                    for k in hypothesis
                    if k != "claims"
                )
            ):
                raise ValueError("Invalid campaign reflection schema")
            (root / "research-report.md").write_text(
                hypothesis.pop("report"), encoding="utf-8"
            )
            atomic_json(root / "hypothesis.json", hypothesis)
            atomic_json(root / "probe.json", probe)
            write_tree(root / "chosen", adapter.deliverable_files(chosen))
            claims = hypothesis.pop("claims")
            if not isinstance(claims, list) or not 1 <= len(claims) <= 64:
                raise ValueError("Finite empirical claim/evidence mapping required")
            allowed_evidence = {
                "probe.json",
                "scientific-analysis.json",
                "chosen/experiment/experiment.py",
                "chosen/hypothesis.json",
            }
            for claim in claims:
                if (
                    not isinstance(claim, dict)
                    or set(claim) != {"id", "text", "evidence"}
                    or not isinstance(claim["id"], str)
                    or not isinstance(claim["text"], str)
                    or not isinstance(claim["evidence"], list)
                    or not claim["evidence"]
                    or not set(claim["evidence"]) <= allowed_evidence
                ):
                    raise ValueError("Invalid report claim evidence mapping")
                claim["evidence"] = {
                    name: digest((root / name).read_bytes())
                    for name in claim["evidence"]
                }
            atomic_json(root / "claims.json", claims)
            atomic_json(root / "hypothesis.json", hypothesis)
        except (ValueError, TypeError, KeyError) as error:
            atomic_json(
                root
                / "report-rejections"
                / (digest(report_operation.encode()) + ".json"),
                {
                    "operation": report_operation,
                    "answer_sha256": digest(reflection["answer"].encode()),
                    "reason": str(error),
                },
            )
            raise
        namespace = Namespace(
            adapter.spec.name, adapter.substrate.name, "agg", 1, None, "campaign"
        )
        harness = (
            namespace.task_dir(core_root, task["id"])
            / "harnesses"
            / selection["chosen_label"]
        )
        write_tree(root / "final-harness", read_tree(harness))
        atomic_json(root / "core-result.json", result.to_dict())
        receipt_files = [
            "research-report.md",
            "claims.json",
            "hypothesis.json",
            "probe.json",
            "core-result.json",
            "selection.json",
            "scientific-analysis.json",
            "chosen/experiment/experiment.py",
            "chosen/report.md",
            "chosen/hypothesis.json",
        ]
        receipt_files += [
            "final-harness/" + name for name in read_tree(root / "final-harness")
        ]
        receipt_files += [
            "library-output/" + name for name in read_tree(root / "library-output")
        ]
        receipt = {
            "schema": "craving-research-round/v1",
            "task": task["id"],
            "manifest_sha256": digest(json_bytes(round_manifest)),
            "execution_manifest_sha256": digest(json_bytes(runtime.execution_manifest)),
            "decision": hypothesis["decision"],
            "execution_complete": probe.get("execution_complete") is True,
            "scientific_acceptance": "manual_required",
            "files": {
                name: digest((root / name).read_bytes()) for name in receipt_files
            },
        }
        return receipt
    except (RunStopped, Exception) as exc:
        stopped = exc
        raise
    finally:
        runtime.finish(
            "paused_quota"
            if isinstance(stopped, QuotaExhaustedError)
            else "inconclusive"
            if stopped
            else "completed"
        )
        atomic_json(
            root / ("attempt-report-" + str(time.time_ns()) + ".json"),
            {
                "schema": "craving-research-attempt/v1",
                "status": runtime.usage["status"],
                "error": {
                    "type": type(stopped).__name__,
                    "message": redact(str(stopped)),
                    "traceback": redact("".join(traceback.format_exception(stopped))),
                }
                if stopped
                else None,
                "core_result": result.to_dict() if result is not None else None,
                "next_step": runtime.journal.data["next_step"],
                "usage": runtime.usage,
                "operations": runtime.journal.data["operations"],
                "scientific_acceptance": "manual_required",
            },
        )
        if stopped is None and receipt is not None:
            receipt["usage"] = runtime.usage
            atomic_json(root / "round-complete.json", receipt)


def execute(args):
    repo, up = args.repo.resolve(), args.upstream.resolve()
    root = output_directory(args.output, repo, up)
    if args.command == "stop":
        atomic_json(
            root / "STOP",
            {
                "requested_at": time.time(),
                "semantics": "stop after the active bounded operation",
            },
        )
        print(json.dumps({"status": "stop_requested", "output": str(root)}))
        return {}
    if args.clear_stop:
        if args.command != "resume":
            raise ValueError("--clear-stop is an explicit operator resume action only")
        (root / "STOP").unlink(missing_ok=True)
    if args.command == "status":
        from .status import operator_status

        snapshot = operator_status(root, args.queue)
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return snapshot
    profile = dict(profile_from(args.profile), stop_file=str(root / "STOP"))
    plan, inputs = load_plan(args.queue, repo)
    upstream(up)
    manifest = frozen_manifest(repo, up, plan, inputs, profile)
    manifest_hash = digest(json_bytes(manifest))
    if args.command == "init":
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "manifest.json", manifest)
        atomic_json(
            root / "queue-state.json",
            {
                "schema": "craving-research-queue-state/v1",
                "manifest_sha256": manifest_hash,
            },
        )
    elif load_json(root / "manifest.json") != manifest:
        raise ResumeMismatchError(
            "Queue/source/evaluator/profile changed; use a new campaign root"
        )
    with (root / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = status(root, plan, manifest_hash)
        if args.command in {"init", "status"}:
            print(json.dumps(snapshot, indent=2, ensure_ascii=False))
            return snapshot
        if args.command == "accept":
            if args.task in {"H6", "H7"}:
                raise ValueError(
                    "Confirmation/production admission is outside this controller"
                )
            stage = (
                None
                if args.task == "H1"
                else load_json(root / "tasks" / args.task / "stage.json")
            )
            if stage and stage.get("status") != "awaiting_scientific_review":
                raise ValueError("Only a frozen stage may receive operator acceptance")
            expected = (
                manifest_hash
                if args.task == "H1"
                else stage_fingerprint(root / "tasks" / args.task)
            )
            if stage and expected != stage["artifact_sha256"]:
                raise ResumeMismatchError(
                    "Frozen stage artifacts changed before operator acceptance"
                )
            if args.artifact_sha256 != expected:
                raise ValueError(
                    "Operator must supply the exact reviewed artifact hash"
                )
            atomic_json(
                root / "acceptance" / (args.task + ".json"),
                {
                    "schema": "craving-research-acceptance/v1",
                    "task": args.task,
                    "decision": "accepted",
                    "manifest_sha256": manifest_hash,
                    "artifact_sha256": expected,
                    "operator": args.operator,
                    "evidence": args.evidence,
                    "authority": "explicit_local_operator_command_not_Governance_human_approval",
                    "at": time.time(),
                },
            )
            print(
                json.dumps(
                    status(root, plan, manifest_hash), indent=2, ensure_ascii=False
                )
            )
            return
        # Explicit task selection prevents an automatic inter-stage DAG launch.
        selected = next(
            (row for row in snapshot["tasks"] if row["id"] == args.task), None
        )
        if not selected or selected["status"] in {
            "blocked_dependency",
            "blocked_manual",
            "accepted",
            "awaiting_scientific_review",
        }:
            raise ValueError(
                "Selected task is not admitted for research; inspect status and dependency evidence"
            )
        task = dict(next(t for t in plan["tasks"] if t["id"] == args.task))
        task["mission"] = plan["mandate"] + "\n\n" + task["mission"]
        stage_root = root / "tasks" / args.task
        stage_root.mkdir(parents=True, exist_ok=True)
        dependencies = dependency_packet(root, task, manifest_hash)
        dependency_path = stage_root / "accepted-dependencies.json"
        if dependency_path.exists() and load_json(dependency_path) != dependencies:
            raise ResumeMismatchError("Accepted dependency snapshot changed on resume")
        if not dependency_path.exists():
            atomic_json(dependency_path, dependencies)
        stage_sources = dict(inputs[args.task])
        stage_sources["campaign/accepted-dependencies.json"] = json_bytes(
            dependencies
        ).decode()
        for index in range(1, task["max_rounds"] + 1):
            round_root = stage_root / f"round-{index:03d}"
            if (round_root / "round-complete.json").exists():
                receipt = verify_round(round_root)
            else:
                budget = aggregate_budget(root)
                if (
                    budget["rounds_started"] >= plan["max_total_rounds"]
                    and not round_root.exists()
                    or budget["process_attempts"] >= plan["max_process_attempts"]
                    or budget["active_seconds"] >= plan["active_seconds"]
                ):
                    raise BudgetExhaustedError("Campaign finite budget exhausted")
                # Reserve one round conservatively; its persistent profile cannot grow on resume.
                if not round_root.exists() and (
                    budget["process_attempts"] + profile["max_calls"]
                    > plan["max_process_attempts"]
                    or budget["active_seconds"] + profile["wall_seconds"]
                    > plan["active_seconds"]
                ):
                    raise BudgetExhaustedError(
                        "Insufficient campaign budget to reserve another complete round"
                    )
                atomic_json(
                    stage_root / "stage.json", {"status": "in_progress", "round": index}
                )
                try:
                    receipt = run_round(
                        round_root,
                        task,
                        stage_sources,
                        profile,
                        manifest,
                        prior=stage_root / f"round-{index - 1:03d}"
                        if index > 1
                        else None,
                    )
                except (RunStopped, Exception) as exc:  # noqa: BLE001 - persist a stopped campaign without auto fallback
                    journal = (
                        load_json(round_root / "run-state.json")
                        if (round_root / "run-state.json").exists()
                        else {}
                    )
                    atomic_json(
                        stage_root / "stage.json",
                        {
                            "status": "paused_quota"
                            if isinstance(exc, QuotaExhaustedError)
                            else "blocked",
                            "round": index,
                            "reason": str(exc),
                            "next_step": journal.get("next_step"),
                            "not_before": max(
                                (
                                    r.get("not_before", 0)
                                    for r in journal.get("operations", {}).values()
                                ),
                                default=0,
                            ),
                        },
                    )
                    snapshot = status(root, plan, manifest_hash)
                    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
                    return snapshot
            # Never discard earlier attempts, even when the new proposal is worse.
            if receipt["decision"] != "continue" or index == task["max_rounds"]:
                artifact_hash = stage_fingerprint(stage_root)
                state = (
                    "awaiting_scientific_review"
                    if receipt["execution_complete"]
                    and receipt["decision"] != "blocked"
                    else "blocked"
                )
                atomic_json(
                    stage_root / "stage.json",
                    {
                        "status": state,
                        "round": index,
                        "artifact_sha256": artifact_hash,
                        "reason": "Finite stage campaign stopped; explicit scientific review required",
                        "scientific_acceptance": "manual_required",
                    },
                )
                break
        snapshot = status(root, plan, manifest_hash)
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return snapshot


def supervise(args):
    snapshot = execute(args)
    if not args.wait_for_quota or args.command not in {"run", "resume"}:
        return
    root = args.output.resolve()
    plan, _ = load_plan(args.queue, args.repo.resolve())
    while True:
        row = next((r for r in snapshot.get("tasks", []) if r["id"] == args.task), {})
        if row.get("status") != "paused_quota" or (root / "STOP").exists():
            return
        delay = max(1, row.get("not_before", 0) - time.time())
        with (root / ".queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = load_json(root / "queue-state.json")
            if (
                state.get("quota_resumes", 0) >= plan["max_quota_resumes"]
                or state.get("quota_wait_seconds_reserved", 0) + delay
                > plan["max_quota_wait_seconds"]
            ):
                print(
                    json.dumps(
                        {
                            "status": "paused_quota",
                            "reason": "finite supervisor wait/resume cap reached",
                        }
                    )
                )
                return
            state["quota_resumes"] = state.get("quota_resumes", 0) + 1
            state["quota_wait_seconds_reserved"] = (
                state.get("quota_wait_seconds_reserved", 0) + delay
            )
            state["retry_at"] = time.time() + delay
            atomic_json(root / "queue-state.json", state)
        # No CLI polling, no transport retry loop. One explicit bounded campaign
        # continuation after the durable eligible time; STOP is checked while idle.
        until = time.monotonic() + delay
        while time.monotonic() < until:
            if (root / "STOP").exists():
                return
            time.sleep(min(30, until - time.monotonic()))
        args.command = "resume"
        snapshot = execute(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["init", "status", "run", "resume", "accept", "stop"]
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["H1", "H2", "H3", "H4", "H5"])
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--operator")
    parser.add_argument("--evidence")
    parser.add_argument(
        "--clear-stop",
        action="store_true",
        help="Explicitly clear a previous operator STOP while resuming",
    )
    parser.add_argument(
        "--wait-for-quota",
        action="store_true",
        help="Finite foreground wait/continue supervisor; CLI retries remain disabled",
    )
    args = parser.parse_args()
    if args.command in {"run", "resume", "accept"} and not args.task:
        parser.error("--task is required; stages never launch automatically")
    if args.command == "accept" and not all(
        (args.artifact_sha256, args.operator, args.evidence)
    ):
        parser.error(
            "accept requires --artifact-sha256, --operator and --evidence from explicit human review"
        )
    supervise(args)


if __name__ == "__main__":
    main()
