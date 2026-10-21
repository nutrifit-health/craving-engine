"""Project-local autonomous development research supervisor. No service installation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from .audit import operation_packet, replay
from .cli import load_json, upstream
from .contracts import PACKAGE, digest, json_bytes, output_directory, read_regular
from .critic import H4_MACHINE, assess, machine_assessment
from .queue import frozen_manifest, load_plan, run_round, verify_round
from .runtime import Runtime, profile_from
from .state import BudgetExhaustedError, ResumeMismatchError, RunStopped, atomic_json, read_journal
from .microsteps import MicrostepInterrupted, compatible, original_manifest
from .subscription import EnvironmentUnavailableError, QuotaExhaustedError, quota_reset_delay, redact

QUALIFIER = """import json
from pathlib import Path
import numpy as np
Path('artifacts/metrics.json').write_text(json.dumps({'configuration_ids':['qualification'],'sum':int(np.array([2,3]).sum())}))
Path('artifacts/observations.md').write_text('Engineering qualification only; no scientific claim.')
"""


def source_revision(repo, plan, inputs, profile, up):
    manifest = frozen_manifest(repo, up, plan, inputs, profile)
    manifest["controller_sources"] = {
        p.relative_to(repo).as_posix(): digest(p.read_bytes())
        for p in sorted((repo / "research_harness").rglob("*"))
        if p.is_file()
        and p.suffix in {".py", ".json", ".md", ".txt"}
        and "__pycache__" not in p.parts
    }
    manifest["controller_sources"]["scripts/craving-research"] = digest(
        (repo / "scripts/craving-research").read_bytes()
    )
    return manifest, digest(json_bytes(manifest))


def immutable(path, value):
    if path.exists():
        if load_json(path) != value:
            raise ValueError("Immutable evidence changed: " + str(path))
    else:
        atomic_json(path, value)


class CompletedExperimentRejected(ValueError):
    """Опыт завершён и сохранён; следующая гипотеза требует отдельного бюджета."""


def classify(error):
    if isinstance(error, EnvironmentUnavailableError):
        return "waiting_environment"
    if isinstance(error, QuotaExhaustedError):
        return "waiting_quota"
    text = str(error).lower()
    if any(
        x in text
        for x in (
            "auth",
            "oauth",
            "credential",
            "login",
            "401",
            "403",
            "docker",
            "not found",
            "not installed",
            "no such file",
            "permission denied",
            "cli binary",
            "settings",
            "environment route",
            "protected evaluator",
            "transport configuration",
        )
    ):
        return "waiting_environment"
    return "repair_pending"


class Supervisor:
    def __init__(self, args, *, clock=time.time, sleep=time.sleep, rng=random.uniform):
        root = output_directory(args.output, args.repo, args.upstream) / "autonomous"
        root.mkdir(parents=True, exist_ok=True)
        self.writer_lock = (root / ".lock").open("a")
        try:
            fcntl.flock(self.writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._initialize(args, clock=clock, sleep=sleep, rng=rng)
        except BaseException:
            self.writer_lock.close()
            raise

    def _initialize(self, args, *, clock, sleep, rng):
        self.args, self.clock, self.sleep, self.rng = args, clock, sleep, rng
        self.repo, self.up = args.repo.resolve(), args.upstream.resolve()
        self.campaign = output_directory(args.output, self.repo, self.up)
        self.root = self.campaign / "autonomous"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop = self.campaign / "STOP"
        self.plan, self.inputs = load_plan(args.queue, self.repo)
        self.profile = dict(
            profile_from(args.profile),
            stop_file=str(self.stop),
            autonomous_quota_wait=True,
            model_sessions="unbounded",
        )
        self.manifest, self.revision = source_revision(
            self.repo, self.plan, self.inputs, self.profile, self.up
        )
        self.path = self.root / "state.json"
        if self.path.exists():
            self.state = load_json(self.path)
            self.goal = load_json(self.root / "goal.json")
            if digest(json_bytes(self.goal)) != self.state["goal_sha256"]:
                raise ValueError("Immutable goal changed")
        else:
            self.goal = {
                "schema": "craving-autonomous-goal/v1",
                "mandate": self.plan["mandate"],
                "tasks": [
                    t
                    for t in self.plan["tasks"]
                    if t["id"] in {"H2", "H3", "H4", "H5"} and t.get("enabled")
                ],
                "protected_evaluator_sha256": digest(
                    (PACKAGE / "worker.py").read_bytes()
                ),
                "scope": "exploratory development evidence; no blind data, humans, product release or paid API",
                "machine_criteria": {"H4": H4_MACHINE},
                "protected_measurement_sources": {
                    name: digest(body.encode())
                    for files in self.inputs.values()
                    for name, body in files.items()
                    if Path(name).name
                    in {
                        "metrics.py",
                        "stats.py",
                        "simulator.py",
                        "population.py",
                        "quality.py",
                    }
                },
                "completion_policy": "all_stages_supported; refuted/inconclusive remain research history and never unlock dependents",
                "criteria": {
                    t["id"]: {
                        t["id"] + ".c" + str(i + 1): line[2:].strip()
                        for i, line in enumerate(
                            [x for x in t["mission"].splitlines() if x.startswith("- ")]
                        )
                    }
                    for t in self.plan["tasks"]
                    if t["id"] in {"H2", "H3", "H4", "H5"} and t.get("enabled")
                },
            }
            immutable(self.root / "goal.json", self.goal)
            self.state = {
                "schema": "craving-autonomous-state/v1",
                "goal_sha256": digest(json_bytes(self.goal)),
                "revision": None,
                "epoch": 0,
                "task_index": 0,
                "round": 1,
                "phase": "qualification",
                "status": "ready",
                "admissions": {},
                "events": 0,
                "head": None,
                "failures": 0,
                "quota_waits": 0,
                "next_retry_at": 0,
                "cumulative": {
                    "process_attempts": 0,
                    "active_seconds": 0,
                    "packets": 0,
                    "declared_configurations": 0,
                    "declared_feature_sets": 0,
                },
            }
            self.save()
        # Recover an event fsynced just before a crash prevented state advancement.
        events = sorted((self.root / "events").glob("*.json"))
        previous = None
        for index, path in enumerate(events, 1):
            row = load_json(path)
            if row["index"] != index or row["previous_sha256"] != previous:
                raise ValueError("Cumulative history hash chain changed")
            previous = digest(json_bytes(row))
        if events:
            tail = load_json(events[-1])
            if tail["index"] > self.state["events"]:
                self.state.update(events=tail["index"], head=digest(json_bytes(tail)))
                self.save()

    def save(self):
        self.state["updated_at"] = self.clock()
        atomic_json(self.path, self.state)

    def event(self, kind, **data):
        row = {
            "schema": "craving-autonomous-event/v1",
            "index": self.state["events"] + 1,
            "at": self.clock(),
            "previous_sha256": self.state["head"],
            "kind": kind,
            **data,
        }
        immutable(self.root / "events" / f"{row['index']:09d}.json", row)
        self.state.update(events=row["index"], head=digest(json_bytes(row)))
        self.save()

    def epoch(self):
        return self.root / "epochs" / f"{self.state['epoch']:06d}"

    def round_root(self):
        task = self.goal["tasks"][self.state["task_index"]]["id"]
        return self.epoch() / "tasks" / task / f"round-{self.state['round']:06d}"

    def held_round(self):
        """Операторский запрет нового опыта действует до допуска данного этапа."""
        path = self.root / "continuation-guard.json"
        reference = self.state.get("continuation_guard")
        if not path.exists():
            if reference:
                raise ResumeMismatchError("Исчезла фиксация текущего раунда")
            return None
        try:
            body = read_regular(path)
            guard = json.loads(body)
        except (OSError, ValueError) as error:
            raise ResumeMismatchError("Не удалось прочитать фиксацию текущего раунда: " + str(error)) from error
        if reference and reference.get("sha256") != digest(body.encode()):
            raise ResumeMismatchError("Изменилась фиксация текущего раунда")
        if (
            not isinstance(guard, dict)
            or guard.get("schema") != "craving-continuation-guard/v1"
            or guard.get("goal_sha256") != self.state["goal_sha256"]
            or guard.get("task") not in {t["id"] for t in self.goal["tasks"]}
        ):
            raise ResumeMismatchError("Некорректная фиксация текущего раунда")
        if guard["task"] in self.state["admissions"]:
            return None
        return guard

    def preserve_continuation(self):
        """Сохранить фазу и ожидание перед блокировкой или сменой процесса."""
        if self.state["phase"] in {"research", "replay", "critic"}:
            self.state["recovery_cursor"] = {
                "round": str(self.round_root().relative_to(self.root)),
                "phase": self.state["phase"],
                "waiting_status": self.state.get("waiting_status", self.state["status"]),
                "next_retry_at": self.state.get("next_retry_at", 0),
            }

    def saved_quota_deadline(self, root, phase):
        """Восстановить срок из проверенной диагностики, не меняя старую попытку."""
        from .contracts import relative

        path = root / (phase + "-state.json")
        if not path.exists():
            return 0
        journal = read_journal(path)
        record = journal["operations"].get(journal.get("next_step"), {})
        if record.get("status") != "paused_quota":
            return 0
        deadline = record.get("not_before", 0)
        tail = record.get("attempt_history", [])[-1:]
        reference = tail[0].get("public_failure") if tail else None
        if reference:
            relative(reference["path"])
            failure = json.loads(read_regular(root / reference["path"], 4 * 1024 * 1024))
            if digest(json_bytes(failure)) != reference["sha256"]:
                raise ResumeMismatchError("Изменилось сообщение сервиса о квоте")
            delay = quota_reset_delay(failure.get("stderr_sanitized", ""))
            if delay is not None:
                deadline = max(deadline, tail[0]["ended_at"] + delay)
        return deadline

    def new_epoch(self, reason):
        if self.held_round():
            raise ResumeMismatchError("Новая эпоха запрещена: требуется завершить закреплённый раунд")
        self.state.update(epoch=self.state["epoch"] + 1, round=1)
        self.state["cumulative"]["packets"] += 1
        immutable(
            self.epoch() / "preregistration.json",
            {
                "schema": "craving-exploratory-packet/v1",
                "epoch": self.state["epoch"],
                "goal_sha256": self.state["goal_sha256"],
                "revision": self.revision,
                "max_configurations": 24,
                "max_feature_sets": 6,
                "cumulative_exposure_before": dict(self.state["cumulative"]),
                "inference": "adaptive exploratory search; lifetime exposure is never reset; no confirmatory claim",
            },
        )
        self.event("epoch_started", reason=reason, epoch=self.state["epoch"])

    def authorize_recovery(self, relative_round, *, hold_round=False):
        """Явный операторский переход к совместимой редакции того же опыта."""
        from .contracts import relative

        relative(relative_round)
        guard = self.held_round()
        if guard and guard["round"] != relative_round:
            raise ValueError("Разрешено восстановление только закреплённого раунда")
        previous_round = str(self.round_root().relative_to(self.root))
        cursor = self.state.get("recovery_cursor", {})
        recovery_phase = self.state["phase"]
        waiting_status = self.state.get("waiting_status", self.state.get("status"))
        retry_at = self.state.get("next_retry_at", 0)
        if recovery_phase == "blocked" and cursor.get("round") == relative_round:
            recovery_phase = cursor["phase"]
            waiting_status = cursor["waiting_status"]
            retry_at = cursor["next_retry_at"]
        root = self.root / relative_round
        if (
            root.resolve() != root.absolute()
            or not root.is_relative_to(self.root / "epochs")
            or root.parent.name not in {t["id"] for t in self.goal["tasks"]}
            or not (root / "run-state.json").exists()
            or (root / "machine-admission.json").exists()
        ):
            raise ValueError("Нужен существующий round без допуска внутри кампании")
        if (root / "round-complete.json").exists():
            receipt = verify_round(root)
            if receipt.get("execution_complete") is not True:
                raise ValueError("Завершённый отрицательный опыт нельзя выдать за незавершённый шаг")
            recovery_phase = recovery_phase if previous_round == relative_round and recovery_phase in {"replay", "critic"} else "replay"
        else:
            recovery_phase = "research"
        quota_root = root / "critics" / f"{self.state.get('critic_attempt', 1):06d}" if recovery_phase == "critic" else root
        quota_phase = "critic" if recovery_phase == "critic" else "run"
        saved_deadline = self.saved_quota_deadline(quota_root, quota_phase)
        if saved_deadline > self.clock():
            waiting_status = "waiting_quota"
            retry_at = max(retry_at, saved_deadline)
        old = original_manifest(root)
        old_revision = load_json(self.root / "revisions" / self.state["revision"] / "manifest.json")
        if not compatible(old_revision, self.manifest):
            raise ValueError("Изменились научные входы/критерии: восстановление несовместимо")
        if (
            old["controller_sources"] != old_revision["controller_sources"]
            and not (root / "controller-recoveries").exists()
        ):
            raise ValueError("Восстанавливаемый раунд относится к другой исходной редакции")
        journal = read_journal(root / "run-state.json")
        if not (root / "controller-recoveries").exists() and any(
            operation.startswith("solver/")
            and operation.rsplit("/", 1)[-1] != "base1"
            and record["status"] == "completed"
            for operation, record in journal["operations"].items()
        ):
            raise ValueError("Историческая ветвь может содержать выполненный fallback; нужен отдельный разбор")
        revision_root = self.root / "revisions" / self.revision
        immutable(revision_root / "manifest.json", self.manifest)
        for name in self.manifest["controller_sources"]:
            target = revision_root / "source" / name
            body = (self.repo / name).read_bytes()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != body:
                raise ValueError("Изменился snapshot новой редакции")
            if not target.exists():
                target.write_bytes(body)
        authorization = {
            "schema": "craving-controller-recovery/v1",
            "original_round": relative_round,
            "old_controller_sources": old["controller_sources"],
            "new_controller_sources": self.manifest["controller_sources"],
            "profile_sha256": digest(json_bytes(self.profile)),
            "original_manifest_sha256": digest(json_bytes(old)),
            "execution_revision": self.revision,
            "policy": "Сохраняются исходные входы, результаты и критерии; меняется операционное восстановление",
        }
        authorization_path = root / "controller-recoveries" / (
            digest(json_bytes(self.manifest["controller_sources"])) + ".json"
        )
        immutable(authorization_path, authorization)
        # Критик имеет собственный журнал и неизменяемые входы. Разрешение
        # совместимой редакции требуется и ему; готовые ответы остаются прежними.
        for critic_state in root.glob("critics/*/critic-state.json"):
            if not read_journal(critic_state)["operations"]:
                continue
            critic_manifest = original_manifest(critic_state.parent, "critic")
            if not compatible(critic_manifest, self.manifest):
                raise ValueError("Изменились научные входы сохранённого критика")
            immutable(
                critic_state.parent / "controller-recoveries" / authorization_path.name,
                dict(authorization,
                     old_controller_sources=critic_manifest["controller_sources"],
                     original_manifest_sha256=digest(json_bytes(critic_manifest))),
            )
        task_id = root.parent.name
        old_revision_id = self.state["revision"]
        if hold_round:
            guard_path = self.root / "continuation-guard.json"
            if guard_path.exists():
                saved_guard = load_json(guard_path)
                if saved_guard.get("round") != relative_round:
                    raise ValueError("Уже закреплён другой раунд")
            else:
                immutable(guard_path, {
                    "schema": "craving-continuation-guard/v1",
                    "goal_sha256": self.state["goal_sha256"],
                    "task": task_id, "round": relative_round,
                    "created_at": self.clock(),
                    "policy": "Готовые шаги сохраняются; новые раунды и эпохи данного этапа запрещены до его допуска",
                })
            self.state["continuation_guard"] = {
                "path": guard_path.name, "sha256": digest(guard_path.read_bytes()),
            }
        self.state.update(
            revision=self.revision,
            recovery_revision=self.revision,
            epoch=int(root.parents[2].name),
            round=int(root.name.removeprefix("round-")),
            task_index=next(i for i, t in enumerate(self.goal["tasks"]) if t["id"] == task_id),
            phase=recovery_phase, status="ready", waiting_status="ready", next_retry_at=0,
        )
        if (
            previous_round == relative_round
            and waiting_status in {"waiting_quota", "waiting_environment"}
            and retry_at > self.clock()
        ):
            self.state.update(
                status=waiting_status, waiting_status=waiting_status,
                next_retry_at=retry_at,
            )
        self.state.pop("recovery_cursor", None)
        self.event(
            "microstep_recovery_authorized",
            round=relative_round, previous_revision=old_revision_id,
            execution_revision=self.revision,
            receipt_sha256=digest(authorization_path.read_bytes()),
        )

    def revise(self):
        if (
            digest((PACKAGE / "worker.py").read_bytes())
            != self.goal["protected_evaluator_sha256"]
        ):
            raise ValueError(
                "Protected evaluator changed; waiting for original environment"
            )
        if any(
            digest((self.repo / name).read_bytes()) != expected
            for name, expected in self.goal["protected_measurement_sources"].items()
        ):
            raise ValueError(
                "Protected evaluator/target measurement source changed; waiting for original environment"
            )
        if self.state["revision"] == self.revision:
            if self.state.get("recovery_revision") == self.revision:
                return
            prereg = self.epoch() / "preregistration.json"
            if not prereg.exists() or load_json(prereg)["revision"] != self.revision:
                self.new_epoch("recover interrupted revision qualification")
            return
        if self.state["revision"] is not None:
            # Изменение исходников не выдаёт новый бюджет и не перезапускает опыт.
            self.preserve_continuation()
            self.state.update(status="blocked", phase="blocked", waiting_status="blocked")
            self.state["last_failure"] = {
                "type": "SourceRevisionRequiresRecovery",
                "message": "Нужен явный start --recover-round для совместимых контрольных точек",
            }
            self.save()
            return
        revision_root = self.root / "revisions" / self.revision
        immutable(revision_root / "manifest.json", self.manifest)
        for name in self.manifest["controller_sources"]:
            target = revision_root / "source" / name
            data = (self.repo / name).read_bytes()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Revision snapshot changed")
            if not target.exists():
                target.write_bytes(data)
        old = self.state["revision"]
        self.state.update(
            revision=self.revision,
            admissions={},
            task_index=0,
            phase="qualification",
            round=1,
            qualification_repair_attempts=0,
            failures=0,
            next_retry_at=0,
            status="ready",
        )
        self.event(
            "source_revision",
            previous_revision=old,
            revision=self.revision,
            semantic_admissions_carried=False,
        )
        self.new_epoch("source revision qualification")

    def qualification(self):
        base = self.root / "revisions" / self.revision / "qualification"
        root = base / f"{self.state.get('qualification_attempt', 1):06d}"
        root.mkdir(parents=True, exist_ok=True)
        receipt_path = root / "receipt.json"
        if not receipt_path.exists():
            runtime = Runtime(self.profile, root, self.manifest, phase="qualification")
            failed = True
            try:
                runtime.prepare(model_access=True)
                result = runtime.execute(
                    {"experiment/experiment.py": QUALIFIER},
                    {
                        "kind": "experiment",
                        "operation_id": "qualification/probe",
                        "outputs": [
                            "artifacts/metrics.json",
                            "artifacts/observations.md",
                        ],
                        "configuration_ids": ["qualification"],
                    },
                )
                metrics = json.loads(
                    result.get("files", {}).get("artifacts/metrics.json", "{}")
                )
                if (
                    result.get("execution_complete") is not True
                    or metrics.get("sum") != 5
                ):
                    raise ValueError("Engineering OCI qualification failed")
                atomic_json(root / "result.json", result)
                failed = False
            finally:
                runtime.finish("inconclusive" if failed else "completed")
            immutable(
                receipt_path,
                {
                    "schema": "craving-machine-qualification/v1",
                    "authority": "autonomous_controller",
                    "revision": self.revision,
                    "result_sha256": digest((root / "result.json").read_bytes()),
                    "transport": runtime.journal.data.get("transport"),
                    "file_isolation": "native tools not proven outside repository; OCI networkless/read-only curated inputs",
                    "human_acceptance": False,
                },
            )
        # Проверяем реальный Propose–Probe–Compose отдельно от доступности OCI.
        harness_root = root / "harness-cycle"
        qualification_task = {
            "id": "H2",
            "dependencies": [],
            "max_configurations": 1,
            "max_feature_sets": 1,
            "mission": "Инженерная квалификация транспорта и Harness², не научное исследование. Подготовь краткую программу, которая записывает artifacts/metrics.json с configuration_ids=['qualification'] и sum=2+3; artifacts/observations.md — русский текст 'Инженерная квалификация; научные выводы не проверялись'. hypothesis.json содержит ровно одну configurations и feature_sets запись: id qualification, definition {'purpose':'engineering'}. report.md имеет статус not_executed. Не исследуй модели и литературу, не устанавливай зависимости. Все улучшения харнесса сохраняют это задание.",
            "experiment_outputs": [
                "artifacts/metrics.json",
                "artifacts/observations.md",
            ],
        }
        cycle = run_round(
            harness_root, qualification_task, {}, self.profile, self.manifest
        )
        qualification_probe = load_json(harness_root / "probe.json")
        if (
            cycle.get("execution_complete") is not True
            or json.loads(qualification_probe["files"]["artifacts/metrics.json"]).get(
                "sum"
            )
            != 5
        ):
            raise ValueError("Полная инженерная квалификация Harness² не завершена")
        immutable(
            root / "harness-qualification.json",
            {
                "schema": "craving-full-harness-qualification/v1",
                "revision": self.revision,
                "round_receipt_sha256": digest(
                    (harness_root / "round-complete.json").read_bytes()
                ),
                "scientific_acceptance": False,
            },
        )
        receipt = load_json(receipt_path)
        if receipt["revision"] != self.revision or receipt["result_sha256"] != digest(
            (root / "result.json").read_bytes()
        ):
            raise ValueError("Qualification evidence changed")
        self.state.update(phase="research", status="running")
        self.event("qualified", receipt=str(receipt_path.relative_to(self.root)))

    def context(self):
        recent = []
        start = max(1, self.state["events"] - 7)
        for i in range(start, self.state["events"] + 1):
            recent.append(load_json(self.root / "events" / f"{i:09d}.json"))
        return json_bytes(
            {
                "cumulative": self.state["cumulative"],
                "ledger_head": self.state["head"],
                "ledger_path": str(self.root / "events"),
                "recent": recent,
                "last_failure": self.state.get("last_failure"),
                "inference": "All packets are exploratory. Full raw history retained by hash; this context is bounded.",
            }
        ).decode()

    def dependencies(self):
        files = {}
        for task_id, admission in self.state["admissions"].items():
            receipt_path = self.root / admission["path"]
            if digest(receipt_path.read_bytes()) != admission["sha256"]:
                raise ValueError("Frozen predecessor admission changed")
            receipt = load_json(receipt_path)
            prior = self.root / receipt["round"]
            verify_round(prior)
            if (
                digest((prior / "round-complete.json").read_bytes())
                != receipt["round_receipt_sha256"]
            ):
                raise ValueError("Accepted predecessor receipt changed")
            if (
                digest(
                    (self.root / receipt["replay"] / "replay-report.json").read_bytes()
                )
                != receipt["replay_sha256"]
            ):
                raise ValueError("Accepted predecessor replay changed")
            if (
                digest((self.root / receipt["critic"] / "review.json").read_bytes())
                != receipt["critic_sha256"]
            ):
                raise ValueError("Accepted predecessor critic changed")
            for name in (
                "probe.json",
                "hypothesis.json",
                "chosen/experiment/experiment.py",
            ):
                files[f"accepted/{task_id}/{name}"] = read_regular(
                    prior / name, 2 * 1024 * 1024
                )
            files[f"accepted/{task_id}/machine-admission.json"] = read_regular(
                receipt_path
            )
        return files

    def research(self):
        task = dict(
            self.goal["tasks"][self.state["task_index"]],
            autonomous=True,
            declarations_root=str(self.epoch()),
        )
        if task["id"] == "H4":
            task["machine_criteria"] = {
                "artifact": "artifacts/metrics.json",
                "object": "autonomous_evidence",
                "predicates": self.goal["machine_criteria"]["H4"],
                "instruction": "Compute these values from actual experiment/checkpoint serialization over at least 365 days and drain pending feedback. Missing evidence triggers another experiment; never prefill or guess. Negative results retain actual values.",
            }
        task["mission"] = self.goal["mandate"] + "\n\n" + task["mission"]
        sources = dict(self.inputs[task["id"]], **self.dependencies())
        sources["research_support.py"] = read_regular(
            PACKAGE / "scientific_evidence.py"
        )
        sources["stage_checkpoint.py"] = read_regular(PACKAGE / "stage_checkpoint.py")
        if task["id"] == "H2":
            sources["h2_reference.py"] = read_regular(PACKAGE / "h2_reference.py")
            task["mission"] += (
                "\nОпорный h2_reference.py — исполняемый пример корректной маркировки гипотетических миров и измеримого контракта. Можно запускать его через runpy.run_path('h2_reference.py', run_name='__main__') с configuration id h2_reference либо улучшить опыт. Он не доказывает первичные источники, независимость семейств или соответствие всех требований; дополни источник/проверки там, где критик найдёт пробел."
            )
        task["measured_evidence_required"] = task["id"] in {"H2", "H3"}
        task["mission"] += (
            "\n\nОбязательный измеримый контракт: artifacts/metrics.json содержит stage_evidence из validate_stage_evidence в research_support.py. Используй этот модуль для stable_seed, observed_features и paired_metrics. H2 проверяет генераторы и наблюдаемость, а не превосходство памяти: не запускай сетку из 18 моделей ради приёмки H2. Сохрани исторический генератор; новые допущения помечай гипотезами. Для H3 сохрани компактные prediction_rows для независимого пересчёта. Уменьшай размер development-опыта, если отдельный файл превышает 128 КиБ или весь пакет превышает 2 МиБ; не обрезай данные и не выдавай малую выборку за клиническую валидацию. Объявляй только реально исполненные configurations и feature_sets. Проектные исходники доступны по именам из входного пакета. stage_checkpoint.py использует явный SHA256 контекста; если применяешь checkpoint, объяви его внутри разрешённого metrics.json либо используй временный путь и включи данные в metrics.json: произвольные дополнительные artifacts запрещены. Не заявляй перенос checkpoint между изменёнными входами. Отчёт разделяет macro AUC и pooled AUC, модельный gate и пуши, байты сериализации и latency на измеренном оборудовании. Скорость/батарея телефона и причинные советы без отдельных опытов неизвестны."
        )
        sources["campaign/cumulative-ledger.json"] = self.context()
        failed_root = self.state.get("failed_round")
        if failed_root:
            prior = self.root / failed_root
            for probe in sorted(prior.glob("rollouts/*/probe.json"))[-2:]:
                prefix = "diagnostic/" + probe.parent.name + "/"
                sources[prefix + "probe.json"] = read_regular(probe, 2 * 1024 * 1024)
                program = probe.parent / "experiment/experiment.py"
                if program.exists():
                    sources[prefix + "experiment.py"] = read_regular(program)
            reports = sorted(prior.glob("attempt-report-*.json"))
            if reports:
                attempt = load_json(reports[-1])
                sources["diagnostic/previous-failure.json"] = json_bytes(
                    {
                        "error": attempt.get("error"),
                        "next_step": attempt.get("next_step"),
                    }
                ).decode()
        round_root = self.round_root()
        # Input context is itself frozen: status events added after quota must not change it.
        packet_path = round_root / "autonomous-inputs.json"
        if packet_path.exists():
            sources = load_json(packet_path)
        else:
            atomic_json(packet_path, sources)
        prior_path = round_root / "autonomous-prior.json"
        if prior_path.exists():
            prior = load_json(prior_path)["prior"]
        else:
            prior = self.state.get("latest_completed", {}).get(task["id"])
            atomic_json(prior_path, {"prior": prior})
        receipt = run_round(
            round_root,
            task,
            sources,
            self.profile,
            self.manifest,
            prior=self.root / prior if prior else None,
        )
        if receipt.get("execution_complete") is not True:
            raise CompletedExperimentRejected("Chosen experiment did not execute successfully")
        self.state.update(phase="replay", critic_attempt=1)
        self.state.setdefault("latest_completed", {})[task["id"]] = str(
            round_root.relative_to(self.root)
        )
        self.event(
            "research_complete",
            round=str(round_root.relative_to(self.root)),
            receipt_sha256=digest((round_root / "round-complete.json").read_bytes()),
        )

    def repeat(self):
        root = self.round_root()
        verify_round(root)
        selection = load_json(root / "selection.json")
        from .contracts import relative

        relative(selection["rollout"])
        chosen = root / selection["rollout"]
        if (
            chosen.parent != root / "rollouts"
            or digest((chosen / "receipt.json").read_bytes())
            != selection["receipt_sha256"]
        ):
            raise ValueError("Selected rollout receipt changed")
        receipt = load_json(chosen / "receipt.json")
        operation = selection["operation"]
        packet, record = operation_packet(root, operation)
        probe = load_json(chosen / "probe.json")
        if (
            operation != "probe/" + receipt["run_id"]
            or packet["model_role"] is not None
            or packet["request"]["kind"] != "experiment"
            or record["status"] != "completed"
            or probe.get("execution_complete") is not True
            or digest(json_bytes(probe)) != record["result_sha256"]
            or load_json(root / "probe.json") != probe
            or any(
                packet["files"].get(name) != read_regular(root / "chosen" / name)
                for name in ("experiment/experiment.py", "hypothesis.json", "report.md")
            )
        ):
            raise ValueError(
                "Selected experiment does not match the successful frozen OCI operation"
            )
        replay_root = (
            self.root
            / "replays"
            / (str(self.state["epoch"]) + "-" + root.parent.name + "-" + root.name)
        )
        replay_report_path = replay_root / "replay-report.json"
        if not replay_report_path.exists() or load_json(replay_report_path).get("result") is None:
            # Завершённый OCI результат остаётся в журнале этого replay.
            # После сбоя публикации повторяется только формирование отчёта.
            replay_root.parent.mkdir(parents=True, exist_ok=True)
            execution_manifest = dict(
                self.manifest,
                **{
                    name: packet["manifest"][name]
                    for name in ("task", "round", "round_inputs", "base_harness", "library_input")
                },
            )
            replay(root, operation, replay_root, execution_manifest=execution_manifest)
        report = load_json(replay_root / "replay-report.json")
        if (
            report.get("exact_output_match") is not True
            or report.get("original_operation") != operation
            or report.get("original_input_sha256") != record["inputs_sha256"]
            or report.get("original_result_sha256") != record["result_sha256"]
        ):
            raise ValueError("Independent OCI replay differs from frozen result")
        self.state.update(
            phase="critic", replay=str(replay_root.relative_to(self.root))
        )
        self.event(
            "replay_complete",
            report_sha256=digest((replay_root / "replay-report.json").read_bytes()),
        )

    def criticism(self):
        root = self.round_root()
        receipt = verify_round(root)
        files = {
            name: read_regular(root / name, 4 * 1024 * 1024)
            for name in receipt["files"]
            if name
            in {
                "probe.json",
                "research-report.md",
                "claims.json",
                "chosen/experiment/experiment.py",
                "chosen/hypothesis.json",
            }
        }
        replay_report = load_json(
            self.root / self.state["replay"] / "replay-report.json"
        )
        files["replay.json"] = json_bytes(
            {
                k: replay_report[k]
                for k in (
                    "schema",
                    "original_operation",
                    "original_input_sha256",
                    "original_result_sha256",
                    "replayed_result_sha256",
                    "exact_output_match",
                )
            }
        ).decode()
        for name, body in load_json(root / "autonomous-inputs.json").items():
            files["input/" + name] = body
        probe = load_json(root / "probe.json")
        files.update(probe.get("files", {}))
        files["exposure.json"] = self.context()
        files["immutable-goal.json"] = json_bytes(self.goal).decode()
        task = self.goal["tasks"][self.state["task_index"]]
        criteria = dict(self.goal["criteria"][task["id"]])
        if not criteria:
            raise ValueError("Immutable stage criteria are missing")
        criteria[task["id"] + ".validity"] = (
            "Program implements preregistration; no leakage; fair baselines; full declared scenarios, units, uncertainty and configuration coverage; exploratory exposure disclosed. A negative result is justified by executed falsification and sufficient coverage, not mere incompleteness."
        )
        critic_root = root / "critics" / f"{self.state.get('critic_attempt', 1):06d}"
        preliminary = machine_assessment(probe, task["id"], "inconclusive")
        files["scientific-analysis.json"] = json_bytes(preliminary).decode()
        # Независимые пересчёты входят в неизменяемый пакет критика.
        snapshot = critic_root / "inputs.json"
        if snapshot.exists():
            files = load_json(snapshot)
            if (
                files.get("scientific-analysis.json")
                != json_bytes(preliminary).decode()
            ):
                raise ValueError("Изменился независимый пересчёт")
        else:
            atomic_json(snapshot, files)
        immutable(critic_root / "scientific-analysis.json", preliminary)
        review = assess(critic_root, files, criteria, self.profile, self.manifest)
        machine = machine_assessment(
            load_json(root / "probe.json"), task["id"], review["outcome"]
        )
        if review["outcome"] != "supported":
            self.state["last_failure"] = {
                "type": "critic_" + review["outcome"],
                "message": review["next_experiment"],
            }
            self.event(
                "hypothesis_assessed",
                outcome=review["outcome"],
                critic_sha256=digest((critic_root / "review.json").read_bytes()),
                machine_criteria=machine,
            )
            self.next_attempt("independent critic requests another experiment")
            return
        admission_path = root / "machine-admission.json"
        immutable(
            admission_path,
            {
                "schema": "craving-agent-admission/v1",
                "authority": "autonomous_controller_with_independent_critic",
                "human_acceptance": False,
                "goal_sha256": self.state["goal_sha256"],
                "revision": self.revision,
                "task": task["id"],
                "outcome": review["outcome"],
                "round": str(root.relative_to(self.root)),
                "round_receipt_sha256": digest(
                    (root / "round-complete.json").read_bytes()
                ),
                "replay_sha256": digest(
                    (
                        self.root / self.state["replay"] / "replay-report.json"
                    ).read_bytes()
                ),
                "critic_sha256": digest((critic_root / "review.json").read_bytes()),
                "critic": str(critic_root.relative_to(self.root)),
                "replay": self.state["replay"],
                "criteria": criteria,
                "machine_criteria": machine,
                "inference": "development exploratory evidence only; not confirmation or deployment",
            },
        )
        self.state["admissions"][task["id"]] = {
            "path": str(admission_path.relative_to(self.root)),
            "sha256": digest(admission_path.read_bytes()),
        }
        self.state["task_index"] += 1
        self.state.update(phase="research", round=self.state["round"] + 1, failures=0)
        if self.state["task_index"] == len(self.goal["tasks"]):
            self.state["phase"] = "finalize"
        self.event(
            "stage_admitted",
            task=task["id"],
            outcome=review["outcome"],
            admission_sha256=digest(admission_path.read_bytes()),
        )
        if self.state["phase"] == "finalize":
            self.finalize()

    def finalize(self):
        if set(self.state["admissions"]) != {t["id"] for t in self.goal["tasks"]}:
            raise ValueError("Every immutable stage needs supported machine admission")
        # Check all dependencies and immutable receipts again before final result.
        self.dependencies()
        self.exposure()
        self.save()
        result_path = self.root / "revisions" / self.revision / "result.json"
        immutable(
            result_path,
            {
                "schema": "craving-development-result/v1",
                "goal_sha256": self.state["goal_sha256"],
                "revision": self.revision,
                "admissions": self.state["admissions"],
                "scope": self.goal["scope"],
                "positive_solution": all(
                    load_json(self.root / a["path"])["outcome"] == "supported"
                    for a in self.state["admissions"].values()
                ),
                "cumulative": self.state["cumulative"],
            },
        )
        self.state.update(
            status="development_evidence_complete",
            phase="complete",
            result=str(result_path.relative_to(self.root)),
        )
        self.event(
            "development_evidence_complete",
            result_sha256=digest(result_path.read_bytes()),
        )

    def exposure(self):
        totals = {
            "process_attempts": 0,
            "active_seconds": 0,
            "packets": self.state["epoch"],
            "declared_configurations": 0,
            "declared_feature_sets": 0,
            "model_active_seconds": 0,
            "compute_active_seconds": 0,
        }
        for path in self.root.rglob("*-state.json"):
            data = read_journal(path)
            totals["process_attempts"] += data["attempts"]
            totals["active_seconds"] += data["active_seconds"]
            totals["model_active_seconds"] += data.get("model_active_seconds", 0)
            totals["compute_active_seconds"] += data["active_seconds"] - data.get(
                "model_active_seconds", 0
            )
        for path in self.root.glob("epochs/*/declarations.json"):
            declarations = load_json(path)
            for field in ("configurations", "feature_sets"):
                signatures = {
                    digest(json_bytes(x["definition"]))
                    for row in declarations.values()
                    for x in row["hypothesis"][field]
                }
                totals["declared_" + field] += len(signatures)
        self.state["cumulative"] = totals
        return totals

    def next_attempt(self, reason):
        guard = self.held_round()
        if guard:
            self.state.update(
                status="blocked", phase="blocked", waiting_status="blocked",
                next_retry_at=0,
            )
            self.event(
                "held_round_needs_repair", round=guard["round"], reason=reason,
                failure=self.state.get("last_failure"),
                next_action="Исправить незавершённый шаг закреплённого раунда; новый опыт не разрешён",
            )
            return
        self.exposure()
        self.state["failed_round"] = str(self.round_root().relative_to(self.root))
        task = self.goal["tasks"][self.state["task_index"]]
        historical = len(list(self.root.glob("epochs/*/tasks/" + task["id"] + "/round-*")))
        stage_attempts = max(
            historical,
            self.state.get("stage_attempts", {}).get(task["id"], 0) + 1,
        )
        self.state.setdefault("stage_attempts", {})[task["id"]] = stage_attempts
        max_stage_attempts = task.get(
            "max_stage_attempts", task.get("max_rounds", 6) * 3
        )
        if stage_attempts >= max_stage_attempts:
            self.state.update(
                status="blocked",
                phase="blocked",
                waiting_status="blocked",
            )
            self.event(
                "stage_attempts_exhausted",
                task=task["id"],
                attempts=stage_attempts,
                limit=max_stage_attempts,
            )
            self.save()
            return
        self.state.update(phase="research", round=self.state["round"] + 1)
        declarations_path = self.epoch() / "declarations.json"
        full = False
        if declarations_path.exists():
            rows = load_json(declarations_path)
            full = any(
                len(
                    {
                        x["id"]
                        for r in rows.values()
                        if r["admitted"]
                        for x in r["hypothesis"][field]
                    }
                )
                >= cap
                for field, cap in (("configurations", 24), ("feature_sets", 6))
            )
        epoch_attempts = sum(
            read_journal(p)["attempts"] for p in self.epoch().rglob("*-state.json")
        )
        epoch_active = sum(
            data["active_seconds"] - data.get("model_active_seconds", 0)
            for p in self.epoch().rglob("*-state.json")
            for data in [read_journal(p)]
        )
        task_max_rounds = task.get("max_rounds", self.plan.get("max_total_rounds", 24))
        if (
            full
            or self.state["round"] > task_max_rounds
            or epoch_attempts >= self.plan["max_process_attempts"]
            or epoch_active >= self.plan["active_seconds"]
        ):
            self.new_epoch(reason)
        self.event(
            "repair_next_attempt",
            reason=reason,
            round=self.state["round"],
            epoch=self.state["epoch"],
        )

    def failed(self, error):
        self.preserve_continuation()
        kind = classify(error)
        self.state["failures"] += 1
        self.state["last_failure"] = {
            "type": type(error).__name__,
            "message": redact(str(error))[:3000],
        }
        if self.stop.exists():
            self.state["status"] = "interrupted"
            self.save()
            return
        if isinstance(error, (BudgetExhaustedError, ResumeMismatchError)):
            self.state.update(status="blocked", phase="blocked", waiting_status="blocked", next_retry_at=0)
            self.event("microstep_blocked", failure=self.state["last_failure"])
            return
        if isinstance(error, CompletedExperimentRejected):
            # Отрицательный завершённый опыт сохраняется целиком. Новый опыт
            # допустим только в оставшемся общем бюджете этапа.
            self.next_attempt(str(error))
            self.save()
            return
        if isinstance(error, MicrostepInterrupted):
            # Runtime сохраняет отдельные попытки роли; готовые операции остаются
            # на прежнем месте. Техническая ошибка не создаёт новую гипотезу/эпоху.
            retries = self.state.setdefault("microstep_retries", {})
            failure_key = (
                str(self.round_root().relative_to(self.root))
                + "/" + digest(str(error).encode())
            )
            retries[failure_key] = retries.get(failure_key, 0) + 1
            if retries[failure_key] >= self.profile["max_role_attempts"]:
                self.state.update(status="blocked", phase="blocked", waiting_status="blocked", next_retry_at=0)
                self.event("microstep_blocked", failure=self.state["last_failure"])
                return
            delay = min(300, 30 * 2 ** min(self.state["failures"] - 1, 3))
            self.state.update(
                status="repair_pending", waiting_status="repair_pending",
                next_retry_at=self.clock() + delay,
            )
            self.event("microstep_retry_pending", failure=self.state["last_failure"],
                       round=str(self.round_root().relative_to(self.root)))
            return
        if kind == "repair_pending":
            # A malformed cached critic must get a new operation root, not poison research.
            if self.state["phase"] == "critic":
                self.state["critic_attempt"] = self.state.get("critic_attempt", 1) + 1
                if self.state["critic_attempt"] > self.profile["max_role_attempts"]:
                    self.next_attempt(
                        "bounded critic attempts exhausted; repair evidence in a new research attempt"
                    )
            elif self.state["phase"] == "qualification":
                # Повторяется только незавершённая роль в том же журнале.
                # Уже выполненные solver/probe не оплачиваются и не вычисляются заново.
                attempts = self.state.get("qualification_repair_attempts", 0) + 1
                self.state["qualification_repair_attempts"] = attempts
                if attempts >= self.profile["max_role_attempts"]:
                    self.state.update(phase="blocked", status="blocked")
            else:
                # Локальный разбор/запись отчёта не сбрасывает готовые расчёты.
                root = self.round_root()
                failures = self.state.setdefault("microstep_failures", {})
                key = str(root.relative_to(self.root))
                failures[key] = failures.get(key, 0) + 1
                if failures[key] >= self.profile["max_role_attempts"]:
                    self.state.update(phase="blocked", status="blocked")
        delay = (
            max(1, float(error.retry_after))
            if kind == "waiting_quota" and getattr(error, "retry_known", True)
            else min(3600, 30 * 2 ** min(self.state["failures"] - 1, 7))
            * self.rng(0.8, 1.2)
        )
        if kind == "waiting_environment":
            delay = max(300, delay)
            if self.state["phase"] == "critic":
                self.state["critic_attempt"] = self.state.get("critic_attempt", 1) + 1
            elif self.state["phase"] == "research":
                # Восстановление среды продолжает прежнюю операцию.
                pass
        if self.state.get("phase") == "blocked":
            self.state.update(
                status="blocked", waiting_status="blocked", next_retry_at=0
            )
            self.save()
            return
        if kind == "waiting_quota":
            self.state["quota_waits"] += 1
        self.state.update(
            status=kind, waiting_status=kind, next_retry_at=self.clock() + delay
        )
        self.event(
            "waiting",
            kind_of_wait=kind,
            next_retry_at=self.state["next_retry_at"],
            failure=self.state["last_failure"],
        )

    def tick(self):
        if self.stop.exists():
            self.state["status"] = "interrupted"
            self.save()
            return False
        if self.state.get("status") in {
            "interrupted",
            "blocked",
            "development_evidence_complete",
        }:
            return False

        try:
            guard = self.held_round()
            if guard and str(self.round_root().relative_to(self.root)) != guard["round"]:
                raise ResumeMismatchError("Контроллер вышел за пределы закреплённого раунда")
        except RunStopped as error:
            self.failed(error)
            return False

        # Check for source updates / self-improvements first
        try:
            upstream(self.up)
            current_plan, current_inputs = load_plan(self.args.queue, self.repo)
            _, current = source_revision(
                self.repo, current_plan, current_inputs, self.profile, self.up
            )
            if current != self.revision:
                self.preserve_continuation()
                self.state["status"] = "source_rollover"
                self.save()
                os.execv(
                    sys.executable,
                    [
                        sys.executable,
                        "-m",
                        "research_harness.autonomy",
                        "work",
                        *self.args.worker_args,
                    ],
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.state["source_check_error"] = {
                "type": type(error).__name__,
                "message": redact(str(error))[:1000],
            }

        if self.clock() < self.state.get("next_retry_at", 0):
            return True
        try:
            upstream(self.up)
            current_plan, current_inputs = load_plan(self.args.queue, self.repo)
            _, current = source_revision(
                self.repo, current_plan, current_inputs, self.profile, self.up
            )
            if current != self.revision:
                self.state["status"] = "source_rollover"
                self.save()
                os.execv(
                    sys.executable,
                    [
                        sys.executable,
                        "-m",
                        "research_harness.autonomy",
                        "work",
                        *self.args.worker_args,
                    ],
                )
            self.revise()
            if self.state.get("status") == "blocked":
                return False
            self.state.update(status="running", next_retry_at=0)
            if self.state["phase"] == "qualification":
                self.qualification()
            elif self.state["phase"] == "research":
                self.research()
            elif self.state["phase"] == "replay":
                self.repeat()
            elif self.state["phase"] == "critic":
                self.criticism()
            elif self.state["phase"] == "finalize":
                self.finalize()
            elif self.state["phase"] == "complete":
                return False
            self.exposure()
            self.save()
        except (
            RunStopped,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            self.failed(error)
        return self.state["status"] not in {
            "interrupted",
            "blocked",
            "development_evidence_complete",
        }

    def work(self):
        with self.writer_lock:
            self.state["pid"] = os.getpid()
            if (
                self.state.get("status") == "blocked"
                and self.state.get("revision") != self.revision
            ):
                self.revise()
            if self.state["status"] == "interrupted":
                self.state["status"] = (
                    self.state.get("waiting_status", "running")
                    if self.state.get("next_retry_at", 0) > self.clock()
                    else "running"
                )
            self.save()

            def stop_requested(*_):
                atomic_json(self.stop, {"requested_at": self.clock(), "by": "signal"})

            signal.signal(signal.SIGTERM, stop_requested)
            signal.signal(signal.SIGINT, stop_requested)
            while self.tick():
                delay = max(0, self.state.get("next_retry_at", 0) - self.clock())
                if delay:
                    self.sleep(min(5, delay))


def active(root):
    path = root / "autonomous" / ".lock"
    if not path.exists():
        return False
    with path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "work", "status", "stop"])
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--recover-round", help="Продолжить совместимый сохранённый round без повторения готовых операций")
    parser.add_argument("--hold-round", action="store_true", help="Запретить новые раунды и эпохи восстанавливаемого этапа до его допуска")
    for name in ("repo", "upstream", "profile", "queue", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.hold_round and (args.command != "start" or not args.recover_round):
        raise ValueError("--hold-round используется только с start --recover-round")
    args.worker_args = [
        item
        for name in ("repo", "upstream", "profile", "queue", "output")
        for item in ("--" + name, str(getattr(args, name).absolute()))
    ]
    root = output_directory(args.output, args.repo, args.upstream)
    if args.command == "status":
        path = root / "autonomous" / "state.json"
        if path.exists():
            result = dict(
                load_json(path), active=active(root), output=str(root / "autonomous")
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            from .status import operator_status

            print(
                json.dumps(
                    operator_status(root, args.queue), ensure_ascii=False, indent=2
                )
            )
        return
    if args.command == "stop":
        atomic_json(root / "STOP", {"requested_at": time.time()})
        print(json.dumps({"status": "stop_requested", "active": active(root)}))
        return
    if active(root):
        print(
            json.dumps(
                {"status": "already_running", "output": str(root / "autonomous")}
            )
        )
        return
    if args.recover_round:
        if args.command != "start":
            raise ValueError("--recover-round разрешён только с start")
        supervisor = Supervisor(args)
        try:
            supervisor.authorize_recovery(args.recover_round, hold_round=args.hold_round)
        finally:
            supervisor.writer_lock.close()
    (root / "STOP").unlink(missing_ok=True)
    if args.command == "start" and not args.foreground:
        (root / "autonomous").mkdir(parents=True, exist_ok=True)
        with (root / "autonomous" / "stdout.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "research_harness.autonomy",
                    "work",
                    *args.worker_args,
                ],
                cwd=args.repo,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        print(
            json.dumps(
                {
                    "status": "starting",
                    "pid": process.pid,
                    "output": str(root / "autonomous"),
                }
            )
        )
        return
    try:
        Supervisor(args).work()
    except BlockingIOError:
        print(json.dumps({"status": "already_running"}))


if __name__ == "__main__":
    main()
