"""Host supervisor: OCI isolation boundary для evaluator и подписочный runner Gemini CLI.

Все модельные роли (solver, proposer, selector) работают через официальный Gemini CLI
с персональной авторизацией (oauth-personal). Платный API и долларовые бюджеты исключены.
Реплей и независимый оценщик (judge) исполняются в изолированном OCI контейнере без сети.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .contracts import MAX_TREE, PACKAGE, digest, json_bytes, write_tree
from .state import BudgetExhaustedError, Journal, ResumeMismatchError, RunStopped
from .microsteps import (
    MicrostepInterrupted,
    bound_manifest,
    compatible,
    persist_recovery,
    recover_result,
    saved_response_draft,
)
from .subscription import GeminiSubscriptionRunner, QuotaExhaustedError

IMAGE = re.compile(r"^(?:[a-zA-Z0-9./:_-]+@)?sha256:[0-9a-f]{64}$")


def positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"Нужен положительный конечный {name}")
    return value


def profile_from(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    schema = p.get("schema")
    if schema not in ("craving-harness-run/v1", "craving-harness-run/v2"):
        raise ValueError("Нужен авторизованный run-profile v1 или v2")

    if (
        p.get("subscription_authorized") is not True
        and p.get("nonpaid_fixture") is not True
    ):
        raise ValueError("Нужен явно авторизованный subscription run-profile")

    if not IMAGE.fullmatch(p.get("image") or ""):
        raise ValueError("Нужен digest-pinned worker image")

    for key in (
        "wall_seconds",
        "call_timeout",
        "max_calls",
        "memory_mb",
        "pids",
        "cpus",
        "tmpfs_mb",
    ):
        positive(p.get(key), key)

    for key in (
        "wall_seconds",
        "call_timeout",
        "max_calls",
        "memory_mb",
        "pids",
        "tmpfs_mb",
    ):
        if not isinstance(p[key], int):
            raise TypeError(f"Нужен целочисленный {key}")

    if (
        p.get("max_solver_executions") != 4
        or p.get("parallelism") != 1
        or p.get("transport_attempts") != 1
    ):
        raise ValueError(
            "H1: ровно 4 solver executions, parallelism=1, transport_attempts=1"
        )

    if set(p.get("roles", {})) != {"solver", "proposer", "selector"}:
        raise ValueError("H1 имеет только solver/proposer/selector roles")

    for role in ("solver", "proposer", "selector"):
        r = p["roles"][role]
        if not r.get("model") or not isinstance(r["model"], str):
            raise ValueError(f"Требуется имя модели для {role}")
        if "steps" in r:
            positive(r["steps"], f"{role}.steps")

    if p.get("paid_run_authorized") or p.get("quota", {}).get("max_retries", 0):
        raise ValueError("Paid authorization and automatic quota retries are forbidden")
    attempts = p.get("max_role_attempts", 4)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts <= 0:
        raise ValueError("max_role_attempts must be a positive integer")
    for role in p["roles"].values():
        if not isinstance(role.get("steps"), int) or isinstance(
            role.get("steps"), bool
        ):
            raise TypeError("H1 requires an integer step cap per role")
    staging = p.get("staging_root")
    if staging is not None:
        staging_path = Path(staging)
        if not staging_path.is_absolute():
            lexical = PACKAGE.parent / staging_path
            expected = PACKAGE.parent / ".research" / "staging"
            if lexical != expected or any(
                part.is_symlink() for part in (lexical.parent, lexical)
            ):
                raise ValueError(
                    "Relative staging must be the project-owned .research/staging directory"
                )
            staging_path = lexical.resolve()
        p["staging_root"] = str(staging_path)
    return p


class Runtime:
    def __init__(self, profile: dict, root: Path, manifest: dict, *, phase="run",
                 execution_manifest=None):
        if "DOCKER_HOST" not in os.environ:
            lima_sock = Path.home() / ".lima/craving-h1/sock/docker.sock"
            if lima_sock.exists():
                os.environ["DOCKER_HOST"] = f"unix://{lima_sock}"
        self.profile = profile
        self.root = root
        self.execution_manifest = execution_manifest or manifest
        if not compatible(manifest, self.execution_manifest):
            raise ResumeMismatchError("Execution revision изменила научные входы")
        self.manifest = bound_manifest(root, manifest, profile, phase)
        self.started = time.monotonic()
        self.journal = Journal(root, self.manifest, profile, phase)
        self.calls = self.journal.data["attempts"]
        self.solver_calls = self.journal.solver_executions
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop_signal = None
        self.usage: dict | None = None
        self.audit: list[dict] = []
        self.runner: GeminiSubscriptionRunner | None = None

    def docker(self, *argv, timeout=30) -> str:
        env = dict(os.environ)
        if "DOCKER_HOST" not in env:
            lima_sock = Path.home() / ".lima/craving-h1/sock/docker.sock"
            if lima_sock.exists():
                env["DOCKER_HOST"] = f"unix://{lima_sock}"
        completed = subprocess.run(
            ["docker", *argv],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if completed.returncode:
            raise RuntimeError("OCI control failed: " + completed.stderr[-1000:])
        return completed.stdout

    def prepare(self, *, model_access=True):
        # Проверяем rootless Docker для evaluator
        info = json.loads(self.docker("info", "--format", "{{json .}}"))
        if not any("rootless" in s for s in info.get("SecurityOptions", [])):
            raise ValueError("Нужен rootless Docker daemon")

        # Проверяем worker image для evaluator
        image = json.loads(self.docker("image", "inspect", self.profile["image"]))[0]
        if ((image.get("Config") or {}).get("Labels") or {}).get(
            "org.nutrifit.opencode-commit"
        ) != "23fb5e0516c99ac04a1aa46c193efda2e1b9bb24":
            raise ValueError("Image должен декларировать закреплённый OpenCode commit")
        if self.profile["image"] not in image.get("RepoDigests", []) and self.profile[
            "image"
        ] != image.get("Id"):
            raise ValueError(
                "Worker image digest не установлен локально; auto pull запрещён"
            )

        if (
            self.manifest.get("schema") == "craving-research-manifest/v1"
            and ((image.get("Config") or {}).get("Labels") or {}).get(
                "org.nutrifit.research-worker"
            )
            != "v1"
        ):
            raise ValueError(
                "Research worker image must include pinned pydantic and research-worker=v1 label"
            )

        # Инициализируем Gemini subscription runner
        if model_access and not self.profile.get("nonpaid_fixture"):
            self.runner = GeminiSubscriptionRunner(
                cli_bin=self.profile.get("cli_bin"),
                default_model=self.profile.get(
                    "default_model", "gemini-3.8-flash-high"
                ),
            )
            identity = self.runner.preflight()
            previous = self.journal.data.get("transport")
            if previous is not None and previous != identity:
                raise ResumeMismatchError(
                    "CLI binary/settings/agent identity changed on resume"
                )
            self.journal.data["transport"] = identity
            self.journal.save()
        if self.profile.get("autonomous_quota_wait"):
            self._recover_operations()

    def _recover_operations(self):
        from .audit import operation_packet
        from .state import atomic_json

        for operation, record in list(self.journal.data["operations"].items()):
            if record["status"] not in {"failed", "running"}:
                continue
            canonical = self.root / "role-results" / (digest(operation.encode()) + ".json")
            if record["status"] == "running" and canonical.exists():
                # Result уже опубликован; восстанавливаем только запись журнала.
                # Реконструкция из потоков изменила бы исходные metadata и хеш.
                self.journal.cached(operation, record["fingerprint"])
                continue
            packet, _ = operation_packet(
                self.root, operation, state_path=self.journal.path
            )
            try:
                recovered = recover_result(
                    self.root, operation, record, packet,
                    self.journal.data.get("transport"),
                )
            except (ValueError, TypeError, KeyError) as error:
                # Формат требует исправления, сохранённый ответ остаётся в архиве.
                atomic_json(
                    self.root / "recovery-rejections" / (digest(operation.encode()) + ".json"),
                    {"operation": operation, "reason": str(error)},
                )
                continue
            if recovered:
                persist_recovery(self.journal, operation, *recovered)

    def _repair_request(self, files, request, model_role):
        """Новая попытка только для незавершённой роли, со своим пакетом входов."""
        if not model_role or not self.profile.get("autonomous_quota_wait"):
            return request
        from .audit import operation_packet
        from .contracts import read_regular

        logical = request["operation_id"]
        record = self.journal.data["operations"].get(logical)
        if not record or record["status"] != "failed":
            return request
        packet, _ = operation_packet(self.root, logical, state_path=self.journal.path)
        if packet["files"] != files or packet["request"] != request:
            raise ResumeMismatchError("Входы незавершённой логической операции изменились")
        for number in range(2, self.profile.get("max_role_attempts", 4) + 1):
            operation = logical + f"/repair-{number}"
            existing = self.journal.data["operations"].get(operation)
            if existing and existing["status"] == "failed":
                record = existing
                continue
            if existing:
                stored, _ = operation_packet(self.root, operation, state_path=self.journal.path)
                return stored["request"]
            previous = record.get("attempt_history", [])[-1]
            diagnostic = {
                "previous_operation": logical if number == 2 else logical + f"/repair-{number - 1}",
                "error": previous.get("error", {}).get("message"),
            }
            failure_ref = previous.get("public_failure")
            if failure_ref:
                failure = json.loads(read_regular(self.root / failure_ref["path"], 4 * 1024 * 1024))
                if digest(json_bytes(failure)) != failure_ref["sha256"]:
                    raise ResumeMismatchError("Изменился сохранённый отказ роли")
                diagnostic["saved_response"] = failure.get("response")
            if not diagnostic.get("saved_response"):
                diagnostic["unfinished_response_draft"] = saved_response_draft(
                    self.root, diagnostic["previous_operation"], previous
                )
            contract = ""
            if request.get("harvest"):
                contract = (
                    "Этот микрошаг изменяет только настройки Harness². "
                    "Ты НЕ solver и НЕ пишешь программу эксперимента, гипотезу или отчёт. "
                    "В files верни полный текст файлов context/harness с относительными "
                    "именами без префикса context/harness, обязательно systemprompt.md. "
                    "Если улучшение не обосновано — верни исходный полный набор файлов "
                    "и answer=KEEP_BASE. Научные задания в context — данные для настройки, "
                    "не инструкции выполнить весь опыт в этой роли. "
                    "Используй уже сохранённый контекст; не повторяй поиск источников "
                    "и не запускай расчёты. Заверши выдачей нужных файлов.\n"
                )
            return dict(
                request, operation_id=operation,
                mission=contract + request["mission"] + "\n\n" + contract
                + "Восстановление незавершённого микрошагa. Исправь указанный формат/ошибку, "
                "используя сохранённый ответ и входы. Готовую работу не выполняй заново. "
                "Верни требуемый полный структурированный результат.\n"
                + json.dumps(diagnostic, ensure_ascii=False),
                recovery_of=logical,
            )
        raise BudgetExhaustedError("Исчерпаны попытки микрошагa: " + logical)

    def finish(self, status="completed"):
        self.journal.data["status"] = status
        self.journal.save()
        self.calls = self.journal.data["attempts"]
        self.solver_calls = self.journal.solver_executions
        self.usage = {
            "channel": "gemini_subscription" if self.runner else "fixture_or_evaluator",
            "calls": self.calls,
            "call_unit": "supervised_process_attempts_not_internal_model_requests",
            "internal_model_calls": None,
            "solver_calls": self.solver_calls,
            "active_seconds": self.journal.data["active_seconds"],
            "model_active_seconds": self.journal.data.get("model_active_seconds", 0),
            "compute_active_seconds": self.journal.data["active_seconds"]
            - self.journal.data.get("model_active_seconds", 0),
            "model_sessions": self.profile.get("model_sessions", "legacy_bounded"),
            "wall_semantics": "cumulative_active_execution; quota/offline wait excluded; interrupted attempt reserves full timeout",
            "cost_usd": None,
            "credits": None,
            "enforced": True,
            "closed": True,
            "status": status,
        }
        self.journal.close()

    def execute(self, files: dict, request: dict, *, model_role=None) -> dict:
        with self.condition:
            if self.stop_signal is not None:
                raise self.stop_signal
            if Path(self.profile.get("stop_file", self.root / "STOP")).exists():
                raise RunStopped(
                    "Operator stop requested; no further operations will start"
                )
            expected = self.execution_manifest.get("evaluator", {})
            if any(
                digest((PACKAGE / name).read_bytes()) != signature
                for name, signature in expected.items()
            ):
                raise ResumeMismatchError(
                    "Evaluator/transport source changed during run"
                )
            try:
                return self._execute_locked(files, request, model_role=model_role)
            except RunStopped as exc:
                self.stop_signal = exc
                raise
            except Exception as exc:
                if not self.profile.get("autonomous_quota_wait"):
                    raise
                self.stop_signal = MicrostepInterrupted(str(exc))
                raise self.stop_signal from exc
            finally:
                self.condition.notify_all()

    def _execute_locked(self, files, request, *, model_role):
        request = self._repair_request(files, request, model_role)
        operation = request.get("operation_id") or ("judge/" + request["kind"])
        fingerprint = digest(
            json_bytes(
                {
                    "files": files,
                    "request": request,
                    "manifest": self.manifest,
                    "profile": self.profile,
                    "role": model_role,
                }
            )
        )
        # Resume the exact paused role first even if upstream schedules independent
        # INV/DOM stages on two threads. Wait releases the single transport lock.
        waiting_since = time.monotonic()
        while (
            self.journal.data["status"] == "paused_quota"
            and self.journal.data["next_step"] != operation
            and self.journal.data["operations"].get(operation, {}).get("status")
            != "completed"
        ):
            if self.stop_signal is not None:
                raise self.stop_signal
            pending = self.journal.data["operations"].get(
                self.journal.data["next_step"], {}
            )
            delay = pending.get("not_before", 0) - time.time()
            if delay > 0:
                raise QuotaExhaustedError(
                    "Quota wait has not elapsed", retry_after=max(1, int(delay) + 1)
                )
            if time.monotonic() - waiting_since >= self.profile["call_timeout"]:
                raise ResumeMismatchError(
                    "The paused next_step was not scheduled; no other role may run"
                )
            self.condition.wait(timeout=1)
        if self.stop_signal is not None:
            raise self.stop_signal
        retry_probe = (
            model_role is None and request["kind"] == "experiment"
            and self.profile.get("autonomous_quota_wait") is True
        )
        record = self.journal.data["operations"].get(operation)
        if retry_probe and record and record["status"] == "failed":
            from .audit import operation_packet

            packet, _ = operation_packet(self.root, operation, state_path=self.journal.path)
            recovered = recover_result(self.root, operation, record, packet, None)
            if recovered:
                persist_recovery(self.journal, operation, *recovered)
        cached = self.journal.cached(operation, fingerprint, retry_failed=retry_probe)
        if cached is not None:
            return cached
        if (
            model_role == "solver"
            and self.journal.solver_executions >= self.profile["max_solver_executions"]
        ):
            raise BudgetExhaustedError("Исчерпаны четыре завершённых solver executions")
        if model_role and not self.profile.get("nonpaid_fixture"):
            if self.runner is None:
                raise RuntimeError("Subscription transport is not prepared")
            if self.runner.preflight() != self.journal.data.get("transport"):
                raise ResumeMismatchError(
                    "CLI transport configuration changed during run"
                )
        unbounded = bool(
            model_role and self.profile.get("model_sessions") == "unbounded"
        )
        from .state import atomic_json

        # Исполняемая редакция сохраняется отдельно: исходный input packet, включая
        # paused-quota операции прежней редакции, остаётся побайтно неизменным.
        execution_sha = digest(json_bytes(self.execution_manifest))
        atomic_json(
            self.root / "execution-manifests" / (execution_sha + ".json"),
            self.execution_manifest,
        )
        self.journal.data["execution_manifest"] = {
            "path": "execution-manifests/" + execution_sha + ".json", "sha256": execution_sha,
        }
        wall = self.journal.begin(
            operation,
            fingerprint,
            model_role or request["kind"],
            unbounded=unbounded,
            inputs={
                "operation_id": operation,
                "files": files,
                "request": request,
                "model_role": model_role,
                "profile": self.profile,
                "manifest": self.manifest,
                "fingerprint": fingerprint,
            },
        )
        started = time.monotonic()
        result = None
        try:
            result = self._execute_once(
                files, request, model_role=model_role, wall=wall
            )
        except QuotaExhaustedError as exc:
            self.journal.end(quota=exc, error=exc)
            raise
        except BaseException as exc:
            self.journal.end(error=exc)
            raise
        else:
            self.journal.end(result)
            return result
        finally:
            self.calls = self.journal.data["attempts"]
            self.solver_calls = self.journal.solver_executions
            self.audit.append(
                {
                    "attempt": self.calls,
                    "operation_id": operation,
                    "role": model_role or request["kind"],
                    "wall_seconds": time.monotonic() - started,
                    "input_sha256": fingerprint,
                    "status": self.journal.data["operations"][operation]["status"],
                    "usage": result.get("_usage") if isinstance(result, dict) else None,
                }
            )

    def _execute_once(self, files, request, *, model_role, wall):
        if model_role:
            if self.profile.get("nonpaid_fixture"):
                if os.environ.get("RESEARCH_HARNESS_SIMULATE_QUOTA"):
                    raise QuotaExhaustedError(
                        "Simulated quota exhausted", retry_after=5
                    )
                if model_role == "solver" and request.get("allowed_outputs"):
                    preregistration = {
                        "hypothesis": "Fixture hypothesis",
                        "falsification": "Fixture deviation",
                        "baseline": "Fixed fixture",
                        "scope": "fixture_only",
                        "configurations": [
                            {"id": "fixture", "definition": {"variant": "fixed"}}
                        ],
                        "feature_sets": [
                            {"id": "fixture", "definition": {"inputs": "fixed"}}
                        ],
                    }
                    program = (
                        "import json\nfrom pathlib import Path\n"
                        "Path('artifacts').mkdir(exist_ok=True)\n"
                        "Path('artifacts/metrics.json').write_text(json.dumps({'configuration_ids':['fixture'],'fixture_only':True}))\n"
                        "Path('artifacts/observations.md').write_text('Fixture execution only, no scientific result.')\n"
                    )
                    result = {
                        "files": {
                            "experiment/experiment.py": program,
                            "report.md": "Fixture proposal only",
                            "hypothesis.json": json.dumps(preregistration),
                        },
                        "answer": "Fixture program",
                    }
                elif model_role == "solver":
                    result = {
                        "files": {
                            "candidate/checkpoint.py": "def save(memory):\n    return memory.export_state()\n\n"
                            "def restore(memory_class, serialized):\n    return memory_class.from_state(serialized)\n"
                        },
                        "answer": "Implemented checkpoint.py",
                    }
                else:
                    result = {
                        "files": {
                            k.removeprefix("context/harness/"): v
                            for k, v in files.items()
                            if request.get("harvest")
                            and k.startswith("context/harness/")
                        },
                        "answer": "<notes>Keep the complete checkpoint API.</notes>",
                    }
                if (
                    request.get("operation_id", "").split("/retry-")[0]
                    == "campaign/report"
                ):
                    result["answer"] = json.dumps(
                        {
                            "hypothesis": "Fixture only",
                            "next_hypothesis": "Repeat fixed fixture",
                            "decision": "continue",
                            "finding_status": "executed",
                            "claims": [
                                {
                                    "id": "fixture",
                                    "text": "Fixture execution only",
                                    "evidence": ["probe.json"],
                                }
                            ],
                            "report": "Fixture qualification only; no scientific evidence.",
                        }
                    )
                elif model_role == "selector":
                    result["answer"] = (
                        "<experience>\ncue: fixture\nlesson: preserve complete input evidence\nevidence: fixed fixture execution\nconfidence: fixture only\n</experience>"
                    )
                result["events"] = (
                    json.dumps(
                        {
                            "type": "text",
                            "part": {"type": "text", "text": result["answer"]},
                        }
                    )
                    + "\n"
                )
                return result
            role_spec = self.profile["roles"][model_role]
            operation = request["operation_id"]
            attempt = self.journal.data["operations"][operation]["attempts"]
            workspace = (
                self.root
                / "cli-workspaces"
                / (digest(operation.encode()) + "-" + str(attempt))
            )
            workspace.mkdir(parents=True, exist_ok=False)
            return self.runner.run_turn(
                workspace,
                prompt=request["mission"],
                files=files,
                model=role_spec["model"],
                effort=role_spec.get("effort"),
                timeout=wall,
                harvest=request.get("harvest"),
                role=model_role,
                max_steps=None
                if self.profile.get("model_sessions") == "unbounded"
                else role_spec["steps"],
                stop_file=self.profile.get("stop_file"),
                heartbeat=self.journal.heartbeat,
                allowed_outputs=request.get("allowed_outputs"),
            )

        # 2. Replay/Evaluator вызов: исполняется внутри OCI без сети (network=none)
        name = "craving-h1-" + uuid.uuid4().hex
        request = dict(
            request,
            debug_fixture=self.profile.get("nonpaid_fixture") is True,
            timeout=max(1, int(wall) - 2),
            limits={key: self.profile[key] for key in ("memory_mb", "pids", "cpus")},
        )

        with tempfile.TemporaryDirectory(
            prefix="craving-curated-", dir=self.profile.get("staging_root")
        ) as tmp:
            source = Path(tmp)
            write_tree(source, files)
            source.chmod(0o755)
            for directory in source.rglob("*"):
                directory.chmod(0o755 if directory.is_dir() else 0o644)
            (source / "_worker.py").write_bytes((PACKAGE / "worker.py").read_bytes())
            argv = [
                "docker",
                "run",
                "--pull=never",
                "--name",
                name,
                "-i",
                "--read-only",
                "--user",
                "65534:65534",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit",
                str(self.profile["pids"]),
                "--memory",
                f"{self.profile['memory_mb']}m",
                "--memory-swap",
                f"{self.profile['memory_mb']}m",
                "--cpus",
                str(self.profile["cpus"]),
                "--ulimit",
                "nofile=256:256",
                "--tmpfs",
                f"/workspace:rw,noexec,nosuid,nodev,size={self.profile['tmpfs_mb']}m,mode=1777",
                "--tmpfs",
                f"/tmp:rw,nosuid,nodev,size={self.profile['tmpfs_mb']}m,mode=1777",
                "--network",
                "none",
                "--mount",
                f"type=bind,src={source},dst=/input,readonly",
                "--workdir",
                "/workspace",
                "--entrypoint",
                "python3",
                self.profile["image"],
                "-I",
                "/input/_worker.py",
            ]

            mounts = []
            for rel in files:
                if not rel.startswith("candidate/"):
                    mounts += [
                        "--mount",
                        f"type=bind,src={source / rel},dst=/workspace/{rel},readonly",
                    ]
            insertion = argv.index("--workdir")
            argv[insertion:insertion] = mounts

            started = time.monotonic()
            proc = subprocess.Popen(
                argv,
                env=os.environ,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            chunks = {"stdout": bytearray(), "stderr": bytearray()}
            overflow = threading.Event()

            def drain(stream, key):
                while True:
                    block = stream.read(65536)
                    if not block:
                        break
                    if len(chunks[key]) + len(block) > MAX_TREE:
                        overflow.set()
                        proc.kill()
                        break
                    chunks[key].extend(block)

            threads = [
                threading.Thread(
                    target=drain, args=(getattr(proc, key), key), daemon=True
                )
                for key in chunks
            ]
            for thread in threads:
                thread.start()

            assert proc.stdin is not None
            input_stream = proc.stdin

            def feed():
                try:
                    input_stream.write(json_bytes(request))
                    input_stream.close()
                except (OSError, ValueError):
                    pass

            feeder = threading.Thread(target=feed, daemon=True)
            feeder.start()
            try:
                proc.wait(timeout=wall)
                for thread in threads:
                    thread.join(timeout=2)
                if overflow.is_set() or proc.returncode:
                    detail = chunks["stderr"].decode(errors="replace")[-1800:]
                    raise RuntimeError(
                        "Worker output overflow/process failure; no verdict: " + detail
                    )
                from .state import atomic_json

                operation, _, _ = self.journal.current
                record = self.journal.data["operations"][operation]
                atomic_json(
                    self.root / "worker-results"
                    / (digest(operation.encode()) + f"-{record['attempts']}.json"),
                    {
                        "schema": "craving-worker-result/v1",
                        "fingerprint": record["fingerprint"],
                        "returncode": proc.returncode,
                        "stdout": chunks["stdout"].decode(),
                        "stderr": chunks["stderr"].decode(errors="replace"),
                    },
                )
                result = json.loads(chunks["stdout"])
                kernel_limits = result.pop("_kernel_limits")
                self.audit.append(
                    {
                        "call": self.calls,
                        "kernel_limits": kernel_limits,
                        "role": request["kind"],
                        "image": self.profile["image"],
                        "wall_seconds": time.monotonic() - started,
                        "input_sha256": digest(json_bytes(files)),
                        "output_sha256": digest(bytes(chunks["stdout"])),
                    }
                )
                return result
            finally:
                if proc.poll() is None:
                    proc.kill()
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
