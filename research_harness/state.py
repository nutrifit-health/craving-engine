"""Атомарный журнал попыток и точных входов; неизвестный исход не повторяется."""

from __future__ import annotations

import fcntl
import json
import os
import time
import traceback
from pathlib import Path

from .contracts import digest, json_bytes, read_regular, relative


class RunStopped(BaseException):
    """Сигнал супервизору, который upstream fallback не должен перехватывать."""


class ResumeMismatchError(RunStopped):
    pass


class BudgetExhaustedError(RunStopped):
    pass


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_journal(path):
    """Read v2 journals, including read-only legacy history before ledger migration.

    Only legacy v2 journals may exceed the normal JSON cap. New journals keep
    their bounded ledger representation; other JSON readers retain their caps.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ResumeMismatchError("Journal must be a regular file")
    oversized = path.stat().st_size > 4 * 1024 * 1024
    if oversized:
        with path.open() as stream:
            data = json.load(stream)
    else:
        data = json.loads(read_regular(path, 4 * 1024 * 1024))
    if not isinstance(data, dict) or data.get("schema") != "craving-harness-state/v2":
        raise ResumeMismatchError("Unknown journal schema")
    if oversized and data.get("autonomous_attempt_ledger"):
        raise ResumeMismatchError("Bounded autonomous journal unexpectedly oversized")
    return data


def archive_original_journal(destination, original):
    """Commit exact original bytes atomically; interrupted temp writes can retry."""
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != original:
            raise ResumeMismatchError("Original journal migration archive changed")
        return
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def attempt_history(root, record):
    """Stream every immutable attempt and verify its chain before the active tail."""
    ledger = record.get("attempt_ledger")
    count = 0
    if ledger:
        relative(ledger["path"])
        previous = None
        for number in range(1, ledger["count"] + 1):
            row = json.loads(
                read_regular(
                    Path(root) / ledger["path"] / f"{number:09d}.json", 4 * 1024 * 1024
                )
            )
            if (
                row["schema"] != "craving-operation-attempt/v1"
                or row["previous_sha256"] != previous
                or row["record"]["attempt"] != number
            ):
                raise ResumeMismatchError("Immutable attempt ledger chain changed")
            previous = digest(json_bytes(row))
            count = number
            yield row["record"]
        if previous != ledger["head_sha256"]:
            raise ResumeMismatchError("Immutable attempt ledger head changed")
    for row in record.get("attempt_history", []):
        if row["attempt"] > count:
            yield row


class Journal:
    def __init__(self, root, manifest, profile, phase="run"):
        self.root, self.profile = Path(root), profile
        self.lock = (self.root / ".supervisor.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ResumeMismatchError("Этот run уже открыт другим супервизором")
        self.path = self.root / (phase + "-state.json")
        self.binding = digest(
            json_bytes({"manifest": manifest, "profile": profile, "phase": phase})
        )
        if self.path.exists():
            self.data = read_journal(self.path)
            if (
                self.data.get("schema") != "craving-harness-state/v2"
                or self.data.get("binding") != self.binding
            ):
                self.close()
                raise ResumeMismatchError("Journal manifest/phase mismatch")
            if profile.get("autonomous_quota_wait") and not self.data.get(
                "autonomous_attempt_ledger"
            ):
                original = self.path.read_bytes()
                name = "journal-migrations/" + digest(original) + ".json"
                destination = self.root / name
                archive_original_journal(destination, original)
                self.data["attempt_history_migration"] = {
                    "path": name,
                    "sha256": digest(original),
                }
                for record in self.data["operations"].values():
                    old = record.get("attempt_history", [])
                    record["quota_attempts"] = sum(
                        a.get("status") == "paused_quota" for a in old
                    )
                    record["quota_active_seconds"] = sum(
                        a.get("active_seconds", 0)
                        for a in old
                        if a.get("status") == "paused_quota"
                    )
                self.data["autonomous_attempt_ledger"] = True
                self.compact_history()
                self.save()
        else:
            self.data = {
                "schema": "craving-harness-state/v2",
                "binding": self.binding,
                "attempts": 0,
                "active_seconds": 0.0,
                "operations": {},
                "next_step": None,
                "status": "in_progress",
                "autonomous_attempt_ledger": profile.get("autonomous_quota_wait")
                is True,
            }
            self.save()
        self.current = None

    def compact_history(self):
        if not self.profile.get("autonomous_quota_wait"):
            return
        for operation, record in self.data["operations"].items():
            ledger = record.setdefault(
                "attempt_ledger",
                {
                    "path": "attempt-ledger/" + digest(operation.encode()),
                    "count": 0,
                    "head_sha256": None,
                },
            )
            for attempt in record.get("attempt_history", []):
                if (
                    attempt["status"] == "running"
                    or attempt["attempt"] <= ledger["count"]
                ):
                    continue
                if attempt["attempt"] != ledger["count"] + 1:
                    raise ResumeMismatchError("Attempt history cannot skip evidence")
                row = {
                    "schema": "craving-operation-attempt/v1",
                    "operation_id": operation,
                    "previous_sha256": ledger["head_sha256"],
                    "record": attempt,
                }
                destination = (
                    self.root / ledger["path"] / f"{attempt['attempt']:09d}.json"
                )
                if destination.exists():
                    if json.loads(read_regular(destination, 4 * 1024 * 1024)) != row:
                        raise ResumeMismatchError("Immutable attempt archive changed")
                else:
                    atomic_json(destination, row)
                ledger.update(
                    count=attempt["attempt"], head_sha256=digest(json_bytes(row))
                )
            record["attempt_history"] = record.get("attempt_history", [])[-2:]

    def save(self):
        atomic_json(self.path, self.data)

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def cached(self, operation, fingerprint):
        record = self.data["operations"].get(operation)
        if not record:
            if self.data["status"] == "paused_quota":
                from .subscription import QuotaExhaustedError

                raise QuotaExhaustedError(
                    "Run приостановлен на другой роли; сначала возобновите next_step"
                )
            return None
        if record["fingerprint"] != fingerprint:
            raise ResumeMismatchError(
                "Изменены входы завершённой/приостановленной роли: " + operation
            )
        output = self.root / "role-results" / (digest(operation.encode()) + ".json")
        if record["status"] in ("completed", "running") and output.exists():
            result = json.loads(read_regular(output, 4 * 1024 * 1024))
            if result.get("fingerprint") != fingerprint or result.get(
                "result_sha256"
            ) != digest(json_bytes(result.get("result"))):
                raise ResumeMismatchError("Повреждён role result: " + operation)
            if (
                record["status"] == "completed"
                and record.get("result_sha256") != result["result_sha256"]
            ):
                raise ResumeMismatchError(
                    "Role result отличается от journal: " + operation
                )
            # Ответ был durable до прерывания записи journal; повтор транспорта не нужен.
            recovered = record["status"] == "running"
            record.update(status="completed", result_sha256=result["result_sha256"])
            if recovered and self.data["next_step"] == operation:
                self.data["next_step"] = None
            if recovered and self.profile.get("autonomous_quota_wait"):
                tail = record["attempt_history"][-1]
                tail.update(
                    status="completed",
                    result_sha256=result["result_sha256"],
                    active_seconds=tail.get("accounted_seconds", 0)
                    if tail["reserved_seconds"] is None
                    else tail["reserved_seconds"],
                    ended_at=time.time(),
                    accounting="durable heartbeat time retained; crash tail duration unknown"
                    if tail["reserved_seconds"] is None
                    else "crash recovery conservatively retains reserved timeout",
                )
            self.save()
            if recovered:
                self.compact_history()
                self.save()
            return result["result"]
        if record["status"] in ("completed", "running", "failed"):
            raise ResumeMismatchError(
                "Исход операции неизвестен либо завершён с ошибкой; автоматический повтор запрещён: "
                + operation
            )
        if time.time() < record.get("not_before", 0):
            from .subscription import QuotaExhaustedError

            raise QuotaExhaustedError(
                "Пауза квоты ещё не истекла",
                retry_after=max(1, int(record["not_before"] - time.time()) + 1),
            )
        return None

    def begin(self, operation, fingerprint, role, inputs=None, *, unbounded=False):
        record = self.data["operations"].get(
            operation,
            {
                "fingerprint": fingerprint,
                "attempts": 0,
                "role": role,
                "quota_attempts": 0,
                "quota_active_seconds": 0,
            },
        )
        # Autonomous waiting renews only transport attempts rejected by quota.
        # The raw attempt ledger and its time remain cumulative and immutable.
        renewable = self.profile.get("autonomous_quota_wait") is True
        quota_attempts = record.get("quota_attempts", 0) if renewable else 0
        all_quota = (
            sum(r.get("quota_attempts", 0) for r in self.data["operations"].values())
            if renewable
            else 0
        )
        quota_seconds = (
            sum(
                r.get("quota_active_seconds", 0)
                for r in self.data["operations"].values()
            )
            if renewable
            else 0
        )
        if record["attempts"] - quota_attempts >= self.profile.get(
            "max_role_attempts", 4
        ):
            raise BudgetExhaustedError("Исчерпаны попытки роли: " + operation)
        remaining = (
            self.profile["wall_seconds"]
            - self.data["active_seconds"]
            + self.data.get("model_active_seconds", 0)
            + quota_seconds
            - sum(
                r.get("model_quota_active_seconds", 0)
                for r in self.data["operations"].values()
            )
        )
        if (not unbounded and remaining <= 0) or self.data[
            "attempts"
        ] - all_quota >= self.profile["max_calls"]:
            raise BudgetExhaustedError(
                "Исчерпан durable active-time/process-attempt budget"
            )
        wall = (
            None if unbounded else min(float(self.profile["call_timeout"]), remaining)
        )
        # Reserve before starting: an abrupt supervisor crash cannot reset consumed budget.
        self.data["active_seconds"] += wall or 0
        self.data["attempts"] += 1
        if inputs is not None:
            input_hash = digest(json_bytes(inputs))
            if record.get("inputs_sha256") not in (None, input_hash):
                raise ResumeMismatchError("Original operation inputs changed")
            atomic_json(
                self.root / "operation-inputs" / (digest(operation.encode()) + ".json"),
                inputs,
            )
            record["inputs_sha256"] = input_hash
        record.update(status="running", attempts=record["attempts"] + 1)
        record.setdefault("attempt_history", []).append(
            {
                "attempt": record["attempts"],
                "started_at": time.time(),
                "status": "running",
                "reserved_seconds": wall,
                "unbounded_model_session": unbounded,
                "accounted_seconds": 0,
            }
        )
        record.pop("not_before", None)
        self.data["operations"][operation] = record
        self.data.update(next_step=operation, status="in_progress")
        self.current = (operation, time.monotonic(), wall)
        self.save()
        return wall

    def heartbeat(self, *, force=False):
        if self.current is None or self.current[2] is not None:
            return
        operation, started, _ = self.current
        attempt = self.data["operations"][operation]["attempt_history"][-1]
        elapsed = time.monotonic() - started
        delta = elapsed - attempt.get("accounted_seconds", 0)
        if delta < 1 and not force:
            return
        self.data["active_seconds"] += delta
        self.data["model_active_seconds"] = (
            self.data.get("model_active_seconds", 0) + delta
        )
        attempt.update(accounted_seconds=elapsed, last_heartbeat_at=time.time())
        self.save()

    def end(self, result=None, *, quota=None, error=None):
        operation, started, reserved = self.current
        record = self.data["operations"][operation]
        if reserved is None:
            self.heartbeat(force=True)
        else:
            self.data["active_seconds"] += (time.monotonic() - started) - reserved
        if result is not None:
            result_hash = digest(json_bytes(result))
            atomic_json(
                self.root / "role-results" / (digest(operation.encode()) + ".json"),
                {
                    "fingerprint": record["fingerprint"],
                    "result": result,
                    "result_sha256": result_hash,
                },
            )
            record.update(
                status="completed",
                result_sha256=result_hash,
                usage=result.get("_usage"),
            )
            self.data["next_step"] = None
        elif quota is not None:
            record.update(
                status="paused_quota", not_before=time.time() + quota.retry_after
            )
            self.data["status"] = "paused_quota"
        else:
            record["status"] = "failed"
            self.data["status"] = "inconclusive"
        attempt = record["attempt_history"][-1]
        attempt.update(
            status=record["status"],
            active_seconds=time.monotonic() - started,
            ended_at=time.time(),
        )
        if quota is not None:
            attempt["not_before"] = record["not_before"]
            record["quota_attempts"] = record.get("quota_attempts", 0) + 1
            record["quota_active_seconds"] = (
                record.get("quota_active_seconds", 0) + attempt["active_seconds"]
            )
            if reserved is None:
                record["model_quota_active_seconds"] = (
                    record.get("model_quota_active_seconds", 0)
                    + attempt["active_seconds"]
                )
        if error is not None:
            from .subscription import redact

            attempt["error"] = {
                "type": type(error).__name__,
                "message": redact(str(error)),
                "traceback": redact("".join(traceback.format_exception(error))),
            }
            if hasattr(error, "public_failure"):
                failure = error.public_failure
                name = (
                    "failed-attempts/"
                    + digest(operation.encode())
                    + "-"
                    + str(record["attempts"])
                    + ".json"
                )
                atomic_json(self.root / name, failure)
                attempt["public_failure"] = {
                    "path": name,
                    "sha256": digest(json_bytes(failure)),
                }
        if result is not None:
            attempt["result_sha256"] = record["result_sha256"]
        archive = (
            self.root
            / "cli-workspaces"
            / (
                digest(operation.encode())
                + "-"
                + str(record["attempts"])
                + "-archive.json"
            )
        )
        if archive.exists():
            attempt["workspace_archive"] = {
                "path": archive.relative_to(self.root).as_posix(),
                "sha256": digest(archive.read_bytes()),
            }
        self.current = None
        # Persist the completed tail first. Archiving and its metadata can then
        # recover idempotently after a crash without losing the previous bytes.
        self.save()
        self.compact_history()
        self.save()

    @property
    def solver_executions(self):
        return sum(
            r["role"] == "solver" and r["status"] == "completed"
            for r in self.data["operations"].values()
        )
