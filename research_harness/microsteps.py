"""Восстановление операций из доказанных результатов без повторного исполнения."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import digest, digest_file, json_bytes, read_regular, relative
from .state import BudgetExhaustedError, ResumeMismatchError, RunStopped, atomic_json, read_journal


class MicrostepInterrupted(RunStopped):
    """Остановить планировщик до fallback и продолжить незавершённую операцию."""


# Только операционные модули. Worker, критик, критерии, данные и harness
# сравниваются полностью. Каждая пара редакций требует явного recover-round.
REPAIR_MODULES = {
    "autonomy.py", "runtime.py", "state.py", "subscription.py",
    "queue.py", "microsteps.py", "audit.py", "research.py",
}


def scientific_manifest(manifest):
    value = json.loads(json_bytes(manifest))
    for key, prefix in (("evaluator", ""), ("controller_sources", "research_harness/")):
        if key in value:
            value[key] = {
                name: sha for name, sha in value[key].items()
                if name not in {prefix + module for module in REPAIR_MODULES}
            }
    return value


def compatible(before, after):
    return scientific_manifest(before) == scientific_manifest(after)


def original_manifest(root, phase="run"):
    state = read_journal(Path(root) / (phase + "-state.json"))
    for operation, record in state["operations"].items():
        packet = json.loads(read_regular(
            Path(root) / "operation-inputs" / (digest(operation.encode()) + ".json"),
            4 * 1024 * 1024,
        ))
        if digest(json_bytes(packet)) != record.get("inputs_sha256"):
            raise ResumeMismatchError("Изменился исходный пакет операции")
        return packet["manifest"]
    raise ResumeMismatchError("Нет исходного пакета для восстановления")


def bound_manifest(root, execution_manifest, profile, phase):
    state_path = Path(root) / (phase + "-state.json")
    if not state_path.exists():
        return execution_manifest
    state = read_journal(state_path)
    expected = digest(json_bytes({
        "manifest": execution_manifest, "profile": profile, "phase": phase,
    }))
    if state["binding"] == expected:
        return execution_manifest
    original = original_manifest(root, phase)
    authorization_path = Path(root) / "controller-recoveries" / (
        digest(json_bytes(execution_manifest.get("controller_sources"))) + ".json"
    )
    if not authorization_path.exists():
        raise ResumeMismatchError("Смена редакции требует явного recover-round")
    authorization = json.loads(read_regular(authorization_path, 4 * 1024 * 1024))
    if (
        not compatible(original, execution_manifest)
        or authorization.get("old_controller_sources") != original.get("controller_sources")
        or authorization.get("new_controller_sources") != execution_manifest.get("controller_sources")
        or authorization.get("profile_sha256") != digest(json_bytes(profile))
        or state["binding"] != digest(json_bytes({
            "manifest": original, "profile": profile, "phase": phase,
        }))
    ):
        raise ResumeMismatchError("Изменились значимые входы или разрешённая редакция")
    return original


def archived_events(workspace, archive):
    """Сверить raw-поток с журналом, сериализованным после удаления секретов."""
    from itertools import zip_longest

    raw_path = workspace / "stream-stdout.jsonl"
    events_path = workspace / "stream-events.jsonl"
    if "stream-events.jsonl" not in archive["files"]:
        with raw_path.open() as source:
            yield from (json.loads(line) for line in source)
        return

    def normalized(source):
        for line in source:
            event = json.loads(line)
            # Проекция answer добавляется после полного transport_result.
            if event.get("event") != "result":
                yield event

    with raw_path.open() as raw, events_path.open() as projected:
        for line, saved in zip_longest(raw, normalized(projected)):
            if line is None or saved is None:
                raise ResumeMismatchError("Потоки CLI содержат разное число событий")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Потоковый redactor может удалить закрывающий JSON-синтаксис
                # вместе со значением токена. Допустима только диагностическая
                # tool-строка с независимо сериализованной копией того же события.
                prefix = line[:512]
                step = saved.get("step_update", {})
                index = re.search(r'"step_index"\s*:\s*(\d+)', prefix)
                state = re.search(r'"state"\s*:\s*"([^"]+)"', prefix)
                if (
                    "[REDACTED]" not in line
                    or not re.match(r'\s*\{\s*"event"\s*:\s*"step_update"\s*,', prefix)
                    or not re.search(r'"step_type"\s*:\s*"tool"', prefix)
                    or saved.get("event") != "step_update"
                    or step.get("step_type") != "tool"
                    or not index or int(index.group(1)) != step.get("step_index")
                    or not state or state.group(1) != step.get("state")
                ):
                    raise ResumeMismatchError("Повреждённое событие CLI не восстановимо")
                event = saved
            kind = event.get("event")
            expected_kind = "transport_result" if kind == "result" else kind
            if saved.get("event") != expected_kind:
                raise ResumeMismatchError("Порядок событий в копиях CLI расходится")
            if kind == "init":
                init = event.get("init", {})
                if init.get("model") != saved.get("model"):
                    raise ResumeMismatchError("Модель в копиях CLI расходится")
            elif kind == "result":
                from .subscription import redact

                if redact(event.get("result")) != saved.get("result"):
                    raise ResumeMismatchError("Завершающие ответы CLI расходятся")
            elif kind == "step_update":
                if event.get("step_update", {}).get("step_index") != saved.get("step_update", {}).get("step_index"):
                    raise ResumeMismatchError("Шаги в копиях CLI расходятся")
            yield event


def recover_result(root, operation, record, packet, identity):
    """Читать только сохранённый полный поток; не исполнять инструменты модели."""
    root = Path(root)
    tail = record.get("attempt_history", [])[-1:]
    if not tail:
        return None
    attempt = tail[0]
    key = digest(operation.encode())
    worker_result = root / "worker-results" / f"{key}-{attempt['attempt']}.json"
    if packet["model_role"] is None:
        if not worker_result.exists():
            return None
        saved = json.loads(read_regular(worker_result, 4 * 1024 * 1024))
        if saved["fingerprint"] != record["fingerprint"] or saved["returncode"] != 0:
            raise ResumeMismatchError("Не совпадает сохранённый ответ worker")
        result = json.loads(saved["stdout"])
        result.pop("_kernel_limits")
        return result, {
            "source": str(worker_result.relative_to(root)),
            "sha256": digest_file(worker_result),
            "kind": "worker_output_recovered",
        }

    workspace = root / "cli-workspaces" / f"{key}-{attempt['attempt']}"
    seal_path = workspace.with_name(workspace.name + "-sealed.json")
    archive_ref = attempt.get("workspace_archive")
    legacy_closure = attempt.get("error", {}).get("message") in {
        "CLI terminal result received but stream did not close",
        "CLI stream did not close after terminal cleanup",
        "Transport readers did not complete",
    }
    if seal_path.exists():
        # Публикация archive могла упасть после завершения и сохранения потоков.
        # Завершаем локальную обработку по отдельной печати, без вызова модели.
        archive_ref = {
            "path": str(seal_path.relative_to(root)), "sha256": digest_file(seal_path),
        }
    elif not archive_ref or (record["status"] == "failed" and not legacy_closure):
        if (workspace / "transport-result.json").exists():
            raise ResumeMismatchError(
                "Ответ модели сохранён, но нет полной печати потоков; "
                "нужно восстановить публикацию, повтор модели запрещён"
            )
        return None
    relative(archive_ref["path"])
    archive_path = root / archive_ref["path"]
    if digest_file(archive_path) != archive_ref["sha256"]:
        raise ResumeMismatchError("Изменился архив ответа модели")
    archive = json.loads(read_regular(archive_path, 4 * 1024 * 1024))
    if (
        seal_path.exists()
        and archive.get("terminal_validated") is not True
        and not legacy_closure
    ):
        if (workspace / "transport-result.json").exists():
            raise ResumeMismatchError(
                "Сохранённый terminal не прошёл проверку завершения транспорта; "
                "нужен разбор, повтор модели запрещён"
            )
        return None
    if not archive.get("evidence_complete") or not archive.get("readers_complete"):
        if (workspace / "transport-result.json").exists():
            raise ResumeMismatchError("Сохранённый ответ ожидает восстановления неполного архива")
        return None
    for name, info in archive["files"].items():
        relative(name)
        path = workspace / name
        if path.is_symlink() or not info.get("sha256") or digest_file(path) != info["sha256"]:
            raise ResumeMismatchError("Изменился сохранённый workspace: " + name)
    stream = workspace / "stream-stdout.jsonl"
    attested, terminal = False, None
    model = packet["profile"]["roles"][packet["model_role"]]["model"]
    for event in archived_events(workspace, archive):
        kind = event.get("event")
        if kind == "init":
            init = event.get("init", {})
            if terminal is not None or attested or init.get("model") != model or not isinstance(init.get("tools"), list):
                raise ResumeMismatchError("Неоднозначная аттестация сохранённой сессии")
            attested = True
        elif not attested:
            raise ResumeMismatchError("Ответ предшествует аттестации")
        elif kind == "result":
            if terminal is not None:
                raise ResumeMismatchError("В потоке несколько завершающих ответов")
            terminal = event["result"]
        elif kind != "step_update" or terminal is not None:
            raise ResumeMismatchError("Неоднозначный порядок событий CLI")
    if not terminal or terminal.get("status") != "SUCCESS":
        return None
    terminal_path = workspace / "transport-result.json"
    if terminal_path.exists():
        saved_terminal = json.loads(read_regular(terminal_path, 4 * 1024 * 1024))
        if (
            saved_terminal.get("model") != model
            or saved_terminal.get("role") != packet["model_role"]
            or saved_terminal.get("attested") is not True
            or saved_terminal.get("terminal") != terminal
        ):
            raise ResumeMismatchError("Сохранённый terminal расходится с полным потоком")
    from .subscription import decode_role_result, redact, usage_fields

    request = packet["request"]
    result = decode_role_result(
        terminal.get("response", ""), role=packet["model_role"],
        harvest=request.get("harvest"), files=packet["files"],
        allowed_outputs=request.get("allowed_outputs"),
    )
    result = redact(result)
    result["events"] = json_bytes({
        "event": "result", "status": "SUCCESS", "type": "text",
        "part": {"type": "text", "text": result["answer"]},
        "recovered_from": str(stream.relative_to(root)),
    }).decode() + "\n"
    result["_usage"] = {
        "model": model, "auth": identity, "tokens": usage_fields(terminal.get("usage")),
        "reused": True, "new_model_calls": 0,
    }
    result["_streams"] = {
        name: {"path": str(workspace / name), **archive["files"][name]}
        for name in ("stream-stdout.jsonl", "stream-events.jsonl", "stream-stderr.log")
        if name in archive["files"]
    }
    return result, {
        "kind": "terminal_response_recovered",
        "source": archive_ref["path"], "sha256": archive_ref["sha256"],
        "stream_sha256": digest_file(stream),
        "original_fingerprint": record["fingerprint"],
        "new_model_calls": 0,
    }


def persist_recovery(journal, operation, result, proof):
    """Старые попытки остаются failed; восстановление — отдельный receipt."""
    record = journal.data["operations"][operation]
    key = digest(operation.encode())
    receipt = {
        "schema": "craving-microstep-recovery/v1",
        "operation": operation, "inputs_sha256": record["inputs_sha256"],
        "result_sha256": digest(json_bytes(result)), "proof": proof,
    }
    before = journal.path.read_bytes()
    snapshot = journal.root / "recovery-journals" / (digest(before) + ".json")
    snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():
        from .state import archive_original_journal

        archive_original_journal(snapshot, before)
    receipt["original_journal"] = {
        "path": str(snapshot.relative_to(journal.root)), "sha256": digest(before),
    }
    destination = journal.root / "recovered-results" / (key + ".json")
    if destination.exists():
        saved = json.loads(read_regular(destination))
        if any(saved.get(k) != v for k, v in receipt.items() if k != "original_journal"):
            raise ResumeMismatchError("Изменился receipt восстановления")
        original = saved["original_journal"]
        relative(original["path"])
        if digest_file(journal.root / original["path"]) != original["sha256"]:
            raise ResumeMismatchError("Изменился исходный journal восстановления")
        receipt = saved
    else:
        atomic_json(destination, receipt)
    result_path = journal.root / "role-results" / (key + ".json")
    result_packet = {
        "fingerprint": record["fingerprint"], "result": result,
        "result_sha256": receipt["result_sha256"],
    }
    if result_path.exists() and json.loads(read_regular(result_path, 4 * 1024 * 1024)) != result_packet:
        raise ResumeMismatchError("Сохранённый результат отличается от восстановления")
    atomic_json(result_path, result_packet)
    record.update(
        status="completed", result_sha256=receipt["result_sha256"],
        recovery={"path": str(destination.relative_to(journal.root)), "sha256": digest_file(destination)},
    )
    if journal.data.get("next_step") == operation:
        journal.data["next_step"] = None
    journal.save()


def repair_hypothesis(runtime, output, run_id, declarations=None):
    """Исправить только форму preregistration до регистрации и первого probe."""
    definitions = {}
    for row in (declarations or {}).values():
        if row.get("admitted"):
            for field in ("configurations", "feature_sets"):
                for item in row["hypothesis"][field]:
                    definitions[(field, item["id"])] = item["definition"]

    def shape(body):
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            return False
        strings = {"hypothesis", "falsification", "baseline", "scope"}
        lists = {"configurations", "feature_sets"}
        return (
            isinstance(data, dict) and set(data) == strings | lists
            and all(isinstance(data[k], str) and data[k].strip() for k in strings)
            and all(
                isinstance(data[k], list) and bool(data[k])
                and all(
                    isinstance(row, dict) and set(row) == {"id", "definition"}
                    and isinstance(row["id"], str)
                    and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", row["id"])
                    and isinstance(row["definition"], dict)
                    for row in data[k]
                )
                for k in lists
            )
        )

    def valid(body):
        if not shape(body):
            return False
        data = json.loads(body)
        return all(
            (field, item["id"]) not in definitions
            or item["definition"] == definitions[(field, item["id"])]
            for field in ("configurations", "feature_sets")
            for item in data[field]
        )

    original = output["files"]["hypothesis.json"]
    original_data = json.loads(original) if shape(original) else None
    if not runtime.profile.get("autonomous_quota_wait"):
        return output
    if "probe/" + run_id in runtime.journal.data["operations"]:
        from .audit import operation_packet

        packet, _ = operation_packet(
            runtime.root, "probe/" + run_id, state_path=runtime.journal.path
        )
        frozen = packet["files"]
        if any(
            frozen.get(name) != body for name, body in output["files"].items()
            if name != "hypothesis.json"
        ):
            raise ResumeMismatchError("После начала probe исходники менять нельзя")
        if frozen["hypothesis.json"] == original:
            return output
        for number in range(1, runtime.profile.get("max_role_attempts", 4) + 1):
            operation = "source-format/" + run_id + f"/{number}"
            receipt_path = runtime.root / "source-corrections" / (digest(operation.encode()) + ".json")
            if not receipt_path.exists():
                continue
            receipt = json.loads(read_regular(receipt_path))
            if (
                receipt.get("accepted_shape") is True
                and receipt.get("original_sha256") == digest(original.encode())
                and receipt.get("corrected_sha256") == digest(frozen["hypothesis.json"].encode())
                and receipt.get("program_sha256") == digest(frozen["experiment/experiment.py"].encode())
            ):
                return dict(
                    output, files=dict(output["files"], **{"hypothesis.json": frozen["hypothesis.json"]}),
                    _source_correction=receipt,
                )
        raise ResumeMismatchError("Нет receipt сохранённой preregistration")
    if valid(original):
        return output
    current = original
    for number in range(1, runtime.profile.get("max_role_attempts", 4) + 1):
        operation = "source-format/" + run_id + f"/{number}"
        repair_files = {
                "experiment/experiment.py": output["files"]["experiment/experiment.py"],
                "hypothesis.json": current,
                "report.md": output["files"]["report.md"],
                "registered-definitions.json": json_bytes([
                    {"field": field, "id": name, "definition": definition}
                    for (field, name), definition in sorted(definitions.items())
                ]).decode(),
            }
        repair_request = {
                "kind": "agent", "role": "proposer", "operation_id": operation,
                "mission": (
                    "Исправь только формат hypothesis.json уже подготовленного опыта. "
                    "Программу и план не меняй, расчёты не запускай, гипотезу не заменяй. "
                    "В answer верни JSON-документ ровно с шестью полями: hypothesis, "
                    "falsification, baseline, scope — непустые строки; configurations "
                    "и feature_sets — непустые списки объектов ровно {id, definition}. "
                    "id соответствует [a-z][a-z0-9_-]{0,63}; definition — объект полных "
                    "параметров фактически сохранённой программы, а не новые параметры. "
                    "Идентификаторы configurations должны совпасть с configuration_ids "
                    "программы. Дай определения на основе предоставленного кода. "
                    "Не включай stage, rationale, declared_configurations, "
                    "falsifiable_hypothesis или другие лишние поля. Перенеси имеющуюся "
                    "фальсификацию в falsification, не смягчая её. Все тексты по-русски. "
                    "Внешний ответ — {answer: документ, files: {}}."
                    " Если идентификатор уже есть в registered-definitions.json, "
                    "сначала сверь его полное определение с сохранённой программой. "
                    "Только при соответствии фактических параметров используй это "
                    "определение без изменений. Если параметры не соответствуют, "
                    "верни в answer объяснение блокера вместо исправленного JSON. "
                    "Для исходного документа с шестью корректными полями разрешено "
                    "исправлять только конфликтующие определения существующих id; "
                    "гипотезу, фальсификацию, baseline, scope, состав списков, "
                    "идентификаторы и неконфликтующие определения сохрани точно."
                ),
            }
        if operation in runtime.journal.data["operations"]:
            from .audit import operation_packet

            packet, _ = operation_packet(
                runtime.root, operation, state_path=runtime.journal.path
            )
            if any(
                packet["files"].get(name) != repair_files[name]
                for name in ("experiment/experiment.py", "hypothesis.json", "report.md")
            ):
                raise ResumeMismatchError("Изменились исходники шага исправления формата")
            repair_files, repair_request = packet["files"], packet["request"]
        corrected = runtime.execute(repair_files, repair_request, model_role="proposer")
        current = corrected["answer"]
        accepted = valid(current)
        if accepted and original_data is not None:
            expected = json.loads(original)
            for field in ("configurations", "feature_sets"):
                for item in expected[field]:
                    if (field, item["id"]) in definitions:
                        item["definition"] = definitions[(field, item["id"])]
            accepted = json.loads(current) == expected
        receipt = {
            "schema": "craving-source-format-repair/v1", "operation": operation,
            "source_operation": "solver/" + run_id,
            "original_sha256": digest(original.encode()),
            "corrected_sha256": digest(current.encode()),
            "program_sha256": digest(output["files"]["experiment/experiment.py"].encode()),
            "accepted_shape": accepted,
            "scientific_acceptance": "not_granted",
        }
        receipt_path = runtime.root / "source-corrections" / (digest(operation.encode()) + ".json")
        if receipt_path.exists():
            if json.loads(read_regular(receipt_path)) != receipt:
                raise ResumeMismatchError("Изменилось сохранённое решение о формате")
        else:
            atomic_json(receipt_path, receipt)
        if accepted:
            return dict(
                output, files=dict(output["files"], **{"hypothesis.json": current}),
                _source_correction=receipt,
            )
    raise BudgetExhaustedError("Исчерпаны адресные исправления формата hypothesis.json: " + run_id)


def saved_response_draft(root, operation, attempt):
    """Незавершённый публичный ответ — контекст исправления, не completed result."""
    ref = attempt.get("workspace_archive")
    if not ref:
        return None
    relative(ref["path"])
    archive_path = Path(root) / ref["path"]
    if digest_file(archive_path) != ref["sha256"]:
        raise ResumeMismatchError("Изменился архив незавершённой попытки")
    archive = json.loads(read_regular(archive_path, 4 * 1024 * 1024))
    stream = (
        Path(root) / "cli-workspaces"
        / f"{digest(operation.encode())}-{attempt['attempt']}" / "stream-stdout.jsonl"
    )
    info = archive.get("files", {}).get("stream-stdout.jsonl", {})
    if not info.get("sha256") or digest_file(stream) != info["sha256"]:
        return None
    step, chunks, size = None, [], 0
    with stream.open() as source:
        for line in source:
            try:
                event = json.loads(line).get("step_update", {})
            except ValueError:
                continue
            if event.get("step_type") != "agent_response":
                continue
            if event.get("step_index") != step:
                step, chunks, size = event.get("step_index"), [], 0
            delta = event.get("text_delta")
            if isinstance(delta, str) and size < 128 * 1024:
                piece = delta[:128 * 1024 - size]
                chunks.append(piece)
                size += len(piece)
    return "".join(chunks) or None
