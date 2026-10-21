"""Адресные регрессии продолжения закреплённого раунда без повторных расчётов."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from research_harness import autonomy
from research_harness.autonomy import Supervisor
from research_harness.contracts import PACKAGE, digest, json_bytes
from research_harness.queue import run_round
from research_harness.runtime import Runtime, profile_from
from research_harness.state import BudgetExhaustedError, ResumeMismatchError, atomic_json
from research_harness.subscription import QuotaExhaustedError, parse_quota_error, quota_reset_delay


@pytest.fixture
def контроллер(tmp_path, monkeypatch):
    """Настоящий контроллер и журнал; модель и OCI заменены инженерными данными."""
    profile_path = tmp_path / "profile.json"
    atomic_json(profile_path, dict(profile_from(PACKAGE / "run-profile.json"), nonpaid_fixture=True))
    args = SimpleNamespace(
        repo=PACKAGE.parent,
        upstream=PACKAGE.parent / ".research/vendor/harness2",
        profile=profile_path,
        queue=PACKAGE / "research-queue.example.json",
        output=tmp_path / "campaign",
        worker_args=[],
    )
    clock = [1000.0]
    monkeypatch.setattr("research_harness.state.time.time", lambda: clock[0])
    monkeypatch.setattr(Runtime, "prepare", lambda *args, **kwargs: None)
    supervisor = Supervisor(args, clock=lambda: clock[0], rng=lambda a, b: 1)
    supervisor.revise()
    supervisor.state["phase"] = "research"
    yield supervisor, clock
    supervisor.writer_lock.close()


@pytest.fixture
def закреплённый_раунд(контроллер):
    supervisor, clock = контроллер
    root = supervisor.round_root()
    root.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(supervisor.profile, root, supervisor.manifest)
    try:
        runtime.execute({}, {"kind": "agent", "operation_id": "fixture/anchor"}, model_role="selector")
    finally:
        runtime.finish()
    relative = str(root.relative_to(supervisor.root))
    supervisor.authorize_recovery(relative, hold_round=True)
    return supervisor, clock, relative


def готовые_операции(root):
    journal = json.loads((root / "run-state.json").read_text())
    return {
        key: (record["attempts"], record["result_sha256"], record["fingerprint"])
        for key, record in journal["operations"].items()
        if record["status"] == "completed"
    }


def следующая_редакция(supervisor):
    """Имитация нового отпечатка операционного модуля без правки живых исходников."""
    manifest = json.loads(json_bytes(supervisor.manifest))
    manifest["controller_sources"]["research_harness/autonomy.py"] = digest(b"fixture revision 2")
    return manifest, digest(json_bytes(manifest))


@pytest.mark.parametrize("сбой", ["ошибка", "квота"])
@pytest.mark.parametrize("шаг", ["solver/inv1", "probe/inv1", "campaign/report"])
def test_восстановление_повторяет_только_незавершённый_шаг(
    контроллер, monkeypatch, сбой, шаг
):
    supervisor, clock = контроллер
    root = supervisor.round_root()
    relative = str(root.relative_to(supervisor.root))
    original = Runtime._execute_once
    calls = []
    failed = []

    def transport(self, files, request, *, model_role, wall):
        operation = request["operation_id"]
        if not calls:
            # Harness уже зафиксировал настоящий пакет первого шага.
            supervisor.authorize_recovery(relative, hold_round=True)
        calls.append(operation)
        matches = operation == шаг or (
            operation.startswith(шаг.split("/")[0] + "/")
            and operation.endswith("/" + шаг.split("/")[-1])
        )
        if matches and not failed:
            failed.append(operation)
            if сбой == "квота":
                raise QuotaExhaustedError("Исчерпана квота инженерного примера", 120)
            raise ValueError("Временная ошибка инженерного примера")
        if model_role is None:
            return {
                "execution_complete": True,
                "files": {
                    "artifacts/metrics.json": '{"configuration_ids":["fixture"]}',
                    "artifacts/observations.md": "Инженерная проверка, научного результата нет",
                },
                "errors": [],
            }
        return original(self, files, request, model_role=model_role, wall=wall)

    monkeypatch.setattr(Runtime, "_execute_once", transport)
    task = {
        "id": "H2", "autonomous": True, "mission": "Инженерная проверка восстановления",
        "max_configurations": 2, "max_feature_sets": 2,
        "experiment_outputs": ["artifacts/metrics.json", "artifacts/observations.md"],
    }

    def research():
        result = run_round(root, task, {}, supervisor.profile, supervisor.manifest)
        assert result["execution_complete"]
        supervisor.state["phase"] = "replay"

    monkeypatch.setattr(supervisor, "research", research)
    assert supervisor.tick()
    assert failed
    assert supervisor.state["status"] == ("waiting_quota" if сбой == "квота" else "repair_pending")
    assert supervisor.state["phase"] == "research"
    ready = готовые_операции(root)
    assert any(operation.startswith("probe/") for operation in ready)
    completed_calls = {operation: calls.count(operation) for operation in ready}
    before = list(calls)
    retry_at = supervisor.state["next_retry_at"]
    supervisor.authorize_recovery(relative, hold_round=True)
    if сбой == "квота":
        assert supervisor.state["next_retry_at"] == retry_at
        assert supervisor.tick()
        assert calls == before
    clock[0] = max(clock[0], retry_at) + 1
    assert supervisor.tick()
    assert supervisor.state["phase"] == "replay"
    assert supervisor.round_root() == root
    assert list(supervisor.root.glob("epochs/*/preregistration.json")) == [supervisor.epoch() / "preregistration.json"]
    assert list(supervisor.root.glob("epochs/*/tasks/*/round-*")) == [root]
    after = готовые_операции(root)
    assert {operation: after[operation] for operation in ready} == ready
    assert {operation: calls.count(operation) for operation in ready} == completed_calls
    assert Counter(operation for operation in calls if operation.startswith("probe/")).most_common(1)[0][1] == (2 if шаг == "probe/inv1" else 1)
    assert supervisor.state["admissions"] == {}


def test_перезапуск_в_ожидании_квоты_сохраняет_срок_и_готовые_шаги(закреплённый_раунд, monkeypatch):
    supervisor, clock, relative = закреплённый_раунд
    root = supervisor.round_root()
    before = готовые_операции(root)
    supervisor.failed(QuotaExhaustedError("Квота", 900))
    supervisor.state["status"] = "interrupted"
    supervisor.save()
    supervisor.writer_lock.close()
    resumed = Supervisor(supervisor.args, clock=lambda: clock[0], rng=lambda a, b: 1)
    try:
        resumed.authorize_recovery(relative, hold_round=True)
        assert resumed.state["status"] == "waiting_quota"
        assert resumed.state["next_retry_at"] == 1900
        monkeypatch.setattr(resumed, "research", lambda: pytest.fail("Ранний запуск до восстановления квоты"))
        assert resumed.tick()
        assert resumed.round_root() == root
        assert готовые_операции(root) == before
    finally:
        resumed.writer_lock.close()


def test_исчерпание_ремонтов_блокирует_текущий_раунд_без_новой_эпохи(закреплённый_раунд):
    supervisor, _, _ = закреплённый_раунд
    root = supervisor.round_root()
    before = готовые_операции(root)
    for _ in range(supervisor.profile["max_role_attempts"]):
        supervisor.failed(autonomy.MicrostepInterrupted("Повторяющаяся ошибка"))
    assert supervisor.state["status"] == "blocked"
    assert supervisor.tick() is False
    assert supervisor.round_root() == root
    assert готовые_операции(root) == before
    assert len(list(supervisor.root.glob("epochs/*"))) == 1


def test_следующая_попытка_заблокирована_даже_при_исчерпанном_пакете(закреплённый_раунд):
    supervisor, _, _ = закреплённый_раунд
    root = supervisor.round_root()
    supervisor.goal["tasks"][0]["max_rounds"] = 1
    supervisor.next_attempt("Пакет исчерпан")
    assert supervisor.state["status"] == "blocked"
    assert supervisor.state["next_retry_at"] == 0
    assert supervisor.round_root() == root
    assert len(list(supervisor.root.glob("epochs/*"))) == 1
    assert supervisor.state["admissions"] == {}


def test_прямой_переход_к_новой_эпохе_не_меняет_состояние(закреплённый_раунд):
    supervisor, _, _ = закреплённый_раунд
    before = json.dumps(supervisor.state, sort_keys=True)
    with pytest.raises(ResumeMismatchError, match="Новая эпоха запрещена"):
        supervisor.new_epoch("Непредусмотренный переход")
    assert json.dumps(supervisor.state, sort_keys=True) == before
    assert len(list(supervisor.root.glob("epochs/*"))) == 1


def test_восстановление_другого_раунда_запрещено(закреплённый_раунд):
    supervisor, _, _ = закреплённый_раунд
    with pytest.raises(ValueError, match="только закреплённого"):
        supervisor.authorize_recovery("epochs/000002/tasks/H2/round-000001")
    assert len(list(supervisor.root.glob("epochs/*"))) == 1


@pytest.mark.parametrize("повреждение", ["удаление", "изменение", "другой_раунд", "каталог", "кодировка"])
def test_повреждение_фиксации_останавливает_исполнение(закреплённый_раунд, monkeypatch, повреждение):
    supervisor, _, _ = закреплённый_раунд
    path = supervisor.root / "continuation-guard.json"
    if повреждение == "удаление":
        path.unlink()
    elif повреждение == "изменение":
        path.write_text("{}")
    elif повреждение == "каталог":
        path.unlink()
        path.mkdir()
    elif повреждение == "кодировка":
        path.write_bytes(b"\xff")
    else:
        supervisor.state["round"] += 1
    monkeypatch.setattr(supervisor, "research", lambda: pytest.fail("Запуск после повреждения фиксации"))
    assert supervisor.tick() is False
    assert supervisor.state["status"] == "blocked"
    assert supervisor.state["last_failure"]["type"] == "ResumeMismatchError"
    assert len(list(supervisor.root.glob("epochs/*"))) == 1


def test_смена_редакции_сохраняет_квоту_до_явного_восстановления(закреплённый_раунд, monkeypatch):
    supervisor, _, relative = закреплённый_раунд
    supervisor.failed(QuotaExhaustedError("Квота", 900))

    class Перезапуск(BaseException):
        pass

    def restart(*_):
        raise Перезапуск()

    manifest, revision = следующая_редакция(supervisor)
    monkeypatch.setattr(autonomy, "source_revision", lambda *args: (manifest, revision))
    monkeypatch.setattr(autonomy.os, "execv", restart)
    with pytest.raises(Перезапуск):
        supervisor.tick()
    assert supervisor.state["next_retry_at"] == 1900
    supervisor.manifest, supervisor.revision = manifest, revision
    supervisor.revise()
    assert supervisor.state["status"] == "blocked"
    supervisor.authorize_recovery(relative, hold_round=True)
    assert supervisor.state["status"] == "waiting_quota"
    assert supervisor.state["next_retry_at"] == 1900
    assert supervisor.state["phase"] == "research"


@pytest.mark.parametrize("фаза", ["replay", "critic"])
def test_совместимое_восстановление_после_готового_исследования(закреплённый_раунд, monkeypatch, фаза):
    from test_autonomous_supervisor import make_round

    supervisor, _, relative = закреплённый_раунд
    root = supervisor.round_root()
    make_round(root, {}, {}, supervisor.profile, supervisor.manifest)
    completed = (root / "round-complete.json").read_bytes()
    supervisor.state["phase"] = фаза
    supervisor.failed(QuotaExhaustedError("Квота поздней фазы", 900))
    supervisor.manifest, supervisor.revision = следующая_редакция(supervisor)
    supervisor.revise()
    supervisor.authorize_recovery(relative, hold_round=True)
    assert supervisor.state["phase"] == фаза
    assert supervisor.state["status"] == "waiting_quota"
    assert supervisor.state["next_retry_at"] == 1900
    assert (root / "round-complete.json").read_bytes() == completed
    assert supervisor.round_root() == root


def test_критик_после_смены_редакции_использует_свой_готовый_ответ(закреплённый_раунд):
    from test_autonomous_supervisor import make_round

    supervisor, clock, relative = закреплённый_раунд
    root = supervisor.round_root()
    make_round(root, {}, {}, supervisor.profile, supervisor.manifest)
    critic = root / "critics/000001"
    critic.mkdir(parents=True)
    request = {"kind": "agent", "operation_id": "critic/ready"}
    pending = {"kind": "agent", "operation_id": "critic/pending"}
    first = Runtime(supervisor.profile, critic, supervisor.manifest, phase="critic")
    first._execute_once = lambda *args, **kwargs: {"answer": "Сохранённый ответ"}
    first.execute({}, request, model_role="selector")

    def quota(*args, **kwargs):
        raise QuotaExhaustedError("Квота критика", 900)

    first._execute_once = quota
    with pytest.raises(QuotaExhaustedError):
        first.execute({}, pending, model_role="selector")
    first.finish("paused_quota")
    supervisor.state["phase"] = "critic"
    supervisor.failed(QuotaExhaustedError("Квота критика", 900))
    supervisor.manifest, supervisor.revision = следующая_редакция(supervisor)
    supervisor.revise()
    supervisor.authorize_recovery(relative, hold_round=True)
    resumed = Runtime(supervisor.profile, critic, supervisor.manifest, phase="critic")
    calls = []

    def transport(files, request, **kwargs):
        calls.append(request["operation_id"])
        return {"answer": "Продолженный ответ"}

    resumed._execute_once = transport
    try:
        assert resumed.execute({}, request, model_role="selector") == {"answer": "Сохранённый ответ"}
        with pytest.raises(QuotaExhaustedError):
            resumed.execute({}, pending, model_role="selector")
        assert calls == []
        resumed.finish("paused_quota")
        clock[0] = 1901
        resumed = Runtime(supervisor.profile, critic, supervisor.manifest, phase="critic")
        resumed._execute_once = transport
        resumed.execute({}, pending, model_role="selector")
        assert calls == ["critic/pending"]
        assert resumed.journal.data["operations"]["critic/ready"]["attempts"] == 1
    finally:
        resumed.finish()


def test_готовый_worker_result_восстанавливается_без_повтора_расчёта(закреплённый_раунд):
    supervisor, _, _ = закреплённый_раунд
    root = supervisor.round_root()
    request = {"kind": "experiment", "operation_id": "probe/publication"}
    expected = {"execution_complete": True, "files": {"result": "5"}, "errors": []}
    first = Runtime(supervisor.profile, root, supervisor.manifest)

    def interrupted(files, request, **kwargs):
        record = first.journal.data["operations"][request["operation_id"]]
        atomic_json(root / "worker-results" / (digest(request["operation_id"].encode()) + "-1.json"), {
            "fingerprint": record["fingerprint"], "returncode": 0,
            "stdout": json.dumps(dict(expected, _kernel_limits={})),
        })
        raise OSError("Сбой публикации уже сохранённого ответа")

    first._execute_once = interrupted
    with pytest.raises(autonomy.MicrostepInterrupted):
        first.execute({}, request)
    first.finish("interrupted")
    resumed = Runtime(supervisor.profile, root, supervisor.manifest)
    resumed._execute_once = lambda *args, **kwargs: pytest.fail("Готовый расчёт запущен повторно")
    try:
        assert resumed.execute({}, request) == expected
        assert resumed.journal.data["operations"][request["operation_id"]]["attempts"] == 1
        assert list(root.glob("recovered-results/*.json"))
    finally:
        resumed.finish()


def test_ошибочный_probe_имеет_конечный_бюджет_повторов(закреплённый_раунд):
    supervisor, _, _ = закреплённый_раунд
    root = supervisor.round_root()
    request = {"kind": "experiment", "operation_id": "probe/failure"}
    calls = []

    def failed(*args, **kwargs):
        calls.append("probe")
        raise OSError("Временная ошибка вычислительной среды")

    limit = supervisor.profile["max_role_attempts"]
    for attempt in range(limit + 1):
        runtime = Runtime(supervisor.profile, root, supervisor.manifest)
        runtime._execute_once = failed
        try:
            expected = autonomy.MicrostepInterrupted if attempt < limit else BudgetExhaustedError
            with pytest.raises(expected):
                runtime.execute({}, request)
        finally:
            runtime.finish("interrupted")
    assert len(calls) == limit
    assert len(list(supervisor.root.glob("epochs/*"))) == 1


def test_неопределённый_исход_расчёта_не_запускается_повторно(закреплённый_раунд):
    supervisor, _, _ = закреплённый_раунд
    root = supervisor.round_root()
    request = {"kind": "experiment", "operation_id": "probe/unknown"}
    runtime = Runtime(supervisor.profile, root, supervisor.manifest)
    fingerprint = digest(json_bytes({"files": {}, "request": request, "manifest": runtime.manifest, "profile": runtime.profile, "role": None}))
    runtime.journal.begin(request["operation_id"], fingerprint, "experiment")
    runtime.journal.close()
    resumed = Runtime(supervisor.profile, root, supervisor.manifest)
    resumed._execute_once = lambda *args, **kwargs: pytest.fail("Неизвестный исход повторён без разбора")
    try:
        with pytest.raises(ResumeMismatchError):
            resumed.execute({}, request)
    finally:
        resumed.finish("interrupted")


@pytest.mark.parametrize("срок,секунды", [("145h2m4s", 522124), ("2d3h4m5s", 183845), ("4m12s", 252), ("45s", 45)])
def test_срок_сброса_квоты_сервиса_распознаётся(срок, секунды):
    text = "Individual quota reached. Resets in " + срок + "."
    assert parse_quota_error(text) == (True, секунды)
    assert quota_reset_delay(text) == секунды


def test_отсутствующий_срок_квоты_не_выдаётся_за_известный():
    assert quota_reset_delay("RESOURCE_EXHAUSTED") is None
    assert parse_quota_error("RESOURCE_EXHAUSTED") == (True, 30)


def test_старое_ожидание_исправляется_по_сохранённому_сообщению_сервиса(закреплённый_раунд):
    supervisor, _, relative = закреплённый_раунд
    root = supervisor.round_root()
    request = {"kind": "agent", "operation_id": "quota/pending"}
    runtime = Runtime(supervisor.profile, root, supervisor.manifest)

    def quota(*args, **kwargs):
        raise QuotaExhaustedError("Старый нераспознанный формат", 30, retry_known=False)

    runtime._execute_once = quota
    with pytest.raises(QuotaExhaustedError):
        runtime.execute({}, request, model_role="selector")
    failure = {"stderr_sanitized": "Individual quota reached. Resets in 145h2m4s."}
    atomic_json(root / "failed-attempts/quota.json", failure)
    record = runtime.journal.data["operations"][request["operation_id"]]
    record["attempt_history"][-1]["public_failure"] = {"path": "failed-attempts/quota.json", "sha256": digest(json_bytes(failure))}
    runtime.finish("paused_quota")
    before = (root / "run-state.json").read_bytes()
    supervisor.failed(QuotaExhaustedError("Старый срок", 30, retry_known=False))
    supervisor.authorize_recovery(relative, hold_round=True)
    assert supervisor.state["status"] == "waiting_quota"
    assert supervisor.state["next_retry_at"] == 523124
    assert (root / "run-state.json").read_bytes() == before
