"""Регрессии автономного перехода, квоты и неизменяемых доказательств."""

import json
from types import SimpleNamespace

import pytest

from research_harness import autonomy
from research_harness.autonomy import Supervisor, active
from research_harness.contracts import PACKAGE, digest
from research_harness.critic import machine_assessment, validate_review
from research_harness.runtime import Runtime, profile_from
from research_harness.state import Journal, atomic_json
from research_harness.subscription import (
    EnvironmentUnavailableError,
    QuotaExhaustedError,
    parse_quota_error,
)


@pytest.fixture
def supervisor(tmp_path):
    repo = PACKAGE.parent
    args = SimpleNamespace(
        repo=repo,
        upstream=repo / ".research/vendor/harness2",
        profile=PACKAGE / "run-profile.json",
        queue=PACKAGE / "research-queue.example.json",
        output=tmp_path / "campaign",
        worker_args=[],
    )
    clock = [1000.0]
    instance = Supervisor(args, clock=lambda: clock[0], rng=lambda a, b: 1)
    instance.test_clock = clock
    yield instance
    instance.writer_lock.close()


def test_двойной_старт_не_пишет_состояние(supervisor):
    before = supervisor.path.read_bytes()
    assert active(supervisor.campaign)
    with pytest.raises(BlockingIOError):
        Supervisor(supervisor.args)
    assert supervisor.path.read_bytes() == before


def test_квота_дольше_лимита_попыток_сохраняет_историю(tmp_path):
    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        autonomous_quota_wait=True,
        max_calls=1,
        max_role_attempts=1,
        wall_seconds=1,
        call_timeout=1,
    )
    journal = Journal(tmp_path, {}, profile)
    try:
        for _ in range(8):
            journal.begin("source", "fingerprint", "solver")
            journal.end(quota=QuotaExhaustedError("quota", retry_after=1))
        journal.begin("source", "fingerprint", "solver")
        journal.end({"answer": "ready"})
        assert journal.data["attempts"] == 9
        from research_harness.state import attempt_history

        assert (
            len(list(attempt_history(tmp_path, journal.data["operations"]["source"])))
            == 9
        )
        assert len(journal.data["operations"]["source"]["attempt_history"]) == 2
        assert journal.cached("source", "fingerprint") == {"answer": "ready"}
    finally:
        journal.close()


def test_старый_исчерпанный_quota_journal_возобновляется(tmp_path):
    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        autonomous_quota_wait=True,
        max_calls=1,
        max_role_attempts=1,
    )
    journal = Journal(tmp_path, {}, profile)
    journal.begin("source", "fp", "solver")
    journal.end(quota=QuotaExhaustedError("quota", 1))
    journal.close()
    resumed = Journal(tmp_path, {}, profile)
    try:
        resumed.begin("source", "fp", "solver")
        resumed.end({"answer": "completed"})
        assert resumed.data["attempts"] == 2
    finally:
        resumed.close()


def test_ожидание_квоты_неограниченно_и_stop_немедленный(supervisor):
    supervisor.revise()
    supervisor.state["phase"] = "critic"
    for _ in range(25):
        supervisor.failed(QuotaExhaustedError("quota", retry_after=90000))
        assert supervisor.state["status"] == "waiting_quota"
        assert supervisor.state["next_retry_at"] == supervisor.test_clock[0] + 90000
        assert supervisor.state["phase"] == "critic"
        supervisor.test_clock[0] += 90001
    atomic_json(supervisor.stop, {"stop": True})
    assert supervisor.tick() is False
    assert supervisor.state["status"] == "interrupted"


def test_auth_не_квота_и_редко_повторяется(supervisor):
    supervisor.failed(EnvironmentUnavailableError("revoked"))
    assert supervisor.state["status"] == "waiting_environment"
    assert supervisor.state["next_retry_at"] >= 1300
    assert supervisor.state["quota_waits"] == 0
    assert (
        parse_quota_error("UNAUTHENTICATED invalid_grant RESOURCE_EXHAUSTED")[0]
        is False
    )


def test_retry_at_и_retry_delay(monkeypatch):
    monkeypatch.setattr("research_harness.subscription.time.time", lambda: 1000)
    assert parse_quota_error('{"code":429,"retryAt":2000}') == (True, 1000)
    assert parse_quota_error('{"code":429,"retryDelay":"4.2s"}') == (True, 5)


def test_неизвестный_срок_квоты_backoff_ограничен(supervisor):
    for _ in range(30):
        supervisor.failed(QuotaExhaustedError("unknown"))
    assert supervisor.state["next_retry_at"] - supervisor.test_clock[0] == 3600


def test_ошибка_кода_создаёт_новую_попытку_и_диагностику(supervisor):
    supervisor.revise()
    old = supervisor.round_root()
    old.mkdir(parents=True, exist_ok=True)
    (old / "raw-failure.txt").write_text("old failure")
    supervisor.state["phase"] = "research"
    supervisor.failed(ValueError("invalid program"))
    assert supervisor.state["round"] == 2
    assert supervisor.state["failed_round"] == str(old.relative_to(supervisor.root))
    assert (old / "raw-failure.txt").read_text() == "old failure"


def test_некорректный_кэш_критика_не_повторяется_с_тем_же_id(supervisor):
    supervisor.revise()
    supervisor.state.update(phase="critic", critic_attempt=1)
    supervisor.failed(ValueError("Invalid critic response"))
    assert supervisor.state["critic_attempt"] == 2
    assert supervisor.state["phase"] == "critic"
    for _ in range(3):
        supervisor.failed(ValueError("Invalid critic response"))
    assert supervisor.state["phase"] == "research"
    assert supervisor.state["round"] == 2


def test_новая_редакция_сбрасывает_semantic_admission(supervisor):
    supervisor.revise()
    old = supervisor.state["revision"]
    supervisor.state["admissions"] = {"H2": {"path": "old", "sha256": "old"}}
    supervisor.revision = "b" * 64
    supervisor.revise()
    assert supervisor.state["revision"] != old
    assert supervisor.state["admissions"] == {}
    assert supervisor.state["phase"] == "qualification"
    assert (supervisor.root / "revisions" / old / "manifest.json").exists()


def test_эпохи_не_сбрасывают_cumulative_экспозицию(supervisor):
    supervisor.revise()
    declaration = {
        "hypothesis": {
            "configurations": [{"id": "a", "definition": {"v": 1}}],
            "feature_sets": [],
        },
        "admitted": True,
    }
    atomic_json(supervisor.epoch() / "declarations.json", {"a": declaration})
    supervisor.exposure()
    supervisor.new_epoch("bounded packet")
    atomic_json(supervisor.epoch() / "declarations.json", {"a": declaration})
    supervisor.exposure()
    assert supervisor.state["cumulative"]["declared_configurations"] == 2
    assert supervisor.state["cumulative"]["packets"] == 2
    assert len(list(supervisor.root.glob("epochs/*/preregistration.json"))) == 2


def test_контекст_ограничен_при_полной_истории(supervisor):
    for index in range(40):
        supervisor.event("history", value=index)
    context = json.loads(supervisor.context())
    assert len(context["recent"]) == 8
    assert len(list((supervisor.root / "events").glob("*.json"))) == 40
    assert context["ledger_head"] == supervisor.state["head"]


def test_критик_не_может_подменить_измеренное_значение():
    files = {"probe.json": json.dumps({"sum": 5})}
    ref = {
        "path": "probe.json",
        "sha256": digest(files["probe.json"].encode()),
        "pointer": "/sum",
        "value": 5,
    }
    review = {
        "outcome": "supported",
        "criteria": [
            {
                "id": "c1",
                "verdict": "supported",
                "reason": "measured",
                "evidence": [ref],
            }
        ],
        "next_experiment": "",
    }
    assert validate_review(review, {"c1": "sum equals 5"}, files) == review
    ref["value"] = 6
    with pytest.raises(ValueError, match="does not match"):
        validate_review(review, {"c1": "sum equals 5"}, files)


@pytest.mark.parametrize(
    "field,value",
    [
        ("delta_macro_brier", 0),
        ("ci_upper", 0),
        ("checkpoint_max_bytes", 16385),
        ("coverage_days", 364),
        ("pending_feedback_drained", False),
    ],
)
def test_h4_supported_требует_численного_доказательства(field, value):
    values = {
        "delta_macro_brier": -0.003,
        "ci_upper": -0.001,
        "checkpoint_max_bytes": 15000,
        "coverage_days": 365,
        "pending_feedback_drained": True,
    }
    values[field] = value
    probe = {
        "files": {"artifacts/metrics.json": json.dumps({"autonomous_evidence": values})}
    }
    with pytest.raises(ValueError):
        machine_assessment(probe, "H4", "supported")


def test_h4_все_численные_критерии_проходят():
    values = {
        "delta_macro_brier": -0.003,
        "ci_upper": -0.001,
        "checkpoint_max_bytes": 15000,
        "coverage_days": 365,
        "pending_feedback_drained": True,
    }
    probe = {
        "files": {"artifacts/metrics.json": json.dumps({"autonomous_evidence": values})}
    }
    assert all(
        v["passes"]
        for v in machine_assessment(probe, "H4", "supported")[
            "numeric_criteria"
        ].values()
    )


def make_round(root, task, sources, profile, manifest, prior=None):
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "round-complete.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    program = "print(5)"
    probe = {
        "execution_complete": True,
        "files": {
            "artifacts/metrics.json": json.dumps(
                {"configuration_ids": ["fixture"], "sum": 5}
            )
        },
        "errors": [],
    }
    runtime = Runtime(profile, root, manifest)
    runtime._execute_once = lambda *a, **k: probe
    runtime.execute(
        {
            "experiment/experiment.py": program,
            "hypothesis.json": "{}",
            "report.md": "fixture",
        },
        {
            "kind": "experiment",
            "operation_id": "probe/fixture",
            "outputs": ["artifacts/metrics.json"],
            "configuration_ids": ["fixture"],
        },
    )
    runtime.finish()
    contents = {
        "probe.json": json.dumps(probe),
        "chosen/experiment/experiment.py": program,
        "chosen/hypothesis.json": "{}",
        "chosen/report.md": "fixture",
        "rollouts/fixture/receipt.json": json.dumps({"run_id": "fixture"}),
        "rollouts/fixture/probe.json": json.dumps(probe),
        "rollouts/fixture/hypothesis.json": "{}",
        "selection.json": json.dumps(
            {
                "rollout": "rollouts/fixture",
                "operation": "probe/fixture",
                "receipt_sha256": digest(json.dumps({"run_id": "fixture"}).encode()),
            }
        ),
        "hypothesis.json": "{}",
        "claims.json": "[]",
        "research-report.md": "Engineering fixture only",
    }
    for name, body in contents.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    receipt = {
        "schema": "craving-research-round/v1",
        "execution_complete": True,
        "files": {n: digest(b.encode()) for n, b in contents.items()},
        "usage": {"closed": True},
    }
    atomic_json(receipt_path, receipt)
    return receipt


def setup_controller(supervisor, monkeypatch):
    supervisor.revise()
    supervisor.state.update(phase="research")
    supervisor.goal["tasks"] = supervisor.goal["tasks"][:1]
    monkeypatch.setattr(autonomy, "run_round", make_round)
    # Эти тесты проверяют orchestration; научный контракт покрыт отдельными тестами.
    monkeypatch.setattr(
        autonomy, "machine_assessment", lambda *args: {"fixture_only": True}
    )

    def fake_replay(root, op, output):
        _, record = autonomy.operation_packet(root, op)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / "replay-report.json",
            {
                "schema": "craving-research-replay/v1",
                "original_operation": op,
                "original_input_sha256": record["inputs_sha256"],
                "original_result_sha256": record["result_sha256"],
                "replayed_result_sha256": record["result_sha256"],
                "exact_output_match": True,
            },
        )

    monkeypatch.setattr(autonomy, "replay", fake_replay)
    return supervisor


def review_for(files, criteria, outcome="supported"):
    ref = {
        "path": "probe.json",
        "sha256": digest(files["probe.json"].encode()),
        "pointer": "/execution_complete",
        "value": True,
    }
    return {
        "outcome": outcome,
        "criteria": [
            {
                "id": c,
                "verdict": outcome,
                "reason": "fixture evidence",
                "evidence": [ref],
            }
            for c in criteria
        ],
        "next_experiment": "Repair missing evidence",
    }


def test_квота_критика_не_повторяет_завершённое_исследование(supervisor, monkeypatch):
    s = setup_controller(supervisor, monkeypatch)
    s.research()
    s.repeat()
    root = s.round_root()
    before = (root / "round-complete.json").read_bytes()

    def quota(*_):
        raise QuotaExhaustedError("critic quota", 123)

    monkeypatch.setattr(autonomy, "assess", quota)
    assert s.tick()
    assert s.state["phase"] == "critic"
    assert s.state["status"] == "waiting_quota"
    s.test_clock[0] += 124

    def critic(root, files, criteria, *_):
        result = review_for(files, criteria)
        atomic_json(root / "review.json", result)
        return result

    monkeypatch.setattr(autonomy, "assess", critic)
    assert s.tick() is False
    assert s.state["status"] == "development_evidence_complete"
    assert (root / "round-complete.json").read_bytes() == before
    assert len(list(s.root.glob("epochs/*/tasks/*/round-*"))) == 1


def test_refuted_не_завершает_цель_и_не_допускает_зависимых(supervisor, monkeypatch):
    s = setup_controller(supervisor, monkeypatch)
    s.research()
    s.repeat()

    def critic(root, files, criteria, *_):
        result = review_for(files, criteria, "refuted")
        atomic_json(root / "review.json", result)
        return result

    monkeypatch.setattr(autonomy, "assess", critic)
    s.criticism()
    assert s.state["task_index"] == 0 and s.state["admissions"] == {}
    assert s.state["phase"] == "research" and s.state["round"] == 2
    assert not list(s.root.rglob("machine-admission.json"))


def test_crash_после_receipt_до_advance_не_дублирует_admission(supervisor, monkeypatch):
    s = setup_controller(supervisor, monkeypatch)
    s.research()
    s.repeat()

    def critic(root, files, criteria, *_):
        result = review_for(files, criteria)
        atomic_json(root / "review.json", result)
        return result

    monkeypatch.setattr(autonomy, "assess", critic)
    original = autonomy.immutable

    def crash(path, value):
        original(path, value)
        if path.name == "machine-admission.json":
            raise InterruptedError("crash after receipt")

    monkeypatch.setattr(autonomy, "immutable", crash)
    with pytest.raises(InterruptedError):
        s.criticism()
    before = next(s.root.rglob("machine-admission.json")).read_bytes()
    monkeypatch.setattr(autonomy, "immutable", original)
    s.criticism()
    assert len(list(s.root.rglob("machine-admission.json"))) == 1
    assert next(s.root.rglob("machine-admission.json")).read_bytes() == before


def test_предыдущий_harness_передаётся_и_фиксируется_через_квоту(
    supervisor, monkeypatch
):
    s = setup_controller(supervisor, monkeypatch)
    s.research()
    prior = s.state["latest_completed"]["H2"]
    s.next_attempt("new hypothesis")
    observed = []

    def paused(root, task, sources, profile, manifest, prior=None):
        observed.append(prior)
        raise QuotaExhaustedError("quota", 60)

    monkeypatch.setattr(autonomy, "run_round", paused)
    with pytest.raises(QuotaExhaustedError):
        s.research()
    s.state["latest_completed"]["H2"] = "changed-later-reference"
    with pytest.raises(QuotaExhaustedError):
        s.research()
    assert observed == [s.root / prior, s.root / prior]


def test_каждый_критерий_зафиксирован_отдельно(supervisor):
    for task in supervisor.goal["tasks"]:
        bullets = [
            line[2:].strip()
            for line in task["mission"].splitlines()
            if line.startswith("- ")
        ]
        assert list(supervisor.goal["criteria"][task["id"]].values()) == bullets
        assert len(bullets) > 1


def test_численные_критерии_не_получаются_из_вердикта_модели():
    with pytest.raises(ValueError, match="needs measured"):
        machine_assessment(
            {"files": {"artifacts/metrics.json": "{}"}}, "H4", "supported"
        )


def test_свежий_процесс_адаптера_не_получает_vertex_defaults():
    import os
    import subprocess
    import sys

    script = """import os
from research_harness.subscription import FORBIDDEN_ENV
before={k:os.environ.get(k) for k in FORBIDDEN_ENV}
from research_harness.research import ResearchAdapter
assert {k:os.environ.get(k) for k in FORBIDDEN_ENV}==before
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )


def test_первый_раунд_не_меняет_provider_environment_при_import_core():
    import os
    import subprocess
    import sys

    script = """import os,tempfile
from pathlib import Path
from research_harness import queue
from research_harness.subscription import FORBIDDEN_ENV
before={k:os.environ.get(k) for k in FORBIDDEN_ENV}
def boundary(*args,**kwargs):
    raise RuntimeError('stop before any execution')
queue.Runtime=boundary
with tempfile.TemporaryDirectory() as folder:
    try:queue.run_round(Path(folder)/'round-001',{'id':'H2'},{},{},{})
    except RuntimeError as error:assert str(error)=='stop before any execution'
assert {k:os.environ.get(k) for k in FORBIDDEN_ENV}==before
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "path", ["/errors/0", "/errors/-1", "/missing", "/execution_complete/child"]
)
def test_невалидный_pointer_становится_исправляемой_ошибкой(path):
    from research_harness.critic import pointer

    with pytest.raises(ValueError, match="does not resolve"):
        pointer({"errors": [], "execution_complete": True}, path)


def test_повреждённый_pointer_критика_переходит_на_новую_попытку(
    supervisor, monkeypatch
):
    s = setup_controller(supervisor, monkeypatch)
    s.research()
    s.repeat()

    def critic(root, files, criteria, *_):
        result = review_for(files, criteria)
        result["criteria"][0]["evidence"][0]["pointer"] = "/errors/0"
        return validate_review(result, criteria, files)

    monkeypatch.setattr(autonomy, "assess", critic)
    assert s.tick()
    assert s.state["phase"] == "critic" and s.state["critic_attempt"] == 2
    assert s.state["last_failure"]["type"] == "ValueError"


def test_большой_v2_quota_journal_мигрирует_без_потери_попыток(tmp_path):
    from research_harness.state import attempt_history

    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        autonomous_quota_wait=True,
        max_calls=1,
        max_role_attempts=1,
        wall_seconds=1,
        call_timeout=1,
    )
    initial = Journal(tmp_path, {}, profile)
    state = initial.data
    initial.close()
    state.pop("autonomous_attempt_ledger")
    attempts = [
        {
            "attempt": i,
            "status": "paused_quota",
            "active_seconds": 1,
            "reserved_seconds": 1,
            "started_at": i,
            "ended_at": i + 1,
            "error": {"message": "x" * 50000},
        }
        for i in range(1, 101)
    ]
    state.update(
        attempts=100,
        active_seconds=100,
        next_step="source",
        status="paused_quota",
        operations={
            "source": {
                "fingerprint": "fp",
                "attempts": 100,
                "role": "solver",
                "status": "paused_quota",
                "attempt_history": attempts,
            }
        },
    )
    atomic_json(tmp_path / "run-state.json", state)
    original = (tmp_path / "run-state.json").read_bytes()
    assert len(original) > 4 * 1024 * 1024
    j = Journal(tmp_path, {}, profile)
    try:
        migration = j.data["attempt_history_migration"]
        archive = tmp_path / migration["path"]
        assert (
            archive.read_bytes() == original and digest(original) == migration["sha256"]
        )
        assert (tmp_path / "run-state.json").stat().st_size < 200000
        record = j.data["operations"]["source"]
        assert record["quota_attempts"] == 100
        assert (
            len(record["attempt_history"]) == 2
            and len(list(attempt_history(tmp_path, record))) == 100
        )
        originals = {
            p: digest(p.read_bytes())
            for p in (tmp_path / "attempt-ledger").rglob("*.json")
        }
        j.begin("source", "fp", "solver")
        j.end(quota=QuotaExhaustedError("quota", 1))
        j.close()
        j = Journal(tmp_path, {}, profile)
        j.begin("source", "fp", "solver")
        j.end({"answer": "completed"})
        assert j.data["attempts"] == 102
        assert j.data["operations"]["source"]["quota_attempts"] == 101
        assert (
            len(list(attempt_history(tmp_path, j.data["operations"]["source"]))) == 102
        )
        assert all(digest(p.read_bytes()) == sha for p, sha in originals.items())
        assert archive.read_bytes() == original
    finally:
        j.close()


def test_crash_между_архивом_и_metadata_не_меняет_попытку(tmp_path, monkeypatch):
    from research_harness.state import attempt_history

    profile = dict(
        profile_from(PACKAGE / "run-profile.json"), autonomous_quota_wait=True
    )
    j = Journal(tmp_path, {}, profile)
    j.begin("source", "fp", "solver")
    save = j.save
    calls = [0]

    def crash():
        calls[0] += 1
        if calls[0] == 2:
            raise InterruptedError("after immutable attempt")
        save()

    monkeypatch.setattr(j, "save", crash)
    with pytest.raises(InterruptedError):
        j.end(quota=QuotaExhaustedError("quota", 1))
    shards = {
        p: digest(p.read_bytes()) for p in (tmp_path / "attempt-ledger").rglob("*.json")
    }
    j.close()
    resumed = Journal(tmp_path, {}, profile)
    try:
        resumed.begin("source", "fp", "solver")
        resumed.end({"answer": "done"})
        assert (
            len(list(attempt_history(tmp_path, resumed.data["operations"]["source"])))
            == 2
        )
        assert all(digest(p.read_bytes()) == sha for p, sha in shards.items())
    finally:
        resumed.close()


def test_public_audit_видит_автономный_round_и_critic(tmp_path):
    from research_harness.audit import index

    campaign = tmp_path / "campaign"
    root = campaign / "autonomous/epochs/000001/tasks/H2/round-000001"
    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        nonpaid_fixture=True,
        autonomous_quota_wait=True,
    )
    make_round(root, {}, {}, profile, {})
    critic = root / "critics/000001"
    critic.mkdir(parents=True)
    runtime = Runtime(profile, critic, {}, phase="critic")
    runtime._execute_once = lambda *args, **kwargs: {"answer": "fixture", "files": {}}
    runtime.execute(
        {},
        {"kind": "agent", "operation_id": "critic", "role": "selector"},
        model_role="selector",
    )
    runtime.finish()
    report = index(campaign)
    assert report["journal_count"] == 2 and report["evidence_complete"]
    assert {row["phase"] for row in report["rounds"]} == {"run", "critic"}
    assert all(
        row["operations"][0]["attempt_ledger"]["count"] == 1 for row in report["rounds"]
    )
    assert index(tmp_path / "empty")["evidence_complete"] is False
    archived = next((critic / "attempt-ledger").rglob("*.json"))
    row = json.loads(archived.read_text())
    row["record"]["active_seconds"] = 123
    atomic_json(archived, row)
    with pytest.raises(autonomy.RunStopped, match="ledger head changed"):
        index(campaign)


def test_autonomous_library_resume_сохраняет_готовый_probe(tmp_path, monkeypatch):
    from research_harness.queue import run_round

    original = Runtime._execute_once
    probes = []
    quota = [False]
    trigger = [False]

    def boundary(self, files, request, *, model_role, wall):
        if model_role is None:
            probes.append((self.root.name, request["operation_id"]))
            return {
                "execution_complete": True,
                "files": {
                    "artifacts/metrics.json": '{"configuration_ids":["fixture"]}',
                    "artifacts/observations.md": "fixture",
                },
                "errors": [],
            }
        if (
            trigger[0]
            and model_role == "proposer"
            and request.get("harvest")
            and not quota[0]
        ):
            quota[0] = True
            raise QuotaExhaustedError("quota", 1)
        return original(self, files, request, model_role=model_role, wall=wall)

    monkeypatch.setattr(Runtime, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(Runtime, "_execute_once", boundary)
    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        nonpaid_fixture=True,
        autonomous_quota_wait=True,
    )
    task = {
        "id": "H2",
        "autonomous": True,
        "mission": "Engineering fixture",
        "max_configurations": 2,
        "max_feature_sets": 2,
        "experiment_outputs": ["artifacts/metrics.json", "artifacts/observations.md"],
    }
    first = tmp_path / "round-001"
    second = tmp_path / "round-002"
    run_round(first, task, {}, profile, {})
    trigger[0] = True
    with pytest.raises(QuotaExhaustedError):
        run_round(second, task, {}, profile, {}, prior=first)
    before = json.loads((second / "run-state.json").read_text())
    completed = {
        op: r["result_sha256"]
        for op, r in before["operations"].items()
        if r["status"] == "completed"
    }
    assert any(op.startswith("probe/") for op in completed)
    monkeypatch.setattr("research_harness.state.time.time", lambda: 10**11)
    run_round(second, task, {}, profile, {}, prior=first)
    after = json.loads((second / "run-state.json").read_text())
    assert before["binding"] == after["binding"]
    assert all(
        after["operations"][op]["result_sha256"] == sha for op, sha in completed.items()
    )
    assert len([p for p in probes if p[0] == "round-002"]) == 4


def test_прерывание_записи_original_archive_не_мешает_миграции(tmp_path, monkeypatch):
    from pathlib import Path

    from research_harness.state import archive_original_journal

    profile = dict(
        profile_from(PACKAGE / "run-profile.json"), autonomous_quota_wait=True
    )
    old = Journal(tmp_path, {}, profile)
    data = old.data
    old.close()
    data.pop("autonomous_attempt_ledger")
    atomic_json(tmp_path / "run-state.json", data)
    original = (tmp_path / "run-state.json").read_bytes()
    destination = tmp_path / "journal-migrations" / (digest(original) + ".json")
    original_open = Path.open

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, body):
            self.stream.write(body[: len(body) // 2])
            self.stream.flush()
            raise InterruptedError("interrupted midway through archive write")

    def open_boundary(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return (
            PartialWrite(stream)
            if path.parent.name == "journal-migrations"
            and path.name.endswith(".tmp")
            and args
            and args[0] == "wb"
            else stream
        )

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_boundary)
        with pytest.raises(InterruptedError):
            archive_original_journal(destination, original)
    assert not destination.exists()
    assert (tmp_path / "run-state.json").read_bytes() == original
    resumed = Journal(tmp_path, {}, profile)
    try:
        assert destination.read_bytes() == original
        assert resumed.data["attempt_history_migration"]["sha256"] == digest(original)
    finally:
        resumed.close()


def test_old_epoch_с_большим_v2_журналом_читается_без_runtime_и_мутаций(
    supervisor, monkeypatch
):
    from research_harness.audit import index
    from research_harness.cli import load_json

    s = supervisor
    s.revise()
    old = s.root / "epochs/000000/tasks/H2/round-000001"
    profile = dict(s.profile, nonpaid_fixture=True)
    make_round(old, {}, {}, profile, {})
    path = old / "run-state.json"
    data = json.loads(path.read_text())
    data.pop("autonomous_attempt_ledger")
    record = next(iter(data["operations"].values()))
    record.pop("attempt_ledger", None)
    record["attempts"] = 100
    record["attempt_history"] = [
        {
            "attempt": i,
            "status": "paused_quota",
            "active_seconds": 1,
            "error": {"message": "x" * 50000},
        }
        for i in range(1, 101)
    ]
    data.update(attempts=100, active_seconds=100)
    atomic_json(path, data)
    original = path.read_bytes()
    assert len(original) > 4 * 1024 * 1024
    with pytest.raises(ValueError):
        load_json(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Old epoch Runtime/Journal must not be opened")

    monkeypatch.setattr(Journal, "__init__", forbidden)
    s.new_epoch("source rollover after old epoch")
    totals = s.exposure()
    assert totals["process_attempts"] == 100 and totals["active_seconds"] == 100
    report = index(s.campaign)
    assert report["journal_count"] == 1 and report["evidence_complete"]
    assert path.read_bytes() == original
    assert not (old / "journal-migrations").exists()


def test_exhausted_attempts_remain_blocked_after_failure(supervisor):
    supervisor.revise()
    supervisor.state.update(phase="research", stage_attempts={"H2": 17})
    supervisor.failed(ValueError("deterministic malformed evidence"))
    assert supervisor.state["phase"] == supervisor.state["status"] == "blocked"
    assert supervisor.state["next_retry_at"] == 0
    assert supervisor.tick() is False


def test_retry_deadline_is_not_shortened_or_other_processes_killed(
    supervisor, monkeypatch
):
    supervisor.revise()
    supervisor.state.update(
        status="repair_pending", phase="research", next_retry_at=2000
    )
    # Проверка источников допускается; любые попытки глобального pkill запрещены.
    original = autonomy.subprocess.run

    def guarded(command, *args, **kwargs):
        assert command[0] != "pkill"
        return original(command, *args, **kwargs)

    monkeypatch.setattr(autonomy.subprocess, "run", guarded)
    assert supervisor.tick() is True
    assert supervisor.state["next_retry_at"] == 2000


def test_replay_uses_selected_operation_not_matching_code(supervisor, monkeypatch):
    s = setup_controller(supervisor, monkeypatch)
    s.research()
    path = s.round_root() / "selection.json"
    value = json.loads(path.read_text())
    value["operation"] = "probe/other"
    atomic_json(path, value)
    # Изменение selection обнаруживается по receipt ещё до replay.
    with pytest.raises(BaseException, match="artifact changed"):
        s.repeat()


def test_critic_receives_frozen_independent_analysis(supervisor, monkeypatch):
    s = setup_controller(supervisor, monkeypatch)
    s.research()
    s.repeat()
    seen = []

    def assess_with_analysis(root, files, criteria, profile, manifest):
        seen.append(json.loads(files["scientific-analysis.json"]))
        review = review_for(files, criteria)
        atomic_json(root / "review.json", review)
        return review

    monkeypatch.setattr(autonomy, "assess", assess_with_analysis)
    s.criticism()
    assert seen == [{"fixture_only": True}]


def test_qualification_format_retry_keeps_completed_journal(supervisor):
    supervisor.revise()
    supervisor.state["qualification_attempt"] = 3
    supervisor.failed(ValueError("Invalid structured role result"))
    assert supervisor.state["phase"] == "qualification"
    assert supervisor.state["qualification_attempt"] == 3
    assert supervisor.state["status"] == "repair_pending"
    assert supervisor.state["next_retry_at"] > supervisor.test_clock[0]
    for _ in range(supervisor.profile["max_role_attempts"] - 1):
        supervisor.failed(ValueError("Invalid structured role result"))
    assert supervisor.state["status"] == "blocked"
    assert supervisor.state["phase"] == "blocked"
    assert supervisor.state["qualification_attempt"] == 3


def test_qualification_environment_retry_keeps_journal(supervisor):
    supervisor.revise()
    supervisor.state["qualification_attempt"] = 3
    supervisor.failed(EnvironmentUnavailableError("temporary connection error"))
    assert supervisor.state["qualification_attempt"] == 3
    assert supervisor.state["status"] == "waiting_environment"


def test_report_retry_reuses_all_four_completed_solver_and_probe_results(
    tmp_path, monkeypatch
):
    from research_harness.queue import run_round

    original = Runtime._execute_once
    calls = []

    def transport(self, files, request, *, model_role, wall):
        operation = request["operation_id"]
        calls.append(operation)
        if operation == "campaign/report" and calls.count(operation) == 1:
            raise ValueError("Invalid structured role result")
        if model_role is None:
            return {
                "execution_complete": True,
                "files": {
                    "artifacts/metrics.json": '{"configuration_ids":["fixture"]}',
                    "artifacts/observations.md": "Инженерная проверка",
                },
                "errors": [],
            }
        return original(self, files, request, model_role=model_role, wall=wall)

    monkeypatch.setattr(Runtime, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(Runtime, "_execute_once", transport)
    profile = dict(profile_from(PACKAGE / "run-profile.json"), nonpaid_fixture=True)
    task = {
        "id": "H2",
        "autonomous": True,
        "mission": "Инженерная проверка",
        "max_configurations": 2,
        "max_feature_sets": 2,
        "experiment_outputs": ["artifacts/metrics.json", "artifacts/observations.md"],
    }
    root = tmp_path / "round-001"
    with pytest.raises(ValueError, match="Invalid structured"):
        run_round(root, task, {}, profile, {})
    before = list(calls)
    result = run_round(root, task, {}, profile, {})
    assert result["execution_complete"]
    assert calls[len(before) :] == ["campaign/report/retry-2"]
    assert len([call for call in calls if call.startswith("solver/")]) == 4
    assert len([call for call in calls if call.startswith("probe/")]) == 4
