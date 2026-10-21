"""Неограниченные сессии модели и ограниченная память transport, без Gemini/API."""

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from research_harness.contracts import MAX_TREE, PACKAGE, digest_file
from research_harness.runtime import profile_from
from research_harness.state import BudgetExhaustedError, Journal, RunStopped
from research_harness.subscription import GeminiSubscriptionRunner, QuotaExhaustedError


def runner(monkeypatch):
    result = GeminiSubscriptionRunner(cli_bin=sys.executable)

    def preflight():
        result.identity = {}
        result.environment = dict(os.environ)

    monkeypatch.setattr(result, "preflight", preflight)
    return result


def native_events(steps=80, text_size=1):
    rows = [
        {
            "event": "init",
            "init": {"model": "gemini-3.8-flash-high", "tools": ["run_command"]},
        }
    ]
    rows.extend(
        {
            "event": "step_update",
            "step_update": {
                "step_index": i,
                "step_type": "agent_response",
                "state": "DONE",
                "text_delta": "x" * text_size,
            },
        }
        for i in range(steps)
    )
    rows.append(
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "response": json.dumps(
                    {
                        "answer": "ready",
                        "files": {"candidate/checkpoint.py": "# complete fixture"},
                    }
                ),
            },
        }
    )
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


class Process:
    pid = 123456789
    returncode = 0

    def __init__(self, stdout, stderr=b""):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.stdin = io.BytesIO()

    def poll(self):
        return self.returncode

    def wait(self, **kwargs):
        return self.returncode


def test_более_50_шагов_и_больше_MAX_TREE_истории_с_ограниченным_result(
    tmp_path, monkeypatch
):
    process = Process(native_events(80, 40000), b"bounded-stderr-tail\n" * 180000)
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen", lambda *a, **kw: process
    )
    result = runner(monkeypatch).run_turn(
        tmp_path, "fixture", timeout=None, max_steps=None
    )
    assert result["_usage"]["observed_step_updates"] == 80
    assert result["_usage"]["session_limits"] == {"steps": None, "seconds": None}
    assert len(result["events"].encode()) < 200000
    assert (tmp_path / "stream-events.jsonl").stat().st_size > MAX_TREE
    assert (tmp_path / "stream-stdout.jsonl").stat().st_size > MAX_TREE
    assert (tmp_path / "stream-stderr.log").stat().st_size > MAX_TREE
    archive = json.loads(
        tmp_path.with_name(tmp_path.name + "-archive.json").read_text()
    )
    assert archive["evidence_complete"]
    for name in ("stream-events.jsonl", "stream-stdout.jsonl", "stream-stderr.log"):
        assert archive["files"][name]["sha256"] == digest_file(tmp_path / name)
        assert result["_streams"][
            {
                "stream-events.jsonl": "events",
                "stream-stdout.jsonl": "stdout",
                "stream-stderr.log": "stderr",
            }[name]
        ]["sha256"] == digest_file(tmp_path / name)


def test_виртуально_дольше_900_секунд_не_обрывает_неограниченную_сессию(
    tmp_path, monkeypatch
):
    process = Process(native_events(2))
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen", lambda *a, **kw: process
    )
    clock = [0.0]
    monkeypatch.setattr(
        "research_harness.subscription.time.monotonic", lambda: clock[0]
    )

    def heartbeat():
        clock[0] += 1000

    result = runner(monkeypatch).run_turn(
        tmp_path, "fixture", timeout=None, max_steps=None, heartbeat=heartbeat
    )
    assert clock[0] > 900 and result["answer"] == "ready"


def test_legacy_сохраняет_фиксированный_лимит_шагов(tmp_path, monkeypatch):
    process = Process(native_events(51))
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen", lambda *a, **kw: process
    )
    with pytest.raises(ValueError, match="step budget"):
        runner(monkeypatch).run_turn(tmp_path, "legacy", timeout=900, max_steps=50)


def test_legacy_сохраняет_deadline(tmp_path, monkeypatch):
    process = Process(native_events(2))
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen", lambda *a, **kw: process
    )
    clock = [0.0]
    monkeypatch.setattr(
        "research_harness.subscription.time.monotonic", lambda: clock[0]
    )

    def heartbeat():
        clock[0] += 1000

    with pytest.raises(TimeoutError):
        runner(monkeypatch).run_turn(
            tmp_path, "legacy", timeout=900, max_steps=50, heartbeat=heartbeat
        )


def test_учёт_времени_LLM_не_исчерпывает_compute_budget(tmp_path, monkeypatch):
    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        model_sessions="unbounded",
        autonomous_quota_wait=True,
        wall_seconds=10,
        call_timeout=10,
    )
    clock = [100.0]
    monkeypatch.setattr("research_harness.state.time.monotonic", lambda: clock[0])
    journal = Journal(tmp_path, {}, profile)
    try:
        assert journal.begin("source", "fp", "solver", unbounded=True) is None
        clock[0] += 20000
        journal.heartbeat()
        state = json.loads((tmp_path / "run-state.json").read_text())
        assert (
            state["active_seconds"] == 20000 and state["model_active_seconds"] == 20000
        )
        journal.end({"answer": "done"})
        assert journal.begin("probe", "other", "experiment") == 10
        clock[0] += 10
        journal.end({"execution_complete": True})
        with pytest.raises(BudgetExhaustedError):
            journal.begin("probe2", "next", "experiment")
        assert (
            journal.data["active_seconds"] == 20010
            and journal.data["model_active_seconds"] == 20000
        )
    finally:
        journal.close()


def test_quota_resume_после_долгой_LLM_не_сбрасывает_usage(tmp_path, monkeypatch):
    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        model_sessions="unbounded",
        autonomous_quota_wait=True,
        wall_seconds=1,
        call_timeout=1,
        max_calls=1,
        max_role_attempts=1,
    )
    clock = [100.0]
    monkeypatch.setattr("research_harness.state.time.monotonic", lambda: clock[0])
    j = Journal(tmp_path, {}, profile)
    try:
        for _ in range(3):
            j.begin("source", "fp", "solver", unbounded=True)
            clock[0] += 1000
            j.end(quota=QuotaExhaustedError("quota", 1))
        assert j.data["active_seconds"] == 3000
        assert j.data["model_active_seconds"] == 3000
        j.close()
        j = Journal(tmp_path, {}, profile)
        j.begin("source", "fp", "solver", unbounded=True)
        clock[0] += 1000
        j.end({"answer": "done"})
        assert j.data["attempts"] == 4 and j.data["active_seconds"] == 4000
        assert j.data["operations"]["source"]["quota_attempts"] == 3
    finally:
        j.close()


def test_STOP_на_молчащем_stream_завершает_process_group(tmp_path, monkeypatch):
    script = tmp_path.parent / "silent-cli.py"
    script.write_text(
        'import json,sys,time\nprint(json.dumps({"event":"init","init":{"model":"gemini-3.8-flash-high","tools":["run_command"]}}),flush=True)\nsys.stdin.readline()\ntime.sleep(120)\n'
    )
    original = subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = original([sys.executable, "-u", str(script)], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("research_harness.subscription.subprocess.Popen", launch)
    stop = tmp_path.parent / "STOP"
    timer = threading.Timer(0.3, lambda: stop.write_text("stop"))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RunStopped, match="STOP") as error:
            runner(monkeypatch).run_turn(
                tmp_path, "fixture", timeout=None, max_steps=None, stop_file=stop
            )
        assert time.monotonic() - started < 3
        assert children[0].poll() is not None
        assert error.value.public_failure["evidence_complete"] is False
        assert error.value.public_failure["streams"]["stdout"]["sha256"] == digest_file(
            tmp_path / "stream-stdout.jsonl"
        )
    finally:
        timer.cancel()
        timer.join()
        if children and children[0].poll() is None:
            os.killpg(children[0].pid, signal.SIGKILL)
            children[0].wait()


def test_runtime_всех_ролей_явно_передаёт_unbounded_и_сохраняет_probe_budget(
    tmp_path, monkeypatch
):
    from research_harness.runtime import Runtime

    profile = dict(
        profile_from(PACKAGE / "run-profile.json"),
        model_sessions="unbounded",
        autonomous_quota_wait=True,
        stop_file=str(tmp_path / "STOP"),
    )
    clock = [0.0]
    monkeypatch.setattr("research_harness.state.time.monotonic", lambda: clock[0])
    runtime = Runtime(profile, tmp_path, {})
    runtime.journal.data["transport"] = {}
    seen = []

    class Subscription:
        def preflight(self):
            return {}

        def run_turn(self, workspace, **kwargs):
            assert kwargs["timeout"] is None and kwargs["max_steps"] is None
            assert kwargs["stop_file"] == str(tmp_path / "STOP")
            seen.append(kwargs["role"])
            clock[0] += 6000
            kwargs["heartbeat"]()
            return {"answer": "fixture", "files": {}}

    runtime.runner = Subscription()
    try:
        for role in ("solver", "proposer", "selector"):
            runtime.execute(
                {},
                {
                    "kind": "agent",
                    "role": role,
                    "operation_id": role,
                    "mission": "fixture",
                },
                model_role=role,
            )
        assert seen == ["solver", "proposer", "selector"]
        assert runtime.journal.data["model_active_seconds"] == 18000
        wall = runtime.journal.begin("probe", "fp", "experiment")
        assert wall == 900
        runtime.journal.end({"execution_complete": True})
    finally:
        runtime.finish()
    assert runtime.usage["model_sessions"] == "unbounded"
    assert runtime.usage["model_active_seconds"] == 18000


def test_секреты_на_границе_чанков_очищаются_до_записи(tmp_path, monkeypatch):
    from research_harness.subscription import StreamRedactor

    samples = [
        b"prefix Authorization: Bearer secret-bearer-value\n",
        b'{"access_token":"secret-json-value","ok":true}\n',
        b"prefix ya29.secret-google-value\n",
        b'{"text":"{\\"access_token\\":\\"secret-embedded-value\\"}"}\n',
    ]
    for sample in samples:
        for split in range(1, len(sample)):
            cleaner = StreamRedactor()
            output = cleaner.feed(sample[:split]) + cleaner.feed(
                sample[split:], final=True
            )
            assert b"secret-" not in output
            assert b"[REDACTED]" in output
    events = native_events(2).replace(
        b'"text_delta": "x"',
        b'"text_delta": "x", "access_token": "secret-json-value"',
    )
    process = Process(
        events, b"x" * 4080 + b"Authorization: Bearer secret-bearer-value\n"
    )
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen", lambda *a, **kw: process
    )
    runner(monkeypatch).run_turn(tmp_path, "fixture", timeout=None, max_steps=None)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"secret-json-value" not in path.read_bytes()
            assert b"secret-bearer-value" not in path.read_bytes()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("operation", ["write", "flush"])
def test_reader_IO_failure_не_висит_и_не_принимает_success(
    tmp_path, monkeypatch, stream, operation
):
    from pathlib import Path

    original_open = Path.open

    class Faulty:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.wrapped.close()

        def __getattr__(self, name):
            if name == operation:

                def fail(*args, **kwargs):
                    raise OSError("injected disk failure")

                return fail
            return getattr(self.wrapped, name)

    def opening(path, *args, **kwargs):
        wrapped = original_open(path, *args, **kwargs)
        expected = "stream-stdout.jsonl" if stream == "stdout" else "stream-stderr.log"
        return Faulty(wrapped) if path.name == expected and args == ("wb",) else wrapped

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen",
        lambda *a, **kw: Process(native_events(2), b"diagnostic"),
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Transport reader failed") as error:
        runner(monkeypatch).run_turn(tmp_path, "fixture", timeout=None, max_steps=None)
    assert time.monotonic() - started < 3
    assert error.value.public_failure["evidence_complete"] is False
    assert error.value.public_failure["readers_complete"] is True
    assert (stream, "OSError") in error.value.public_failure["reader_errors"]
    archive = json.loads(
        tmp_path.with_name(tmp_path.name + "-archive.json").read_text()
    )
    assert archive["evidence_complete"] is False


def test_STOP_убивает_оставшегося_ребёнка_до_hash_выросшего_stderr(
    tmp_path, monkeypatch
):
    script = tmp_path.parent / "orphan-cli.py"
    child_pid = tmp_path.parent / "child.pid"
    script.write_text(
        "import os,sys,json,time,signal\n"
        'print(json.dumps({"event":"init","init":{"model":"gemini-3.8-flash-high","tools":[]}}),flush=True)\n'
        "sys.stdin.readline()\n"
        "pid=os.fork()\n"
        "if pid:\n os._exit(0)\n"
        f'open({str(child_pid)!r},"w").write(str(os.getpid()))\n'
        'while True:\n os.write(2,b"diagnostic\\n"*10000)\n time.sleep(0.002)\n'
    )
    original_popen = subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = original_popen([sys.executable, "-u", str(script)], **kwargs)
        children.append(child)
        return child

    original_hash = digest_file
    hashes_after_group_stop = []
    killed = threading.Event()
    original_killpg = os.killpg

    def killpg(pid, sig):
        original_killpg(pid, sig)
        killed.set()

    def hashing(path):
        if path.name.startswith("stream-"):
            assert killed.is_set(), "hash started before terminating producers"
            hashes_after_group_stop.append(str(path))
        return original_hash(path)

    monkeypatch.setattr("research_harness.subscription.subprocess.Popen", launch)
    monkeypatch.setattr("research_harness.subscription.os.killpg", killpg)
    monkeypatch.setattr("research_harness.subscription.digest_file", hashing)
    stop = tmp_path.parent / "ORPHAN-STOP"

    def stop_after_output():
        path = tmp_path / "stream-stderr.log"
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if path.exists() and path.stat().st_size > MAX_TREE and child_pid.exists():
                stop.write_text("stop")
                return
            time.sleep(0.01)
        stop.write_text("stop")

    timer = threading.Thread(target=stop_after_output)
    timer.start()
    try:
        with pytest.raises(RunStopped, match="STOP") as error:
            runner(monkeypatch).run_turn(
                tmp_path, "fixture", timeout=None, max_steps=None, stop_file=stop
            )
        assert (
            children[0].poll() == 0
        )  # leader exited before STOP; child still held pipes
        assert (
            hashes_after_group_stop and error.value.public_failure["readers_complete"]
        )
        assert (tmp_path / "stream-stderr.log").stat().st_size > MAX_TREE
    finally:
        timer.join()
        if children:
            try:
                original_killpg(children[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_structured_OAuth_tokens_не_попадают_в_events_и_на_диск(tmp_path, monkeypatch):
    raw = native_events(2).replace(
        b'"text_delta": "x"',
        b'"text_delta": "x", "tool_info": {"id_token": "secret-id-value", "oauth_token": "secret-oauth-value"}',
    )
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen",
        lambda *a, **kw: Process(raw),
    )
    result = runner(monkeypatch).run_turn(
        tmp_path, "fixture", timeout=None, max_steps=None
    )
    assert "secret-id-value" not in json.dumps(result)
    assert "secret-oauth-value" not in json.dumps(result)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"secret-id-value" not in path.read_bytes()
            assert b"secret-oauth-value" not in path.read_bytes()


@pytest.mark.parametrize(
    "sample,secret",
    [
        (b"Authorization: Basic dXNlcjpwYXNz\n", b"dXNlcjpwYXNz"),
        (b'{"authorization":"Basic dXNlcjpwYXNz","ok":true}\n', b"dXNlcjpwYXNz"),
        (
            b'Authorization: Digest realm="private realm", nonce="hidden nonce"\n',
            b"hidden nonce",
        ),
        (
            b'authorization="Custom arbitrary multi word credential"\n',
            b"multi word credential",
        ),
        (
            b'{"authorization":"Negotiate multi word credential","ok":true}\n',
            b"multi word credential",
        ),
        (
            b'{"text":"{\\"authorization\\":\\"Custom multi word credential\\"}"}\n',
            b"multi word credential",
        ),
        (
            b'{"authorization":Custom multi word credential,"ok":true}\n',
            b"multi word credential",
        ),
    ],
)
def test_sensitive_field_полностью_скрыт_при_любом_chunk_split(sample, secret):
    from research_harness.subscription import StreamRedactor

    for split in range(len(sample) + 1):
        cleaner = StreamRedactor()
        safe = cleaner.feed(sample[:split]) + cleaner.feed(sample[split:], final=True)
        assert secret not in safe
        assert b"[REDACTED]" in safe
        if b'"ok":true' in sample:
            assert b'"ok":true' in safe


def test_длинное_секретное_значение_не_накапливается_в_RAM_и_на_диске(
    tmp_path, monkeypatch
):
    from research_harness.subscription import StreamRedactor

    cleaner = StreamRedactor()
    cleaner.feed(b'{"authorization": "Custom ')
    for _ in range(100):
        assert cleaner.feed(b"private-credential " * 4096) == b""
        assert len(cleaner.pending) <= 256
    assert b"private" not in cleaner.feed(b'last secret", "ok":true}', final=True)
    process = Process(native_events(1), b"Authorization: Basic dXNlcjpwYXNz\n")
    monkeypatch.setattr(
        "research_harness.subscription.subprocess.Popen", lambda *a, **kw: process
    )
    result = runner(monkeypatch).run_turn(
        tmp_path, "fixture", timeout=None, max_steps=None
    )
    assert "dXNlcjpwYXNz" not in json.dumps(result)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"dXNlcjpwYXNz" not in path.read_bytes()
