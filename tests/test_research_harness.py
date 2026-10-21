"""H1 nonpaid subscription regression coverage. HARNESS2_SOURCE is an explicit pinned checkout."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, os.environ["HARNESS2_SOURCE"])
from research_harness.adapter import CravingAdapter, IsolatedProposer, compare
from research_harness.cli import (
    evaluator_manifest,
    require_complete_run,
    require_harness_stages,
    write_json,
)
from research_harness.contracts import (
    CANDIDATE,
    PACKAGE,
    digest,
    fixture,
    json_bytes,
    read_tree,
    source_packet,
)
from research_harness.runtime import profile_from
from research_harness.subscription import (
    GeminiSubscriptionRunner,
    QuotaExhaustedError,
    find_gemini_cli,
)


def test_profile_validation_v2(tmp_path):
    valid_profile = {
        "schema": "craving-harness-run/v2",
        "subscription_authorized": True,
        "image": "sha256:6b568b2e2be9a8514ec921f60bb35a891b7fedff229d68f7c25402fa87edfae2",
        "wall_seconds": 600,
        "call_timeout": 60,
        "max_calls": 20,
        "memory_mb": 512,
        "pids": 64,
        "cpus": 1.0,
        "tmpfs_mb": 128,
        "max_solver_executions": 4,
        "parallelism": 1,
        "transport_attempts": 1,
        "roles": {
            "solver": {"model": "gemini-3.8-flash-high", "steps": 4},
            "proposer": {"model": "gemini-3.8-flash-high", "steps": 4},
            "selector": {"model": "gemini-3.8-flash-high", "steps": 4},
        },
    }
    p_file = tmp_path / "profile.json"
    write_json(p_file, valid_profile)
    assert profile_from(p_file)["schema"] == "craving-harness-run/v2"

    # Missing subscription authorization fails closed
    unauth = dict(valid_profile, subscription_authorized=False)
    write_json(p_file, unauth)
    with pytest.raises(ValueError, match="subscription"):
        profile_from(p_file)

    # Invalid image digest fails closed
    bad_image = dict(valid_profile, image="craving-worker:latest")
    write_json(p_file, bad_image)
    with pytest.raises(ValueError, match="digest-pinned"):
        profile_from(p_file)

    # Missing role fails closed
    bad_roles = dict(
        valid_profile, roles={"solver": {"model": "gemini-3.8-flash-high"}}
    )
    write_json(p_file, bad_roles)
    with pytest.raises(ValueError, match="solver/proposer/selector"):
        profile_from(p_file)


def test_subscription_quota_detection():
    from research_harness.subscription import parse_quota_error

    is_q, delay = parse_quota_error("Error: RESOURCE_EXHAUSTED: Quota exceeded")
    assert is_q and delay == 30

    is_q, delay = parse_quota_error("HTTP 429 Too Many Requests")
    assert is_q and delay == 30

    is_q, delay = parse_quota_error("Usage limit reached for model gemini-3.8-flash")
    assert is_q and delay == 30

    is_q, delay = parse_quota_error(
        '{"error": {"code": 429, "details": [{"retryDelayMs": 45000}]}}'
    )
    assert is_q and delay == 45

    is_q, delay = parse_quota_error(
        "Rate limit reached. Please retry after 120 seconds."
    )
    assert is_q and delay == 120

    is_q, delay = parse_quota_error("SyntaxError: invalid syntax")
    assert not is_q and delay == 0


def test_subscription_candidate_extraction():
    from research_harness.subscription import extract_python_code

    md_block = """Here is the checkpoint module:
```python
def save(memory):
    return memory.export_state()

def restore(memory_class, serialized):
    return memory_class.from_state(serialized)
```
Explanation of the solution.
"""
    extracted = extract_python_code(md_block)
    assert extracted is not None
    assert "def save(" in extracted
    assert "def restore(" in extracted
    assert "Explanation" not in extracted

    assert extract_python_code("def foo(): pass") is None


def test_subscription_cli_binary_lookup(tmp_path, monkeypatch):
    custom_bin = tmp_path / "mock_gemini"
    custom_bin.write_text("#!/bin/sh\necho ok\n")
    custom_bin.chmod(0o755)

    monkeypatch.setenv("GEMINI_CLI_BIN", str(custom_bin))
    assert find_gemini_cli() == str(custom_bin)


def test_simulated_quota_hook(tmp_path, monkeypatch):
    from research_harness.subscription import parse_quota_error

    monkeypatch.setenv("RESEARCH_HARNESS_SIMULATE_QUOTA", "15")
    is_q, delay = parse_quota_error("some normal text")
    assert is_q and delay == 15

    runner = GeminiSubscriptionRunner(
        cli_bin="/bin/echo",
        max_quota_retries=0,
    )
    with pytest.raises(QuotaExhaustedError) as exc_info:
        runner.run_turn(
            workspace=tmp_path,
            prompt="solve task",
        )
    assert exc_info.value.retry_after == 15


def test_manifest_ignores_bytecode_before_reading(tmp_path, monkeypatch):
    from research_harness import cli

    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/bad.pyc").write_bytes(b"\xff\x00")
    (tmp_path / "worker.py").write_text("x = 1")
    monkeypatch.setattr(cli, "PACKAGE", tmp_path)
    assert evaluator_manifest() == {"worker.py": digest(b"x = 1")}


def test_original_run_close_failure_cannot_be_judged(tmp_path):
    write_json(tmp_path / "selection.json", {})
    with pytest.raises((OSError, ValueError)):
        require_complete_run(tmp_path, {})


def test_complete_marker_hashes_report_and_closed_accounting(tmp_path):
    frozen, selection = {"source": "frozen"}, {"chosen": "x"}
    report = {
        "command": "run",
        "status": "selected_not_judged",
        "error": None,
        "manifest_sha256": digest(json_bytes(frozen)),
        "usage": {"closed": True, "enforced": True},
    }
    write_json(tmp_path / "selection.json", selection)
    write_json(tmp_path / "run-report-1.json", report)
    write_json(
        tmp_path / "run-complete.json",
        {
            "schema": "craving-harness-complete/v1",
            "report": "run-report-1.json",
            "manifest_sha256": digest(json_bytes(frozen)),
            "selection_sha256": digest(json_bytes(selection)),
            "report_sha256": digest(json_bytes(report)),
        },
    )
    assert require_complete_run(tmp_path, frozen) == selection
    (tmp_path / "run-report-1.json").write_text(
        json.dumps(dict(report, error="close failed"))
    )
    with pytest.raises(ValueError):
        require_complete_run(tmp_path, frozen)


def test_fallback_stages_block_core_success():
    result = SimpleNamespace(
        status="ok",
        errors=[],
        modes={"agg": {"ok": True, "errors": [], "fell_back": ["inv1"], "stages": []}},
    )
    with pytest.raises(ValueError, match="стадия"):
        require_harness_stages(result, SimpleNamespace(completed_executions=[]))


def worker_call(kind, candidate, checkpoints=None):
    script = (
        "import json,sys; "
        "from research_harness.worker import replay; "
        "print(json.dumps(replay(json.load(sys.stdin),candidate_path=sys.argv[1]),allow_nan=False))"
    )
    body = {"kind": kind, "fixture": fixture()}
    if checkpoints:
        body["checkpoints"] = checkpoints
    result = subprocess.run(
        [sys.executable, "-c", script, str(candidate)],
        input=json.dumps(body),
        text=True,
        capture_output=True,
        timeout=40,
        check=True,
    )
    return json.loads(result.stdout)


def test_actual_fresh_process_restart_and_full_state(tmp_path):
    candidate = tmp_path / "checkpoint.py"
    candidate.write_text(
        "def save(m): return m.export_state()\ndef restore(c,s): return c.from_state(s)\n"
    )
    reference = worker_call("reference", candidate)
    saved = worker_call("candidate_save", candidate)
    checkpoints = {k: v["checkpoint"] for k, v in saved.items()}
    restored = worker_call("candidate_restore", candidate, checkpoints)
    observed = {
        k: {
            "checkpoint": checkpoints[k],
            "trace": saved[k]["trace"] + restored[k]["trace"],
        }
        for k in checkpoints
    }
    assert compare(reference, observed)["all_pass"]
    observed["TunedScalarMemory"]["trace"][-1]["state"] = "{}"
    assert not compare(reference, observed)["all_pass"]


def test_global_stash_cannot_cross_fresh_process(tmp_path):
    candidate = tmp_path / "checkpoint.py"
    candidate.write_text(
        "stash={}\ndef save(m): stash[type(m).__name__]=m; return m.export_state()\ndef restore(c,s): return stash[c.__name__]\n"
    )
    saved = worker_call("candidate_save", candidate)
    with pytest.raises(subprocess.CalledProcessError):
        worker_call(
            "candidate_restore",
            candidate,
            {k: v["checkpoint"] for k, v in saved.items()},
        )


def test_invalid_profile_and_symlink_artifacts_fail_closed(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text('{"schema":"craving-harness-run/v1","subscription_authorized":false}')
    with pytest.raises(ValueError):
        profile_from(p)
    (tmp_path / "link").symlink_to(p)
    with pytest.raises(ValueError, match="Symlink"):
        read_tree(tmp_path)


@pytest.mark.parametrize("proposer_failure", [False, True])
def test_real_pinned_core_nonpaid_transport(tmp_path, proposer_failure):
    from harness2.core.schedule import run_task

    class LocalFixtureRuntime:
        profile: ClassVar[dict] = {
            "call_timeout": 60,
            "roles": {
                role: {"model": "gemini-3.8-flash-high", "steps": 4, "effort": "medium"}
                for role in ("solver", "proposer", "selector")
            },
        }
        calls: ClassVar[list] = []

        def execute(self, files, request, *, model_role=None):
            self.calls.append(model_role)
            if model_role == "proposer" and proposer_failure:
                raise RuntimeError("fixture: no model available")
            output = (
                {CANDIDATE: (PACKAGE / CANDIDATE).read_text()}
                if model_role == "solver"
                else {
                    k.removeprefix("context/harness/"): v
                    for k, v in files.items()
                    if k.startswith("context/harness/")
                }
            )
            return {
                "files": output,
                "answer": "<notes>Keep complete checkpoint API.</notes>",
                "events": json.dumps(
                    {
                        "type": "text",
                        "part": {"type": "text", "text": "Checkpoint module saved."},
                    }
                ),
                "_usage": {"tokens": 0, "cost_usd": 0},
            }

    runtime = LocalFixtureRuntime()
    adapter = CravingAdapter(
        tmp_path, runtime, source_packet(Path(__file__).parents[1])
    )
    adapter.task_files(TASK := "checkpoint_restore").mkdir()
    result = run_task(
        TASK,
        1,
        adapter=adapter,
        prop=IsolatedProposer(adapter, "proposer"),
        selector_prop=IsolatedProposer(adapter, "selector"),
        runs_root=tmp_path / "core",
        modes=("agg",),
        seed=None,
        library=None,
        tag="subscription",
    )
    assert runtime.calls.count("solver") == 4
    assert "selector" not in runtime.calls
    if proposer_failure:
        with pytest.raises(ValueError, match="стадия"):
            require_harness_stages(result, adapter)
    else:
        require_harness_stages(result, adapter)
        assert result.status == "ok"


def test_rollout_resumption_does_not_repeat_finished(tmp_path):
    """Test that existing valid rollouts are reused without re-executing."""

    class CounterRuntime:
        profile: ClassVar[dict] = {
            "call_timeout": 60,
            "roles": {
                "solver": {"model": "gemini-3.8-flash-high", "steps": 4},
                "proposer": {"model": "gemini-3.8-flash-high", "steps": 4},
                "selector": {"model": "gemini-3.8-flash-high", "steps": 4},
            },
        }
        solver_invocations: int = 0

        def execute(self, files, request, *, model_role=None):
            if model_role == "solver":
                self.solver_invocations += 1
            return {
                "files": {CANDIDATE: (PACKAGE / CANDIDATE).read_text()},
                "answer": "Candidate solved",
                "events": json.dumps({"type": "text"}),
            }

    runtime = CounterRuntime()
    adapter = CravingAdapter(
        tmp_path, runtime, source_packet(Path(__file__).parents[1])
    )
    adapter.task_files(TASK := "checkpoint_restore").mkdir()

    # First execution creates rollout
    first_path = adapter.execute(TASK, None, "test-rollout-1")
    assert runtime.solver_invocations == 1
    assert first_path.exists()
    assert adapter.delivered(first_path)

    # Second execution of same run_id reuses existing rollout and does not call solver
    second_path = adapter.execute(TASK, None, "test-rollout-1")
    assert runtime.solver_invocations == 1
    assert second_path == first_path


def test_quota_checkpoint_and_resume_integrity(tmp_path):
    """Test cycle_checkpoint schema, status transitions and manifest verification on resume."""
    from research_harness.cli import load_json

    root = tmp_path / "run_dir"
    root.mkdir()

    # Write initial paused checkpoint
    cp_file = root / "cycle_checkpoint.json"
    write_json(
        cp_file,
        {
            "schema": "craving-harness-checkpoint/v1",
            "status": "paused_quota",
            "method": "harness",
            "retry_after": 45,
            "paused_at": time.time(),
            "completed_executions": ["run-1", "run-2"],
            "reason": "QuotaExhaustedError: Resource exhausted",
        },
    )

    loaded = load_json(cp_file)
    assert loaded["schema"] == "craving-harness-checkpoint/v1"
    assert loaded["status"] == "paused_quota"
    assert loaded["retry_after"] == 45
    assert loaded["completed_executions"] == ["run-1", "run-2"]


def journal_profile():
    return {
        "call_timeout": 60,
        "wall_seconds": 600,
        "max_calls": 20,
        "max_role_attempts": 4,
        "max_solver_executions": 4,
        "nonpaid_fixture": True,
        "roles": {
            role: {"model": "gemini-3.8-flash-high", "steps": 50}
            for role in ("solver", "proposer", "selector")
        },
    }


def test_quota_is_supervisor_control_signal():
    # Every known upstream fallback catches Exception; quota must cross that boundary.
    from research_harness.state import RunStopped

    assert issubclass(QuotaExhaustedError, RunStopped)
    assert not issubclass(QuotaExhaustedError, Exception)


def test_journal_persists_budget_role_outputs_and_pending_step(tmp_path):
    from research_harness.state import (
        BudgetExhaustedError,
        Journal,
        ResumeMismatchError,
    )

    profile = journal_profile()
    first = Journal(tmp_path, {"source": "frozen"}, profile)
    first.begin("proposer/inv1.a1", "input-a", "proposer")
    first.end({"answer": "done", "files": {}})
    first.begin("solver/inv1", "input-b", "solver")
    first.end(quota=QuotaExhaustedError("quota", retry_after=1))
    first.data["operations"]["solver/inv1"]["not_before"] = 0
    first.save()
    first.close()
    resumed = Journal(tmp_path, {"source": "frozen"}, profile)
    assert resumed.data["attempts"] == 2
    assert resumed.cached("proposer/inv1.a1", "input-a")["answer"] == "done"
    assert resumed.data["next_step"] == "solver/inv1"
    with pytest.raises(ResumeMismatchError):
        resumed.cached("proposer/inv1.a1", "changed-input")
    assert resumed.cached("solver/inv1", "input-b") is None
    resumed.data["attempts"] = profile["max_calls"]
    with pytest.raises(BudgetExhaustedError):
        resumed.begin("solver/inv1", "input-b", "solver")
    resumed.close()


def test_uncertain_inflight_result_is_never_replayed(tmp_path):
    from research_harness.state import Journal, ResumeMismatchError

    journal = Journal(tmp_path, {}, journal_profile())
    journal.begin("solver/base1", "exact-input", "solver")
    reserved = journal.data["active_seconds"]
    journal.close()  # abrupt shutdown before result/quota acknowledgement
    resumed = Journal(tmp_path, {}, journal_profile())
    assert resumed.data["active_seconds"] == reserved
    with pytest.raises(ResumeMismatchError, match="неизвестен"):
        resumed.cached("solver/base1", "exact-input")
    resumed.close()


def test_actual_core_quota_resume_reuses_completed_proposer_and_solver(
    tmp_path, monkeypatch
):
    from harness2.core.schedule import run_task

    from research_harness.contracts import write_tree
    from research_harness.runtime import Runtime

    calls = []
    paused = False
    original = Runtime._execute_once

    def transport(self, files, request, *, model_role, wall):
        nonlocal paused
        operation = request["operation_id"]
        calls.append(operation)
        if model_role == "proposer" and operation.endswith("dom1.a1") and not paused:
            paused = True
            raise QuotaExhaustedError("fixture quota", retry_after=1)
        return original(self, files, request, model_role=model_role, wall=wall)

    monkeypatch.setattr(Runtime, "_execute_once", transport)
    sources = source_packet(Path(__file__).parents[1])
    write_tree(tmp_path / "task_source", sources)
    profile = journal_profile()

    def drive(runtime):
        adapter = CravingAdapter(tmp_path, runtime, sources)
        result = run_task(
            "checkpoint_restore",
            1,
            adapter=adapter,
            prop=IsolatedProposer(adapter, "proposer"),
            selector_prop=IsolatedProposer(adapter, "selector"),
            runs_root=tmp_path / "core",
            modes=("agg",),
            library=None,
            seed=None,
            tag="h1",
        )
        require_harness_stages(result, adapter)
        return result

    first = Runtime(profile, tmp_path, {"frozen": "source/config"})
    with pytest.raises(QuotaExhaustedError):
        drive(first)
    assert calls[-1].endswith("dom1.a1")
    assert first.journal.data["next_step"].endswith("dom1.a1")
    for record in first.journal.data["operations"].values():
        record["not_before"] = 0
    first.finish("paused_quota")
    before = list(calls)
    resumed = Runtime(profile, tmp_path, {"frozen": "source/config"})
    result = drive(resumed)
    assert result.status == "ok"
    assert resumed.solver_calls == 4
    for operation in before[:-1]:
        assert calls.count(operation) == 1
    assert (
        calls.count(before[-1]) == 2
    )  # only the known incomplete quota turn is retried
    resumed.finish()
    assert resumed.usage["calls"] == len(calls)
    assert resumed.usage["internal_model_calls"] is None
    assert resumed.usage["cost_usd"] is None


def test_solver_receipt_rejects_changed_harness(tmp_path):
    from research_harness.contracts import write_tree
    from research_harness.runtime import Runtime
    from research_harness.state import ResumeMismatchError

    runtime = Runtime(journal_profile(), tmp_path, {"frozen": "source"})
    adapter = CravingAdapter(
        tmp_path, runtime, source_packet(Path(__file__).parents[1])
    )
    adapter.execute("checkpoint_restore", None, "same-run-id")
    changed = tmp_path / "changed-harness"
    tree = read_tree(PACKAGE / "base_harness")
    tree["systemprompt.md"] += "\nChanged instructions"
    write_tree(changed, tree)
    with pytest.raises(ResumeMismatchError):
        adapter.execute("checkpoint_restore", changed, "same-run-id")
    runtime.finish()


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Completed.\n{}\nDone."])
@pytest.mark.parametrize("make_symlink", [False, True])
@pytest.mark.parametrize("role_case", ["solver", "harvest", "harvest_empty", "text"])
def test_subscription_stream_attests_tools_before_sending_any_input(
    tmp_path, monkeypatch, wrapper, make_symlink, role_case
):
    import io
    import queue

    from research_harness import subscription

    sent = []
    lines = queue.Queue()
    answer = {
        "answer": "Literal 429 in candidate explanation is not a quota diagnostic",
        "files": {
            CANDIDATE: "def save(m): return m.export_state()\ndef restore(c,s): return c.from_state(s)\n"
        },
    }
    if role_case == "harvest":
        answer["files"] = {
            "systemprompt.md": "Research contract",
            "guardrails.md": "Keep scope",
        }
    elif role_case in {"harvest_empty", "text"}:
        answer["files"] = {}
    harvest = "context/harness" if role_case.startswith("harvest") else None
    role = "proposer" if harvest else "selector" if role_case == "text" else "solver"
    inputs = {
        "context/harness/systemprompt.md": "Research contract",
        "context/harness/guardrails.md": "Keep scope",
    }
    lines.put(
        json.dumps(
            {
                "event": "init",
                "init": {
                    "tools": [],
                    "model": "gemini-3.8-flash-high",
                    "agent": "h1-transport",
                },
            }
        ).encode()
        + b"\n"
    )

    class Input:
        def write(self, value):
            sent.append(json.loads(value))
            payload = json.loads(sent[-1]["message"]["content"])
            assert payload["harvest"] == harvest
            if harvest:
                assert set(payload["output"]["files"]) == {
                    "systemprompt.md",
                    "guardrails.md",
                }
                assert (
                    "KEEP_BASE still requires the complete files tree"
                    in payload["role_instructions"]
                )
                assert "files={} is invalid" in payload["role_instructions"]
            elif role_case == "text":
                assert payload["output"]["files"] == {}
            else:
                assert set(payload["output"]["files"]) == {CANDIDATE}
            lines.put(
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_index": 1,
                            "state": "DONE",
                            "step_type": "agent_response",
                            "text_delta": "429 is text",
                        },
                    }
                ).encode()
                + b"\n"
            )
            lines.put(
                json.dumps(
                    {
                        "event": "result",
                        "result": {
                            "status": "SUCCESS",
                            "response": wrapper.format(json.dumps(answer)),
                            "usage": {"input_tokens": 5, "output_tokens": 10},
                        },
                    }
                ).encode()
                + b"\n"
            )
            lines.put(None)

        def close(self):
            pass

    class Output:
        def readline(self, cap):
            return lines.get() or b""

    class Process:
        pid = 123456789  # A completed Popen still exposes its original process group.
        stdin = Input()
        stdout = Output()
        stderr = io.BytesIO()
        returncode = 0

        def wait(self, **kwargs):
            return 0

        def poll(self):
            return 0

    def launch(argv, **kwargs):
        assert not sent
        assert "--dangerously-skip-permissions" not in argv
        assert "--input-format" in argv and "--sandbox" in argv
        assert kwargs["env"] == {"PATH": "/usr/bin:/bin"}
        assert (
            tmp_path / "context/harness/systemprompt.md"
        ).read_text() == "Research contract"
        assert (
            tmp_path / "inputs/context/harness/systemprompt.md"
        ).read_text() == "Research contract"
        assert not (tmp_path / "context").is_symlink()
        if make_symlink:
            (tmp_path / "untrusted-link").symlink_to(
                "context", target_is_directory=True
            )
        return Process()

    monkeypatch.setattr(subscription.subprocess, "Popen", launch)
    runner = GeminiSubscriptionRunner(cli_bin="/bin/echo")

    def preflight():
        runner.identity = {"auth_verified": False}
        runner.environment = {"PATH": "/usr/bin:/bin"}
        return runner.identity

    monkeypatch.setattr(runner, "preflight", preflight)
    if make_symlink:
        # Preserve an existing transport failure; audit still records the symlink.
        error_type = ValueError if role_case == "harvest_empty" else RuntimeError
        error_text = (
            "Harvest response"
            if role_case == "harvest_empty"
            else "Workspace audit incomplete"
        )
        with pytest.raises(error_type, match=error_text):
            runner.run_turn(
                tmp_path,
                "Implement checkpoint",
                files=inputs,
                role=role,
                harvest=harvest,
            )
        assert (tmp_path / "untrusted-link").is_symlink()
        archive = json.loads(
            (tmp_path.parent / (tmp_path.name + "-archive.json")).read_text()
        )
        assert not archive["evidence_complete"]
        assert archive["files"]["untrusted-link"]["followed"] is False
        return
    if role_case == "harvest_empty":
        with pytest.raises(
            ValueError, match="Harvest response must include systemprompt.md"
        ):
            runner.run_turn(
                tmp_path, "KEEP_BASE", files=inputs, role=role, harvest=harvest
            )
        raw_events = list(
            map(json.loads, (tmp_path / "stream-events.jsonl").read_text().splitlines())
        )
        terminal = next(e for e in raw_events if e.get("event") == "transport_result")
        assert terminal["result"]["response"] == wrapper.format(json.dumps(answer))
        assert (
            tmp_path / "context/harness/systemprompt.md"
        ).read_text() == "Research contract"
        return
    result = runner.run_turn(
        tmp_path,
        "Implement checkpoint",
        files=inputs,
        role=role,
        harvest=harvest,
    )
    assert len(sent) == 1 and sent[0]["event"] == "user"
    assert result["_usage"]["cost_usd"] is None
    rendered = subscription.render_subscription_events(
        result["events"], obs_trunc=50000
    )
    assert "STEP 1 agent_response DONE" in rendered
    assert "429 is text" in rendered
    assert result["_usage"]["tokens"]["output_tokens"] == 10
    terminal = [
        e
        for e in map(json.loads, result["events"].splitlines())
        if e.get("event") == "transport_result"
    ]
    assert terminal[0]["result"]["response"] == wrapper.format(json.dumps(answer))


def test_subscription_rejects_wrong_model_init_before_curated_data(
    tmp_path, monkeypatch
):
    import io

    from research_harness import subscription

    sent = []

    class Process:
        pid = 123456789  # A completed Popen still exposes its original process group.
        stdin = SimpleNamespace(write=sent.append, close=lambda: None)
        stdout = io.BytesIO(
            json.dumps(
                {
                    "event": "init",
                    "init": {
                        "tools": ["run_command"],
                        "model": "gemini-other",
                        "agent": "h1-transport",
                    },
                }
            ).encode()
            + b"\n"
        )
        stderr = io.BytesIO()
        returncode = 0

        def wait(self, **kwargs):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(subscription.subprocess, "Popen", lambda *a, **k: Process())
    runner = GeminiSubscriptionRunner(cli_bin="/bin/echo")

    def preflight():
        runner.identity, runner.environment = {}, {}

    monkeypatch.setattr(runner, "preflight", preflight)
    with pytest.raises(ValueError, match="exact model"):
        runner.run_turn(tmp_path, "Never send this task to the wrong model")
    assert sent == []


def test_queue_blocks_science_dependency_without_operator_acceptance(tmp_path):
    from research_harness.queue import status

    plan = {
        "tasks": [
            {"id": "H2", "enabled": True, "dependencies": ["H1"]},
            {"id": "H6", "enabled": False, "dependencies": ["H2"]},
        ]
    }
    snapshot = status(tmp_path, plan, "frozen")
    assert [row["status"] for row in snapshot["tasks"]] == [
        "blocked_dependency",
        "blocked_manual",
    ]
    assert snapshot["budget"]["internal_model_calls"] is None


def test_worker_experiment_uses_fixed_container_argv_and_reports_coverage(
    tmp_path, monkeypatch
):
    from research_harness import worker

    # This unit fixture replaces the container subprocess; it is not isolation proof.
    workspace = tmp_path / "workspace"
    (workspace / "experiment").mkdir(parents=True)
    (workspace / "experiment/experiment.py").write_text(
        "# generated source is never imported by supervisor"
    )
    real_path = Path

    def local_path(name):
        if str(name) == "/workspace":
            return workspace
        if str(name).startswith("/tmp/experiment-"):
            return tmp_path / real_path(name).name
        return real_path(name)

    monkeypatch.setattr(worker, "Path", local_path)
    commands = []

    class Child:
        returncode = 0

        def wait(self, **kwargs):
            return 0

    def launch(argv, **kwargs):
        commands.append(argv)
        assert isinstance(argv, list) and argv[1:3] == ["-I", "-c"]
        assert "shell" not in kwargs
        assert "HOME" not in kwargs["env"] and "GEMINI_API_KEY" not in kwargs["env"]
        (workspace / "artifacts/metrics.json").write_text(
            json.dumps({"configuration_ids": ["fixed"]})
        )
        (workspace / "artifacts/observations.md").write_text(
            "Measured fixture observation"
        )
        return Child()

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    result = worker.experiment(
        {
            "timeout": 3,
            "outputs": ["artifacts/metrics.json", "artifacts/observations.md"],
            "configuration_ids": ["fixed"],
        }
    )
    assert result["execution_complete"]
    assert result["scientific_acceptance"] == "manual_required"
    assert len(commands) == 1


def test_research_two_rounds_library_stop_resume_audit_and_replay(
    tmp_path, monkeypatch
):
    from research_harness import audit
    from research_harness.queue import dependency_packet, run_round, stage_fingerprint
    from research_harness.runtime import Runtime
    from research_harness.state import ResumeMismatchError, RunStopped, atomic_json

    campaign = tmp_path / "campaign"
    stage = campaign / "tasks/H2"
    stage.mkdir(parents=True)
    original = Runtime._execute_once
    probes = []

    def fixture_transport(self, files, request, *, model_role, wall):
        if model_role is None:
            assert request["kind"] == "experiment"
            probes.append(request["operation_id"])
            return {
                "execution_complete": True,
                "files": {
                    "artifacts/metrics.json": '{"configuration_ids":["fixture"],"fixture_only":true}',
                    "artifacts/observations.md": "Fixture only",
                },
                "errors": [],
                "scientific_acceptance": "manual_required",
            }
        return original(self, files, request, model_role=model_role, wall=wall)

    monkeypatch.setattr(Runtime, "prepare", lambda self, **kwargs: None)
    monkeypatch.setattr(Runtime, "_execute_once", fixture_transport)
    profile = dict(
        journal_profile(),
        wall_seconds=1200,
        max_calls=30,
        stop_file=str(campaign / "STOP"),
    )
    task = {
        "id": "H2",
        "dependencies": ["H1"],
        "mission": "Tiny deterministic engineering fixture only",
        "max_configurations": 2,
        "max_feature_sets": 2,
        "experiment_outputs": ["artifacts/metrics.json", "artifacts/observations.md"],
    }
    sources = {"fixture.py": "# fixture data"}
    manifest = {"schema": "craving-research-manifest/v1", "fixture": True}
    first = stage / "round-001"
    receipt = run_round(first, task, sources, profile, manifest)
    assert receipt["usage"]["closed"]
    assert len(probes) == 4
    second = stage / "round-002"
    (campaign / "STOP").write_text("stop")
    with pytest.raises(RunStopped):
        run_round(second, task, sources, profile, manifest, prior=first)
    assert len(probes) == 4
    (campaign / "STOP").unlink()
    run_round(second, task, sources, profile, manifest, prior=first)
    assert len(probes) == 8
    journal = json.loads((second / "run-state.json").read_text())
    assert any(op.startswith("selector/") for op in journal["operations"])
    packet_paths = list((second / "operation-inputs").glob("*.json"))
    packets = [json.loads(p.read_text()) for p in packet_paths]
    assert any(
        "round-001"
        in p["files"].get("campaign/hypothesis-and-negative-ledger.json", "")
        for p in packets
    )
    assert len(list((second / "library-output").rglob("*.md"))) >= len(
        list((first / "library-output").rglob("*.md"))
    )
    index = audit.index(campaign)
    assert len(index["rounds"]) == 2
    operation = next(
        name for name in journal["operations"] if name.startswith("probe/")
    )
    replay_output = tmp_path / "replayed"
    audit.replay(second, operation, replay_output)
    assert json.loads((replay_output / "replay-report.json").read_text())[
        "exact_output_match"
    ]
    atomic_json(
        stage / "stage.json",
        {
            "status": "awaiting_scientific_review",
            "artifact_sha256": stage_fingerprint(stage),
        },
    )
    atomic_json(
        campaign / "acceptance/H2.json",
        {
            "schema": "craving-research-acceptance/v1",
            "task": "H2",
            "manifest_sha256": "frozen",
            "decision": "accepted",
            "operator": "fixture-human",
            "evidence": "fixture review",
            "artifact_sha256": stage_fingerprint(stage),
        },
    )
    dependencies = dependency_packet(campaign, {"dependencies": ["H2"]}, "frozen")
    assert dependencies["H2"]["rounds"][0]["files"]["chosen/experiment/experiment.py"]
    bad = packet_paths[0]
    bad.write_text("{}")
    with pytest.raises(ResumeMismatchError):
        audit.index(campaign)


def test_role_json_parser_is_scoped_and_rejects_ambiguity():
    from research_harness.subscription import parse_role_response

    value = {"answer": "literal { braces }", "files": {}}
    assert parse_role_response("Text\n```json\n" + json.dumps(value) + "\n```") == value
    for text in ("{} {}", "{broken}", "[]", '{"answer": 1} [1]'):
        with pytest.raises(ValueError):
            parse_role_response(text)
    with pytest.raises(json.JSONDecodeError):
        json.loads('Text\n{"answer":"x","files":{}}')


def test_upstream_bootstrap_preserves_environment_without_hooks():
    env = dict(os.environ)
    for key in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        env.pop(key, None)
    env["HARNESS2_VERTEX_PROJECT"] = "synthetic-project-for-regression"
    env["PYTHONPATH"] = str(PACKAGE.parent) + os.pathsep + os.environ["HARNESS2_SOURCE"]
    code = """import os, builtins, json
from pathlib import Path
before = dict(os.environ)
loads, importer, rglob = json.loads, builtins.__import__, Path.rglob
from research_harness.adapter import CravingAdapter
assert dict(os.environ) == before
assert (loads, importer, rglob) == (json.loads, builtins.__import__, Path.rglob)
"""
    subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True
    )


def test_h1_status_separates_queue_and_interrupted_active_runs(tmp_path):
    import fcntl

    from research_harness.status import operator_status

    runs = tmp_path / "runs"
    run = runs / "saved-h1"
    run.mkdir(parents=True)
    write_json(run / "manifest.json", {"evaluator": evaluator_manifest()})
    write_json(
        run / "cycle_checkpoint.json", {"status": "in_progress", "method": "harness"}
    )
    write_json(
        run / "run-state.json",
        {
            "attempts": 1,
            "next_step": "solver/base1",
            "operations": {"solver/base1": {"status": "running", "attempts": 1}},
        },
    )
    queue_path = tmp_path / "queue.json"
    write_json(queue_path, {"tasks": []})
    before = {str(p): p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    snapshot = operator_status(runs / "campaign", queue_path)
    assert snapshot["queue_status"] == "not_initialized"
    assert snapshot["h1"][0]["status"] == "interrupted"
    assert snapshot["h1"][0]["next_step"] == "solver/base1"
    assert {str(p): p.read_bytes() for p in runs.rglob("*") if p.is_file()} == before
    assert not (runs / "campaign").exists()
    with (run / ".supervisor.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = operator_status(runs / "campaign", queue_path)
        assert active["h1"][0]["status"] == "running"
        assert active["h1"][0]["supervisor_active"]
    write_json(
        run / "run-report-1.json",
        {"status": "inconclusive", "error": "retained failure"},
    )
    failed = operator_status(runs / "campaign", queue_path)["h1"][0]
    assert failed["status"] == "inconclusive"
    assert failed["error"] == "retained failure"


def test_h1_status_completed_and_judged_revision_evidence(tmp_path):
    from research_harness.status import h1_run

    frozen = {"evaluator": evaluator_manifest()}
    selection = {"chosen_label": "plain"}
    report = {
        "status": "selected_not_judged",
        "command": "run",
        "error": None,
        "manifest_sha256": digest(json_bytes(frozen)),
        "usage": {"closed": True, "enforced": True},
    }
    write_json(tmp_path / "manifest.json", frozen)
    write_json(
        tmp_path / "cycle_checkpoint.json", {"status": "completed", "method": "plain"}
    )
    write_json(tmp_path / "selection.json", selection)
    write_json(tmp_path / "run-report-1.json", report)
    write_json(
        tmp_path / "run-complete.json",
        {
            "schema": "craving-harness-complete/v1",
            "report": "run-report-1.json",
            "manifest_sha256": digest(json_bytes(frozen)),
            "selection_sha256": digest(json_bytes(selection)),
            "report_sha256": digest(json_bytes(report)),
        },
    )
    assert h1_run(tmp_path)["status"] == "selected_not_judged"
    write_json(
        tmp_path / "judge-report-1.json",
        {
            "status": "graded",
            "error": None,
            "manifest_sha256": digest(json_bytes(frozen)),
            "result": {"score": {"all_pass": True}},
        },
    )
    snapshot = h1_run(tmp_path)
    assert snapshot["status"] == "graded"
    assert snapshot["evaluator_revision_current"]
    assert snapshot["judge"]["score"]["all_pass"]
    (tmp_path / "selection.json").write_text("{}")
    assert h1_run(tmp_path)["status"] == "evidence_incomplete"


def test_h1_interrupt_is_saved_as_interrupted_not_success(tmp_path, monkeypatch):
    from research_harness import cli, runtime

    statuses = []

    class InterruptedRuntime:
        def __init__(self, profile, root, frozen, phase):
            self.profile, self.started, self.audit = profile, time.monotonic(), []
            self.usage = None
            self.journal = SimpleNamespace(data={"operations": {}, "next_step": None})

        def prepare(self, **kwargs):
            raise KeyboardInterrupt

        def finish(self, status):
            statuses.append(status)
            self.usage = {"closed": True, "enforced": True, "status": status}

    monkeypatch.setattr(runtime, "Runtime", InterruptedRuntime)
    monkeypatch.setattr(cli, "upstream", lambda path: None)
    monkeypatch.setattr(cli, "manifest", lambda *args: {"frozen": True})
    run = tmp_path / "interrupted-h1"
    args = SimpleNamespace(
        command="run",
        method="plain",
        resume=False,
        repo=PACKAGE.parent,
        upstream=tmp_path / "vendor",
        output=run,
        profile=PACKAGE / "run-profile.json",
    )
    with pytest.raises(RuntimeError, match="Interrupted"):
        cli.execute(args)
    assert statuses == ["interrupted"]
    assert (
        json.loads((run / "cycle_checkpoint.json").read_text())["status"]
        == "interrupted"
    )
    report = json.loads(next(run.glob("run-report-*.json")).read_text())
    assert report["status"] == "interrupted"
    assert not (run / "run-complete.json").exists()
    assert not (run / "selection.json").exists()


@pytest.mark.parametrize("missing", ["run-complete.json", "selection.json"])
@pytest.mark.parametrize("claimed", ["selected_not_judged", "graded"])
def test_h1_status_never_claims_success_without_receipt(tmp_path, missing, claimed):
    from research_harness.status import h1_run

    frozen = {"evaluator": evaluator_manifest()}
    write_json(tmp_path / "manifest.json", frozen)
    write_json(
        tmp_path / "cycle_checkpoint.json", {"status": "completed", "method": "plain"}
    )
    write_json(tmp_path / "run-report-1.json", {"status": "selected_not_judged"})
    if missing != "run-complete.json":
        write_json(tmp_path / "run-complete.json", {"report": "run-report-1.json"})
    if missing != "selection.json":
        write_json(tmp_path / "selection.json", {})
    if claimed == "graded":
        write_json(
            tmp_path / "judge-report-1.json",
            {"status": "graded", "manifest_sha256": digest(json_bytes(frozen))},
        )
    assert h1_run(tmp_path)["status"] == "evidence_incomplete"


@pytest.mark.parametrize(
    "file,value",
    [
        ("manifest.json", []),
        ("cycle_checkpoint.json", []),
        ("run-state.json", []),
        ("run-state.json", {"operations": []}),
        ("run-state.json", {"operations": {"solver/base": []}}),
        ("run-report-1.json", []),
        ("judge-report-1.json", []),
    ],
)
def test_h1_status_corrupt_shape_is_local_to_one_run(tmp_path, file, value):
    from research_harness.status import h1_inventory

    runs = tmp_path / "runs"
    bad, good = runs / "bad-h1", runs / "good-h1"
    for run in (bad, good):
        write_json(run / "manifest.json", {"evaluator": evaluator_manifest()})
        write_json(
            run / "cycle_checkpoint.json", {"status": "in_progress", "method": "plain"}
        )
        write_json(run / "run-state.json", {"operations": {}})
    write_json(bad / file, value)
    result = h1_inventory(runs)
    assert [(r["name"], r["status"]) for r in result] == [
        ("bad-h1", "evidence_incomplete"),
        ("good-h1", "interrupted"),
    ]


def test_research_solver_contract_separates_sources_from_measured_artifacts(tmp_path):
    from research_harness.research import EDITABLE, ResearchAdapter

    task = {
        "id": "ENGINEERING",
        "mission": "One deterministic engineering fixture only",
        "max_configurations": 1,
        "max_feature_sets": 1,
        "experiment_outputs": ["artifacts/metrics.json", "artifacts/observations.md"],
    }
    adapter = ResearchAdapter(
        tmp_path, SimpleNamespace(profile=journal_profile()), {}, task
    )
    prompt = adapter.task_prompt("ENGINEERING")
    contract, end = json.JSONDecoder().raw_decode(prompt)
    assert contract["phase"] == "source_generation_before_probe"
    assert set(contract["editable_outputs"]) == EDITABLE
    assert set(contract["experiment_outputs"]).isdisjoint(EDITABLE)
    assert "not_executed" in contract["report"]
    assert "No numerical results" in contract["report"]
    assert (
        "do not create" not in contract["execution"]["native_role"]
    )  # research tools remain available
    assert (
        "File/shell/web tools remain available" in contract["execution"]["native_role"]
    )
    assert (
        "Project Python dependencies are not installed in this native shell"
        in contract["execution"]["native_role"]
    )
    assert (
        "After receiving the structured response" in contract["execution"]["supervisor"]
    )
    assert (
        "model must not create or fill these artifacts"
        in contract["execution"]["program"]
    )
    assert "not a claim that you read primary papers" in contract["source_provenance"]
    assert "Return immediately" in contract["response"]
    assert "One deterministic engineering fixture only" in prompt[end:]


def test_research_queue_has_consistent_pre_probe_and_source_context():
    plan = json.loads((PACKAGE / "research-queue.example.json").read_text())
    assert "научные кампании H2–H5 не запускаются" not in plan["mandate"]
    assert "solver сначала возвращает только программу" in plan["mandate"]
    for task in plan["tasks"]:
        if task["id"] in {"H6", "H7"}:
            continue
        assert "web-инструмент недоступны" not in task["mission"]
        assert "solver не пишет эти измерения до исполнения" in task["mission"]
        assert "только фактически полученный текст" in task["mission"]
        assert all(
            source["status"] == "seed_summary_not_parameter_validation"
            for source in task["literature"]["sources"]
        )
        assert all(
            "модель не имеет свободного web-доступа" not in gap
            for gap in task["literature"]["gaps"]
        )
