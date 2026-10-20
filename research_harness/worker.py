"""Worker entrypoint в OCI image. Не запускать непосредственно на host.

Один JSON request stdin / один bounded JSON response stdout. Это недоверенный
output процесса; только supervisor вычисляет verdict. Candidate не импортируется
в процессе supervisor или reference worker.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

LIMIT = 2 * 1024 * 1024


def text_file(path: Path, cap: int = 128 * 1024) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError("Artifact должен быть обычным ограниченным файлом")
    return path.read_text()


def collect(root: Path) -> dict:
    result = {}
    if root.is_symlink():
        raise ValueError("Symlink output root forbidden")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink в output запрещён")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = text_file(path)
    if len(json.dumps(result).encode()) > LIMIT:
        raise ValueError("Output превышает лимит")
    return result


def replay(
    request: dict, *, candidate_path: str = "/workspace/candidate/checkpoint.py"
) -> dict:
    from datetime import datetime, timedelta

    import numpy as np

    from research_v12.models import HierarchicalRidgeMemory, TunedScalarMemory

    kind = request["kind"]
    adapter = None
    if kind in ("candidate_save", "candidate_restore"):
        spec = importlib.util.spec_from_file_location(
            "candidate_checkpoint", candidate_path
        )
        if spec is None or spec.loader is None:
            raise ValueError("Candidate module cannot be loaded")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
    fixture = request["fixture"]
    start = datetime.fromisoformat(fixture["start"])
    result = {}
    for cls in (TunedScalarMemory, HierarchicalRidgeMemory):
        memory = None if kind == "candidate_restore" else cls()
        trace = []

        def predict(memory, day, hour=0, cls=cls):
            args = {
                "event_id": f"e{day}",
                "at": start + timedelta(days=day, hours=hour),
                "p0": fixture["p0"],
            }
            if cls is HierarchicalRidgeMemory:
                args["z"] = np.array(fixture["z"])
            snap = memory.predict(**args)
            return float(snap["p_final"])

        if kind == "candidate_restore":
            # Новый контейнер: live memory, module globals и /tmp save worker отсутствуют.
            saved = request["checkpoints"][cls.__name__]
            if adapter is None:
                raise ValueError("Restore adapter missing")
            memory = adapter.restore(cls, saved)
        else:
            assert memory is not None
            for day in range(fixture["initial_pending"]):
                p = predict(memory, day)
                trace.append(
                    {"phase": "predict", "p": p, "state": memory.export_state()}
                )
            assert memory is not None
            saved = memory.export_state() if adapter is None else adapter.save(memory)
        if not isinstance(saved, str) or len(saved.encode()) > 16384:
            raise ValueError("Checkpoint превышает 16 KiB или не является строкой")
        if kind == "candidate_save":
            result[cls.__name__] = {"trace": trace, "checkpoint": saved}
            continue
        assert memory is not None
        trace.append({"phase": "restore", "state": memory.export_state()})
        for day in fixture["days"]:
            memory.observe(
                f"e{day - 3}",
                start + timedelta(days=day),
                day % 2,
                confidence=fixture["confidence"],
            )
            trace.append({"phase": "observe", "state": memory.export_state()})
            p = predict(memory, day, 1)
            trace.append({"phase": "predict", "p": p, "state": memory.export_state()})
        result[cls.__name__] = {"trace": trace, "checkpoint": saved}
    return result


def experiment(request):
    """Generated Python executes only here, inside the existing networkless OCI."""
    work = Path("/workspace")
    program = work / "experiment/experiment.py"
    text_file(program)
    outputs = request["outputs"]
    if not isinstance(outputs, list) or len(outputs) > 8:
        raise ValueError("Finite named experiment outputs required")
    for name in outputs:
        path = Path(name)
        if (
            not name.startswith("artifacts/")
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
        ):
            raise ValueError("Experiment output outside artifacts allowlist")
    (work / "artifacts").mkdir(exist_ok=True)
    # Fixed argv, no shell. -I ignores user Python startup/environment; only curated
    # workspace is added for imports by this trusted launcher.
    launcher = "import runpy,sys;sys.path.insert(0,'/workspace');runpy.run_path(sys.argv[1],run_name='__main__')"
    argv = [sys.executable, "-I", "-c", launcher, str(program)]
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    timed_out = False
    with (
        Path("/tmp/experiment-stdout").open("wb") as stdout,
        Path("/tmp/experiment-stderr").open("wb") as stderr,
    ):
        proc = subprocess.Popen(
            argv,
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=request["timeout"])
        except subprocess.TimeoutExpired:
            import signal

            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True
    files = {}
    errors = []
    for name in outputs:
        try:
            files[name] = text_file(work / name)
        except (OSError, ValueError, UnicodeError):
            errors.append("Missing/invalid named artifact: " + name)
    try:
        metrics = json.loads(files.get("artifacts/metrics.json", ""))
        reported = metrics.get("configuration_ids")
        expected = request["configuration_ids"]
        if (
            not isinstance(reported, list)
            or len(reported) != len(set(reported))
            or set(reported) != set(expected)
        ):
            errors.append(
                "Reported configuration coverage differs from preregistration"
            )
    except (ValueError, TypeError, AttributeError):
        errors.append(
            "Metrics must report exact configuration_ids; this is not semantic/scientific proof"
        )
    # Reject extra files or symlinks: bounded, explicitly named UTF-8 outputs only.
    try:
        actual = collect(work / "artifacts")
        if set(actual) != {name.removeprefix("artifacts/") for name in outputs}:
            errors.append("Unexpected artifact paths")
    except (OSError, ValueError, UnicodeError):
        errors.append("Invalid artifact tree")
    logs = {}
    for name in ("stdout", "stderr"):
        try:
            logs[name] = text_file(Path("/tmp/experiment-" + name), 64 * 1024)
        except (OSError, ValueError, UnicodeError):
            logs[name] = "[output exceeded UTF-8/64KiB cap]"
            errors.append(name + " limit exceeded")
    return {
        "execution_complete": proc.returncode == 0 and not timed_out and not errors,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "files": files,
        "errors": errors,
        "logs": logs,
        "scientific_acceptance": "manual_required",
    }


def verify_limits(limits):
    if os.getuid() != 65534:
        raise ValueError("Worker must run as uid 65534")
    root = Path("/sys/fs/cgroup")
    memory = int((root / "memory.max").read_text().strip())
    pids = int((root / "pids.max").read_text().strip())
    quota, period = (root / "cpu.max").read_text().split()
    cpus = int(quota) / int(period)
    if (
        memory > limits["memory_mb"] * 1024 * 1024
        or pids > limits["pids"]
        or cpus > limits["cpus"] + 0.001
    ):
        raise ValueError("Kernel resource limits exceed profile")
    return {
        "uid": os.getuid(),
        "memory_max": memory,
        "pids_max": pids,
        "cpu_max": [quota, period],
    }


def main():
    request = json.loads(sys.stdin.buffer.read(LIMIT + 1))
    applied_limits = verify_limits(request["limits"])
    source = Path("/input")
    work = Path("/workspace")
    # Все остальные curated input paths смонтированы runtime непосредственно
    # readonly по /workspace/<rel>; не создаём writable copies источников/evidence.
    if request["kind"] not in {
        "reference",
        "candidate_save",
        "candidate_restore",
        "experiment",
    }:
        raise ValueError("Unknown worker kind; model/API transports never run in OCI")
    editable = ("candidate",) if request["kind"].startswith("candidate_") else ()
    for name in editable:
        editable_root = source / name
        if editable_root.is_dir():
            for original in editable_root.rglob("*"):
                if original.is_file():
                    target = work / name / original.relative_to(editable_root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original.read_bytes())
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(work))
    os.chdir(work)
    # Candidate prints не становятся доверенным JSON; результат всё равно недоверенный.
    with contextlib.redirect_stdout(io.StringIO()):
        result = (
            experiment(request) if request["kind"] == "experiment" else replay(request)
        )
    result["_kernel_limits"] = applied_limits
    encoded = json.dumps(result, allow_nan=False)
    if len(encoded.encode()) > LIMIT:
        raise ValueError("Worker response too large")
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
