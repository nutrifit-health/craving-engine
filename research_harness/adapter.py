"""BenchmarkAdapter и изолированный proposer для закреплённого Harness² core."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .upstream import load_core_config

load_core_config()

from harness2.core.proposer import ProposerError, Reply
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.substrate.opencode import OpencodeSubstrate

from .contracts import (
    CANDIDATE,
    PACKAGE,
    TASK,
    digest,
    fixture,
    json_bytes,
    read_regular,
    read_tree,
    relative,
    write_tree,
)
from .state import ResumeMismatchError
from .subscription import render_subscription_events

TASK_PROMPT = """Implement candidate/checkpoint.py: save(memory) -> JSON string and
restore(memory_class, serialized) -> memory_class instance. Preserve full V12
configuration, clocks, pending decisions and subsequent delayed-feedback learning.
Use memory.export_state() and memory_class.from_state(serialized).
Do not fit, tune, generate data or change any other source.
Return one JSON object: answer is your concise final explanation, files contains
the complete candidate/checkpoint.py implementation. This structured response is
the sole deliverable; do not create output/answer.md. Once the two wrapper calls
are established, finish immediately. Do not run model-equivalence experiments or
repeat file inventories: independent correctness probes run in the OCI judge.
"""


def safe_id(value: str) -> str:
    return digest(value.encode())[:24]


class IsolatedProposer:
    """Duck-typed core proposer: никогда не вызывает host Proposer.call."""

    def __init__(self, adapter, role):
        self.adapter, self.role = adapter, role
        self.spec = adapter.spec.channels[role]

    def call(
        self,
        mission,
        context_dir,
        label,
        *,
        system="",
        timeout=None,
        max_attempts=None,
        harvest=None,
    ):
        started = time.monotonic()
        files = (
            {
                "context/" + name: body
                for name, body in read_tree(Path(context_dir)).items()
            }
            if context_dir
            else {}
        )
        try:
            result = self.adapter.runtime.execute(
                files,
                {
                    "kind": "agent",
                    "role": self.role,
                    "mission": system + "\n" + mission,
                    "operation_id": self.role + "/" + label,
                    "harvest": harvest,
                },
                model_role=self.role,
            )
            if (
                not isinstance(result.get("answer"), str)
                or not result["answer"].strip()
            ):
                raise ValueError("Нет ответа proposer")
            workspace = None
            if harvest:
                if harvest != "context/harness":
                    raise ValueError("Неизвестный harvest path")
                self.adapter.validate_tree(result["files"])
                workspace = (
                    self.adapter.root / "harvest" / safe_id(self.role + "/" + label)
                )
                if workspace.exists() and read_tree(workspace) != result["files"]:
                    raise ResumeMismatchError("Harvest изменён после завершения роли")
                write_tree(workspace, result["files"])
            # Upstream numeric aggregation is not a billing report; real usage remains unknown in supervisor ledger.
            return Reply(
                label,
                result["answer"],
                True,
                wall_s=time.monotonic() - started,
                attempts=1,
                channel="gemini_subscription",
                workspace=workspace,
                cost_usd=0.0,
            )
        except Exception as exc:
            raise ProposerError(str(exc), attempts=1) from exc


class CravingAdapter:
    def __init__(self, root, runtime, sources):
        self.root, self.runtime, self.sources = Path(root), runtime, sources
        self.substrate = OpencodeSubstrate(launcher=self._launch)
        channels = {
            role: ChannelSpec(
                "opencode",
                runtime.profile["roles"][role]["model"],
                effort=runtime.profile["roles"][role].get("effort", ""),
                steps=int(runtime.profile["roles"][role]["steps"]),
                timeout=int(runtime.profile["call_timeout"]),
            )
            for role in ("proposer", "selector")
        }
        self.spec = BenchSpec(
            "craving",
            self.root / "core",
            "workspace_agents_md",
            BenchPrompts(
                work="Maintain complete personal state across restart and delayed replies.",
                domain_word="personal memory",
                deliverable_word="checkpoint module",
            ),
            channels,
            domain_word="personal memory",
            allow_subagents=False,
        )
        self.committed = None
        self.completed_executions = []

    def tasks(self, split="all"):
        if split not in ("all", "development"):
            raise ValueError("H1 доступен только development")
        return [TASK]

    def domain_of(self, task_id):
        self._task(task_id)
        return "checkpoint"

    def _task(self, task_id):
        if task_id != TASK:
            raise ValueError("H1 имеет единственное заранее фиксированное задание")

    def task_prompt(self, task_id):
        self._task(task_id)
        return TASK_PROMPT

    def task_files(self, task_id):
        self._task(task_id)
        return self.root / "task_source"

    def base_harness(self):
        return PACKAGE / "base_harness"

    def validate_tree(self, tree):
        # Harness prompts/helpers — недоверенные данные. Ни один из них не запускается host.
        if not isinstance(tree, dict) or len(json_bytes(tree)) > 256 * 1024:
            raise ValueError("Harness превышает лимит")
        for name, body in tree.items():
            relative(name)
            if not isinstance(body, str) or len(body.encode()) > 128 * 1024:
                raise ValueError("Недопустимый harness file")
            # Middleware/plugins и nested agents не нужны для первого пилота.
            if not (
                name in self.substrate.surface().always_on
                or name == "scripts/README.md"
                or name == "skills/README.md"
                or name == "plugin/README.md"
                or name == "sub_agents/README.md"
                or re.fullmatch(r"scripts/[a-z0-9_]+\.py", name)
            ):
                raise ValueError(f"H1 harness path вне конечной поверхности: {name}")
        if "systemprompt.md" not in tree:
            raise ValueError("Harness должен сохранять systemprompt.md")

    def execute(self, task_id, harness_dir, run_id, *, timeout=None):
        self._task(task_id)
        tree = read_tree(Path(harness_dir or self.base_harness()))
        self.validate_tree(tree)
        # Реальный supplied launcher через стандартные substrate Install/RunRequest.
        install = self.substrate.materialise(
            Path(harness_dir or self.base_harness()), self.spec, {}
        )
        try:
            return self.substrate.execute(
                task_id, install, [run_id], int(self.runtime.profile["call_timeout"])
            )[0]
        finally:
            import shutil

            shutil.rmtree(install.staging_dir)

    def _launch(self, request, install):
        self._task(request.task_ref)
        if len(request.run_ids) != 1:
            raise ValueError("H1 launcher не батчит solver")
        run_id = request.run_ids[0]
        tree = read_tree(install.staging_dir)
        self.validate_tree(tree)
        files = dict(self.sources)
        files[CANDIDATE] = read_regular(PACKAGE / CANDIDATE)
        files.update({"harness/" + name: body for name, body in tree.items()})
        # workspace_agents_md: always-on текст устанавливается в AGENTS.md;
        # helpers остаются данными внутри isolated workspace, никогда не host execution.
        files["AGENTS.md"] = "\n\n".join(
            tree.get(name, "") for name in self.substrate.surface().always_on
        )
        for name, body in tree.items():
            if name.startswith("scripts/"):
                files[name] = body
        request = {
            "kind": "agent",
            "role": "solver",
            "mission": TASK_PROMPT,
            "operation_id": "solver/" + run_id,
        }
        provenance = digest(
            json_bytes(
                {
                    "run_id": run_id,
                    "files": files,
                    "request": request,
                    "profile": self.runtime.profile,
                    "manifest": getattr(self.runtime, "manifest", {}),
                }
            )
        )
        destination = self.root / "rollouts" / safe_id(run_id)
        if destination.exists():
            receipt = json.loads(read_regular(destination / "receipt.json"))
            if (
                not self.delivered(destination)
                or receipt.get("input_sha256") != provenance
                or receipt.get("run_id") != run_id
            ):
                raise ResumeMismatchError(
                    "Rollout/source/config/harness provenance mismatch: " + run_id
                )
            if not any(
                entry["run_id"] == run_id for entry in self.completed_executions
            ):
                self.completed_executions.append(
                    {"run_id": run_id, "run_dir": str(destination.resolve())}
                )
            return [destination]
        output = self.runtime.execute(files, request, model_role="solver")
        if set(output.get("files", {})) != {CANDIDATE}:
            raise ValueError("Solver не вернул единственный allowlisted patch")
        staging = (
            self.root / "rollouts" / (safe_id(run_id) + ".tmp-" + uuid.uuid4().hex)
        )
        staging.mkdir(parents=True, exist_ok=False)
        write_tree(staging, output["files"])
        if not isinstance(output.get("answer"), str) or not isinstance(
            output.get("events"), str
        ):
            raise TypeError("Неполная solver trajectory")
        (staging / "answer.md").write_text(output["answer"])
        (staging / "events.jsonl").write_text(output["events"])
        (staging / "receipt.json").write_text(
            json.dumps(
                {
                    "schema": "craving-rollout-receipt/v2",
                    "run_id": run_id,
                    "input_sha256": provenance,
                    "candidate_sha256": digest(
                        read_regular(staging / CANDIDATE).encode()
                    ),
                    "answer_sha256": digest(output["answer"].encode()),
                    "events_sha256": digest(output["events"].encode()),
                }
            )
        )
        staging.replace(destination)
        self.completed_executions.append(
            {"run_id": run_id, "run_dir": str(destination.resolve())}
        )
        return [destination]

    def delivered(self, run_dir):
        try:
            body = read_regular(Path(run_dir) / CANDIDATE)
            receipt = json.loads(read_regular(Path(run_dir) / "receipt.json"))
            return (
                bool(body.strip())
                and receipt.get("schema") == "craving-rollout-receipt/v2"
                and isinstance(receipt.get("input_sha256"), str)
                and digest(body.encode()) == receipt["candidate_sha256"]
                and digest(read_regular(Path(run_dir) / "answer.md").encode())
                == receipt["answer_sha256"]
                and digest(
                    read_regular(
                        Path(run_dir) / "events.jsonl", 2 * 1024 * 1024
                    ).encode()
                )
                == receipt["events_sha256"]
            )
        except (OSError, ValueError, KeyError):
            return False

    def trajectory(self, run_dir, *, obs_trunc):
        return render_subscription_events(
            read_regular(Path(run_dir) / "events.jsonl", 2 * 1024 * 1024),
            obs_trunc=obs_trunc,
        )

    def deliverable_files(self, run_dir):
        return {CANDIDATE: read_regular(Path(run_dir) / CANDIDATE)}

    def deliverable_text(self, run_dir):
        return read_regular(Path(run_dir) / CANDIDATE)

    def score(self, run_dir, task_id):
        self._task(task_id)
        if (
            not self.committed
            or str(Path(run_dir).resolve()) != self.committed["chosen_run_dir"]
        ):
            raise ValueError("Judge разрешён только после committed selection")
        if not self.delivered(run_dir):
            return None
        candidate = read_regular(Path(run_dir) / CANDIDATE)
        if digest(candidate.encode()) != self.committed["candidate_sha256"]:
            raise ValueError("Candidate изменён после selection")
        # Три независимых контейнера без сети: reference, save и fresh restore.
        expected = self.runtime.execute(
            self.sources, {"kind": "reference", "fixture": fixture()}
        )
        candidate_sources = dict(self.sources, **{CANDIDATE: candidate})
        saved = self.runtime.execute(
            candidate_sources, {"kind": "candidate_save", "fixture": fixture()}
        )
        if not isinstance(saved, dict) or set(saved) != set(fixture()["models"]):
            raise ValueError("Неполный save evidence")
        checkpoints = {}
        for name in fixture()["models"]:
            record = saved[name]
            if not isinstance(record, dict) or set(record) != {"trace", "checkpoint"}:
                raise ValueError("Неполный save artifact")
            value = record["checkpoint"]
            if not isinstance(value, str) or len(value.encode()) > 16384:
                raise ValueError("Неверный serialized checkpoint")
            checkpoints[name] = value
        # Только checkpoint strings и fixed fixture пересекают restart boundary.
        restored = self.runtime.execute(
            candidate_sources,
            {
                "kind": "candidate_restore",
                "fixture": fixture(),
                "checkpoints": checkpoints,
            },
        )
        if not isinstance(restored, dict) or set(restored) != set(checkpoints):
            raise ValueError("Неполный restore evidence")
        observed = {}
        for name, checkpoint in checkpoints.items():
            record = restored[name]
            if (
                not isinstance(record, dict)
                or set(record) != {"trace", "checkpoint"}
                or record["checkpoint"] != checkpoint
                or not isinstance(record["trace"], list)
                or not isinstance(saved[name]["trace"], list)
            ):
                raise ValueError("Неверный restore response")
            observed[name] = {
                "checkpoint": checkpoint,
                "trace": saved[name]["trace"] + record["trace"],
            }
        return compare(expected, observed)


def compare(expected, observed):
    """Verdict только из supervisor. Ошибка/недостаток evidence не нулевой score."""
    models = fixture()["models"]
    if not isinstance(observed, dict) or set(observed) != set(models):
        raise ValueError("Неполный candidate response")
    criteria = {}
    peaks = {}
    for name in models:
        ref, got = expected[name], observed[name]
        if not isinstance(got, dict) or set(got) != {"trace", "checkpoint"}:
            raise ValueError("Неполный replay evidence")
        if not isinstance(got["trace"], list) or len(got["trace"]) != len(ref["trace"]):
            raise ValueError("Отсутствуют операции replay")
        criteria[name + ":checkpoint"] = got["checkpoint"] == ref["checkpoint"]
        criteria[name + ":continuity"] = got["trace"] == ref["trace"]
        sizes = []
        for row in got["trace"]:
            if not isinstance(row, dict) or not isinstance(row.get("state"), str):
                raise TypeError("Отсутствует полное состояние")
            sizes.append(len(row["state"].encode()))
        peaks[name] = max(sizes)
        criteria[name + ":bytes"] = peaks[name] <= 16384
    return {
        "n_passed": sum(criteria.values()),
        "n_criteria": len(criteria),
        "all_pass": all(criteria.values()),
        "criteria": criteria,
        "peak_bytes": peaks,
        "scope": "development_checkpoint_continuity_only",
        "future_event_quality": "not_evaluated",
    }
