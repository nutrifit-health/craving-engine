"""Research task adapter: pinned Harness² Propose–Probe–Compose, same OCI worker."""

from __future__ import annotations

import fcntl
import json
import re
import uuid
from dataclasses import replace
from pathlib import Path

from .upstream import load_core_config

load_core_config()

from harness2.core.spec import BenchPrompts

from .adapter import CravingAdapter, safe_id
from .contracts import (
    PACKAGE,
    digest,
    json_bytes,
    read_regular,
    read_tree,
    relative,
    write_tree,
)
from .state import ResumeMismatchError, atomic_json
from .microsteps import MicrostepInterrupted, repair_hypothesis
from .subscription import redact

EDITABLE = {"experiment/experiment.py", "report.md", "hypothesis.json"}


class ResearchAdapter(CravingAdapter):
    def __init__(self, root, runtime, sources, task, *, base=None):
        self.task = task
        self.research_base = Path(base) if base else PACKAGE / "research_base_harness"
        super().__init__(root, runtime, sources)
        self.spec = replace(
            self.spec,
            name="craving-research",
            prompts=BenchPrompts(
                work="Develop a reproducible bounded research experiment from explicit supplied evidence.",
                domain_word="research question",
                deliverable_word="research program",
            ),
            domain_word="research question",
        )

    def _task(self, task_id):
        if task_id != self.task["id"]:
            raise ValueError("Unknown research task")

    def tasks(self, split="all"):
        if split not in ("all", "development"):
            raise ValueError("Development research only")
        return [self.task["id"]]

    def domain_of(self, task_id):
        self._task(task_id)
        return "development-research"

    def task_prompt(self, task_id):
        self._task(task_id)
        return (
            json.dumps(
                {
                    "phase": "source_generation_before_probe",
                    "editable_outputs": sorted(EDITABLE),
                    "experiment_outputs": self.task["experiment_outputs"],
                    "response": "Return exactly one JSON object with answer and files containing exactly the three editable_outputs with complete UTF-8 contents. Return immediately when those sources are ready; do not wait for execution or prepare the final findings report in this turn.",
                    "execution": {
                        "native_role": "Prepare source only. File/shell/web tools remain available for bounded source inspection and research. Project Python dependencies are not installed in this native shell: do not import or execute project models, search the host for Python, install dependencies or repair the environment.",
                        "supervisor": "After receiving the structured response, validate and freeze hypothesis.json, then launch experiment/experiment.py in the existing networkless OCI with read-only curated sources and cwd=/workspace.",
                        "oci_dependencies": [
                            "numpy",
                            "scipy",
                            "scikit-learn",
                            "joblib",
                            "threadpoolctl",
                            "pydantic",
                        ],
                        "program": "Compute and write the exact experiment_outputs only when the supervisor executes this program in OCI. The model must not create or fill these artifacts during source generation, guess measurements, or prepopulate output tables.",
                    },
                    "research_repair": "Act autonomously within the editable experiment program and permitted harness: repair code failures and revise the experiment implementation or research architecture using the saved previous-failure evidence. Return sources for another independently executed and reviewed attempt; unfinished errors are repair work, never goal completion. Preserve every attempt and negative/equal evidence. Compare candidates by preregistered criteria and retain the best verified baseline: a weaker, unverified or tied candidate without demonstrated benefit requires KEEP_BASE. Never alter the immutable goal, target, evaluator, source checkout, credentials or paid-route policy. Curated sources stay read-only; implement any required compatible workaround inside experiment/experiment.py. The controller continues subsequent attempts until all fixed goal criteria have verified evidence.",
                    "report": "report.md — предварительный план исследования со статусом: not_executed. Все разделы, гипотеза, предпосылки, методология и ожидаемая схема артефактов должны быть написаны исключительно на русском языке. Pre-execution plan must be written strictly in Russian. Include the hypothesis, assumptions, method and expected artifact schema only. No numerical results, confidence intervals, completed acceptance checklist or directions to advance stages. A separate post-probe role writes findings from the actual saved probe.",
                    "acceptance": "Execution/artifact completeness only. A separate evidence-bound critic and deterministic controller assess the frozen goal; do not claim acceptance or ask the owner questions."
                    if self.task.get("autonomous")
                    else "Execution/artifact completeness only. Scientific acceptance is manual.",
                    "preregistration": "Before execution return hypothesis.json with exactly hypothesis/falsification/baseline/scope strings and configurations/feature_sets finite lists. Each list entry is {id: stable_slug, definition: complete_parameter_object}; include scalar tuning and controls. artifacts/metrics.json must report configuration_ids exactly matching the preregistered configurations. The supervisor checks declarations and reported coverage, not arbitrary program semantics; scientific review audits actual code.",
                    "stage_caps": {
                        "configurations": self.task["max_configurations"],
                        "feature_sets": self.task["max_feature_sets"],
                    },
                    "machine_criteria": self.task.get("machine_criteria", {}),
                    "literature": self.task.get("literature", []),
                    "source_provenance": "Seed summaries are starting evidence, not a claim that you read primary papers or validated numerical parameters. Use only actually retrieved excerpts for stronger source claims; label missing evidence and uncalibrated parameters explicitly. Scientific criteria below describe the stage outcome, not claims to mark completed before this probe.",
                },
                ensure_ascii=False,
            )
            + "\n\n# Current stage mission and scientific constraints\n\n"
            + self.task["mission"]
        )

    def base_harness(self):
        return self.research_base

    def validate_tree(self, tree):
        """Поверхность харнесса совпадает с закреплённым upstream-контрактом."""
        from harness2.core.tree import is_allowed

        if not isinstance(tree, dict) or len(json_bytes(tree)) > 512 * 1024:
            raise ValueError("Harness превышает лимит")
        for name, body in tree.items():
            relative(name)
            if not isinstance(body, str) or len(body.encode()) > 256 * 1024:
                raise ValueError("Недопустимый harness file")
            if not is_allowed(self.substrate.surface(), name) and name not in {
                p for p, _ in self.substrate.surface().stubs
            }:
                raise ValueError("Research harness path вне поверхности: " + name)
        if "systemprompt.md" not in tree:
            raise ValueError("Harness должен сохранять systemprompt.md")

    def _launch(self, request, install):
        try:
            return self._launch_once(request, install)
        except Exception as error:
            if not self.runtime.profile.get("autonomous_quota_wait"):
                raise
            # Ошибка локального разбора/сохранения также не должна превращаться
            # в dead rollout и запуск зависимой ветви upstream.
            atomic_json(
                self.root / "local-failures" / (digest(request.run_ids[0].encode()) + ".json"),
                {"type": type(error).__name__, "message": redact(str(error))},
            )
            raise MicrostepInterrupted(str(error)) from error

    def _launch_once(self, request, install):
        self._task(request.task_ref)
        if len(request.run_ids) != 1:
            raise ValueError("One research rollout at a time")
        run_id = request.run_ids[0]
        tree = read_tree(install.staging_dir)
        self.validate_tree(tree)
        files = dict(self.sources)
        files.update({"harness/" + name: text for name, text in tree.items()})
        files["AGENTS.md"] = "\n\n".join(
            tree.get(name, "") for name in self.substrate.surface().always_on
        )
        request = {
            "kind": "agent",
            "role": "solver",
            "mission": self.task_prompt(self.task["id"]),
            "operation_id": "solver/" + run_id,
            "allowed_outputs": sorted(EDITABLE),
        }
        fingerprint = digest(
            json_bytes(
                {
                    "files": files,
                    "request": request,
                    "task": self.task,
                    "manifest": self.runtime.manifest,
                    "profile": self.runtime.profile,
                }
            )
        )
        destination = self.root / "rollouts" / safe_id(run_id)
        if destination.exists():
            receipt = json.loads(read_regular(destination / "receipt.json"))
            if (
                not self.delivered(destination)
                or receipt.get("input_sha256") != fingerprint
                or receipt.get("run_id") != run_id
            ):
                raise ResumeMismatchError(
                    "Research rollout provenance mismatch: " + run_id
                )
        else:
            output = self.runtime.execute(files, request, model_role="solver")
            if set(output.get("files", {})) != EDITABLE:
                raise ValueError(
                    "Research solver returned files outside editable allowlist"
                )
            stage_root = Path(self.task.get("declarations_root", self.root.parent))
            declaration_path = stage_root / "declarations.json"
            try:
                source_hypothesis = json.loads(output["files"]["hypothesis.json"])
            except ValueError:
                source_hypothesis = None
            output = repair_hypothesis(
                self.runtime, output, run_id,
                json.loads(read_regular(declaration_path, 2 * 1024 * 1024))
                if declaration_path.exists() else {},
            )
            hypothesis = json.loads(output["files"]["hypothesis.json"])
            required = {
                "hypothesis",
                "falsification",
                "baseline",
                "scope",
                "configurations",
                "feature_sets",
            }
            if (
                not isinstance(hypothesis, dict)
                or set(hypothesis) != required
                or any(
                    not isinstance(hypothesis[k], str) or not hypothesis[k].strip()
                    for k in ("hypothesis", "falsification", "baseline", "scope")
                )
                or any(
                    not isinstance(hypothesis[k], list)
                    for k in ("configurations", "feature_sets")
                )
            ):
                raise ValueError(
                    "Hypothesis, falsification, baseline and finite declared configurations required before probe"
                )
            # Preserve every declaration, including failed attempts. These are declared
            # scientific configurations; semantic compliance remains a human review duty.
            with (stage_root / ".declarations.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                declarations = (
                    json.loads(read_regular(declaration_path, 2 * 1024 * 1024))
                    if declaration_path.exists()
                    else {}
                )
                for field in ("configurations", "feature_sets"):
                    for declaration in hypothesis[field]:
                        if (
                            not isinstance(declaration, dict)
                            or set(declaration) != {"id", "definition"}
                            or not isinstance(declaration["id"], str)
                            or not re.fullmatch(
                                "[a-z][a-z0-9_-]{0,63}", declaration["id"]
                            )
                            or not isinstance(declaration["definition"], dict)
                        ):
                            raise ValueError(
                                "Stable id and complete definition required for every declared "
                                + field
                            )
                key = self.root.name + "/" + run_id
                if (
                    key in declarations
                    and declarations[key]["hypothesis"] != hypothesis
                ):
                    correction = output.get("_source_correction")
                    if (
                        declarations[key].get("admitted")
                        or not correction
                        or correction.get("accepted_shape") is not True
                        or declarations[key]["hypothesis"] != source_hypothesis
                    ):
                        raise ResumeMismatchError("A previous preregistration changed")
                    # Отказанная регистрация остаётся неизменной; исправленный
                    # документ регистрируется отдельно до первого расчёта.
                    key += "/source-correction/" + digest(json_bytes(hypothesis))
                    if key in declarations and declarations[key]["hypothesis"] != hypothesis:
                        raise ResumeMismatchError("A corrected preregistration changed")
                    if key not in declarations and "probe/" + run_id in self.runtime.journal.data["operations"]:
                        raise ResumeMismatchError("Cannot register a correction after probe")
                declarations[key] = {"hypothesis": hypothesis, "admitted": True}
                for field, cap in (
                    ("configurations", self.task["max_configurations"]),
                    ("feature_sets", self.task["max_feature_sets"]),
                ):
                    declared = {}
                    for row in declarations.values():
                        if row["admitted"]:
                            for value in row["hypothesis"][field]:
                                signature = digest(json_bytes(value["definition"]))
                                if (
                                    value["id"] in declared
                                    and declared[value["id"]] != signature
                                ):
                                    declarations[key]["admitted"] = False
                                    atomic_json(declaration_path, declarations)
                                    raise ValueError(
                                        "Configuration/feature id reused with a changed definition"
                                    )
                                declared[value["id"]] = signature
                    if len(declared) > cap:
                        declarations[key]["admitted"] = False
                        atomic_json(declaration_path, declarations)
                        raise ValueError(
                            "Stage-wide declared " + field + " cap exceeded"
                        )
                atomic_json(declaration_path, declarations)
            # Candidate program never imports/executes on host. Probe is development feedback,
            # not a hidden/scientific judge. Both successful and failed probes are useful data.
            probe = self.runtime.execute(
                dict(self.sources, **output["files"]),
                {
                    "kind": "experiment",
                    "outputs": self.task["experiment_outputs"],
                    "configuration_ids": [
                        value["id"] for value in hypothesis["configurations"]
                    ],
                    "operation_id": "probe/" + run_id,
                },
            )
            staging = (
                self.root / "rollouts" / (safe_id(run_id) + ".tmp-" + uuid.uuid4().hex)
            )
            write_tree(staging, output["files"])
            (staging / "answer.md").write_text(output["answer"], encoding="utf-8")
            scientific = {"status": "not_required"}
            if self.task.get("measured_evidence_required"):
                from .critic import machine_assessment

                try:
                    scientific = {
                        "status": "complete_for_review",
                        "assessment": machine_assessment(
                            probe, self.task["id"], "inconclusive"
                        ),
                    }
                except (ValueError, KeyError, TypeError) as error:
                    scientific = {"status": "evidence_incomplete", "reason": str(error)}
            projection = {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "[DEVELOPMENT PROBE; NOT SCIENTIFIC ACCEPTANCE]\n"
                    + json.dumps(
                        {"probe": redact(probe), "scientific_validation": scientific},
                        ensure_ascii=False,
                    ),
                },
            }
            events = (
                output["events"] + json.dumps(projection, ensure_ascii=False) + "\n"
            )
            (staging / "events.jsonl").write_text(events, encoding="utf-8")
            atomic_json(staging / "probe.json", redact(probe))
            covered = sorted(EDITABLE | {"answer.md", "events.jsonl", "probe.json"})
            atomic_json(
                staging / "receipt.json",
                {
                    "schema": "craving-research-receipt/v1",
                    "run_id": run_id,
                    "input_sha256": fingerprint,
                    "files": {
                        name: digest((staging / name).read_bytes()) for name in covered
                    },
                },
            )
            staging.replace(destination)
        if not any(row["run_id"] == run_id for row in self.completed_executions):
            self.completed_executions.append(
                {"run_id": run_id, "run_dir": str(destination.resolve())}
            )
        return [destination]

    def delivered(self, run_dir):
        try:
            root = Path(run_dir)
            receipt = json.loads(read_regular(root / "receipt.json"))
            covered = EDITABLE | {"answer.md", "events.jsonl", "probe.json"}
            return (
                receipt.get("schema") == "craving-research-receipt/v1"
                and set(receipt.get("files", {})) == covered
                and all(
                    digest(read_regular(root / name, 2 * 1024 * 1024).encode())
                    == expected
                    for name, expected in receipt["files"].items()
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def deliverable_files(self, run_dir):
        return {name: read_regular(Path(run_dir) / name) for name in sorted(EDITABLE)}

    def deliverable_text(self, run_dir):
        return read_regular(Path(run_dir) / "report.md")

    def score(self, run_dir, task_id):
        raise ValueError(
            "Research artifacts require manual scientific review; H1 judge is inapplicable"
        )
