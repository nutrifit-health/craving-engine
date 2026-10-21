"""Independent subscription critic; hash-bound evidence is necessary for admission."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .contracts import digest, relative
from .runtime import Runtime
from .state import RunStopped, atomic_json
from .subscription import EnvironmentUnavailableError, QuotaExhaustedError

H4_MACHINE = {
    "delta_macro_brier": {"maximum": -0.002},
    "ci_upper": {"exclusive_maximum": 0},
    "checkpoint_max_bytes": {"maximum": 16384},
    "coverage_days": {"minimum": 365},
    "pending_feedback_drained": {"equals": True},
}


def machine_assessment(probe, task_id, outcome):
    """Fixed goal predicates; the critic cannot supply or change measured values."""
    if probe.get("execution_complete") is False:
        raise ValueError("Незавершённое выполнение не является научным результатом")
    if task_id in {"H2", "H3"}:
        from .scientific_evidence import validate_stage_evidence

        return validate_stage_evidence(
            json.loads(probe["files"]["artifacts/metrics.json"]), task_id
        )
    if task_id != "H4":
        return {
            "numeric_criteria": {},
            "semantic_criteria": "independent evidence critic",
        }
    metrics = json.loads(probe["files"]["artifacts/metrics.json"])
    values = metrics.get("autonomous_evidence")
    if not isinstance(values, dict) or not set(H4_MACHINE) <= set(values):
        raise ValueError(
            "H4 needs measured autonomous_evidence for fixed numerical criteria"
        )
    for key in H4_MACHINE:
        value = values[key]
        if key == "pending_feedback_drained":
            if not isinstance(value, bool):
                raise ValueError("Measured feedback drain flag required")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Finite measured numeric criterion required: " + key)
    if values["coverage_days"] < 365 or values["pending_feedback_drained"] is not True:
        raise ValueError(
            "Incomplete 365-day/drained-feedback evidence is not a negative result"
        )
    passes = {
        "delta_macro_brier": values["delta_macro_brier"] <= -0.002,
        "ci_upper": values["ci_upper"] < 0,
        "checkpoint_max_bytes": 0 < values["checkpoint_max_bytes"] <= 16384,
        "coverage_days": True,
        "pending_feedback_drained": True,
    }
    if outcome == "supported" and not all(passes.values()):
        raise ValueError("Critic supported outcome contradicts fixed numerical goal")
    return {
        "numeric_criteria": {
            key: {
                "value": values[key],
                "passes": passes[key],
                "predicate": H4_MACHINE[key],
                "pointer": "/files/artifacts~1metrics.json (JSON string)/autonomous_evidence/"
                + key,
            }
            for key in H4_MACHINE
        },
        "metrics_sha256": digest(probe["files"]["artifacts/metrics.json"].encode()),
        "semantic_criteria": "independent evidence critic; negative rationale is not inferred from exit status",
    }


def pointer(document, path):
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Evidence needs a JSON pointer")
    for part in path[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(document, list):
                if not key.isdigit() or str(int(key)) != key:
                    raise ValueError("Invalid JSON array index")
                document = document[int(key)]
            elif isinstance(document, dict):
                document = document[key]
            else:
                raise TypeError("JSON pointer traverses a scalar")
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Evidence JSON pointer does not resolve: " + path
            ) from error
    return document


def validate_review(review, criteria, files):
    if not isinstance(review, dict) or set(review) != {
        "outcome",
        "criteria",
        "next_experiment",
    }:
        raise ValueError("Invalid critic response")
    if review["outcome"] not in {"supported", "refuted", "inconclusive"}:
        raise ValueError("Invalid research outcome")
    rows = review["criteria"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(criteria)
        or {r.get("id") for r in rows if isinstance(r, dict)} != set(criteria)
    ):
        raise ValueError("Critic must assess every immutable criterion exactly once")
    for row in rows:
        if (
            set(row) != {"id", "verdict", "reason", "evidence"}
            or row["verdict"] not in {"supported", "refuted", "inconclusive"}
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
        ):
            raise ValueError("Invalid criterion assessment")
        if not isinstance(row["evidence"], list) or not row["evidence"]:
            raise ValueError("No evidence for criterion")
        for ref in row["evidence"]:
            if not isinstance(ref, dict) or set(ref) != {
                "path",
                "sha256",
                "pointer",
                "value",
            }:
                raise ValueError("Invalid evidence reference")
            relative(ref["path"])
            if ref["path"] in {"immutable-goal.json", "exposure.json"} or ref[
                "path"
            ].endswith("hypothesis.json"):
                raise ValueError(
                    "План/цель не является измеренным доказательством критерия"
                )
            body = files[ref["path"]]
            if (
                digest(body.encode()) != ref["sha256"]
                or pointer(json.loads(body), ref["pointer"]) != ref["value"]
            ):
                raise ValueError("Critic evidence does not match frozen measured value")
    if not isinstance(review["next_experiment"], str):
        raise TypeError("Repair guidance required")
    if review["outcome"] != "inconclusive" and any(
        r["verdict"] == "inconclusive" for r in rows
    ):
        raise ValueError("Incomplete criteria cannot admit a stage")
    if review["outcome"] == "supported" and any(
        r["verdict"] != "supported" for r in rows
    ):
        raise ValueError("Positive outcome contradicts criteria")
    if review["outcome"] == "refuted" and not any(
        r["verdict"] == "refuted" for r in rows
    ):
        raise ValueError("Negative outcome lacks a falsified criterion")
    return review


def assess(root: Path, files, criteria, profile, manifest):
    """Fresh independent role, its own journal; completed research is never repeated."""
    root.mkdir(parents=True, exist_ok=True)
    request = {
        "kind": "agent",
        "role": "selector",
        "operation_id": "independent-critic",
        "mission": 'You are an independent scientific critic, not the source author. Assess ALL immutable criteria against actual program, preregistration, measured OCI artifacts and exact replay. Do not accept just because execution succeeds. Inspect leakage, temporal availability, baseline parity, configuration search exposure, uncertainty, units and limits. Missing evidence means inconclusive; a negative result needs actual falsification and coverage, never just no positive finding. Only development exploratory claims; no confirmatory or real-user benefit. Return exactly the outer envelope {"answer": review_object, "files":{}}. answer may be the review object directly or a JSON-encoded string. The review object has exactly outcome (supported|refuted|inconclusive), criteria (one {id,verdict,reason,evidence:[{path,sha256,pointer,value}]} per criterion), next_experiment (bounded corrective guidance). Each evidence reference must match a JSON value and SHA256 in the supplied files. Use artifacts/metrics.json for measured evidence, never immutable-goal.json, exposure.json or hypothesis.json. Input sources are supplied under input/. A source URL without retrieved excerpt is not evidence. All conclusions and reasons must be in Russian. files must be {}. No questions to the owner, no environment repair. Criteria: '
        + json.dumps(criteria, ensure_ascii=False),
    }
    runtime = Runtime(profile, root, manifest, phase="critic")
    status = "inconclusive"
    try:
        runtime.prepare(model_access=True)
        output = runtime.execute(files, request, model_role="selector")
        review = validate_review(json.loads(output["answer"]), criteria, files)
        atomic_json(root / "review.json", review)
        status = "completed"
        return review
    except QuotaExhaustedError:
        status = "paused_quota"
        raise
    except EnvironmentUnavailableError:
        status = "waiting_environment"
        raise
    except RunStopped:
        status = "interrupted"
        raise
    finally:
        runtime.finish(status)
