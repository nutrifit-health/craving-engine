"""Read-only audit index and explicit fixed-OCI replay; never regenerates with an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import load_json
from .contracts import (
    PACKAGE,
    digest,
    digest_file,
    json_bytes,
    output_directory,
    read_regular,
)
from .runtime import Runtime
from .state import ResumeMismatchError, atomic_json, attempt_history, read_journal


def operation_packet(round_root, operation, *, state_path=None):
    journal = read_journal(state_path or round_root / "run-state.json")
    record = journal["operations"][operation]
    key = digest(operation.encode())
    packet_path = round_root / "operation-inputs" / (key + ".json")
    packet = load_json(packet_path)
    if (
        digest(json_bytes(packet)) != record.get("inputs_sha256")
        or packet.get("operation_id") != operation
    ):
        raise ResumeMismatchError(
            "Stored operation input packet changed or is missing provenance"
        )
    expected = digest(
        json_bytes(
            {
                "files": packet["files"],
                "request": packet["request"],
                "manifest": packet["manifest"],
                "profile": packet["profile"],
                "role": packet["model_role"],
            }
        )
    )
    if expected != record["fingerprint"] or packet.get("fingerprint") != expected:
        raise ResumeMismatchError(
            "Original operation fingerprint does not match input packet"
        )
    return packet, record


def index(campaign, *, strict=True, states=None):
    rows = []
    if states is None:
        states = set(campaign.glob("tasks/H*/round-*/run-state.json"))
        states.update((campaign / "autonomous").rglob("*-state.json"))
    if not strict:
        incidents = []
        for state in sorted(states):
            try:
                rows.extend(index(campaign, states=[state])["rounds"])
            except (
                ResumeMismatchError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as error:
                incidents.append(
                    {
                        "journal": str(state.relative_to(campaign)),
                        "error": str(error),
                        "type": type(error).__name__,
                    }
                )
        return {
            "schema": "craving-research-audit/v2",
            "campaign": str(campaign),
            "rounds": rows,
            "incidents": incidents,
            "journal_count": len(states),
            "evidence_complete": bool(rows)
            and not incidents
            and all(r["evidence_complete"] for r in rows),
            "policy": "История не изменяется; повреждённые записи не годятся для допуска",
        }
    for state_path in sorted(states):
        round_root = state_path.parent
        state = read_journal(state_path)
        phase = state_path.name.removesuffix("-state.json")
        migration = state.get("attempt_history_migration")
        if (
            migration
            and digest((round_root / migration["path"]).read_bytes())
            != migration["sha256"]
        ):
            raise ResumeMismatchError("Original journal migration evidence changed")
        operations = []
        workspace_evidence_complete = True
        for operation, record in state["operations"].items():
            key = digest(operation.encode())
            packet_path = round_root / "operation-inputs" / (key + ".json")
            packet, _ = operation_packet(round_root, operation, state_path=state_path)
            result_path = round_root / "role-results" / (key + ".json")
            if record.get("status") == "completed":
                result_packet = load_json(result_path)
                if digest(json_bytes(result_packet["result"])) != record.get(
                    "result_sha256"
                ) or result_packet.get("result_sha256") != record.get("result_sha256"):
                    raise ResumeMismatchError("Recorded operation result changed")
            for attempt in attempt_history(round_root, record):
                archive_ref = attempt.get("workspace_archive")
                if packet["model_role"] and not packet["profile"].get(
                    "nonpaid_fixture"
                ):
                    if not archive_ref:
                        workspace_evidence_complete = False
                    else:
                        archive_path = round_root / archive_ref["path"]
                        if digest(archive_path.read_bytes()) != archive_ref["sha256"]:
                            raise ResumeMismatchError("Workspace audit archive changed")
                        archive = load_json(archive_path)
                        workspace = archive_path.with_name(
                            archive_path.name.removesuffix("-archive.json")
                        )
                        for name, entry in archive["files"].items():
                            from .contracts import relative

                            relative(name)
                            if entry.get("sha256") is None:
                                workspace_evidence_complete = False
                            elif (workspace / name).is_symlink() or digest_file(
                                workspace / name
                            ) != entry["sha256"]:
                                raise ResumeMismatchError(
                                    "Intermediate workspace evidence changed: "
                                    + str(workspace / name)
                                )
                        workspace_evidence_complete = (
                            workspace_evidence_complete
                            and archive.get("evidence_complete") is True
                        )
                failure = attempt.get("public_failure")
                if (
                    failure
                    and digest(json_bytes(load_json(round_root / failure["path"])))
                    != failure["sha256"]
                ):
                    raise ResumeMismatchError(
                        "Recorded failed-attempt evidence changed"
                    )
            operations.append(
                {
                    "id": operation,
                    "role": record["role"],
                    "status": record["status"],
                    "input_packet": str(packet_path.relative_to(campaign)),
                    "input_sha256": record["inputs_sha256"],
                    "input_files": {
                        name: digest(text.encode())
                        for name, text in packet["files"].items()
                    },
                    "environment_profile_sha256": digest(json_bytes(packet["profile"])),
                    "source_manifest_sha256": digest(json_bytes(packet["manifest"])),
                    "result": str(
                        (round_root / "role-results" / (key + ".json")).relative_to(
                            campaign
                        )
                    ),
                    "result_sha256": record.get("result_sha256"),
                    "attempts": record.get("attempt_history", []),
                    "attempt_ledger": record.get("attempt_ledger"),
                    "total_attempts": record["attempts"],
                    "quota_attempts": record.get("quota_attempts", 0),
                    "fixed_oci_replay_supported": packet["model_role"] is None
                    and packet["request"]["kind"] == "experiment",
                }
            )
        rollouts = []
        for receipt_path in sorted(round_root.glob("rollouts/*/receipt.json")):
            root = receipt_path.parent
            receipt = load_json(receipt_path)
            for name, expected in receipt.get("files", {}).items():
                if (
                    digest(read_regular(root / name, 2 * 1024 * 1024).encode())
                    != expected
                ):
                    raise ResumeMismatchError(
                        "Rollout artifact changed: " + str(root / name)
                    )
            rollouts.append(
                {
                    "path": str(root.relative_to(campaign)),
                    "receipt": receipt,
                    "hypothesis": load_json(root / "hypothesis.json"),
                    "probe": str((root / "probe.json").relative_to(campaign)),
                }
            )
        claims_path = round_root / "claims.json"
        claims = load_json(claims_path) if claims_path.exists() else []
        for claim in claims:
            for name, expected in claim["evidence"].items():
                if digest((round_root / name).read_bytes()) != expected:
                    raise ResumeMismatchError("Report claim evidence changed")
        if (round_root / "round-complete.json").exists():
            from .queue import verify_round

            verify_round(round_root)
        rows.append(
            {
                "evidence_complete": workspace_evidence_complete
                and (
                    (round_root / "round-complete.json").exists()
                    if phase == "run"
                    else state.get("status") == "completed"
                )
                and all(op["status"] == "completed" for op in operations),
                "round": str(round_root.relative_to(campaign)),
                "binding": state["binding"],
                "phase": phase,
                "journal": str(state_path.relative_to(campaign)),
                "attempt_history_migration": migration,
                "next_step": state["next_step"],
                "operations": operations,
                "rollouts": rollouts,
                "claims": claims,
                "report": str(
                    (round_root / "research-report.md").relative_to(campaign)
                ),
            }
        )
    return {
        "schema": "craving-research-audit/v1",
        "campaign": str(campaign),
        "rounds": rows,
        "journal_count": len(rows),
        "evidence_complete": bool(rows)
        and all(row["evidence_complete"] for row in rows),
        "limits": "Model turns are stochastic. Replay repeats only saved generated Python in the pinned OCI; it does not establish scientific acceptance.",
    }


def replay(round_root, operation, output, *, execution_manifest=None):
    packet, record = operation_packet(round_root, operation)
    if packet["model_role"] is not None or packet["request"]["kind"] != "experiment":
        raise ValueError(
            "Replay supports only saved experiment probes, never LLM/model calls"
        )
    if execution_manifest is not None:
        from .microsteps import bound_manifest

        original = bound_manifest(round_root, execution_manifest, packet["profile"], "run")
        if original != packet["manifest"]:
            raise ValueError("Replay revision не разрешена для исходного пакета")
    repo = PACKAGE.parent.resolve()
    output = output_directory(output, repo, repo / ".research/vendor/harness2")
    if (
        output == round_root
        or round_root in output.parents
        or output in round_root.parents
    ):
        raise ValueError("Replay output must not overlap its original round")
    if output.exists() and any(output.iterdir()) and not (output / "replay-state.json").exists():
        raise ValueError("Replay output содержит несвязанные данные")
    output.mkdir(parents=True, exist_ok=True)
    profile = dict(packet["profile"], stop_file=str(output / "STOP"))
    runtime = Runtime(
        profile, output, packet["manifest"], phase="replay",
        execution_manifest=execution_manifest,
    )
    result = None
    failure = None
    try:
        runtime.prepare(model_access=False)
        result = runtime.execute(packet["files"], packet["request"])
    except BaseException as exc:
        failure = exc
        raise
    finally:
        runtime.finish("inconclusive" if failure else "completed")
        previous_report = output / "replay-report.json"
        if previous_report.exists():
            previous = previous_report.read_bytes()
            from .state import archive_original_journal

            snapshot = output / "report-history" / (digest(previous) + ".json")
            snapshot.parent.mkdir(exist_ok=True)
            if not snapshot.exists():
                archive_original_journal(snapshot, previous)
        atomic_json(
            output / "replay-report.json",
            {
                "schema": "craving-research-replay/v1",
                "original_operation": operation,
                "original_round": str(round_root),
                "original_input_sha256": record["inputs_sha256"],
                "result": result,
                "original_result_sha256": record.get("result_sha256"),
                "replayed_result_sha256": digest(json_bytes(result))
                if result is not None
                else None,
                "exact_output_match": result is not None
                and digest(json_bytes(result)) == record.get("result_sha256"),
                "usage": runtime.usage,
                "scientific_acceptance": "not_granted_by_replay",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("index")
    inspect.add_argument("--campaign", type=Path, required=True)
    inspect.add_argument("--collect-errors", action="store_true")
    repeat = subs.add_parser("replay")
    repeat.add_argument("--round", type=Path, required=True)
    repeat.add_argument("--operation", required=True)
    repeat.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "index":
        print(
            json.dumps(
                index(args.campaign.resolve(), strict=not args.collect_errors),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        replay(args.round.resolve(), args.operation, args.output)


if __name__ == "__main__":
    main()
