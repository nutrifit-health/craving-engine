"""Последовательная конечная серия: tuning -> freeze -> development -> новые сценарии."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from research_v8.archive import RunArchive, digest
from research_v8.context_representation import (
    ContextTransform,
    FixedContextRepresentation,
)
from research_v8.evaluation import choose_thresholds
from research_v8.replay import graph_matrix
from research_v8.reuse import _verify_files
from research_v10.analysis import (
    POINT,
    adaptation,
    compact_report,
    comparison,
    macro_brier,
    report,
)
from research_v10.replay import parse_records, policy_rows, replay
from research_v10.scenarios import generate
from research_v10.spiking import encode, training_scale
from research_v10.types import EVALUATION_SEEDS, SCENARIOS, Variant, variants

ROOT = Path(__file__).resolve().parents[1]


def save_rows(archive: RunArchive, name: str, rows: list[dict]) -> None:
    archive.write_checkpoint(
        name,
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def encoded(
    variant: Variant, codes: np.ndarray, p0: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if variant.family == "scalar":
        return np.ones((len(codes), 1)), None, {}
    if variant.lif is None:
        return codes, None, {}
    start = time.perf_counter()
    batch = encode(codes, p0, scale, variant.lif)
    return (
        batch.rates,
        batch.eligibility,
        {
            "encoding_seconds": time.perf_counter() - start,
            "spike_fraction": batch.spike_fraction,
            "silent_fraction": batch.silent_fraction,
        },
    )


def run_variant(
    variant: Variant, rows: list[dict], codes: np.ndarray, p0: np.ndarray, scale: float
):
    h, eligibility, diagnostics = encoded(variant, codes, p0, scale)
    scores, costs = replay(rows, h, p0, variant, eligibility)
    return scores, {**costs, **diagnostics}


def thresholds(rows: list[dict]) -> dict:
    return {
        policy: choose_thresholds(policy_rows(rows, policy))[POINT]
        for policy in ("gated", "ungated")
    }


def reports(rows: list[dict], frozen: dict) -> dict:
    return {
        policy: report(policy_rows(rows, policy), frozen[policy])
        for policy in ("gated", "ungated")
    }


def audit_inputs(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    _verify_files(bundle, manifest["artifacts"])
    cohorts, events = {}, set()
    for split in ("training", "calibration", "development"):
        rows = parse_records(bundle / f"{split}.jsonl")
        cohorts[split] = {r["user_id"] for r in rows}
        for row in rows:
            if (
                row["event_id"] in events
                or row["available_at"] > row["decision_at"]
                or (row["outcome"] is None) != (row["feedback_at"] is None)
                or (
                    row["feedback_at"] is not None
                    and row["feedback_at"] <= row["decision_at"]
                )
            ):
                raise ValueError("Нарушена причинность или уникальность событий")
            events.add(row["event_id"])
    if any(
        cohorts[a] & cohorts[b]
        for a, b in (
            ("training", "calibration"),
            ("training", "development"),
            ("calibration", "development"),
        )
    ):
        raise ValueError("Пересечение пользователей")
    return {
        "manifest_sha256": digest(bundle / "manifest.json"),
        "users": {k: len(v) for k, v in cohorts.items()},
        "events": len(events),
        "status": "passed",
        "evaluation_metrics_not_read": True,
    }


def run(output: Path, bundle: Path) -> None:
    start = time.perf_counter()
    matrix = variants()
    archive = RunArchive(
        output,
        {
            "iteration": "V10",
            "variants": [asdict(v) for v in matrix],
            "protocol": "https://gov.nutrifit.health/documents/6ac6b48e56965bd66f608816",
            "protocol_sha256": digest(ROOT / "docs/protocol-v10.ru.md"),
            "evaluation_seeds": EVALUATION_SEEDS,
            "scenarios": SCENARIOS,
            "calibration_seed": 6101,
            "synthetic_only": True,
        },
    )
    hashes = {}
    for folder in (
        "research_v6",
        "research_v7",
        "research_v8",
        "research_v9",
        "research_v10",
        "synthetic_benchmark",
    ):
        for source in sorted((ROOT / folder).glob("*.py")):
            relative = source.relative_to(ROOT).as_posix()
            archive._path("sources/" + relative).write_bytes(source.read_bytes())
            hashes[relative] = digest(source)
    for name in (
        "hybrid_engine.py",
        "mushroom_body.py",
        "pyproject.toml",
        "docs/protocol-v10.ru.md",
    ):
        archive._path("sources/" + name).write_bytes((ROOT / name).read_bytes())
        hashes[name] = digest(ROOT / name)
    archive.write_json("source_manifest.json", hashes)
    archive.write_json(
        "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "pydantic",
                    "threadpoolctl",
                )
            },
            "threads": 1,
        },
    )
    archive.write_json("input_audit.json", audit_inputs(bundle))
    print("V10: input audit passed", flush=True)
    schema = json.loads((bundle / "schema.json").read_text())
    training = np.load(bundle / "training.X.npy", allow_pickle=False)
    transform = ContextTransform.fit(training, schema)
    q = transform.transform(training)
    graph, _ = graph_matrix(bundle / "connectome.npz")
    representation = FixedContextRepresentation.fit(q, graph, 11)
    scale = training_scale(representation.encode(q))
    archive.write_json("transform.json", transform.to_dict())
    representation.save(archive._path("representation.npz"))
    del training, q
    cal_rows = parse_records(bundle / "calibration.jsonl")
    cal_h = representation.encode(
        transform.transform(np.load(bundle / "calibration.X.npy", allow_pickle=False))
    )
    cal_p = np.load(bundle / "calibration.gbdt.npy", allow_pickle=False)
    users = sorted({r["user_id"] for r in cal_rows})
    tune_users = set(users[:10])
    tune_mask = np.asarray([r["user_id"] in tune_users for r in cal_rows])
    tune_rows = [r for r, keep in zip(cal_rows, tune_mask, strict=True) if keep]
    threshold_rows = [
        r for r, keep in zip(cal_rows, tune_mask, strict=True) if not keep
    ]
    ranking, best = [], {}
    for variant in matrix:
        scores, costs = run_variant(
            variant, tune_rows, cal_h[tune_mask], cal_p[tune_mask], scale
        )
        quality = macro_brier(scores)
        ranking.append(
            {"variant": asdict(variant), "macro_brier": quality, "costs": costs}
        )
        if (
            variant.family not in best
            or (quality, variant.name) < best[variant.family][:2]
        ):
            best[variant.family] = (quality, variant.name, variant)
        print(f"tuning {variant.name}: Brier={quality:.8f}", flush=True)
    archive.write_json("tuning.json", ranking)
    selected = {family: item[2] for family, item in best.items()}
    chosen = selected["lif_rstdp"]
    for rule in ("direct", "stdp"):
        family = f"lif_{rule}_matched"
        selected[family] = Variant(family, family, chosen.rate, chosen.lif, rule)
    frozen = {}
    for family, variant in selected.items():
        rows, _ = run_variant(
            variant, threshold_rows, cal_h[~tune_mask], cal_p[~tune_mask], scale
        )
        frozen[family] = thresholds(rows)
        save_rows(archive, f"predictions/calibration/{family}.jsonl.gz", rows)
    frozen_path = archive.write_json(
        "frozen.json",
        {
            "selected": {k: asdict(v) for k, v in selected.items()},
            "thresholds": frozen,
            "training_scale": scale,
            "tuning_users": users[:10],
            "threshold_users": users[10:],
        },
    )
    frozen_sha = digest(frozen_path)
    archive.event("frozen_before_development", sha256=frozen_sha)
    print("V10: configuration and thresholds frozen", flush=True)
    dev_rows = parse_records(bundle / "development.jsonl")
    dev_h = representation.encode(
        transform.transform(np.load(bundle / "development.X.npy", allow_pickle=False))
    )
    dev_p = np.load(bundle / "development.gbdt.npy", allow_pickle=False)
    development, costs = {}, {}
    for family, variant in selected.items():
        rows, costs[family] = run_variant(variant, dev_rows, dev_h, dev_p, scale)
        development[family] = reports(rows, frozen[family])
        save_rows(archive, f"predictions/development/{family}.jsonl.gz", rows)
        print(f"development {family}: Brier={macro_brier(rows):.8f}", flush=True)
    archive.write_json("development.json", development)
    archive.write_json("development_costs.json", costs)
    archive.write_json(
        "development_comparisons.json",
        {
            f"{candidate}_vs_{baseline}_{policy}": comparison(
                development[candidate][policy], development[baseline][policy]
            )
            for candidate, baseline in (
                ("lif_rstdp", "lif_direct_matched"),
                ("lif_rstdp", "lif_stdp_matched"),
                ("lif_rstdp", "lif_direct"),
                ("lif_rstdp", "direct"),
                ("lif_direct", "direct"),
                ("lif_rstdp", "scalar"),
                ("direct", "population"),
                ("lif_stdp", "lif_rstdp"),
            )
            for policy in ("gated", "ungated")
        },
    )
    del dev_h, cal_h
    summary = {
        "development": {
            family: {
                policy: compact_report(value) for policy, value in policies.items()
            }
            for family, policies in development.items()
        },
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        cal_records, h, p = generate(6101, scenario, split="calibration")
        scenario_scale = training_scale(h)
        scenario_thresholds = {}
        for family, variant in selected.items():
            rows, _ = run_variant(variant, cal_records, h, p, scenario_scale)
            scenario_thresholds[family] = thresholds(rows)
        archive.write_json(
            f"scenarios/{scenario}/frozen_thresholds.json",
            {"scale": scenario_scale, "thresholds": scenario_thresholds},
        )
        accumulated = {family: [] for family in selected}
        seed_results = {}
        for seed in EVALUATION_SEEDS:
            records, h, p = generate(seed, scenario)
            seed_results[str(seed)] = {}
            for family, variant in selected.items():
                rows, cost = run_variant(variant, records, h, p, scenario_scale)
                accumulated[family].extend(rows)
                seed_results[str(seed)][family] = {
                    "reports": reports(rows, scenario_thresholds[family]),
                    "costs": cost,
                }
                save_rows(
                    archive, f"predictions/{scenario}/{seed}-{family}.jsonl.gz", rows
                )
        pooled = {
            family: reports(rows, scenario_thresholds[family])
            for family, rows in accumulated.items()
        }
        contrasts = {
            f"{candidate}_vs_{baseline}_{policy}": comparison(
                pooled[candidate][policy], pooled[baseline][policy]
            )
            for candidate, baseline in (
                ("lif_rstdp", "lif_direct_matched"),
                ("lif_rstdp", "lif_stdp_matched"),
                ("lif_rstdp", "lif_direct"),
                ("lif_rstdp", "direct"),
                ("lif_direct", "direct"),
                ("lif_rstdp", "scalar"),
            )
            for policy in ("gated", "ungated")
        }
        payload = {
            "per_seed": seed_results,
            "combined": pooled,
            "comparisons": contrasts,
            "adaptation": {
                family: adaptation(rows) for family, rows in accumulated.items()
            },
        }
        archive.write_json(f"scenarios/{scenario}/report.json", payload)
        summary["scenarios"][scenario] = {
            family: {
                policy: compact_report(value) for policy, value in policies.items()
            }
            for family, policies in pooled.items()
        }
        print(f"scenario {scenario} complete", flush=True)
    if digest(frozen_path) != frozen_sha:
        raise RuntimeError("Зафиксированные настройки изменились")
    summary["seconds"] = time.perf_counter() - start
    summary["selected"] = {k: asdict(v) for k, v in selected.items()}
    archive.write_json("summary.json", summary)
    archive.finish()
    print(f"V10 complete: {archive.path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/v9-frozen")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output_dir, args.data_dir)


if __name__ == "__main__":
    main()
