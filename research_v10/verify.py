"""Независимая проверка артефактов, повтор прогнозов и отрицательные контроли."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from research_v8.archive import digest
from research_v8.context_representation import (
    ContextTransform,
    FixedContextRepresentation,
)
from research_v8.evaluation import interval
from research_v8.reuse import _verify_files
from research_v10.analysis import POINT, macro_brier
from research_v10.experiment import ROOT, run_variant
from research_v10.replay import parse_records
from research_v10.types import LIFConfig, Variant


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def paired_phase(candidate: list[dict], baseline: list[dict], count: int) -> dict:
    grouped = {}
    for c, b in zip(candidate, baseline, strict=True):
        if c["event_id"] != b["event_id"]:
            raise ValueError("Порядок пар нарушен")
        if c["day"] >= c["change_day"] and c["post_change_feedback"] == count:
            grouped.setdefault(c["user_id"], []).append((c, b))
    result = {"users": len(grouped), "decisions": sum(map(len, grouped.values()))}
    for policy, field in (("gated", "probability"), ("ungated", "ungated")):
        for metric, truth in (
            ("brier", "outcome"),
            ("probability_mse", "true_probability"),
        ):
            deltas = [
                float(
                    np.mean(
                        [
                            (c[field] - c[truth]) ** 2 - (b[field] - b[truth]) ** 2
                            for c, b in rows
                        ]
                    )
                )
                for rows in grouped.values()
            ]
            result[f"{policy}_{metric}_delta"] = interval(deltas, seed=20261007)
    return result


def admission(run: Path) -> dict:
    development = json.loads((run / "development_comparisons.json").read_text())
    abrupt = json.loads((run / "scenarios/abrupt/report.json").read_text())
    stable = json.loads((run / "scenarios/stable/report.json").read_text())
    decisions = {}
    for baseline in ("lif_direct_matched", "direct", "scalar"):
        key = f"lif_rstdp_vs_{baseline}_gated"
        checks = {}
        for name, contrasts in (
            ("v9_development", development),
            ("new_abrupt", abrupt["comparisons"]),
        ):
            diff = contrasts[key]
            brier = diff["brier_delta"]
            checks[name] = {
                "brier_gain_at_least_0002": brier["mean"] <= -0.002,
                "brier_ci_below_zero": brier["ci95"][1] < 0,
                "fpr_increase_at_most_001": diff["false_positive_rate_delta"]["mean"]
                <= 0.01,
                "j_not_lower": diff["youden_j_delta"]["mean"] >= 0,
            }
        stable_diff = stable["comparisons"][key]["brier_delta"]["mean"]
        wins = sum(
            seed["lif_rstdp"]["reports"]["gated"]["operating_points"][POINT]["macro"][
                "brier"
            ]["mean"]
            < seed[baseline]["reports"]["gated"]["operating_points"][POINT]["macro"][
                "brier"
            ]["mean"]
            for seed in abrupt["per_seed"].values()
        )
        checks["stable_brier_degradation_at_most_0002"] = stable_diff <= 0.002
        checks["improvement_on_at_least_two_seeds"] = wins >= 2
        values = [
            v
            for group in checks.values()
            for v in (group.values() if isinstance(group, dict) else [group])
        ]
        decisions[baseline] = {
            "passed": all(values),
            "checks": checks,
            "seed_wins": wins,
        }
    return {
        "recommendation_supported": all(d["passed"] for d in decisions.values()),
        "comparisons": decisions,
        "scope": "finite original implementation; synthetic evidence only; no universal claim about SNNs",
    }


def verify(run: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / "artifact_manifest.json").read_text())
    _verify_files(run, manifest)
    frozen = json.loads((run / "frozen.json").read_text())
    bundle = ROOT / "data/v9-frozen"
    records = parse_records(bundle / "development.jsonl")
    transform = ContextTransform.from_dict(
        json.loads((run / "transform.json").read_text())
    )
    rep = FixedContextRepresentation.load(run / "representation.npz")
    h = rep.encode(
        transform.transform(np.load(bundle / "development.X.npy", allow_pickle=False))
    )
    p0 = np.load(bundle / "development.gbdt.npy", allow_pickle=False)
    repeated = {}
    for family, payload in frozen["selected"].items():
        v = Variant(
            **{
                **payload,
                "lif": LIFConfig(**payload["lif"]) if payload["lif"] else None,
            }
        )
        new, _ = run_variant(v, records, h, p0, frozen["training_scale"])
        old = read_rows(run / f"predictions/development/{family}.jsonl.gz")
        repeated[family] = {"exact_rows_match": new == old, "rows": len(new)}
        if new != old:
            raise AssertionError("Повтор прогнозов отличается: " + family)
    controls = {}
    rng = np.random.default_rng(20261007)
    shuffled = [dict(r) for r in records]
    for user in sorted({r["user_id"] for r in shuffled}):
        items = [
            r for r in shuffled if r["user_id"] == user and r["outcome"] is not None
        ]
        labels = rng.permutation([r["outcome"] for r in items])
        for row, value in zip(items, labels, strict=True):
            row["outcome"] = int(value)
    none = [{**r, "outcome": None, "feedback_at": None} for r in records]
    for family in ("direct", "lif_rstdp"):
        payload = frozen["selected"][family]
        v = Variant(
            **{
                **payload,
                "lif": LIFConfig(**payload["lif"]) if payload["lif"] else None,
            }
        )
        scores, _ = run_variant(v, none, h, p0, frozen["training_scale"])
        raw_drift = max(abs(row["ungated"] - row["p_population"]) for row in scores)
        # p_learning проходит sigmoid(logit(p0)); допускается только погрешность
        # float64 round-trip. Итоговый gated-прогноз должен совпасть точно.
        no_feedback = (
            all(row["probability"] == row["p_population"] for row in scores)
            and raw_drift <= 1e-14
        )
        if not no_feedback:
            raise AssertionError("Память изменила прогноз без feedback")
        scores, _ = run_variant(v, shuffled, h, p0, frozen["training_scale"])
        controls[family] = {
            "no_feedback_equals_population": no_feedback,
            "no_feedback_ungated_max_delta": raw_drift,
            "ungated_round_trip_tolerance": 1e-14,
            "shuffled_macro_brier": macro_brier(scores),
        }
    few_shot, oracle = {}, {}
    for scenario in (
        "abrupt",
        "gradual",
        "transient",
        "noise15",
        "delay72",
        "report50",
        "report10",
    ):
        all_rows = {
            family: [
                r
                for seed in (7101, 7102, 7103)
                for r in read_rows(
                    run / f"predictions/{scenario}/{seed}-{family}.jsonl.gz"
                )
            ]
            for family in (
                "lif_rstdp",
                "lif_direct_matched",
                "direct",
                "scalar",
                "population",
            )
        }
        few_shot[scenario] = {
            baseline: {
                str(n): paired_phase(all_rows["lif_rstdp"], all_rows[baseline], n)
                for n in (0, 1, 2, 5, 10, 20)
            }
            for baseline in ("lif_direct_matched", "direct", "scalar", "population")
        }
        reference = all_rows["population"]
        true = np.asarray([r["true_probability"] for r in reference])
        p = np.asarray([r["p_population"] for r in reference])
        bounded = expit(logit(p) + np.clip(logit(true) - logit(p), -0.7, 0.7))
        oracle[scenario] = {
            "population_probability_mse": float(np.mean((p - true) ** 2)),
            "best_probability_mse_with_logit_cap_07": float(
                np.mean((bounded - true) ** 2)
            ),
            "unreachable_fraction": float(
                np.mean(np.abs(logit(true) - logit(p)) > 0.7)
            ),
            "interpretation": "Evaluator-only bound; not a fitted model or clinical ceiling",
        }
    result = {
        "manifest_files_verified": len(manifest),
        "repeated": repeated,
        "controls": controls,
        "admission": admission(run),
        "oracle_cap_diagnostics": oracle,
    }
    (output / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output / "few-shot-paired.json").write_text(
        json.dumps(few_shot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output / "source_sha256.json").write_text(
        json.dumps({"verify.py": digest(Path(__file__))}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "verified": True,
                "recommendation_supported": result["admission"][
                    "recommendation_supported"
                ],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.run, args.output_dir)


if __name__ == "__main__":
    main()
