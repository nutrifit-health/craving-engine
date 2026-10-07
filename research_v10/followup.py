"""Послеоценочный контроль скорости адаптации без изменения выбранных моделей."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from research_v8.archive import digest
from research_v10.experiment import run_variant
from research_v10.scenarios import generate
from research_v10.types import LIFConfig, Variant
from research_v10.verify import paired_phase, read_rows


def run(root: Path, output: Path) -> None:
    frozen = json.loads((root / "frozen.json").read_text())
    scenario = json.loads(
        (root / "scenarios/abrupt/frozen_thresholds.json").read_text()
    )
    result = {
        "status": "post_evaluation_diagnostic_no_retuning",
        "frozen_sha256": digest(root / "frozen.json"),
        "source_sha256": digest(Path(__file__)),
        "families": {},
    }
    for family in ("scalar", "direct", "lif_direct_matched", "lif_rstdp"):
        payload = frozen["selected"][family]
        variant = Variant(
            **{
                **payload,
                "lif": LIFConfig(**payload["lif"]) if payload["lif"] else None,
            }
        )
        factual, withheld, cold = [], [], []
        for seed in (7101, 7102, 7103):
            records, h, p = generate(seed, "abrupt")
            modified = [
                {**r, "outcome": None, "feedback_at": None} if i % 120 >= 60 else r
                for i, r in enumerate(records)
            ]
            scores, _ = run_variant(variant, modified, h, p, scenario["scale"])
            original = read_rows(root / f"predictions/abrupt/{seed}-{family}.jsonl.gz")
            factual.extend(original)
            withheld.extend(scores)
            mask = np.asarray([i % 120 >= 60 for i in range(len(records))])
            fresh_rows = [
                {**r, "change_day": 0}
                for r, keep in zip(records, mask, strict=True)
                if keep
            ]
            fresh, _ = run_variant(
                variant, fresh_rows, h[mask], p[mask], scenario["scale"]
            )
            cold.extend(fresh)
            for a, b in zip(original, scores, strict=True):
                if a["day"] < 60 and a["probability"] != b["probability"]:
                    raise AssertionError("Контроль изменил прогноз до вмешательства")
        population = [
            {**r, "probability": r["p_population"], "ungated": r["p_population"]}
            for r in cold
        ]
        result["families"][family] = {
            "learning_vs_withheld": {
                str(n): paired_phase(factual, withheld, n) for n in (0, 1, 2, 5, 10, 20)
            },
            "cold_start_vs_population": {
                str(n): paired_phase(cold, population, n) for n in (0, 1, 2, 5, 10, 20)
            },
        }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print("counterfactual and cold-start diagnostics completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.run, args.output)


if __name__ == "__main__":
    main()
