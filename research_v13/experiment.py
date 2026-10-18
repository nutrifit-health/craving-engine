"""Два отдельных этапа: выбор политики на dev_1 и поисковая оценка dev_2/dev_3."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from research_v13.evaluation import evaluate, paired_intervals
from research_v13.types import PolicyConfig

PROTOCOL = "https://gov.nutrifit.health/documents/6ad2e714556a2264e7a13682/versions/1"
BASE_GRID = tuple(i / 20 for i in range(21)) + (1.01,)
PENALTIES = (0.05, 0.10, 0.20)
FAMILY = "tuned_scalar_v12"
EPS = 1e-12


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hashes() -> dict:
    root = Path(__file__).parent
    return {
        name: _sha(root / name)
        for name in ("types.py", "policy.py", "evaluation.py", "experiment.py")
    }


def _load(stage2_dir: Path) -> tuple[dict, dict]:
    summary_path = stage2_dir / "search_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "exploratory_replay_no_new_selection":
        raise ValueError(
            "Требуется исправленный replay V12; исторический прогон не подходит"
        )
    score_path = stage2_dir / "dev_scores_all.json"
    scores = json.loads(score_path.read_text())[FAMILY]
    if set(scores) != {"dev_1", "dev_2", "dev_3"}:
        raise ValueError("Нужны три development-когорты; holdout здесь не используется")
    seen_users = set()
    for split in ("dev_1", "dev_2", "dev_3"):
        users = {row["user_id"] for row in scores[split]}
        if users & seen_users:
            raise ValueError("Пользователи development-когорт пересекаются")
        seen_users.update(users)
    provenance = {
        "scores_sha256": _sha(score_path),
        "summary_sha256": _sha(summary_path),
    }
    return scores, provenance


def _write(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "decisions"}


def _rank(candidate: dict) -> tuple:
    report, cfg = candidate["metrics"], candidate["config"]
    return (
        report["macro"]["recall"],
        -report["macro"]["fpr"],
        -report["pooled"]["alerts_per_user_week"],
        -cfg["reserve_penalty"],
        cfg["base_threshold"],
    )


def _budget_ok(candidate: dict, baseline: dict) -> bool:
    cp, bp = candidate["pooled"], baseline["pooled"]
    return (
        candidate["macro"]["fpr"] <= min(0.15, baseline["macro"]["fpr"]) + EPS
        and cp["alerts_per_user_week"] <= bp["alerts_per_user_week"] + EPS
        and cp["precision"] is not None
        and bp["precision"] is not None
        and cp["precision"] + EPS >= bp["precision"]
    )


def select_policies(rows: list[dict], days: int = 180) -> dict:
    """Полная фиксированная сетка; получатель передаёт только tuning-строки."""
    trials = []
    for penalty in (0.0, *PENALTIES):
        for base in BASE_GRID:
            cfg = PolicyConfig(base, penalty)
            metrics = _compact(evaluate(rows, cfg, days))
            if (
                metrics["pooled"]["unknown"]
                or metrics["macro"]["recall"] is None
                or metrics["macro"]["fpr"] is None
            ):
                raise ValueError("Выбор требует полных исходов обоих классов")
            trials.append({"config": asdict(cfg), "metrics": metrics})
    baseline = max(
        (
            r
            for r in trials
            if r["config"]["reserve_penalty"] == 0
            and r["metrics"]["macro"]["fpr"] <= 0.15 + EPS
        ),
        key=_rank,
    )
    allowed = [
        r
        for r in trials
        if r["config"]["reserve_penalty"] > 0
        and _budget_ok(r["metrics"], baseline["metrics"])
    ]
    candidate = max(allowed, key=_rank) if allowed else None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "trials": trials,
        "status": "frozen_for_exploratory_evaluation"
        if candidate
        else "stop_no_feasible_candidate",
    }


def freeze(stage2_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    scores, provenance = _load(stage2_dir)
    # Не использовать dev_2/dev_3 даже для диагностики/выбора порогов.
    selection = select_policies(scores["dev_1"])
    payload = {
        "protocol": PROTOCOL,
        "family": FAMILY,
        "days_per_user": 180,
        "selection_split": "dev_1",
        "evaluation_splits": ["dev_2", "dev_3"],
        "provenance": provenance,
        "implementation_sha256": _implementation_hashes(),
        "created_at": datetime.now(UTC).isoformat(),
        **selection,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "frozen_policy.json", payload)


def evaluate_frozen(stage2_dir: Path, frozen_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    scores, provenance = _load(stage2_dir)
    frozen = json.loads(frozen_path.read_text())
    if frozen["implementation_sha256"] != _implementation_hashes():
        raise ValueError("Реализация изменилась после заморозки; нужен отдельный аудит")
    if frozen["provenance"] != provenance:
        raise ValueError("Прогнозы или итог V12 изменились после заморозки")
    if (
        frozen["protocol"] != PROTOCOL
        or frozen["family"] != FAMILY
        or frozen["days_per_user"] != 180
        or frozen["selection_split"] != "dev_1"
        or frozen["evaluation_splits"] != ["dev_2", "dev_3"]
    ):
        raise ValueError("Изменён контракт замороженного эксперимента")
    if frozen["candidate"] is None:
        raise ValueError("Сработала остановка: допустимый кандидат отсутствует")
    configurations = {
        key: PolicyConfig(**frozen[key]["config"]) for key in ("baseline", "candidate")
    }
    if (
        configurations["baseline"].reserve_penalty != 0
        or configurations["candidate"].reserve_penalty not in PENALTIES
        or any(c.base_threshold not in BASE_GRID for c in configurations.values())
    ):
        raise ValueError("Конфигурация вне зафиксированной сетки")
    reports, decisions, gains, admissible = {}, {}, [], []
    for split in ("dev_2", "dev_3"):
        results = {
            name: evaluate(scores[split], cfg) for name, cfg in configurations.items()
        }
        baseline, candidate = results["baseline"], results["candidate"]
        for result in results.values():
            if (
                result["pooled"]["unknown"]
                or result["macro"]["recall"] is None
                or result["macro"]["fpr"] is None
            ):
                raise ValueError("Оценка требует полных исходов обоих классов")
        delta = candidate["macro"]["recall"] - baseline["macro"]["recall"]
        gains.append(delta)
        admissible.append(_budget_ok(candidate, baseline) and delta > 0)
        reports[split] = {
            **{name: _compact(result) for name, result in results.items()},
            "paired": paired_intervals(baseline, candidate),
            "passes_seed_constraints": admissible[-1],
        }
        decisions[split] = {
            name: result["decisions"] for name, result in results.items()
        }
    mean_gain = sum(gains) / len(gains)
    report = {
        "protocol": PROTOCOL,
        "status": "exploratory_reused_development_only",
        "provenance": provenance,
        "frozen_policy_sha256": _sha(frozen_path),
        "mean_macro_recall_gain": mean_gain,
        "eligible_for_new_confirmatory_protocol": all(admissible)
        and mean_gain + EPS >= 0.05,
        "confirmation_run": False,
        "production_ready": False,
        "splits": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "evaluation.json", report)
    _write(output_dir / "decisions.json", decisions)
    _write(output_dir / "frozen_policy.json", frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.frozen_policy is not None:
            parser.error("freeze не принимает готовую политику")
        freeze(args.stage2_dir, args.output_dir)
    else:
        if args.frozen_policy is None:
            parser.error("evaluate требует --frozen-policy")
        evaluate_frozen(args.stage2_dir, args.frozen_policy, args.output_dir)
