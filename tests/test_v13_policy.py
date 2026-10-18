"""Регрессии причинности, лимитов и раздельных этапов политики предупреждений."""

import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from research_v13.evaluation import evaluate, paired_intervals
from research_v13.experiment import evaluate_frozen, freeze, select_policies
from research_v13.policy import AlertPolicy
from research_v13.types import PolicyConfig

START = datetime(2026, 1, 1, 16, 30, tzinfo=UTC)


def trajectory(user="u", days=14):
    return [
        {
            "user_id": user,
            "event_id": f"{user}:{day}",
            "decision_at": (START + timedelta(days=day)).isoformat(),
            "p_final": 0.9 if day % 7 in (0, 3) else 0.1,
            "evaluation_outcome": int(day % 7 in (0, 3)),
        }
        for day in range(days)
    ]


def test_sliding_window_exact_boundary_and_no_calendar_reset():
    policy = AlertPolicy(PolicyConfig(0.5))
    assert policy.decide(START, 0.9)["alert"]
    assert policy.decide(START + timedelta(hours=1), 0.9)["alert"]
    assert not policy.decide(START + timedelta(hours=167, minutes=59), 0.9)["alert"]
    assert policy.decide(START + timedelta(hours=168), 0.9)["alert"]
    assert not policy.decide(START + timedelta(hours=168, minutes=30), 0.9)["alert"]


def test_dynamic_reservation_and_no_future_information():
    policy = AlertPolicy(PolicyConfig(0.5, 0.2))
    assert policy.decide(START, 0.6)["alert"]
    result = policy.decide(START + timedelta(hours=1), 0.6)
    assert not result["alert"]
    assert result["threshold"] == pytest.approx(0.5 + 0.2 * 167 / 168)
    assert policy.decide(START + timedelta(hours=167), 0.6)["alert"]
    with pytest.raises(ValueError, match="возрастанию"):
        policy.decide(START, 0.9)


def test_outcomes_do_not_change_decisions_and_input_stays_unchanged():
    rows = trajectory()
    before = copy.deepcopy(rows)
    altered = [
        {**row, "evaluation_outcome": 1 - row["evaluation_outcome"]} for row in rows
    ]
    left = evaluate(rows, PolicyConfig(0.5, 0.1), days=14)
    right = evaluate(altered, PolicyConfig(0.5, 0.1), days=14)
    assert [r["alert"] for r in left["decisions"]] == [
        r["alert"] for r in right["decisions"]
    ]
    assert rows == before
    with pytest.raises(ValueError, match="исходы"):
        paired_intervals(left, right)


def test_metrics_budget_and_missing_outcomes_are_distinct():
    rows = trajectory(days=7)
    rows[0]["evaluation_outcome"] = None
    report = evaluate(rows, PolicyConfig(0.5), days=7)
    counts = report["pooled"]
    assert (counts["tp"], counts["fp"], counts["tn"], counts["fn"]) == (1, 0, 5, 0)
    assert counts["unknown_alerts"] == 1
    assert counts["alerts_per_user_week"] == 2
    assert counts["precision"] == 1
    assert report["label_scope"] == "observed_only_not_population_identified"
    with pytest.raises(ValueError, match="полных исходов"):
        select_policies(rows, days=7)
    quiet = evaluate(rows, PolicyConfig(1.01), days=7)
    assert quiet["pooled"]["precision"] is None


def test_daily_completeness_duplicate_and_user_isolation():
    rows = trajectory("a") + trajectory("b")
    report = evaluate(rows, PolicyConfig(0.5), days=14)
    assert report["per_user"]["a"]["tp"] == report["per_user"]["b"]["tp"] == 4
    with pytest.raises(ValueError, match="Повтор"):
        evaluate(rows + [rows[0]], PolicyConfig(0.5), days=14)
    with pytest.raises(ValueError, match="полные ежедневные"):
        evaluate(rows[:-1], PolicyConfig(0.5), days=14)
    intervals = paired_intervals(report, report, draws=100)
    assert intervals["contrasts"]["recall"]["ci95"] == [0.0, 0.0]


def test_precision_and_fpr_have_different_denominators():
    rows = [{**row, "p_final": 0.9} for row in trajectory(days=7)]
    report = evaluate(rows, PolicyConfig(0.5), days=7)
    counts = report["pooled"]
    assert (counts["tp"], counts["fp"], counts["tn"], counts["fn"]) == (1, 1, 4, 1)
    assert counts["precision"] == counts["recall"] == 0.5
    assert counts["fpr"] == 0.2
    assert counts["youden_j"] == pytest.approx(0.3)


def test_stop_when_no_policy_satisfies_false_alert_budget():
    rows = [{**row, "p_final": 0.9} for row in trajectory(days=7)]
    result = select_policies(rows, days=7)
    assert result["baseline"]["metrics"]["pooled"]["alerts"] == 0
    assert result["candidate"] is None
    assert result["status"] == "stop_no_feasible_candidate"


def test_freeze_then_evaluate_without_selection_or_overwrite(tmp_path, monkeypatch):
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    scores = {
        split: trajectory(split, days=180) for split in ("dev_1", "dev_2", "dev_3")
    }
    (stage2 / "search_summary.json").write_text(
        json.dumps({"status": "exploratory_replay_no_new_selection"})
    )
    score_path = stage2 / "dev_scores_all.json"
    score_path.write_text(json.dumps({"tuned_scalar_v12": scores}))
    freeze_dir = tmp_path / "frozen"
    freeze(stage2, freeze_dir)
    frozen_path = freeze_dir / "frozen_policy.json"
    frozen_before = frozen_path.read_bytes()
    frozen = json.loads(frozen_before)
    assert len(frozen["trials"]) == 88
    assert frozen["candidate"] is not None

    def forbidden_selection(*args, **kwargs):
        raise AssertionError("Оценка не должна повторять подбор")

    monkeypatch.setattr("research_v13.experiment.select_policies", forbidden_selection)
    out = tmp_path / "evaluation"
    evaluate_frozen(stage2, frozen_path, out)
    result = json.loads((out / "evaluation.json").read_text())
    assert result["mean_macro_recall_gain"] == 0
    assert not result["eligible_for_new_confirmatory_protocol"]
    assert frozen_path.read_bytes() == frozen_before
    with pytest.raises(FileExistsError):
        evaluate_frozen(stage2, frozen_path, out)
    scores["dev_2"][0]["p_final"] = 0.8
    score_path.write_text(json.dumps({"tuned_scalar_v12": scores}))
    with pytest.raises(ValueError, match="изменились после заморозки"):
        evaluate_frozen(stage2, frozen_path, tmp_path / "changed")
