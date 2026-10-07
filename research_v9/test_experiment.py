"""Критерии допуска V9 и изоляция персональной памяти от чужого baseline.

Тесты используют готовые искусственные scores, без fit популяционных моделей.
Исполнение поручено Луне; импорт не создаёт архивов и не запускает проверки.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_v8.archive import RunArchive
from research_v8.data_types import V8Record
from research_v8.evaluation import model_report, paired_comparison
from research_v8.information_audit import score_rows
from research_v8.replay import score_records
from research_v9.experiment import admission, choose_simple_baseline


POINT = "calibration_macro_fpr20"
START = datetime(2026, 1, 1, 16, 30, tzinfo=timezone.utc)
USERS = 20
WEEKS = 20


def population_rows(positive_days: tuple[int, ...] = (0, 1), split: str = "development") -> list[dict]:
    """Baseline пропускает последний положительный день каждой недели."""
    rows = []
    for person in range(USERS):
        user = f"person-{person:02d}"
        for day in range(WEEKS * 7):
            at = START + timedelta(days=day)
            outcome = int(day % 7 in positive_days)
            probability = .4 if day % 7 == max(positive_days) else .9 if outcome else .1
            rows.append({
                "model": "population", "split": split, "user_id": user,
                "event_id": f"{user}:{day}", "decision_at": at.isoformat(),
                "outcome": outcome, "observed_feedback": True,
                "probability": probability, "candidate": probability, "gamma": 0.0,
            })
    return rows


def improve_rows(rows: list[dict], recovered_weeks: int = WEEKS) -> list[dict]:
    """Исправляет пропущенные положительные дни, не добавляя ложных тревог."""
    result = deepcopy(rows)
    for row in result:
        day = (datetime.fromisoformat(row["decision_at"]) - START).days
        if row["outcome"] and day // 7 < recovered_weeks:
            row["probability"] = row["candidate"] = .9
    return result


def report(rows: list[dict], positive_days: int = 2) -> dict:
    """Реальные macro, bootstrap и rolling7 отчёты на одном фиксированном пороге."""
    return model_report(rows, {POINT: .5}, positive_days / 7)


def report_family(population: dict, anatomical: dict, scalar: dict | None = None, rewired: dict | None = None) -> dict:
    """Сохраняет заранее заданное имя единственного кандидата."""
    return {
        "population": population, "scalar": scalar if scalar is not None else population,
        "anatomical_population": population, "anatomical_personal": anatomical,
        "rewired_population": population, "rewired_personal": rewired if rewired is not None else anatomical,
    }


class SimpleBaselineSelectionTests(unittest.TestCase):
    """Выбор ограничен population/scalar и заранее заданными tie-break."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = population_rows(split="calibration")
        cls.base = report(cls.rows)
        cls.better = report(improve_rows(cls.rows))

    def test_higher_calibration_recall_wins_without_selecting_mb(self) -> None:
        """Простой контроль выбирается из двух семейств, даже если MB лучше обоих."""
        reports = report_family(self.base, self.better, scalar=self.base)
        self.assertEqual(choose_simple_baseline(reports), "population")
        reports["scalar"] = report(improve_rows(self.rows, recovered_weeks=2))
        self.assertEqual(choose_simple_baseline(reports), "scalar")

    def test_equal_recall_uses_brier_before_lexical_name(self) -> None:
        """Более уверенный пропуск остаётся пропуском, но имеет меньшую ошибку Brier."""
        improved_probabilities = deepcopy(self.rows)
        for row in improved_probabilities:
            if row["outcome"] and row["probability"] < .5:
                row["probability"] = row["candidate"] = .49
        scalar = report(improved_probabilities)
        self.assertEqual(scalar["operating_points"][POINT]["macro"]["recall"]["mean"],
                         self.base["operating_points"][POINT]["macro"]["recall"]["mean"])
        self.assertEqual(choose_simple_baseline(report_family(self.base, self.better, scalar=scalar)), "scalar")

    def test_exact_tie_uses_name_and_does_not_modify_reports(self) -> None:
        """При полном равенстве population выигрывает по имени; вход не меняется."""
        reports = report_family(self.base, self.better)
        before = deepcopy(reports)
        self.assertEqual(choose_simple_baseline(reports), "population")
        self.assertEqual(reports, before)


class AdmissionTests(unittest.TestCase):
    """Допуск требует совместного выполнения критериев относительно обоих controls."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = population_rows()
        cls.improved = improve_rows(cls.rows)
        cls.base = report(cls.rows)
        cls.good = report(cls.improved)

    def test_clear_paired_improvement_passes_all_checks(self) -> None:
        """Улучшение recall/J, Brier и охвата при прежнем FPR допускается."""
        result = admission(report_family(self.base, self.good), "population")
        self.assertTrue(result["eligible"])
        self.assertIsInstance(result["checks"], dict)
        self.assertTrue(result["checks"])
        self.assertIsInstance(result["comparisons"], dict)
        self.assertTrue(result["comparisons"])

    def test_candidate_must_beat_selected_scalar_as_well_as_population(self) -> None:
        """Преимущество над GBDT недостаточно при равенстве выбранному scalar."""
        reports = report_family(self.base, self.good, scalar=self.good)
        self.assertFalse(admission(reports, "scalar")["eligible"])

    def test_stronger_rewired_cannot_replace_preregistered_anatomical_candidate(self) -> None:
        """Контроль с идеальным результатом не спасает неизменившийся primary."""
        reports = report_family(self.base, self.base, rewired=self.good)
        self.assertFalse(admission(reports, "population")["eligible"])

    def test_small_positive_recall_gain_does_not_meet_five_percentage_points(self) -> None:
        """Один дополнительный исход из40 даёт положительный CI, но только2.5п.п."""
        small = report(improve_rows(self.rows, recovered_weeks=1))
        comparison = paired_comparison(small, self.base)
        self.assertGreater(comparison["recall_delta"]["ci95"][0], 0)
        self.assertAlmostEqual(comparison["recall_delta"]["mean"], .025)
        self.assertFalse(admission(report_family(self.base, small), "population")["eligible"])

    def test_mean_improvement_without_positive_paired_ci_is_rejected(self) -> None:
        """Средний выигрыш10п.п. не скрывает противоположные эффекты между людьми."""
        heterogeneous = deepcopy(self.improved)
        for row in heterogeneous:
            if int(row["user_id"].rsplit("-", 1)[1]) >= 12 and row["outcome"]:
                row["probability"] = row["candidate"] = .4
        candidate = report(heterogeneous)
        comparison = paired_comparison(candidate, self.base)
        self.assertGreaterEqual(comparison["recall_delta"]["mean"], .05)
        self.assertLessEqual(comparison["recall_delta"]["ci95"][0], 0)
        self.assertLessEqual(comparison["youden_j_delta"]["ci95"][0], 0)
        self.assertFalse(admission(report_family(self.base, candidate), "population")["eligible"])

    def test_fpr_increase_above_one_point_fails_even_below_twenty_percent(self) -> None:
        """Нельзя купить большой recall приростом FPR2п.п. при абсолютном FPR2%."""
        extra_alerts = deepcopy(self.improved)
        for row in extra_alerts:
            day = (datetime.fromisoformat(row["decision_at"]) - START).days
            if day in (2, 9):
                row["probability"] = row["candidate"] = .6
        candidate = report(extra_alerts)
        point = candidate["operating_points"][POINT]
        self.assertAlmostEqual(point["macro"]["false_positive_rate"]["mean"], .02)
        self.assertGreater(point["macro"]["recall"]["mean"], .9)
        self.assertFalse(admission(report_family(self.base, candidate), "population")["eligible"])

    def test_exact_fpr_margin_is_allowed_when_other_checks_pass(self) -> None:
        """Ровно1п.п. FPR включён в допуск; поздняя лишняя тревога не занимает пуш."""
        extra_alerts = deepcopy(self.improved)
        for row in extra_alerts:
            if row["decision_at"] == (START + timedelta(days=2)).isoformat():
                row["probability"] = row["candidate"] = .6
        candidate = report(extra_alerts)
        self.assertAlmostEqual(candidate["operating_points"][POINT]["macro"]["false_positive_rate"]["mean"], .01)
        self.assertTrue(admission(report_family(self.base, candidate), "population")["eligible"])

    def test_perfect_detection_with_worse_probability_quality_is_rejected(self) -> None:
        """Верные стороны порога не компенсируют плохой Brier и отрицательный BSS."""
        uncertain = deepcopy(self.improved)
        for row in uncertain:
            row["probability"] = row["candidate"] = .51 if row["outcome"] else .49
        candidate = report(uncertain)
        point = candidate["operating_points"][POINT]
        self.assertEqual(point["macro"]["recall"]["mean"], 1)
        self.assertEqual(point["macro"]["false_positive_rate"]["mean"], 0)
        self.assertLess(point["pooled"]["bss_training_prevalence"], 0)
        self.assertFalse(admission(report_family(self.base, candidate), "population")["eligible"])

    def test_individual_auc_degradation_limit_is_inclusive_and_blocks_excess(self) -> None:
        """Один из20 ухудшившихся допустим; два из20 запрещены при хорошем среднем."""
        for degraded_users, expected in ((1, True), (2, False)):
            rows = deepcopy(self.improved)
            for row in rows:
                user = int(row["user_id"].rsplit("-", 1)[1])
                day = (datetime.fromisoformat(row["decision_at"]) - START).days
                if user < degraded_users and day % 7 == 1:
                    row["probability"] = row["candidate"] = .05
            candidate = report(rows)
            comparison = paired_comparison(candidate, self.base)
            with self.subTest(degraded_users=degraded_users):
                self.assertAlmostEqual(comparison["individual_auc_degradation"]["fraction"], degraded_users / USERS)
                self.assertEqual(admission(report_family(self.base, candidate), "population")["eligible"], expected)

    def test_classifier_gain_without_notification_gain_is_rejected(self) -> None:
        """При трёх событиях за неделю третий найденный день не увеличивает лимит2."""
        rows = population_rows(positive_days=(0, 1, 2))
        population, candidate = report(rows, 3), report(improve_rows(rows), 3)
        base_point, point = population["operating_points"][POINT], candidate["operating_points"][POINT]
        self.assertGreater(point["macro"]["recall"]["mean"], base_point["macro"]["recall"]["mean"])
        self.assertEqual(point["notifications"]["2"]["recall"], base_point["notifications"]["2"]["recall"])
        self.assertFalse(admission(report_family(population, candidate), "population")["eligible"])

    def test_increased_false_notifications_per_caught_event_blocks_admission(self) -> None:
        """Ложная ранняя тревога повышает burden даже при допустимом FPR1%."""
        rows = population_rows(positive_days=(2, 3))
        improved = improve_rows(rows)
        for row in improved:
            if row["decision_at"] == START.isoformat():
                row["probability"] = row["candidate"] = .6
        population, candidate = report(rows), report(improved)
        base_point, point = population["operating_points"][POINT], candidate["operating_points"][POINT]
        self.assertAlmostEqual(point["macro"]["false_positive_rate"]["mean"], .01)
        self.assertGreater(point["notifications"]["2"]["recall"], base_point["notifications"]["2"]["recall"])
        self.assertGreater(point["notifications"]["2"]["false_alarms_per_caught"], 0)
        self.assertFalse(admission(report_family(population, candidate), "population")["eligible"])


def memory_records(user: str, outcome: int) -> list[V8Record]:
    """Исходный p_population намеренно не совпадает с передаваемым MB baseline."""
    return [V8Record(
        split="development", user_id=user, target="evening_relapse", event_id=f"{user}:{day}",
        decision_at=START + timedelta(days=day), available_at=START + timedelta(days=day),
        pn=[.5] * 16, p_population=.02, outcome=outcome, evaluation_outcome=outcome,
        evaluation_probability=.5, feedback_at=START + timedelta(days=day, hours=20),
    ) for day in range(4)]


class OwnPopulationReplayTests(unittest.TestCase):
    """Предобученная MB задаёт собственный p0; личные состояния не смешиваются."""

    def test_interleaved_users_match_isolated_replay_and_other_user_cannot_change_memory(self) -> None:
        """Обучение второго человека не меняет ни один прогноз первого."""
        alice, bob = memory_records("alice", 1), memory_records("bob", 0)
        combined = [record for pair in zip(alice, bob, strict=True) for record in pair]
        changed = [row.model_copy(update={"outcome": 1, "evaluation_outcome": 1})
                   if row.user_id == "bob" else row for row in combined]
        with tempfile.TemporaryDirectory() as directory:
            def run(name: str, records: list[V8Record]) -> list[dict]:
                archive = RunArchive(Path(directory) / name, {"test": name})
                return score_records(records, np.ones((len(records), 1)), np.full(len(records), .7),
                                     "anatomical_personal", "fixed-mb", archive, "development")[0]
            separate, mixed, poisoned = run("separate", alice), run("mixed", combined), run("changed", changed)
        for scores in (mixed, poisoned):
            selected = [row for row in scores if row["user_id"] == "alice"]
            for key in ("probability", "candidate", "p_learning", "gamma", "familiarity"):
                self.assertEqual([row[key] for row in separate], [row[key] for row in selected])
        self.assertEqual(separate[0]["probability"], .7)
        self.assertGreater(separate[1]["p_learning"], .7)
        self.assertLess([row for row in mixed if row["user_id"] == "bob"][1]["p_learning"], .7)

    def test_own_mb_baseline_ignores_record_gbdt_probability_and_evaluator_truth(self) -> None:
        """Смена чужого offset и latent truth не влияет на MB прогноз или обновление."""
        original = memory_records("alice", 1)
        altered = [row.model_copy(update={"p_population": .98, "evaluation_probability": .001,
                                          "evaluation_outcome": 0}) for row in original]
        with tempfile.TemporaryDirectory() as directory:
            def run(name: str, records: list[V8Record], p0: float) -> list[dict]:
                archive = RunArchive(Path(directory) / name, {"test": name})
                return score_records(records, np.ones((len(records), 1)), np.full(len(records), p0),
                                     "anatomical_personal", "fixed-mb", archive, "development")[0]
            first, second, different_mb = run("first", original, .75), run("altered", altered, .75), run("own", original, .25)
        for key in ("probability", "candidate", "p_learning", "gamma", "familiarity", "p_population"):
            self.assertEqual([row[key] for row in first], [row[key] for row in second])
        self.assertTrue(all(row["p_population"] == .75 for row in first))
        self.assertEqual(first[0]["probability"], .75)
        self.assertEqual(different_mb[0]["probability"], .25)
        self.assertNotEqual(first[1]["p_learning"], different_mb[1]["p_learning"])

    def test_population_only_view_preserves_p0_without_personal_updates(self) -> None:
        """p0-вариант использует тот же предобученный прогноз без личной памяти."""
        records = memory_records("alice", 1)
        probabilities = np.asarray([.72, .61, .75, .63])
        view = score_rows(records, probabilities, name="anatomical_population")
        with tempfile.TemporaryDirectory() as directory:
            archive = RunArchive(Path(directory) / "p0", {"test": "p0"})
            replay, _ = score_records(records, None, probabilities, "anatomical_population",
                                      "fixed-mb-p0", archive, "development")
        self.assertEqual([row["probability"] for row in view], probabilities.tolist())
        for key in ("probability", "candidate", "p_learning", "p_population"):
            self.assertEqual([row[key] for row in replay], probabilities.tolist())
        self.assertTrue(all(row["gamma"] == 0 and row["familiarity"] == 0 for row in replay))


if __name__ == "__main__":
    unittest.main()
