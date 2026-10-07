"""Изоляция пользователей и fold-local MB; запуск поручен Luna."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.special import expit
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from research_v8.context_representation import ContextTransform, FixedContextRepresentation
from research_v9.models import MBClassifier, calibration_plan, fit_model, save_model_artifacts


def sample_schema() -> list[dict]:
    """Один наблюдаемый признак без скрытых ID и меток в X."""
    return [{"name": "amount", "unit": "g", "semantics": "Тестовый измеренный сигнал", "available_if": None}]


def small_graph() -> np.ndarray:
    """Два PN покрыты разными KC; top-5% оставляет один положительный победитель."""
    return np.tile(np.eye(2), (10, 1))


def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Все три пользователя имеют оба класса, без генерации benchmark-корпуса."""
    values = np.asarray([-3., -2., -1., -.5, .5, 1., 2., 3.])
    X = np.tile(values, 3).reshape(-1, 1)
    return X, (X[:, 0] > 0).astype(int), np.repeat(["alpha", "beta", "gamma"], len(values))


class PopulationMBTests(unittest.TestCase):
    """Readout и sigmoid проверяются отдельно от персональной online-памяти."""

    def test_explicit_folds_cover_rows_once_and_exclude_whole_users(self) -> None:
        """Сохранённый порядок indices не заменяется новым автоматическим split."""
        X, y, groups = sample_data()
        folds, _ = calibration_plan(X, y, groups)
        explicit = [(fit[::-1], cal[::-1]) for fit, cal in folds[::-1]]
        copied, audit = calibration_plan(X, y, groups, explicit)
        coverage = np.zeros(len(X), dtype=int)
        for (actual_fit, actual_cal), (expected_fit, expected_cal), membership in zip(copied, explicit, audit, strict=True):
            np.testing.assert_array_equal(actual_fit, expected_fit)
            np.testing.assert_array_equal(actual_cal, expected_cal)
            self.assertFalse(set(membership["fit_users"]) & set(membership["calibration_users"]))
            self.assertEqual(set(membership["fit_users"]) | set(membership["calibration_users"]), set(groups))
            coverage[actual_cal] += 1
        np.testing.assert_array_equal(coverage, 1)
        copied[0][0][0] = 999
        self.assertNotEqual(explicit[0][0][0], 999)

    def test_fold_plan_rejects_row_loss_duplicates_user_overlap_and_single_class(self) -> None:
        """Повреждённый fold не исправляется отбором данных или новым seed."""
        X, y, groups = sample_data()
        folds, _ = calibration_plan(X, y, groups)
        missing = deepcopy(folds)
        missing[0] = (missing[0][0][1:], missing[0][1])
        duplicate = deepcopy(folds)
        duplicate[0] = (duplicate[0][0], np.append(duplicate[0][1], duplicate[0][1][0]))
        repeated = deepcopy(folds)
        repeated[1] = deepcopy(repeated[0])
        crossed = deepcopy(folds)
        crossed[0][0][0], crossed[0][1][0] = crossed[0][1][0], crossed[0][0][0]
        fractional = deepcopy(folds)
        fractional[0] = (fractional[0][0].astype(float), fractional[0][1])
        for invalid in (missing, duplicate, repeated, crossed, fractional, folds[:2]):
            with self.subTest(folds=len(invalid)), self.assertRaises(ValueError):
                calibration_plan(X, y, groups, invalid)
        single_class = y.copy()
        single_class[folds[0][1]] = 0
        with self.assertRaises(ValueError):
            calibration_plan(X, single_class, groups, folds)

    def test_calibration_user_changes_do_not_enter_that_fold_normalization_or_readout(self) -> None:
        """Внешние строки меняют sigmoid своего fold, но не его training-код."""
        X, y, groups = sample_data()
        X[groups == "beta"] += 10
        X[groups == "gamma"] += 100
        folds, _ = calibration_plan(X, y, groups)
        first, metadata = fit_model("mb", X, y, groups, sample_schema(), small_graph(), folds)
        changed_X, changed_y = X.copy(), y.copy()
        changed_X[folds[0][1]] += 500
        changed_y[folds[0][1]] = 1 - changed_y[folds[0][1]]
        second, _ = fit_model("mb", changed_X, changed_y, groups, sample_schema(), small_graph(), folds)
        for calibrated, (fit_indices, _), state in zip(first.calibrated_classifiers_, folds, metadata["folds"], strict=True):
            expected = ContextTransform.fit(X[fit_indices], sample_schema())
            self.assertEqual(calibrated.estimator.transform_.to_dict(), expected.to_dict())
            np.testing.assert_allclose(state["transform"]["state"]["mean"], X[fit_indices].mean(axis=0))
            self.assertEqual(state["representation"]["geometry"]["training_rows"], len(fit_indices))
        before, after = first.calibrated_classifiers_[0].estimator, second.calibrated_classifiers_[0].estimator
        self.assertEqual(before.transform_.to_dict(), after.transform_.to_dict())
        self.assertEqual(before.representation_.representation_id, after.representation_.representation_id)
        np.testing.assert_array_equal(before.readout_.coef_, after.readout_.coef_)
        np.testing.assert_array_equal(before.readout_.intercept_, after.readout_.intercept_)

    def test_standalone_mb_learns_signal_and_calibrates_raw_logits_without_offset(self) -> None:
        """Сохранённые sigmoid воспроизводят среднее вероятностей трёх readout."""
        X, y, groups = sample_data()
        model, metadata = fit_model("mb", X, y, groups, sample_schema(), small_graph())
        probabilities = model.predict_proba(X)
        np.testing.assert_array_equal(probabilities[:, 1] >= .5, y.astype(bool))
        self.assertGreater(probabilities[y == 1, 1].mean() - probabilities[y == 0, 1].mean(), .3)
        fold_probabilities = []
        for calibrated, state in zip(model.calibrated_classifiers_, metadata["folds"], strict=True):
            estimator = calibrated.estimator
            logits = estimator.decision_function(X)
            np.testing.assert_allclose(estimator.predict_proba(X)[:, 1], expit(logits), rtol=1e-13, atol=1e-13)
            sigmoid = state["sigmoid"]
            fold_probabilities.append(expit(-(sigmoid["a"] * logits + sigmoid["b"])))
            self.assertTrue(is_classifier(estimator))
            self.assertIsInstance(clone(estimator), MBClassifier)
        np.testing.assert_allclose(probabilities[:, 1], np.mean(fold_probabilities, axis=0), rtol=1e-13, atol=1e-13)
        self.assertFalse(metadata["gbdt_offset_used"])
        self.assertFalse(metadata["global_head_used"])
        self.assertEqual(metadata["calibration"]["mapping_seed"], 11)
        self.assertTrue(metadata["calibration"]["ensemble"])

    def test_artifacts_roundtrip_preserves_predictions_parameters_and_frozen_preprocessing(self) -> None:
        """Архивный JSON и NPZ воспроизводят fit; существующий каталог не перезаписывается."""
        X, y, groups = sample_data()
        model, metadata = fit_model("mb", X, y, groups, sample_schema(), small_graph())
        expected = model.predict_proba(X)
        frozen = deepcopy(model.research_metadata_)
        metadata["folds"][0]["readout"]["coef"][0][0] = 999
        model.predict_proba(np.asarray([[10000.], [-10000.]]))
        self.assertEqual(model.research_metadata_, frozen)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            manifest = save_model_artifacts(model, output)
            for relative, entry in manifest["files"].items():
                content = (output / relative).read_bytes()
                self.assertEqual(len(content), entry["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
            # Только pickle, созданный самим тестом и уже сверенный с manifest.
            restored = pickle.loads((output / "model.pickle").read_bytes())
            np.testing.assert_array_equal(restored.predict_proba(X), expected)
            transform = ContextTransform.from_dict(json.loads((output / "fold_0_transform.json").read_bytes()))
            representation = FixedContextRepresentation.load(output / "fold_0_representation.npz")
            original = model.calibrated_classifiers_[0].estimator
            np.testing.assert_array_equal(representation.encode(transform.transform(X)), original._codes(X))
            self.assertEqual(transform.to_dict(), original.transform_.to_dict())
            self.assertFalse((output / ".incomplete").exists())
            with self.assertRaises(FileExistsError):
                save_model_artifacts(model, output)

    def test_convergence_warning_is_fatal_and_invalid_input_never_reaches_readout(self) -> None:
        """Ни предупреждение, ни отсутствующая метка не запускают скрытый подбор."""
        X, y, groups = sample_data()

        def failed_fit(*args: object, **kwargs: object) -> None:
            warnings.warn("Тестовая несходимость readout", ConvergenceWarning)

        with patch.object(LogisticRegression, "fit", side_effect=failed_fit) as fit:
            with self.assertRaises(ConvergenceWarning):
                fit_model("mb", X, y, groups, sample_schema(), small_graph())
            self.assertEqual(fit.call_count, 1)
        with patch.object(LogisticRegression, "fit") as fit:
            null_labels = y.astype(object)
            null_labels[0] = None
            invalid_X = X.copy()
            invalid_X[0] = np.nan
            for values, labels in ((invalid_X, y), (X, null_labels), (X, np.ones(len(y))), (X, y[:-1])):
                with self.assertRaises(ValueError):
                    fit_model("mb", values, labels, groups, sample_schema(), small_graph())
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
