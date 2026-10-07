"""Популяционный MB без GBDT-offset: fold-local код и логистический readout.

Обучение получает только observed X/y и группы пользователей. Внешний caller
отвечает за отбор training и исключение отсутствующего reported outcome.
Калибровка и усреднение трёх моделей не используют внешнюю calibration-выборку.
Полный training-код для online-памяти обучается отдельно в runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

from research_v8.context_representation import ContextTransform, FixedContextRepresentation
from research_v9.types import CalibrationFolds, MBCalibrationConfig, MBReadoutConfig, ModelKind


FORMAT_VERSION = 1


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _array_hash(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8")
    return _sha(np.asarray(array.shape, dtype="<i8").tobytes() + array.tobytes(order="C"))


def _features(X: np.ndarray, columns: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Ожидалась числовая матрица признаков") from error
    if (result.ndim != 2 or result.shape[1] < 1 or not np.isfinite(result).all()
            or (columns is not None and result.shape[1] != columns)):
        raise ValueError("Неверная форма либо нечисловые признаки")
    return result


def _labels(y: np.ndarray, rows: int) -> np.ndarray:
    result = np.asarray(y)
    if (result.ndim != 1 or len(result) != rows or not rows
            or not np.all(np.isin(result, [0, 1])) or set(result.tolist()) != {0, 1}):
        raise ValueError("Нужны наблюдаемые бинарные метки обоих классов без пропусков")
    return result.astype(np.int64, copy=True)


def _groups(groups: np.ndarray, rows: int) -> np.ndarray:
    result = np.asarray(groups)
    if (result.ndim != 1 or len(result) != rows
            or any(not isinstance(group, str) or not group for group in result.tolist())
            or len(set(result.tolist())) < MBCalibrationConfig().folds):
        raise ValueError("Нужны строковые ID минимум трёх training-пользователей")
    return result.copy()


def _indices(value: object, rows: int) -> np.ndarray:
    indices = np.asarray(value)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= rows)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Fold требует уникальные целые индексы в пределах training")
    return indices.astype(np.int64, copy=True)


def calibration_plan(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: CalibrationFolds | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    """Проверяет весь план до первого fit, не исправляя неудачные fold."""
    values = _features(X)
    labels = _labels(y, len(values))
    people = _groups(groups, len(values))
    supplied = list(GroupKFold(n_splits=3).split(values, labels, people)) if folds is None else list(folds)
    if len(supplied) != 3:
        raise ValueError("Протокол требует ровно три calibration fold")
    plan, membership = [], []
    coverage = np.zeros(len(values), dtype=int)
    universe = set(range(len(values)))
    for fold, pair in enumerate(supplied):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Fold должен содержать fit/calibration индексы")
        fit_indices, cal_indices = (_indices(part, len(values)) for part in pair)
        fit_users, cal_users = set(people[fit_indices].tolist()), set(people[cal_indices].tolist())
        if (set(fit_indices) & set(cal_indices) or set(fit_indices) | set(cal_indices) != universe
                or fit_users & cal_users
                or any(set(labels[index].tolist()) != {0, 1} for index in (fit_indices, cal_indices))):
            raise ValueError("Fold пересекает пользователей, теряет строки либо не содержит оба класса")
        coverage[cal_indices] += 1
        plan.append((fit_indices, cal_indices))
        membership.append({
            "fold": fold, "fit_indices": fit_indices.tolist(), "calibration_indices": cal_indices.tolist(),
            "fit_users": sorted(fit_users), "calibration_users": sorted(cal_users),
            "fit_rows": len(fit_indices), "calibration_rows": len(cal_indices),
        })
    if not np.all(coverage == 1):
        raise ValueError("Каждая training-строка должна ровно один раз калибровать исключившую её модель")
    return plan, membership


class MBClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn estimator: нормировка и RMS принадлежат только его fit-строкам."""

    def __init__(self, schema: list[dict], graph: np.ndarray) -> None:
        # Параметры конструктора не копируются и не меняются: это контракт clone.
        self.schema = schema
        self.graph = graph

    def fit(self, X: np.ndarray, y: np.ndarray) -> MBClassifier:
        values = _features(X)
        labels = _labels(y, len(values))
        if self.graph is None:
            raise ValueError("Standalone MB требует явный ALPN→KC граф")
        transform = ContextTransform.fit(values, self.schema)
        q = transform.transform(values)
        representation = FixedContextRepresentation.fit(
            q, self.graph, MBCalibrationConfig().mapping_seed,
        )
        readout = LogisticRegression(**asdict(MBReadoutConfig()))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            readout.fit(representation.encode(q), labels)
        self.transform_ = transform
        self.representation_ = representation
        self.readout_ = readout
        self.classes_ = readout.classes_.copy()
        self.n_features_in_ = values.shape[1]
        return self

    def _codes(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("transform_", "representation_", "readout_", "classes_", "n_features_in_"))
        values = _features(X, self.n_features_in_)
        return self.representation_.encode(self.transform_.transform(values))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Platt получает исходный logit readout, а не predict_proba."""
        codes = self._codes(X)
        return self.readout_.decision_function(codes) if len(codes) else np.empty(0, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        codes = self._codes(X)
        return self.readout_.predict_proba(codes) if len(codes) else np.empty((0, 2), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        return self.classes_[(logits > 0).astype(int)]


def _fold_state(calibrated: object, membership: dict) -> dict:
    estimator = calibrated.estimator
    if not isinstance(estimator, MBClassifier) or len(calibrated.calibrators) != 1:
        raise ValueError("Экспорт ожидает бинарный MB и один sigmoid на fold")
    sigmoid = calibrated.calibrators[0]
    return {
        **membership,
        "transform": estimator.transform_.to_dict(),
        "representation": {
            "representation_id": estimator.representation_.representation_id,
            "scale": estimator.representation_.scale,
            "geometry": estimator.representation_.geometry,
            "mapping_sha256": _array_hash(estimator.representation_.mapping),
            "graph_sha256": _array_hash(estimator.graph),
        },
        "readout": {
            "parameters": estimator.readout_.get_params(deep=False),
            "coef": estimator.readout_.coef_.tolist(), "intercept": estimator.readout_.intercept_.tolist(),
            "classes": estimator.classes_.tolist(), "n_iter": estimator.readout_.n_iter_.tolist(),
            "input": "h=L2([1, positiveTopK(WrowL2 A q)/training_RMS])",
            "anchor_is_context_coordinate": True, "separate_intercept": True,
        },
        "sigmoid": {"a": float(sigmoid.a_), "b": float(sigmoid.b_),
                    "formula": "P(y=1)=expit(-(a*raw_readout_logit+b))"},
    }


def fit_model(
    kind: ModelKind, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    schema: list[dict], graph: np.ndarray, folds: CalibrationFolds | None = None,
) -> tuple[CalibratedClassifierCV, dict]:
    """Обучает standalone MB; GBDT baseline повторно берётся runner из V8.3."""
    if kind != "mb":
        raise ValueError("Этот fit поддерживает только mb; GBDT V8.3 должен переиспользоваться без fit")
    values = _features(X)
    labels = _labels(y, len(values))
    people = _groups(groups, len(values))
    plan, membership = calibration_plan(values, labels, people, folds)
    config = MBCalibrationConfig()
    model = CalibratedClassifierCV(
        MBClassifier(schema=schema, graph=graph), method=config.method, cv=plan,
        ensemble=config.ensemble, n_jobs=config.n_jobs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(values, labels)
    if len(model.calibrated_classifiers_) != config.folds or not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Калиброванный ансамбль не соответствует бинарному протоколу из трёх fold")
    metadata = {
        "format_version": FORMAT_VERSION, "kind": kind, "training_rows": len(values),
        "training_users": len(set(people.tolist())), "feature_count": values.shape[1],
        "schema": deepcopy(schema), "observed_training_prevalence": float(labels.mean()),
        "input_sha256": {"X": _array_hash(values), "y": _array_hash(labels),
                         "groups": _sha(_json_bytes(people.tolist())), "schema": _sha(_json_bytes(schema)),
                         "graph": _array_hash(graph)},
        "calibration": {**asdict(config), "split_source": "explicit_archived_indices" if folds is not None else "GroupKFold3",
                        "score_source": "decision_function_raw_logit", "prediction": "mean of three sigmoid probabilities",
                        "training_user_weighting": "none; each observed row has weight1", "class_weight": None},
        "folds": [_fold_state(calibrated, audit) for calibrated, audit in zip(model.calibrated_classifiers_, membership, strict=True)],
        "source_sha256": {str(path): _sha(path.read_bytes()) for path in (
            Path(__file__), Path(__file__).with_name("types.py"),
            Path(__file__).parents[1] / "research_v8/context_representation.py",
            Path(__file__).parents[1] / "research_v8/context_types.py",
        )},
        "gbdt_offset_used": False, "global_head_used": False,
        "full_training_online_representation_included": False,
        "convergence_warning_policy": "error; no retry or parameter tuning",
        "limitations": [
            "MB и GBDT используют разные классы моделей; одинаковые folds не уравнивают ёмкость.",
            "Внутренние sigmoid обучены на пользователях, исключённых из соответствующего readout и нормировки.",
            "Итоговый ансамбль включает другие fold-модели, обученные на этих пользователях: его training-прогнозы не являются OOF.",
            "Архивируется модель наблюдаемых synthetic outcome; это не подтверждение качества на реальных пользователях.",
        ],
    }
    # Независимая копия позволяет caller дополнять метаданные, не меняя архивную модель.
    model.research_metadata_ = deepcopy(metadata)
    return model, metadata


def save_model_artifacts(model: CalibratedClassifierCV, path: Path) -> dict:
    """Новый каталог: JSON параметров, NPZ представлений и trusted-local pickle.

    Pickle нельзя загружать из непроверенного источника. Проверка manifest должна
    предшествовать его загрузке; этот модуль не предоставляет внешнего unpickle.
    """
    check_is_fitted(model, ("calibrated_classifiers_", "research_metadata_"))
    metadata = deepcopy(model.research_metadata_)
    if len(model.calibrated_classifiers_) != 3 or metadata.get("kind") != "mb":
        raise ValueError("Неизвестный калиброванный MB")
    # Параметры экспорта обязаны соответствовать модели, которая будет сохранена.
    for calibrated, state in zip(model.calibrated_classifiers_, metadata["folds"], strict=True):
        membership = {key: state[key] for key in ("fold", "fit_indices", "calibration_indices", "fit_users",
                                                  "calibration_users", "fit_rows", "calibration_rows")}
        if _fold_state(calibrated, membership) != state:
            raise ValueError("Параметры MB изменились после fit")
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / ".incomplete"
    marker.write_text("Архив не завершён.\n", encoding="utf-8")
    files = {}

    def write(relative: str, content: bytes) -> None:
        target = output / relative
        with target.open("xb") as stream:
            stream.write(content)
        target.chmod(0o444)
        files[relative] = {"sha256": _sha(content), "bytes": len(content)}

    write("metadata.json", _json_bytes(metadata))
    for index, calibrated in enumerate(model.calibrated_classifiers_):
        state = metadata["folds"][index]
        write(f"fold_{index}_transform.json", _json_bytes(state["transform"]))
        write(f"fold_{index}_readout.json", _json_bytes(state["readout"]))
        write(f"fold_{index}_sigmoid.json", _json_bytes(state["sigmoid"]))
        name = f"fold_{index}_representation.npz"
        calibrated.estimator.representation_.save(output / name)
        content = (output / name).read_bytes()
        (output / name).chmod(0o444)
        files[name] = {"sha256": _sha(content), "bytes": len(content)}
    write("model.pickle", pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    manifest = {"format_version": FORMAT_VERSION, "complete": True, "files": files,
                "pickle_scope": "trusted-local; verify bytes and hashes before loading",
                "metadata_sha256": files["metadata.json"]["sha256"]}
    with (output / "manifest.json").open("xb") as stream:
        stream.write(_json_bytes(manifest))
    (output / "manifest.json").chmod(0o444)
    marker.unlink()
    return manifest
