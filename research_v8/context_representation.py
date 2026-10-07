"""Фиксированный observed-контекст V8.4 без старого 32-канального адаптера.

Статистики и общий RMS-масштаб оцениваются только явно переданным training.
Смещение и величина сигнала сохраняются относительно опорной единицы:
h = L2([1, raw / training_rms]). Опорная координата позволяет учить общую
персональную поправку, но её вклад после L2 зависит от нормы контекста. Это
не обученная глобальная голова и не готовая поправка к population.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from research_v8.context_types import ContextFeature, ContextRepresentationState, ContextTransformState


FORMAT_VERSION = 1
WINNER_FRACTION = 0.05
BATCH_SIZE = 256
PROTOCOL = "https://github.com/nutrifit-health/craving-engine/blob/main/docs/protocol-v9.md#representation"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _hash_json(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _frozen(values: np.ndarray) -> np.ndarray:
    """Буфер bytes запрещает и запись, и обратное включение writeable."""
    values = np.asarray(values, dtype=np.float64)
    return np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(values.shape)


def _sources() -> tuple[tuple[str, str], ...]:
    return tuple((path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in (Path(__file__), Path(__file__).with_name("context_types.py")))


def _matrix(value: np.ndarray, columns: int | None = None, *, training: bool = False) -> np.ndarray:
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Нужна числовая матрица") from error
    if (values.ndim != 2 or values.shape[1] < 1 or (training and values.shape[0] == 0)
            or (columns is not None and values.shape[1] != columns) or not np.isfinite(values).all()):
        raise ValueError("Неверная форма или нечисловые значения матрицы")
    return values


def _schema(values: list[dict]) -> tuple[ContextFeature, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("Нужна непустая схема признаков")
    features = []
    for item in values:
        if not isinstance(item, dict) or set(item) != {"name", "unit", "semantics", "available_if"}:
            raise ValueError("Схема требует name, unit, semantics, available_if без дополнительных полей")
        if any(not isinstance(item[key], str) or not item[key] for key in ("name", "unit", "semantics")):
            raise ValueError("Имя, единица и семантика признака должны быть непустыми строками")
        if item["available_if"] is not None and not isinstance(item["available_if"], str):
            raise ValueError("available_if должен быть именем индикатора либо null")
        features.append(ContextFeature(**item))
    names = {feature.name for feature in features}
    if len(names) != len(features):
        raise ValueError("Повтор имени признака")
    by_name = {feature.name: feature for feature in features}
    for feature in features:
        indicator = feature.available_if
        if indicator is not None and (indicator not in names or indicator == feature.name
                                      or by_name[indicator].available_if is not None):
            raise ValueError("Индикатор наличия должен быть отдельным всегда доступным признаком")
    return tuple(features)


def _availability(values: np.ndarray, schema: tuple[ContextFeature, ...]) -> np.ndarray:
    columns = {feature.name: i for i, feature in enumerate(schema)}
    available = np.ones(values.shape, dtype=bool)
    for index, feature in enumerate(schema):
        if feature.available_if is not None:
            mask = values[:, columns[feature.available_if]]
            if np.any((mask != 0) & (mask != 1)):
                raise ValueError("Индикатор наличия должен быть равен 0 или 1")
            available[:, index] = mask == 1
    return available


class ContextTransform:
    """Маскированный asinh-контраст, затем все ON и все OFF каналы."""

    def __init__(self, state: ContextTransformState) -> None:
        self._state = state
        self._mean = _frozen(np.asarray(state.mean))
        self._scale = _frozen(np.asarray(state.scale))

    @classmethod
    def fit(cls, X: np.ndarray, schema: list[dict]) -> ContextTransform:
        """Принимает только training X; labels и внешние выборки не нужны."""
        features = _schema(schema)
        values = _matrix(X, len(features), training=True)
        available = _availability(values, features)
        means, scales, counts = [], [], []
        for index in range(len(features)):
            observed = values[available[:, index], index]
            count = len(observed)
            with np.errstate(over="ignore", invalid="ignore"):
                mean = float(observed.mean()) if count else 0.0
                std = float(observed.std()) if count else 0.0
            if not math.isfinite(mean) or not math.isfinite(std):
                raise ValueError("Переполнение training-статистик")
            means.append(mean)
            scales.append(std if std > 0 else 1.0)
            counts.append(count)
        return cls(ContextTransformState(features, tuple(means), tuple(scales), tuple(counts),
                                         len(values), _array_hash(values), _sources()))

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @property
    def representation_id(self) -> str:
        return _hash_json(self._payload())

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Пропуск даёт два нуля; его отдельный индикатор остаётся в коде."""
        values = _matrix(X, len(self._state.schema))
        available = _availability(values, self._state.schema)
        # Отсутствующие placeholder-значения не участвуют даже в арифметике.
        safe = np.where(available, values, self._mean)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z = np.arcsinh((safe - self._mean) / self._scale)
        if not np.isfinite(z).all():
            raise ValueError("Переполнение контрастного преобразования")
        return _frozen(np.concatenate((np.maximum(z, 0), np.maximum(-z, 0)), axis=1))

    def _payload(self) -> dict:
        return {"version": FORMAT_VERSION, "kind": "masked_asinh_on_off", "protocol": PROTOCOL,
                "state": asdict(self._state), "constant_scale": 1.0,
                "channels": "all ON columns, then all OFF columns; no channel selection"}

    def to_dict(self) -> dict:
        """Свежая JSON-совместимая копия контракта и параметров training."""
        payload = json.loads(_json(self._payload()))
        return {**payload, "representation_id": self.representation_id}

    @classmethod
    def from_dict(cls, value: dict) -> ContextTransform:
        if not isinstance(value, dict):
            raise ValueError("Неизвестный формат ContextTransform")
        payload = {key: item for key, item in value.items() if key != "representation_id"}
        if (value.get("version") != FORMAT_VERSION or value.get("kind") != "masked_asinh_on_off"
                or value.get("representation_id") != _hash_json(payload)):
            raise ValueError("Версия или хеш ContextTransform не совпали")
        try:
            state = value["state"]
            schema = _schema(state["schema"])
            means = tuple(float(item) for item in state["mean"])
            scales = tuple(float(item) for item in state["scale"])
            counts = tuple(state["observed_counts"])
            rows = state["training_rows"]
            sources = tuple(tuple(item) for item in state["sources"])
            if (len(means) != len(schema) or len(scales) != len(schema) or len(counts) != len(schema)
                    or type(rows) is not int or rows < 1 or sources != _sources()
                    or any(not math.isfinite(item) for item in (*means, *scales))
                    or any(item <= 0 for item in scales)
                    or any(type(count) is not int or not 0 <= count <= rows for count in counts)):
                raise ValueError("Неверные сохранённые training-статистики")
            result = cls(ContextTransformState(schema, means, scales, counts, rows, state["training_sha256"], sources))
        except (KeyError, TypeError) as error:
            raise ValueError("Повреждён контракт ContextTransform") from error
        if result.to_dict() != value:
            raise ValueError("Сохранённый контракт не соответствует каноническому ContextTransform")
        return result


def _mapping(pn_count: int, channels: int, seed: int) -> np.ndarray:
    """Каждый канал покрыт; большее пространство циклически покрывает меньшее."""
    rng = np.random.default_rng(seed)
    edges = max(pn_count, channels)
    pn_indices = np.resize(rng.permutation(pn_count), edges)
    input_indices = np.resize(rng.permutation(channels), edges)
    mapping = np.zeros((pn_count, channels), dtype=np.float64)
    mapping[pn_indices, input_indices] = 1.0
    mapping /= mapping.sum(axis=0, keepdims=True)
    return mapping


def _graph(value: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    graph = _matrix(value, training=True)
    if np.any(graph < 0) or np.any(np.count_nonzero(graph, axis=0) == 0):
        raise ValueError("Граф должен быть неотрицательным и иметь путь от каждого PN к KC")
    norm = np.linalg.norm(graph, axis=1)
    if not np.isfinite(norm).all():
        raise ValueError("Переполнение нормы структурных весов")
    normalized = np.divide(graph, norm[:, None], out=np.zeros_like(graph), where=norm[:, None] > 0)
    return _frozen(graph), sparse.csr_matrix(normalized)


def _raw(q: np.ndarray, graph: sparse.csr_matrix | None, mapping: sparse.csr_matrix | None, k: int) -> np.ndarray:
    if graph is None:
        return q
    if mapping is None:
        raise ValueError("Для графа требуется отображение входных каналов")
    raw = np.asarray(graph @ (mapping @ q.T)).T
    if not np.isfinite(raw).all():
        raise ValueError("Переполнение активаций KC")
    if not len(raw):
        return raw
    cutoff = np.partition(raw, raw.shape[1] - k, axis=1)[:, raw.shape[1] - k]
    above, equal = raw > cutoff[:, None], raw == cutoff[:, None]
    remaining = k - above.sum(axis=1)
    winners = (raw > 0) & (above | (equal & (np.cumsum(equal, axis=1) <= remaining[:, None])))
    return np.where(winners, raw, 0.0)


class FixedContextRepresentation:
    """Simple либо фиксированный ALPN→KC, с единым training RMS на весь код."""

    def __init__(self, state: ContextRepresentationState, graph: np.ndarray | None, mapping: np.ndarray | None) -> None:
        self._state = state
        self._graph, self._graph_csr = (None, None) if graph is None else _graph(graph)
        self._mapping = None if mapping is None else _frozen(mapping)
        self._mapping_csr = None if mapping is None else sparse.csr_matrix(self._mapping)

    @classmethod
    def fit(cls, q_training: np.ndarray, graph: np.ndarray | None, seed: int) -> FixedContextRepresentation:
        """Оценивает только RMS raw, не веса или качество представления."""
        q = _matrix(q_training, training=True)
        if np.any(q < 0) or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("ON/OFF должен быть неотрицательным; seed — целым неотрицательным")
        seed = int(seed)
        if graph is None:
            graph_values, graph_csr, mapping, mapping_csr, k = None, None, None, None, 0
            dimension = q.shape[1]
        else:
            graph_values, graph_csr = _graph(graph)
            mapping = _mapping(graph_values.shape[1], q.shape[1], seed)
            mapping_csr = sparse.csr_matrix(mapping)
            dimension = graph_values.shape[0]
            k = max(1, round(dimension * WINNER_FRACTION))
            reach = np.asarray((graph_values > 0).astype(np.int64) @ (mapping > 0).astype(np.int64))
            if np.any(np.count_nonzero(reach, axis=0) == 0) or not np.allclose(mapping.sum(axis=0), 1, rtol=0, atol=1e-14):
                raise ValueError("Не все входные каналы покрыты доступными путями к KC")
        squared_sum, active_sum, zero_rows = 0.0, 0, 0
        for start in range(0, len(q), BATCH_SIZE):
            raw = _raw(q[start:start + BATCH_SIZE], graph_csr, mapping_csr, k)
            with np.errstate(over="ignore", invalid="ignore"):
                squared_sum += float(np.sum(raw * raw))
            active = np.count_nonzero(raw, axis=1)
            active_sum += int(active.sum())
            zero_rows += int(np.sum(active == 0))
        if not math.isfinite(squared_sum):
            raise ValueError("Переполнение training RMS")
        rms = math.sqrt(squared_sum / len(q))
        scale = rms if rms > 0 else 1.0
        geometry = {
            "input_channels": q.shape[1], "raw_dimension": dimension, "encoded_dimension": dimension + 1,
            "pn_count": graph_values.shape[1] if graph_values is not None else None,
            "kc_count": dimension if graph_values is not None else None, "winner_count": k,
            "mapped_input_channels": q.shape[1] if mapping is not None else None,
            "reachable_input_channels": q.shape[1] if mapping is not None else None,
            "training_rows": len(q), "training_zero_raw_rows": zero_rows,
            "training_mean_active_raw": active_sum / len(q), "training_rms_raw_norm": rms,
            "scale_fallback_used": rms == 0, "bias_coordinate": 0,
            "normalization": "h=L2([1,raw/a]); a=sqrt(mean_training(sum(raw**2))) or1 if0",
            "topk_ties": "ascending KC index; only positive winners",
        }
        state = ContextRepresentationState(
            "simple" if graph is None else "mb", seed, q.shape[1], dimension, k, scale, len(q), _array_hash(q),
            None if graph_values is None else _array_hash(graph_values),
            None if mapping is None else _array_hash(mapping), _json(geometry), _sources(),
        )
        return cls(state, graph_values, mapping)

    @property
    def mapping(self) -> np.ndarray | None:
        return self._mapping

    @property
    def scale(self) -> float:
        return self._state.scale

    @property
    def geometry(self) -> dict:
        return json.loads(self._state.geometry_json)

    @property
    def representation_id(self) -> str:
        return _hash_json(self._payload())

    def _payload(self) -> dict:
        return {"version": FORMAT_VERSION, "kind": "fixed_context_rms_anchor", "protocol": PROTOCOL,
                "winner_fraction": WINNER_FRACTION, "state": asdict(self._state)}

    def encode(self, q: np.ndarray) -> np.ndarray:
        """Не меняет RMS при calibration/development; нулевой raw даёт [1,0,…]."""
        values = _matrix(q, self._state.input_channels)
        if np.any(values < 0):
            raise ValueError("ON/OFF каналы не могут быть отрицательными")
        output = np.empty((len(values), self._state.raw_dimension + 1), dtype=np.float64)
        for start in range(0, len(values), BATCH_SIZE):
            raw = _raw(values[start:start + BATCH_SIZE], self._graph_csr, self._mapping_csr, self._state.winner_count)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                anchored = np.column_stack((np.ones(len(raw)), raw / self.scale))
                norm = np.linalg.norm(anchored, axis=1, keepdims=True)
            if not np.isfinite(anchored).all() or not np.isfinite(norm).all():
                raise ValueError("Переполнение нормировки представления")
            output[start:start + len(raw)] = anchored / norm
        return _frozen(output)

    def save(self, path: Path) -> None:
        """Новый NPZ без pickle; существующий артефакт не перезаписывается."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self._payload(), "representation_id": self.representation_id}
        with path.open("xb") as stream:
            np.savez_compressed(stream, metadata=np.asarray(_json(metadata)),
                                graph=np.empty((0, 0)) if self._graph is None else self._graph,
                                mapping=np.empty((0, 0)) if self._mapping is None else self._mapping)

    @classmethod
    def load(cls, path: Path) -> FixedContextRepresentation:
        """Проверяет формат, идентичность параметров и неизменность реализации."""
        with np.load(Path(path), allow_pickle=False) as data:
            if set(data.files) != {"metadata", "graph", "mapping"} or data["metadata"].ndim != 0:
                raise ValueError("Неизвестный формат фиксированного контекстного представления")
            metadata = json.loads(str(data["metadata"]))
            graph, mapping = np.asarray(data["graph"], dtype=float), np.asarray(data["mapping"], dtype=float)
        if not isinstance(metadata, dict):
            raise ValueError("Некорректный manifest представления")
        payload = {key: value for key, value in metadata.items() if key != "representation_id"}
        if (metadata.get("version") != FORMAT_VERSION or metadata.get("kind") != "fixed_context_rms_anchor"
                or metadata.get("representation_id") != _hash_json(payload)):
            raise ValueError("Версия или идентичность представления не совпали")
        try:
            raw_state = dict(metadata["state"])
            raw_state["sources"] = tuple(tuple(item) for item in raw_state["sources"])
            state = ContextRepresentationState(**raw_state)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Повреждены параметры представления") from error
        if (state.mode not in ("simple", "mb") or state.sources != _sources()
                or type(state.seed) is not int or state.seed < 0
                or any(type(value) is not int or value < 1 for value in (state.input_channels, state.raw_dimension, state.training_rows))
                or type(state.scale) not in (int, float) or not math.isfinite(state.scale) or state.scale <= 0):
            raise ValueError("Некорректные параметры или другие исходники представления")
        if state.mode == "simple":
            if (graph.shape != (0, 0) or mapping.shape != (0, 0) or state.graph_sha256 is not None
                    or state.mapping_sha256 is not None or state.winner_count != 0 or state.raw_dimension != state.input_channels):
                raise ValueError("Simple-представление не должно содержать граф")
            result = cls(state, None, None)
        else:
            graph, _ = _graph(graph)
            mapping = _matrix(mapping, state.input_channels, training=True)
            if (graph.shape != (state.raw_dimension, len(mapping)) or np.any(mapping < 0)
                    or not np.allclose(mapping.sum(axis=0), 1, rtol=0, atol=1e-14)
                    or not np.array_equal(mapping, _mapping(len(mapping), state.input_channels, state.seed))
                    or _array_hash(graph) != state.graph_sha256 or _array_hash(mapping) != state.mapping_sha256
                    or state.winner_count != max(1, round(state.raw_dimension * WINNER_FRACTION))):
                raise ValueError("Граф, mapping, покрытие или top-K не совпали с manifest")
            result = cls(state, graph, mapping)
        if result.representation_id != metadata["representation_id"]:
            raise ValueError("Сохранённый контракт отличается от канонического представления")
        return result
