"""Неизменяемые контракты контекстного представления V8.4."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextFeature:
    """Единица и маска наличия являются частью идентичности признака."""

    name: str
    unit: str
    semantics: str
    available_if: str | None = None


@dataclass(frozen=True)
class ContextTransformState:
    """Статистики только training; отсутствующие значения не участвуют в fit."""

    schema: tuple[ContextFeature, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    observed_counts: tuple[int, ...]
    training_rows: int
    training_sha256: str
    sources: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ContextRepresentationState:
    """Один фиксированный кодировщик; supervised-параметров здесь нет."""

    mode: str
    seed: int
    input_channels: int
    raw_dimension: int
    winner_count: int
    scale: float
    training_rows: int
    training_sha256: str
    graph_sha256: str | None
    mapping_sha256: str | None
    geometry_json: str
    sources: tuple[tuple[str, str], ...]
