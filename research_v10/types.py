"""Явные параметры конечной исследовательской матрицы V10."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LIFConfig:
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    threshold: float = 2.5
    gain: float = 0.1
    steps: int = 32
    tau_trace: float = 20.0
    a_plus: float = 1.0
    a_minus: float = 1.2


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    rate: float = 0.2
    lif: LIFConfig | None = None
    rule: str = "direct"


@dataclass(frozen=True)
class EncodedBatch:
    rates: np.ndarray
    eligibility: np.ndarray
    spike_fraction: float
    silent_fraction: float


PROFILES = ((5.0, 0.1), (20.0, 0.03), (20.0, 0.1), (20.0, 0.3), (80.0, 0.1))
SCENARIOS = (
    "stable",
    "abrupt",
    "gradual",
    "transient",
    "noise15",
    "delay72",
    "report50",
    "report10",
)
EVALUATION_SEEDS = (7101, 7102, 7103)


def variants() -> list[Variant]:
    """Список фиксируется до вычисления качества на development."""
    result = [Variant("population", "population")]
    for rate in (0.2, 0.8):
        for family in ("scalar", "direct"):
            result.append(Variant(f"{family}_r{rate}", family, rate))
        for tau, gain in PROFILES:
            for rule in ("direct", "stdp", "rstdp"):
                family = f"lif_{rule}"
                name = f"{family}_t{tau}_g{gain}_r{rate}"
                result.append(
                    Variant(name, family, rate, LIFConfig(tau_mem=tau, gain=gain), rule)
                )
    return result
