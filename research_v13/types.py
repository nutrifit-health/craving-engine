"""Контракты изолированной политики V13."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    base_threshold: float
    reserve_penalty: float = 0.0
    limit: int = 2
    window_hours: float = 168.0
