"""Общий bootstrap по пользователям, выделенный из исходной V6."""

import numpy as np


def bootstrap_interval(values: list[float], seed: int, draws: int) -> list[float] | None:
    """Парный bootstrap по людям; один человек с несколькими целями не дублируется."""
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = np.asarray(values)
    means = [float(np.mean(rng.choice(samples, size=len(samples), replace=True))) for _ in range(draws)]
    return np.quantile(means, [0.025, 0.975]).tolist()


