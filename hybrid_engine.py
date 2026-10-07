"""Числовые функции общей исследовательской памяти."""

import math
import numpy as np


def safe_logit(p: float, eps: float = 1e-4) -> float:
    """Безопасный расчет логита log(p / (1 - p))."""
    p_clipped = float(np.clip(p, eps, 1.0 - eps))
    return float(math.log(p_clipped / (1.0 - p_clipped)))


def safe_sigmoid(z: float) -> float:
    """Безопасная логистическая сигмоида 1 / (1 + exp(-z))."""
    z_clipped = float(np.clip(z, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-z_clipped)))


