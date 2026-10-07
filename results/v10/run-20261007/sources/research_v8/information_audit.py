"""Присоединение оценочных меток после прогноза, без генерации данных."""
import numpy as np
from research_v8.data_types import V8Record

def score_rows(records: list[V8Record], predictions: np.ndarray, name: str = "diagnostic") -> list[dict]:
    """Оценочные labels присоединяются только после сохранённых прогнозов."""
    p = np.asarray(predictions, dtype=float)
    if p.shape != (len(records),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Размер или диапазон прогнозов не соответствует решениям")
    return [{
        "model": name, "split": row.split, "user_id": row.user_id,
        "event_id": row.event_id, "decision_at": row.decision_at.isoformat(),
        "available_at": row.available_at.isoformat(),
        "outcome": row.evaluation_outcome, "observed_outcome": row.outcome,
        "observed_feedback": row.outcome is not None,
        "feedback_at": row.feedback_at.isoformat() if row.feedback_at else None,
        "probability": float(probability), "candidate": float(probability),
        "gamma": 0.0,
    } for row, probability in zip(records, p, strict=True)]


