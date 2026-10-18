"""Причинное решение без доступа к исходам или будущим вероятностям."""

from collections import deque
from datetime import datetime, timedelta
from math import isfinite

from research_v13.types import PolicyConfig


class AlertPolicy:
    def __init__(self, config: PolicyConfig):
        if config.limit != 2 or config.window_hours != 168.0:
            raise ValueError("Протокол V13 фиксирует лимит 2 за 168 часов")
        if (
            not isfinite(config.base_threshold)
            or not 0 <= config.base_threshold <= 1.01
        ):
            raise ValueError("Недопустимый базовый порог")
        if (
            not isfinite(config.reserve_penalty)
            or not 0 <= config.reserve_penalty <= 0.2
        ):
            raise ValueError("Недопустимая надбавка резервирования")
        self.config = config
        self.sent: deque[datetime] = deque()
        self.last_at: datetime | None = None

    def decide(self, at: datetime, probability: float) -> dict:
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Вероятность должна принадлежать [0, 1]")
        if self.last_at is not None and at <= self.last_at:
            raise ValueError("Решения должны поступать строго по возрастанию времени")
        self.last_at = at
        window = timedelta(hours=self.config.window_hours)
        while self.sent and self.sent[0] <= at - window:
            self.sent.popleft()
        free = self.config.limit - len(self.sent)
        threshold = self.config.base_threshold
        if free == 1:
            remaining = (self.sent[0] + window - at) / window
            threshold += self.config.reserve_penalty * remaining
        alert = free > 0 and probability >= threshold
        if alert:
            self.sent.append(at)
        return {"alert": alert, "threshold": threshold, "free_slots_before": free}
