"""LIF-кодировка и локальный след readout; исходная реализация для V10.

Каждое событие разворачивается в отдельные такты симуляции. Состояние между
событиями не переносится. Постсинаптический train задаётся общим p0; исход
события недоступен до observe. Это абляция механизмов, не копия коры автора.
"""

import math

import numpy as np

from research_v10.types import EncodedBatch, LIFConfig


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def training_scale(codes: np.ndarray) -> float:
    """Масштаб только из положительных training-координат, без anchor."""
    values = codes[:, 1:]
    positive = values[values > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def stdp_trace(pre: np.ndarray, post: np.ndarray, config: LIFConfig) -> np.ndarray:
    """Положительная причинная и отрицательная антипричинная пара."""
    if pre.ndim != 2 or post.shape != (len(pre),):
        raise ValueError("Нужны [такты, синапсы] и [такты]")
    a = np.zeros(pre.shape[1])
    b = 0.0
    e = np.zeros_like(a)
    decay = math.exp(-1 / config.tau_trace)
    for x, y in zip(pre, post, strict=True):
        a *= decay
        b *= decay
        e += config.a_plus * y * a - config.a_minus * x * b
        a += x
        b += y
    return e


def encode(
    codes: np.ndarray, p0: np.ndarray, scale: float, config: LIFConfig
) -> EncodedBatch:
    """Векторизация по событиям не смешивает нейронные состояния пользователей."""
    if (
        codes.ndim != 2
        or codes.shape[1] < 2
        or len(codes) != len(p0)
        or not np.isfinite(codes).all()
        or np.any(codes < 0)
        or not np.isfinite(p0).all()
        or np.any((p0 <= 0) | (p0 >= 1))
        or not math.isfinite(scale)
        or scale <= 0
        or min(
            config.tau_mem,
            config.tau_syn,
            config.threshold,
            config.tau_trace,
            config.steps,
        )
        <= 0
    ):
        raise ValueError("Некорректные входы LIF")
    all_rates, all_eligibility = [], []
    total_spikes, silent = 0, 0
    for start in range(0, len(codes), 128):
        current = codes[start : start + 128, 1:] / scale * config.gain
        probability = p0[start : start + 128]
        syn, mem, count, pre_trace, eligibility = (
            np.zeros_like(current) for _ in range(5)
        )
        post_mem = np.zeros(len(current))
        post_trace = np.zeros(len(current))
        for _ in range(config.steps):
            syn = math.exp(-1 / config.tau_syn) * syn + current
            mem = math.exp(-1 / config.tau_mem) * mem + syn
            pre = mem >= config.threshold
            mem[pre] = 0.0
            count += pre
            # Частота пост-импульсов связана с p0, а не с будущей меткой.
            post_mem += probability * 0.5
            post = post_mem >= 1
            post_mem[post] -= 1.0
            pre_trace *= math.exp(-1 / config.tau_trace)
            post_trace *= math.exp(-1 / config.tau_trace)
            eligibility += (
                config.a_plus * post[:, None] * pre_trace
                - config.a_minus * pre * post_trace[:, None]
            )
            pre_trace += pre
            post_trace += post
        rate = count / config.steps
        # Anchor и относительный масштаб сохраняют сопоставимую L2-норму.
        rates = normalize(np.column_stack((np.ones(len(rate)), rate / 0.1)))
        # Anchor учится обычным сигналом; временная часть может быть знаковой.
        eligibility = normalize(
            np.column_stack((np.ones(len(rate)), eligibility / config.steps))
        )
        all_rates.append(rates)
        all_eligibility.append(eligibility)
        total_spikes += int(count.sum())
        silent += int(np.sum(count.sum(axis=1) == 0))
    n = len(codes)
    return EncodedBatch(
        np.concatenate(all_rates),
        np.concatenate(all_eligibility),
        total_spikes / (n * (codes.shape[1] - 1) * config.steps),
        silent / n,
    )
