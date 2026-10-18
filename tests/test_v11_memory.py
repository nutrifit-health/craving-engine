"""Тесты точной эквивалентности scalar, инвариантов памяти и гейта для V11."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from research_v7.types import MemoryConfig as MemoryConfigV7
from research_v8.memory import EncodedMemory
from research_v11.memory import CompactMemoryV11
from research_v11.types import MemoryConfigV11


@pytest.mark.parametrize("config_type", ["default", "frozen"])
def test_compact_memory_d0_exact_scalar_equivalence(config_type: str):
    """Проверяет численное совпадение прогнозов и состояния при d=0 со scalar из V8 на 60 шагах.

    Включает пропуски ответов (30%) и нерегулярные задержки (от 2 до 48 часов),
    в том числе превышающие суточный шаг между решениями.
    """
    if config_type == "frozen":
        fast_rate, slow_rate, l2 = 0.20, 0.05, 0.01
    else:
        fast_rate, slow_rate, l2 = 0.10, 0.025, 0.01

    cfg_v8 = MemoryConfigV7(
        fast_rate=fast_rate,
        slow_rate=slow_rate,
        fast_half_life_days=14.0,
        slow_half_life_days=90.0,
        exposure_half_life_days=30.0,
        l2=l2,
        max_logit_residual=0.7,
        gate_window_days=60.0,
        gate_min_effective_samples=12.0,
        gate_min_class_weight=2.0,
        minimum_gain=0.005,
        uncertainty_penalty=1.64,
        gate_temperature=0.008,
    )
    cfg_v11 = MemoryConfigV11(
        fast_rate=fast_rate,
        slow_rate=slow_rate,
        fast_half_life_days=14.0,
        slow_half_life_days=90.0,
        exposure_half_life_days=30.0,
        l2=l2,
        max_logit_residual=0.7,
        gate_window_days=60.0,
        gate_min_effective_samples=12.0,
        gate_min_class_weight=2.0,
        minimum_gain=0.005,
        uncertainty_penalty=1.64,
        gate_temperature=0.008,
    )

    mem_v8 = EncodedMemory(dimension=1, representation_id="scalar_test", config=cfg_v8)
    mem_v11 = CompactMemoryV11(dimension=0, config=cfg_v11)

    base_time = datetime(2026, 1, 1, 16, 30, tzinfo=UTC)
    rng = np.random.default_rng(20261101)

    # Очередь задержанных ответов для проверки задержек, в том числе > 24 часов
    pending_feedbacks: list[tuple[datetime, str, int, float]] = []

    for step in range(60):
        t_decision = base_time + timedelta(days=step)

        # Обработка ответов, созревших строго до t_decision
        pending_feedbacks.sort(key=lambda x: x[0])
        while pending_feedbacks and pending_feedbacks[0][0] < t_decision:
            fb_time, fb_id, fb_outcome, fb_conf = pending_feedbacks.pop(0)
            mem_v8.observe(fb_id, fb_time, fb_outcome, fb_conf)
            mem_v11.observe(fb_id, fb_time, fb_outcome, fb_conf)

            # Проверка компонентов сразу после обновления
            assert math.isclose(
                mem_v8.fast[0], mem_v11.b_fast, rel_tol=1e-12, abs_tol=1e-12
            )
            assert math.isclose(
                mem_v8.slow[0], mem_v11.b_slow, rel_tol=1e-12, abs_tol=1e-12
            )
            assert math.isclose(
                mem_v8.exposure[0], mem_v11.exposure, rel_tol=1e-12, abs_tol=1e-12
            )

        p0 = float(rng.uniform(0.1, 0.6))
        event_id = f"ev_{step}"

        # Прогноз
        snap_v8 = mem_v8.predict(event_id, t_decision, np.array([1.0]), p0)
        snap_v11 = mem_v11.predict(event_id, t_decision, None, p0)

        # Проверка прогнозов с машинной точностью 1e-12
        assert math.isclose(
            snap_v8.p_learning, snap_v11.p_learning, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            snap_v8.p_candidate, snap_v11.p_candidate, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            snap_v8.p_final, snap_v11.p_final, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(snap_v8.gamma, snap_v11.gamma, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(
            snap_v8.familiarity, snap_v11.familiarity, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            snap_v8.residual, snap_v11.residual, rel_tol=1e-12, abs_tol=1e-12
        )

        # Проверка компонентов состояния
        assert math.isclose(
            mem_v8.fast[0], mem_v11.b_fast, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            mem_v8.slow[0], mem_v11.b_slow, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            mem_v8.exposure[0], mem_v11.exposure, rel_tol=1e-12, abs_tol=1e-12
        )

        # Задержка ответа: от 2 до 48 часов (в 70% случаев)
        if rng.random() < 0.70:
            delay_hours = float(rng.uniform(2.0, 48.0))
            t_feedback = t_decision + timedelta(hours=delay_hours)
            outcome = int(rng.random() < p0)
            confidence = float(rng.uniform(0.7, 1.0))
            pending_feedbacks.append((t_feedback, event_id, outcome, confidence))
        else:
            mem_v8.discard(event_id)
            mem_v11.discard(event_id)


def test_compact_memory_export_restore_with_active_pending():
    """Проверяет восстановление состояния с непустой очередью pending и совпадение дальнейшей траектории."""
    cfg = MemoryConfigV11(fast_rate=0.20, slow_rate=0.05, l2=0.01)
    mem_orig = CompactMemoryV11(dimension=2, config=cfg)

    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    z1 = np.array([1.2, -0.8], dtype=np.float64)
    z2 = np.array([-0.5, 1.5], dtype=np.float64)
    z3 = np.array([0.4, 0.2], dtype=np.float64)

    # Шаг 1: прогноз ev1 (12:00) и закрытие наблюдением (16:00)
    mem_orig.predict("ev1", t0, z1, 0.35)
    mem_orig.observe("ev1", t0 + timedelta(hours=4), outcome=1, confidence=1.0)

    # Шаги 2 и 3: делаем прогнозы ev2 (день 2, 12:00) и ev3 (день 2, 14:00)
    t1 = t0 + timedelta(days=1)
    mem_orig.predict("ev2", t1, z2, 0.40)

    t2 = t1 + timedelta(hours=2)
    mem_orig.predict("ev3", t2, z3, 0.25)

    assert len(mem_orig.pending) == 2, "В pending должно быть ровно 2 активных снимка"

    # Экспорт в JSON и восстановление в независимый объект
    json_state = mem_orig.export_state()
    mem_restored = CompactMemoryV11.from_state(json_state)

    # Проверка идентичности всех компонентов при восстановлении
    assert mem_restored.dimension == mem_orig.dimension
    assert math.isclose(mem_restored.b_fast, mem_orig.b_fast, abs_tol=1e-15)
    assert math.isclose(mem_restored.b_slow, mem_orig.b_slow, abs_tol=1e-15)
    assert np.allclose(mem_restored.w_fast, mem_orig.w_fast, atol=1e-15)
    assert np.allclose(mem_restored.w_slow, mem_orig.w_slow, atol=1e-15)
    assert math.isclose(mem_restored.exposure, mem_orig.exposure, abs_tol=1e-15)
    assert mem_restored.seen == mem_orig.seen
    assert mem_restored.clock == mem_orig.clock
    assert len(mem_restored.pending) == 2

    # Продолжение траектории: доставляем задержанные ответы в оба экземпляра хронологически
    t_fb2 = t1 + timedelta(hours=8)  # 20:00 дня 2
    mem_orig.observe("ev2", t_fb2, outcome=0, confidence=0.9)
    mem_restored.observe("ev2", t_fb2, outcome=0, confidence=0.9)

    t_fb3 = t1 + timedelta(hours=10)  # 22:00 дня 2
    mem_orig.observe("ev3", t_fb3, outcome=1, confidence=0.95)
    mem_restored.observe("ev3", t_fb3, outcome=1, confidence=0.95)

    # Выполняем следующие 20 шагов в обоих экземплярах и проверяем совпадение
    rng = np.random.default_rng(2026)
    for i in range(20):
        t_step = t0 + timedelta(days=i + 2)
        z_step = rng.normal(size=2)
        p0_step = float(rng.uniform(0.1, 0.6))
        ev_id = f"ev_cont_{i}"

        s_orig = mem_orig.predict(ev_id, t_step, z_step, p0_step)
        s_rest = mem_restored.predict(ev_id, t_step, z_step, p0_step)

        assert math.isclose(s_orig.p_final, s_rest.p_final, abs_tol=1e-14)
        assert math.isclose(s_orig.p_learning, s_rest.p_learning, abs_tol=1e-14)
        assert math.isclose(s_orig.residual, s_rest.residual, abs_tol=1e-14)

        if rng.random() < 0.8:
            fb_time = t_step + timedelta(hours=12)
            out = int(rng.random() < p0_step)
            mem_orig.observe(ev_id, fb_time, out, 1.0)
            mem_restored.observe(ev_id, fb_time, out, 1.0)
        else:
            mem_orig.discard(ev_id)
            mem_restored.discard(ev_id)

        assert math.isclose(mem_orig.b_fast, mem_restored.b_fast, abs_tol=1e-14)
        assert np.allclose(mem_orig.w_fast, mem_restored.w_fast, atol=1e-14)


def test_compact_memory_gating_threshold_strict():
    """Проверяет строгое закрытие гейта при N_eff < 12 и открытие при подтверждённом выигрыше."""
    cfg = MemoryConfigV11(
        fast_rate=0.20,
        slow_rate=0.05,
        gate_min_effective_samples=12.0,
        gate_min_class_weight=2.0,
        minimum_gain=0.005,
    )
    mem = CompactMemoryV11(dimension=0, config=cfg)
    base_t = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    # 11 последовательных наблюдений (N_eff < 12)
    for i in range(11):
        t_dec = base_t + timedelta(days=i)
        ev_id = f"obs_early_{i}"
        snap = mem.predict(ev_id, t_dec, None, 0.40)
        # На каждом из первых 11 шагов гейт строго закрыт!
        assert snap.gamma == 0.0
        assert snap.p_final == snap.p_population

        # Наблюдаем исход (чередуем 0 и 1 для баланса классов)
        outcome = i % 2
        mem.observe(ev_id, t_dec + timedelta(hours=4), outcome=outcome, confidence=1.0)

    # 12-й шаг: еще до добавления 12-го наблюдения N_eff < 12
    t_12 = base_t + timedelta(days=11)
    snap_12 = mem.predict("obs_12_pre", t_12, None, 0.40)
    assert snap_12.gamma == 0.0
    mem.observe("obs_12_pre", t_12 + timedelta(hours=4), outcome=1, confidence=1.0)

    # Добавляем следующие наблюдения, где кандидат явно выигрывает
    for i in range(12, 25):
        t_dec = base_t + timedelta(days=i)
        ev_id = f"obs_win_{i}"
        outcome = 1  # кандидат предсказывает высокую вероятность при outcome=1
        mem.predict(ev_id, t_dec, None, 0.20)
        mem.observe(ev_id, t_dec + timedelta(hours=4), outcome=outcome, confidence=1.0)

    # Теперь N_eff >= 12, кандидат демонстрирует выигрыш над p_pop
    snap_mature = mem.predict("obs_mature", base_t + timedelta(days=26), None, 0.20)
    assert snap_mature.gamma > 0.0, (
        "Гейт должен открыться при N_eff >= 12 и подтверждённом выигрыше"
    )
    assert snap_mature.p_final > snap_mature.p_population


def test_compact_memory_signed_coordinates_and_amplitudes():
    """Проверяет сохранение знаков и чувствительность к амплитудам без Dual-Channel и без построчной L2."""
    cfg = MemoryConfigV11(fast_rate=0.20, slow_rate=0.05, l2=0.0)
    mem_unit = CompactMemoryV11(dimension=1, config=cfg)
    mem_large = CompactMemoryV11(dimension=1, config=cfg)

    t0 = datetime(2026, 1, 1, 16, 30, tzinfo=UTC)

    # Случай unit: z = [1.0]
    mem_unit.predict("e1", t0, np.array([1.0]), 0.30)
    mem_unit.observe("e1", t0 + timedelta(hours=5), outcome=1, confidence=1.0)

    # Случай large: z = [10.0]
    mem_large.predict("e1", t0, np.array([10.0]), 0.30)
    mem_large.observe("e1", t0 + timedelta(hours=5), outcome=1, confidence=1.0)

    # При построчной нормализации z/||z|| веса обновились бы одинаково.
    # В линейной модели градиент пропорционален амплитуде z (10x больше).
    assert math.isclose(mem_large.w_fast[0], 10.0 * mem_unit.w_fast[0], rel_tol=1e-5), (
        "Амплитуда координат должна влиять на величину градиентного обновления линейно"
    )

    # Проверка знаков: отрицательная координата обновляет вес в противоположную сторону
    mem_neg = CompactMemoryV11(dimension=1, config=cfg)
    mem_neg.predict("e1", t0, np.array([-2.5]), 0.30)
    mem_neg.observe("e1", t0 + timedelta(hours=5), outcome=1, confidence=1.0)
    assert mem_neg.w_fast[0] < 0.0, (
        "Отрицательный z при положительной ошибке должен давать отрицательный w"
    )


def test_no_future_feedback_leakage():
    """Проверяет отсутствие утечки будущих ответов в более ранние решения."""
    cfg = MemoryConfigV11(fast_rate=0.20, slow_rate=0.05)
    mem = CompactMemoryV11(dimension=1, config=cfg)

    t_dec1 = datetime(2026, 1, 1, 16, 30, tzinfo=UTC)
    mem.predict("e1", t_dec1, np.array([1.0]), 0.40)

    # Ответ на e1 запланирован на +20 часов
    t_fb1 = t_dec1 + timedelta(hours=20)

    # Решение в момент t_dec1 + 10 часов (ответ еще не поступил!)
    t_dec2 = t_dec1 + timedelta(hours=10)
    snap2 = mem.predict("e2", t_dec2, np.array([1.0]), 0.40)

    # До момента t_fb1 ответ не поступил, поэтому веса должны оставаться строго нулевыми
    assert snap2.residual == 0.0
    assert snap2.p_learning == snap2.p_population

    # Доставляем ответ в момент t_fb1
    mem.observe("e1", t_fb1, outcome=1, confidence=1.0)

    # Теперь в момент t_dec1 + 24 часа ответ уже обработан
    t_dec3 = t_dec1 + timedelta(hours=24)
    snap3 = mem.predict("e3", t_dec3, np.array([1.0]), 0.40)
    assert snap3.w_residual > 0.0
    assert snap3.p_learning > snap3.p_population
