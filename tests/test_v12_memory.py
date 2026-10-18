from datetime import UTC, datetime, timedelta

import numpy as np

from research_v12.models import (
    HierarchicalRidgeMemory,
    MixtureOfExpertsModel,
    SupervisedBottleneckEncoder,
    TunedScalarMemory,
)


def test_tuned_scalar_live_stream_and_memory_cap():
    """Тест непрерывного потока на 365 днях: удаление старых наблюдений и стабилизация размера состояния."""
    memory = TunedScalarMemory(
        fast_rate=0.4,
        slow_ratio=0.5,
        l2=0.005,
        gate_window_days=60.0,
    )

    start_date = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    rng = np.random.RandomState(42)

    size_180 = None
    size_365 = None

    for day in range(365):
        at = start_date + timedelta(days=day)
        event_id = f"evt-day-{day}"
        p0 = float(rng.uniform(0.1, 0.4))

        # Принятие решения
        snap = memory.predict(event_id=event_id, at=at, p0=p0)
        assert 0.0 <= snap["p_final"] <= 1.0

        # Обратная связь поступает через 2 часа (с 10% вероятностью отсутствия ответа)
        if rng.uniform() > 0.10:
            outcome = int(rng.binomial(1, snap["p_final"]))
            memory.observe(
                event_id=event_id, at=at + timedelta(hours=2), outcome=outcome
            )
        else:
            memory.discard(event_id=event_id)

        # Проверяем, что в очереди наблюдений нет записей старше gate_window_days (60 дней)
        cutoff_sec = 60.0 * 86400.0
        for obs in memory.observations:
            assert (at - obs["decision_at"]).total_seconds() <= cutoff_sec + 86400.0

        # Фиксация контрольных точек
        if day == 180:
            size_180 = memory.state_size_bytes()
            assert size_180 <= 16384, (
                f"Размер состояния на 180 дн ({size_180} байт) превышает 16 КиБ"
            )

        if day == 364:
            size_365 = memory.state_size_bytes()
            assert size_365 <= 16384, (
                f"Размер состояния на 365 дн ({size_365} байт) превышает 16 КиБ"
            )

    assert size_180 is not None and size_365 is not None
    # Количество активных наблюдений в окне не может превышать ~65
    assert len(memory.observations) <= 65
    # Размер стабилен и не растет неограниченно
    assert abs(size_365 - size_180) < 1500, (
        f"Размер состояния не стабилизировался: 180d={size_180}B, 365d={size_365}B"
    )


def test_hierarchical_ridge_live_stream_and_memory_cap():
    """Тест HierarchicalRidgeMemory на потоке 365 дней: плато памяти <= 16 КиБ."""
    d = 8
    memory = HierarchicalRidgeMemory(
        dimension=d,
        fast_rate=0.15,
        slow_ratio=0.5,
        l2=0.01,
        gate_window_days=60.0,
    )

    start_date = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    rng = np.random.RandomState(42)

    size_180 = None
    size_365 = None

    for day in range(365):
        at = start_date + timedelta(days=day)
        event_id = f"evt-ridge-{day}"
        p0 = float(rng.uniform(0.1, 0.4))
        z = rng.randn(d)

        snap = memory.predict(event_id=event_id, at=at, p0=p0, z=z)
        assert 0.0 <= snap["p_final"] <= 1.0

        if rng.uniform() > 0.15:
            outcome = int(rng.binomial(1, snap["p_final"]))
            memory.observe(
                event_id=event_id, at=at + timedelta(hours=3), outcome=outcome
            )
        else:
            memory.discard(event_id=event_id)

        if day == 180:
            size_180 = memory.state_size_bytes()
            assert size_180 <= 16384

        if day == 364:
            size_365 = memory.state_size_bytes()
            assert size_365 <= 16384

    assert len(memory.observations) <= 65
    assert abs(size_365 - size_180) < 1500


def test_supervised_bottleneck_respects_observed_mask():
    """Тест: SupervisedBottleneckEncoder обучается строго на наблюдаемых строках, не подменяя пропуски нулями."""
    rng = np.random.RandomState(123)
    n = 200
    p = 10
    d = 4

    X = rng.randn(n, p)
    p0 = rng.uniform(0.1, 0.5, size=n)
    y_true = rng.binomial(1, p0)
    # 40% пропусков
    observed_mask = rng.uniform(size=n) > 0.40

    y_with_poison = y_true.copy()
    y_with_poison[~observed_mask] = -99  # Недопустимые значения для проверки изоляции

    encoder = SupervisedBottleneckEncoder(n_components=d, alpha=0.1, seed=42)
    encoder.fit(X, y_with_poison, p0, observed_mask=observed_mask)

    assert encoder.projection_matrix is not None
    assert encoder.projection_matrix.shape == (p, d)
    assert not np.isnan(encoder.projection_matrix).any()

    control = SupervisedBottleneckEncoder(n_components=d, alpha=0.1, seed=42)
    control.fit(X[observed_mask], y_true[observed_mask], p0[observed_mask])
    np.testing.assert_allclose(
        encoder.projection_matrix, control.projection_matrix, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        encoder.transform(X), control.transform(X), rtol=0, atol=1e-12
    )

    # Проекция на новые данные
    Z = encoder.transform(X[:10])
    assert Z.shape == (10, d)
    assert not np.isnan(Z).any()


def test_moe_respects_observed_mask():
    """Тест: MixtureOfExpertsModel обучается экспертами только на наблюдаемых метках."""
    rng = np.random.RandomState(456)
    n = 300
    p = 8

    X = rng.randn(n, p)
    y_true = rng.binomial(1, 0.3, size=n)
    user_ids = [f"user_{i % 15}" for i in range(n)]
    observed_mask = rng.uniform(size=n) > 0.30

    y_with_poison = y_true.copy()
    y_with_poison[
        ~observed_mask
    ] = -1  # Значения вне [0, 1] для проверки, что не используются

    moe = MixtureOfExpertsModel(k_experts=3, seed=42)
    moe.fit(X, y_with_poison, user_ids=user_ids, observed_mask=observed_mask)

    assert len(moe.experts) == 3
    preds = moe.predict_experts(X[:15])
    assert preds.shape == (15, 3)
    assert ((preds >= 0.0) & (preds <= 1.0)).all()

    control = MixtureOfExpertsModel(k_experts=3, seed=42)
    alternative = y_true.copy()
    alternative[~observed_mask] = 99
    control.fit(X, alternative, user_ids=user_ids, observed_mask=observed_mask)
    np.testing.assert_allclose(
        moe.predict_experts(X), control.predict_experts(X), rtol=0, atol=1e-12
    )
