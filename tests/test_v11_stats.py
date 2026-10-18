"""Тесты статистических процедур V11: bootstrap, односторонний тест и поправка Холма."""

import numpy as np

from research_v11.stats import holm_bonferroni_correction, paired_bootstrap_test


def test_paired_bootstrap_significant_negative_diff():
    """Тест: при выраженном отрицательном эффекте bootstrap отвергает H0."""
    rng = np.random.default_rng(42)
    # 40 пользователей со средним выигрышем -0.005 и шумом 0.002
    diffs = rng.normal(loc=-0.005, scale=0.002, size=40)

    res = paired_bootstrap_test(diffs, n_resamples=5_000, seed=123)

    assert res["mean_diff"] < -0.003
    assert res["ci_upper"] < 0.0
    assert res["ci_below_zero"] is True
    assert res["p_value"] < 0.01


def test_paired_bootstrap_null_diff():
    """Тест: под нулевой гипотезой ДИ накрывает ноль, p-value не значим."""
    rng = np.random.default_rng(42)
    # Нулевой эффект
    diffs = rng.normal(loc=0.0, scale=0.005, size=40)

    res = paired_bootstrap_test(diffs, n_resamples=5_000, seed=123)

    assert res["ci_lower"] < 0.0 < res["ci_upper"]
    assert res["ci_below_zero"] is False
    assert res["p_value"] > 0.1


def test_holm_bonferroni_step_down():
    """Тест: поправка Холма последовательно проверяет гипотезы с ранней остановкой."""
    # 1. p=[0.015, 0.04]: обе отвергаются (p_1 <= 0.025, p_2 <= 0.050)
    p_vals_both = [("hypo_b", 0.040), ("hypo_a", 0.015)]
    res_both = holm_bonferroni_correction(p_vals_both, alpha=0.05)
    assert res_both["hypo_a"]["significant"] is True
    assert res_both["hypo_a"]["threshold"] == 0.025
    assert res_both["hypo_b"]["significant"] is True
    assert res_both["hypo_b"]["threshold"] == 0.050

    # 2. p=[0.03, 0.04]: обе НЕ отвергаются (первая p=0.03 > 0.025 -> остановка)
    p_vals_neither = [("hypo_b", 0.040), ("hypo_a", 0.030)]
    res_neither = holm_bonferroni_correction(p_vals_neither, alpha=0.05)
    assert res_neither["hypo_a"]["significant"] is False
    assert res_neither["hypo_b"]["significant"] is False

    # 3. Значима только первая: p=[0.015, 0.06] (p_1 <= 0.025, p_2 > 0.050)
    p_vals_first_only = [("hypo_a", 0.015), ("hypo_b", 0.060)]
    res_first_only = holm_bonferroni_correction(p_vals_first_only, alpha=0.05)
    assert res_first_only["hypo_a"]["significant"] is True
    assert res_first_only["hypo_b"]["significant"] is False

    # 4. Независимость сопоставления от исходного порядка в списке
    order_1 = holm_bonferroni_correction(
        [("alpha", 0.015), ("beta", 0.060)], alpha=0.05
    )
    order_2 = holm_bonferroni_correction(
        [("beta", 0.060), ("alpha", 0.015)], alpha=0.05
    )
    assert order_1["alpha"] == order_2["alpha"]
    assert order_1["beta"] == order_2["beta"]

    # 5. Проверка на 3 гипотезах: h1 отвергается, h2 нет -> h3 останавливается
    p_vals_3 = [("h3", 0.045), ("h1", 0.010), ("h2", 0.030)]
    res_3 = holm_bonferroni_correction(p_vals_3, alpha=0.05)
    assert res_3["h1"]["significant"] is True
    assert res_3["h2"]["significant"] is False
    assert res_3["h3"]["significant"] is False
