"""Проверки причинности и механизмов до открытия результатов V10."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from research_v7.types import MemoryConfig
from research_v8.memory import EncodedMemory
from research_v10.memory import PlasticMemory
from research_v10.replay import replay
from research_v10.scenarios import generate
from research_v10.spiking import encode, normalize, stdp_trace
from research_v10.types import LIFConfig, Variant

AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_stdp_causal_and_anticausal_signs():
    """Направление пары, а не простое совместное число импульсов."""
    pre = np.array([[1.0], [0], [0]])
    post = np.array([0.0, 0, 1])
    assert stdp_trace(pre, post, LIFConfig())[0] > 0
    assert stdp_trace(pre[::-1], post[::-1], LIFConfig())[0] < 0
    assert stdp_trace(pre, pre[:, 0], LIFConfig())[0] == 0


def test_lif_batch_order_and_repetition_are_identical():
    """Eval не обновляет глобальную нормировку и не смешивает события."""
    h = normalize(np.random.default_rng(3).uniform(size=(12, 9)))
    p = np.linspace(0.1, 0.9, 12)
    first = encode(h, p, 0.2, LIFConfig())
    reverse = encode(h[::-1], p[::-1], 0.2, LIFConfig())
    np.testing.assert_array_equal(first.rates, reverse.rates[::-1])
    np.testing.assert_array_equal(first.eligibility, reverse.eligibility[::-1])
    assert 0 < first.spike_fraction < 1
    np.testing.assert_allclose(np.linalg.norm(first.rates, axis=1), 1)


def test_lif_threshold_controls_firing():
    h = normalize(np.ones((3, 9)))
    p = np.full(3, 0.5)
    low = encode(h, p, 0.2, LIFConfig(threshold=0.1))
    high = encode(h, p, 0.2, LIFConfig(threshold=10))
    assert low.spike_fraction > high.spike_fraction


def test_direct_memory_matches_v9_exactly():
    """Новая обвязка не улучшает baseline изменением его математики."""
    cfg = MemoryConfig()
    old = EncodedMemory(3, "test", cfg)
    new = PlasticMemory(3, "test", cfg)
    h = normalize(np.array([1.0, 2, 3]))
    for i in range(30):
        at = AT + timedelta(days=i)
        a = old.predict(str(i), at, h, 0.4)
        b = new.predict_with_trace(str(i), at, h, 0.4, h)
        assert a == b
        old.observe(str(i), at + timedelta(hours=20), i % 2)
        new.observe(str(i), at + timedelta(hours=20), i % 2)
    assert old.export_state() == new.export_state()


@pytest.mark.parametrize("rule", ["direct", "stdp", "rstdp"])
def test_checkpoint_continuation_and_duplicate_feedback(rule):
    a = PlasticMemory(3, "test", MemoryConfig(), rule)
    h = normalize(np.array([1.0, 2, 3]))
    a.predict_with_trace("a", AT, h, 0.4, h)
    b = PlasticMemory.restore(a.checkpoint())
    a.observe("a", AT + timedelta(hours=20), 1)
    b.observe("a", AT + timedelta(hours=20), 1)
    assert a.checkpoint() == b.checkpoint()
    with pytest.raises(ValueError):
        b.observe("a", AT + timedelta(days=1), 1)


def test_error_modulator_changes_update_sign():
    h = normalize(np.ones(3))
    weights = []
    for outcome in (0, 1):
        memory = PlasticMemory(3, "test", MemoryConfig(), "rstdp")
        memory.predict_with_trace("a", AT, h, 0.5, h)
        memory.observe("a", AT + timedelta(hours=20), outcome)
        weights.append(memory.fast.copy())
    np.testing.assert_allclose(weights[0], -weights[1])


def test_no_feedback_leaves_population_and_user_isolation():
    records, h, p = generate(17, "abrupt", users=3)
    variant = Variant("direct", "direct")
    none = [{**r, "outcome": None, "feedback_at": None} for r in records]
    scores, _ = replay(none, h, p, variant)
    np.testing.assert_array_equal([r["probability"] for r in scores], p)
    all_scores, _ = replay(records, h, p, variant)
    only, _ = replay(records[120:240], h[120:240], p[120:240], variant)
    assert all_scores[120:240] == only


def test_future_evaluation_labels_cannot_change_predictions():
    records, h, p = generate(17, "abrupt", users=2)
    a, _ = replay(records, h, p, Variant("rstdp", "lif_rstdp", rule="rstdp"), h)
    altered = [
        {
            **r,
            "evaluation_outcome": 1 - r["evaluation_outcome"],
            "evaluation_probability": 0.99,
        }
        for r in records
    ]
    b, _ = replay(altered, h, p, Variant("rstdp", "lif_rstdp", rule="rstdp"), h)
    np.testing.assert_array_equal(
        [r["probability"] for r in a], [r["probability"] for r in b]
    )
    np.testing.assert_array_equal([r["ungated"] for r in a], [r["ungated"] for r in b])


def test_feedback_at_decision_is_not_used_early():
    records, h, p = generate(17, "stable", users=1)
    records[0]["feedback_at"] = records[1]["decision_at"]
    a, _ = replay(records[:2], h[:2], p[:2], Variant("direct", "direct"))
    assert a[1]["ungated"] == a[1]["p_population"]


def test_scenarios_preserve_inputs_when_only_feedback_changes():
    a, h, p = generate(7101, "abrupt", users=2)
    for scenario in ("noise15", "report50", "report10", "delay72"):
        b, other_h, other_p = generate(7101, scenario, users=2)
        np.testing.assert_array_equal(h, other_h)
        np.testing.assert_array_equal(p, other_p)
        assert [r["evaluation_outcome"] for r in a] == [
            r["evaluation_outcome"] for r in b
        ]


def test_gate_does_not_open_on_two_examples():
    memory = PlasticMemory(2, "test", MemoryConfig())
    h = normalize(np.ones(2))
    for i in range(2):
        at = AT + timedelta(days=i)
        memory.predict_with_trace(str(i), at, h, 0.5, h)
        memory.observe(str(i), at + timedelta(hours=20), i)
    snap = memory.predict_with_trace("next", AT + timedelta(days=2), h, 0.5, h)
    assert snap.gamma == 0 and snap.p_final == 0.5


def test_corrupted_trace_rejected_on_restore():
    memory = PlasticMemory(2, "test", MemoryConfig())
    h = normalize(np.ones(2))
    memory.predict_with_trace("a", AT, h, 0.5, h)
    state = json.loads(memory.checkpoint())
    state["traces"] = {}
    with pytest.raises(ValueError):
        PlasticMemory.restore(json.dumps(state))
