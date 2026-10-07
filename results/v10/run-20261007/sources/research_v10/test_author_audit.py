"""Проверяет наблюдаемые свойства прочитанного временного листинга автора."""

import copy
import os
import unittest
from pathlib import Path

import torch

from research_v10.author_audit import (
    SMALL_CONFIG,
    eval_probe,
    feedback_probe,
    load_author,
    range_probe,
    seed_all,
    stdp_probe,
    training_probe,
)


class AuthorAuditTests(unittest.TestCase):
    """Диагностирует механизм; отсутствие градиента не объявляет reservoir ошибкой."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source = Path(
            os.environ.get("V10_AUTHOR_SOURCE", "/tmp/nutrifit-v10-brain_spiking.py")
        )
        if not cls.source.exists():
            raise unittest.SkipTest(
                "Сначала сохраните и прочитайте точный листинг статьи в /tmp"
            )
        cls.author = load_author(cls.source)

    def setUp(self):
        seed_all(7)
        self.model = self.author.DynamicTopologyNetwork(**SMALL_CONFIG)
        self.x = torch.randn(4, 40) * 1.5 + 1
        self.other = torch.randn(3, 40) * 1.5 - 1
        self.y = torch.arange(4) % 3

    def test_gradient_and_optimizer_routes(self):
        """Readout и межколоночные веса обучаются; encoder и колонки без градиента."""
        result = training_probe(self.model, self.x, self.y)
        self.assertTrue(result["completed"])
        self.assertTrue(result["finite"])
        self.assertFalse(result["rate_loss_requires_grad"])
        for name, row in result["parameters"].items():
            if name == "input_scale" or name.startswith("columns."):
                self.assertIsNone(row["gradient"], name)
                self.assertEqual(row["optimizer_delta"], 0, name)
                self.assertEqual(row["forward_delta"], 0, name)
            elif name.startswith("readout.") or name == "inter_weights":
                self.assertGreater(row["gradient"]["nonzero"], 0, name)
                self.assertGreater(row["optimizer_delta"], 0, name)

    def test_hard_encoder_has_no_autograd_path(self):
        """Оба режима encoder используют дискретное сравнение без surrogate."""
        for deterministic in (False, True):
            self.assertFalse(
                self.model._poisson_encode(self.x, deterministic).requires_grad
            )

    def test_evaluation_mutates_average_and_logits(self):
        """Повтор и порядок меняют logits; восстановление среднего убирает отличие."""
        result = eval_probe(self.model, self.x, self.other)
        self.assertGreater(result["eval_avg_max_delta"], 0)
        self.assertGreater(result["repeated_logits_max_delta"], 0)
        self.assertGreater(result["order_logits_max_delta"], 0)
        self.assertGreater(result["batch_partition_logits_max_delta"], 0)
        self.assertEqual(result["repeated_raw_rate_max_delta"], 0)
        self.assertEqual(result["restored_avg_logits_max_delta"], 0)

    def test_intercolumn_weights_do_not_feed_lif_state(self):
        """Межколоночное вмешательство меняет readout, но не входы/состояния LIF."""
        result = feedback_probe(self.model, self.x)
        self.assertEqual(result["column_input_spike_state_max_delta"], 0)
        self.assertEqual(result["no_edges_column_input_spike_state_max_delta"], 0)
        self.assertGreater(result["logits_extreme_weight_max_delta"], 0)

    def test_stdp_and_trace_carry(self):
        """STDP меняет параметры; trace сохраняется между независимыми batches."""
        result = stdp_probe(self.model, self.x, self.other, 7)
        self.assertGreater(result["total_stdp_weight_delta"], 0)
        self.assertEqual(result["off_weight_delta"], 0)
        self.assertEqual(result["paired_batches"][0]["logits_max_delta"], 0)
        self.assertGreater(result["trace_before_next_batch"]["l2"], 0)
        self.assertGreater(result["carried_vs_reset_trace_delta"], 0)
        self.assertGreater(result["post_trace_before_next_batch"]["l2"], 0)
        self.assertGreater(result["carried_vs_reset_post_trace_delta"], 0)
        self.assertGreater(result["carried_vs_reset_weight_delta"], 0)

    def test_effective_weight_sign_and_formation_bound(self):
        """Sigmoid убирает знак raw веса; стандартный potential не достигает 0.8."""
        result = range_probe(self.model, self.author)
        self.assertEqual(result["initial_negative_effective_edges"], 0)
        self.assertGreater(result["initial_effective_active_edges"]["min"], 0)
        self.assertLess(
            result["default_correlation_upper_bound"], result["formation_threshold"]
        )
        self.assertEqual(result["formation_candidates_after_1000"], 0)

    def test_column_membrane_resets_on_each_forward(self):
        """При одинаковом EMA eval повторяет локальные спайки после сброса состояний."""
        self.model.eval()
        initial = copy.deepcopy(self.model.state_dict())
        with torch.no_grad():
            first, rates_first = self.model(self.x)
            self.model(self.other)
            self.model.load_state_dict(initial)
            second, rates_second = self.model(self.x)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(rates_first, rates_second))


if __name__ == "__main__":
    unittest.main()
