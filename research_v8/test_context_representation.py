"""Регрессии V8.4: амплитуда, доступность и полное покрытие входных каналов.

Запуск этого файла поручен Luna; импорт не готовит корпус и не запускает replay.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_v8.context_representation import ContextTransform, FixedContextRepresentation


def feature(name: str, unit: str = "g", available_if: str | None = None) -> dict:
    """Минимальный явный контракт признака для математических тестов."""
    return {"name": name, "unit": unit, "semantics": "Наблюдаемый тестовый признак", "available_if": available_if}


class ContextTransformTests(unittest.TestCase):
    """Нормировка использует только доступные training-значения."""

    def test_varying_features_are_invariant_to_positive_unit_conversion(self) -> None:
        """Граммы и миллиграммы дают одинаковый контраст после своего training-fit."""
        training = np.asarray([[10.0, 1.0], [20.0, 1.0], [900.0, 0.0], [40.0, 1.0]])
        evaluation = np.asarray([[15.0, 1.0], [777.0, 0.0], [100.0, 1.0]])
        schema = [feature("amount", "g", "present"), feature("present", "0/1")]
        first = ContextTransform.fit(training, schema)
        converted_training, converted_evaluation = training.copy(), evaluation.copy()
        converted_training[:, 0] *= 1000
        converted_evaluation[:, 0] *= 1000
        converted_schema = [feature("amount", "mg", "present"), feature("present", "0/1")]
        second = ContextTransform.fit(converted_training, converted_schema)

        np.testing.assert_allclose(first.transform(evaluation), second.transform(converted_evaluation), rtol=1e-13, atol=1e-13)
        self.assertNotEqual(first.representation_id, second.representation_id)

    def test_missing_placeholder_is_not_a_measurement_or_a_true_zero(self) -> None:
        """Пропущенный placeholder не влияет на mean/scale и обнуляет оба канала."""
        schema = [feature("amount", available_if="present"), feature("present", "0/1")]
        training = np.asarray([[10.0, 1.0], [999.0, 0.0], [30.0, 1.0]])
        model = ContextTransform.fit(training, schema)
        first = model.transform(np.asarray([[123.0, 0.0], [0.0, 1.0]]))
        second = model.transform(np.asarray([[-9999.0, 0.0], [0.0, 1.0]]))

        self.assertEqual(model.mean[0], 20.0)
        self.assertEqual(model.scale[0], 10.0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0, [0, 2]], [0.0, 0.0])
        self.assertGreater(first[1, 2], 0)
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_constant_and_never_observed_channels_remain_in_the_contract(self) -> None:
        """Scale=1 не удаляет постоянные столбцы и сохраняет будущие отклонения."""
        schema = [feature("constant"), feature("unknown", available_if="present"), feature("present", "0/1")]
        model = ContextTransform.fit(np.asarray([[3.0, 100.0, 0.0], [3.0, -20.0, 0.0]]), schema)
        encoded = model.transform(np.asarray([[5.0, 7.0, 1.0]]))

        np.testing.assert_array_equal(model.scale, [1.0, 1.0, 1.0])
        self.assertEqual(encoded.shape, (1, 6))
        self.assertGreater(encoded[0, 0], 0)
        self.assertGreater(encoded[0, 1], 0)
        self.assertEqual(model.to_dict()["state"]["observed_counts"], [2, 0, 2])

    def test_transform_is_frozen_and_roundtrips_with_feature_semantics(self) -> None:
        """Внешние массивы, schema и вызовы transform не меняют сохранённый fit."""
        training = np.asarray([[1.0], [2.0], [4.0]])
        schema = [feature("amount")]
        model = ContextTransform.fit(training, schema)
        frozen = model.to_dict()
        expected = model.transform(np.asarray([[3.0]]))
        training[:] = 999
        schema[0]["unit"] = "changed"
        model.transform(np.asarray([[10000.0]]))
        clone = ContextTransform.from_dict(json.loads(json.dumps(frozen)))

        self.assertEqual(model.to_dict(), frozen)
        self.assertEqual(clone.representation_id, model.representation_id)
        np.testing.assert_array_equal(clone.transform(np.asarray([[3.0]])), expected)
        with self.assertRaises(ValueError):
            model.mean.setflags(write=True)
        corrupted = copy.deepcopy(frozen)
        corrupted["state"]["schema"][0]["unit"] = "kg"
        with self.assertRaises(ValueError):
            ContextTransform.from_dict(corrupted)

    def test_transform_rejects_invalid_matrix_schema_and_mask(self) -> None:
        """Неверные размерности, NaN и дробный индикатор не превращаются в данные."""
        schema = [feature("amount", available_if="present"), feature("present", "0/1")]
        for values in (np.empty((0, 2)), np.ones(2), np.ones((2, 1)), np.asarray([[np.nan, 1.0]]), np.asarray([[1.0, 0.5]])):
            with self.subTest(values=values.shape), self.assertRaises(ValueError):
                ContextTransform.fit(values, schema)
        with self.assertRaises(ValueError):
            ContextTransform.fit(np.ones((2, 2)), [feature("duplicate"), feature("duplicate")])
        with self.assertRaises(ValueError):
            ContextTransform.fit(np.ones((2, 1)), [feature("self", available_if="self")])
        model = ContextTransform.fit(np.asarray([[1.0, 1.0], [2.0, 1.0]]), schema)
        with self.assertRaises(ValueError):
            model.transform(np.asarray([[np.inf, 1.0]]))


class FixedContextRepresentationTests(unittest.TestCase):
    """Все входы достигают графа; RMS и опорная единица сохраняют амплитуду."""

    def test_all_188_channels_including_last_reach_164_pn(self) -> None:
        """Переполнение числа ALPN смешивает каналы, но не обрезает последние."""
        training = np.ones((2, 188))
        graph = np.eye(164)
        first = FixedContextRepresentation.fit(training, graph, 11)
        second = FixedContextRepresentation.fit(training, graph[::-1], 11)
        mapping = first.mapping

        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.shape, (164, 188))
        np.testing.assert_allclose(mapping.sum(axis=0), np.ones(188), rtol=0, atol=1e-14)
        self.assertTrue(np.all(np.count_nonzero(mapping, axis=0) >= 1))
        self.assertLessEqual(np.ptp(np.count_nonzero(mapping, axis=1)), 1)
        np.testing.assert_array_equal(mapping, second.mapping)
        last = np.zeros((1, 188))
        last[0, -1] = 1.0
        self.assertGreater(np.count_nonzero(first.encode(last)[0, 1:]), 0)
        self.assertEqual(first.geometry["reachable_input_channels"], 188)
        with self.assertRaises(ValueError):
            mapping.setflags(write=True)

    def test_mapping_also_covers_all_pn_when_inputs_are_fewer(self) -> None:
        """Повтор входа распределяется по ALPN с сохранением суммы каждого столбца."""
        model = FixedContextRepresentation.fit(np.ones((2, 4)), np.eye(11), 23)
        np.testing.assert_allclose(model.mapping.sum(axis=0), np.ones(4), rtol=0, atol=1e-14)
        self.assertTrue(np.all(model.mapping.sum(axis=1) > 0))

    def test_anchor_preserves_signal_amplitude_for_simple_and_mb(self) -> None:
        """Два пропорциональных контекста различимы, а отношение к bias восстанавливает raw/a."""
        training = np.asarray([[1.0, 2.0], [2.0, 1.0]])
        examples = np.asarray([[1.0, 2.0], [3.0, 6.0]])
        for graph in (None, np.eye(2)):
            with self.subTest(graph=graph is not None):
                model = FixedContextRepresentation.fit(training, graph, 11)
                encoded = model.encode(examples)
                np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), 1.0, rtol=1e-14)
                self.assertFalse(np.allclose(encoded[0], encoded[1]))
                np.testing.assert_allclose(encoded[1, 1:] / encoded[1, 0], 3 * encoded[0, 1:] / encoded[0, 0], rtol=1e-13)
                if graph is None:
                    np.testing.assert_allclose(encoded[:, 1:] / encoded[:, :1], examples / model.scale, rtol=1e-13)

    def test_rms_uses_training_energy_not_the_raw_dimension(self) -> None:
        """Добавление нулевых координат не ослабляет одномерный контрольный сигнал."""
        narrow = np.asarray([[2.0], [4.0]])
        wide = np.zeros((2, 188))
        wide[:, 0] = narrow[:, 0]
        simple = FixedContextRepresentation.fit(narrow, None, 11)
        padded = FixedContextRepresentation.fit(wide, None, 11)

        self.assertAlmostEqual(simple.scale, np.sqrt(10))
        self.assertEqual(simple.scale, padded.scale)
        np.testing.assert_allclose(simple.encode(narrow), padded.encode(wide)[:, :2], rtol=1e-14, atol=1e-14)

    def test_zero_signal_has_only_bias_and_deterministic_positive_topk(self) -> None:
        """Нулевой вход не создаёт KC; равные положительные KC выбираются по индексу."""
        zero = np.zeros((2, 4))
        for graph in (None, np.ones((40, 4))):
            model = FixedContextRepresentation.fit(zero, graph, 47)
            self.assertEqual(model.scale, 1.0)
            encoded = model.encode(zero)
            np.testing.assert_array_equal(encoded[:, 0], [1.0, 1.0])
            self.assertEqual(np.count_nonzero(encoded[:, 1:]), 0)
        mb = FixedContextRepresentation.fit(np.ones((2, 4)), np.ones((40, 4)), 47)
        self.assertEqual(mb.geometry["winner_count"], 2)
        np.testing.assert_array_equal(np.flatnonzero(mb.encode(np.ones((1, 4)))[0, 1:]), [0, 1])

    def test_checkpoint_roundtrip_freezes_training_scale_graph_and_mapping(self) -> None:
        """NPZ восстанавливает тот же код; последующий encode не переоценивает RMS."""
        training = np.asarray([[1.0, 0.0], [0.0, 3.0]])
        for graph in (None, np.eye(2)):
            with self.subTest(graph=graph is not None), tempfile.TemporaryDirectory() as temporary:
                model = FixedContextRepresentation.fit(training, graph, 23)
                original_id, scale, geometry = model.representation_id, model.scale, model.geometry
                path = Path(temporary) / "representation.npz"
                model.save(path)
                restored = FixedContextRepresentation.load(path)
                model.encode(training * 100)
                self.assertEqual(restored.representation_id, original_id)
                self.assertEqual(model.scale, scale)
                self.assertEqual(model.geometry, geometry)
                np.testing.assert_array_equal(restored.encode(training), model.encode(training))
                with self.assertRaises(FileExistsError):
                    model.save(path)
                if graph is not None:
                    with np.load(path, allow_pickle=False) as saved:
                        content = {key: saved[key].copy() for key in saved.files}
                    content["graph"][0, 0] += 1
                    broken = Path(temporary) / "broken.npz"
                    np.savez_compressed(broken, **content)
                    with self.assertRaises(ValueError):
                        FixedContextRepresentation.load(broken)

    def test_representation_rejects_invalid_shapes_nan_negative_and_unreachable_inputs(self) -> None:
        """Ошибочные значения и ALPN без пути к KC останавливают подготовку."""
        for q in (np.empty((0, 2)), np.ones(2), np.asarray([[np.nan, 0.0]]), np.asarray([[-1.0, 0.0]])):
            with self.subTest(shape=q.shape), self.assertRaises(ValueError):
                FixedContextRepresentation.fit(q, None, 11)
        for graph in (np.zeros((3, 2)), np.asarray([[1.0, np.inf]]), np.asarray([[1.0, -1.0]])):
            with self.subTest(shape=graph.shape), self.assertRaises(ValueError):
                FixedContextRepresentation.fit(np.ones((2, 2)), graph, 11)
        with self.assertRaises(ValueError):
            FixedContextRepresentation.fit(np.ones((2, 2)), None, True)
        model = FixedContextRepresentation.fit(np.ones((2, 2)), None, 11)
        for q in (np.ones((1, 3)), np.asarray([[np.inf, 1.0]]), np.asarray([[-1.0, 1.0]])):
            with self.subTest(shape=q.shape), self.assertRaises(ValueError):
                model.encode(q)


if __name__ == "__main__":
    unittest.main()
