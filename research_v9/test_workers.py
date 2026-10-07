"""Небольшая сквозная интеграция V9 workers; запуск выполняет только Luna."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_v8.archive import digest
from research_v8.data_types import V8Record
from research_v8.parallel import execute_jobs
from research_v8.reuse import _verify_files
from research_v9.experiment import NAMES, fit_worker, replay_worker
from research_v9.experiment_types import PretrainJob, ReplayJob
from research_v9.models import calibration_plan
from research_v9.test_models import sample_data, sample_schema, small_graph


class WorkerIntegrationTests(unittest.TestCase):
    """Два малых fit разделяются всеми тестами; benchmark-корпус не используется."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.X, cls.y, cls.groups = sample_data()
        folds, _ = calibration_plan(cls.X, cls.y, cls.groups)
        event_ids = tuple(f"train:{user}:{index}" for index, user in enumerate(cls.groups))
        graph = small_graph()
        jobs = [PretrainJob(
            name=name, X=cls.X, y=cls.y, groups=cls.groups, event_ids=event_ids,
            folds=folds, schema=sample_schema(), graph=matrix,
            online_training=cls.X, output=cls.root / "fits" / name,
        ) for name, matrix in (("anatomical", graph), ("rewired", graph[:, ::-1].copy()))]
        # Именно spawn и настоящие worker-функции: локальные mock не проверяют
        # сериализуемость задания и обязательный result["key"] общего executor.
        cls.results = execute_jobs(fit_worker, jobs, workers=2)
        cls.fits = {result["family"]: result for result in cls.results}

    def make_replay_job(self, case: str, split: str = "calibration", fit_path: Path | None = None) -> ReplayJob:
        """Фиксирует небольшой вход и SHA завершённого fit до вызова worker."""
        directory = self.root / case
        directory.mkdir()
        records = []
        start = datetime(2026, 1, 1, 16, 30, tzinfo=timezone.utc)
        for index, (user, label) in enumerate(zip(self.groups, self.y, strict=True)):
            day = index % 8
            at = start + timedelta(days=day)
            observed = day != 2
            records.append(V8Record(
                split=split, user_id=f"new-{user}", event_id=f"{split}:{user}:{day}",
                target="evening_relapse", decision_at=at, available_at=at,
                # Заведомо посторонний prior в record должен быть проигнорирован MB.
                p_population=.999, pn=[.25] * 16, outcome=int(label) if observed else None,
                evaluation_outcome=int(label), evaluation_probability=.5,
                feedback_at=at + timedelta(hours=20) if observed else None,
            ))
        records_path = directory / "records.jsonl"
        records_path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")
        features_path = directory / "features.npy"
        with features_path.open("xb") as stream:
            np.save(stream, self.X, allow_pickle=False)
        fitted = self.fits["anatomical"]
        fit_path = fit_path or Path(fitted["directory"])
        hashes = {str(path): digest(path) for path in (records_path, features_path)}
        for relative, key in (
            ("artifact_manifest.json", "artifact_manifest_sha256"),
            ("model/model.pickle", "model_sha256"),
            ("online_transform.json", "online_transform_sha256"),
            ("online_representation.npz", "online_representation_sha256"),
        ):
            hashes[str(fit_path / relative)] = fitted[key]
        return ReplayJob(
            family="anatomical", split=split, records_path=records_path,
            features_path=features_path, population_path=None, fit_path=fit_path,
            input_hashes=hashes, output=directory / "replay", training_prevalence=.5,
        )

    def test_parallel_fit_workers_return_keys_and_closed_archives_with_pinned_hashes(self) -> None:
        """Два сериализуемых задания возвращают полный контракт общего executor."""
        self.assertEqual([result["key"] for result in self.results], ["anatomical", "rewired"])
        for result in self.results:
            with self.subTest(family=result["family"]):
                path = Path(result["directory"])
                self.assertEqual(result["key"], result["family"])
                self.assertEqual(result["fit_rows"], 24)
                self.assertEqual(result["fit_users"], 3)
                self.assertGreaterEqual(result["seconds"], 0)
                manifest = json.loads((path / "artifact_manifest.json").read_text())
                _verify_files(path, manifest)
                for relative, key in (
                    ("artifact_manifest.json", "artifact_manifest_sha256"),
                    ("model/model.pickle", "model_sha256"),
                    ("online_transform.json", "online_transform_sha256"),
                    ("online_representation.npz", "online_representation_sha256"),
                ):
                    self.assertEqual(digest(path / relative), result[key])
                events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["status"], "completed")
                metadata = json.loads((path / "model/metadata.json").read_text())
                self.assertFalse(metadata["gbdt_offset_used"])
                self.assertEqual(len(metadata["folds"]), 3)
                self.assertFalse((path / "model/.incomplete").exists())

    def test_mb_replay_emits_two_variants_with_own_population_and_independent_personal_state(self) -> None:
        """Персональная память стартует с MB p0; record.p_population её не подменяет."""
        job = self.make_replay_job("valid-replay")
        result = replay_worker(job)
        self.assertEqual(result["key"], "calibration:anatomical")
        names = {"anatomical_population", "anatomical_personal"}
        self.assertEqual(set(result["reports"]), names)
        self.assertEqual(set(result["thresholds"]), names)
        fit_manifest = json.loads((job.fit_path / "artifact_manifest.json").read_text())
        _verify_files(job.fit_path, fit_manifest)
        self.assertEqual(digest(job.fit_path / "model/model.pickle"), self.fits["anatomical"]["model_sha256"])
        # Pickle создан в setUpClass самим тестом, а его SHA проверен до чтения.
        with (job.fit_path / "model/model.pickle").open("rb") as stream:
            model = pickle.load(stream)
        expected = model.predict_proba(self.X)[:, 1]
        p0 = np.load(job.output / "p0.npy", allow_pickle=False)
        np.testing.assert_allclose(p0, expected, rtol=1e-12, atol=1e-12)
        self.assertFalse(np.allclose(p0, .999))
        records = [V8Record.model_validate_json(line) for line in job.records_path.read_text().splitlines()]
        by_event = {record.event_id: p0[index] for index, record in enumerate(records)}
        for name in names:
            path = job.output / "predictions/calibration" / f"{name}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["event_id"] for row in rows}, set(by_event))
            for row in rows:
                self.assertEqual(row["p_population"], by_event[row["event_id"]])
                self.assertEqual(row["global_residual"], 0)
                if name.endswith("_population") or row["event_id"].endswith(":0"):
                    self.assertEqual(row["probability"], row["p_population"])
                    self.assertEqual(row["personal_residual"], 0)
            self.assertEqual(set(rows[0]["decisions"]), set(result["thresholds"][name]))
        checkpoint = job.output / "checkpoints/calibration/anatomical_personal/new-alpha.json.gz"
        with gzip.open(checkpoint, "rt", encoding="utf-8") as stream:
            state = json.load(stream)
        self.assertTrue(any(value != 0 for value in state["fast"]))
        self.assertEqual(state["head_mode"], "off")
        _verify_files(job.output, json.loads((job.output / "artifact_manifest.json").read_text()))

    def test_changed_fit_or_threshold_is_rejected_before_unpickle_and_replay(self) -> None:
        """Подмена закреплённых байтов не превращается в новый доверенный SHA."""
        bad_fit = self.root / "altered-fit"
        shutil.copytree(Path(self.fits["anatomical"]["directory"]), bad_fit)
        model_job = self.make_replay_job("bad-model", fit_path=bad_fit)
        model_path = bad_fit / "model/model.pickle"
        model_path.chmod(0o644)
        model_path.write_bytes(b"changed after fit was frozen")
        threshold_job = self.make_replay_job("bad-threshold", split="development")
        frozen = threshold_job.output.parent / "frozen_thresholds.json"
        thresholds = {name: {"fixed_030": .3, "calibration_macro_fpr20": .5,
                             "calibration_macro_recall94": .1} for name in NAMES}
        frozen.write_text(json.dumps(thresholds), encoding="utf-8")
        threshold_job = replace(threshold_job, frozen_path=frozen, frozen_sha256=digest(frozen))
        thresholds["anatomical_personal"]["calibration_macro_fpr20"] = .7
        frozen.write_text(json.dumps(thresholds), encoding="utf-8")
        for job, message in ((model_job, "Вход replay изменился"),
                             (threshold_job, "Development требует заранее зафиксированные пороги")):
            with self.subTest(case=job.output.parent.name):
                with patch("research_v9.experiment.pickle.load") as load, patch("research_v9.experiment.score_records") as replay:
                    with self.assertRaisesRegex(ValueError, message):
                        replay_worker(job)
                    load.assert_not_called()
                    replay.assert_not_called()
                events = [json.loads(line) for line in (job.output / "events.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["status"], "failed")
                self.assertFalse((job.output / "artifact_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
