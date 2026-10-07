"""V9: два параллельных standalone MB-fit и четыре равных семейства.

Все модели и пороги фиксируются до development scoring. Повторная попытка
пишется в новый каталог; import не обучает и не запускает проверки.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits

from research_v7.connectome import rewire
from research_v7.types import MemoryConfig
from research_v8.archive import RunArchive, digest
from research_v9.bundle import SOURCE_PARENT_SHA256, _array, _copy, _records, load_bundle
from research_v8.context_representation import ContextTransform, FixedContextRepresentation
from research_v8.data_types import V8Record
from research_v8.evaluation import choose_thresholds, model_report, paired_comparison
from research_v8.parallel import execute_jobs
from research_v8.replay import ROOT, graph_matrix, save_predictions, score_records
from research_v8.reuse import _verify_files
from research_v9.experiment_types import PretrainJob, ReplayJob
from research_v9.models import fit_model, save_model_artifacts


PROTOCOL_URL = "https://github.com/nutrifit-health/craving-engine/blob/main/docs/protocol-v9.md"
CONNECTOME_SHA256 = "9af289f6854acc05d8656545b1b3bc76aa68de6af6b1cbb280f411f1372c8a9e"
POINT = "calibration_macro_fpr20"
NAMES = ("population", "scalar", "anatomical_population", "anatomical_personal",
         "rewired_population", "rewired_personal")
CANDIDATE = "anatomical_personal"
# Усреднение float не должно исключать математически точную границу допуска.
NUMERIC_TOLERANCE = 1e-12


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot_sources(archive: RunArchive) -> dict[str, str]:
    """Сохраняет автономные исходники без метаданных приватного монорепозитория."""
    return archive.snapshot_sources(ROOT)


def fit_worker(job: PretrainJob) -> dict:
    """Наблюдаемые training labels и полные параметры сохраняются до replay."""
    started = time.monotonic()
    archive = RunArchive(job.output, {"protocol": PROTOCOL_URL, "stage": "pretraining",
        "family": job.name, "threads": 1, "gbdt_offset": False})
    try:
        archive.write_json("training_contract.json", {"event_ids": job.event_ids,
            "fields": ["X", "observed_outcome", "user_id"], "evaluation_labels_received": False,
            "rows": len(job.y), "prevalence": float(job.y.mean())})
        with threadpool_limits(limits=1):
            model, metadata = fit_model("mb", job.X, job.y, job.groups, job.schema, job.graph, job.folds)
            save_model_artifacts(model, archive.path / "model")
            transform = ContextTransform.fit(job.online_training, job.schema)
            archive.write_json("online_transform.json", transform.to_dict())
            representation = FixedContextRepresentation.fit(transform.transform(job.online_training), job.graph, 11)
            representation.save(archive.path / "online_representation.npz")
        archive.write_json("online_geometry.json", representation.geometry)
        result = {"key": job.name, "family": job.name, "directory": str(archive.path),
            "model_sha256": digest(archive.path / "model/model.pickle"),
            "fit_users": metadata["training_users"], "fit_rows": metadata["training_rows"],
            "seconds": time.monotonic() - started}
        archive.write_json("result.json", result)
        archive.finish()
        return {**result, "artifact_manifest_sha256": digest(archive.path / "artifact_manifest.json"),
            "online_transform_sha256": digest(archive.path / "online_transform.json"),
            "online_representation_sha256": digest(archive.path / "online_representation.npz")}
    except Exception as error:
        archive.event("failed", type=type(error).__name__, message=str(error))
        raise


def choose_simple_baseline(reports: dict) -> str:
    """Выбор сильнейшего простого контроля разрешён только вызывающему calibration."""
    def rank(name: str) -> tuple:
        macro = reports[name]["operating_points"][POINT]["macro"]
        if macro["false_positive_rate"]["mean"] > .20 + 1e-12:
            raise ValueError("Calibration baseline нарушает FPR20")
        return (-macro["recall"]["mean"], macro["brier"]["mean"], name)
    return min(("population", "scalar"), key=rank)


def admission(reports: dict, baseline: str) -> dict:
    """Допуск только заранее выбранного anatomy; контроли не становятся победителем."""
    if baseline not in ("population", "scalar"):
        raise ValueError("Простой baseline должен быть population либо scalar")
    checks, comparisons = {}, {}
    candidate = reports[CANDIDATE]
    cp = candidate["operating_points"][POINT]
    for name in sorted({"population", baseline}):
        bp = reports[name]["operating_points"][POINT]
        delta = paired_comparison(candidate, reports[name])
        comparisons[name] = delta
        recall, j = delta["recall_delta"], delta["youden_j_delta"]
        notification, reference = cp["notifications"]["2"], bp["notifications"]["2"]
        checks[name] = {
            "recall_gain_5pp": recall["mean"] is not None and recall["mean"] >= .05 - NUMERIC_TOLERANCE,
            "recall_ci_positive": recall["ci95"] is not None and recall["ci95"][0] > 0,
            "j_gain_002": j["mean"] is not None and j["mean"] >= .02 - NUMERIC_TOLERANCE,
            "j_ci_positive": j["ci95"] is not None and j["ci95"][0] > 0,
            "fpr_at_most_020": cp["macro"]["false_positive_rate"]["mean"] <= .20 + NUMERIC_TOLERANCE,
            "fpr_increase_at_most_001": delta["false_positive_rate_delta"]["mean"] <= .01 + NUMERIC_TOLERANCE,
            "brier_not_worse": delta["brier_delta"]["mean"] <= NUMERIC_TOLERANCE,
            "positive_bss": cp["pooled"]["bss_training_prevalence"] is not None
                and cp["pooled"]["bss_training_prevalence"] > 0,
            "individual_degradation_at_most_005": delta["individual_auc_degradation"]["fraction"] is not None
                and delta["individual_auc_degradation"]["fraction"] <= .05,
            "notification_recall_improved": notification["recall"] is not None and reference["recall"] is not None
                and notification["recall"] > reference["recall"],
            "notification_burden_not_worse": notification["false_alarms_per_caught"] is not None
                and reference["false_alarms_per_caught"] is not None
                and notification["false_alarms_per_caught"] <= reference["false_alarms_per_caught"],
        }
    return {"candidate": CANDIDATE, "simple_baseline": baseline,
        "eligible": all(all(group.values()) for group in checks.values()), "checks": checks,
        "comparisons": comparisons, "blind_test_performed": False,
        "interpretation": "development screening; no clinical or population safety claim"}


def enrich_report(rows: list[dict], thresholds: dict, prevalence: float) -> dict:
    """Дополняет общие метрики явными четырьмя счётчиками и возрастом памяти."""
    result = model_report(rows, thresholds, prevalence)
    for point in result["operating_points"].values():
        m = point["pooled"]
        m["false_negatives"] = m["events"] - m["true_positives"]
        m["true_negatives"] = m["scored"] - m["events"] - m["false_positives"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["user_id"]].append(row)
    phases = {"days_001_030": [], "days_031_090": [], "days_091_180": []}
    for items in groups.values():
        ordered = sorted(items, key=lambda row: (row["decision_at"], row["event_id"]))
        first = datetime.fromisoformat(ordered[0]["decision_at"])
        for row in ordered:
            day = (datetime.fromisoformat(row["decision_at"]) - first).days + 1
            phase = "days_001_030" if day <= 30 else "days_031_090" if day <= 90 else "days_091_180"
            phases[phase].append(row)
    # Уведомления по фазам не публикуются: сброс очереди в начале фазы меняет политику.
    result["memory_age"] = {name: {"operating_points": {
        key: {field: value for field, value in data.items() if field != "notifications"}
        for key, data in model_report(items, {POINT: thresholds[POINT]}, prevalence)["operating_points"].items()}}
        for name, items in phases.items() if items}
    return result


def replay_worker(job: ReplayJob) -> dict:
    """Независимый причинный replay; у MB p0 получен только от MB-ансамбля."""
    started = time.monotonic()
    archive = RunArchive(job.output, {"protocol": PROTOCOL_URL, "stage": job.split,
        "family": job.family, "inputs": job.input_hashes, "threads": 1})
    try:
        for filename, expected in job.input_hashes.items():
            if digest(Path(filename)) != expected:
                raise ValueError("Вход replay изменился после фиксации: " + filename)
        frozen = None
        if job.split != "calibration":
            if job.frozen_path is None or digest(job.frozen_path) != job.frozen_sha256:
                raise ValueError("Development требует заранее зафиксированные пороги")
            frozen = _json(job.frozen_path)
            if set(frozen) != set(NAMES):
                raise ValueError("Не сохранены пороги всех шести вариантов")
        records = [V8Record.model_validate_json(line) for line in job.records_path.read_text().splitlines()]
        X = np.load(job.features_path, allow_pickle=False)
        if len(X) != len(records):
            raise ValueError("Признаки и события не выровнены")
        with threadpool_limits(limits=1):
            if job.family == "gbdt":
                if job.population_path is None or job.fit_path is not None:
                    raise ValueError("GBDT требует закреплённые прогнозы")
                p0 = np.load(job.population_path, allow_pickle=False)
                codes = np.ones((len(records), 1))
                variants = (("population", None), ("scalar", codes))
            else:
                if job.population_path is not None or job.fit_path is None:
                    raise ValueError("MB не должен получать GBDT offset")
                fit = job.fit_path
                _verify_files(fit, _json(fit / "artifact_manifest.json"))
                # Pickle собственного завершённого fit: bytes сверены по закреплённым SHA.
                with (fit / "model/model.pickle").open("rb") as stream:
                    model = pickle.load(stream)
                p0 = model.predict_proba(X)[:, 1]
                transform = ContextTransform.from_dict(_json(fit / "online_transform.json"))
                representation = FixedContextRepresentation.load(fit / "online_representation.npz")
                codes = representation.encode(transform.transform(X))
                variants = ((job.family + "_population", None), (job.family + "_personal", codes))
            if p0.shape != (len(records),) or not np.isfinite(p0).all() or np.any((p0 <= 0) | (p0 >= 1)):
                raise ValueError("Некорректный популяционный прогноз")
            _array(archive, "p0.npy", p0)
            reports, thresholds, geometries = {}, {}, {}
            for name, h in variants:
                rep_id = _identity({"model": name, "inputs": job.input_hashes,
                    "memory": MemoryConfig().model_dump(mode="json"), "protocol": PROTOCOL_URL})
                scores, geometry = score_records(records, h, p0, name, rep_id, archive, job.split)
                threshold = choose_thresholds(scores) if frozen is None else frozen[name]
                save_predictions(archive, name, job.split, scores, threshold)
                reports[name] = enrich_report(scores, threshold, job.training_prevalence)
                thresholds[name], geometries[name] = threshold, geometry
        result = {"key": f"{job.split}:{job.family}", "family": job.family, "split": job.split, "reports": reports,
            "thresholds": thresholds, "geometry": geometries, "seconds": time.monotonic() - started}
        archive.write_json("result.json", result)
        archive.finish()
        return result
    except Exception as error:
        archive.event("failed", type=type(error).__name__, message=str(error))
        raise


def run(bundle: Path, connectome: Path, output: Path, workers: int = 2) -> dict:
    """Одна конечная серия; запрет автоматического подбора и открытия holdout."""
    started = time.monotonic()
    if workers not in (1, 2) or isinstance(workers, bool):
        raise ValueError("V9 разрешает один или два worker")
    if digest(connectome) != CONNECTOME_SHA256:
        raise ValueError("Коннектом отличается от закреплённого источника")
    archive = RunArchive(output, {"iteration": "V9", "protocol": PROTOCOL_URL,
        "source_parent_manifest_sha256": SOURCE_PARENT_SHA256, "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "connectome_sha256": CONNECTOME_SHA256,
        "names": NAMES, "candidate": CANDIDATE, "mapping_seed": 11,
        "rewiring_seed": 20260923, "workers": workers, "threads_per_worker": 1,
        "memory": MemoryConfig().model_dump(mode="json"),
        "status": "known_synthetic_development_no_blind_test"})
    try:
        sources = snapshot_sources(archive)
        corpus, matrices, population, schema = load_bundle(bundle, archive)
        archive.write_json("schema.json", schema)
        _copy(connectome, archive.path / "connectome_source.npz")
        graph, metadata = graph_matrix(archive.path / "connectome_source.npz")
        archive.write_json("graph_source.json", metadata)
        randomized, rewiring = rewire(graph, seed=20260923)
        archive.write_json("rewiring.json", rewiring)
        _array(archive, "rewired_graph.npy", randomized)
        observed = np.asarray([row.outcome is not None for row in corpus["training"]])
        records = [row for row in corpus["training"] if row.outcome is not None]
        y = np.asarray([row.outcome for row in records], dtype=int)
        groups = np.asarray([row.user_id for row in records])
        ids = tuple(row.event_id for row in records)
        folds_source = _json(archive.path / "inputs/folds.json")
        if tuple(folds_source["training_event_ids"]) != ids:
            raise ValueError("Training event IDs отличаются от GBDT")
        folds = [(np.asarray(fold["fit"]), np.asarray(fold["sigmoid"])) for fold in folds_source["indices"]]
        archive.write_json("fit_contract.json", {"observed_rows": len(records),
            "total_rows": len(corpus["training"]), "users": len(set(groups.tolist())),
            "training_prevalence": float(y.mean()), "folds_source": "inputs/folds.json",
            "same_observed_rows_and_folds_as_gbdt": True, "gbdt_refitted": False,
            "fit_fields": ["observed94", "observed_outcome", "user_id", "event_id"],
            "calibration_labels": "full synthetic evaluator labels for operating thresholds only"})
        archive.event("input_audit_completed", split_rows={split: len(rows) for split, rows in corpus.items()})
        print("V9: входы закреплены; два независимых предобучения MB", flush=True)
        jobs = [PretrainJob(name, matrices["training"][observed], y, groups, ids, folds, schema,
            current, matrices["training"], archive.path / "fits" / name)
            for name, current in (("anatomical", graph), ("rewired", randomized))]
        fits = {result["family"]: result for result in execute_jobs(fit_worker, jobs, workers)}
        archive.write_json("frozen_models.json", fits)
        archive.event("all_fits_completed", families=list(fits))
        reports, thresholds, baseline, frozen_sha256 = {}, {}, None, None
        for split in ("calibration", "development"):
            if split == "development" and (baseline is None or not (archive.path / "frozen_thresholds.json").is_file()):
                raise ValueError("Development запрещён до calibration")
            rows_path = _records(archive, f"replay_inputs/{split}.jsonl", corpus[split])
            features_path = _array(archive, f"replay_inputs/{split}.X.npy", matrices[split])
            p_path = _array(archive, f"replay_inputs/{split}.gbdt.npy", population[split])
            jobs = []
            for family in ("gbdt", "anatomical", "rewired"):
                fit_path = None if family == "gbdt" else Path(fits[family]["directory"])
                inputs = {str(path): digest(path) for path in (rows_path, features_path)}
                if fit_path is None:
                    inputs[str(p_path)] = digest(p_path)
                else:
                    for relative, key in (("artifact_manifest.json", "artifact_manifest_sha256"),
                                          ("model/model.pickle", "model_sha256"),
                                          ("online_transform.json", "online_transform_sha256"),
                                          ("online_representation.npz", "online_representation_sha256")):
                        inputs[str(fit_path / relative)] = fits[family][key]
                frozen = archive.path / "frozen_thresholds.json" if split == "development" else None
                jobs.append(ReplayJob(family, split, rows_path, features_path,
                    p_path if family == "gbdt" else None, fit_path, inputs,
                    archive.path / "replays" / split / family, float(y.mean()),
                    frozen, frozen_sha256 if frozen is not None else None))
            results = execute_jobs(replay_worker, jobs, workers)
            reports[split] = {name: report for result in results for name, report in result["reports"].items()}
            if set(reports[split]) != set(NAMES):
                raise ValueError("Не все отчётные варианты завершились")
            archive.write_json(f"{split}_report.json", reports[split])
            if split == "calibration":
                thresholds = {name: value for result in results for name, value in result["thresholds"].items()}
                baseline = choose_simple_baseline(reports[split])
                archive.write_json("frozen_thresholds.json", thresholds)
                frozen_sha256 = digest(archive.path / "frozen_thresholds.json")
                archive.write_json("calibration_selection.json", {"candidate": CANDIDATE,
                    "simple_baseline": baseline, "candidate_is_preregistered": True,
                    "selected_before_development": True, "alternative_mb_selection_allowed": False,
                    "frozen_thresholds_sha256": frozen_sha256})
                archive.event("thresholds_and_baseline_frozen", baseline=baseline)
                print("V9: пороги зафиксированы; начинаем development", flush=True)
        dev = reports["development"]
        comparisons = {name: paired_comparison(dev[candidate], dev[reference])
            for name, candidate, reference in (
                ("anatomy_vs_rewired", "anatomical_personal", "rewired_personal"),
                ("anatomy_population_vs_gbdt", "anatomical_population", "population"),
                ("rewired_population_vs_gbdt", "rewired_population", "population"),
                ("anatomy_personalization", "anatomical_personal", "anatomical_population"),
                ("rewired_personalization", "rewired_personal", "rewired_population"),
                ("scalar_personalization", "scalar", "population"))}
        decision = admission(dev, baseline)
        summary = []
        for name in NAMES:
            point = dev[name]["operating_points"][POINT]
            macro, pooled = point["macro"], point["pooled"]
            summary.append({"model": name, "macro_recall": macro["recall"]["mean"],
                "macro_fpr": macro["false_positive_rate"]["mean"], "macro_j": macro["youden_j"]["mean"],
                "macro_auc": macro["auc"]["mean"], "macro_brier": macro["brier"]["mean"],
                "pooled_bss": pooled["bss_training_prevalence"], "mean_gamma": pooled["mean_gamma"],
                "confusion": {key: pooled[key] for key in ("true_positives", "false_positives", "true_negatives", "false_negatives")},
                "notifications2": point["notifications"]["2"]})
        archive.write_json("summary.json", summary)
        archive.write_json("comparisons.json", comparisons)
        archive.write_json("admission.json", decision)
        if any(digest(ROOT / relative) != sha for relative, sha in sources.items()):
            raise RuntimeError("Исходники изменились во время V9")
        if digest(archive.path / "frozen_thresholds.json") != frozen_sha256:
            raise RuntimeError("Пороги изменились после calibration")
        for fitted in fits.values():
            fit_path = Path(fitted["directory"])
            if digest(fit_path / "artifact_manifest.json") != fitted["artifact_manifest_sha256"]:
                raise RuntimeError("Manifest предобученной модели изменился")
            _verify_files(fit_path, _json(fit_path / "artifact_manifest.json"))
        if digest(bundle / "manifest.json") != _json(archive.path / "protocol.json")["bundle_manifest_sha256"]:
            raise RuntimeError("Manifest входного набора изменился")
        _verify_files(bundle, _json(bundle / "manifest.json")["artifacts"])
        result = {"status": "completed_known_synthetic_development", "protocol": PROTOCOL_URL,
            "splits": reports, "comparisons": comparisons, "admission": decision,
            "seconds": time.monotonic() - started, "fit_seconds_sum": sum(r["seconds"] for r in fits.values()),
            "limitations": ["Только известная синтетика; новый blind test пока не проводился",
                "Полная разметка calibration идеализирована; fit и online используют только observed outcomes",
                "Partial ALPN→KC и инженерное отображение сигналов не являются полной симуляцией мозга",
                "Обнаружение не доказывает предотвращение; 94% при малой доле ложных тревог не гарантированы",
                "GBDT не переобучается: используются сохранённые прогнозы исходной серии",
                "Автономное выделение кода требует отдельной проверки воспроизводимости"]}
        archive.write_json("report.json", result)
        archive.finish()
        print(f"V9 завершена: {archive.path / 'summary.json'}; blind eligibility={decision['eligible']}", flush=True)
        return result
    except Exception as error:
        archive.event("failed", type=type(error).__name__, message=str(error), seconds=time.monotonic() - started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/v9-frozen")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    bundle = args.data_dir.resolve()
    run(bundle, bundle / "connectome.npz", args.output_dir, args.workers)


if __name__ == "__main__":
    main()
