"""Регрессии ошибок аудита: неоднозначный JSON, утечки, ложное сравнение и кэш."""

import numpy as np
import pytest

from research_harness.scientific_evidence import (
    observed_features,
    paired_metrics,
    stable_seed,
    validate_stage_evidence,
)
from research_harness.stage_checkpoint import StageCheckpoint
from research_harness.subscription import normalize_role_answer, parse_role_response


@pytest.mark.parametrize(
    "text",
    [
        r'{"answer":"\q\q\q\q\q\q\q\q","files":{}} {}',
        '{"answer":1,"answer":2}',
        '{"answer":NaN}',
        '{"answer":"a\nb"}',
        '{"answer":"KEEP_BASE"} []',
    ],
)
def test_rejects_invalid_or_ambiguous_json(text):
    with pytest.raises(ValueError):
        parse_role_response(text)


def test_seed_is_stable_and_distinguishes_worlds():
    assert stable_seed(1, "person", "additive") == stable_seed(1, "person", "additive")
    assert stable_seed(1, "person", "additive") != stable_seed(
        1, "person", "null_world"
    )


@pytest.mark.parametrize(
    "change", [{"available_at": 11}, {"source": "latent"}, {"received_at": 0}]
)
def test_rejects_unavailable_or_latent_features(change):
    feature = {
        "value": 3,
        "source": "diary",
        "event_at": 1,
        "received_at": 2,
        "available_at": 3,
    }
    feature.update(change)
    with pytest.raises(ValueError):
        observed_features({"sleep": feature}, 10)


def rows():
    return [
        {
            "world": "world",
            "user": u,
            "event": d,
            "model": m,
            "feature_set": "base",
            "y": d,
            "p": 0.2 + 0.6 * d,
            "p0": 0.5,
            "feedback_id": d,
        }
        for u in ("a", "b")
        for d in (0, 1)
        for m in ("scalar", "candidate")
    ]


def test_paired_metrics_recompute_and_cluster_by_user():
    metrics = paired_metrics(rows(), ["training"], bootstrap=100)
    assert len(metrics) == 2
    assert metrics[0]["delta_macro_brier"] == pytest.approx(-0.21)
    assert metrics[0]["macro_auc"] == 1
    assert metrics[0]["n_users"] == 2


@pytest.mark.parametrize("mode", ["leak", "label", "missing", "feedback", "p0"])
def test_invalid_comparisons_rejected(mode):
    data = rows()
    train = ["a"] if mode == "leak" else []
    if mode == "label":
        data[0]["y"] = 1
    if mode == "missing":
        data.pop(0)
    if mode == "feedback":
        data[0]["feedback_id"] = "different"
    if mode == "p0":
        data[0]["p0"] = 0.7
    with pytest.raises(ValueError):
        paired_metrics(data, train)


def test_checkpoint_roundtrip_array_and_context(tmp_path):
    path = tmp_path / "checkpoint.json"
    cache = StageCheckpoint("a" * 64, path)
    array = np.arange(3000, dtype=np.float32)
    cache.save("dataset", array)
    restored = StageCheckpoint("a" * 64, tmp_path / "copy.json", resume=path)
    np.testing.assert_array_equal(restored.load("dataset"), array)
    assert restored.load("dataset").dtype == array.dtype
    with pytest.raises(ValueError):
        StageCheckpoint("b" * 64, tmp_path / "bad.json", resume=path)
    before = path.read_bytes()
    with pytest.raises(TypeError):
        cache.save("unsupported", object())
    assert path.read_bytes() == before
    assert "unsupported" not in cache.values


def test_h2_incomplete_evidence_not_admitted():
    with pytest.raises(ValueError):
        validate_stage_evidence(
            {"configuration_ids": ["fake"], "status": "completed"}, "H2"
        )


def h3_packet():
    from research_harness.scientific_evidence import WORLD_FAMILIES

    worlds = sorted(WORLD_FAMILIES)
    keys = (
        "temporal_availability",
        "latent_isolation",
        "missing_is_unknown",
        "matched_world_targets",
        "stable_rng",
        "same_features",
        "same_feedback",
        "frozen_population",
        "cold_start_reported",
        "rare_feedback_reported",
    )
    return {
        "configuration_ids": ["scalar", "candidate"],
        "stage_evidence": {
            "schema": "craving-stage-evidence/v1",
            "stage": "H3",
            "scope": "synthetic_development",
            "seed": 11,
            "splits": {"train_users": ["train"], "evaluation_users": ["a", "b"]},
            "checks": {k: {"tested": 1, "violations": 0} for k in keys},
            "worlds": [
                {
                    "family": w,
                    "n_users": 2,
                    "n_decisions": 4,
                    "observed_feedback_fraction": 0.5,
                }
                for w in worlds
            ],
            "feature_set_ids": ["base", "sleep"],
            "prediction_rows": [
                {**row, "world": w, "feature_set": f}
                for w in worlds
                for f in ("base", "sleep")
                for row in rows()
            ],
        },
    }


def test_h3_full_factorial_measured_comparisons():
    metrics = validate_stage_evidence(h3_packet(), "H3")["recomputed_metrics"]
    assert len(metrics) == 28
    assert all(
        r["n_users"] == 2 and r["p_holm_within_packet"] >= r["p_user_signflip"]
        for r in metrics
    )


@pytest.mark.parametrize("fault", ["model", "feature", "user", "event", "outcome"])
def test_h3_global_coverage_cannot_hide_missing_pairs(fault):
    packet = h3_packet()
    evidence = packet["stage_evidence"]
    data = evidence["prediction_rows"]
    first_world = data[0]["world"]
    if fault in {"model", "feature", "user"}:
        field, value = {
            "model": ("model", "candidate"),
            "feature": ("feature_set", "sleep"),
            "user": ("user", "a"),
        }[fault]
        evidence["prediction_rows"] = [
            r for r in data if not (r["world"] == first_world and r[field] == value)
        ]
    else:
        for row in data:
            if row["world"] == first_world and row["feature_set"] == "sleep":
                row["event" if fault == "event" else "y"] += (
                    10 if fault == "event" else 1
                )
    with pytest.raises(ValueError):
        validate_stage_evidence(packet, "H3")


def test_text_role_accepts_json_object_without_changing_values():
    import json

    original = {"answer": {"report": "Строка\n\\Delta", "claims": []}, "files": {}}
    normalized = normalize_role_answer(original, role="proposer")
    assert json.loads(normalized["answer"]) == original["answer"]
    assert isinstance(original["answer"], dict)
    assert normalize_role_answer(original, role="selector") == normalized
    assert normalize_role_answer(original, role="solver") == original
    assert (
        normalize_role_answer(original, role="proposer", harvest="context/harness")
        == original
    )
    invalid = {"answer": [], "files": {}}
    assert normalize_role_answer(invalid, role="proposer") == invalid
