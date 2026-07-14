from __future__ import annotations

from dataclasses import fields
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from trec_mlm.profile_experiment import (
    FROZEN_RANKING_KEYS,
    ProfileExperimentConfig,
    ProfileExperimentResult,
    compare_profile_repeats,
    freeze_profile_rankings,
    load_profile_artifacts,
    run_profile_experiment,
    write_profile_artifacts,
)
from trec_mlm.profile_retrieval import ProfileScoreResult


QUERY_IDS = ("q1", "q2")
CANDIDATE_IDS = ("A", "B", "C")
DENSE = {"q1": ["A", "B", "C"], "q2": ["C", "B", "A"]}
E5 = {"q1": ["B", "A", "C"], "q2": ["A", "C", "B"]}
PROFILE = {"q1": ["C", "A", "B"], "q2": ["B", "C", "A"]}


def _result() -> ProfileExperimentResult:
    profile = ProfileScoreResult(
        scores=np.asarray([[0.7, 0.6, 0.8], [0.1, 0.9, 0.8]], dtype=np.float32),
        winning_centroid_ids=np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int32),
        rankings={key: list(value) for key, value in PROFILE.items()},
    )
    frozen = freeze_profile_rankings(
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        dense_rankings=DENSE,
        e5_rankings=E5,
        profile_scores=profile,
        unavailable_profile_ids=frozenset(),
    )
    return ProfileExperimentResult(
        config=ProfileExperimentConfig(
            query_path="queries.tsv",
            discovery_path="discovery.json",
            candidate_path="candidates.json",
            dense_artifact_path="dense",
            e5_artifact_path="e5",
        ),
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        profile_scores=profile.scores,
        winning_centroid_ids=profile.winning_centroid_ids,
        rankings=frozen.rankings,
        fusion_scores=frozen.scores,
        metadata={"qrel_free": True, "fixture": True},
        profile_payload={"A": {"coverage": 1}, "B": {"coverage": 1}, "C": {"coverage": 1}},
    )


def test_public_construction_boundary_contains_no_qrels() -> None:
    assert not any("qrel" in field.name.lower() for field in fields(ProfileExperimentConfig))
    for function in (
        run_profile_experiment,
        write_profile_artifacts,
        load_profile_artifacts,
        freeze_profile_rankings,
    ):
        assert "qrel" not in str(inspect.signature(function)).lower()


def test_frozen_ranking_matrix_has_one_primary_and_fixed_ablations() -> None:
    result = _result()
    assert tuple(result.rankings) == FROZEN_RANKING_KEYS
    assert result.rankings["profile_only"] == PROFILE
    assert result.rankings["dense_profile_equal"]["q1"] == ["A", "C", "B"]
    assert result.rankings["dense_e5_equal"]["q1"] == ["A", "B", "C"]
    assert result.fusion_scores["dense_profile_equal"].shape == (2, 3)


def test_atomic_artifact_round_trip_and_hash_validation(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    paths = write_profile_artifacts(_result(), destination)
    assert set(paths) >= {"config.json", "manifest.json", "validation.json"}
    loaded = load_profile_artifacts(destination)
    assert loaded.rankings == _result().rankings
    np.testing.assert_array_equal(loaded.profile_scores, _result().profile_scores)
    assert loaded.manifest["qrel_free"] is True
    assert loaded.manifest["aggregate_sha256"]
    with pytest.raises(FileExistsError):
        write_profile_artifacts(_result(), destination)

    ranking_path = destination / "rankings" / "profile_only.json"
    ranking_path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_profile_artifacts(destination)


def test_repeat_receipt_compares_canonical_rankings_and_score_tolerance(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_profile_artifacts(_result(), first)
    write_profile_artifacts(_result(), second)
    receipt = compare_profile_repeats(first, second)
    assert receipt["status"] == "passed"
    assert receipt["rankings_identical"] is True
    assert receipt["maximum_absolute_score_difference"] == 0.0

    changed = _result()
    changed.profile_scores[0, 0] += 2e-5
    third = tmp_path / "third"
    write_profile_artifacts(changed, third)
    with pytest.raises(ValueError, match="1e-05"):
        compare_profile_repeats(first, third)


def test_writer_failure_does_not_leave_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trec_mlm import profile_experiment

    destination = tmp_path / "broken"
    monkeypatch.setattr(
        profile_experiment, "write_trec_run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    with pytest.raises(RuntimeError, match="stop"):
        write_profile_artifacts(_result(), destination)
    assert not destination.exists()


def test_manifest_contains_no_qrel_path_or_payload(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    text = (destination / "manifest.json").read_text().lower()
    assert '"qrels"' not in text
    assert "qrel_path" not in text
    config = json.loads((destination / "config.json").read_text())
    assert config["qrel_free"] is True
