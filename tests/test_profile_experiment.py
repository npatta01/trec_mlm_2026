from __future__ import annotations

from dataclasses import asdict, fields, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from trec_mlm.profile_experiment import (
    OFFICIAL_DISCOVERY_SHA256,
    FROZEN_DENSE_RUN_SHA256,
    FROZEN_E5_VALIDATION_SHA256,
    FROZEN_RANKING_KEYS,
    ProfileExperimentConfig,
    ProfileExperimentResult,
    compare_profile_repeats,
    freeze_profile_rankings,
    load_profile_artifacts,
    run_profile_experiment,
    validate_official_inference_handoff,
    write_profile_artifacts,
    candidate_truncation_audits,
)
from trec_mlm import profile_experiment
from trec_mlm.profile_retrieval import (
    CandidateProfile,
    aggregate_cluster_labels,
    score_profiles,
)
from trec_mlm.profile_inference_artifact import (
    ProfileInferenceArtifactConfig,
    ProfileInferenceArtifactResult,
    load_profile_inference_artifacts,
    write_profile_inference_artifacts,
)


QUERY_IDS = ("q1", "q2")
CANDIDATE_IDS = ("A", "B", "C")
DENSE = {"q1": ["A", "B", "C"], "q2": ["C", "B", "A"]}
E5 = {"q1": ["B", "A", "C"], "q2": ["A", "C", "B"]}
PROFILE = {"q1": ["C", "A", "B"], "q2": ["B", "C", "A"]}


def _fixture_question(candidate_id: str, index: int) -> str:
    return f"question {candidate_id}{index}"


def _fixture_response(candidate_id: str, index: int) -> str:
    return f"answer {candidate_id}{index}"


def _fixture_record_id(candidate_id: str, index: int) -> str:
    return profile_experiment.canonical_profile_record_id(
        candidate_id,
        f"s{index}",
        _fixture_question(candidate_id, index),
        _fixture_response(candidate_id, index),
    )


def _refresh_manifest(root: Path, relative_path: str) -> None:
    path = root / relative_path
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative_path] = {
        "byte_size": path.stat().st_size,
        "sha256": profile_experiment._sha256(path),
    }
    manifest["aggregate_sha256"] = profile_experiment._aggregate_hash(manifest["files"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _result() -> ProfileExperimentResult:
    centroids = {
        "A": np.asarray([[0.7, 0.0, np.sqrt(0.51)], [0.0, 0.1, np.sqrt(0.99)]], dtype=np.float32),
        "B": np.asarray([[0.6, 0.0, 0.8], [0.0, 0.9, np.sqrt(0.19)]], dtype=np.float32),
        "C": np.asarray([[0.8, 0.0, 0.6], [0.0, 0.8, 0.6]], dtype=np.float32),
    }
    query_embeddings = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    label_question_embeddings = query_embeddings.copy()
    label_embeddings = np.eye(3, dtype=np.float32)
    question_label_scores = label_question_embeddings @ label_embeddings.T
    profile = score_profiles(
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        query_embeddings=query_embeddings,
        profiles={
            candidate_id: CandidateProfile(candidate_id, 2, values, (), True, 1)
            for candidate_id, values in centroids.items()
        },
    )
    assert profile.rankings == PROFILE
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
        ),
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        profile_scores=profile.scores,
        winning_centroid_ids=profile.winning_centroid_ids,
        rankings=frozen.rankings,
        fusion_scores=frozen.scores,
        metadata={
            "qrel_free": True,
            "fixture": True,
        },
        profile_payload={
            candidate_id: {
                "available": True,
                "candidate_id": candidate_id,
                "requested_clusters": 2,
                "actual_clusters": 2,
                "centroids": values.tolist(),
                "converged": True,
                "iteration_count": 1,
                "audit": {"usable": 2},
                "clusters": [
                    {
                        "cluster_id": index,
                        "member_record_ids": [_fixture_record_id(candidate_id, index)],
                        "support_count": 1,
                        "dispersion": 0.0,
                        "medoid_record_id": _fixture_record_id(candidate_id, index),
                        "representative_questions": [_fixture_question(candidate_id, index)],
                        "membership_sha256": profile_experiment._length_framed_strings(
                            [_fixture_record_id(candidate_id, index)]
                        ),
                        "annotation": asdict(
                            aggregate_cluster_labels(
                                member_question_indices=[index],
                                question_label_scores=question_label_scores,
                                label_ids=CANDIDATE_IDS,
                            )
                        ),
                    }
                    for index in range(2)
                ],
            }
            for candidate_id, values in centroids.items()
        },
        source_rankings={"dense_reference": DENSE, "contextual_e5_control": E5},
        record_payload=[
            {
                "record_id": _fixture_record_id(candidate_id, index),
                "candidate_id": candidate_id,
                "source_query_id": f"s{index}",
                "raw_membership_sha256": profile_experiment._length_framed_strings(
                    [_fixture_question(candidate_id, index), _fixture_response(candidate_id, index)]
                ),
                "formatted_sha256": hashlib.sha256(
                    (
                        f"Question: {_fixture_question(candidate_id, index)}\n"
                        f"Response: {_fixture_response(candidate_id, index)}"
                    ).encode()
                ).hexdigest(),
                "question": _fixture_question(candidate_id, index),
                "response": _fixture_response(candidate_id, index),
            }
            for candidate_id in CANDIDATE_IDS
            for index in range(2)
        ],
        formatted_inputs={
            "target_queries": ["query 1", "query 2"],
            "profile_documents": [
                f"Question: {_fixture_question(candidate_id, index)}\n"
                f"Response: {_fixture_response(candidate_id, index)}"
                for candidate_id in CANDIDATE_IDS
                for index in range(2)
            ],
            "label_questions": ["label question 0", "label question 1"],
        },
        label_question_ids=("s0", "s1"),
        query_embeddings=query_embeddings,
        record_embeddings=np.concatenate([centroids[item] for item in CANDIDATE_IDS]),
        label_question_embeddings=label_question_embeddings,
        label_embeddings=label_embeddings,
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
    source = inspect.getsource(run_profile_experiment)
    assert "encode_profile_inputs" not in source
    assert "_encode_contextual_e5" not in source


def test_official_discovery_digest_is_frozen() -> None:
    assert OFFICIAL_DISCOVERY_SHA256 == (
        "b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25"
    )
    assert FROZEN_DENSE_RUN_SHA256 == (
        "1ec008416befc8996be01ef26afb2701a2765983fa5b475acafc21c217ca7a92"
    )
    assert FROZEN_E5_VALIDATION_SHA256 == (
        "0049f337106aa0bafa183d43cb8bd08f6426ffc2bb3a4d9d3991f5fb46806c35"
    )


def test_candidate_truncation_audits_preserve_candidate_order() -> None:
    assert candidate_truncation_audits(
        ("A", "B"), {"A": 2, "B": 1}, ("none", "partial", "full")
    ) == {
        "A": {"not_truncated": 1, "partial_response_truncation": 1, "full_response_truncation": 0},
        "B": {"not_truncated": 0, "partial_response_truncation": 0, "full_response_truncation": 1},
    }


def test_stage2_handoff_revalidates_labels_records_hashes_and_audits() -> None:
    result = _result()
    base_audits = {item: {"usable": 2} for item in CANDIDATE_IDS}
    audits = {
        item: {
            "usable": 2,
            "not_truncated": 2,
            "partial_response_truncation": 0,
            "full_response_truncation": 0,
        }
        for item in CANDIDATE_IDS
    }
    inference = ProfileInferenceArtifactResult(
        config=ProfileInferenceArtifactConfig("q", "d", "c", "e5"),
        query_ids=result.query_ids,
        candidate_ids=result.candidate_ids,
        label_question_ids=result.label_question_ids,
        records=result.record_payload,
        audits=audits,
        formatted_inputs=result.formatted_inputs,
        query_embeddings=result.query_embeddings,
        record_embeddings=result.record_embeddings,
        label_question_embeddings=result.label_question_embeddings,
        label_embeddings=result.label_embeddings,
        e5_rankings=result.source_rankings["contextual_e5_control"],
        metadata={
            "qrel_free": True,
            "fixture": True,
            "query_source_sha256": "query-hash",
            "candidate_source_sha256": "candidate-hash",
        },
    )
    arguments = {
        "inference": inference,
        "query_ids": result.query_ids,
        "candidate_ids": result.candidate_ids,
        "expected_records": result.record_payload,
        "base_audits": base_audits,
        "expected_formatted_inputs": result.formatted_inputs,
        "expected_label_question_ids": result.label_question_ids,
        "query_source_sha256": "query-hash",
        "candidate_source_sha256": "candidate-hash",
        "enforce_official_counts": False,
    }
    validate_official_inference_handoff(**arguments)
    with pytest.raises(ValueError, match="label-question"):
        validate_official_inference_handoff(
            **{
                **arguments,
                "inference": replace(inference, label_question_ids=("changed",)),
            }
        )
    changed_records = [dict(item) for item in result.record_payload]
    changed_records[0]["response"] += " changed"
    with pytest.raises(ValueError, match="record payload"):
        validate_official_inference_handoff(
            **{
                **arguments,
                "inference": replace(inference, records=changed_records),
            }
        )


def test_frozen_ranking_matrix_has_one_primary_and_fixed_ablations() -> None:
    result = _result()
    assert tuple(result.rankings) == FROZEN_RANKING_KEYS
    assert result.rankings["profile_only"] == PROFILE
    assert result.rankings["dense_profile_equal"]["q1"] == ["A", "C", "B"]
    assert result.rankings["dense_e5_equal"]["q1"] == ["A", "B", "C"]
    assert result.fusion_scores["dense_profile_equal"].shape == (2, 3)


def test_atomic_artifact_round_trip_and_hash_validation(tmp_path: Path) -> None:
    assert profile_experiment.ARTIFACT_VERSION == 2
    destination = tmp_path / "artifact"
    paths = write_profile_artifacts(_result(), destination)
    assert set(paths) >= {
        "config.json",
        "manifest.json",
        "validation.json",
        "records.json",
        "formatted_inputs.json",
        "predictions/dense_profile_equal.tsv",
    }
    loaded = load_profile_artifacts(destination)
    assert loaded.rankings == _result().rankings
    assert loaded.source_rankings["dense_reference"] == DENSE
    np.testing.assert_array_equal(loaded.profile_scores, _result().profile_scores)
    assert loaded.manifest["qrel_free"] is True
    assert loaded.manifest["aggregate_sha256"]
    assert all(
        set(entry) == {"byte_size", "sha256"}
        for entry in loaded.manifest["files"].values()
    )
    assert all(loaded.validation["integrity_checks"].values())
    with pytest.raises(FileExistsError):
        write_profile_artifacts(_result(), destination)

    ranking_path = destination / "rankings" / "profile_only.json"
    ranking_path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_profile_artifacts(destination)


def test_loader_rejects_files_not_closed_by_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    (destination / "unlisted.txt").write_text("mutation")
    with pytest.raises(ValueError, match="file inventory"):
        load_profile_artifacts(destination)


def test_loader_rejects_nonfinite_profile_scores_even_after_rehash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    scores = np.load(destination / "profile_scores.npy", allow_pickle=False)
    scores[0, 0] = np.nan
    np.save(destination / "profile_scores.npy", scores)
    _refresh_manifest(destination, "profile_scores.npy")
    with pytest.raises(ValueError, match="profile scores must be finite"):
        load_profile_artifacts(destination)


def test_loader_recomputes_prediction_rows_even_after_rehash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    relative = "predictions/dense_profile_equal.tsv"
    path = destination / relative
    path.write_text(path.read_text().replace("\t1\t", "\t99\t", 1))
    _refresh_manifest(destination, relative)
    with pytest.raises(ValueError, match="prediction rows"):
        load_profile_artifacts(destination)


def test_loader_recomputes_rrf_arithmetic_even_after_rehash(tmp_path: Path) -> None:
    result = _result()
    destination = tmp_path / "artifact"
    write_profile_artifacts(result, destination)
    key = "dense_profile_equal"
    score_relative = f"scores/{key}.npy"
    scores = np.load(destination / score_relative, allow_pickle=False)
    scores += np.float32(1e-4)
    np.save(destination / score_relative, scores)
    prediction_relative = f"predictions/{key}.tsv"
    (destination / prediction_relative).write_text(
        profile_experiment._prediction_text(
            query_ids=result.query_ids,
            candidate_ids=result.candidate_ids,
            rankings=result.rankings[key],
            scores=scores,
            profile_scores=result.profile_scores,
            winners=result.winning_centroid_ids,
        )
    )
    _refresh_manifest(destination, score_relative)
    _refresh_manifest(destination, prediction_relative)
    with pytest.raises(ValueError, match="fusion arithmetic"):
        load_profile_artifacts(destination)


def test_loader_recomputes_centroids_and_winners_even_after_rehash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    profiles = json.loads((destination / "profiles.json").read_text())
    profiles["A"]["centroids"][0] = [0.0, 1.0, 0.0]
    (destination / "profiles.json").write_text(
        json.dumps(profiles, indent=2, sort_keys=True) + "\n"
    )
    _refresh_manifest(destination, "profiles.json")
    with pytest.raises(ValueError, match="centroid arithmetic"):
        load_profile_artifacts(destination)


def test_loader_recomputes_cluster_diagnostics_even_after_rehash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    profiles = json.loads((destination / "profiles.json").read_text())
    profiles["A"]["clusters"][0]["dispersion"] = 0.5
    (destination / "profiles.json").write_text(
        json.dumps(profiles, indent=2, sort_keys=True) + "\n"
    )
    _refresh_manifest(destination, "profiles.json")
    with pytest.raises(ValueError, match="cluster diagnostics"):
        load_profile_artifacts(destination)


def test_loader_recomputes_annotations_even_after_rehash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    profiles = json.loads((destination / "profiles.json").read_text())
    profiles["A"]["clusters"][0]["annotation"]["top_score"] = 0.25
    (destination / "profiles.json").write_text(
        json.dumps(profiles, indent=2, sort_keys=True) + "\n"
    )
    _refresh_manifest(destination, "profiles.json")
    with pytest.raises(ValueError, match="annotation arithmetic"):
        load_profile_artifacts(destination)


def test_loader_recomputes_formatted_and_raw_hashes_even_after_rehash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_profile_artifacts(_result(), destination)
    formatted = json.loads((destination / "formatted_inputs.json").read_text())
    formatted["profile_documents"][0] += " mutation"
    (destination / "formatted_inputs.json").write_text(
        json.dumps(formatted, indent=2, sort_keys=True) + "\n"
    )
    _refresh_manifest(destination, "formatted_inputs.json")
    with pytest.raises(ValueError, match="formatted record inventory"):
        load_profile_artifacts(destination)


def test_construction_embeds_byte_identical_inference_handoff(tmp_path: Path) -> None:
    result = _result()
    inference_dir = tmp_path / "inference"
    write_profile_inference_artifacts(
        ProfileInferenceArtifactResult(
            config=ProfileInferenceArtifactConfig("q", "d", "c", "e5"),
            query_ids=result.query_ids,
            candidate_ids=result.candidate_ids,
            label_question_ids=result.label_question_ids,
            records=result.record_payload,
            audits={item: result.profile_payload[item]["audit"] for item in result.candidate_ids},
            formatted_inputs=result.formatted_inputs,
            query_embeddings=result.query_embeddings,
            record_embeddings=result.record_embeddings,
            label_question_embeddings=result.label_question_embeddings,
            label_embeddings=result.label_embeddings,
            e5_rankings=result.source_rankings["contextual_e5_control"],
            metadata={"qrel_free": True, "fixture": True},
        ),
        inference_dir,
    )
    inference = load_profile_inference_artifacts(inference_dir)
    construction_result = replace(
        result,
        metadata={
            **result.metadata,
            "inference_aggregate_sha256": inference.manifest["aggregate_sha256"],
        },
        inference_artifact_root=inference.root,
        inference_aggregate_sha256=inference.manifest["aggregate_sha256"],
    )
    construction = tmp_path / "construction"
    write_profile_artifacts(construction_result, construction)
    loaded = load_profile_artifacts(construction)
    assert loaded.manifest["files"]["query_embeddings.npy"]["sha256"] == (
        loaded.manifest["files"]["inference/query_embeddings.npy"]["sha256"]
    )


def test_repeat_receipt_compares_canonical_rankings_and_score_tolerance(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _result()
    second_result = _result()

    def attach_inference(
        result: ProfileExperimentResult, name: str, seconds: float
    ) -> tuple[ProfileExperimentResult, str]:
        root = tmp_path / name
        write_profile_inference_artifacts(
            ProfileInferenceArtifactResult(
                config=ProfileInferenceArtifactConfig("q", "d", "c", "e5"),
                query_ids=result.query_ids,
                candidate_ids=result.candidate_ids,
                label_question_ids=result.label_question_ids,
                records=result.record_payload,
                audits={
                    item: result.profile_payload[item]["audit"]
                    for item in result.candidate_ids
                },
                formatted_inputs=result.formatted_inputs,
                query_embeddings=result.query_embeddings,
                record_embeddings=result.record_embeddings,
                label_question_embeddings=result.label_question_embeddings,
                label_embeddings=result.label_embeddings,
                e5_rankings=result.source_rankings["contextual_e5_control"],
                metadata={
                    "qrel_free": True,
                    "fixture": True,
                    "profile_encoder": {"encoding_seconds": seconds},
                },
            ),
            root,
        )
        loaded = load_profile_inference_artifacts(root)
        digest = loaded.manifest["aggregate_sha256"]
        return (
            replace(
                result,
                metadata={**result.metadata, "inference_aggregate_sha256": digest},
                inference_artifact_root=root,
                inference_aggregate_sha256=digest,
            ),
            digest,
        )

    first_result, infer1 = attach_inference(first_result, "infer1", 1.0)
    second_result, infer2 = attach_inference(second_result, "infer2", 2.0)
    write_profile_artifacts(
        first_result,
        first,
    )
    write_profile_artifacts(
        second_result,
        second,
    )
    inference_receipt = tmp_path / "inference_repeat.json"
    inference_receipt.write_text(
        json.dumps(
            {
                "status": "passed",
                "first_aggregate_sha256": infer1,
                "second_aggregate_sha256": infer2,
                "maximum_absolute_embedding_difference": 0.0,
                "tolerance": 1e-5,
                "integrity_checks": {
                    "canonical_inputs_and_rankings_identical": True,
                    "metadata_and_token_audits_identical": True,
                    "embedding_tolerance": True,
                },
            }
        )
    )
    receipt = compare_profile_repeats(first, second, inference_receipt)
    assert receipt["status"] == "passed"
    assert receipt["rankings_identical"] is True
    assert receipt["maximum_absolute_score_difference"] == 0.0
    assert receipt["integrity_checks"] == {
        "inventories": True,
        "formatted_inputs": True,
        "memberships_centroids_annotations_winners": True,
        "source_and_frozen_rankings": True,
        "embedding_and_score_tolerance": True,
        "inference_repeat_gate": True,
    }
    with pytest.raises(ValueError, match="distinct"):
        compare_profile_repeats(first, first, inference_receipt)

    changed = _result()
    changed.profile_scores[0, 0] += 2e-5
    third = tmp_path / "third"
    write_profile_artifacts(changed, third)
    with pytest.raises(ValueError, match="profile score arithmetic"):
        compare_profile_repeats(first, third, inference_receipt)


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


def test_mutating_qrels_cannot_change_any_construction_hash(tmp_path: Path) -> None:
    first_qrels = tmp_path / "qrels-a.txt"
    second_qrels = tmp_path / "qrels-b.txt"
    first_qrels.write_text("q1 0 A 0\n")
    second_qrels.write_text("q1 0 A 3\n")
    first = tmp_path / "construction-a"
    second = tmp_path / "construction-b"
    write_profile_artifacts(_result(), first)
    write_profile_artifacts(_result(), second)
    assert first_qrels.read_bytes() != second_qrels.read_bytes()
    assert json.loads((first / "manifest.json").read_text())["aggregate_sha256"] == json.loads(
        (second / "manifest.json").read_text()
    )["aggregate_sha256"]


def test_mutating_every_annotation_leaves_scores_centroids_and_rankings_unchanged(
    tmp_path: Path,
) -> None:
    original = _result()
    mutated_payload = json.loads(json.dumps(original.profile_payload))
    for candidate in mutated_payload.values():
        for cluster in candidate["clusters"]:
            cluster["annotation"] = {
                **cluster["annotation"],
                "primary_label": "annotation-mutated",
                "secondary_labels": [],
                "affects_ranking": False,
            }
    changed = replace(original, profile_payload=mutated_payload)
    first = tmp_path / "labels-a"
    second = tmp_path / "labels-b"
    write_profile_artifacts(original, first)
    write_profile_artifacts(changed, second)
    first_manifest = json.loads((first / "manifest.json").read_text())["files"]
    second_manifest = json.loads((second / "manifest.json").read_text())["files"]
    assert first_manifest["profiles.json"] != second_manifest["profiles.json"]
    protected = {
        "profile_scores.npy",
        "winning_centroid_ids.npy",
        *(f"scores/{key}.npy" for key in FROZEN_RANKING_KEYS),
        *(f"rankings/{key}.json" for key in FROZEN_RANKING_KEYS),
        *(f"runs/{key}.trec" for key in FROZEN_RANKING_KEYS),
    }
    assert all(first_manifest[name] == second_manifest[name] for name in protected)
