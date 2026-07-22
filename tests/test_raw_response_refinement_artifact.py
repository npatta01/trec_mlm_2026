from __future__ import annotations

from dataclasses import fields, replace
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from trec_mlm import raw_response_refinement
from trec_mlm.artifact_io import (
    manifest_aggregate_sha256,
    manifest_entries,
    write_pretty_json,
)
from trec_mlm.raw_response_refinement import (
    RawRefinementConfig,
    compare_raw_response_refinement_repeats,
    construct_raw_response_refinement,
    load_raw_response_refinement_artifact,
    write_raw_response_refinement_artifact,
)
from trec_mlm.runio import rankings_from_run, write_trec_run


QUERY_IDS = ("q1", "q2")
CANDIDATE_IDS = ("A", "B")
DENSE = {"q1": ["B", "A"], "q2": ["A", "B"]}


def _dependencies(tmp_path: Path):
    inference_root = tmp_path / "inference"
    construction_root = tmp_path / "construction"
    dense_root = tmp_path / "dense"
    for root in (inference_root, construction_root, dense_root):
        root.mkdir(parents=True)
    write_trec_run(dense_root / "run.trec", DENSE, run_id="dense")
    dense_entries = manifest_entries(dense_root)
    write_pretty_json(
        dense_root / "manifest.json",
        {
            "artifact_type": "fixture_dense_manifest",
            "version": 1,
            "files": dense_entries,
            "aggregate_sha256": manifest_aggregate_sha256(dense_entries),
        },
    )
    records = [
        {"record_id": "A-1", "candidate_id": "A"},
        {"record_id": "A-2", "candidate_id": "A"},
        {"record_id": "B-1", "candidate_id": "B"},
        {"record_id": "B-2", "candidate_id": "B"},
    ]
    query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    record_embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]],
        dtype=np.float32,
        order="C",
    )
    inference = SimpleNamespace(
        root=inference_root,
        manifest={"aggregate_sha256": "a" * 64},
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        records=records,
        query_embeddings=query_embeddings,
        record_embeddings=record_embeddings,
        metadata={"qrel_free": True, "fixture": True},
    )
    construction = SimpleNamespace(
        root=construction_root,
        manifest={"aggregate_sha256": "b" * 64},
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        profile_scores=np.asarray([[0.9, 0.2], [0.1, 0.8]], dtype=np.float32),
        source_rankings={"dense_reference": DENSE},
        metadata={"inference_aggregate_sha256": "a" * 64},
        records=[dict(record) for record in records],
        query_embeddings=query_embeddings.copy(order="C"),
        record_embeddings=record_embeddings.copy(order="C"),
    )
    fixture = SimpleNamespace(
        inference=inference,
        construction=construction,
        dense_rankings=DENSE,
        dense_aggregate_sha256=manifest_aggregate_sha256(dense_entries),
    )
    config = RawRefinementConfig(
        str(inference_root), str(construction_root), str(dense_root)
    )
    return config, fixture


def _write_fixture(tmp_path: Path, name: str = "artifact"):
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    root = tmp_path / name
    raw_response_refinement._write_raw_response_refinement_artifact_fixture(result, root)
    return root, dependencies, result


def _refresh_manifest(root: Path, relative_path: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest_entries(root)
    manifest["files"] = entries
    manifest["aggregate_sha256"] = manifest_aggregate_sha256(entries)
    write_pretty_json(manifest_path, manifest)


def test_qrel_free_config_and_construction_boundary() -> None:
    assert [field.name for field in fields(RawRefinementConfig)] == [
        "inference_path",
        "construction_path",
        "dense_artifact_path",
    ]
    for function in (
        construct_raw_response_refinement,
        write_raw_response_refinement_artifact,
        load_raw_response_refinement_artifact,
        compare_raw_response_refinement_repeats,
    ):
        assert "qrel" not in str(inspect.signature(function)).lower()
    assert "fixture" not in str(inspect.signature(construct_raw_response_refinement))
    assert "fixture" not in str(inspect.signature(write_raw_response_refinement_artifact))


def test_dense_dependency_identity_is_exactly_pinned(tmp_path: Path) -> None:
    assert raw_response_refinement.DENSE_CLOSED_TREE_SHA256 == (
        "63c2a536bf77f967bba2b5aaf25e03926a633aa04e6522441a3a7c85c95a261c"
    )
    root = tmp_path / "plausible-dense"
    (root / "embeddings").mkdir(parents=True)
    for name in (
        "config.json", "encoder_metadata.json", "inner_scores.tsv",
        "selections.tsv", "neighbors.tsv", "predictions.tsv", "run.trec",
        "tfidf_control.trec", "global_prior_control.trec", "per_query.tsv",
        "summary.json", "validation.json", "embeddings/minilm_l6.npy",
        "embeddings/e5_small_v2.npy", "embeddings/bge_small_en_v1_5.npy",
        "embeddings/mpnet_base_v2.npy",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("internally consistent\n", encoding="utf-8")

    with pytest.raises(ValueError, match="closed-tree aggregate"):
        raw_response_refinement._closed_dense_dependency(str(root))

    write_pretty_json(root / "manifest.json", {"files": {}})
    with pytest.raises(ValueError, match="manifestless frozen"):
        raw_response_refinement._closed_dense_dependency(str(root))


def test_dense_dependency_rejects_nested_manifest_even_if_digest_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "nested-manifest"
    (root / "embeddings").mkdir(parents=True)
    for name in (
        "config.json", "encoder_metadata.json", "inner_scores.tsv",
        "selections.tsv", "neighbors.tsv", "predictions.tsv", "run.trec",
        "tfidf_control.trec", "global_prior_control.trec", "per_query.tsv",
        "summary.json", "validation.json", "embeddings/minilm_l6.npy",
        "embeddings/e5_small_v2.npy", "embeddings/bge_small_en_v1_5.npy",
        "embeddings/mpnet_base_v2.npy",
        "embeddings/manifest.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload\n", encoding="utf-8")
    monkeypatch.setattr(
        raw_response_refinement,
        "manifest_aggregate_sha256",
        lambda entries: raw_response_refinement.DENSE_CLOSED_TREE_SHA256,
    )
    monkeypatch.setattr(raw_response_refinement, "rankings_from_run", lambda path: DENSE)

    with pytest.raises(ValueError, match="manifestless frozen"):
        raw_response_refinement._closed_dense_dependency(str(root))


def test_qrel_isolation_detects_scalar_metadata_paths() -> None:
    assert raw_response_refinement._contains_qrel_reference(
        {"source_path": "/sealed/dev_qrels.txt"}
    )


def test_constructs_complete_fixed_width_evidence_and_equal_dense_rrf(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )

    assert result.top_record_indices.shape == (2, 2, 10)
    assert result.top_record_similarities.shape == (2, 2, 10)
    assert result.selected_record_ids[0][0] == ("A-1", "A-2")
    assert np.all(result.top_record_indices[:, :, 2:] == -1)
    assert np.isnan(result.top_record_similarities[:, :, 2:]).all()
    assert set(result.rankings) == {
        "raw_hierarchical",
        "raw_hierarchical_dense_equal",
    }
    assert all(
        len(row) == 2 and set(row) == set(CANDIDATE_IDS)
        for ranking in result.rankings.values()
        for row in ranking.values()
    )
    assert result.metadata["dependency_aggregate_sha256"] == {
        "inference": "a" * 64,
        "construction": "b" * 64,
        "dense": dependencies.dense_aggregate_sha256,
    }
    assert result.execution_receipt["qrel_free"] is True
    assert result.execution_receipt["dependency_aggregate_sha256"] == result.metadata[
        "dependency_aggregate_sha256"
    ]


def test_equal_rrf_skips_raw_signal_for_unavailable_candidate(tmp_path: Path) -> None:
    config, dependencies = _dependencies(tmp_path)
    dependencies.inference.records = dependencies.inference.records[:2]
    dependencies.inference.record_embeddings = (
        dependencies.inference.record_embeddings[:2].copy(order="C")
    )
    dependencies.construction.record_embeddings = (
        dependencies.construction.record_embeddings[:2].copy(order="C")
    )
    dependencies.construction.records = dependencies.construction.records[:2]
    dependencies.construction.profile_scores[:, 1] = np.float32(-2.0)

    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )

    assert result.diagnostics["unavailable_candidate_ids"] == ["B"]
    np.testing.assert_allclose(
        result.fusion_scores[:, 1],
        np.asarray([1.0 / 61.0, 1.0 / 62.0], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_round_trip_has_exact_manifest_closure_and_semantic_reconstruction(
    tmp_path: Path,
) -> None:
    root, dependencies, result = _write_fixture(tmp_path)
    expected_files = {
        "broad_scores.npy",
        "config.json",
        "diagnostics.json",
        "hierarchical_scores.npy",
        "inputs.json",
        "metadata.json",
        "rankings/dense_reference.json",
        "rankings/raw_hierarchical.json",
        "rankings/raw_hierarchical_dense_equal.json",
        "raw_scores.npy",
        "runs/dense_reference.trec",
        "runs/raw_hierarchical.trec",
        "runs/raw_hierarchical_dense_equal.trec",
        "scores/raw_hierarchical_dense_equal.npy",
        "selected_record_ids.json",
        "top_record_indices.npy",
        "top_record_similarities.npy",
        "validation.json",
        "execution.json",
    }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == expected_files
    assert set(manifest_entries(root)) == expected_files

    loaded = raw_response_refinement._load_raw_response_refinement_artifact_fixture(
        root, dependencies
    )
    np.testing.assert_array_equal(loaded.raw_scores, result.raw_scores)
    np.testing.assert_array_equal(
        loaded.hierarchical_scores, result.hierarchical_scores
    )
    assert loaded.selected_record_ids == result.selected_record_ids
    for key, rankings in {
        "dense_reference": DENSE,
        **loaded.rankings,
    }.items():
        assert rankings_from_run(root / "runs" / f"{key}.trec") == rankings


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda value: value.astype(np.float64), "float32"),
        (lambda value: np.asfortranarray(value), "C-contiguous"),
        (lambda value: value * np.float32(2.0), "unit-normalized"),
    ],
)
def test_construction_rejects_noncanonical_embeddings_before_scoring(
    tmp_path: Path, transform, message: str
) -> None:
    config, dependencies = _dependencies(tmp_path)
    dependencies.inference.query_embeddings = transform(
        dependencies.inference.query_embeddings
    )
    with pytest.raises(ValueError, match=message):
        raw_response_refinement._construct_raw_response_refinement_fixture(
            config, dependencies
    )


def test_construction_requires_construction_to_bind_supplied_inference(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    dependencies.construction.metadata["inference_aggregate_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="construction inference aggregate"):
        raw_response_refinement._construct_raw_response_refinement_fixture(
            config, dependencies
    )


def test_production_dependency_branch_uses_official_loaders_and_revalidates_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dependencies = _dependencies(tmp_path)
    construction = dependencies.construction
    np.save(
        construction.root / "query_embeddings.npy",
        construction.query_embeddings,
        allow_pickle=False,
    )
    np.save(
        construction.root / "record_embeddings.npy",
        construction.record_embeddings,
        allow_pickle=False,
    )
    write_pretty_json(construction.root / "metadata.json", construction.metadata)
    write_pretty_json(construction.root / "records.json", construction.records)
    calls: list[str] = []

    def load_inference(path: str):
        calls.append(f"inference:{path}")
        return dependencies.inference

    def load_construction(path: str):
        calls.append(f"construction:{path}")
        return construction

    monkeypatch.setattr(raw_response_refinement, "load_profile_inference_artifacts", load_inference)
    monkeypatch.setattr(raw_response_refinement, "load_profile_artifacts", load_construction)
    monkeypatch.setattr(
        raw_response_refinement,
        "_closed_dense_dependency",
        lambda path: (DENSE, raw_response_refinement.DENSE_CLOSED_TREE_SHA256),
    )

    with pytest.raises(ValueError, match="complete official qrel-free"):
        construct_raw_response_refinement(config)

    assert calls == [
        f"inference:{config.inference_path}",
        f"construction:{config.construction_path}",
    ]


def test_public_writer_rejects_fixture_sized_result(tmp_path: Path) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )

    with pytest.raises(ValueError, match="complete official inventory"):
        write_raw_response_refinement_artifact(result, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_writer_rejects_same_length_forged_selected_record_ids(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    selected = [[list(candidate) for candidate in query] for query in result.selected_record_ids]
    selected[0][0][0] = "forged-record"

    with pytest.raises(ValueError, match="selected record identity arithmetic"):
        raw_response_refinement._write_raw_response_refinement_artifact_fixture(
            replace(
                result,
                selected_record_ids=tuple(
                    tuple(tuple(candidate) for candidate in query) for query in selected
                ),
            ),
            tmp_path / "forged",
        )
    assert not (tmp_path / "forged").exists()


def test_writer_rejects_forged_frozen_metadata_values(tmp_path: Path) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    metadata = {**result.metadata, "raw_top_k": 9}

    with pytest.raises(ValueError, match="semantic reconstruction"):
        raw_response_refinement._write_raw_response_refinement_artifact_fixture(
            replace(result, metadata=metadata), tmp_path / "forged-metadata"
        )
    assert not (tmp_path / "forged-metadata").exists()


def test_writer_refuses_overwrite_and_cleans_failed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    destination = tmp_path / "failed"
    cleanup_observations: list[bool] = []
    original_rmtree = shutil.rmtree

    def fail_on_validation(path: Path, payload: object) -> None:
        if path.name == "validation.json":
            raise RuntimeError("injected failure")
        write_pretty_json(path, payload)

    monkeypatch.setattr(
        "trec_mlm.raw_response_refinement._write_json", fail_on_validation
    )
    def observe_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".failed.") and "failure-log" not in Path(path).name:
            cleanup_observations.append(
                bool(list(tmp_path.glob(".failed.failure-*.json")))
            )
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(raw_response_refinement.shutil, "rmtree", observe_cleanup)
    with pytest.raises(RuntimeError, match="injected failure"):
        raw_response_refinement._write_raw_response_refinement_artifact_fixture(result, destination)
    assert not destination.exists()
    failure_logs = list(tmp_path.glob(".failed.failure-*.json"))
    assert len(failure_logs) == 1
    failure = json.loads(failure_logs[0].read_text(encoding="utf-8"))
    assert failure == {
        "artifact_type": "raw_response_refinement_failure",
        "version": 1,
        "status": "failed",
        "execution_id": result.execution_receipt["execution_id"],
        "error_type": "RuntimeError",
        "staging_cleaned_after_log": True,
    }
    assert not [path for path in tmp_path.glob(".failed.*") if path.is_dir()]
    assert cleanup_observations == [True]

    monkeypatch.undo()
    raw_response_refinement._write_raw_response_refinement_artifact_fixture(result, destination)
    with pytest.raises(FileExistsError):
        raw_response_refinement._write_raw_response_refinement_artifact_fixture(result, destination)


def test_atomic_promotion_does_not_replace_concurrently_created_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    destination = tmp_path / "raced"
    original = raw_response_refinement._write_json

    def create_destination_before_manifest(path: Path, payload: object) -> None:
        if path.name == "manifest.json":
            destination.mkdir()
        original(path, payload)

    monkeypatch.setattr(raw_response_refinement, "_write_json", create_destination_before_manifest)
    with pytest.raises(FileExistsError):
        raw_response_refinement._write_raw_response_refinement_artifact_fixture(result, destination)
    assert destination.is_dir()
    assert len(list(tmp_path.glob(".raced.failure-*.json"))) == 1
    assert not [path for path in tmp_path.glob(".raced.*") if path.is_dir() and path != destination]


def test_loader_rejects_hash_tamper_and_rehashed_semantic_tamper(
    tmp_path: Path,
) -> None:
    root, dependencies, _ = _write_fixture(tmp_path)
    ranking_path = root / "rankings" / "raw_hierarchical.json"
    ranking_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        raw_response_refinement._load_raw_response_refinement_artifact_fixture(
            root, dependencies
    )

    root2, dependencies2, _ = _write_fixture(tmp_path / "second")
    scores = np.load(root2 / "hierarchical_scores.npy", allow_pickle=False)
    scores[0, 0] += 0.01
    np.save(root2 / "hierarchical_scores.npy", scores, allow_pickle=False)
    _refresh_manifest(root2, "hierarchical_scores.npy")
    with pytest.raises(ValueError, match="hierarchical score arithmetic"):
        raw_response_refinement._load_raw_response_refinement_artifact_fixture(
            root2, dependencies2
    )


def test_loader_rejects_rehashed_metadata_schema_tamper(tmp_path: Path) -> None:
    root, dependencies, _ = _write_fixture(tmp_path)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["qrels_path"] = "forged-qrels.txt"
    metadata["source_tree_sha256"] = "not-a-digest"
    write_pretty_json(metadata_path, metadata)
    _refresh_manifest(root, "metadata.json")

    with pytest.raises(ValueError, match="dependency or frozen configuration"):
        raw_response_refinement._load_raw_response_refinement_artifact_fixture(
            root, dependencies
    )


def test_repeat_comparison_requires_identical_selections_and_rankings(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    first_result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    second_result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    raw_response_refinement._write_raw_response_refinement_artifact_fixture(first_result, first)
    raw_response_refinement._write_raw_response_refinement_artifact_fixture(second_result, second)

    receipt = raw_response_refinement._compare_raw_response_refinement_repeats_fixture(
        first, second, dependencies
    )
    assert receipt["status"] == "passed"
    assert receipt["selected_record_ids_identical"] is True
    assert receipt["rankings_identical"] is True
    assert receipt["maximum_absolute_score_difference"] <= 1e-12
    assert receipt["tolerance"] == 1e-12
    assert receipt["first_execution_id"] != receipt["second_execution_id"]


def test_repeat_rejects_one_execution_written_twice(tmp_path: Path) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )
    first = tmp_path / "same-first"
    second = tmp_path / "same-second"
    raw_response_refinement._write_raw_response_refinement_artifact_fixture(result, first)
    raw_response_refinement._write_raw_response_refinement_artifact_fixture(result, second)

    with pytest.raises(ValueError, match="distinct construction executions"):
        raw_response_refinement._compare_raw_response_refinement_repeats_fixture(
            first, second, dependencies
        )


def test_official_scale_scoring_is_candidate_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    candidate_ids = tuple(f"c{index:03d}" for index in range(150))
    base, remainder = divmod(182_764, len(candidate_ids))
    counts = [base + (index < remainder) for index in range(len(candidate_ids))]
    records = [
        {"record_id": f"{candidate_id}-{offset:04d}", "candidate_id": candidate_id}
        for candidate_id, count in zip(candidate_ids, counts, strict=True)
        for offset in range(count)
    ]
    queries = np.ones((len(query_ids), 1), dtype=np.float32)
    record_embeddings = np.ones((len(records), 1), dtype=np.float32)
    dense = {query_id: list(candidate_ids) for query_id in query_ids}
    inference = SimpleNamespace(
        root=tmp_path / "inference",
        manifest={"aggregate_sha256": "a" * 64},
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        records=records,
        query_embeddings=queries,
        record_embeddings=record_embeddings,
        metadata={"qrel_free": True, "fixture": True},
    )
    construction = SimpleNamespace(
        root=tmp_path / "construction",
        manifest={"aggregate_sha256": "b" * 64},
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        records=records,
        query_embeddings=queries,
        record_embeddings=record_embeddings,
        profile_scores=np.zeros((len(query_ids), len(candidate_ids)), dtype=np.float32),
        source_rankings={"dense_reference": dense},
        metadata={"qrel_free": True, "fixture": True, "inference_aggregate_sha256": "a" * 64},
    )
    dependencies = SimpleNamespace(
        inference=inference,
        construction=construction,
        dense_rankings=dense,
        dense_aggregate_sha256="c" * 64,
    )
    observed_blocks: list[int] = []

    def fake_score(query_matrix, record_matrix, record_ids, *, top_k):
        observed_blocks.append(record_matrix.shape[0])
        selected = np.tile(np.arange(top_k, dtype=np.int32), (len(query_matrix), 1))
        identities = tuple(tuple(record_ids[:top_k]) for _ in query_matrix)
        return raw_response_refinement.CandidateRawScores(
            np.zeros(len(query_matrix), dtype=np.float64),
            selected,
            identities,
            np.zeros((len(query_matrix), top_k), dtype=np.float64),
        )

    monkeypatch.setattr(raw_response_refinement, "score_candidate_records", fake_score)
    config = RawRefinementConfig("inference", "construction", "dense")
    result = raw_response_refinement._construct_raw_response_refinement_fixture(
        config, dependencies
    )

    assert result.raw_scores.shape == (322, 150)
    assert len(observed_blocks) == 150
    assert sum(observed_blocks) == 182_764
    assert max(observed_blocks) == max(counts)
    assert max(observed_blocks) < 182_764
