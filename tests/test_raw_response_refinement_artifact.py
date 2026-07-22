from __future__ import annotations

from dataclasses import fields, replace
import inspect
import json
from pathlib import Path
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
    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
    )
    root = tmp_path / name
    write_raw_response_refinement_artifact(result, root)
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


def test_constructs_complete_fixed_width_evidence_and_equal_dense_rrf(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
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

    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
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
    }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == expected_files
    assert set(manifest_entries(root)) == expected_files

    loaded = load_raw_response_refinement_artifact(
        root, _fixture_dependencies=dependencies
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
        construct_raw_response_refinement(
            config, _fixture_dependencies=dependencies
        )


def test_construction_requires_construction_to_bind_supplied_inference(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    dependencies.construction.metadata["inference_aggregate_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="construction inference aggregate"):
        construct_raw_response_refinement(
            config, _fixture_dependencies=dependencies
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

    result = construct_raw_response_refinement(config)

    assert calls == [
        f"inference:{config.inference_path}",
        f"construction:{config.construction_path}",
    ]
    assert result.metadata["dependency_aggregate_sha256"] == {
        "inference": "a" * 64,
        "construction": "b" * 64,
        "dense": dependencies.dense_aggregate_sha256,
    }


def test_writer_rejects_same_length_forged_selected_record_ids(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
    )
    selected = [[list(candidate) for candidate in query] for query in result.selected_record_ids]
    selected[0][0][0] = "forged-record"

    with pytest.raises(ValueError, match="selected record identity arithmetic"):
        write_raw_response_refinement_artifact(
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
    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
    )
    metadata = {**result.metadata, "raw_top_k": 9}

    with pytest.raises(ValueError, match="semantic reconstruction"):
        write_raw_response_refinement_artifact(
            replace(result, metadata=metadata), tmp_path / "forged-metadata"
        )
    assert not (tmp_path / "forged-metadata").exists()


def test_writer_refuses_overwrite_and_cleans_failed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
    )
    destination = tmp_path / "failed"

    def fail_on_validation(path: Path, payload: object) -> None:
        if path.name == "validation.json":
            raise RuntimeError("injected failure")
        write_pretty_json(path, payload)

    monkeypatch.setattr(
        "trec_mlm.raw_response_refinement._write_json", fail_on_validation
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        write_raw_response_refinement_artifact(result, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.*"))

    monkeypatch.undo()
    write_raw_response_refinement_artifact(result, destination)
    with pytest.raises(FileExistsError):
        write_raw_response_refinement_artifact(result, destination)


def test_atomic_promotion_does_not_replace_concurrently_created_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
    )
    destination = tmp_path / "raced"
    original = raw_response_refinement._write_json

    def create_destination_before_manifest(path: Path, payload: object) -> None:
        if path.name == "manifest.json":
            destination.mkdir()
        original(path, payload)

    monkeypatch.setattr(raw_response_refinement, "_write_json", create_destination_before_manifest)
    with pytest.raises(FileExistsError):
        write_raw_response_refinement_artifact(result, destination)
    assert destination.is_dir()
    assert not list(tmp_path.glob(".raced.*"))


def test_loader_rejects_hash_tamper_and_rehashed_semantic_tamper(
    tmp_path: Path,
) -> None:
    root, dependencies, _ = _write_fixture(tmp_path)
    ranking_path = root / "rankings" / "raw_hierarchical.json"
    ranking_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_raw_response_refinement_artifact(
            root, _fixture_dependencies=dependencies
        )

    root2, dependencies2, _ = _write_fixture(tmp_path / "second")
    scores = np.load(root2 / "hierarchical_scores.npy", allow_pickle=False)
    scores[0, 0] += 0.01
    np.save(root2 / "hierarchical_scores.npy", scores, allow_pickle=False)
    _refresh_manifest(root2, "hierarchical_scores.npy")
    with pytest.raises(ValueError, match="hierarchical score arithmetic"):
        load_raw_response_refinement_artifact(
            root2, _fixture_dependencies=dependencies2
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
        load_raw_response_refinement_artifact(
            root, _fixture_dependencies=dependencies
        )


def test_repeat_comparison_requires_identical_selections_and_rankings(
    tmp_path: Path,
) -> None:
    config, dependencies = _dependencies(tmp_path)
    result = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_raw_response_refinement_artifact(result, first)
    write_raw_response_refinement_artifact(replace(result), second)

    receipt = compare_raw_response_refinement_repeats(
        first, second, _fixture_dependencies=dependencies
    )
    assert receipt["status"] == "passed"
    assert receipt["selected_record_ids_identical"] is True
    assert receipt["rankings_identical"] is True
    assert receipt["maximum_absolute_score_difference"] <= 1e-12
    assert receipt["tolerance"] == 1e-12
