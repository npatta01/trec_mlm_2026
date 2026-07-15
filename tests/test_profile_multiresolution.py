from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from trec_mlm.profile_multiresolution import (
    EXPECTED_SYSTEMS,
    MultiResolutionConfig,
    construct_multiresolution,
    load_multiresolution_artifact,
    write_multiresolution_artifact,
)


def test_config_rejects_protocol_drift_and_public_api_is_qrel_free() -> None:
    base = MultiResolutionConfig(
        construction_path="source", output_root="output", source_aggregate_sha256="a" * 64
    )
    for changed in (
        replace(base, resolutions=(5, 10)),
        replace(base, top_count=3),
        replace(base, seed=7),
        replace(base, rrf_k=61),
        replace(base, source_commit="bad"),
        replace(base, source_aggregate_sha256="bad"),
    ):
        with pytest.raises((TypeError, ValueError)):
            construct_multiresolution(changed)
    for function in (
        construct_multiresolution,
        write_multiresolution_artifact,
        load_multiresolution_artifact,
    ):
        assert "qrel" not in str(inspect.signature(function)).lower()


def test_source_aggregate_binding_is_checked_before_construction(monkeypatch) -> None:
    config = MultiResolutionConfig(
        construction_path="source", output_root="output", source_aggregate_sha256="a" * 64
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution.load_profile_artifacts",
        lambda _: type("Loaded", (), {"manifest": {"aggregate_sha256": "b" * 64}})(),
    )
    with pytest.raises(ValueError, match="aggregate"):
        construct_multiresolution(config)


def _fixture_result(tmp_path: Path):
    from trec_mlm import profile_multiresolution as module

    query_ids = ("q1", "q2")
    candidate_ids = ("A", "B")
    records = [
        {"record_id": "A0", "candidate_id": "A", "source_query_id": "s0", "question": "qa", "response": "ra"},
        {"record_id": "A1", "candidate_id": "A", "source_query_id": "s1", "question": "qb", "response": "rb"},
        {"record_id": "B0", "candidate_id": "B", "source_query_id": "s0", "question": "qc", "response": "rc"},
        {"record_id": "B1", "candidate_id": "B", "source_query_id": "s1", "question": "qd", "response": "rd"},
    ]
    query_embeddings = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    record_embeddings = np.asarray([[1, 0], [0, 1], [.8, .6], [.6, .8]], dtype=np.float32)
    profiles = {}
    for resolution in (5, 10, 20):
        profiles[str(resolution)] = {
            candidate: {
                "available": True,
                "candidate_id": candidate,
                "requested_clusters": 2,
                "actual_clusters": 2,
                "centroids": record_embeddings[offset : offset + 2].tolist(),
                "clusters": [
                    {
                        "cluster_id": index,
                        "member_record_ids": [records[offset + index]["record_id"]],
                        "support_count": 1,
                        "membership_sha256": module._length_framed_hash([records[offset + index]["record_id"]]),
                    }
                    for index in range(2)
                ],
            }
            for candidate, offset in (("A", 0), ("B", 2))
        }
    dense = {"q1": ["B", "A"], "q2": ["A", "B"]}
    return module._assemble_result_for_test(
        config=MultiResolutionConfig(
            construction_path="fixture-source",
            output_root=str(tmp_path / "artifact"),
            source_aggregate_sha256="a" * 64,
        ),
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        records=records,
        query_embeddings=query_embeddings,
        record_embeddings=record_embeddings,
        profiles=profiles,
        dense_rankings=dense,
    )


def test_exact_c0_reconstruction_complete_ranks_and_system_inventory(tmp_path: Path) -> None:
    result = _fixture_result(tmp_path)
    assert set(result.rankings) == set(EXPECTED_SYSTEMS)
    for ranking in result.rankings.values():
        assert set(ranking) == {"q1", "q2"}
        assert all(len(row) == 2 and set(row) == {"A", "B"} for row in ranking.values())
    assert result.profiles["10"] == result.source_c0_profiles
    np.testing.assert_array_equal(
        result.system_scores["c0_max_k10"], result.component_scores["max_k10"]
    )


def test_atomic_artifact_has_exact_manifest_and_rejects_tampering(tmp_path: Path) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _fixture_result(tmp_path)
    destination = tmp_path / "artifact"
    paths = write_multiresolution_artifact(result, destination)
    loaded = load_multiresolution_artifact(destination)
    assert set(loaded.rankings) == set(EXPECTED_SYSTEMS)
    manifest = json.loads((destination / "manifest.json").read_text())
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["files"]) == actual == set(paths) - {"manifest.json"}
    for name, entry in manifest["files"].items():
        payload = (destination / name).read_bytes()
        assert entry == {"byte_size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    ranking_path = destination / "rankings" / "c0_max_k10.json"
    ranking = json.loads(ranking_path.read_text())
    ranking["q1"].reverse()
    ranking_path.write_text(json.dumps(ranking, sort_keys=True) + "\n")
    with pytest.raises(ValueError):
        load_multiresolution_artifact(destination)

    entry = {
        "byte_size": ranking_path.stat().st_size,
        "sha256": hashlib.sha256(ranking_path.read_bytes()).hexdigest(),
    }
    manifest["files"]["rankings/c0_max_k10.json"] = entry
    manifest["aggregate_sha256"] = module._aggregate_hash(manifest["files"])
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError):
        load_multiresolution_artifact(destination)
