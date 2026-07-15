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


AUDIT_FINAL_ROOT = "artifacts/candidate_semantic_profiles/run1_record_id_v2_auditfinal"
AUDIT_FINAL_AGGREGATE = "04d6be23d2f9046ed93eb1c01bb5ad56545bb51a9470d408b96d3241b1ee54f4"


def test_config_rejects_protocol_drift_and_public_api_is_qrel_free() -> None:
    assert MultiResolutionConfig().construction_path == AUDIT_FINAL_ROOT
    assert MultiResolutionConfig().source_aggregate_sha256 == AUDIT_FINAL_AGGREGATE
    base = MultiResolutionConfig(output_root="output")
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


def test_evaluated_source_constants_are_public_and_exact() -> None:
    from trec_mlm import profile_multiresolution as module

    assert module.EVALUATED_CONSTRUCTION_PATH == AUDIT_FINAL_ROOT
    assert module.EVALUATED_CONSTRUCTION_AGGREGATE_SHA256 == AUDIT_FINAL_AGGREGATE


@pytest.mark.parametrize(
    "changed",
    (
        replace(
            MultiResolutionConfig(),
            construction_path="artifacts/alternate-valid-source",
        ),
        replace(MultiResolutionConfig(), source_aggregate_sha256="a" * 64),
    ),
)
def test_validate_config_rejects_alternate_valid_source(changed) -> None:
    from trec_mlm import profile_multiresolution as module

    with pytest.raises(ValueError, match="evaluated issue-12 source"):
        module._validate_config(changed)


def test_source_aggregate_binding_is_checked_before_construction(monkeypatch) -> None:
    config = MultiResolutionConfig(output_root="output")
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
                "converged": True,
                "iteration_count": 1,
                "audit": {"usable": 2},
                "clusters": [
                    {
                        "cluster_id": index,
                        "member_record_ids": [records[offset + index]["record_id"]],
                        "support_count": 1,
                        "membership_sha256": module._length_framed_hash([records[offset + index]["record_id"]]),
                        "dispersion": 0.0,
                        "medoid_record_id": records[offset + index]["record_id"],
                        "representative_questions": [records[offset + index]["question"]],
                        "annotation": {
                            "primary_label": candidate,
                            "secondary_label": None,
                            "top_score": 1.0,
                            "second_score": 0.0,
                            "margin": 1.0,
                            "support_count": 1,
                            "status": "primary",
                            "affects_ranking": False,
                        },
                    }
                    for index in range(2)
                ],
            }
            for candidate, offset in (("A", 0), ("B", 2))
        }
    dense = {"q1": ["B", "A"], "q2": ["A", "B"]}
    return module._assemble_result_for_test(
        config=MultiResolutionConfig(
            output_root=str(tmp_path / "artifact"),
        ),
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        records=records,
        query_embeddings=query_embeddings,
        record_embeddings=record_embeddings,
        profiles=profiles,
        dense_rankings=dense,
    )


def _refresh_manifest(destination: Path, relative_path: str) -> None:
    from trec_mlm import profile_multiresolution as module

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    changed = destination / relative_path
    manifest["files"][relative_path] = {
        "byte_size": changed.stat().st_size,
        "sha256": hashlib.sha256(changed.read_bytes()).hexdigest(),
    }
    manifest["aggregate_sha256"] = module._aggregate_hash(manifest["files"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _bind_expected(monkeypatch: pytest.MonkeyPatch, result) -> None:
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution.load_profile_artifacts",
        lambda _: type(
            "Loaded",
            (),
            {"manifest": {"aggregate_sha256": AUDIT_FINAL_AGGREGATE}},
        )(),
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution.construct_multiresolution", lambda config: result
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


def test_atomic_artifact_has_exact_manifest_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fixture_result(tmp_path)
    _bind_expected(monkeypatch, result)
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


@pytest.mark.parametrize(
    ("relative_path", "mutate"),
    (
        ("records.json", lambda value: value[0].__setitem__("response", "changed")),
        ("query_embeddings.npy", lambda value: value.__imul__(2.0)),
        ("record_embeddings.npy", lambda value: value.__imul__(2.0)),
        ("rankings/dense_reference.json", lambda value: value["q1"].reverse()),
    ),
)
def test_loader_binds_embedded_source_inputs_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    mutate,
) -> None:
    result = _fixture_result(tmp_path)
    _bind_expected(monkeypatch, result)
    destination = tmp_path / "artifact"
    write_multiresolution_artifact(result, destination)
    target = destination / relative_path
    if target.suffix == ".npy":
        value = np.load(target, allow_pickle=False)
        mutate(value)
        np.save(target, value)
    else:
        value = json.loads(target.read_text())
        mutate(value)
        target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    _refresh_manifest(destination, relative_path)
    with pytest.raises(ValueError, match="bound issue-12 source"):
        load_multiresolution_artifact(destination)


@pytest.mark.parametrize(
    ("location", "replacement"),
    (
        (("audit", "usable"), 99),
        (("clusters", 0, "annotation", "status"), "changed"),
        (("clusters", 0, "dispersion"), 0.25),
        (("clusters", 0, "medoid_record_id"), "A1"),
        (("clusters", 0, "representative_questions"), ["changed"]),
        (("converged",), False),
        (("iteration_count",), 99),
    ),
)
def test_loader_reconstructs_complete_canonical_profile_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: tuple[object, ...],
    replacement: object,
) -> None:
    result = _fixture_result(tmp_path)
    _bind_expected(monkeypatch, result)
    destination = tmp_path / "artifact"
    write_multiresolution_artifact(result, destination)
    relative_path = "profiles/k5.json"
    target = destination / relative_path
    payload = json.loads(target.read_text())
    cursor = payload["A"]
    for key in location[:-1]:
        cursor = cursor[key]
    cursor[location[-1]] = replacement
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _refresh_manifest(destination, relative_path)
    with pytest.raises(ValueError, match="canonical profile"):
        load_multiresolution_artifact(destination)


@pytest.mark.parametrize("target_name", ("config.json", "manifest.json"))
def test_loader_rejects_extra_schema_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    result = _fixture_result(tmp_path)
    _bind_expected(monkeypatch, result)
    destination = tmp_path / "artifact"
    write_multiresolution_artifact(result, destination)
    target = destination / target_name
    payload = json.loads(target.read_text())
    payload["qrels"] = "forbidden"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if target_name != "manifest.json":
        _refresh_manifest(destination, target_name)
    with pytest.raises(ValueError, match="schema"):
        load_multiresolution_artifact(destination)
