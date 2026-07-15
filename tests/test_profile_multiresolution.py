from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from trec_mlm.profile_multiresolution import (
    EXPECTED_SYSTEMS,
    MultiResolutionConfig,
    assess_multiresolution_viability,
    compare_multiresolution_artifacts,
    construct_multiresolution,
    load_multiresolution_artifact,
    validate_multiresolution_viability_receipt,
    write_multiresolution_receipt,
    write_multiresolution_artifact,
)


AUDIT_FINAL_ROOT = "artifacts/candidate_semantic_profiles/run1_record_id_v2_auditfinal"
AUDIT_FINAL_AGGREGATE = "04d6be23d2f9046ed93eb1c01bb5ad56545bb51a9470d408b96d3241b1ee54f4"
ISSUE18_ROOT = Path("artifacts/multiresolution_profiles/issue18_pr17head")
ISSUE18_AGGREGATE = "2a3909cfea23af246eed59565557d57ec4c5b8eb1e2f1d5c2b11d2f9f8c6d5b9"


def _set_locked_manifest_mtime(root: Path) -> None:
    seconds = int(datetime.fromisoformat("2026-07-15T12:41:18-04:00").timestamp())
    nanoseconds = seconds * 1_000_000_000 + 692_102_441
    os.utime(root / "manifest.json", ns=(nanoseconds, nanoseconds))


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
        assess_multiresolution_viability,
        compare_multiresolution_artifacts,
        write_multiresolution_receipt,
    ):
        assert "qrel" not in str(inspect.signature(function)).lower()


def test_evaluated_source_constants_are_public_and_exact() -> None:
    from trec_mlm import profile_multiresolution as module

    assert module.EVALUATED_CONSTRUCTION_PATH == AUDIT_FINAL_ROOT
    assert module.EVALUATED_CONSTRUCTION_AGGREGATE_SHA256 == AUDIT_FINAL_AGGREGATE


def test_issue18_frozen_loader_is_nonconstructive_and_pins_exact_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trec_mlm import profile_multiresolution as module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen loader invoked construction or scoring")

    for name in (
        "construct_multiresolution",
        "canonical_kmeans",
        "score_profiles",
        "_freeze_systems",
        "reciprocal_rank_fusion",
    ):
        monkeypatch.setattr(module, name, forbidden)

    result = module.load_issue18_frozen_snapshot(ISSUE18_ROOT)
    assert result.config == MultiResolutionConfig()
    assert result.query_ids and len(result.query_ids) == 322
    assert result.candidate_ids and len(result.candidate_ids) == 150
    assert (
        json.loads((ISSUE18_ROOT / "manifest.json").read_text())["aggregate_sha256"]
        == ISSUE18_AGGREGATE
    )


def test_real_frozen_assessment_avoids_every_reconstruction_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trec_mlm import profile_experiment
    from trec_mlm import profile_multiresolution as module
    from trec_mlm import profile_retrieval

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen assessment invoked reconstruction")

    for owner, names in (
        (
            module,
            (
                "construct_multiresolution",
                "canonical_kmeans",
                "score_profiles",
                "_freeze_systems",
                "reciprocal_rank_fusion",
                "load_profile_artifacts",
                "_recompute_profile_centroid",
                "_rank_scores",
            ),
        ),
        (
            profile_experiment,
            ("load_profile_artifacts", "score_profiles", "_recompute_profile_centroid"),
        ),
        (profile_retrieval, ("canonical_kmeans", "score_profiles", "reciprocal_rank_fusion")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden, raising=False)

    receipt = module.assess_multiresolution_viability(ISSUE18_ROOT)
    assert receipt["artifact_aggregate_sha256"] == ISSUE18_AGGREGATE
    assert receipt["qrel_free"] is True


def test_issue18_frozen_loader_rejects_noncanonical_root(tmp_path: Path) -> None:
    from trec_mlm import profile_multiresolution as module

    with pytest.raises(ValueError, match="canonical issue-18 artifact root"):
        module.load_issue18_frozen_snapshot(tmp_path)


def test_frozen_manifest_rejects_alternate_self_consistent_aggregate(
    tmp_path: Path,
) -> None:
    from trec_mlm import profile_multiresolution as module

    alternate = module._aggregate_hash({})
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "issue18_multiresolution_profile_manifest",
                "version": 1,
                "qrel_free": True,
                "files": {},
                "aggregate_sha256": alternate,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="locked snapshot"):
        module._validate_frozen_manifest(
            tmp_path,
            artifact_type="issue18_multiresolution_profile_manifest",
            version=1,
            aggregate_sha256=ISSUE18_AGGREGATE,
        )


@pytest.mark.parametrize(
    "relative_path",
    (
        "records.json",
        "query_embeddings.npy",
        "record_embeddings.npy",
        "profiles.json",
    ),
)
def test_issue12_snapshot_token_detects_every_direct_bound_file_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    from trec_mlm import profile_multiresolution as module

    for name in (
        "records.json",
        "query_embeddings.npy",
        "record_embeddings.npy",
        "profiles.json",
        "inputs.json",
    ):
        (tmp_path / name).write_bytes(f"original:{name}\n".encode())
    entries = {
        name: {
            "byte_size": (tmp_path / name).stat().st_size,
            "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
        }
        for name in (
            "records.json",
            "query_embeddings.npy",
            "record_embeddings.npy",
            "profiles.json",
            "inputs.json",
        )
    }
    aggregate = module._aggregate_hash(entries)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "candidate_semantic_profile_manifest",
                "version": 2,
                "qrel_free": True,
                "files": entries,
                "aggregate_sha256": aggregate,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "EVALUATED_CONSTRUCTION_PATH", str(tmp_path))
    monkeypatch.setattr(module, "EVALUATED_CONSTRUCTION_AGGREGATE_SHA256", aggregate)
    token = module._capture_frozen_tree_token(tmp_path)
    snapshot = SimpleNamespace(
        root=tmp_path,
        tree_token=token,
        manifest=json.loads((tmp_path / "manifest.json").read_text()),
    )

    (tmp_path / relative_path).write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="token|file changed"):
        module.validate_issue12_source_snapshot(snapshot)


def test_viability_repair_locks_external_canonical_receipt_path(tmp_path: Path) -> None:
    from scripts import run_multiresolution_profiles as script

    with pytest.raises(ValueError, match="canonical viability receipt"):
        script._validate_viability_repair_paths(
            ISSUE18_ROOT, ISSUE18_ROOT / "receipt.json"
        )
    with pytest.raises(ValueError, match="canonical viability receipt"):
        script._validate_viability_repair_paths(ISSUE18_ROOT, tmp_path / "receipt.json")


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


@pytest.mark.parametrize("true_content_change", (False, True))
def test_construct_compares_profiles_using_strict_json_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    true_content_change: bool,
) -> None:
    from trec_mlm import profile_multiresolution as module

    root = tmp_path / "source"
    root.mkdir()
    records = [
        {
            "record_id": "r1",
            "source_query_id": "s1",
            "candidate_id": "A",
            "question": "question",
            "response": "response",
        }
    ]
    tuple_payload = {
        "available": True,
        "candidate_id": "A",
        "requested_clusters": 1,
        "actual_clusters": 1,
        "centroids": [[1.0, 0.0]],
        "clusters": [
            {
                "cluster_id": 0,
                "member_record_ids": ("r1",),
                "support_count": 1,
                "membership_sha256": module._length_framed_hash(["r1"]),
                "dispersion": 0.0,
                "medoid_record_id": "r1",
                "representative_questions": ("question",),
                "annotation": {
                    "primary_label": "A",
                    "secondary_label": None,
                    "top_score": 1.0,
                    "second_score": None,
                    "margin": None,
                    "support_count": 1,
                    "status": "primary",
                    "affects_ranking": False,
                },
            }
        ],
        "converged": True,
        "iteration_count": 1,
        "audit": {"usable": 1},
    }
    source_payload = json.loads(json.dumps(tuple_payload, allow_nan=False))
    if true_content_change:
        source_payload["clusters"][0]["support_count"] = 2
    (root / "records.json").write_text(json.dumps(records), encoding="utf-8")
    (root / "profiles.json").write_text(
        json.dumps({"A": source_payload}), encoding="utf-8"
    )
    (root / "inputs.json").write_text(
        json.dumps({"label_question_ids": ["s1"]}), encoding="utf-8"
    )
    source = SimpleNamespace(
        manifest={"aggregate_sha256": AUDIT_FINAL_AGGREGATE},
        root=root,
        query_ids=("q1",),
        candidate_ids=("A",),
        source_rankings={"dense_reference": {"q1": ["A"]}},
        profile_scores=np.asarray([[1.0]], dtype=np.float32),
        rankings={"profile_only": {"q1": ["A"]}},
    )
    arrays = {
        "query_embeddings.npy": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "record_embeddings.npy": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "label_question_embeddings.npy": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "label_embeddings.npy": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }
    monkeypatch.setattr(module, "load_profile_artifacts", lambda _: source)
    monkeypatch.setattr(
        module,
        "canonical_kmeans",
        lambda *_args, **_kwargs: module.CandidateProfile(
            "A",
            1,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            (),
            True,
            1,
        ),
    )
    monkeypatch.setattr(
        module.np,
        "load",
        lambda path, **_kwargs: arrays[Path(path).name],
    )
    monkeypatch.setattr(
        module,
        "_profile_to_json",
        lambda _profile, _audit: tuple_payload,
    )

    if true_content_change:
        with pytest.raises(ValueError, match="C0 reconstruction differs"):
            construct_multiresolution(MultiResolutionConfig())
    else:
        result = construct_multiresolution(MultiResolutionConfig())
        assert result.profiles["10"] == {"A": source_payload}
        assert isinstance(
            result.profiles["10"]["A"]["clusters"][0]["member_record_ids"], list
        )


def test_json_profile_canonicalization_rejects_nonfinite_and_nonstring_keys() -> None:
    from trec_mlm import profile_multiresolution as module

    with pytest.raises(ValueError, match="finite JSON"):
        module._json_native_profile_payload({"score": float("nan")})
    with pytest.raises(ValueError, match="string keys"):
        module._json_native_profile_payload({1: "changed"})


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


def _diagnostic_fixture(tmp_path: Path):
    result = _fixture_result(tmp_path)
    profiles = {
        resolution: {
            candidate: {
                **profile,
                "requested_clusters": 2,
                "actual_clusters": 2,
                "centroids": [[1.0, 0.0], [0.0, 1.0]],
                "clusters": [
                    {**profile["clusters"][0], "support_count": 3},
                    {**profile["clusters"][1], "support_count": 1},
                ],
            }
            for candidate, profile in payload.items()
        }
        for resolution, payload in result.profiles.items()
    }
    rankings = {
        system: {
            "q1": ["A", "B"],
            "q2": ["B", "A"],
        }
        for system in EXPECTED_SYSTEMS
    }
    system_scores = {
        system: np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        for system in EXPECTED_SYSTEMS
    }
    return replace(
        result, profiles=profiles, rankings=rankings, system_scores=system_scores
    )


def test_viability_reports_exact_cluster_formulas_and_threshold_names(
    tmp_path: Path,
) -> None:
    from trec_mlm import profile_multiresolution as module

    receipt = module._assess_multiresolution_viability(
        _diagnostic_fixture(tmp_path), require_official_coverage=False
    )
    thresholds = receipt["thresholds"]
    expected = {
        "candidate_occupied_cluster_retention",
        "largest_cluster_share",
        "median_normalized_occupancy_entropy",
        "minimum_normalized_occupancy_entropy",
        "singleton_cluster_share",
        "exact_duplicate_centroids",
        "within_candidate_near_duplicate_centroid_share",
        "median_centroid_separation",
        "unique_top10_rankings",
        "maximum_top1_candidate_share",
        "top10_candidate_coverage",
        "maximum_candidate_top10_query_share",
        "incremental_top10_difference_query_share",
        "incremental_mean_top10_jaccard",
        "incremental_unique_top10_retention",
        "incremental_candidate_coverage_retention",
    }
    assert set(thresholds) == expected
    cluster = receipt["cluster_diagnostics"]["k5"]
    assert cluster["largest_cluster_share"] == pytest.approx(0.75)
    assert cluster["median_normalized_occupancy_entropy"] == pytest.approx(
        -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)) / np.log(2)
    )
    assert cluster["singleton_cluster_share"] == pytest.approx(0.5)
    assert cluster["exact_duplicate_centroids"] == 0
    assert cluster["within_candidate_near_duplicate_centroid_share"] == 0.0
    assert cluster["median_centroid_separation"] == pytest.approx(1.0)


def test_viability_reports_query_collapse_dominance_and_incremental_difference(
    tmp_path: Path,
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    changed = {
        **result.rankings,
        "a1_top2_k10": {"q1": ["B", "A"], "q2": ["B", "A"]},
    }
    receipt = module._assess_multiresolution_viability(
        replace(result, rankings=changed), require_official_coverage=False
    )
    c0 = receipt["ranking_diagnostics"]["c0_max_k10"]
    a1 = receipt["ranking_diagnostics"]["a1_top2_k10"]
    assert c0["unique_top10_rankings"] == 2
    assert c0["maximum_top1_candidate_share"] == pytest.approx(0.5)
    assert c0["top10_candidate_coverage"] == 2
    assert c0["maximum_candidate_top10_query_share"] == pytest.approx(1.0)
    assert a1["top10_difference_query_share"] == pytest.approx(0.5)
    assert a1["mean_top10_jaccard"] == pytest.approx(1.0)
    assert a1["unique_top10_retention"] == pytest.approx(0.5)
    assert a1["candidate_coverage_retention"] == pytest.approx(1.0)


def test_control_comparison_reports_exact_top10_overlap_statistics() -> None:
    from trec_mlm import profile_multiresolution as module

    observed = {
        "q1": ["A", "B", "C"],
        "q2": ["B", "A", "C"],
    }
    control = {
        "q1": ["A", "C", "D"],
        "q2": ["A", "B", "C"],
    }
    comparison = module._compare_top10_rankings(observed, control)
    assert comparison == {
        "mean_top10_jaccard": pytest.approx(0.75),
        "top10_difference_query_share": pytest.approx(1.0),
        "mean_top10_overlap_count": pytest.approx(2.5),
        "mean_common_top10_absolute_rank_displacement": pytest.approx(0.6),
    }


def test_control_comparisons_cover_all_frozen_systems_and_both_controls(
    tmp_path: Path,
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    comparisons = module._control_comparisons(
        result.rankings,
        {
            "query_knn": result.dense_rankings,
            "contextual_e5": {
                "q1": ["A", "B"],
                "q2": ["A", "B"],
            },
        },
    )
    assert set(comparisons) == set(EXPECTED_SYSTEMS)
    assert all(set(values) == {"query_knn", "contextual_e5"} for values in comparisons.values())


def test_control_sources_bind_exact_hashes_and_inventories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    artifact_root = tmp_path / "artifact"
    source_root = tmp_path / "source"
    (artifact_root / "rankings").mkdir(parents=True)
    (source_root / "rankings").mkdir(parents=True)
    dense_path = artifact_root / "rankings" / "dense_reference.json"
    contextual_path = source_root / "rankings" / "contextual_e5_control.json"
    dense_path.write_text(json.dumps(result.dense_rankings), encoding="utf-8")
    contextual = {"q1": ["A", "B"], "q2": ["B", "A"]}
    contextual_path.write_text(json.dumps(contextual), encoding="utf-8")
    dense_sha = hashlib.sha256(dense_path.read_bytes()).hexdigest()
    contextual_sha = hashlib.sha256(contextual_path.read_bytes()).hexdigest()
    (artifact_root / "manifest.json").write_text(json.dumps({
        "aggregate_sha256": "f" * 64,
        "files": {"rankings/dense_reference.json": {"sha256": dense_sha}},
    }))
    source = SimpleNamespace(
        root=source_root,
        manifest={
            "aggregate_sha256": AUDIT_FINAL_AGGREGATE,
            "files": {
                "rankings/contextual_e5_control.json": {"sha256": contextual_sha},
                "rankings/dense_reference.json": {"sha256": dense_sha},
            },
        },
        query_ids=result.query_ids,
        candidate_ids=result.candidate_ids,
        source_rankings={
            "dense_reference": result.dense_rankings,
            "contextual_e5_control": contextual,
        },
    )
    monkeypatch.setattr(module, "validate_issue12_source_snapshot", lambda _source: None)
    result = replace(result, source_snapshot=source)

    sources, controls = module._load_control_rankings(result, artifact_root)
    assert set(sources) == {"query_knn", "contextual_e5"}
    assert sources["query_knn"]["ranking_sha256"] == dense_sha
    assert sources["contextual_e5"]["ranking_sha256"] == contextual_sha
    assert sources["contextual_e5"]["source_aggregate_sha256"] == AUDIT_FINAL_AGGREGATE
    assert controls == {
        "query_knn": result.dense_rankings,
        "contextual_e5": contextual,
    }


@pytest.mark.parametrize("mismatch", ("candidate_inventory", "contextual_hash"))
def test_control_sources_reject_inventory_or_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str,
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    artifact_root = tmp_path / "artifact"
    source_root = tmp_path / "source"
    (artifact_root / "rankings").mkdir(parents=True)
    (source_root / "rankings").mkdir(parents=True)
    dense_path = artifact_root / "rankings" / "dense_reference.json"
    contextual_path = source_root / "rankings" / "contextual_e5_control.json"
    dense_path.write_text(json.dumps(result.dense_rankings))
    contextual_path.write_text(json.dumps(result.dense_rankings))
    dense_sha = hashlib.sha256(dense_path.read_bytes()).hexdigest()
    contextual_sha = hashlib.sha256(contextual_path.read_bytes()).hexdigest()
    (artifact_root / "manifest.json").write_text(json.dumps({
        "aggregate_sha256": "f" * 64,
        "files": {"rankings/dense_reference.json": {"sha256": dense_sha}},
    }))
    source = SimpleNamespace(
        root=source_root,
        manifest={
            "aggregate_sha256": AUDIT_FINAL_AGGREGATE,
            "files": {
                "rankings/contextual_e5_control.json": {
                    "sha256": "0" * 64 if mismatch == "contextual_hash" else contextual_sha
                },
                "rankings/dense_reference.json": {"sha256": dense_sha},
            },
        },
        query_ids=result.query_ids,
        candidate_ids=("A",) if mismatch == "candidate_inventory" else result.candidate_ids,
        source_rankings={
            "dense_reference": result.dense_rankings,
            "contextual_e5_control": result.dense_rankings,
        },
    )
    monkeypatch.setattr(module, "validate_issue12_source_snapshot", lambda _source: None)
    result = replace(result, source_snapshot=source)
    with pytest.raises(ValueError, match="inventory|hash"):
        module._load_control_rankings(result, artifact_root)


def test_execution_provenance_is_complete_and_exact() -> None:
    from trec_mlm import profile_multiresolution as module

    provenance = module._execution_provenance(
        "a" * 64, receipt_generated_at="2026-07-15T18:00:00+00:00"
    )
    assert provenance == {
        "execution_code_commit": module.PASS_ONE_EXECUTION_CODE_COMMIT,
        "command": module.PASS_ONE_COMMAND,
        "thread_environment": module.PASS_ONE_THREAD_ENVIRONMENT,
        "dependency_versions": module.PASS_ONE_DEPENDENCY_VERSIONS,
        "construction_completed_at": module.PASS_ONE_CONSTRUCTION_COMPLETED_AT,
        "artifact_aggregate_sha256": "a" * 64,
        "receipt_generated_at": "2026-07-15T18:00:00+00:00",
    }


def test_assessor_produces_complete_supplemental_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"aggregate_sha256": "a" * 64}), encoding="utf-8"
    )
    _set_locked_manifest_mtime(root)
    controls = {
        "query_knn": result.dense_rankings,
        "contextual_e5": result.dense_rankings,
    }
    sources = {
        name: {
            "ranking_name": "dense_reference" if name == "query_knn" else "contextual_e5_control",
            "ranking_file": "rankings/control.json",
            "ranking_sha256": "b" * 64,
            "source_aggregate_sha256": AUDIT_FINAL_AGGREGATE,
            "query_count": 2,
            "candidate_count": 2,
            "query_inventory_sha256": module._length_framed_hash(result.query_ids),
            "candidate_inventory_sha256": module._length_framed_hash(result.candidate_ids),
        }
        for name in module.CONTROL_NAMES
    }
    monkeypatch.setattr(module, "_load_control_rankings", lambda *_args: (sources, controls))
    monkeypatch.setattr(module, "_tree_bytes", lambda path: 100 if path == root else 50)
    monkeypatch.setattr(module, "_source_elapsed_seconds", lambda _path: 10.0)
    receipt = module._assess_multiresolution_viability(
        result, require_official_coverage=False, root=root
    )
    assert set(receipt["control_comparisons"]) == set(EXPECTED_SYSTEMS)
    assert set(receipt["control_comparisons"]["x1_multires_top2_dense_equal"]) == {
        "query_knn", "contextual_e5"
    }
    assert receipt["control_sources"] == sources
    assert receipt["execution_provenance"]["artifact_aggregate_sha256"] == "a" * 64
    assert (
        receipt["execution_provenance"]["construction_completed_at"]
        == module.PASS_ONE_CONSTRUCTION_COMPLETED_AT
    )
    assert receipt["failed_query_sensitivity_thresholds"]
    assert receipt["decision"] == "reject-invalid"
    assert receipt["repeat_authorized"] is False

    copied = tmp_path / "copied-artifact"
    shutil.copytree(root, copied, copy_function=shutil.copy)
    assert (copied / "manifest.json").read_bytes() == (root / "manifest.json").read_bytes()
    changed_ns = (copied / "manifest.json").stat().st_mtime_ns + 1_000_000_000
    os.utime(copied / "manifest.json", ns=(changed_ns, changed_ns))
    with pytest.raises(ValueError, match="locked construction completion timestamp"):
        module._assess_multiresolution_viability(
            result, require_official_coverage=False, root=copied
        )

    incomplete = dict(receipt)
    del incomplete["execution_provenance"]
    with pytest.raises(ValueError, match="authoritative"):
        validate_multiresolution_viability_receipt(incomplete, root)


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        (
            "ranking diagnostics",
            lambda receipt: receipt["ranking_diagnostics"]["c0_max_k10"].__setitem__(
                "unique_top10_rankings", 999
            ),
        ),
        (
            "control sources",
            lambda receipt: receipt["control_sources"]["contextual_e5"].__setitem__(
                "query_inventory_sha256", "f" * 64
            ),
        ),
        (
            "control comparisons",
            lambda receipt: receipt["control_comparisons"]["c0_max_k10"][
                "query_knn"
            ].__setitem__("mean_top10_jaccard", 0.123),
        ),
        (
            "cost evidence",
            lambda receipt: receipt["cost_evidence"].__setitem__(
                "multiresolution_artifact_storage_bytes", 999
            ),
        ),
        (
            "provenance",
            lambda receipt: receipt["execution_provenance"].__setitem__(
                "execution_code_commit", "f" * 40
            ),
        ),
        (
            "provenance timestamp",
            lambda receipt: receipt["execution_provenance"].__setitem__(
                "receipt_generated_at", "not-a-timestamp"
            ),
        ),
    ),
)
def test_receipt_validator_recomputes_authoritative_artifact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutate,
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"aggregate_sha256": "a" * 64}), encoding="utf-8"
    )
    _set_locked_manifest_mtime(root)
    controls = {
        "query_knn": result.dense_rankings,
        "contextual_e5": result.dense_rankings,
    }
    inventory = {
        "query_count": len(result.query_ids),
        "candidate_count": len(result.candidate_ids),
        "query_inventory_sha256": module._length_framed_hash(result.query_ids),
        "candidate_inventory_sha256": module._length_framed_hash(result.candidate_ids),
    }
    sources = {
        name: {
            "ranking_name": "dense_reference" if name == "query_knn" else "contextual_e5_control",
            "ranking_file": "rankings/dense_reference.json" if name == "query_knn" else "rankings/contextual_e5_control.json",
            "ranking_sha256": ("b" if name == "query_knn" else "c") * 64,
            "source_aggregate_sha256": AUDIT_FINAL_AGGREGATE,
            **inventory,
        }
        for name in module.CONTROL_NAMES
    }
    monkeypatch.setattr(module, "load_issue18_frozen_snapshot", lambda path: result)
    monkeypatch.setattr(module, "_load_control_rankings", lambda *_args: (sources, controls))
    monkeypatch.setattr(module, "_tree_bytes", lambda path: 100 if path == root else 50)
    monkeypatch.setattr(module, "_source_elapsed_seconds", lambda _path: 10.0)
    canonical = module._assess_multiresolution_viability(
        result, require_official_coverage=True, root=root
    )
    assert validate_multiresolution_viability_receipt(canonical, root) == canonical

    tampered = copy.deepcopy(canonical)
    mutate(tampered)
    with pytest.raises(ValueError, match="authoritative"):
        validate_multiresolution_viability_receipt(tampered, root)


def test_receipt_validator_requires_persisted_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="persisted artifact"):
        validate_multiresolution_viability_receipt({}, tmp_path / "missing")


@pytest.mark.parametrize(
    ("failed_integrity", "failed_query_sensitivity", "failed_incremental", "expected"),
    (
        (["finite_scores"], ["maximum_top1_candidate_share"], ["incremental"], "reject-invalid"),
        ([], ["maximum_top1_candidate_share"], [], "stop-query-sensitivity"),
        ([], [], ["incremental_mean_top10_jaccard"], "stop-no-incremental-value"),
        ([], [], [], "continue"),
    ),
)
def test_viability_decision_uses_strict_failure_precedence(
    failed_integrity: list[str],
    failed_query_sensitivity: list[str],
    failed_incremental: list[str],
    expected: str,
) -> None:
    from trec_mlm import profile_multiresolution as module

    assert module._viability_decision(
        failed_integrity,
        failed_query_sensitivity,
        failed_incremental,
    ) == expected


def test_viability_classifies_integrity_before_incremental_value(
    tmp_path: Path,
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    invalid = replace(
        result,
        system_scores={
            **result.system_scores,
            "c0_max_k10": np.full((2, 2), np.nan, dtype=np.float32),
        },
    )
    invalid_receipt = module._assess_multiresolution_viability(
        invalid, require_official_coverage=False
    )
    assert invalid_receipt["decision"] == "reject-invalid"
    assert "finite_scores" in invalid_receipt["failed_thresholds"]

    structurally_valid_profiles = {
        resolution: {
            candidate: {
                **profile,
                "clusters": [
                    {**profile["clusters"][0], "support_count": 3},
                    {**profile["clusters"][1], "support_count": 2},
                ],
            }
            for candidate, profile in payload.items()
        }
        for resolution, payload in result.profiles.items()
    }
    no_increment = module._assess_multiresolution_viability(
        replace(result, profiles=structurally_valid_profiles),
        require_official_coverage=False,
    )
    assert no_increment["decision"] == "stop-query-sensitivity"
    assert not no_increment["failed_integrity_thresholds"]
    assert no_increment["failed_incremental_thresholds"]


def _official_viability_fixture(tmp_path: Path):
    fixture = _fixture_result(tmp_path)
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    candidate_ids = tuple(f"C{index:03d}" for index in range(150))
    ranking = list(candidate_ids)
    rankings = {
        system: {query_id: list(ranking) for query_id in query_ids}
        for system in EXPECTED_SYSTEMS
    }
    profiles = {
        str(resolution): {
            candidate_id: {
                "available": True,
                "candidate_id": candidate_id,
                "requested_clusters": 1,
                "actual_clusters": 1,
                "centroids": [[1.0, 0.0]],
                "clusters": [{"support_count": 2}],
            }
            for candidate_id in candidate_ids
        }
        for resolution in (5, 10, 20)
    }
    component_scores = {
        key: np.zeros((322, 150), dtype=np.float32)
        for key in fixture.component_scores
    }
    system_scores = {
        system: np.zeros((322, 150), dtype=np.float32)
        for system in EXPECTED_SYSTEMS
    }
    contributors = {
        key: np.zeros(
            (322, 150, 1 if key.startswith("max_") else 2), dtype=np.int32
        )
        for key in fixture.contributing_centroid_ids
    }
    return replace(
        fixture,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        profiles=profiles,
        source_c0_profiles=profiles["10"],
        component_scores=component_scores,
        contributing_centroid_ids=contributors,
        rankings=rankings,
        system_scores=system_scores,
    )


def test_public_viability_rejects_nonofficial_coverage(tmp_path: Path) -> None:
    from trec_mlm import profile_multiresolution as module

    receipt = module._assess_multiresolution_viability(
        _diagnostic_fixture(tmp_path), require_official_coverage=True
    )
    assert receipt["decision"] == "reject-invalid"
    assert "official_322_by_150_coverage" in receipt["failed_integrity_thresholds"]


@pytest.mark.parametrize(
    "failure", ("shape", "nonfinite", "tie_order", "missing_ranking")
)
def test_public_viability_rejects_component_and_ranking_score_drift(
    tmp_path: Path, failure: str
) -> None:
    result = _official_viability_fixture(tmp_path)
    if failure == "shape":
        components = dict(result.component_scores)
        components["max_k5"] = components["max_k5"][:, :-1]
        result = replace(result, component_scores=components)
    elif failure == "nonfinite":
        components = {key: value.copy() for key, value in result.component_scores.items()}
        components["max_k5"][0, 0] = np.nan
        result = replace(result, component_scores=components)
    elif failure == "tie_order":
        rankings = {
            system: {query_id: list(row) for query_id, row in values.items()}
            for system, values in result.rankings.items()
        }
        rankings["c0_max_k10"]["q000"] = list(reversed(result.candidate_ids))
        result = replace(result, rankings=rankings)
    else:
        rankings = dict(result.rankings)
        del rankings["x1_multires_top2_dense_equal"]
        result = replace(result, rankings=rankings)
    from trec_mlm import profile_multiresolution as module

    receipt = module._assess_multiresolution_viability(
        result, require_official_coverage=True
    )
    assert receipt["decision"] == "reject-invalid"
    expected = (
        "component_score_schema"
        if failure in {"shape", "nonfinite"}
        else "ranking_score_consistency"
        if failure == "tie_order"
        else "complete_rankings"
    )
    assert expected in receipt["failed_integrity_thresholds"]


def test_public_viability_requires_persisted_artifact(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="persisted artifact"):
        assess_multiresolution_viability(_diagnostic_fixture(tmp_path))


@pytest.mark.parametrize("system", EXPECTED_SYSTEMS)
def test_collapse_and_dominance_gates_cover_every_frozen_ranking(
    tmp_path: Path, system: str
) -> None:
    from trec_mlm import profile_multiresolution as module

    result = _diagnostic_fixture(tmp_path)
    rankings = {
        **result.rankings,
        system: {"q1": ["A", "B"], "q2": ["A", "B"]},
    }
    receipt = module._assess_multiresolution_viability(
        replace(result, rankings=rankings), require_official_coverage=False
    )
    assert system in receipt["ranking_diagnostics"]
    assert receipt["ranking_diagnostics"][system]["maximum_top1_candidate_share"] == 1.0
    assert (
        receipt["thresholds"]["maximum_top1_candidate_share"]["values"][system]
        == 1.0
    )


def test_repeat_comparison_requires_independence_and_exact_scientific_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_result = _fixture_result(tmp_path / "first")
    second_result = replace(
        first_result,
        config=replace(first_result.config, output_root=str(tmp_path / "second")),
    )
    _bind_expected(monkeypatch, first_result)
    first = tmp_path / "first-artifact"
    second = tmp_path / "second-artifact"
    write_multiresolution_artifact(first_result, first)
    write_multiresolution_artifact(second_result, second)
    receipt = compare_multiresolution_artifacts(first, second)
    assert receipt["status"] == "passed"
    assert receipt["first_aggregate_sha256"] != receipt["second_aggregate_sha256"]
    assert receipt["source_arrays_reused"] is True
    assert receipt["canonical_scientific_files_identical"] is True
    assert receipt["maximum_absolute_score_difference"] == 0.0
    json.dumps(receipt, allow_nan=False)

    with pytest.raises(ValueError, match="independent"):
        compare_multiresolution_artifacts(first, first)


def test_repeat_comparison_reports_deliberately_changed_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fixture_result(tmp_path)
    _bind_expected(monkeypatch, result)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_multiresolution_artifact(result, first)
    write_multiresolution_artifact(
        replace(result, config=replace(result.config, output_root="different")), second
    )
    score_path = second / "scores" / "a1_top2_k10.npy"
    scores = np.load(score_path, allow_pickle=False)
    scores[0, 0] += np.float32(0.01)
    np.save(score_path, scores)
    _refresh_manifest(second, "scores/a1_top2_k10.npy")
    with pytest.raises(ValueError, match="semantic reconstruction"):
        compare_multiresolution_artifacts(first, second)


def test_repeat_comparison_strictly_rejects_identically_rehashed_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fixture_result(tmp_path)
    _bind_expected(monkeypatch, result)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_multiresolution_artifact(result, first)
    write_multiresolution_artifact(
        replace(result, config=replace(result.config, output_root="different")), second
    )
    for root in (first, second):
        target = root / "profiles" / "k5.json"
        payload = json.loads(target.read_text())
        payload["A"]["audit"]["usable"] = 99
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        _refresh_manifest(root, "profiles/k5.json")
    with pytest.raises(ValueError, match="canonical profile"):
        compare_multiresolution_artifacts(first, second)


def test_receipt_serialization_is_atomic_finite_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reports" / "receipt.json"
    payload = {"status": "passed", "maximum": 0.0}
    write_multiresolution_receipt(payload, destination)
    assert json.loads(destination.read_text()) == payload
    with pytest.raises(FileExistsError):
        write_multiresolution_receipt(payload, destination)
    with pytest.raises(ValueError, match="finite"):
        write_multiresolution_receipt({"maximum": float("nan")}, tmp_path / "bad.json")


def test_receipt_atomic_commit_preserves_file_created_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trec_mlm import profile_multiresolution as module

    destination = tmp_path / "receipt.json"
    attacker = b"attacker-owned\n"
    original_replace = module.os.replace
    original_link = module.os.link

    def race_replace(source, target):
        Path(target).write_bytes(attacker)
        return original_replace(source, target)

    def race_link(source, target):
        Path(target).write_bytes(attacker)
        return original_link(source, target)

    monkeypatch.setattr(module.os, "replace", race_replace)
    monkeypatch.setattr(module.os, "link", race_link)

    with pytest.raises(FileExistsError):
        write_multiresolution_receipt({"status": "valid"}, destination)
    assert destination.read_bytes() == attacker


def test_cli_accepts_only_qrel_free_viability_and_repeat_receipts() -> None:
    from scripts.run_multiresolution_profiles import _parser

    args = _parser().parse_args(
        [
            "--config", "config.json",
            "--viability-receipt", "viability.json",
            "--first-artifact", "first",
            "--first-viability-receipt", "first-viability.json",
            "--repeat-artifact", "repeat",
            "--repeatability-receipt", "repeatability.json",
        ]
    )
    assert args.viability_receipt == "viability.json"
    assert args.first_artifact == "first"
    assert args.first_viability_receipt == "first-viability.json"
    assert args.repeat_artifact == "repeat"
    assert args.repeatability_receipt == "repeatability.json"
    with pytest.raises(SystemExit):
        _parser().parse_args(["--qrels", "forbidden"])


def _authorization_receipt(
    *, authorized: bool, aggregate_sha256: str = "a" * 64
) -> dict[str, object]:
    return {
        "artifact_type": "issue18_multiresolution_viability_receipt",
        "version": 1,
        "qrel_free": True,
        "decision": "continue" if authorized else "stop-no-incremental-value",
        "artifact_aggregate_sha256": aggregate_sha256,
        "repeat_authorized": authorized,
        "cost_evidence": {
            "storage_ratio": 1.0 if authorized else 5.0,
            "wall_time_ratio": 1.0 if authorized else 5.0,
            "storage_at_most_four_times_issue12": authorized,
            "wall_time_at_most_four_times_issue12": authorized,
            "repeat_cost_authorized": authorized,
        },
    }


def _write_cli_config(path: Path, *, output_root: str) -> None:
    payload = asdict(MultiResolutionConfig(output_root=output_root))
    path.write_text(json.dumps(payload))


def test_cli_exact_canonical_config_preserves_pass_one_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_multiresolution_profiles as script

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.json"
    _write_cli_config(config_path, output_root=MultiResolutionConfig().output_root)
    monkeypatch.setattr(script, "_run_construction", lambda args, config: 17)
    assert script.main(["--config", str(config_path)]) == 17


def test_cli_refuses_existing_canonical_pass_one_before_construct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    monkeypatch.chdir(tmp_path)
    config = MultiResolutionConfig()
    config_path = tmp_path / "config.json"
    _write_cli_config(config_path, output_root=config.output_root)
    Path(config.output_root).mkdir(parents=True)
    monkeypatch.setattr(
        script,
        "construct_multiresolution",
        lambda config: pytest.fail("ambiguous canonical pass one was reconstructed"),
    )
    with pytest.raises(FileExistsError, match="canonical pass-one"):
        script.main(["--config", str(config_path)])


def test_cli_existing_canonical_artifact_repairs_receipt_without_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    monkeypatch.chdir(tmp_path)
    config = MultiResolutionConfig()
    config_path = tmp_path / "config.json"
    _write_cli_config(config_path, output_root=config.output_root)
    Path(config.output_root).mkdir(parents=True)
    output = script.CANONICAL_VIABILITY_RECEIPT_PATH
    expected = {"artifact_type": "issue18_multiresolution_viability_receipt"}
    monkeypatch.setattr(
        script,
        "construct_multiresolution",
        lambda _config: pytest.fail("receipt repair reconstructed production artifacts"),
    )
    monkeypatch.setattr(
        script, "assess_multiresolution_viability", lambda _path: expected
    )
    assert script.main(
        [
            "--config", str(config_path),
            "--viability-receipt", str(output),
        ]
    ) == 0
    assert json.loads(output.read_text()) == expected


@pytest.mark.parametrize(
    "receipt_kind",
    ("absent", "invalid", "noncontinue", "stop-query-sensitivity", "overbudget"),
)
def test_cli_repeat_construction_is_blocked_before_construct_without_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_kind: str,
) -> None:
    from scripts import run_multiresolution_profiles as script

    config_path = tmp_path / "repeat-config.json"
    _write_cli_config(config_path, output_root=str(tmp_path / "repeat-output"))
    first = tmp_path / "first"
    first.mkdir()
    (first / "manifest.json").write_text(
        json.dumps({"aggregate_sha256": "a" * 64})
    )
    arguments = ["--config", str(config_path), "--first-artifact", str(first)]
    if receipt_kind != "absent":
        authorization = tmp_path / "viability.json"
        if receipt_kind == "invalid":
            authorization.write_text("not-json")
        else:
            payload = _authorization_receipt(authorized=True)
            if receipt_kind == "noncontinue":
                payload["decision"] = "stop-no-incremental-value"
            elif receipt_kind == "stop-query-sensitivity":
                payload["decision"] = "stop-query-sensitivity"
                payload["repeat_authorized"] = False
            else:
                payload["cost_evidence"]["wall_time_ratio"] = 4.01
            authorization.write_text(json.dumps(payload))
        arguments.extend(["--first-viability-receipt", str(authorization)])
    monkeypatch.setattr(
        script,
        "construct_multiresolution",
        lambda config: pytest.fail("unauthorized repeat construction began"),
    )
    with pytest.raises(ValueError):
        script.main(arguments)


def test_cli_omitted_repeat_flag_cannot_bypass_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    config_path = tmp_path / "repeat-config.json"
    _write_cli_config(config_path, output_root=str(tmp_path / "repeat-output"))
    monkeypatch.setattr(
        script,
        "construct_multiresolution",
        lambda config: pytest.fail("noncanonical construction bypassed authorization"),
    )
    with pytest.raises(ValueError, match="requires first artifact"):
        script.main(["--config", str(config_path)])


def test_cli_forged_minimal_first_artifact_cannot_authorize_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    config_path = tmp_path / "repeat-config.json"
    _write_cli_config(config_path, output_root=str(tmp_path / "repeat-output"))
    first = tmp_path / "first"
    first.mkdir()
    (first / "manifest.json").write_text(json.dumps({"aggregate_sha256": "a" * 64}))
    authorization = tmp_path / "viability.json"
    authorization.write_text(json.dumps(_authorization_receipt(authorized=True)))
    monkeypatch.setattr(
        script,
        "construct_multiresolution",
        lambda config: pytest.fail("forged first artifact authorized construction"),
    )
    with pytest.raises(ValueError, match="invalid first viability receipt"):
        script.main(
            [
                "--config", str(config_path),
                "--first-artifact", str(first),
                "--first-viability-receipt", str(authorization),
            ]
        )


def test_cli_strict_valid_first_artifact_authorizes_noncanonical_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    result = replace(
        _fixture_result(tmp_path),
        config=MultiResolutionConfig(),
    )
    _bind_expected(monkeypatch, result)
    first = tmp_path / "first"
    write_multiresolution_artifact(result, first)
    aggregate = json.loads((first / "manifest.json").read_text())["aggregate_sha256"]
    authorization = tmp_path / "viability.json"
    authorization.write_text(
        json.dumps(
            _authorization_receipt(
                authorized=True, aggregate_sha256=aggregate
            )
        )
    )
    config_path = tmp_path / "repeat-config.json"
    _write_cli_config(config_path, output_root=str(tmp_path / "repeat-output"))
    called = []
    monkeypatch.setattr(
        script,
        "_run_construction",
        lambda args, config: called.append((args, config)) or 0,
    )
    validated = []
    monkeypatch.setattr(
        script,
        "validate_multiresolution_viability_receipt",
        lambda payload, artifact: validated.append((payload, artifact)) or payload,
    )
    assert script.main(
        [
            "--config", str(config_path),
            "--first-artifact", str(first),
            "--first-viability-receipt", str(authorization),
        ]
    ) == 0
    assert len(called) == 1
    assert validated == [(_authorization_receipt(authorized=True, aggregate_sha256=aggregate), str(first))]


def test_cli_strict_but_noncanonical_first_artifact_cannot_authorize_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    result = _fixture_result(tmp_path)
    _bind_expected(monkeypatch, result)
    first = tmp_path / "first"
    write_multiresolution_artifact(result, first)
    aggregate = json.loads((first / "manifest.json").read_text())["aggregate_sha256"]
    authorization = tmp_path / "viability.json"
    authorization.write_text(
        json.dumps(
            _authorization_receipt(authorized=True, aggregate_sha256=aggregate)
        )
    )
    config_path = tmp_path / "repeat-config.json"
    _write_cli_config(config_path, output_root=str(tmp_path / "repeat-output"))
    monkeypatch.setattr(
        script,
        "_run_construction",
        lambda args, config: pytest.fail("noncanonical first artifact authorized repeat"),
    )
    monkeypatch.setattr(
        script,
        "validate_multiresolution_viability_receipt",
        lambda payload, artifact: payload,
    )
    with pytest.raises(ValueError, match="canonical pass-one"):
        script.main(
            [
                "--config", str(config_path),
                "--first-artifact", str(first),
                "--first-viability-receipt", str(authorization),
            ]
        )


def test_cli_repeat_comparison_mode_never_constructs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    authorization = tmp_path / "viability.json"
    authorization.write_text(json.dumps(_authorization_receipt(authorized=True)))
    first = tmp_path / "first"
    first.mkdir()
    (first / "manifest.json").write_text(
        json.dumps({"aggregate_sha256": "a" * 64})
    )
    output = tmp_path / "repeatability.json"
    monkeypatch.setattr(
        script,
        "construct_multiresolution",
        lambda config: pytest.fail("comparison mode constructed another artifact"),
    )
    monkeypatch.setattr(
        script,
        "load_multiresolution_artifact",
        lambda path: type("Loaded", (), {"config": MultiResolutionConfig()})(),
    )
    monkeypatch.setattr(
        script,
        "validate_multiresolution_viability_receipt",
        lambda payload, artifact: payload,
    )
    monkeypatch.setattr(
        script,
        "compare_multiresolution_artifacts",
        lambda first, second: {
            "status": "passed",
            "first_aggregate_sha256": "a" * 64,
        },
    )
    assert script.main(
        [
            "--first-artifact", str(first),
            "--first-viability-receipt", str(authorization),
            "--repeat-artifact", str(tmp_path / "repeat"),
            "--repeatability-receipt", str(output),
        ]
    ) == 0
    assert json.loads(output.read_text())["status"] == "passed"


def test_cli_refuses_unbound_authorization_before_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    authorization = tmp_path / "viability.json"
    authorization.write_text(json.dumps(_authorization_receipt(authorized=True)))
    first = tmp_path / "first"
    first.mkdir()
    (first / "manifest.json").write_text(
        json.dumps({"aggregate_sha256": "b" * 64})
    )
    monkeypatch.setattr(
        script,
        "compare_multiresolution_artifacts",
        lambda first, second: pytest.fail("unbound authorization compared repeats"),
    )
    monkeypatch.setattr(
        script,
        "load_multiresolution_artifact",
        lambda path: type("Loaded", (), {"config": MultiResolutionConfig()})(),
    )
    monkeypatch.setattr(
        script,
        "validate_multiresolution_viability_receipt",
        lambda payload, artifact: payload,
    )
    with pytest.raises(ValueError, match="bind"):
        script.main(
            [
                "--first-artifact", str(first),
                "--first-viability-receipt", str(authorization),
                "--repeat-artifact", str(tmp_path / "repeat"),
                "--repeatability-receipt", str(tmp_path / "repeatability.json"),
            ]
        )


def test_cli_refuses_repeat_comparison_without_viability_and_cost_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_multiresolution_profiles as script

    authorization = tmp_path / "viability.json"
    authorization.write_text(json.dumps(_authorization_receipt(authorized=False)))
    output = tmp_path / "repeatability.json"
    monkeypatch.setattr(
        script,
        "compare_multiresolution_artifacts",
        lambda first, second: pytest.fail("unauthorized repeat was compared"),
    )
    monkeypatch.setattr(
        script,
        "validate_multiresolution_viability_receipt",
        lambda payload, artifact: payload,
    )
    with pytest.raises(ValueError, match="authorize"):
        script.main(
            [
                "--first-artifact", str(tmp_path / "first"),
                "--first-viability-receipt", str(authorization),
                "--repeat-artifact", str(tmp_path / "repeat"),
                "--repeatability-receipt", str(output),
            ]
        )
    assert not output.exists()
