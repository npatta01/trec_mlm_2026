"""Qrel-free construction and immutable artifacts for issue-18 profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.profile_experiment import (
    _profile_to_json,
    _recompute_profile_centroid,
    _validate_complete_rankings,
    load_profile_artifacts,
)
from trec_mlm.profile_retrieval import (
    CandidateProfile,
    ProfileRecord,
    aggregate_cluster_labels,
    canonical_kmeans,
    reciprocal_rank_fusion,
    score_profiles,
)
from trec_mlm.runio import rankings_from_run, write_trec_run


ARTIFACT_VERSION = 1
LOCKED_SOURCE_COMMIT = "2db10261bac607d7f8c67c0aa0c01728484a42f9"
LOCKED_RESOLUTIONS = (5, 10, 20)
EVALUATED_CONSTRUCTION_PATH = (
    "artifacts/candidate_semantic_profiles/run1_record_id_v2_auditfinal"
)
EVALUATED_CONSTRUCTION_AGGREGATE_SHA256 = (
    "04d6be23d2f9046ed93eb1c01bb5ad56545bb51a9470d408b96d3241b1ee54f4"
)
EXPECTED_SYSTEMS = (
    "c0_max_k10",
    "c0_max_k10_dense_equal",
    "a1_top2_k10",
    "a1_top2_k10_dense_equal",
    "r1_multires_max",
    "r1_multires_max_dense_equal",
    "x1_multires_top2",
    "x1_multires_top2_dense_equal",
)
CONTROL_NAMES = ("query_knn", "contextual_e5")
PASS_ONE_EXECUTION_CODE_COMMIT = "bf6bb6689f1b589453fe1eb8905291317ef71241"
PASS_ONE_COMMAND = (
    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
    ".venv/bin/python scripts/run_multiresolution_profiles.py --config "
    "configs/trec_2026_multiresolution_profiles.json --viability-receipt "
    "reports/trec_2026_issue18_pr17head_viability.json"
)
PASS_ONE_CONSTRUCTION_COMPLETED_AT = "2026-07-15T12:41:18.692102441-04:00"
PASS_ONE_DEPENDENCY_VERSIONS = {
    "python": "3.13.9",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
}
PASS_ONE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
PASS_ONE_ELAPSED_SECONDS = 428.64752057695296
NEW_PROFILE_SYSTEMS = (
    "a1_top2_k10",
    "r1_multires_max",
    "x1_multires_top2",
)

VIABILITY_THRESHOLDS: dict[str, tuple[str, float | int]] = {
    "candidate_occupied_cluster_retention": (">=", 0.95),
    "largest_cluster_share": ("<=", 0.80),
    "median_normalized_occupancy_entropy": (">=", 0.65),
    "minimum_normalized_occupancy_entropy": (">=", 0.35),
    "singleton_cluster_share": ("<=", 0.25),
    "exact_duplicate_centroids": ("==", 0),
    "within_candidate_near_duplicate_centroid_share": ("<=", 0.01),
    "median_centroid_separation": (">=", 0.005),
    "unique_top10_rankings": (">=", 258),
    "maximum_top1_candidate_share": ("<=", 0.10),
    "top10_candidate_coverage": (">=", 90),
    "maximum_candidate_top10_query_share": ("<=", 0.50),
    "incremental_top10_difference_query_share": (">=", 0.10),
    "incremental_mean_top10_jaccard": ("<", 0.95),
    "incremental_unique_top10_retention": (">=", 0.90),
    "incremental_candidate_coverage_retention": (">=", 0.90),
}

VIABILITY_INTEGRITY_CHECKS = frozenset(
    {
        "source_binding",
        "query_identities",
        "candidate_identities",
        "official_322_by_150_coverage",
        "complete_rankings",
        "component_score_schema",
        "system_score_schema",
        "contributor_schema",
        "finite_scores",
        "ranking_score_consistency",
        "normalized_centroids",
    }
)
VIABILITY_COMPONENT_SYSTEMS = frozenset(
    f"{label}_k{resolution}"
    for resolution in LOCKED_RESOLUTIONS
    for label in ("max", "top2")
)
VIABILITY_DIAGNOSTIC_SYSTEMS = frozenset(
    {*VIABILITY_COMPONENT_SYSTEMS, *EXPECTED_SYSTEMS}
)


def _expected_artifact_files() -> set[str]:
    names = {
        "config.json",
        "source.json",
        "inputs.json",
        "records.json",
        "query_embeddings.npy",
        "record_embeddings.npy",
        "rankings/dense_reference.json",
        "validation.json",
    }
    for resolution in LOCKED_RESOLUTIONS:
        names.add(f"profiles/k{resolution}.json")
        for label in ("max", "top2"):
            key = f"{label}_k{resolution}"
            names.add(f"component_scores/{key}.npy")
            names.add(f"contributors/{key}.npy")
    for system in EXPECTED_SYSTEMS:
        names.update(
            {
                f"rankings/{system}.json",
                f"scores/{system}.npy",
                f"runs/{system}.trec",
            }
        )
    return names


@dataclass(frozen=True, slots=True)
class MultiResolutionConfig:
    construction_path: str = EVALUATED_CONSTRUCTION_PATH
    output_root: str = "artifacts/multiresolution_profiles/issue18_pr17head"
    resolutions: tuple[int, ...] = LOCKED_RESOLUTIONS
    top_count: int = 2
    seed: int = 2026
    rrf_k: int = 60
    source_commit: str = LOCKED_SOURCE_COMMIT
    source_aggregate_sha256: str = EVALUATED_CONSTRUCTION_AGGREGATE_SHA256


@dataclass(frozen=True, slots=True)
class MultiResolutionResult:
    config: MultiResolutionConfig
    source_aggregate_sha256: str
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    records: list[dict[str, Any]]
    query_embeddings: NDArray[np.float32]
    record_embeddings: NDArray[np.float32]
    profiles: dict[str, dict[str, Any]]
    source_c0_profiles: dict[str, Any]
    dense_rankings: dict[str, list[str]]
    component_scores: dict[str, NDArray[np.float32]]
    contributing_centroid_ids: dict[str, NDArray[np.int32]]
    rankings: dict[str, dict[str, list[str]]]
    system_scores: dict[str, NDArray[np.float32]]


def _validate_config(config: MultiResolutionConfig) -> None:
    if not isinstance(config, MultiResolutionConfig):
        raise TypeError("config must be MultiResolutionConfig")
    if not config.construction_path or not config.output_root:
        raise ValueError("construction_path and output_root must be non-empty")
    if (
        config.construction_path != EVALUATED_CONSTRUCTION_PATH
        or config.source_aggregate_sha256
        != EVALUATED_CONSTRUCTION_AGGREGATE_SHA256
    ):
        raise ValueError("construction must use the evaluated issue-12 source")
    if config.resolutions != LOCKED_RESOLUTIONS:
        raise ValueError(f"resolutions must be {LOCKED_RESOLUTIONS!r}")
    if config.top_count != 2:
        raise ValueError("top_count must be 2")
    if config.seed != 2026:
        raise ValueError("seed must be 2026")
    if config.rrf_k != 60:
        raise ValueError("rrf_k must be 60")
    if config.source_commit != LOCKED_SOURCE_COMMIT:
        raise ValueError("source_commit differs from the locked issue-18 protocol")
    if (
        not isinstance(config.source_aggregate_sha256, str)
        or len(config.source_aggregate_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config.source_aggregate_sha256)
    ):
        raise ValueError("source_aggregate_sha256 must be a lowercase SHA-256")


def _length_framed_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid artifact JSON: {path.name}") from error


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _json_native_profile_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize tuple containers without weakening exact JSON comparison."""

    def require_string_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("profile payload mappings must use string keys")
            for nested in value.values():
                require_string_keys(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                require_string_keys(nested)

    if not isinstance(payload, Mapping):
        raise TypeError("profile payload must be a mapping")
    require_string_keys(payload)
    try:
        normalized = json.loads(
            json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("profile payload must contain finite JSON values") from error
    if not isinstance(normalized, dict):
        raise AssertionError("JSON-normalized profile payload must remain an object")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_hash(files: Mapping[str, Mapping[str, Any]]) -> str:
    entries = [
        [name, int(files[name]["byte_size"]), str(files[name]["sha256"])]
        for name in sorted(files)
    ]
    return hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _profiles_from_payload(
    payload: Mapping[str, Any], candidate_ids: Sequence[str]
) -> dict[str, CandidateProfile | None]:
    profiles: dict[str, CandidateProfile | None] = {}
    if set(payload) != set(candidate_ids):
        raise ValueError("profile candidate inventory changed")
    for candidate_id in candidate_ids:
        item = payload[candidate_id]
        if item.get("available") is False:
            profiles[candidate_id] = None
            continue
        profiles[candidate_id] = CandidateProfile(
            candidate_id=candidate_id,
            requested_clusters=int(item["requested_clusters"]),
            centroids=np.asarray(item["centroids"], dtype=np.float32),
            clusters=(),
            converged=bool(item.get("converged", True)),
            iteration_count=int(item.get("iteration_count", 1)),
        )
    return profiles


def _freeze_systems(
    *,
    query_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    dense_rankings: Mapping[str, Sequence[str]],
    profiles: Mapping[str, Mapping[str, Any]],
    query_embeddings: NDArray[np.float32],
    top_count: int,
    rrf_k: int,
) -> tuple[
    dict[str, NDArray[np.float32]],
    dict[str, NDArray[np.int32]],
    dict[str, dict[str, list[str]]],
    dict[str, NDArray[np.float32]],
]:
    component_scores: dict[str, NDArray[np.float32]] = {}
    contributors: dict[str, NDArray[np.int32]] = {}
    component_rankings: dict[str, dict[str, list[str]]] = {}
    unavailable = frozenset(
        candidate_id
        for candidate_id in candidate_ids
        if profiles[str(LOCKED_RESOLUTIONS[0])][candidate_id].get("available") is False
    )
    for resolution in LOCKED_RESOLUTIONS:
        candidates = _profiles_from_payload(profiles[str(resolution)], candidate_ids)
        for label, count in (("max", 1), ("top2", top_count)):
            scored = score_profiles(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                query_embeddings=query_embeddings,
                profiles=candidates,
                top_n=count,
            )
            key = f"{label}_k{resolution}"
            component_scores[key] = scored.scores
            contributors[key] = scored.contributing_centroid_ids
            component_rankings[key] = scored.rankings
    base: dict[str, tuple[NDArray[np.float32], dict[str, list[str]]]] = {
        "c0_max_k10": (component_scores["max_k10"], component_rankings["max_k10"]),
        "a1_top2_k10": (component_scores["top2_k10"], component_rankings["top2_k10"]),
    }
    for system, prefix in (
        ("r1_multires_max", "max"),
        ("x1_multires_top2", "top2"),
    ):
        fused = reciprocal_rank_fusion(
            query_ids=query_ids,
            candidate_ids=candidate_ids,
            rankings_by_signal={
                f"k{resolution}": component_rankings[f"{prefix}_k{resolution}"]
                for resolution in LOCKED_RESOLUTIONS
            },
            weights={f"k{resolution}": 1.0 for resolution in LOCKED_RESOLUTIONS},
            unavailable_profile_ids=unavailable,
            k=rrf_k,
        )
        base[system] = (fused.scores, fused.rankings)
    rankings: dict[str, dict[str, list[str]]] = {}
    system_scores: dict[str, NDArray[np.float32]] = {}
    for system, (scores, ranking) in base.items():
        rankings[system] = ranking
        system_scores[system] = scores
        fused = reciprocal_rank_fusion(
            query_ids=query_ids,
            candidate_ids=candidate_ids,
            rankings_by_signal={"dense": dense_rankings, "profile": ranking},
            weights={"dense": 1.0, "profile": 1.0},
            unavailable_profile_ids=unavailable,
            k=rrf_k,
        )
        dense_name = f"{system}_dense_equal"
        rankings[dense_name] = fused.rankings
        system_scores[dense_name] = fused.scores
    if set(rankings) != set(EXPECTED_SYSTEMS):
        raise AssertionError("frozen system inventory changed")
    return component_scores, contributors, rankings, system_scores


def _assemble_result_for_test(
    *,
    config: MultiResolutionConfig,
    query_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    records: list[dict[str, Any]],
    query_embeddings: NDArray[np.float32],
    record_embeddings: NDArray[np.float32],
    profiles: dict[str, dict[str, Any]],
    dense_rankings: dict[str, list[str]],
) -> MultiResolutionResult:
    """Assemble a small already-clustered payload for artifact contract tests."""

    _validate_config(config)
    component_scores, contributors, rankings, system_scores = _freeze_systems(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        dense_rankings=dense_rankings,
        profiles=profiles,
        query_embeddings=query_embeddings,
        top_count=config.top_count,
        rrf_k=config.rrf_k,
    )
    return MultiResolutionResult(
        config=config,
        source_aggregate_sha256=config.source_aggregate_sha256,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        records=records,
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        record_embeddings=np.asarray(record_embeddings, dtype=np.float32),
        profiles=profiles,
        source_c0_profiles=profiles["10"],
        dense_rankings={key: list(value) for key, value in dense_rankings.items()},
        component_scores=component_scores,
        contributing_centroid_ids=contributors,
        rankings=rankings,
        system_scores=system_scores,
    )


def construct_multiresolution(config: MultiResolutionConfig) -> MultiResolutionResult:
    """Reconstruct every issue-18 profile and freeze all preregistered rankings."""

    _validate_config(config)
    source = load_profile_artifacts(config.construction_path)
    observed_aggregate = source.manifest.get("aggregate_sha256")
    if observed_aggregate != config.source_aggregate_sha256:
        raise ValueError("issue-12 source aggregate differs from the frozen binding")
    root = source.root
    records = _read_json(root / "records.json")
    source_profiles = _read_json(root / "profiles.json")
    query_embeddings = np.load(root / "query_embeddings.npy", allow_pickle=False)
    record_embeddings = np.load(root / "record_embeddings.npy", allow_pickle=False)
    label_question_embeddings = np.load(
        root / "label_question_embeddings.npy", allow_pickle=False
    )
    label_embeddings = np.load(root / "label_embeddings.npy", allow_pickle=False)
    inputs = _read_json(root / "inputs.json")
    label_question_ids = tuple(inputs["label_question_ids"])
    label_positions = {value: index for index, value in enumerate(label_question_ids)}
    question_label_scores = np.asarray(
        label_question_embeddings @ label_embeddings.T, dtype=np.float32
    )
    profile_records = [
        ProfileRecord(
            record_id=item["record_id"],
            source_query_id=item["source_query_id"],
            candidate_id=item["candidate_id"],
            question=item["question"],
            response=item["response"],
        )
        for item in records
    ]
    record_lookup = {item.record_id: item for item in profile_records}
    profiles: dict[str, dict[str, Any]] = {}
    for resolution in config.resolutions:
        resolution_payload: dict[str, Any] = {}
        cursor = 0
        for candidate_id in source.candidate_ids:
            candidate_records = tuple(
                item for item in profile_records if item.candidate_id == candidate_id
            )
            count = len(candidate_records)
            if not count:
                resolution_payload[candidate_id] = _json_native_profile_payload(
                    {
                        "available": False,
                        "audit": source_profiles[candidate_id]["audit"],
                        "clusters": [],
                    }
                )
                continue
            profile = canonical_kmeans(
                candidate_id,
                candidate_records,
                record_embeddings[cursor : cursor + count],
                requested_clusters=min(resolution, count),
            )
            cursor += count
            annotated = []
            for cluster in profile.clusters:
                annotation = aggregate_cluster_labels(
                    member_question_indices=[
                        label_positions[record_lookup[item].source_query_id]
                        for item in cluster.member_record_ids
                    ],
                    question_label_scores=question_label_scores,
                    label_ids=source.candidate_ids,
                )
                annotated.append(replace(cluster, annotation=annotation))
            profile = replace(profile, clusters=tuple(annotated))
            resolution_payload[candidate_id] = _json_native_profile_payload(
                _profile_to_json(profile, source_profiles[candidate_id]["audit"])
            )
        if cursor != len(records):
            raise ValueError("record embedding inventory was not exactly consumed")
        profiles[str(resolution)] = resolution_payload
    if profiles["10"] != source_profiles:
        raise ValueError("C0 reconstruction differs from validated issue-12 profiles")
    dense_rankings = source.source_rankings["dense_reference"]
    component_scores, contributors, rankings, system_scores = _freeze_systems(
        query_ids=source.query_ids,
        candidate_ids=source.candidate_ids,
        dense_rankings=dense_rankings,
        profiles=profiles,
        query_embeddings=query_embeddings,
        top_count=config.top_count,
        rrf_k=config.rrf_k,
    )
    if not np.array_equal(component_scores["max_k10"], source.profile_scores):
        raise ValueError("C0 score reconstruction differs from issue-12")
    if rankings["c0_max_k10"] != source.rankings["profile_only"]:
        raise ValueError("C0 ranking reconstruction differs from issue-12")
    return MultiResolutionResult(
        config=config,
        source_aggregate_sha256=observed_aggregate,
        query_ids=source.query_ids,
        candidate_ids=source.candidate_ids,
        records=records,
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        record_embeddings=np.asarray(record_embeddings, dtype=np.float32),
        profiles=profiles,
        source_c0_profiles=source_profiles,
        dense_rankings=dense_rankings,
        component_scores=component_scores,
        contributing_centroid_ids=contributors,
        rankings=rankings,
        system_scores=system_scores,
    )


def write_multiresolution_artifact(
    result: MultiResolutionResult, output: str | Path
) -> dict[str, str]:
    """Atomically write one complete multi-resolution construction artifact."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_json(
            staging / "config.json",
            {
                "artifact_type": "issue18_multiresolution_profiles",
                "version": ARTIFACT_VERSION,
                "qrel_free": True,
                **asdict(result.config),
            },
        )
        _write_json(
            staging / "source.json",
            {
                "source_commit": result.config.source_commit,
                "construction_aggregate_sha256": result.source_aggregate_sha256,
            },
        )
        _write_json(
            staging / "inputs.json",
            {"query_ids": result.query_ids, "candidate_ids": result.candidate_ids},
        )
        _write_json(staging / "records.json", result.records)
        np.save(staging / "query_embeddings.npy", result.query_embeddings)
        np.save(staging / "record_embeddings.npy", result.record_embeddings)
        _write_json(staging / "rankings" / "dense_reference.json", result.dense_rankings)
        for resolution, payload in result.profiles.items():
            _write_json(staging / "profiles" / f"k{resolution}.json", payload)
        for key, scores in result.component_scores.items():
            (staging / "component_scores").mkdir(parents=True, exist_ok=True)
            np.save(staging / "component_scores" / f"{key}.npy", scores)
            (staging / "contributors").mkdir(parents=True, exist_ok=True)
            np.save(
                staging / "contributors" / f"{key}.npy",
                result.contributing_centroid_ids[key],
            )
        for system in EXPECTED_SYSTEMS:
            _write_json(staging / "rankings" / f"{system}.json", result.rankings[system])
            (staging / "scores").mkdir(parents=True, exist_ok=True)
            np.save(staging / "scores" / f"{system}.npy", result.system_scores[system])
            write_trec_run(
                staging / "runs" / f"{system}.trec",
                result.rankings[system],
                run_id=system,
            )
        _write_json(
            staging / "validation.json",
            {
                "status": "passed",
                "qrel_free": True,
                "systems": EXPECTED_SYSTEMS,
                "integrity_checks": {
                    "source_binding": True,
                    "inventory": True,
                    "semantic_reconstruction": True,
                    "complete_rankings": True,
                },
            },
        )
        names = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        )
        entries = {
            name: {
                "byte_size": (staging / name).stat().st_size,
                "sha256": _sha256(staging / name),
            }
            for name in names
        }
        _write_json(
            staging / "manifest.json",
            {
                "artifact_type": "issue18_multiresolution_profile_manifest",
                "version": ARTIFACT_VERSION,
                "qrel_free": True,
                "files": entries,
                "aggregate_sha256": _aggregate_hash(entries),
            },
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        path.relative_to(destination).as_posix(): str(path)
        for path in destination.rglob("*")
        if path.is_file()
    }


def _validate_profile_semantics(
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    candidate_ids: tuple[str, ...],
    records: Sequence[Mapping[str, Any]],
    record_embeddings: NDArray[np.float32],
) -> None:
    positions = {item["record_id"]: index for index, item in enumerate(records)}
    if len(positions) != len(records):
        raise ValueError("record identities are not unique")
    for resolution in LOCKED_RESOLUTIONS:
        payload = profiles[str(resolution)]
        if set(payload) != set(candidate_ids):
            raise ValueError("profile candidate inventory changed")
        for candidate_id in candidate_ids:
            expected = {
                item["record_id"] for item in records if item["candidate_id"] == candidate_id
            }
            item = payload[candidate_id]
            if item.get("available") is False:
                if expected:
                    raise ValueError("available record inventory lost its profile")
                continue
            centroids = np.asarray(item["centroids"], dtype=np.float32)
            requested = min(resolution, len(expected))
            if item["requested_clusters"] != requested or len(centroids) != item["actual_clusters"]:
                raise ValueError("requested or occupied cluster count changed")
            if not np.isfinite(centroids).all() or not np.allclose(
                np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5, rtol=0.0
            ):
                raise ValueError("centroids are not finite normalized vectors")
            observed: list[str] = []
            for index, cluster in enumerate(item["clusters"]):
                members = list(cluster["member_record_ids"])
                observed.extend(members)
                if (
                    cluster["cluster_id"] != index
                    or cluster["support_count"] != len(members)
                    or members != sorted(members)
                    or cluster["membership_sha256"] != _length_framed_hash(members)
                ):
                    raise ValueError("cluster identities or support changed")
                recomputed = _recompute_profile_centroid(
                    record_embeddings[np.asarray([positions[value] for value in members])]
                )
                if not np.array_equal(recomputed, centroids[index]):
                    raise ValueError("centroid reconstruction drifted")
            if len(observed) != len(set(observed)) or set(observed) != expected:
                raise ValueError("cluster memberships do not exactly cover records")


def load_multiresolution_artifact(path: str | Path) -> MultiResolutionResult:
    """Load only after inventory, hashes, source, schema, and semantics validate."""

    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    expected_manifest_keys = {
        "artifact_type",
        "version",
        "qrel_free",
        "files",
        "aggregate_sha256",
    }
    if (
        set(manifest) != expected_manifest_keys
        or
        manifest.get("artifact_type") != "issue18_multiresolution_profile_manifest"
        or manifest.get("version") != ARTIFACT_VERSION
        or manifest.get("qrel_free") is not True
    ):
        raise ValueError("multi-resolution manifest schema changed")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or manifest.get("aggregate_sha256") != _aggregate_hash(entries):
        raise ValueError("manifest aggregate hash changed")
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    }
    if actual != set(entries) or actual != _expected_artifact_files():
        raise ValueError("artifact file inventory changed")
    for name, expected in entries.items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"byte_size", "sha256"}
            or (root / name).stat().st_size != expected["byte_size"]
            or _sha256(root / name) != expected["sha256"]
        ):
            raise ValueError(f"artifact hash mismatch: {name}")
    config_payload = _read_json(root / "config.json")
    expected_config_keys = {
        "artifact_type",
        "version",
        "qrel_free",
        *(field.name for field in fields(MultiResolutionConfig)),
    }
    if (
        set(config_payload) != expected_config_keys
        or
        config_payload.get("artifact_type") != "issue18_multiresolution_profiles"
        or config_payload.get("version") != ARTIFACT_VERSION
        or config_payload.get("qrel_free") is not True
    ):
        raise ValueError("artifact config schema changed")
    config_values = {
        field.name: config_payload[field.name] for field in fields(MultiResolutionConfig)
    }
    config_values["resolutions"] = tuple(config_values["resolutions"])
    config = MultiResolutionConfig(**config_values)
    _validate_config(config)
    source = _read_json(root / "source.json")
    if source != {
        "source_commit": config.source_commit,
        "construction_aggregate_sha256": config.source_aggregate_sha256,
    }:
        raise ValueError("source binding changed")
    construction_path = Path(config.construction_path)
    if construction_path.exists():
        construction = load_profile_artifacts(construction_path)
        if construction.manifest.get("aggregate_sha256") != config.source_aggregate_sha256:
            raise ValueError("bound issue-12 source aggregate changed")
    validation = _read_json(root / "validation.json")
    if (
        set(validation) != {"status", "qrel_free", "systems", "integrity_checks"}
        or
        validation.get("status") != "passed"
        or validation.get("qrel_free") is not True
        or tuple(validation.get("systems", ())) != EXPECTED_SYSTEMS
        or validation.get("integrity_checks")
        != {
            "source_binding": True,
            "inventory": True,
            "semantic_reconstruction": True,
            "complete_rankings": True,
        }
    ):
        raise ValueError("validation receipt changed")
    inputs = _read_json(root / "inputs.json")
    if set(inputs) != {"query_ids", "candidate_ids"}:
        raise ValueError("artifact input schema changed")
    query_ids = tuple(inputs["query_ids"])
    candidate_ids = tuple(inputs["candidate_ids"])
    if not query_ids or not candidate_ids or len(set(query_ids)) != len(query_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("input identities are invalid")
    records = _read_json(root / "records.json")
    query_embeddings = np.load(root / "query_embeddings.npy", allow_pickle=False)
    record_embeddings = np.load(root / "record_embeddings.npy", allow_pickle=False)
    if (
        query_embeddings.dtype.str != "<f4"
        or record_embeddings.dtype.str != "<f4"
        or query_embeddings.shape[0] != len(query_ids)
        or record_embeddings.shape[0] != len(records)
        or query_embeddings.ndim != 2
        or record_embeddings.ndim != 2
        or query_embeddings.shape[1] != record_embeddings.shape[1]
    ):
        raise ValueError("embedding schema or inventory changed")
    profiles = {
        str(resolution): _read_json(root / "profiles" / f"k{resolution}.json")
        for resolution in LOCKED_RESOLUTIONS
    }
    expected = construct_multiresolution(config)
    source_inputs_match = (
        expected.source_aggregate_sha256 == config.source_aggregate_sha256
        and expected.query_ids == query_ids
        and expected.candidate_ids == candidate_ids
        and expected.records == records
        and np.array_equal(expected.query_embeddings, query_embeddings)
        and np.array_equal(expected.record_embeddings, record_embeddings)
        and expected.dense_rankings == _read_json(
            root / "rankings" / "dense_reference.json"
        )
    )
    if not source_inputs_match:
        raise ValueError("embedded inputs differ from the bound issue-12 source")
    if expected.profiles != profiles:
        raise ValueError("canonical profile reconstruction differs from bound source")
    _validate_profile_semantics(
        profiles=profiles,
        candidate_ids=candidate_ids,
        records=records,
        record_embeddings=record_embeddings,
    )
    dense_rankings = _read_json(root / "rankings" / "dense_reference.json")
    _validate_complete_rankings(dense_rankings, query_ids, candidate_ids)
    component_scores: dict[str, NDArray[np.float32]] = {}
    contributors: dict[str, NDArray[np.int32]] = {}
    for resolution in LOCKED_RESOLUTIONS:
        for label, width in (("max", 1), ("top2", config.top_count)):
            key = f"{label}_k{resolution}"
            component_scores[key] = np.load(
                root / "component_scores" / f"{key}.npy", allow_pickle=False
            )
            contributors[key] = np.load(
                root / "contributors" / f"{key}.npy", allow_pickle=False
            )
            if (
                component_scores[key].dtype.str != "<f4"
                or component_scores[key].shape != (len(query_ids), len(candidate_ids))
                or not np.isfinite(component_scores[key]).all()
                or contributors[key].dtype.str != "<i4"
                or contributors[key].shape != (len(query_ids), len(candidate_ids), width)
            ):
                raise ValueError("component score or contributor schema changed")
    reconstructed_components, reconstructed_contributors, reconstructed_rankings, reconstructed_system_scores = _freeze_systems(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        dense_rankings=dense_rankings,
        profiles=profiles,
        query_embeddings=query_embeddings,
        top_count=config.top_count,
        rrf_k=config.rrf_k,
    )
    for key in reconstructed_components:
        if (
            not np.array_equal(component_scores[key], expected.component_scores[key])
            or not np.array_equal(
                contributors[key], expected.contributing_centroid_ids[key]
            )
            or not np.array_equal(
                component_scores[key], reconstructed_components[key]
            )
            or not np.array_equal(
                contributors[key], reconstructed_contributors[key]
            )
        ):
            raise ValueError(f"component semantic reconstruction drift: {key}")
    rankings: dict[str, dict[str, list[str]]] = {}
    system_scores: dict[str, NDArray[np.float32]] = {}
    for system in EXPECTED_SYSTEMS:
        rankings[system] = _read_json(root / "rankings" / f"{system}.json")
        system_scores[system] = np.load(root / "scores" / f"{system}.npy", allow_pickle=False)
        _validate_complete_rankings(rankings[system], query_ids, candidate_ids)
        if rankings_from_run(root / "runs" / f"{system}.trec") != rankings[system]:
            raise ValueError(f"TREC run disagrees for {system}")
        if (
            system_scores[system].dtype.str != "<f4"
            or system_scores[system].shape != (len(query_ids), len(candidate_ids))
            or not np.array_equal(system_scores[system], reconstructed_system_scores[system])
            or rankings[system] != reconstructed_rankings[system]
            or not np.array_equal(system_scores[system], expected.system_scores[system])
            or rankings[system] != expected.rankings[system]
        ):
            raise ValueError(f"system semantic reconstruction drift: {system}")
    return MultiResolutionResult(
        config=config,
        source_aggregate_sha256=config.source_aggregate_sha256,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        records=records,
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        record_embeddings=np.asarray(record_embeddings, dtype=np.float32),
        profiles=profiles,
        source_c0_profiles=profiles["10"],
        dense_rankings=dense_rankings,
        component_scores=component_scores,
        contributing_centroid_ids=contributors,
        rankings=rankings,
        system_scores=system_scores,
    )


def _passes(value: float | int, operator: str, threshold: float | int) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    if operator == "==":
        return value == threshold
    raise AssertionError(f"unsupported threshold operator: {operator}")


def _finite_receipt_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _compare_top10_rankings(
    ranking: Mapping[str, Sequence[str]],
    control: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Compute deterministic qrel-free top-10 overlap and displacement metrics."""

    if set(ranking) != set(control) or not ranking:
        raise ValueError("control ranking query inventory differs from frozen systems")
    jaccards: list[float] = []
    overlaps: list[int] = []
    difference_count = 0
    displacement_total = 0
    common_count = 0
    for query_id in ranking:
        observed = tuple(ranking[query_id][:10])
        baseline = tuple(control[query_id][:10])
        if len(observed) != len(set(observed)) or len(baseline) != len(set(baseline)):
            raise ValueError("top-10 control comparison contains duplicate candidates")
        observed_set = set(observed)
        baseline_set = set(baseline)
        common = observed_set & baseline_set
        union = observed_set | baseline_set
        jaccards.append(len(common) / len(union) if union else 1.0)
        overlaps.append(len(common))
        difference_count += int(observed != baseline)
        observed_positions = {candidate: index for index, candidate in enumerate(observed)}
        baseline_positions = {candidate: index for index, candidate in enumerate(baseline)}
        displacement_total += sum(
            abs(observed_positions[candidate] - baseline_positions[candidate])
            for candidate in common
        )
        common_count += len(common)
    query_count = len(ranking)
    return {
        "mean_top10_jaccard": statistics.fmean(jaccards),
        "top10_difference_query_share": difference_count / query_count,
        "mean_top10_overlap_count": statistics.fmean(overlaps),
        "mean_common_top10_absolute_rank_displacement": (
            displacement_total / common_count if common_count else 0.0
        ),
    }


def _control_comparisons(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    controls: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, dict[str, float]]]:
    if set(rankings) != set(EXPECTED_SYSTEMS) or set(controls) != set(CONTROL_NAMES):
        raise ValueError("frozen system or control ranking inventory changed")
    return {
        system: {
            control: _compare_top10_rankings(ranking, controls[control])
            for control in CONTROL_NAMES
        }
        for system, ranking in rankings.items()
    }


def _validate_control_ranking_inventory(
    ranking: Mapping[str, Sequence[str]], result: MultiResolutionResult
) -> None:
    if set(ranking) != set(result.query_ids):
        raise ValueError("control query inventory differs from frozen artifact")
    expected_candidates = set(result.candidate_ids)
    if any(
        len(row) != len(result.candidate_ids) or set(row) != expected_candidates
        for row in ranking.values()
    ):
        raise ValueError("control candidate inventory differs from frozen artifact")


def _load_control_rankings(
    result: MultiResolutionResult, artifact_root: Path
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Sequence[str]]],
]:
    """Load only the two strictly bound qrel-free control rankings."""

    source = load_profile_artifacts(result.config.construction_path)
    if (
        source.manifest.get("aggregate_sha256")
        != result.config.source_aggregate_sha256
        or tuple(source.query_ids) != result.query_ids
        or tuple(source.candidate_ids) != result.candidate_ids
    ):
        raise ValueError("control source aggregate or inventory differs from frozen artifact")
    source_files = source.manifest.get("files")
    if not isinstance(source_files, Mapping):
        raise ValueError("control source manifest file inventory changed")
    artifact_manifest = _read_json(artifact_root / "manifest.json")
    artifact_files = artifact_manifest.get("files")
    if not isinstance(artifact_files, Mapping):
        raise ValueError("frozen artifact manifest file inventory changed")

    dense = source.source_rankings.get("dense_reference")
    contextual = source.source_rankings.get("contextual_e5_control")
    if not isinstance(dense, Mapping) or not isinstance(contextual, Mapping):
        raise ValueError("required control rankings are absent")
    if dict(dense) != result.dense_rankings:
        raise ValueError("query-kNN control differs from frozen artifact copy")
    controls = {"query_knn": dense, "contextual_e5": contextual}
    for ranking in controls.values():
        _validate_control_ranking_inventory(ranking, result)

    dense_relative = "rankings/dense_reference.json"
    contextual_relative = "rankings/contextual_e5_control.json"
    dense_path = artifact_root / dense_relative
    contextual_path = Path(source.root) / contextual_relative
    dense_entry = artifact_files.get(dense_relative)
    source_dense_entry = source_files.get(dense_relative)
    contextual_entry = source_files.get(contextual_relative)
    if any(
        not isinstance(entry, Mapping) or set(entry) < {"sha256"}
        for entry in (dense_entry, source_dense_entry, contextual_entry)
    ):
        raise ValueError("control ranking hash entries changed")
    dense_sha = _sha256(dense_path)
    contextual_sha = _sha256(contextual_path)
    if (
        dense_sha != dense_entry["sha256"]
        or dense_sha != source_dense_entry["sha256"]
        or contextual_sha != contextual_entry["sha256"]
    ):
        raise ValueError("control ranking hash mismatch")

    inventory = {
        "query_count": len(result.query_ids),
        "candidate_count": len(result.candidate_ids),
        "query_inventory_sha256": _length_framed_hash(result.query_ids),
        "candidate_inventory_sha256": _length_framed_hash(result.candidate_ids),
    }
    sources = {
        "query_knn": {
            "ranking_name": "dense_reference",
            "ranking_file": dense_relative,
            "ranking_sha256": dense_sha,
            "source_aggregate_sha256": result.config.source_aggregate_sha256,
            **inventory,
        },
        "contextual_e5": {
            "ranking_name": "contextual_e5_control",
            "ranking_file": contextual_relative,
            "ranking_sha256": contextual_sha,
            "source_aggregate_sha256": result.config.source_aggregate_sha256,
            **inventory,
        },
    }
    return sources, controls


def _execution_provenance(
    artifact_aggregate_sha256: str,
    *,
    receipt_generated_at: str | None = None,
) -> dict[str, Any]:
    generated = receipt_generated_at or datetime.now(timezone.utc).isoformat()
    return {
        "execution_code_commit": PASS_ONE_EXECUTION_CODE_COMMIT,
        "command": PASS_ONE_COMMAND,
        "thread_environment": dict(PASS_ONE_THREAD_ENVIRONMENT),
        "dependency_versions": dict(PASS_ONE_DEPENDENCY_VERSIONS),
        "construction_completed_at": PASS_ONE_CONSTRUCTION_COMPLETED_AT,
        "artifact_aggregate_sha256": artifact_aggregate_sha256,
        "receipt_generated_at": generated,
    }


def _viability_decision(
    failed_integrity: Sequence[str],
    failed_query_sensitivity: Sequence[str],
    failed_incremental: Sequence[str],
) -> str:
    if failed_integrity:
        return "reject-invalid"
    if failed_query_sensitivity:
        return "stop-query-sensitivity"
    if failed_incremental:
        return "stop-no-incremental-value"
    return "continue"


def validate_multiresolution_viability_receipt(
    payload: Mapping[str, Any],
    artifact_aggregate_sha256: str,
    artifact: MultiResolutionResult | str | Path | None = None,
) -> Mapping[str, Any]:
    """Require the complete, internally consistent viability receipt."""

    top_level_keys = {
        "artifact_type",
        "version",
        "qrel_free",
        "artifact_aggregate_sha256",
        "decision",
        "thresholds",
        "integrity_checks",
        "failed_thresholds",
        "failed_integrity_thresholds",
        "failed_query_sensitivity_thresholds",
        "failed_incremental_thresholds",
        "cluster_diagnostics",
        "ranking_diagnostics",
        "control_sources",
        "control_comparisons",
        "execution_provenance",
        "cost_evidence",
        "repeat_authorized",
    }
    if set(payload) != top_level_keys:
        raise ValueError("viability receipt schema is incomplete or changed")
    if payload.get("artifact_aggregate_sha256") != artifact_aggregate_sha256:
        raise ValueError("viability receipt does not bind the evaluated artifact")
    if (
        payload.get("artifact_type")
        != "issue18_multiresolution_viability_receipt"
        or payload.get("version") != 1
        or payload.get("qrel_free") is not True
    ):
        raise ValueError("viability receipt schema is incomplete or changed")

    integrity = payload.get("integrity_checks")
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != VIABILITY_INTEGRITY_CHECKS
        or not all(value is True for value in integrity.values())
    ):
        raise ValueError("viability receipt integrity checks are inconsistent")

    cluster_diagnostics = payload.get("cluster_diagnostics")
    cluster_fields = {"candidate_count", *tuple(VIABILITY_THRESHOLDS)[:8]}
    if (
        not isinstance(cluster_diagnostics, Mapping)
        or set(cluster_diagnostics)
        != {f"k{resolution}" for resolution in LOCKED_RESOLUTIONS}
        or any(
            not isinstance(values, Mapping)
            or set(values) != cluster_fields
            or any(not _finite_receipt_number(value) for value in values.values())
            for values in cluster_diagnostics.values()
        )
    ):
        raise ValueError("viability receipt cluster diagnostic schema changed")

    ranking_diagnostics = payload.get("ranking_diagnostics")
    base_ranking_fields = set(tuple(VIABILITY_THRESHOLDS)[8:12])
    incremental_diagnostic_fields = {
        "top10_difference_query_share",
        "mean_top10_jaccard",
        "unique_top10_retention",
        "candidate_coverage_retention",
    }
    if (
        not isinstance(ranking_diagnostics, Mapping)
        or set(ranking_diagnostics) != VIABILITY_DIAGNOSTIC_SYSTEMS
    ):
        raise ValueError("viability receipt ranking diagnostic schema changed")
    for system, values in ranking_diagnostics.items():
        expected_fields = base_ranking_fields | (
            incremental_diagnostic_fields if system in NEW_PROFILE_SYSTEMS else set()
        )
        if (
            not isinstance(values, Mapping)
            or set(values) != expected_fields
            or any(not _finite_receipt_number(value) for value in values.values())
        ):
            raise ValueError("viability receipt ranking diagnostic schema changed")

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(
        VIABILITY_THRESHOLDS
    ):
        raise ValueError("viability receipt threshold schema changed")
    incremental_fields = {
        "incremental_top10_difference_query_share": "top10_difference_query_share",
        "incremental_mean_top10_jaccard": "mean_top10_jaccard",
        "incremental_unique_top10_retention": "unique_top10_retention",
        "incremental_candidate_coverage_retention": "candidate_coverage_retention",
    }
    for index, (name, (operator, threshold)) in enumerate(
        VIABILITY_THRESHOLDS.items()
    ):
        item = thresholds[name]
        if not isinstance(item, Mapping) or set(item) != {
            "operator",
            "threshold",
            "values",
            "passed",
        }:
            raise ValueError("viability receipt threshold schema changed")
        if item["operator"] != operator or item["threshold"] != threshold:
            raise ValueError("viability receipt threshold contract is inconsistent")
        if index < 8:
            expected_values = {
                key: values[name] for key, values in cluster_diagnostics.items()
            }
        elif name in incremental_fields:
            field = incremental_fields[name]
            expected_values = {
                system: ranking_diagnostics[system][field]
                for system in NEW_PROFILE_SYSTEMS
            }
        else:
            expected_values = {
                system: values[name] for system, values in ranking_diagnostics.items()
            }
        values = item["values"]
        if (
            not isinstance(values, Mapping)
            or dict(values) != expected_values
            or any(not _finite_receipt_number(value) for value in values.values())
        ):
            raise ValueError("viability receipt thresholds and diagnostics are inconsistent")
        recomputed_pass = all(
            _passes(value, operator, threshold) for value in values.values()
        )
        if item["passed"] is not recomputed_pass:
            raise ValueError("viability receipt threshold pass state is inconsistent")

    cluster_threshold_names = tuple(VIABILITY_THRESHOLDS)[:8]
    query_sensitivity_names = tuple(VIABILITY_THRESHOLDS)[8:12]
    incremental_threshold_names = tuple(VIABILITY_THRESHOLDS)[12:]
    failed_integrity = [
        name for name in cluster_threshold_names if not thresholds[name]["passed"]
    ]
    failed_query_sensitivity = [
        name for name in query_sensitivity_names if not thresholds[name]["passed"]
    ]
    failed_incremental = [
        name for name in incremental_threshold_names if not thresholds[name]["passed"]
    ]
    if (
        payload.get("failed_integrity_thresholds") != failed_integrity
        or payload.get("failed_query_sensitivity_thresholds")
        != failed_query_sensitivity
        or payload.get("failed_incremental_thresholds") != failed_incremental
        or payload.get("failed_thresholds")
        != [*failed_integrity, *failed_query_sensitivity, *failed_incremental]
    ):
        raise ValueError("viability receipt failure lists are inconsistent")

    control_sources = payload.get("control_sources")
    source_fields = {
        "ranking_name",
        "ranking_file",
        "ranking_sha256",
        "source_aggregate_sha256",
        "query_count",
        "candidate_count",
        "query_inventory_sha256",
        "candidate_inventory_sha256",
    }
    expected_control_identities = {
        "query_knn": (
            "dense_reference",
            "rankings/dense_reference.json",
            "734d96fd42d9cbff940c9c29acb6efe6459a0452d9cadd24a6fa3074beb96fb8",
        ),
        "contextual_e5": (
            "contextual_e5_control",
            "rankings/contextual_e5_control.json",
            "a88a0454494098b683eb745bd9ec909b9f4266c94ac2a538b2b552d95754681f",
        ),
    }
    if not isinstance(control_sources, Mapping) or set(control_sources) != set(
        CONTROL_NAMES
    ):
        raise ValueError("viability receipt control source schema changed")
    for name, expected_identity in expected_control_identities.items():
        source = control_sources[name]
        if (
            not isinstance(source, Mapping)
            or set(source) != source_fields
            or (
                source["ranking_name"],
                source["ranking_file"],
                source["ranking_sha256"],
            )
            != expected_identity
            or source["source_aggregate_sha256"]
            != EVALUATED_CONSTRUCTION_AGGREGATE_SHA256
            or source["query_count"] != 322
            or source["candidate_count"] != 150
            or any(
                not isinstance(source[field], str) or len(source[field]) != 64
                for field in (
                    "query_inventory_sha256",
                    "candidate_inventory_sha256",
                )
            )
        ):
            raise ValueError("viability receipt control source identity changed")
    if (
        control_sources["query_knn"]["query_inventory_sha256"]
        != control_sources["contextual_e5"]["query_inventory_sha256"]
        or control_sources["query_knn"]["candidate_inventory_sha256"]
        != control_sources["contextual_e5"]["candidate_inventory_sha256"]
    ):
        raise ValueError("viability receipt control inventories differ")

    comparison_fields = {
        "mean_top10_jaccard",
        "top10_difference_query_share",
        "mean_top10_overlap_count",
        "mean_common_top10_absolute_rank_displacement",
    }
    control_comparisons = payload.get("control_comparisons")
    if (
        not isinstance(control_comparisons, Mapping)
        or set(control_comparisons) != set(EXPECTED_SYSTEMS)
        or any(
            not isinstance(values, Mapping)
            or set(values) != set(CONTROL_NAMES)
            or any(
                not isinstance(comparison, Mapping)
                or set(comparison) != comparison_fields
                or any(
                    not _finite_receipt_number(value)
                    for value in comparison.values()
                )
                for comparison in values.values()
            )
            for values in control_comparisons.values()
        )
    ):
        raise ValueError("viability receipt control comparison schema changed")

    if artifact is not None:
        result = (
            artifact
            if isinstance(artifact, MultiResolutionResult)
            else load_multiresolution_artifact(artifact)
        )
        if isinstance(artifact, MultiResolutionResult):
            raise ValueError("persisted artifact path is required to validate controls")
        _, controls = _load_control_rankings(result, Path(artifact))
        if control_comparisons != _control_comparisons(result.rankings, controls):
            raise ValueError("viability receipt control comparisons are inconsistent")

    provenance = payload.get("execution_provenance")
    provenance_fields = {
        "execution_code_commit",
        "command",
        "thread_environment",
        "dependency_versions",
        "construction_completed_at",
        "artifact_aggregate_sha256",
        "receipt_generated_at",
    }
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != provenance_fields
        or provenance["execution_code_commit"] != PASS_ONE_EXECUTION_CODE_COMMIT
        or provenance["command"] != PASS_ONE_COMMAND
        or provenance["thread_environment"] != PASS_ONE_THREAD_ENVIRONMENT
        or provenance["dependency_versions"] != PASS_ONE_DEPENDENCY_VERSIONS
        or provenance["construction_completed_at"]
        != PASS_ONE_CONSTRUCTION_COMPLETED_AT
        or provenance["artifact_aggregate_sha256"] != artifact_aggregate_sha256
    ):
        raise ValueError("viability receipt execution provenance changed")
    try:
        completed = datetime.fromisoformat(PASS_ONE_CONSTRUCTION_COMPLETED_AT)
        generated = datetime.fromisoformat(provenance["receipt_generated_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("viability receipt provenance timestamp is invalid") from error
    if generated.tzinfo is None or generated < completed:
        raise ValueError("viability receipt provenance timestamp is inconsistent")

    cost = payload.get("cost_evidence")
    cost_keys = {
        "issue12_construction_storage_bytes",
        "multiresolution_artifact_storage_bytes",
        "storage_ratio",
        "storage_at_most_four_times_issue12",
        "issue12_construction_elapsed_seconds",
        "multiresolution_construction_elapsed_seconds",
        "wall_time_ratio",
        "wall_time_at_most_four_times_issue12",
        "repeat_cost_authorized",
    }
    if not isinstance(cost, Mapping) or set(cost) != cost_keys:
        raise ValueError("viability receipt cost schema changed")
    numeric_cost_fields = cost_keys - {
        "storage_at_most_four_times_issue12",
        "wall_time_at_most_four_times_issue12",
        "repeat_cost_authorized",
    }
    if any(not _finite_receipt_number(cost[field]) for field in numeric_cost_fields):
        raise ValueError("viability receipt cost values are invalid")
    source_storage = cost["issue12_construction_storage_bytes"]
    artifact_storage = cost["multiresolution_artifact_storage_bytes"]
    source_elapsed = cost["issue12_construction_elapsed_seconds"]
    artifact_elapsed = cost["multiresolution_construction_elapsed_seconds"]
    cost_consistent = (
        source_storage > 0
        and artifact_storage >= 0
        and source_elapsed > 0
        and artifact_elapsed >= 0
        and cost["storage_ratio"] == artifact_storage / source_storage
        and cost["wall_time_ratio"] == artifact_elapsed / source_elapsed
        and cost["storage_at_most_four_times_issue12"]
        is (artifact_storage <= 4 * source_storage)
        and cost["wall_time_at_most_four_times_issue12"]
        is (artifact_elapsed <= 4 * source_elapsed)
        and cost["repeat_cost_authorized"]
        is (
            cost["storage_at_most_four_times_issue12"] is True
            and cost["wall_time_at_most_four_times_issue12"] is True
        )
    )
    if not cost_consistent:
        raise ValueError("viability receipt cost evidence is inconsistent")

    expected_decision = _viability_decision(
        failed_integrity, failed_query_sensitivity, failed_incremental
    )
    expected_repeat_authorized = (
        expected_decision == "continue" and cost["repeat_cost_authorized"] is True
    )
    if (
        payload.get("decision") != expected_decision
        or payload.get("repeat_authorized") is not expected_repeat_authorized
    ):
        raise ValueError("viability receipt decision is inconsistent")
    if expected_decision == "reject-invalid":
        raise ValueError("viability receipt does not authorize diagnostic evaluation")
    return payload


def _threshold_receipt(
    name: str, values: Mapping[str, float | int]
) -> dict[str, Any]:
    operator, threshold = VIABILITY_THRESHOLDS[name]
    return {
        "operator": operator,
        "threshold": threshold,
        "values": dict(values),
        "passed": all(_passes(value, operator, threshold) for value in values.values()),
    }


def _normalized_occupancy_entropy(support: Sequence[int]) -> float:
    if len(support) <= 1:
        return 1.0
    probabilities = np.asarray(support, dtype=np.float64)
    probabilities /= float(probabilities.sum())
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(support)))


def _cluster_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    retained: list[bool] = []
    largest_shares: list[float] = []
    entropies: list[float] = []
    singleton_count = 0
    cluster_count = 0
    exact_duplicates = 0
    near_duplicates = 0
    pair_count = 0
    separations: list[float] = []
    for item in payload.values():
        if item.get("available") is False:
            continue
        requested = int(item["requested_clusters"])
        actual = int(item["actual_clusters"])
        retained.append(actual >= 0.8 * requested)
        support = [int(cluster["support_count"]) for cluster in item["clusters"]]
        largest_shares.append(max(support) / sum(support))
        entropies.append(_normalized_occupancy_entropy(support))
        singleton_count += sum(value == 1 for value in support)
        cluster_count += len(support)
        centroids = np.asarray(item["centroids"], dtype=np.float64)
        for left in range(len(centroids)):
            for right in range(left + 1, len(centroids)):
                cosine = float(centroids[left] @ centroids[right])
                pair_count += 1
                exact_duplicates += int(np.array_equal(centroids[left], centroids[right]))
                near_duplicates += int(cosine >= 0.9999)
                separations.append(1.0 - cosine)
    return {
        "candidate_count": len(retained),
        "candidate_occupied_cluster_retention": (
            sum(retained) / len(retained) if retained else 0.0
        ),
        "largest_cluster_share": max(largest_shares, default=1.0),
        "median_normalized_occupancy_entropy": (
            statistics.median(entropies) if entropies else 0.0
        ),
        "minimum_normalized_occupancy_entropy": min(entropies, default=0.0),
        "singleton_cluster_share": singleton_count / cluster_count if cluster_count else 1.0,
        "exact_duplicate_centroids": exact_duplicates,
        "within_candidate_near_duplicate_centroid_share": (
            near_duplicates / pair_count if pair_count else 0.0
        ),
        "median_centroid_separation": (
            statistics.median(separations) if separations else 1.0
        ),
    }


def _top10_rows(ranking: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {query_id: tuple(values[:10]) for query_id, values in ranking.items()}


def _rank_scores(
    result: MultiResolutionResult, scores: NDArray[np.float32]
) -> dict[str, list[str]]:
    return {
        query_id: [
            result.candidate_ids[index]
            for index in sorted(
                range(len(result.candidate_ids)),
                key=lambda index: (-float(scores[row, index]), result.candidate_ids[index]),
            )
        ]
        for row, query_id in enumerate(result.query_ids)
    }


def _ranking_diagnostics(
    ranking: Mapping[str, Sequence[str]],
    baseline: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    rows = _top10_rows(ranking)
    top1_counts: dict[str, int] = {}
    top10_counts: dict[str, int] = {}
    for row in rows.values():
        if row:
            top1_counts[row[0]] = top1_counts.get(row[0], 0) + 1
        for candidate_id in row:
            top10_counts[candidate_id] = top10_counts.get(candidate_id, 0) + 1
    query_count = len(rows)
    result: dict[str, Any] = {
        "unique_top10_rankings": len(set(rows.values())),
        "maximum_top1_candidate_share": (
            max(top1_counts.values(), default=0) / query_count if query_count else 1.0
        ),
        "top10_candidate_coverage": len(top10_counts),
        "maximum_candidate_top10_query_share": (
            max(top10_counts.values(), default=0) / query_count if query_count else 1.0
        ),
    }
    if baseline is None:
        return result
    baseline_rows = _top10_rows(baseline)
    if set(rows) != set(baseline_rows):
        raise ValueError("ranking query inventory differs from C0")
    jaccards: list[float] = []
    differences = 0
    for query_id, row in rows.items():
        observed = set(row)
        control = set(baseline_rows[query_id])
        union = observed | control
        jaccard = len(observed & control) / len(union) if union else 1.0
        jaccards.append(jaccard)
        differences += int(row != baseline_rows[query_id])
    baseline_unique = len(set(baseline_rows.values()))
    baseline_coverage = len({value for row in baseline_rows.values() for value in row})
    result.update(
        {
            "top10_difference_query_share": differences / query_count if query_count else 0.0,
            "mean_top10_jaccard": statistics.fmean(jaccards) if jaccards else 1.0,
            "unique_top10_retention": (
                result["unique_top10_rankings"] / baseline_unique
                if baseline_unique
                else 1.0
            ),
            "candidate_coverage_retention": (
                result["top10_candidate_coverage"] / baseline_coverage
                if baseline_coverage
                else 1.0
            ),
        }
    )
    return result


def _tree_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def assess_multiresolution_viability(
    artifact: MultiResolutionResult | str | Path,
) -> dict[str, Any]:
    """Apply every locked issue-18 qrel-free viability threshold."""

    root = None if isinstance(artifact, MultiResolutionResult) else Path(artifact)
    result = (
        artifact
        if isinstance(artifact, MultiResolutionResult)
        else load_multiresolution_artifact(root)
    )
    return _assess_multiresolution_viability(
        result, require_official_coverage=True, root=root
    )


def _assess_multiresolution_viability(
    result: MultiResolutionResult,
    *,
    require_official_coverage: bool,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compute a receipt; tests may disable only the official inventory-size gate."""

    expected_shape = (len(result.query_ids), len(result.candidate_ids))
    expected_component_keys = {
        f"{label}_k{resolution}"
        for resolution in LOCKED_RESOLUTIONS
        for label in ("max", "top2")
    }
    component_score_schema = (
        set(result.component_scores) == expected_component_keys
        and all(
            scores.dtype.str == "<f4"
            and scores.shape == expected_shape
            and np.isfinite(scores).all()
            for scores in result.component_scores.values()
        )
    )
    system_score_schema = set(result.system_scores) == set(EXPECTED_SYSTEMS) and all(
        scores.dtype.str == "<f4"
        and scores.shape == expected_shape
        and np.isfinite(scores).all()
        for scores in result.system_scores.values()
    )
    contributor_schema = set(result.contributing_centroid_ids) == expected_component_keys
    if contributor_schema:
        for key, contributors in result.contributing_centroid_ids.items():
            width = 1 if key.startswith("max_") else result.config.top_count
            if (
                contributors.dtype.str != "<i4"
                or contributors.shape != (*expected_shape, width)
            ):
                contributor_schema = False
                break
    cluster_diagnostics = {
        f"k{resolution}": _cluster_diagnostics(result.profiles[str(resolution)])
        for resolution in LOCKED_RESOLUTIONS
    }
    component_rankings = {
        key: (
            _rank_scores(result, scores)
            if scores.dtype.str == "<f4"
            and scores.shape == expected_shape
            and np.isfinite(scores).all()
            else {query_id: [] for query_id in result.query_ids}
        )
        for key, scores in result.component_scores.items()
    }
    for key in expected_component_keys - set(component_rankings):
        component_rankings[key] = {query_id: [] for query_id in result.query_ids}
    empty_ranking = {query_id: [] for query_id in result.query_ids}
    c0_ranking = result.rankings.get("c0_max_k10", empty_ranking)
    diagnostic_rankings = {
        **component_rankings,
        **{system: result.rankings.get(system, empty_ranking) for system in EXPECTED_SYSTEMS},
    }
    ranking_diagnostics = {
        system: _ranking_diagnostics(
            ranking,
            c0_ranking if system in NEW_PROFILE_SYSTEMS else None,
        )
        for system, ranking in diagnostic_rankings.items()
    }
    thresholds: dict[str, dict[str, Any]] = {}
    cluster_names = tuple(list(VIABILITY_THRESHOLDS)[:8])
    for name in cluster_names:
        thresholds[name] = _threshold_receipt(
            name,
            {key: diagnostics[name] for key, diagnostics in cluster_diagnostics.items()},
        )
    ranking_names = tuple(list(VIABILITY_THRESHOLDS)[8:12])
    for name in ranking_names:
        thresholds[name] = _threshold_receipt(
            name,
            {
                system: diagnostics[name]
                for system, diagnostics in ranking_diagnostics.items()
            },
        )
    incremental_fields = {
        "incremental_top10_difference_query_share": "top10_difference_query_share",
        "incremental_mean_top10_jaccard": "mean_top10_jaccard",
        "incremental_unique_top10_retention": "unique_top10_retention",
        "incremental_candidate_coverage_retention": "candidate_coverage_retention",
    }
    for name, field in incremental_fields.items():
        thresholds[name] = _threshold_receipt(
            name,
            {system: ranking_diagnostics[system][field] for system in NEW_PROFILE_SYSTEMS},
        )
    integrity_checks = {
        "source_binding": (
            result.source_aggregate_sha256 == result.config.source_aggregate_sha256
            == EVALUATED_CONSTRUCTION_AGGREGATE_SHA256
        ),
        "query_identities": bool(result.query_ids)
        and len(set(result.query_ids)) == len(result.query_ids),
        "candidate_identities": bool(result.candidate_ids)
        and len(set(result.candidate_ids)) == len(result.candidate_ids),
        "official_322_by_150_coverage": (
            not require_official_coverage
            or (len(result.query_ids), len(result.candidate_ids)) == (322, 150)
        ),
        "complete_rankings": set(result.rankings) == set(EXPECTED_SYSTEMS)
        and all(
            set(ranking) == set(result.query_ids)
            and all(
                len(row) == len(result.candidate_ids)
                and set(row) == set(result.candidate_ids)
                for row in ranking.values()
            )
            for ranking in result.rankings.values()
        ),
        "component_score_schema": component_score_schema,
        "system_score_schema": system_score_schema,
        "contributor_schema": contributor_schema,
        "finite_scores": component_score_schema and system_score_schema,
        "ranking_score_consistency": system_score_schema
        and set(result.rankings) == set(EXPECTED_SYSTEMS)
        and all(
            result.rankings[system] == _rank_scores(result, result.system_scores[system])
            for system in EXPECTED_SYSTEMS
        ),
        "normalized_centroids": all(
            item.get("available") is False
            or (
                np.isfinite(np.asarray(item["centroids"])).all()
                and np.allclose(
                    np.linalg.norm(np.asarray(item["centroids"]), axis=1),
                    1.0,
                    atol=1e-5,
                    rtol=0.0,
                )
            )
            for payload in result.profiles.values()
            for item in payload.values()
        ),
    }
    failed_integrity = [name for name, passed in integrity_checks.items() if not passed]
    failed_viability = [name for name, item in thresholds.items() if not item["passed"]]
    cluster_threshold_names = set(tuple(VIABILITY_THRESHOLDS)[:8])
    query_sensitivity_names = set(tuple(VIABILITY_THRESHOLDS)[8:12])
    failed_integrity.extend(
        name for name in failed_viability if name in cluster_threshold_names
    )
    failed_query_sensitivity = [
        name for name in failed_viability if name in query_sensitivity_names
    ]
    failed_incremental = [
        name for name in incremental_fields if not thresholds[name]["passed"]
    ]
    decision = _viability_decision(
        failed_integrity, failed_query_sensitivity, failed_incremental
    )
    artifact_bytes = _tree_bytes(root) if root is not None else None
    source_root = Path(result.config.construction_path)
    source_bytes = _tree_bytes(source_root)
    artifact_aggregate = (
        _read_json(root / "manifest.json")["aggregate_sha256"]
        if root is not None
        else None
    )
    if root is not None:
        control_sources, controls = _load_control_rankings(result, root)
    else:
        in_memory_control = result.rankings.get("c0_max_k10", result.dense_rankings)
        controls = {
            "query_knn": in_memory_control,
            "contextual_e5": in_memory_control,
        }
        inventory = {
            "query_count": len(result.query_ids),
            "candidate_count": len(result.candidate_ids),
            "query_inventory_sha256": _length_framed_hash(result.query_ids),
            "candidate_inventory_sha256": _length_framed_hash(result.candidate_ids),
        }
        control_sources = {
            name: {
                "ranking_name": (
                    "dense_reference" if name == "query_knn" else "contextual_e5_control"
                ),
                "ranking_file": "unpersisted",
                "ranking_sha256": hashlib.sha256(
                    json.dumps(
                        controls[name], sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "source_aggregate_sha256": result.config.source_aggregate_sha256,
                **inventory,
            }
            for name in CONTROL_NAMES
        }
    if set(result.rankings) == set(EXPECTED_SYSTEMS):
        control_comparisons = _control_comparisons(result.rankings, controls)
    else:
        empty_comparison = {
            "mean_top10_jaccard": 0.0,
            "top10_difference_query_share": 0.0,
            "mean_top10_overlap_count": 0.0,
            "mean_common_top10_absolute_rank_displacement": 0.0,
        }
        control_comparisons = {
            system: {
                control: dict(empty_comparison) for control in CONTROL_NAMES
            }
            for system in EXPECTED_SYSTEMS
        }
    source_elapsed = _source_elapsed_seconds(source_root)
    artifact_elapsed = PASS_ONE_ELAPSED_SECONDS
    storage_ratio = (
        artifact_bytes / source_bytes
        if artifact_bytes is not None and source_bytes
        else None
    )
    wall_time_ratio = artifact_elapsed / source_elapsed if source_elapsed else None
    storage_ok = (
        artifact_bytes <= 4 * source_bytes
        if artifact_bytes is not None and source_bytes
        else None
    )
    wall_time_ok = artifact_elapsed <= 4 * source_elapsed if source_elapsed else None
    repeat_cost_authorized = storage_ok is True and wall_time_ok is True
    repeat_authorized = decision == "continue" and repeat_cost_authorized
    return {
        "artifact_type": "issue18_multiresolution_viability_receipt",
        "version": 1,
        "qrel_free": True,
        "artifact_aggregate_sha256": artifact_aggregate,
        "decision": decision,
        "thresholds": thresholds,
        "integrity_checks": integrity_checks,
        "failed_thresholds": [
            *failed_integrity,
            *failed_query_sensitivity,
            *failed_incremental,
        ],
        "failed_integrity_thresholds": failed_integrity,
        "failed_query_sensitivity_thresholds": failed_query_sensitivity,
        "failed_incremental_thresholds": failed_incremental,
        "cluster_diagnostics": cluster_diagnostics,
        "ranking_diagnostics": ranking_diagnostics,
        "control_sources": control_sources,
        "control_comparisons": control_comparisons,
        "execution_provenance": _execution_provenance(artifact_aggregate),
        "cost_evidence": {
            "issue12_construction_storage_bytes": source_bytes,
            "multiresolution_artifact_storage_bytes": artifact_bytes,
            "storage_ratio": storage_ratio,
            "storage_at_most_four_times_issue12": storage_ok,
            "issue12_construction_elapsed_seconds": source_elapsed,
            "multiresolution_construction_elapsed_seconds": artifact_elapsed,
            "wall_time_ratio": wall_time_ratio,
            "wall_time_at_most_four_times_issue12": wall_time_ok,
            "repeat_cost_authorized": repeat_cost_authorized,
        },
        "repeat_authorized": repeat_authorized,
    }


def _source_elapsed_seconds(source_root: Path) -> float | None:
    metadata_path = source_root / "metadata.json"
    if not metadata_path.is_file():
        return None
    payload = _read_json(metadata_path)
    value = payload.get("construction_runtime", {}).get("elapsed_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _validated_manifest_root(path: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    entries = manifest.get("files")
    if (
        set(manifest) != {
            "artifact_type", "version", "qrel_free", "files", "aggregate_sha256"
        }
        or manifest.get("artifact_type") != "issue18_multiresolution_profile_manifest"
        or manifest.get("version") != ARTIFACT_VERSION
        or manifest.get("qrel_free") is not True
        or not isinstance(entries, dict)
        or set(entries) != _expected_artifact_files()
        or manifest.get("aggregate_sha256") != _aggregate_hash(entries)
    ):
        raise ValueError("repeat artifact manifest contract changed")
    for name, entry in entries.items():
        item = root / name
        if (
            not item.is_file()
            or set(entry) != {"byte_size", "sha256"}
            or item.stat().st_size != entry["byte_size"]
            or _sha256(item) != entry["sha256"]
        ):
            raise ValueError(f"repeat artifact hash mismatch: {name}")
    return root, manifest


def compare_multiresolution_artifacts(
    first: str | Path, second: str | Path
) -> dict[str, Any]:
    """Compare two independent qrel-free construction artifacts."""

    load_multiresolution_artifact(first)
    load_multiresolution_artifact(second)
    first_root, first_manifest = _validated_manifest_root(first)
    second_root, second_manifest = _validated_manifest_root(second)
    first_aggregate = first_manifest["aggregate_sha256"]
    second_aggregate = second_manifest["aggregate_sha256"]
    if first_aggregate == second_aggregate:
        raise ValueError("repeat artifacts are not independent aggregate hashes")
    source_files = {
        "source.json", "inputs.json", "records.json", "query_embeddings.npy",
        "record_embeddings.npy", "rankings/dense_reference.json",
    }
    source_arrays_reused = all(
        first_manifest["files"][name]["sha256"]
        == second_manifest["files"][name]["sha256"]
        for name in source_files
    )
    if not source_arrays_reused:
        raise ValueError("repeat artifacts do not reuse the exact bound source")
    excluded = {"config.json", "validation.json"}
    scientific_files = {
        name
        for name in first_manifest["files"]
        if name not in excluded
        and not name.startswith("component_scores/")
        and not name.startswith("scores/")
    }
    identical_files = {
        name: first_manifest["files"][name]["sha256"]
        == second_manifest["files"][name]["sha256"]
        for name in sorted(scientific_files)
    }
    maximum = 0.0
    for directory in ("component_scores", "scores"):
        for first_score in (first_root / directory).glob("*.npy"):
            second_score = second_root / directory / first_score.name
            left = np.load(first_score, allow_pickle=False)
            right = np.load(second_score, allow_pickle=False)
            if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError("repeat score arrays are not finite with identical shapes")
            maximum = max(
                maximum,
                float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))),
            )
    rankings_identical = all(
        identical
        for name, identical in identical_files.items()
        if name.startswith("rankings/") or name.startswith("runs/")
    )
    canonical_identical = all(identical_files.values())
    checks = {
        "independent_aggregate_hashes": True,
        "exact_bound_source_reused": source_arrays_reused,
        "canonical_scientific_files_identical": canonical_identical,
        "rankings_identical": rankings_identical,
        "maximum_absolute_score_difference_at_most_1e-5": maximum <= 1e-5,
    }
    return {
        "artifact_type": "issue18_multiresolution_repeatability_receipt",
        "version": 1,
        "qrel_free": True,
        "status": "passed" if all(checks.values()) else "failed",
        "first_aggregate_sha256": first_aggregate,
        "second_aggregate_sha256": second_aggregate,
        "source_arrays_reused": source_arrays_reused,
        "canonical_scientific_files_identical": canonical_identical,
        "rankings_identical": rankings_identical,
        "maximum_absolute_score_difference": maximum,
        "tolerance": 1e-5,
        "integrity_checks": checks,
        "canonical_file_comparison": identical_files,
    }


def write_multiresolution_receipt(
    receipt: Mapping[str, Any], output: str | Path
) -> Path:
    """Serialize one finite qrel-free receipt atomically without overwriting."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"receipt already exists: {destination}")
    try:
        encoded = json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except ValueError as error:
        raise ValueError("receipt values must be finite JSON") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        if destination.exists():
            raise FileExistsError(f"receipt already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "EVALUATED_CONSTRUCTION_AGGREGATE_SHA256",
    "EVALUATED_CONSTRUCTION_PATH",
    "EXPECTED_SYSTEMS",
    "MultiResolutionConfig",
    "MultiResolutionResult",
    "VIABILITY_THRESHOLDS",
    "assess_multiresolution_viability",
    "compare_multiresolution_artifacts",
    "construct_multiresolution",
    "load_multiresolution_artifact",
    "write_multiresolution_receipt",
    "write_multiresolution_artifact",
]
