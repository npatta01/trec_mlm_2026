"""Qrel-free construction and immutable artifacts for issue-18 profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    construction_path: str = "artifacts/candidate_semantic_profiles/run1_record_id_v2_final"
    output_root: str = "artifacts/multiresolution_profiles/issue18_pr17head"
    resolutions: tuple[int, ...] = LOCKED_RESOLUTIONS
    top_count: int = 2
    seed: int = 2026
    rrf_k: int = 60
    source_commit: str = LOCKED_SOURCE_COMMIT
    source_aggregate_sha256: str = (
        "9bd4a81b3d3c83dbd3903d6a285f152b297cfb81a22c5a8a0ef31d0b323f35c1"
    )


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
                resolution_payload[candidate_id] = {
                    "available": False,
                    "audit": source_profiles[candidate_id]["audit"],
                    "clusters": [],
                }
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
            resolution_payload[candidate_id] = _profile_to_json(
                profile, source_profiles[candidate_id]["audit"]
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
    if (
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
    if (
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
        if not np.array_equal(component_scores[key], reconstructed_components[key]) or not np.array_equal(contributors[key], reconstructed_contributors[key]):
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


__all__ = [
    "EXPECTED_SYSTEMS",
    "MultiResolutionConfig",
    "MultiResolutionResult",
    "construct_multiresolution",
    "load_multiresolution_artifact",
    "write_multiresolution_artifact",
]
