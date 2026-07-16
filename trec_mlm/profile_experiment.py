"""Qrel-free orchestration and immutable artifacts for semantic profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import resource
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.artifact_io import (
    length_framed_sha256,
    manifest_aggregate_sha256,
    sha256_file,
    source_tree_sha256,
    write_pretty_json,
)
from trec_mlm.label_experiment import load_candidate_manifest
from trec_mlm.label_retrieval import format_query, rank_score_matrix
from trec_mlm.profile_inference import (
    PROFILE_BATCH_SIZE,
    PROFILE_SEED,
    format_profile_document,
    format_profile_query,
    hash_formatted_inputs,
)
from trec_mlm.profile_retrieval import (
    CandidateProfile,
    ProfileScoreResult,
    aggregate_cluster_labels,
    canonical_kmeans,
    canonical_profile_record_id,
    filter_and_deduplicate_records,
    reciprocal_rank_fusion,
    score_profiles,
)
from trec_mlm.profile_inference_artifact import (
    OFFICIAL_CANDIDATE_COUNT,
    OFFICIAL_LABEL_QUESTION_COUNT,
    OFFICIAL_QUERY_COUNT,
    OFFICIAL_RECORD_COUNT,
    ProfileInferenceArtifactResult,
    load_profile_inference_artifacts,
    ordered_source_questions,
)
from trec_mlm.query_knn import load_queries_tsv
from trec_mlm.response_retrieval import load_discovery_records
from trec_mlm.runio import (
    rankings_from_run,
    validate_trec_run,
    write_trec_run,
)


ARTIFACT_VERSION = 2
PROFILE_DIAGNOSTIC_ATOL = float(2.0 * np.finfo(np.float32).eps)
OFFICIAL_DISCOVERY_SHA256 = (
    "b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25"
)
FROZEN_DENSE_RUN_SHA256 = (
    "1ec008416befc8996be01ef26afb2701a2765983fa5b475acafc21c217ca7a92"
)
FROZEN_E5_VALIDATION_SHA256 = (
    "0049f337106aa0bafa183d43cb8bd08f6426ffc2bb3a4d9d3991f5fb46806c35"
)
FROZEN_RANKING_KEYS = (
    "profile_only",
    "dense_e5_equal",
    "dense_profile_equal",
    "dense_profile_dense_heavy",
    "dense_profile_profile_heavy",
    "dense_profile_e5_equal",
)
FUSION_WEIGHTS: dict[str, dict[str, float]] = {
    "dense_e5_equal": {"dense": 1.0, "e5": 1.0},
    "dense_profile_equal": {"dense": 1.0, "profile": 1.0},
    "dense_profile_dense_heavy": {"dense": 2.0, "profile": 1.0},
    "dense_profile_profile_heavy": {"dense": 1.0, "profile": 2.0},
    "dense_profile_e5_equal": {"dense": 1.0, "profile": 1.0, "e5": 1.0},
}
INFERENCE_REPEAT_CHECKS = frozenset(
    {
        "canonical_inputs_and_rankings_identical",
        "metadata_and_token_audits_identical",
        "embedding_tolerance",
    }
)
CONSTRUCTION_REPEAT_CHECKS = frozenset(
    {
        "inventories",
        "formatted_inputs",
        "memberships_centroids_annotations_winners",
        "source_and_frozen_rankings",
        "embedding_and_score_tolerance",
        "inference_repeat_gate",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileExperimentConfig:
    query_path: str = "data/dev/llm_dev_data.tsv"
    discovery_path: str = "data/dev/llm_discovery_data.json"
    candidate_path: str = "reports/trec_2026_discovery_profile_audit.json"
    dense_artifact_path: str = "artifacts/dense_query_knn_baseline"
    inference_artifact_path: str = "artifacts/candidate_semantic_profiles/inference"
    batch_size: int = PROFILE_BATCH_SIZE
    seed: int = PROFILE_SEED
    numerical_threads: int = 1


@dataclass(frozen=True, slots=True)
class FrozenProfileRankings:
    scores: dict[str, NDArray[np.float32]]
    rankings: dict[str, dict[str, list[str]]]


@dataclass(frozen=True, slots=True)
class ProfileExperimentResult:
    config: ProfileExperimentConfig
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    profile_scores: NDArray[np.float32]
    winning_centroid_ids: NDArray[np.int32]
    rankings: dict[str, dict[str, list[str]]]
    fusion_scores: dict[str, NDArray[np.float32]]
    metadata: dict[str, Any]
    profile_payload: dict[str, Any]
    source_rankings: dict[str, dict[str, list[str]]]
    record_payload: list[dict[str, Any]]
    formatted_inputs: dict[str, list[str]]
    label_question_ids: tuple[str, ...]
    query_embeddings: NDArray[np.float32] | None = None
    record_embeddings: NDArray[np.float32] | None = None
    label_question_embeddings: NDArray[np.float32] | None = None
    label_embeddings: NDArray[np.float32] | None = None
    inference_artifact_root: Path | None = None
    inference_aggregate_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedProfileArtifacts:
    config: ProfileExperimentConfig
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    profile_scores: NDArray[np.float32]
    winning_centroid_ids: NDArray[np.int32]
    rankings: dict[str, dict[str, list[str]]]
    fusion_scores: dict[str, NDArray[np.float32]]
    source_rankings: dict[str, dict[str, list[str]]]
    validation: dict[str, Any]
    manifest: dict[str, Any]
    root: Path


def _validate_config(config: ProfileExperimentConfig) -> None:
    for field in fields(ProfileExperimentConfig):
        value = getattr(config, field.name)
        if field.name.endswith("_path") and (not isinstance(value, str) or not value):
            raise ValueError(f"{field.name} must be a non-empty path")
    if config.batch_size != PROFILE_BATCH_SIZE:
        raise ValueError(f"batch_size must be {PROFILE_BATCH_SIZE}")
    if config.seed != PROFILE_SEED:
        raise ValueError(f"seed must be {PROFILE_SEED}")
    if config.numerical_threads != 1:
        raise ValueError("numerical_threads must be 1")


def _validate_complete_rankings(
    rankings: Mapping[str, Sequence[str]],
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> None:
    if set(rankings) != set(query_ids):
        raise ValueError("ranking query inventory mismatch")
    expected = set(candidate_ids)
    for query_id in query_ids:
        values = list(rankings[query_id])
        if len(values) != len(candidate_ids) or set(values) != expected:
            raise ValueError(f"ranking candidate inventory mismatch for {query_id!r}")


def candidate_truncation_audits(
    candidate_ids: Sequence[str],
    record_counts: Mapping[str, int],
    classifications: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Aggregate ordered per-record truncation classes by candidate."""

    expected = sum(record_counts[item] for item in candidate_ids)
    if len(classifications) != expected:
        raise ValueError("truncation inventory differs from usable record inventory")
    output: dict[str, dict[str, int]] = {}
    cursor = 0
    for candidate_id in candidate_ids:
        count = record_counts[candidate_id]
        values = classifications[cursor : cursor + count]
        cursor += count
        if any(item not in {"none", "partial", "full"} for item in values):
            raise ValueError("unknown response truncation class")
        output[candidate_id] = {
            "not_truncated": values.count("none"),
            "partial_response_truncation": values.count("partial"),
            "full_response_truncation": values.count("full"),
        }
    return output


def validate_official_inference_handoff(
    *,
    inference: ProfileInferenceArtifactResult,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    expected_records: Sequence[Mapping[str, Any]],
    base_audits: Mapping[str, Mapping[str, Any]],
    expected_formatted_inputs: Mapping[str, Sequence[str]],
    expected_label_question_ids: Sequence[str],
    query_source_sha256: str,
    candidate_source_sha256: str,
    enforce_official_counts: bool = True,
) -> None:
    """Re-derive and validate the complete immutable Stage 1→2 handoff."""

    if inference.query_ids != tuple(query_ids) or inference.candidate_ids != tuple(
        candidate_ids
    ):
        raise ValueError("inference artifact inventory differs from construction inputs")
    if enforce_official_counts and (
        len(query_ids) != OFFICIAL_QUERY_COUNT
        or len(candidate_ids) != OFFICIAL_CANDIDATE_COUNT
        or len(expected_records) != OFFICIAL_RECORD_COUNT
        or len(expected_label_question_ids) != OFFICIAL_LABEL_QUESTION_COUNT
    ):
        raise ValueError("official inference handoff inventory counts changed")
    if inference.label_question_ids != tuple(expected_label_question_ids):
        raise ValueError("inference label-question row mapping differs from official sources")
    canonical_formatted = {
        key: list(value) for key, value in expected_formatted_inputs.items()
    }
    if inference.formatted_inputs != canonical_formatted:
        raise ValueError("inference formatted inputs differ from official sources")
    if inference.records != [dict(item) for item in expected_records]:
        raise ValueError("inference record payload differs from official sources")
    if inference.metadata.get("query_source_sha256") != query_source_sha256 or inference.metadata.get(
        "candidate_source_sha256"
    ) != candidate_source_sha256:
        raise ValueError("inference official input hashes differ from construction inputs")
    if set(inference.audits) != set(candidate_ids):
        raise ValueError("inference candidate audit inventory differs")
    truncation_keys = (
        "not_truncated",
        "partial_response_truncation",
        "full_response_truncation",
    )
    for candidate_id in candidate_ids:
        actual = inference.audits[candidate_id]
        base = dict(base_audits[candidate_id])
        if any(actual.get(key) != value for key, value in base.items()):
            raise ValueError("inference filtering audits differ from official sources")
        if set(actual) != {*base, *truncation_keys}:
            raise ValueError("inference truncation audit schema differs")
        if any(
            not isinstance(actual[key], int) or actual[key] < 0
            for key in truncation_keys
        ) or sum(actual[key] for key in truncation_keys) != base["usable"]:
            raise ValueError("inference truncation audit arithmetic differs")
    _validate_complete_rankings(inference.e5_rankings, query_ids, candidate_ids)


def freeze_profile_rankings(
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    dense_rankings: Mapping[str, Sequence[str]],
    e5_rankings: Mapping[str, Sequence[str]],
    profile_scores: ProfileScoreResult,
    unavailable_profile_ids: frozenset[str],
) -> FrozenProfileRankings:
    """Freeze the sole primary and every preregistered ablation without qrels."""

    queries = tuple(query_ids)
    candidates = tuple(candidate_ids)
    for ranking in (dense_rankings, e5_rankings, profile_scores.rankings):
        _validate_complete_rankings(ranking, queries, candidates)
    signals = {
        "dense": dense_rankings,
        "e5": e5_rankings,
        "profile": profile_scores.rankings,
    }
    rankings: dict[str, dict[str, list[str]]] = {
        "profile_only": {
            query_id: list(profile_scores.rankings[query_id]) for query_id in queries
        }
    }
    scores: dict[str, NDArray[np.float32]] = {
        "profile_only": np.asarray(profile_scores.scores, dtype=np.float32)
    }
    for key in FROZEN_RANKING_KEYS[1:]:
        weights = FUSION_WEIGHTS[key]
        fused = reciprocal_rank_fusion(
            query_ids=queries,
            candidate_ids=candidates,
            rankings_by_signal={name: signals[name] for name in weights},
            weights=weights,
            unavailable_profile_ids=unavailable_profile_ids,
            k=60,
        )
        rankings[key] = fused.rankings
        scores[key] = fused.scores
    return FrozenProfileRankings(scores=scores, rankings=rankings)


def run_profile_experiment(config: ProfileExperimentConfig) -> ProfileExperimentResult:
    """Construct profiles and all frozen rankings without accepting judgments."""

    started = time.perf_counter()
    _validate_config(config)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must equal 1 before profile construction")
    queries = load_queries_tsv(config.query_path)
    query_ids = tuple(queries)
    candidate_ids = load_candidate_manifest(config.candidate_path)
    corpus = load_discovery_records(config.discovery_path, candidate_ids)
    if corpus.source_sha256 != OFFICIAL_DISCOVERY_SHA256:
        raise ValueError("discovery source SHA-256 differs from the frozen protocol")
    filtered = filter_and_deduplicate_records(corpus, candidate_ids)
    flat_records = tuple(
        record
        for candidate_id in candidate_ids
        for record in filtered.records_by_candidate[candidate_id]
    )
    formatted_queries = tuple(format_profile_query(text) for text in queries.values())
    formatted_records = tuple(
        format_profile_document(record.question, record.response) for record in flat_records
    )
    expected_label_question_ids, unique_questions = ordered_source_questions(corpus)
    formatted_label_questions = tuple(
        format_query(unique_questions[item]) for item in expected_label_question_ids
    )
    expected_record_payload = [
        {
            "record_id": record.record_id,
            "candidate_id": record.candidate_id,
            "source_query_id": record.source_query_id,
            "question": record.question,
            "response": record.response,
            "raw_membership_sha256": _length_framed_strings(
                (record.question, record.response)
            ),
            "formatted_sha256": hashlib.sha256(
                format_profile_document(record.question, record.response).encode("utf-8")
            ).hexdigest(),
        }
        for record in flat_records
    ]
    inference = load_profile_inference_artifacts(config.inference_artifact_path)
    validate_official_inference_handoff(
        inference=inference,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        expected_records=expected_record_payload,
        base_audits={
            item: asdict(filtered.audit_by_candidate[item]) for item in candidate_ids
        },
        expected_formatted_inputs={
            "target_queries": formatted_queries,
            "profile_documents": formatted_records,
            "label_questions": formatted_label_questions,
        },
        expected_label_question_ids=expected_label_question_ids,
        query_source_sha256=_sha256(Path(config.query_path)),
        candidate_source_sha256=_sha256(Path(config.candidate_path)),
    )
    query_embeddings = inference.query_embeddings
    record_embeddings = inference.record_embeddings
    label_question_embeddings = inference.label_question_embeddings
    label_embeddings = inference.label_embeddings
    source_query_ids = inference.label_question_ids
    question_label_scores = np.asarray(
        label_question_embeddings @ label_embeddings.T, dtype=np.float32
    )
    source_positions = {item: index for index, item in enumerate(source_query_ids)}
    record_by_id = {record.record_id: record for record in flat_records}

    profiles: dict[str, CandidateProfile | None] = {}
    profile_payload: dict[str, Any] = {}
    cursor = 0
    for candidate_id in candidate_ids:
        records = filtered.records_by_candidate[candidate_id]
        count = len(records)
        if not count:
            profiles[candidate_id] = None
            profile_payload[candidate_id] = {
                "available": False,
                "audit": inference.audits[candidate_id],
                "clusters": [],
            }
            continue
        profile = canonical_kmeans(
            candidate_id,
            records,
            record_embeddings[cursor : cursor + count],
        )
        cursor += count
        annotated_clusters = []
        for cluster in profile.clusters:
            indices = tuple(
                source_positions[record_by_id[item].source_query_id]
                for item in cluster.member_record_ids
            )
            annotation = aggregate_cluster_labels(
                member_question_indices=indices,
                question_label_scores=question_label_scores,
                label_ids=candidate_ids,
            )
            annotated_clusters.append(replace(cluster, annotation=annotation))
        profile = replace(profile, clusters=tuple(annotated_clusters))
        profiles[candidate_id] = profile
        profile_payload[candidate_id] = _profile_to_json(
            profile,
            {
                **inference.audits[candidate_id],
            },
        )
    if cursor != len(flat_records):
        raise AssertionError("profile embedding cursor did not cover all records")

    scored = score_profiles(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        query_embeddings=query_embeddings,
        profiles=profiles,
    )
    dense_root = Path(config.dense_artifact_path)
    dense_run = dense_root / "run.trec" if dense_root.is_dir() else dense_root
    if _sha256(dense_run) != FROZEN_DENSE_RUN_SHA256:
        raise ValueError("dense run SHA-256 differs from frozen reference")
    dense_rankings = rankings_from_run(dense_run)
    frozen = freeze_profile_rankings(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        dense_rankings=dense_rankings,
        e5_rankings=inference.e5_rankings,
        profile_scores=scored,
        unavailable_profile_ids=frozenset(
            item for item, profile in profiles.items() if profile is None
        ),
    )
    cluster_supports = [
        cluster.support_count
        for profile in profiles.values()
        if profile is not None
        for cluster in profile.clusters
    ]
    return ProfileExperimentResult(
        config=config,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        profile_scores=scored.scores,
        winning_centroid_ids=scored.winning_centroid_ids,
        rankings=frozen.rankings,
        fusion_scores=frozen.scores,
        metadata={
            "qrel_free": True,
            "source_tree_sha256": _source_tree_sha256(),
            "source_sha256": corpus.source_sha256,
            "query_source_sha256": _sha256(Path(config.query_path)),
            "candidate_source_sha256": _sha256(Path(config.candidate_path)),
            "profile_encoder": inference.metadata["profile_encoder"],
            "label_encoder": inference.metadata["label_encoder"],
            "dense_run_sha256": _sha256(dense_run),
            "e5_validation_sha256": FROZEN_E5_VALIDATION_SHA256,
            "record_count": len(flat_records),
            "source_question_count": len(source_query_ids),
            "inference_aggregate_sha256": inference.manifest["aggregate_sha256"],
            "construction_runtime": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "scipy_version": importlib.metadata.version("scipy"),
                "scikit_learn_version": importlib.metadata.version("scikit-learn"),
                "elapsed_seconds": time.perf_counter() - started,
                "platform": platform.platform(),
                "logical_cpu_count": os.cpu_count(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "numerical_threads": config.numerical_threads,
            },
            "coverage": {
                "available_candidate_count": sum(profile is not None for profile in profiles.values()),
                "unavailable_candidate_count": sum(profile is None for profile in profiles.values()),
                "cluster_count": len(cluster_supports),
                "minimum_cluster_support": min(cluster_supports),
                "median_cluster_support": float(np.median(cluster_supports)),
                "maximum_cluster_support": max(cluster_supports),
            },
        },
        profile_payload=profile_payload,
        source_rankings={
            "dense_reference": {key: list(value) for key, value in dense_rankings.items()},
            "contextual_e5_control": {
                key: list(value) for key, value in inference.e5_rankings.items()
            },
        },
        record_payload=inference.records,
        formatted_inputs=inference.formatted_inputs,
        label_question_ids=source_query_ids,
        query_embeddings=query_embeddings,
        record_embeddings=record_embeddings,
        label_question_embeddings=label_question_embeddings,
        label_embeddings=label_embeddings,
        inference_artifact_root=inference.root,
        inference_aggregate_sha256=inference.manifest["aggregate_sha256"],
    )


def _profile_to_json(profile: CandidateProfile, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        "candidate_id": profile.candidate_id,
        "requested_clusters": profile.requested_clusters,
        "actual_clusters": len(profile.clusters),
        "centroids": profile.centroids.tolist(),
        "converged": profile.converged,
        "iteration_count": profile.iteration_count,
        "audit": audit,
        "clusters": [
            {
                **{
                    field.name: getattr(cluster, field.name)
                    for field in fields(cluster)
                    if field.name != "annotation"
                },
                "annotation": asdict(cluster.annotation)
                if hasattr(cluster.annotation, "__dataclass_fields__")
                else cluster.annotation,
            }
            for cluster in profile.clusters
        ],
    }


_write_json = write_pretty_json
_sha256 = sha256_file


def _source_tree_sha256() -> str:
    return source_tree_sha256(Path(__file__).resolve().parent)


def _prediction_text(
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    rankings: Mapping[str, Sequence[str]],
    scores: NDArray[np.float32],
    profile_scores: NDArray[np.float32],
    winners: NDArray[np.int32],
) -> str:
    positions = {item: index for index, item in enumerate(candidate_ids)}
    lines = ["query_id\tcandidate_id\trank\tscore\tprofile_score\twinning_centroid_id"]
    for query_index, query_id in enumerate(query_ids):
        for rank, candidate_id in enumerate(rankings[query_id], start=1):
            candidate_index = positions[candidate_id]
            lines.append(
                "\t".join(
                    (
                        query_id,
                        candidate_id,
                        str(rank),
                        format(float(scores[query_index, candidate_index]), ".9g"),
                        format(float(profile_scores[query_index, candidate_index]), ".9g"),
                        str(int(winners[query_index, candidate_index])),
                    )
                )
            )
    return "\n".join(lines) + "\n"


def _write_predictions(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_prediction_text(**kwargs), encoding="utf-8")


_aggregate_hash = manifest_aggregate_sha256
_length_framed_strings = length_framed_sha256


def _normalize_profile_members(
    member_matrix: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Mirror the constructor's persisted-row normalization exactly."""

    values = np.asarray(member_matrix, dtype=np.float32, order="C")
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(row_norms).all()
        or np.any(row_norms == 0.0)
    ):
        raise ValueError("profile centroid members must be finite nonzero vectors")
    return np.asarray(values / row_norms, dtype=np.float32, order="C")


def _recompute_profile_centroid(member_matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Mirror the constructor's float64 mean then float32 normalization exactly."""

    normalized_members = _normalize_profile_members(member_matrix)
    mean = np.mean(normalized_members, axis=0, dtype=np.float64)
    values = np.asarray([mean], dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norm).all() or np.any(norm == 0.0):
        raise ValueError("profile centroid members do not define a finite nonzero mean")
    return np.asarray(values / norm, dtype=np.float32)[0]


def write_profile_artifacts(
    result: ProfileExperimentResult, output_dir: str | Path
) -> dict[str, str]:
    """Atomically write complete qrel-free construction artifacts."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_json(
            staging / "config.json",
            {
                "artifact_type": "candidate_semantic_profiles",
                "version": ARTIFACT_VERSION,
                "qrel_free": True,
                **asdict(result.config),
            },
        )
        _write_json(
            staging / "inputs.json",
            {
                "query_ids": result.query_ids,
                "candidate_ids": result.candidate_ids,
                "label_question_ids": result.label_question_ids,
            },
        )
        _write_json(staging / "metadata.json", result.metadata)
        _write_json(staging / "profiles.json", result.profile_payload)
        if result.inference_artifact_root is not None:
            inference_copy = staging / "inference"
            shutil.copytree(
                result.inference_artifact_root, inference_copy, copy_function=os.link
            )
            for name in ("records.json", "formatted_inputs.json"):
                os.link(result.inference_artifact_root / name, staging / name)
        else:
            _write_json(staging / "records.json", result.record_payload)
            _write_json(staging / "formatted_inputs.json", result.formatted_inputs)
        np.save(staging / "profile_scores.npy", result.profile_scores)
        np.save(staging / "winning_centroid_ids.npy", result.winning_centroid_ids)
        optional_arrays = {
            "query_embeddings.npy": result.query_embeddings,
            "record_embeddings.npy": result.record_embeddings,
            "label_question_embeddings.npy": result.label_question_embeddings,
            "label_embeddings.npy": result.label_embeddings,
        }
        for name, values in optional_arrays.items():
            if values is not None:
                if result.inference_artifact_root is not None:
                    os.link(result.inference_artifact_root / name, staging / name)
                else:
                    np.save(staging / name, values)
        (staging / "scores").mkdir(parents=True, exist_ok=True)
        for key, ranking in result.source_rankings.items():
            _validate_complete_rankings(ranking, result.query_ids, result.candidate_ids)
            _write_json(staging / "rankings" / f"{key}.json", ranking)
            write_trec_run(staging / "runs" / f"{key}.trec", ranking, run_id=key)
        for key in FROZEN_RANKING_KEYS:
            _write_json(staging / "rankings" / f"{key}.json", result.rankings[key])
            np.save(staging / "scores" / f"{key}.npy", result.fusion_scores[key])
            write_trec_run(
                staging / "runs" / f"{key}.trec",
                result.rankings[key],
                run_id=key,
            )
            _write_predictions(
                staging / "predictions" / f"{key}.tsv",
                query_ids=result.query_ids,
                candidate_ids=result.candidate_ids,
                rankings=result.rankings[key],
                scores=result.fusion_scores[key],
                profile_scores=result.profile_scores,
                winners=result.winning_centroid_ids,
            )
        run_validation = {
            key: validate_trec_run(
                staging / "runs" / f"{key}.trec",
                expected_query_ids=result.query_ids,
                expected_model_ids=result.candidate_ids,
            )
            for key in FROZEN_RANKING_KEYS
        }
        _write_json(
            staging / "validation.json",
            {
                "status": "passed",
                "qrel_free": True,
                "query_count": len(result.query_ids),
                "candidate_count": len(result.candidate_ids),
                "ranking_keys": FROZEN_RANKING_KEYS,
                "runs": run_validation,
                "integrity_checks": {
                    "qrel_independence": True,
                    "label_independence": True,
                    "source_identity": True,
                    "semantic_reconstruction": True,
                    "inventory": True,
                    "trec_completeness": True,
                },
            },
        )
        files = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        )
        hashes = {
            name: {
                "byte_size": (staging / name).stat().st_size,
                "sha256": _sha256(staging / name),
            }
            for name in files
        }
        manifest = {
            "artifact_type": "candidate_semantic_profile_manifest",
            "version": ARTIFACT_VERSION,
            "qrel_free": True,
            "files": hashes,
            "aggregate_sha256": _aggregate_hash(hashes),
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        path.relative_to(destination).as_posix(): str(path)
        for path in destination.rglob("*")
        if path.is_file()
    }


def load_profile_artifacts(artifact_dir: str | Path) -> LoadedProfileArtifacts:
    """Load construction artifacts only after immutable hash validation."""

    root = Path(artifact_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_type") != "candidate_semantic_profile_manifest"
        or manifest.get("version") != ARTIFACT_VERSION
        or manifest.get("qrel_free") is not True
    ):
        raise ValueError("profile manifest contract changed")
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or manifest.get("aggregate_sha256") != _aggregate_hash(hashes):
        raise ValueError("profile manifest aggregate hash changed")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.json"
    }
    if actual_files != set(hashes):
        raise ValueError("profile artifact file inventory differs from manifest")
    for name, expected in hashes.items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"byte_size", "sha256"}
            or (root / name).stat().st_size != expected["byte_size"]
            or _sha256(root / name) != expected["sha256"]
        ):
            raise ValueError(f"artifact hash mismatch: {name}")
    config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    metadata_payload = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    inference_sha = metadata_payload.get("inference_aggregate_sha256")
    if inference_sha is not None:
        inference = load_profile_inference_artifacts(root / "inference")
        if inference.manifest["aggregate_sha256"] != inference_sha:
            raise ValueError("embedded inference aggregate hash changed")
        for name in (
            "records.json",
            "formatted_inputs.json",
            "query_embeddings.npy",
            "record_embeddings.npy",
            "label_question_embeddings.npy",
            "label_embeddings.npy",
        ):
            if _sha256(root / name) != _sha256(root / "inference" / name):
                raise ValueError("construction inference copies are not byte-identical")
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("qrel_free") is not True:
        raise ValueError("profile validation receipt did not pass")
    expected_integrity = {
        "qrel_independence",
        "label_independence",
        "source_identity",
        "semantic_reconstruction",
        "inventory",
        "trec_completeness",
    }
    checks = validation.get("integrity_checks")
    if not isinstance(checks, dict) or set(checks) != expected_integrity or not all(checks.values()):
        raise ValueError("profile validation integrity checks changed")
    config = ProfileExperimentConfig(
        **{field.name: config_payload[field.name] for field in fields(ProfileExperimentConfig)}
    )
    _validate_config(config)
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    query_ids = tuple(inputs["query_ids"])
    candidate_ids = tuple(inputs["candidate_ids"])
    label_question_ids = tuple(inputs["label_question_ids"])
    profile_scores = np.load(root / "profile_scores.npy", allow_pickle=False)
    winners = np.load(root / "winning_centroid_ids.npy", allow_pickle=False)
    if profile_scores.dtype.str != "<f4" or winners.dtype.str != "<i4":
        raise ValueError("persisted profile score or winner dtype changed")
    profile_payload = json.loads((root / "profiles.json").read_text(encoding="utf-8"))
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    formatted_inputs = json.loads(
        (root / "formatted_inputs.json").read_text(encoding="utf-8")
    )
    if set(formatted_inputs) != {
        "target_queries",
        "profile_documents",
        "label_questions",
    }:
        raise ValueError("formatted input groups changed")
    if len(formatted_inputs["target_queries"]) != len(query_ids) or len(
        formatted_inputs["profile_documents"]
    ) != len(records) or len(formatted_inputs["label_questions"]) != len(
        label_question_ids
    ):
        raise ValueError("formatted input inventory shape changed")
    candidate_order = {item: index for index, item in enumerate(candidate_ids)}
    previous_key: tuple[int, str, str] | None = None
    for record, formatted in zip(
        records, formatted_inputs["profile_documents"], strict=True
    ):
        raw_hash = _length_framed_strings(
            (record["question"], record["response"])
        )
        expected_formatted = format_profile_document(
            record["question"], record["response"]
        )
        expected_id = canonical_profile_record_id(
            record["candidate_id"],
            record["source_query_id"],
            record["question"],
            record["response"],
        )
        current_key = (
            candidate_order[record["candidate_id"]],
            record["source_query_id"],
            record["record_id"],
        )
        valid = (
            record["raw_membership_sha256"] == raw_hash
            and record["formatted_sha256"]
            == hashlib.sha256(expected_formatted.encode("utf-8")).hexdigest()
            and record["record_id"] == expected_id
            and formatted == expected_formatted
            and (previous_key is None or current_key > previous_key)
        )
        if not valid:
            raise ValueError("formatted record inventory or raw hashes changed")
        previous_key = current_key
    expected_profile_input_hash = hash_formatted_inputs(
        (
            *formatted_inputs["target_queries"],
            *formatted_inputs["profile_documents"],
        )
    )
    if metadata_payload.get("profile_encoder", {}).get(
        "input_sha256", expected_profile_input_hash
    ) != expected_profile_input_hash:
        raise ValueError("formatted profile encoder input hash changed")
    expected_label_input_hash = hash_formatted_inputs(
        formatted_inputs["label_questions"]
    )
    if metadata_payload.get("label_encoder", {}).get(
        "input_sha256", expected_label_input_hash
    ) != expected_label_input_hash:
        raise ValueError("formatted label encoder input hash changed")
    query_embeddings = np.load(root / "query_embeddings.npy", allow_pickle=False)
    record_embeddings = np.load(root / "record_embeddings.npy", allow_pickle=False)
    label_question_embeddings = np.load(root / "label_question_embeddings.npy", allow_pickle=False)
    label_embeddings = np.load(root / "label_embeddings.npy", allow_pickle=False)
    if any(matrix.dtype.str != "<f4" for matrix in (query_embeddings, record_embeddings, label_question_embeddings, label_embeddings)):
        raise ValueError("persisted embedding dtype changed")
    if profile_scores.shape != (len(query_ids), len(candidate_ids)) or winners.shape != profile_scores.shape:
        raise ValueError("profile score shape changed")
    if not np.isfinite(profile_scores).all():
        raise ValueError("profile scores must be finite")
    if query_embeddings.shape[0] != len(query_ids) or record_embeddings.shape[0] != len(records):
        raise ValueError("embedding inventory shape changed")
    if query_embeddings.ndim != 2 or record_embeddings.ndim != 2 or query_embeddings.shape[1] != record_embeddings.shape[1]:
        raise ValueError("embedding dimensions changed")
    for name, matrix in (("query", query_embeddings), ("record", record_embeddings)):
        if not np.isfinite(matrix).all() or not np.allclose(
            np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5, rtol=0.0
        ):
            raise ValueError(f"{name} embeddings are not finite normalized vectors")
    if label_question_embeddings.shape != (
        len(label_question_ids), query_embeddings.shape[1]
    ) or label_embeddings.shape != (len(candidate_ids), query_embeddings.shape[1]):
        raise ValueError("label embedding inventory shape changed")
    for name, matrix in (
        ("label question", label_question_embeddings),
        ("label", label_embeddings),
    ):
        if not np.isfinite(matrix).all() or not np.allclose(
            np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5, rtol=0.0
        ):
            raise ValueError(f"{name} embeddings are not finite normalized vectors")
    question_label_scores = np.asarray(
        label_question_embeddings @ label_embeddings.T, dtype=np.float32
    )
    label_question_positions = {
        item: index for index, item in enumerate(label_question_ids)
    }
    record_positions = {item["record_id"]: index for index, item in enumerate(records)}
    records_by_id = {item["record_id"]: item for item in records}
    if len(record_positions) != len(records):
        raise ValueError("record IDs are not unique")
    profiles: dict[str, CandidateProfile | None] = {}
    for candidate_id in candidate_ids:
        payload = profile_payload[candidate_id]
        expected_records = {
            item["record_id"] for item in records if item["candidate_id"] == candidate_id
        }
        if payload.get("available") is False:
            if expected_records:
                raise ValueError("unavailable profile has usable records")
            profiles[candidate_id] = None
            continue
        centroids = np.asarray(payload["centroids"], dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[1] != query_embeddings.shape[1]:
            raise ValueError("profile centroid dimensions changed")
        if not np.isfinite(centroids).all() or not np.allclose(
            np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5, rtol=0.0
        ):
            raise ValueError("profile centroids are not finite normalized vectors")
        members: list[str] = []
        for index, cluster in enumerate(payload["clusters"]):
            if cluster["cluster_id"] != index or cluster["support_count"] != len(cluster["member_record_ids"]):
                raise ValueError("cluster support or ordering changed")
            if cluster.get("annotation", {}).get("affects_ranking") is not False:
                raise ValueError("cluster annotations must not affect ranking")
            cluster_members = list(cluster["member_record_ids"])
            members.extend(cluster_members)
            member_matrix = record_embeddings[
                np.asarray([record_positions[item] for item in cluster_members])
            ]
            recomputed = _recompute_profile_centroid(member_matrix)
            if not np.array_equal(recomputed, centroids[index]):
                raise ValueError("centroid arithmetic disagrees with memberships")
            similarities = np.asarray(
                member_matrix @ centroids[index], dtype=np.float32
            )
            ranked_offsets = sorted(
                range(len(cluster_members)),
                key=lambda offset: (-float(similarities[offset]), cluster_members[offset]),
            )
            representative_questions: list[str] = []
            for offset in ranked_offsets:
                question = records_by_id[cluster_members[offset]]["question"]
                if question not in representative_questions:
                    representative_questions.append(question)
                if len(representative_questions) == 3:
                    break
            expected_dispersion = float(
                np.mean(1.0 - similarities, dtype=np.float64)
            )
            diagnostics_valid = (
                cluster_members == sorted(cluster_members)
                and cluster["membership_sha256"]
                == _length_framed_strings(cluster_members)
                and cluster["medoid_record_id"]
                == cluster_members[ranked_offsets[0]]
                and cluster["representative_questions"] == representative_questions
                # Construction and validation may take distinct float32 BLAS
                # paths. Two float32 machine epsilons bound their observed
                # dot-product rounding without masking a meaningful change.
                and abs(float(cluster["dispersion"]) - expected_dispersion)
                <= PROFILE_DIAGNOSTIC_ATOL
            )
            if not diagnostics_valid:
                raise ValueError("cluster diagnostics disagree with embeddings")
            recomputed_annotation = aggregate_cluster_labels(
                member_question_indices=[
                    label_question_positions[
                        records_by_id[record_id]["source_query_id"]
                    ]
                    for record_id in cluster_members
                ],
                question_label_scores=question_label_scores,
                label_ids=candidate_ids,
            )
            if asdict(recomputed_annotation) != cluster["annotation"]:
                raise ValueError("annotation arithmetic disagrees")
        if len(members) != len(set(members)) or set(members) != expected_records:
            raise ValueError("cluster memberships do not exactly cover candidate records")
        profiles[candidate_id] = CandidateProfile(
            candidate_id,
            int(payload["requested_clusters"]),
            centroids,
            (),
            bool(payload["converged"]),
            int(payload["iteration_count"]),
        )
    recomputed_profiles = score_profiles(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        query_embeddings=query_embeddings,
        profiles=profiles,
    )
    if not np.array_equal(recomputed_profiles.scores, profile_scores):
        raise ValueError("profile score arithmetic disagrees")
    if not np.array_equal(recomputed_profiles.winning_centroid_ids, winners):
        raise ValueError("winning centroid arithmetic disagrees")
    rankings: dict[str, dict[str, list[str]]] = {}
    scores: dict[str, NDArray[np.float32]] = {}
    source_rankings: dict[str, dict[str, list[str]]] = {}
    for key in ("dense_reference", "contextual_e5_control"):
        source_rankings[key] = json.loads(
            (root / "rankings" / f"{key}.json").read_text(encoding="utf-8")
        )
        _validate_complete_rankings(source_rankings[key], query_ids, candidate_ids)
        if rankings_from_run(root / "runs" / f"{key}.trec") != source_rankings[key]:
            raise ValueError(f"TREC run disagrees with source ranking {key!r}")
    for key in FROZEN_RANKING_KEYS:
        rankings[key] = json.loads(
            (root / "rankings" / f"{key}.json").read_text(encoding="utf-8")
        )
        _validate_complete_rankings(rankings[key], query_ids, candidate_ids)
        if rankings_from_run(root / "runs" / f"{key}.trec") != rankings[key]:
            raise ValueError(f"TREC run disagrees with ranking {key!r}")
        scores[key] = np.load(root / "scores" / f"{key}.npy", allow_pickle=False)
        if scores[key].dtype.str != "<f4":
            raise ValueError(f"persisted frozen score dtype changed for {key!r}")
        if scores[key].shape != profile_scores.shape or not np.isfinite(scores[key]).all():
            raise ValueError(f"frozen score shape changed for {key!r}")
        reconstructed = rank_score_matrix(
            scores=scores[key], query_ids=query_ids, candidate_ids=candidate_ids
        )
        if reconstructed.rankings != rankings[key]:
            raise ValueError(f"frozen scores disagree with ranking {key!r}")
        expected_predictions = _prediction_text(
            query_ids=query_ids,
            candidate_ids=candidate_ids,
            rankings=rankings[key],
            scores=scores[key],
            profile_scores=profile_scores,
            winners=winners,
        )
        if (root / "predictions" / f"{key}.tsv").read_text(encoding="utf-8") != expected_predictions:
            raise ValueError(f"prediction rows disagree for {key!r}")
    if not np.array_equal(scores["profile_only"], profile_scores):
        raise ValueError("profile-only score arithmetic disagrees")
    unavailable = frozenset(
        candidate_id
        for candidate_id in candidate_ids
        if profile_payload[candidate_id].get("available") is False
    )
    signals = {
        "dense": source_rankings["dense_reference"],
        "e5": source_rankings["contextual_e5_control"],
        "profile": rankings["profile_only"],
    }
    for key in FROZEN_RANKING_KEYS[1:]:
        weights = FUSION_WEIGHTS[key]
        recomputed = reciprocal_rank_fusion(
            query_ids=query_ids,
            candidate_ids=candidate_ids,
            rankings_by_signal={name: signals[name] for name in weights},
            weights=weights,
            unavailable_profile_ids=unavailable,
            k=60,
        )
        if not np.array_equal(recomputed.scores, scores[key]):
            raise ValueError(f"fusion arithmetic disagrees for {key!r}")
        if recomputed.rankings != rankings[key]:
            raise ValueError(f"fusion ranking arithmetic disagrees for {key!r}")
    return LoadedProfileArtifacts(
        config, query_ids, candidate_ids, np.asarray(profile_scores, dtype=np.float32),
        np.asarray(winners, dtype=np.int32), rankings, scores, source_rankings,
        validation, manifest, root
    )


def compare_profile_repeats(
    first_dir: str | Path,
    second_dir: str | Path,
    inference_repeat_receipt_path: str | Path,
) -> dict[str, Any]:
    """Require canonical ranking identity and at most 1e-5 score drift."""

    first = load_profile_artifacts(first_dir)
    second = load_profile_artifacts(second_dir)
    if first.root.resolve() == second.root.resolve():
        raise ValueError("construction repeats must use distinct artifact roots")
    if first.manifest["aggregate_sha256"] == second.manifest["aggregate_sha256"]:
        raise ValueError("construction repeats must have distinct aggregate digests")
    inference_receipt = json.loads(
        Path(inference_repeat_receipt_path).read_text(encoding="utf-8")
    )
    first_metadata = json.loads(
        (first.root / "metadata.json").read_text(encoding="utf-8")
    )
    second_metadata = json.loads(
        (second.root / "metadata.json").read_text(encoding="utf-8")
    )
    first_inference_sha = first_metadata.get("inference_aggregate_sha256")
    second_inference_sha = second_metadata.get("inference_aggregate_sha256")
    inference_checks = inference_receipt.get("integrity_checks")
    inference_repeat_pass = (
        inference_receipt.get("status") == "passed"
        and inference_receipt.get("first_aggregate_sha256") == first_inference_sha
        and inference_receipt.get("second_aggregate_sha256") == second_inference_sha
        and isinstance(first_inference_sha, str)
        and isinstance(second_inference_sha, str)
        and first_inference_sha != second_inference_sha
        and isinstance(inference_checks, dict)
        and set(inference_checks) == INFERENCE_REPEAT_CHECKS
        and all(inference_checks.values())
        and inference_receipt.get("tolerance") == 1e-5
        and isinstance(
            inference_receipt.get("maximum_absolute_embedding_difference"),
            (int, float),
        )
        and inference_receipt["maximum_absolute_embedding_difference"] <= 1e-5
    )
    if not inference_repeat_pass:
        raise ValueError("inference repeat receipt does not bind both constructions")
    if first.query_ids != second.query_ids or first.candidate_ids != second.candidate_ids:
        raise ValueError("repeat inventories differ")
    rankings_identical = (
        first.rankings == second.rankings
        and first.source_rankings == second.source_rankings
    )
    groups = {
        "inventories": ("inputs.json", "records.json"),
        "formatted_inputs": ("formatted_inputs.json",),
        "memberships_centroids_annotations_winners": (
            "profiles.json",
            "winning_centroid_ids.npy",
        ),
        "source_and_frozen_rankings": tuple(
            [
                f"rankings/{key}.json"
                for key in ("dense_reference", "contextual_e5_control", *FROZEN_RANKING_KEYS)
            ]
            + [
                f"runs/{key}.trec"
                for key in ("dense_reference", "contextual_e5_control", *FROZEN_RANKING_KEYS)
            ]
        ),
    }
    integrity_checks = {
        name: all(
            first.manifest["files"].get(path)
            == second.manifest["files"].get(path)
            for path in paths
        )
        for name, paths in groups.items()
    }
    numeric_files = (
        "query_embeddings.npy",
        "record_embeddings.npy",
        "label_question_embeddings.npy",
        "label_embeddings.npy",
        "profile_scores.npy",
        *(f"scores/{key}.npy" for key in FROZEN_RANKING_KEYS),
    )
    maximum = 0.0
    for name in numeric_files:
        left = np.load(first.root / name, allow_pickle=False)
        right = np.load(second.root / name, allow_pickle=False)
        if left.shape != right.shape:
            raise ValueError(f"repeat numeric shape differs for {name!r}")
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    integrity_checks["embedding_and_score_tolerance"] = maximum <= 1e-5
    integrity_checks["inference_repeat_gate"] = inference_repeat_pass
    if not rankings_identical:
        raise ValueError("repeat rankings are not byte-equivalent canonically")
    if not all(integrity_checks.values()):
        raise ValueError("repeat memberships, centroids, annotations, or winners differ")
    if maximum > 1e-5:
        raise ValueError(f"repeat maximum score difference exceeds 1e-05: {maximum}")
    return {
        "status": "passed",
        "rankings_identical": True,
        "canonical_profile_files_identical": True,
        "integrity_checks": integrity_checks,
        "maximum_absolute_score_difference": maximum,
        "tolerance": 1e-5,
        "first_aggregate_sha256": first.manifest["aggregate_sha256"],
        "second_aggregate_sha256": second.manifest["aggregate_sha256"],
        "inference_repeat_receipt": inference_receipt,
        "inference_repeat_receipt_sha256": _sha256(
            Path(inference_repeat_receipt_path)
        ),
    }


profile_to_json = _profile_to_json
recompute_profile_centroid = _recompute_profile_centroid
validate_complete_rankings = _validate_complete_rankings


__all__ = [
    "FROZEN_RANKING_KEYS",
    "FrozenProfileRankings",
    "LoadedProfileArtifacts",
    "ProfileExperimentConfig",
    "ProfileExperimentResult",
    "compare_profile_repeats",
    "freeze_profile_rankings",
    "load_profile_artifacts",
    "profile_to_json",
    "recompute_profile_centroid",
    "run_profile_experiment",
    "validate_complete_rankings",
    "write_profile_artifacts",
]
