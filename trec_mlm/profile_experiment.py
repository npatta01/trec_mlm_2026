"""Qrel-free orchestration and immutable artifacts for semantic profiles."""

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

from trec_mlm.label_experiment import (
    load_candidate_manifest,
    load_label_experiment_artifacts,
)
from trec_mlm.label_inference import _encode_contextual_e5
from trec_mlm.label_retrieval import format_query
from trec_mlm.label_retrieval import rank_score_matrix
from trec_mlm.profile_inference import (
    PROFILE_BATCH_SIZE,
    PROFILE_SEED,
    encode_profile_inputs,
    format_profile_document,
    format_profile_query,
)
from trec_mlm.profile_retrieval import (
    CandidateProfile,
    ProfileScoreResult,
    aggregate_cluster_labels,
    canonical_kmeans,
    filter_and_deduplicate_records,
    reciprocal_rank_fusion,
    score_profiles,
)
from trec_mlm.query_knn import load_queries_tsv
from trec_mlm.response_retrieval import load_discovery_records
from trec_mlm.runio import (
    rankings_from_run,
    validate_trec_run,
    write_trec_run,
)


ARTIFACT_VERSION = 1
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


@dataclass(frozen=True, slots=True)
class ProfileExperimentConfig:
    query_path: str = "data/dev/llm_dev_data.tsv"
    discovery_path: str = "data/dev/llm_discovery_data.json"
    candidate_path: str = "reports/trec_2026_discovery_profile_audit.json"
    dense_artifact_path: str = "artifacts/dense_query_knn_baseline"
    e5_artifact_path: str = "artifacts/contextual_e5"
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
    query_embeddings: NDArray[np.float32] | None = None
    record_embeddings: NDArray[np.float32] | None = None
    label_question_embeddings: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class LoadedProfileArtifacts:
    config: ProfileExperimentConfig
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    profile_scores: NDArray[np.float32]
    winning_centroid_ids: NDArray[np.int32]
    rankings: dict[str, dict[str, list[str]]]
    fusion_scores: dict[str, NDArray[np.float32]]
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

    _validate_config(config)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must equal 1 before profile construction")
    queries = load_queries_tsv(config.query_path)
    query_ids = tuple(queries)
    query_texts = tuple(queries.values())
    candidate_ids = load_candidate_manifest(config.candidate_path)
    corpus = load_discovery_records(config.discovery_path, candidate_ids)
    filtered = filter_and_deduplicate_records(corpus, candidate_ids)
    flat_records = tuple(
        record
        for candidate_id in candidate_ids
        for record in filtered.records_by_candidate[candidate_id]
    )
    formatted_queries = tuple(format_profile_query(text) for text in query_texts)
    formatted_records = tuple(
        format_profile_document(record.question, record.response) for record in flat_records
    )
    query_embeddings, record_embeddings, profile_encoder_metadata = encode_profile_inputs(
        formatted_queries=formatted_queries,
        formatted_documents=formatted_records,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    unique_questions: dict[str, str] = {}
    for record in flat_records:
        existing = unique_questions.setdefault(record.source_query_id, record.question)
        if existing != record.question:
            raise ValueError("source query ID maps to multiple question strings")
    source_query_ids = tuple(sorted(unique_questions))
    formatted_label_queries = tuple(
        format_query(unique_questions[item]) for item in source_query_ids
    )
    label_question_embeddings, label_encoder_metadata = _encode_contextual_e5(
        formatted_label_queries, batch_size=32, seed=config.seed
    )
    e5 = load_label_experiment_artifacts(config.e5_artifact_path)
    if e5.query_ids != query_ids or e5.candidate_ids != candidate_ids:
        raise ValueError("contextual E5 artifact inventory differs from profile inputs")
    question_label_scores = np.asarray(
        label_question_embeddings @ e5.label_embeddings.T, dtype=np.float32
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
                "audit": asdict(filtered.audit_by_candidate[candidate_id]),
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
            profile, asdict(filtered.audit_by_candidate[candidate_id])
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
    dense_rankings = rankings_from_run(dense_run)
    frozen = freeze_profile_rankings(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        dense_rankings=dense_rankings,
        e5_rankings=e5.rankings,
        profile_scores=scored,
        unavailable_profile_ids=frozenset(
            item for item, profile in profiles.items() if profile is None
        ),
    )
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
            "source_sha256": corpus.source_sha256,
            "profile_encoder": profile_encoder_metadata,
            "label_encoder": label_encoder_metadata,
            "dense_run_sha256": _sha256(dense_run),
            "e5_manifest_path": str(Path(config.e5_artifact_path) / "validation.json"),
            "record_count": len(flat_records),
            "source_question_count": len(source_query_ids),
        },
        profile_payload=profile_payload,
        query_embeddings=query_embeddings,
        record_embeddings=record_embeddings,
        label_question_embeddings=label_question_embeddings,
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


def _aggregate_hash(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(hashes.items())), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
            {"query_ids": result.query_ids, "candidate_ids": result.candidate_ids},
        )
        _write_json(staging / "metadata.json", result.metadata)
        _write_json(staging / "profiles.json", result.profile_payload)
        np.save(staging / "profile_scores.npy", result.profile_scores)
        np.save(staging / "winning_centroid_ids.npy", result.winning_centroid_ids)
        optional_arrays = {
            "query_embeddings.npy": result.query_embeddings,
            "record_embeddings.npy": result.record_embeddings,
            "label_question_embeddings.npy": result.label_question_embeddings,
        }
        for name, values in optional_arrays.items():
            if values is not None:
                np.save(staging / name, values)
        (staging / "scores").mkdir(parents=True, exist_ok=True)
        for key in FROZEN_RANKING_KEYS:
            _write_json(staging / "rankings" / f"{key}.json", result.rankings[key])
            np.save(staging / "scores" / f"{key}.npy", result.fusion_scores[key])
            write_trec_run(
                staging / "runs" / f"{key}.trec",
                result.rankings[key],
                run_id=key,
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
            },
        )
        files = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        )
        hashes = {name: _sha256(staging / name) for name in files}
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
    for name, expected in hashes.items():
        if _sha256(root / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config = ProfileExperimentConfig(
        **{field.name: config_payload[field.name] for field in fields(ProfileExperimentConfig)}
    )
    _validate_config(config)
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    query_ids = tuple(inputs["query_ids"])
    candidate_ids = tuple(inputs["candidate_ids"])
    profile_scores = np.load(root / "profile_scores.npy", allow_pickle=False)
    winners = np.load(root / "winning_centroid_ids.npy", allow_pickle=False)
    if profile_scores.shape != (len(query_ids), len(candidate_ids)) or winners.shape != profile_scores.shape:
        raise ValueError("profile score shape changed")
    rankings: dict[str, dict[str, list[str]]] = {}
    scores: dict[str, NDArray[np.float32]] = {}
    for key in FROZEN_RANKING_KEYS:
        rankings[key] = json.loads(
            (root / "rankings" / f"{key}.json").read_text(encoding="utf-8")
        )
        _validate_complete_rankings(rankings[key], query_ids, candidate_ids)
        if rankings_from_run(root / "runs" / f"{key}.trec") != rankings[key]:
            raise ValueError(f"TREC run disagrees with ranking {key!r}")
        scores[key] = np.asarray(
            np.load(root / "scores" / f"{key}.npy", allow_pickle=False),
            dtype=np.float32,
        )
        if scores[key].shape != profile_scores.shape or not np.isfinite(scores[key]).all():
            raise ValueError(f"frozen score shape changed for {key!r}")
        reconstructed = rank_score_matrix(
            scores=scores[key], query_ids=query_ids, candidate_ids=candidate_ids
        )
        if reconstructed.rankings != rankings[key]:
            raise ValueError(f"frozen scores disagree with ranking {key!r}")
    return LoadedProfileArtifacts(
        config, query_ids, candidate_ids, np.asarray(profile_scores, dtype=np.float32),
        np.asarray(winners, dtype=np.int32), rankings, scores, manifest, root
    )


def compare_profile_repeats(
    first_dir: str | Path, second_dir: str | Path
) -> dict[str, Any]:
    """Require canonical ranking identity and at most 1e-5 score drift."""

    first = load_profile_artifacts(first_dir)
    second = load_profile_artifacts(second_dir)
    if first.query_ids != second.query_ids or first.candidate_ids != second.candidate_ids:
        raise ValueError("repeat inventories differ")
    rankings_identical = first.rankings == second.rankings
    canonical_files = (
        "inputs.json",
        "profiles.json",
        "winning_centroid_ids.npy",
    )
    canonical_files_identical = all(
        first.manifest["files"].get(name) == second.manifest["files"].get(name)
        for name in canonical_files
    )
    maximum = float(np.max(np.abs(first.profile_scores - second.profile_scores)))
    for key in FROZEN_RANKING_KEYS:
        maximum = max(
            maximum,
            float(np.max(np.abs(first.fusion_scores[key] - second.fusion_scores[key]))),
        )
    if not rankings_identical:
        raise ValueError("repeat rankings are not byte-equivalent canonically")
    if not canonical_files_identical:
        raise ValueError("repeat memberships, centroids, annotations, or winners differ")
    if maximum > 1e-5:
        raise ValueError(f"repeat maximum score difference exceeds 1e-05: {maximum}")
    return {
        "status": "passed",
        "rankings_identical": True,
        "canonical_profile_files_identical": True,
        "maximum_absolute_score_difference": maximum,
        "tolerance": 1e-5,
        "first_aggregate_sha256": first.manifest["aggregate_sha256"],
        "second_aggregate_sha256": second.manifest["aggregate_sha256"],
    }


__all__ = [
    "FROZEN_RANKING_KEYS",
    "FrozenProfileRankings",
    "LoadedProfileArtifacts",
    "ProfileExperimentConfig",
    "ProfileExperimentResult",
    "compare_profile_repeats",
    "freeze_profile_rankings",
    "load_profile_artifacts",
    "run_profile_experiment",
    "write_profile_artifacts",
]
