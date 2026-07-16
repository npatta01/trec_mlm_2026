"""Qrel-free candidate semantic profiles, scoring, annotations, and fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.artifact_io import length_framed_sha256
from trec_mlm.response_retrieval import DiscoveryRecord, response_filter_reason


FloatArray = NDArray[np.floating[Any]]
REQUIRED_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
CLUSTERING_RUNTIME = ("3.13.9", "2.5.1", "1.18.0", "1.9.0")


def validate_clustering_runtime_identity(
    *,
    python_version: str,
    numpy_version: str,
    scipy_version: str,
    sklearn_version: str,
) -> None:
    """Reject drift from the advisor-approved Stage 2 numerical runtime."""

    actual = (python_version, numpy_version, scipy_version, sklearn_version)
    if actual != CLUSTERING_RUNTIME:
        raise RuntimeError(
            f"clustering runtime differs from frozen contract: {actual!r}"
        )


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    record_id: str
    source_query_id: str
    candidate_id: str
    question: str
    response: str


@dataclass(frozen=True, slots=True)
class CandidateRecordAudit:
    raw: int
    removed_null: int
    removed_blank: int
    removed_exact_idk: int
    duplicates: int
    usable: int


@dataclass(frozen=True, slots=True)
class FilteredProfileRecords:
    candidate_ids: tuple[str, ...]
    records_by_candidate: dict[str, tuple[ProfileRecord, ...]]
    audit_by_candidate: dict[str, CandidateRecordAudit]


@dataclass(frozen=True, slots=True)
class ClusterAnnotation:
    primary_label: str | None
    secondary_label: str | None
    top_score: float
    second_score: float
    margin: float
    support_count: int
    status: str
    affects_ranking: bool = False


@dataclass(frozen=True, slots=True)
class CanonicalCluster:
    cluster_id: int
    member_record_ids: tuple[str, ...]
    support_count: int
    dispersion: float
    medoid_record_id: str
    representative_questions: tuple[str, ...]
    membership_sha256: str
    annotation: ClusterAnnotation | Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    candidate_id: str
    requested_clusters: int
    centroids: NDArray[np.float32]
    clusters: tuple[CanonicalCluster, ...]
    converged: bool
    iteration_count: int


@dataclass(frozen=True, slots=True)
class ProfileScoreResult:
    scores: NDArray[np.float32]
    winning_centroid_ids: NDArray[np.int32]
    contributing_centroid_ids: NDArray[np.int32]
    rankings: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class RRFResult:
    scores: NDArray[np.float32]
    rankings: dict[str, list[str]]


_length_framed_hash = length_framed_sha256


def canonical_profile_record_id(
    candidate_id: str,
    source_query_id: str,
    question: str,
    response: str,
) -> str:
    """Return the schema-v2 globally unique identity for one profile record."""

    values = (candidate_id, source_query_id, question, response)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("profile record identity fields must be non-empty strings")
    return f"profile-record-v2:{_length_framed_hash(values)}"


def filter_and_deduplicate_records(
    records: Sequence[DiscoveryRecord], candidate_ids: Sequence[str]
) -> FilteredProfileRecords:
    """Apply the frozen basic filter and candidate-local raw exact deduplication."""

    candidates = tuple(sorted(candidate_ids))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate IDs must be non-empty and unique")
    candidate_set = set(candidates)
    grouped: dict[str, list[ProfileRecord]] = {item: [] for item in candidates}
    counts: dict[str, dict[str, int]] = {
        item: defaultdict(int) for item in candidates
    }
    seen: dict[str, set[tuple[str, str]]] = {item: set() for item in candidates}
    for record in records:
        if record.model_id not in candidate_set:
            raise ValueError(f"unexpected discovery candidate: {record.model_id!r}")
        candidate_counts = counts[record.model_id]
        candidate_counts["raw"] += 1
        reason = response_filter_reason(record.response, "basic")
        if reason is not None:
            candidate_counts[f"removed_{reason}"] += 1
            continue
        assert record.response is not None
        raw_key = (record.query_text, record.response)
        if raw_key in seen[record.model_id]:
            candidate_counts["duplicates"] += 1
            continue
        seen[record.model_id].add(raw_key)
        grouped[record.model_id].append(
            ProfileRecord(
                record_id=canonical_profile_record_id(
                    record.model_id,
                    record.source_query_id,
                    record.query_text,
                    record.response,
                ),
                source_query_id=record.source_query_id,
                candidate_id=record.model_id,
                question=record.query_text,
                response=record.response,
            )
        )
    output: dict[str, tuple[ProfileRecord, ...]] = {}
    audits: dict[str, CandidateRecordAudit] = {}
    for candidate_id in candidates:
        ordered = tuple(
            sorted(
                grouped[candidate_id],
                key=lambda item: (item.source_query_id, item.record_id),
            )
        )
        output[candidate_id] = ordered
        values = counts[candidate_id]
        audits[candidate_id] = CandidateRecordAudit(
            raw=values["raw"],
            removed_null=values["removed_null"],
            removed_blank=values["removed_blank"],
            removed_exact_idk=values["removed_exact_idk"],
            duplicates=values["duplicates"],
            usable=len(ordered),
        )
    record_ids = [record.record_id for records in output.values() for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("profile record IDs are not globally unique")
    return FilteredProfileRecords(candidates, output, audits)


def _normalize_rows(values: FloatArray) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32, order="C")
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("embeddings must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embeddings must be nonzero")
    return np.asarray(matrix / norms, dtype=np.float32, order="C")


def _validate_thread_contract() -> None:
    invalid = {
        name: os.environ.get(name)
        for name in REQUIRED_THREAD_ENVIRONMENT
        if os.environ.get(name) != "1"
    }
    if invalid:
        raise RuntimeError(f"clustering requires one-thread environment: {invalid!r}")


def canonical_kmeans(
    candidate_id: str,
    records: Sequence[ProfileRecord],
    embeddings: FloatArray,
    *,
    requested_clusters: int | None = None,
    enforce_thread_contract: bool = True,
) -> CandidateProfile:
    """Fit and canonicalize one candidate-local deterministic K-means profile."""

    import platform
    import scipy
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    validate_clustering_runtime_identity(
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        sklearn_version=sklearn.__version__,
    )

    record_tuple = tuple(records)
    if not record_tuple:
        raise ValueError("at least one profile record is required")
    if any(item.candidate_id != candidate_id for item in record_tuple):
        raise ValueError("all profile records must belong to the requested candidate")
    if len({item.record_id for item in record_tuple}) != len(record_tuple):
        raise ValueError("profile record IDs must be unique")
    matrix = _normalize_rows(embeddings)
    if matrix.shape[0] != len(record_tuple):
        raise ValueError("record and embedding counts differ")
    default_clusters = min(10, len(record_tuple))
    requested = default_clusters if requested_clusters is None else requested_clusters
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError("requested cluster count must be an integer")
    if requested < 1 or requested > len(record_tuple):
        raise ValueError("requested cluster count exceeds the usable-record rule")
    if enforce_thread_contract:
        _validate_thread_contract()

    model = KMeans(
        n_clusters=requested,
        init="k-means++",
        random_state=2026,
        n_init=10,
        algorithm="lloyd",
        max_iter=300,
        tol=1e-4,
    )
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = np.asarray(model.fit_predict(matrix), dtype=np.int32)

    occupied: list[tuple[bytes, list[int]]] = []
    by_centroid: dict[bytes, list[int]] = {}
    order: list[bytes] = []
    for fitted_id in sorted(set(int(item) for item in labels)):
        members = np.flatnonzero(labels == fitted_id).astype(int).tolist()
        if not members:
            continue
        key = np.asarray(model.cluster_centers_[fitted_id], dtype=np.float32).tobytes()
        if key not in by_centroid:
            order.append(key)
            by_centroid[key] = []
        by_centroid[key].extend(members)
    occupied = [(key, sorted(by_centroid[key])) for key in order]
    occupied.sort(key=lambda item: min(record_tuple[index].record_id for index in item[1]))

    centroids: list[NDArray[np.float32]] = []
    clusters: list[CanonicalCluster] = []
    for cluster_id, (_, members) in enumerate(occupied):
        mean = np.mean(matrix[members], axis=0, dtype=np.float64)
        centroid = _normalize_rows(np.asarray([mean], dtype=np.float32))[0]
        similarities = np.asarray(matrix[members] @ centroid, dtype=np.float32)
        ranked_members = sorted(
            range(len(members)),
            key=lambda offset: (
                -float(similarities[offset]),
                record_tuple[members[offset]].record_id,
            ),
        )
        ordered_records = [record_tuple[members[offset]] for offset in ranked_members]
        representative_questions: list[str] = []
        for item in ordered_records:
            if item.question not in representative_questions:
                representative_questions.append(item.question)
            if len(representative_questions) == 3:
                break
        member_ids = tuple(sorted(record_tuple[index].record_id for index in members))
        clusters.append(
            CanonicalCluster(
                cluster_id=cluster_id,
                member_record_ids=member_ids,
                support_count=len(members),
                dispersion=float(np.mean(1.0 - similarities, dtype=np.float64)),
                medoid_record_id=ordered_records[0].record_id,
                representative_questions=tuple(representative_questions),
                membership_sha256=_length_framed_hash(member_ids),
            )
        )
        centroids.append(centroid)
    centroid_matrix = np.asarray(centroids, dtype=np.float32, order="C")
    return CandidateProfile(
        candidate_id=candidate_id,
        requested_clusters=requested,
        centroids=centroid_matrix,
        clusters=tuple(clusters),
        converged=int(model.n_iter_) <= 300,
        iteration_count=int(model.n_iter_),
    )


def score_profiles(
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    query_embeddings: FloatArray,
    profiles: Mapping[str, CandidateProfile | None],
    top_n: int = 1,
) -> ProfileScoreResult:
    """Score by the mean of each query/candidate's top centroid cosines."""

    queries = _normalize_rows(query_embeddings)
    query_tuple = tuple(query_ids)
    candidates = tuple(candidate_ids)
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError("top_n must be an integer")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if queries.shape[0] != len(query_tuple):
        raise ValueError("query and embedding counts differ")
    if len(set(query_tuple)) != len(query_tuple) or len(set(candidates)) != len(candidates):
        raise ValueError("query and candidate IDs must be unique")
    if set(profiles) != set(candidates):
        raise ValueError("profiles must match the candidate inventory")
    scores = np.full((len(query_tuple), len(candidates)), -2.0, dtype=np.float32)
    winners = np.full((len(query_tuple), len(candidates)), -1, dtype=np.int32)
    contributors = np.full(
        (len(query_tuple), len(candidates), top_n), -1, dtype=np.int32
    )
    available = np.zeros(len(candidates), dtype=bool)
    for candidate_index, candidate_id in enumerate(candidates):
        profile = profiles[candidate_id]
        if profile is None:
            continue
        if profile.candidate_id != candidate_id:
            raise ValueError("profile candidate ID mismatch")
        centroids = _normalize_rows(profile.centroids)
        if centroids.shape[1] != queries.shape[1]:
            raise ValueError("query and centroid dimensions differ")
        similarities = np.asarray(queries @ centroids.T, dtype=np.float32)
        centroid_ids = np.arange(centroids.shape[0], dtype=np.int32)
        contributor_count = min(top_n, centroids.shape[0])
        for query_index in range(len(query_tuple)):
            order = np.lexsort((centroid_ids, -similarities[query_index]))
            selected = np.asarray(order[:contributor_count], dtype=np.int32)
            contributors[
                query_index, candidate_index, :contributor_count
            ] = selected
            winners[query_index, candidate_index] = selected[0]
            scores[query_index, candidate_index] = np.mean(
                similarities[query_index, selected], dtype=np.float64
            )
        available[candidate_index] = True
    names = np.asarray(candidates)
    rankings: dict[str, list[str]] = {}
    for query_index, query_id in enumerate(query_tuple):
        order = np.lexsort((names, -scores[query_index], -available.astype(np.int8)))
        rankings[query_id] = [candidates[int(index)] for index in order]
    return ProfileScoreResult(scores, winners, contributors, rankings)


def aggregate_cluster_labels(
    *,
    member_question_indices: Sequence[int],
    question_label_scores: FloatArray,
    label_ids: Sequence[str],
) -> ClusterAnnotation:
    """Aggregate explanation-only contextual label similarities."""

    scores = np.asarray(question_label_scores, dtype=np.float32)
    labels = tuple(label_ids)
    members = tuple(member_question_indices)
    if not members:
        raise ValueError("at least one supporting question is required")
    if scores.ndim != 2 or scores.shape[1] != len(labels) or len(labels) < 2:
        raise ValueError("label score shape is invalid")
    if any(index < 0 or index >= scores.shape[0] for index in members):
        raise IndexError("supporting question index is out of range")
    means = np.mean(scores[np.asarray(members)], axis=0, dtype=np.float64)
    names = np.asarray(labels)
    order = np.lexsort((names, -means))
    first, second = int(order[0]), int(order[1])
    top = float(means[first])
    runner_up = float(means[second])
    margin = top - runner_up
    if top >= 0.60 and margin >= 0.02:
        primary, secondary, status = labels[first], None, "primary"
    elif top >= 0.60:
        primary, secondary, status = labels[first], labels[second], "primary_secondary"
    else:
        primary, secondary, status = None, None, "mixed_or_unclear"
    return ClusterAnnotation(
        primary, secondary, top, runner_up, margin, len(members), status, False
    )


def reciprocal_rank_fusion(
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    rankings_by_signal: Mapping[str, Mapping[str, Sequence[str]]],
    weights: Mapping[str, float],
    unavailable_profile_ids: frozenset[str] = frozenset(),
    k: int = 60,
) -> RRFResult:
    """Fuse complete fixed rankings without judgments or score calibration."""

    queries = tuple(query_ids)
    candidates = tuple(candidate_ids)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("RRF k must be a positive integer")
    if set(rankings_by_signal) != set(weights) or not weights:
        raise ValueError("fusion signals and weights must match")
    if any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("fusion weights must be positive and finite")
    candidate_set = set(candidates)
    scores = np.zeros((len(queries), len(candidates)), dtype=np.float64)
    candidate_index = {item: index for index, item in enumerate(candidates)}
    for signal, signal_rankings in rankings_by_signal.items():
        if set(signal_rankings) != set(queries):
            raise ValueError(f"signal {signal!r} query inventory mismatch")
        for query_index, query_id in enumerate(queries):
            ranking = tuple(signal_rankings[query_id])
            if len(ranking) != len(candidates) or set(ranking) != candidate_set:
                raise ValueError(f"signal {signal!r} candidate inventory mismatch")
            for rank, candidate_id in enumerate(ranking, start=1):
                if signal == "profile" and candidate_id in unavailable_profile_ids:
                    continue
                scores[query_index, candidate_index[candidate_id]] += weights[signal] / (
                    k + rank
                )
    values = np.asarray(scores, dtype=np.float32)
    names = np.asarray(candidates)
    rankings = {
        query_id: [candidates[int(index)] for index in np.lexsort((names, -values[row]))]
        for row, query_id in enumerate(queries)
    }
    return RRFResult(values, rankings)


__all__ = [
    "CandidateProfile",
    "CandidateRecordAudit",
    "CanonicalCluster",
    "ClusterAnnotation",
    "FilteredProfileRecords",
    "ProfileRecord",
    "ProfileScoreResult",
    "RRFResult",
    "aggregate_cluster_labels",
    "canonical_kmeans",
    "canonical_profile_record_id",
    "filter_and_deduplicate_records",
    "reciprocal_rank_fusion",
    "score_profiles",
]
