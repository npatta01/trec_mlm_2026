from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
import os
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from trec_mlm.dense_aggregation_hashing import (
    canonical_json,
    canonical_membership_sha256,
    framed_sha256,
    logical_array_sha256,
)
from trec_mlm.dense_aggregation_runtime import THREAD_ENVIRONMENT


_KMEANS_ARGUMENTS = {
    "init": "k-means++",
    "random_state": 2026,
    "n_init": 10,
    "algorithm": "lloyd",
    "max_iter": 300,
    "tol": 1e-4,
}
_THREADPOOL_KEYS = (
    "user_api",
    "internal_api",
    "prefix",
    "filepath",
    "version",
    "num_threads",
)
_THREADPOOL_SORT_KEYS = _THREADPOOL_KEYS[:5]


@dataclass(frozen=True, slots=True)
class FitKey:
    support_query_ids: tuple[str, ...]
    cluster_count: int
    embedding_sha256: str
    clustering_contract_sha256: str


@dataclass(slots=True)
class CanonicalClusterFit:
    key: FitKey
    centers: NDArray[np.float32]
    labels: NDArray[np.uint16]
    memberships: tuple[tuple[str, ...], ...]
    fit_membership_sha256: str
    cluster_membership_sha256: tuple[str, ...]
    center_sha256: str
    cluster_inertia: tuple[float, ...]
    sklearn_inertia: float
    recomputed_total_inertia: float
    n_iter: int


@dataclass(frozen=True, slots=True)
class TargetAssignment:
    query_id: str
    canonical_cluster_id: int
    squared_distance: float
    member_indices: tuple[int, ...]
    cluster_support: int


@dataclass(frozen=True, slots=True)
class NumericalEnvironmentReceipt:
    thread_environment: Mapping[str, str]
    threadpool_info: tuple[Mapping[str, str | int | None], ...]
    threadpool_info_sha256: str
    schema: str = "trec-mlm-dense-aggregation-numerical-environment-v1"


def _canonical_threadpool_row(
    row: Mapping[str, object],
) -> dict[str, str | int | None]:
    canonical: dict[str, str | int | None] = {}
    for key in _THREADPOOL_KEYS[:4]:
        value = row.get(key)
        if type(value) is not str:
            raise RuntimeError(f"threadpool_info {key} must be a string")
        canonical[key] = value
    version = row.get("version")
    if version is not None and type(version) is not str:
        raise RuntimeError("threadpool_info version must be a string or null")
    canonical["version"] = version
    num_threads = row.get("num_threads")
    if type(num_threads) is not int or num_threads != 1:
        raise RuntimeError(
            "threadpool_info num_threads must be the exact integer 1"
        )
    canonical["num_threads"] = num_threads
    return canonical


def _threadpool_sort_key(
    row: Mapping[str, str | int | None],
) -> tuple[tuple[bool, str], ...]:
    return tuple(
        (value is not None, "" if value is None else str(value))
        for value in (row[key] for key in _THREADPOOL_SORT_KEYS)
    )


def capture_numerical_environment() -> NumericalEnvironmentReceipt:
    with threadpool_limits(limits=1):
        thread_environment = {
            name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT
        }
        for name, value in thread_environment.items():
            if value != "1":
                raise RuntimeError(
                    f"{name} must equal '1' in the numerical environment"
                )
        canonical_rows = tuple(
            sorted(
                (_canonical_threadpool_row(row) for row in threadpool_info()),
                key=_threadpool_sort_key,
            )
        )
    return NumericalEnvironmentReceipt(
        thread_environment=thread_environment,
        threadpool_info=canonical_rows,
        threadpool_info_sha256=framed_sha256(
            "threadpool-info", canonical_json(canonical_rows)
        ),
    )


def require_same_numerical_environment(
    expected: NumericalEnvironmentReceipt,
) -> None:
    observed = capture_numerical_environment()
    if observed != expected:
        raise RuntimeError("numerical environment changed during worker execution")


def numerical_environment_file_bytes(
    receipt: NumericalEnvironmentReceipt,
) -> bytes:
    return canonical_json(asdict(receipt)) + b"\n"


def _validate_embedding_query_inventory(
    embeddings: NDArray[np.float32], query_ids: Sequence[str]
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    embedding_array = np.asarray(embeddings)
    if embedding_array.ndim != 2:
        raise ValueError("embeddings must be exactly two-dimensional")
    canonical_query_ids = tuple(query_ids)
    if len(canonical_query_ids) != embedding_array.shape[0]:
        raise ValueError("query_ids length must match embedding rows")
    if (
        not canonical_query_ids
        or any(
            type(query_id) is not str or not query_id
            for query_id in canonical_query_ids
        )
        or len(set(canonical_query_ids)) != len(canonical_query_ids)
    ):
        raise ValueError("query_ids must contain nonempty unique IDs")
    return embedding_array, canonical_query_ids


def _validate_cluster_count(cluster_count: object) -> int:
    if type(cluster_count) is not int or not 1 <= cluster_count <= 65_536:
        raise ValueError("cluster_count must be an exact int in 1..65536")
    return cluster_count


def _validate_indices(
    indices: Sequence[int],
    *,
    inventory_size: int,
    kind: str,
) -> tuple[int, ...]:
    values = tuple(indices)
    if any(type(index) is not int for index in values):
        raise ValueError(f"{kind} indices must be exact ints")
    if len(set(values)) != len(values):
        raise ValueError(f"{kind} indices must be unique")
    if any(index < 0 or index >= inventory_size for index in values):
        raise ValueError(f"{kind} index out of range")
    return values


def fit_canonical_clusters(
    *,
    embeddings: NDArray[np.float32],
    query_ids: Sequence[str],
    support_indices: Sequence[int],
    cluster_count: int,
    embedding_sha256: str,
    clustering_contract_sha256: str,
) -> CanonicalClusterFit:
    embedding_array, canonical_query_ids = _validate_embedding_query_inventory(
        embeddings, query_ids
    )
    validated_cluster_count = _validate_cluster_count(cluster_count)
    validated_support = _validate_indices(
        support_indices,
        inventory_size=embedding_array.shape[0],
        kind="support",
    )
    support = tuple(
        sorted(validated_support, key=lambda index: canonical_query_ids[index])
    )
    support_ids = tuple(canonical_query_ids[index] for index in support)
    rows = np.ascontiguousarray(
        embedding_array[np.asarray(support, dtype=np.intp)], dtype=np.dtype("<f4")
    )
    if rows.shape[0] < validated_cluster_count:
        raise ValueError("cluster support has fewer rows than C")
    if not np.isfinite(rows).all() or np.any(
        np.linalg.norm(rows, axis=1) == 0.0
    ):
        raise ValueError("cluster support contains a non-finite or zero row")
    if len({row.tobytes() for row in rows}) < validated_cluster_count:
        raise ValueError("cluster support has fewer than C distinct float32 rows")

    estimator = KMeans(n_clusters=validated_cluster_count, **_KMEANS_ARGUMENTS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            raw_labels = estimator.fit_predict(rows)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("K-means emitted a convergence warning")
    if not 1 <= int(estimator.n_iter_) <= 299:
        raise RuntimeError(
            f"K-means n_iter_ outside 1..299: {estimator.n_iter_}"
        )

    raw_label_array = np.asarray(raw_labels)
    if (
        raw_label_array.ndim != 1
        or raw_label_array.shape[0] != rows.shape[0]
        or np.issubdtype(raw_label_array.dtype, np.bool_)
        or not np.issubdtype(raw_label_array.dtype, np.integer)
    ):
        raise RuntimeError(
            "K-means labels must be a one-dimensional integer vector "
            f"of support length {rows.shape[0]}"
        )
    original_labels = np.ascontiguousarray(raw_label_array, dtype=np.int64)
    present = tuple(sorted(set(int(label) for label in original_labels)))
    expected_labels = set(range(validated_cluster_count))
    if not set(present).issubset(expected_labels):
        raise RuntimeError("K-means labels must be exactly range(cluster_count)")
    if len(present) != validated_cluster_count:
        raise RuntimeError("K-means did not occupy every cluster")
    raw_centers = np.asarray(estimator.cluster_centers_)
    expected_center_shape = (validated_cluster_count, rows.shape[1])
    if raw_centers.shape != expected_center_shape:
        raise RuntimeError(
            "K-means cluster_centers_ must have shape "
            f"{expected_center_shape}"
        )
    original_members = {
        label: tuple(
            index
            for index, assigned in enumerate(original_labels)
            if assigned == label
        )
        for label in present
    }
    canonical_original = tuple(
        sorted(
            present,
            key=lambda label: min(
                support_ids[index] for index in original_members[label]
            ),
        )
    )
    remap = {
        original: canonical
        for canonical, original in enumerate(canonical_original)
    }
    labels = np.asarray(
        [remap[int(label)] for label in original_labels], dtype=np.dtype("<u2")
    )
    centers = np.ascontiguousarray(
        np.asarray(raw_centers, dtype=np.float32)[list(canonical_original)],
        dtype=np.dtype("<f4"),
    )
    if not np.isfinite(centers).all():
        raise RuntimeError("K-means produced a non-finite center")

    memberships = tuple(
        tuple(
            support_ids[index]
            for index, label in enumerate(labels)
            if int(label) == cluster_id
        )
        for cluster_id in range(validated_cluster_count)
    )
    inertia_values: list[float] = []
    for cluster_id in range(validated_cluster_count):
        member_delta = np.ascontiguousarray(
            rows[np.flatnonzero(labels == cluster_id)], dtype=np.float64
        ) - np.ascontiguousarray(centers[cluster_id], dtype=np.float64)
        inertia_values.append(
            float(np.sum(member_delta * member_delta, dtype=np.float64))
        )
    cluster_inertia = tuple(inertia_values)
    if not all(
        math.isfinite(value) and value >= 0.0 for value in cluster_inertia
    ):
        raise RuntimeError("canonical cluster inertia is non-finite or negative")
    recomputed = math.fsum(cluster_inertia)
    if not math.isclose(
        float(estimator.inertia_),
        recomputed,
        rel_tol=1e-5,
        abs_tol=1e-8,
    ):
        raise RuntimeError("sklearn inertia differs from canonical recomputation")

    key = FitKey(
        support_ids,
        validated_cluster_count,
        embedding_sha256,
        clustering_contract_sha256,
    )
    return CanonicalClusterFit(
        key=key,
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=canonical_membership_sha256(
            tuple(enumerate(memberships))
        ),
        cluster_membership_sha256=tuple(
            canonical_membership_sha256(
                ((cluster_id, memberships[cluster_id]),)
            )
            for cluster_id in range(validated_cluster_count)
        ),
        center_sha256=logical_array_sha256(
            centers,
            tag="canonical-centers",
            dtype="<f4",
            row_inventory=tuple(range(validated_cluster_count)),
            column_inventory=(),
        ),
        cluster_inertia=cluster_inertia,
        sklearn_inertia=float(estimator.inertia_),
        recomputed_total_inertia=recomputed,
        n_iter=int(estimator.n_iter_),
    )


def assign_targets(
    *,
    fit: CanonicalClusterFit,
    embeddings: NDArray[np.float32],
    query_ids: Sequence[str],
    target_indices: Sequence[int],
    support_index_by_id: Mapping[str, int],
) -> tuple[TargetAssignment, ...]:
    embedding_array, canonical_query_ids = _validate_embedding_query_inventory(
        embeddings, query_ids
    )
    cluster_count = _validate_cluster_count(fit.key.cluster_count)
    support_ids = tuple(fit.key.support_query_ids)
    expected_center_shape = (cluster_count, embedding_array.shape[1])
    center_array = np.asarray(fit.centers)
    if center_array.shape != expected_center_shape:
        raise ValueError(f"fit centers must have shape {expected_center_shape}")

    label_array = np.asarray(fit.labels)
    if label_array.ndim != 1 or label_array.shape[0] != len(support_ids):
        raise ValueError(
            "fit labels must be a one-dimensional vector of support length "
            f"{len(support_ids)}"
        )
    if (
        np.issubdtype(label_array.dtype, np.bool_)
        or not np.issubdtype(label_array.dtype, np.integer)
        or any(
            int(label) < 0 or int(label) >= cluster_count
            for label in label_array
        )
    ):
        raise ValueError("fit labels contain invalid canonical cluster IDs")

    try:
        memberships = tuple(tuple(membership) for membership in fit.memberships)
    except TypeError as error:
        raise ValueError("fit memberships must partition fit support IDs") from error
    if len(memberships) != cluster_count:
        raise ValueError("fit memberships count must equal cluster_count")
    if any(not membership for membership in memberships):
        raise ValueError("fit memberships must not contain an empty cluster")
    flattened_memberships = tuple(
        query_id for membership in memberships for query_id in membership
    )
    valid_support_partition = (
        bool(support_ids)
        and all(
            type(query_id) is str and bool(query_id)
            for query_id in support_ids
        )
        and len(set(support_ids)) == len(support_ids)
        and all(
            type(query_id) is str and bool(query_id)
            for query_id in flattened_memberships
        )
        and len(flattened_memberships) == len(support_ids)
        and len(set(flattened_memberships)) == len(flattened_memberships)
        and set(flattened_memberships) == set(support_ids)
    )
    if not valid_support_partition:
        raise ValueError("fit memberships must partition fit support IDs")
    expected_memberships = tuple(
        tuple(
            support_ids[index]
            for index, label in enumerate(label_array)
            if int(label) == cluster_id
        )
        for cluster_id in range(cluster_count)
    )
    if memberships != expected_memberships:
        raise ValueError("fit memberships are inconsistent with labels")

    mapping_keys = tuple(support_index_by_id)
    if len(mapping_keys) != len(support_ids) or set(mapping_keys) != set(
        support_ids
    ):
        raise ValueError(
            "support_index_by_id keys must exactly match fit support IDs"
        )
    mapping_values = tuple(
        support_index_by_id[query_id] for query_id in support_ids
    )
    if any(type(index) is not int for index in mapping_values):
        raise ValueError("support_index_by_id values must be exact ints")
    if len(set(mapping_values)) != len(mapping_values):
        raise ValueError("support_index_by_id values must be unique")
    if any(
        index < 0 or index >= embedding_array.shape[0]
        for index in mapping_values
    ):
        raise ValueError("support_index_by_id index out of range")
    if any(
        canonical_query_ids[index] != query_id
        for query_id, index in zip(support_ids, mapping_values, strict=True)
    ):
        raise ValueError(
            "support_index_by_id must map each support ID to the matching query"
        )

    validated_targets = _validate_indices(
        target_indices,
        inventory_size=embedding_array.shape[0],
        kind="target",
    )
    support_id_set = set(support_ids)
    if any(
        canonical_query_ids[target_index] in support_id_set
        for target_index in validated_targets
    ):
        raise ValueError("target query IDs must be disjoint from fit support IDs")

    centers = np.ascontiguousarray(center_array, dtype=np.float64)
    assignments: list[TargetAssignment] = []
    for target_index in validated_targets:
        target = np.ascontiguousarray(
            embedding_array[target_index], dtype=np.float64
        )
        if not np.isfinite(target).all():
            raise ValueError("target embedding contains a non-finite value")
        delta = centers - target
        distances = np.asarray(
            [
                np.sum(delta[index] * delta[index], dtype=np.float64)
                for index in range(cluster_count)
            ],
            dtype=np.float64,
        )
        if not np.isfinite(distances).all():
            raise RuntimeError("target-to-center distance is non-finite")
        cluster_id = int(np.argmin(distances))
        member_indices = tuple(
            support_index_by_id[query_id]
            for query_id in memberships[cluster_id]
        )
        assignments.append(
            TargetAssignment(
                query_id=canonical_query_ids[target_index],
                canonical_cluster_id=cluster_id,
                squared_distance=float(distances[cluster_id]),
                member_indices=member_indices,
                cluster_support=len(member_indices),
            )
        )
    return tuple(assignments)
