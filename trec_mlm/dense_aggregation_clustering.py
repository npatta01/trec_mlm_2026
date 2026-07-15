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


def fit_canonical_clusters(
    *,
    embeddings: NDArray[np.float32],
    query_ids: Sequence[str],
    support_indices: Sequence[int],
    cluster_count: int,
    embedding_sha256: str,
    clustering_contract_sha256: str,
) -> CanonicalClusterFit:
    support = tuple(sorted(support_indices, key=lambda index: query_ids[index]))
    support_ids = tuple(query_ids[index] for index in support)
    rows = np.ascontiguousarray(
        embeddings[np.asarray(support, dtype=np.intp)], dtype=np.dtype("<f4")
    )
    if rows.shape[0] < cluster_count:
        raise ValueError("cluster support has fewer rows than C")
    if not np.isfinite(rows).all() or np.any(
        np.linalg.norm(rows, axis=1) == 0.0
    ):
        raise ValueError("cluster support contains a non-finite or zero row")
    if len({row.tobytes() for row in rows}) < cluster_count:
        raise ValueError("cluster support has fewer than C distinct float32 rows")

    estimator = KMeans(n_clusters=cluster_count, **_KMEANS_ARGUMENTS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            original_labels = np.asarray(
                estimator.fit_predict(rows), dtype=np.int64
            )
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("K-means emitted a convergence warning")
    if not 1 <= int(estimator.n_iter_) <= 299:
        raise RuntimeError(
            f"K-means n_iter_ outside 1..299: {estimator.n_iter_}"
        )

    present = tuple(sorted(set(int(label) for label in original_labels)))
    if len(present) != cluster_count:
        raise RuntimeError("K-means did not occupy every cluster")
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
        np.asarray(estimator.cluster_centers_, dtype=np.float32)[
            list(canonical_original)
        ],
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
        for cluster_id in range(cluster_count)
    )
    inertia_values: list[float] = []
    for cluster_id in range(cluster_count):
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
        cluster_count,
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
            for cluster_id in range(cluster_count)
        ),
        center_sha256=logical_array_sha256(
            centers,
            tag="canonical-centers",
            dtype="<f4",
            row_inventory=tuple(range(cluster_count)),
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
    centers = np.ascontiguousarray(fit.centers, dtype=np.float64)
    assignments: list[TargetAssignment] = []
    for target_index in target_indices:
        target = np.ascontiguousarray(
            embeddings[target_index], dtype=np.float64
        )
        if not np.isfinite(target).all():
            raise ValueError("target embedding contains a non-finite value")
        delta = centers - target
        distances = np.asarray(
            [
                np.sum(delta[index] * delta[index], dtype=np.float64)
                for index in range(fit.key.cluster_count)
            ],
            dtype=np.float64,
        )
        if not np.isfinite(distances).all():
            raise RuntimeError("target-to-center distance is non-finite")
        cluster_id = int(np.argmin(distances))
        member_indices = tuple(
            support_index_by_id[query_id]
            for query_id in fit.memberships[cluster_id]
        )
        assignments.append(
            TargetAssignment(
                query_id=query_ids[target_index],
                canonical_cluster_id=cluster_id,
                squared_distance=float(distances[cluster_id]),
                member_indices=member_indices,
                cluster_support=len(member_indices),
            )
        )
    return tuple(assignments)
