from __future__ import annotations

from collections.abc import Callable
import copy
import dataclasses
from dataclasses import FrozenInstanceError
import errno
import gc
import hashlib
import inspect
from itertools import combinations
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any, Literal, get_type_hints
import weakref

import numpy as np
import pytest

from trec_mlm.dense_aggregation import (
    DenseAggregationCandidate,
    combine_and_rank,
    domain_scores,
)
import trec_mlm.dense_aggregation_cache as cache_module
from trec_mlm.dense_aggregation_cache import (
    CACHE_HEADER_KEYS,
    CACHE_HEADER_SCHEMA,
    StoredFitRecord,
    StreamingFitCache,
    _fit_key_sha256,
)
from trec_mlm.dense_aggregation_clustering import (
    CanonicalClusterFit,
    FitKey,
    TargetAssignment,
)


_HEADER_SCHEMA = "trec-mlm-dense-aggregation-fit-cache-header-v1"
_HEADER_KEYS = {
    "schema",
    "query_ids",
    "query_input_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "clustering_contract_sha256",
    "implementation_inventory_sha256",
    "dependency_inventory_sha256",
    "runtime_inventory_sha256",
    "numerical_environment_sha256",
    "shard_schemas",
    "expected_inventory",
    "expected_counts",
}
_HASH_FIELDS = (
    "query_input_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "clustering_contract_sha256",
    "implementation_inventory_sha256",
    "dependency_inventory_sha256",
    "runtime_inventory_sha256",
    "numerical_environment_sha256",
)
_SHARD_SCHEMAS = {
    "centers": {
        "path": "centers.f4",
        "kind": "binary",
        "dtype": "<f4",
        "layout": "C",
    },
    "labels": {
        "path": "labels.u2",
        "kind": "binary",
        "dtype": "<u2",
        "layout": "C",
    },
    "cluster_inertia": {
        "path": "cluster_inertia.f8",
        "kind": "binary",
        "dtype": "<f8",
        "layout": "C",
    },
    "fit_receipts": {
        "path": "fit_receipts.jsonl",
        "kind": "canonical-jsonl",
        "record_schema": "trec-mlm-dense-aggregation-fit-receipt-v1",
    },
    "timings": {
        "path": "timings.jsonl",
        "kind": "canonical-jsonl",
        "record_schema": "trec-mlm-dense-aggregation-fit-timing-v1",
    },
}
_STREAM_PATHS = {
    "centers.f4",
    "labels.u2",
    "cluster_inertia.f8",
    "fit_receipts.jsonl",
    "timings.jsonl",
}
_QUERY_IDS = ("q1", "q2", "q3", "q4")
_SYNTHETIC_EMBEDDINGS = np.ascontiguousarray(
    np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [11.0, 1.0, 1.0],
            [11.0, 11.0, 1.0],
        ],
        dtype="<f4",
    )
)
_CANONICAL_INVENTORY = [
    {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 1},
    {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
    {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    {"exclusion_kind": "single", "inventory_index": 0, "cluster_count": 1},
    {"exclusion_kind": "single", "inventory_index": 0, "cluster_count": 2},
    {"exclusion_kind": "single", "inventory_index": 1, "cluster_count": 2},
]
_INVENTORY_KEYS = {"exclusion_kind", "inventory_index", "cluster_count"}
_COUNT_KEYS = {
    "pair_fit_count",
    "single_fit_count",
    "fit_count",
    "start_count",
    "timing_record_count",
}
_TIMING_KEYS = {
    "schema",
    "sequence",
    "exclusion_kind",
    "inventory_index",
    "cluster_count",
    "elapsed_ns",
    "n_iter",
}
_SHARD_SCHEMA_LEAVES = tuple(
    (shard_name, field)
    for shard_name, schema in _SHARD_SCHEMAS.items()
    for field in schema
)


def _valid_header(
    *, inventory: list[dict[str, object]] | None = None
) -> dict[str, Any]:
    rows = (
        [{"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2}]
        if inventory is None
        else copy.deepcopy(inventory)
    )
    pair_count = sum(row["exclusion_kind"] == "pair" for row in rows)
    single_count = len(rows) - pair_count
    fit_count = len(rows)
    return {
        "schema": _HEADER_SCHEMA,
        "query_ids": list(_QUERY_IDS),
        "query_input_sha256": "0" * 64,
        "embedding_sha256": "1" * 64,
        "similarity_sha256": "2" * 64,
        "clustering_contract_sha256": "3" * 64,
        "implementation_inventory_sha256": "4" * 64,
        "dependency_inventory_sha256": "5" * 64,
        "runtime_inventory_sha256": "6" * 64,
        "numerical_environment_sha256": "7" * 64,
        "shard_schemas": copy.deepcopy(_SHARD_SCHEMAS),
        "expected_inventory": rows,
        "expected_counts": {
            "pair_fit_count": pair_count,
            "single_fit_count": single_count,
            "fit_count": fit_count,
            "start_count": 10 * fit_count,
            "timing_record_count": fit_count,
        },
    }


def _independent_canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _independent_framed_sha256(tag: str, *payloads: bytes) -> str:
    digest = hashlib.sha256(b"trec-mlm-dense-aggregation-hash-v1\0")
    for payload in (tag.encode("ascii"), *payloads):
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _independent_center_sha256(centers: np.ndarray[Any, Any]) -> str:
    value = np.ascontiguousarray(centers, dtype="<f4")
    return _independent_framed_sha256(
        "canonical-centers",
        value.dtype.str.encode("ascii"),
        _independent_canonical_json(list(value.shape)),
        _independent_canonical_json(list(range(value.shape[0]))),
        _independent_canonical_json([]),
        value.tobytes(order="C"),
    )


def _independent_membership_sha256(
    rows: tuple[tuple[int, tuple[str, ...]], ...],
) -> str:
    payload = [[cluster_id, list(query_ids)] for cluster_id, query_ids in rows]
    return _independent_framed_sha256(
        "canonical-membership", _independent_canonical_json(payload)
    )


def _independent_support_sha256(support_query_ids: tuple[str, ...]) -> str:
    return _independent_framed_sha256(
        "support-query-ids",
        _independent_canonical_json(list(support_query_ids)),
    )


def _independent_fit_key_sha256(key: FitKey) -> str:
    return _independent_framed_sha256(
        "fit-cache-key",
        _independent_canonical_json(
            {
                "schema": "trec-mlm-dense-aggregation-fit-key-v1",
                "support_query_ids": list(key.support_query_ids),
                "cluster_count": key.cluster_count,
                "embedding_sha256": key.embedding_sha256,
                "clustering_contract_sha256": key.clustering_contract_sha256,
            }
        ),
    )


def _pair_excluded_query_ids(inventory_index: int) -> tuple[str, str]:
    return tuple(combinations(_QUERY_IDS, 2))[inventory_index]


def _support_without(excluded_query_ids: tuple[str, ...]) -> tuple[str, ...]:
    excluded = set(excluded_query_ids)
    return tuple(query_id for query_id in _QUERY_IDS if query_id not in excluded)


def _synthetic_target_assignments(
    *,
    excluded_query_ids: tuple[str, ...],
    centers: np.ndarray[Any, Any],
    memberships: tuple[tuple[str, ...], ...],
) -> tuple[TargetAssignment, ...]:
    centers64 = np.ascontiguousarray(centers, dtype=np.float64)
    assignments: list[TargetAssignment] = []
    for query_id in excluded_query_ids:
        target64 = np.ascontiguousarray(
            _SYNTHETIC_EMBEDDINGS[_QUERY_IDS.index(query_id)],
            dtype=np.float64,
        )
        distances = tuple(
            float(
                np.sum(
                    (centers64[cluster_id] - target64) ** 2,
                    dtype=np.float64,
                )
            )
            for cluster_id in range(len(memberships))
        )
        canonical_cluster_id = min(
            range(len(distances)), key=lambda cluster_id: distances[cluster_id]
        )
        member_indices = tuple(
            _QUERY_IDS.index(member_id)
            for member_id in memberships[canonical_cluster_id]
        )
        assignments.append(
            TargetAssignment(
                query_id=query_id,
                canonical_cluster_id=canonical_cluster_id,
                squared_distance=distances[canonical_cluster_id],
                member_indices=member_indices,
                cluster_support=len(member_indices),
            )
        )
    return tuple(assignments)


def _synthetic_pair_fit(
    inventory_index: int,
) -> tuple[
    CanonicalClusterFit,
    tuple[TargetAssignment, ...],
    tuple[str, str],
]:
    excluded_query_ids = _pair_excluded_query_ids(inventory_index)
    support_query_ids = _support_without(excluded_query_ids)
    centers = np.ascontiguousarray(
        _SYNTHETIC_EMBEDDINGS[
            np.asarray(
                tuple(_QUERY_IDS.index(query_id) for query_id in support_query_ids)
            )
        ],
        dtype="<f4",
    )
    labels = np.ascontiguousarray(np.asarray([0, 1], dtype="<u2"))
    memberships = ((support_query_ids[0],), (support_query_ids[1],))
    cluster_inertia = (0.0, 0.0)
    membership_rows = tuple(
        (cluster_id, memberships[cluster_id]) for cluster_id in range(2)
    )
    total_inertia = math.fsum(cluster_inertia)
    fit = CanonicalClusterFit(
        key=FitKey(
            support_query_ids,
            2,
            "1" * 64,
            "3" * 64,
        ),
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=_independent_membership_sha256(membership_rows),
        cluster_membership_sha256=tuple(
            _independent_membership_sha256((membership_rows[cluster_id],))
            for cluster_id in range(2)
        ),
        center_sha256=_independent_center_sha256(centers),
        cluster_inertia=cluster_inertia,
        sklearn_inertia=total_inertia,
        recomputed_total_inertia=total_inertia,
        n_iter=1 + inventory_index,
    )
    assignments = _synthetic_target_assignments(
        excluded_query_ids=excluded_query_ids,
        centers=centers,
        memberships=memberships,
    )
    return fit, assignments, excluded_query_ids


def _synthetic_c1_pair_fit() -> tuple[
    CanonicalClusterFit,
    tuple[TargetAssignment, ...],
    tuple[str, str],
]:
    excluded_query_ids = _pair_excluded_query_ids(0)
    support_query_ids = _support_without(excluded_query_ids)
    support_rows = np.ascontiguousarray(
        _SYNTHETIC_EMBEDDINGS[
            np.asarray(
                tuple(_QUERY_IDS.index(query_id) for query_id in support_query_ids)
            )
        ],
        dtype=np.float64,
    )
    centers = np.ascontiguousarray(
        np.asarray([support_rows.mean(axis=0, dtype=np.float64)], dtype="<f4")
    )
    labels = np.ascontiguousarray(np.asarray([0, 0], dtype="<u2"))
    memberships = (support_query_ids,)
    center64 = np.ascontiguousarray(centers[0], dtype=np.float64)
    cluster_inertia = (
        float(
            np.sum(
                (support_rows - center64) ** 2,
                dtype=np.float64,
            )
        ),
    )
    membership_rows = ((0, support_query_ids),)
    total_inertia = math.fsum(cluster_inertia)
    fit = CanonicalClusterFit(
        key=FitKey(support_query_ids, 1, "1" * 64, "3" * 64),
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=_independent_membership_sha256(membership_rows),
        cluster_membership_sha256=(_independent_membership_sha256(membership_rows),),
        center_sha256=_independent_center_sha256(centers),
        cluster_inertia=cluster_inertia,
        sklearn_inertia=total_inertia,
        recomputed_total_inertia=total_inertia,
        n_iter=2,
    )
    assignments = _synthetic_target_assignments(
        excluded_query_ids=excluded_query_ids,
        centers=centers,
        memberships=memberships,
    )
    return fit, assignments, excluded_query_ids


def _synthetic_single_c1_fit(
    inventory_index: int = 1,
) -> tuple[
    CanonicalClusterFit,
    tuple[TargetAssignment, ...],
    tuple[str],
]:
    excluded_query_ids = (_QUERY_IDS[inventory_index],)
    support_query_ids = _support_without(excluded_query_ids)
    support_rows = np.ascontiguousarray(
        _SYNTHETIC_EMBEDDINGS[
            np.asarray(
                tuple(_QUERY_IDS.index(query_id) for query_id in support_query_ids)
            )
        ],
        dtype=np.float64,
    )
    centers = np.ascontiguousarray(
        np.asarray([support_rows.mean(axis=0, dtype=np.float64)], dtype="<f4")
    )
    labels = np.ascontiguousarray(np.zeros(len(support_query_ids), dtype="<u2"))
    memberships = (support_query_ids,)
    center64 = np.ascontiguousarray(centers[0], dtype=np.float64)
    cluster_inertia = (float(np.sum((support_rows - center64) ** 2, dtype=np.float64)),)
    membership_rows = ((0, support_query_ids),)
    total_inertia = math.fsum(cluster_inertia)
    fit = CanonicalClusterFit(
        key=FitKey(support_query_ids, 1, "1" * 64, "3" * 64),
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=_independent_membership_sha256(membership_rows),
        cluster_membership_sha256=(_independent_membership_sha256(membership_rows),),
        center_sha256=_independent_center_sha256(centers),
        cluster_inertia=cluster_inertia,
        sklearn_inertia=total_inertia,
        recomputed_total_inertia=total_inertia,
        n_iter=3,
    )
    assignments = _synthetic_target_assignments(
        excluded_query_ids=excluded_query_ids,
        centers=centers,
        memberships=memberships,
    )
    return fit, assignments, excluded_query_ids


def _synthetic_reordered_support_pair_fit() -> tuple[
    CanonicalClusterFit,
    tuple[TargetAssignment, ...],
    tuple[str, str],
]:
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    support_query_ids = tuple(reversed(fit.key.support_query_ids))
    memberships = ((support_query_ids[0],), (support_query_ids[1],))
    membership_rows = ((0, memberships[0]), (1, memberships[1]))
    reordered = dataclasses.replace(
        fit,
        key=dataclasses.replace(fit.key, support_query_ids=support_query_ids),
        memberships=memberships,
        fit_membership_sha256=_independent_membership_sha256(membership_rows),
        cluster_membership_sha256=tuple(
            _independent_membership_sha256((row,)) for row in membership_rows
        ),
    )
    reordered_assignments = tuple(
        dataclasses.replace(
            assignment,
            member_indices=(_QUERY_IDS.index(support_query_ids[cluster_id]),),
        )
        for cluster_id, assignment in enumerate(assignments)
    )
    return reordered, reordered_assignments, excluded_query_ids


def _expected_fit_receipt(
    *,
    fit: CanonicalClusterFit,
    assignments: tuple[TargetAssignment, ...],
    inventory_index: int,
    excluded_query_ids: tuple[str, ...],
    sequence: int,
    center_offset: int,
    label_offset: int,
    inertia_offset: int,
) -> dict[str, object]:
    centers = np.ascontiguousarray(fit.centers, dtype="<f4")
    labels = np.ascontiguousarray(fit.labels, dtype="<u2")
    inertia = np.ascontiguousarray(fit.cluster_inertia, dtype="<f8")
    membership_rows = tuple(
        (cluster_id, fit.memberships[cluster_id])
        for cluster_id in range(fit.key.cluster_count)
    )
    record_without_hash: dict[str, object] = {
        "schema": "trec-mlm-dense-aggregation-fit-receipt-v1",
        "sequence": sequence,
        "exclusion_kind": "pair",
        "inventory_index": inventory_index,
        "excluded_query_ids": list(excluded_query_ids),
        "cluster_count": fit.key.cluster_count,
        "fit_key_sha256": _independent_fit_key_sha256(fit.key),
        "support_query_ids_sha256": _independent_support_sha256(
            fit.key.support_query_ids
        ),
        "embedding_sha256": fit.key.embedding_sha256,
        "clustering_contract_sha256": fit.key.clustering_contract_sha256,
        "centers": {
            "offset": center_offset,
            "bytes": centers.nbytes,
            "shape": list(centers.shape),
            "dtype": "<f4",
        },
        "labels": {
            "offset": label_offset,
            "bytes": labels.nbytes,
            "shape": list(labels.shape),
            "dtype": "<u2",
        },
        "cluster_inertia": {
            "offset": inertia_offset,
            "bytes": inertia.nbytes,
            "shape": [len(fit.cluster_inertia)],
            "dtype": "<f8",
        },
        "center_sha256": _independent_center_sha256(centers),
        "fit_membership_sha256": _independent_membership_sha256(membership_rows),
        "cluster_membership_sha256": [
            _independent_membership_sha256((membership_rows[cluster_id],))
            for cluster_id in range(fit.key.cluster_count)
        ],
        "cluster_inertia_hex": [value.hex() for value in fit.cluster_inertia],
        "sklearn_inertia_hex": fit.sklearn_inertia.hex(),
        "recomputed_total_inertia_hex": fit.recomputed_total_inertia.hex(),
        "n_iter": fit.n_iter,
        "assignments": [
            {
                "query_id": assignment.query_id,
                "canonical_cluster_id": assignment.canonical_cluster_id,
                "squared_distance_hex": assignment.squared_distance.hex(),
                "cluster_support": assignment.cluster_support,
            }
            for assignment in assignments
        ],
        "fit_count_increment": 1,
        "start_count_increment": 10,
    }
    return {
        **record_without_hash,
        "record_sha256": _independent_framed_sha256(
            "fit-cache-record",
            _independent_canonical_json(record_without_hash),
        ),
    }


def _fit_record_semantics(
    *,
    fit: CanonicalClusterFit,
    assignments: tuple[TargetAssignment, ...],
    exclusion_kind: str,
    inventory_index: int,
    excluded_query_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "key": fit.key,
        "centers_shape": fit.centers.shape,
        "centers_dtype": fit.centers.dtype.str,
        "centers_c_contiguous": fit.centers.flags.c_contiguous,
        "centers_bytes": fit.centers.tobytes(order="C"),
        "labels_shape": fit.labels.shape,
        "labels_dtype": fit.labels.dtype.str,
        "labels_c_contiguous": fit.labels.flags.c_contiguous,
        "labels_bytes": fit.labels.tobytes(order="C"),
        "memberships": fit.memberships,
        "fit_membership_sha256": fit.fit_membership_sha256,
        "cluster_membership_sha256": fit.cluster_membership_sha256,
        "center_sha256": fit.center_sha256,
        "cluster_inertia_hex": tuple(value.hex() for value in fit.cluster_inertia),
        "sklearn_inertia_hex": fit.sklearn_inertia.hex(),
        "recomputed_total_inertia_hex": fit.recomputed_total_inertia.hex(),
        "n_iter": fit.n_iter,
        "assignments": tuple(
            (
                assignment.query_id,
                assignment.canonical_cluster_id,
                assignment.squared_distance.hex(),
                assignment.member_indices,
                assignment.cluster_support,
            )
            for assignment in assignments
        ),
        "exclusion_kind": exclusion_kind,
        "inventory_index": inventory_index,
        "excluded_query_ids": excluded_query_ids,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class _CacheMutationSnapshot:
    streams: tuple[tuple[str, bytes, int, int], ...]
    index_count: int
    resident_count: int
    fit_count: int
    timing_count: int


def _cache_stream_handles(cache: StreamingFitCache) -> tuple[tuple[str, Any], ...]:
    return (
        ("centers.f4", cache._center_handle),
        ("labels.u2", cache._label_handle),
        ("cluster_inertia.f8", cache._inertia_handle),
        ("fit_receipts.jsonl", cache._receipt_handle),
        ("timings.jsonl", cache._timing_handle),
    )


def _cache_mutation_snapshot(cache: StreamingFitCache) -> _CacheMutationSnapshot:
    handles = _cache_stream_handles(cache)
    assert all(not handle.closed for _, handle in handles)
    for _, handle in handles:
        handle.flush()
    stream_rows: list[tuple[str, bytes, int, int]] = []
    for name, handle in handles:
        path = Path(handle.name)
        payload = path.read_bytes()
        stream_rows.append((name, payload, path.stat().st_size, handle.tell()))
    return _CacheMutationSnapshot(
        streams=tuple(stream_rows),
        index_count=len(cache._index),
        resident_count=len(cache._resident),
        fit_count=cache._fit_count,
        timing_count=cache._timing_count,
    )


def _assert_append_rejected_without_side_effects(
    cache: StreamingFitCache,
    *,
    fit: CanonicalClusterFit,
    assignments: tuple[TargetAssignment, ...],
    exclusion_kind: str,
    inventory_index: int,
    excluded_query_ids: tuple[str, ...],
    match: str | None = None,
) -> None:
    before = _cache_mutation_snapshot(cache)
    header_path = Path(cache._center_handle.name).parent / "header.json"
    assert not header_path.exists()

    with pytest.raises(ValueError, match=match):
        cache.append_fit(
            fit,
            assignments,
            exclusion_kind=exclusion_kind,  # type: ignore[arg-type]
            inventory_index=inventory_index,
            excluded_query_ids=excluded_query_ids,
        )

    assert _cache_mutation_snapshot(cache) == before
    assert not header_path.exists()


@dataclasses.dataclass(frozen=True, slots=True)
class _TimingMutationSnapshot:
    payload: bytes
    file_size: int
    handle_position: int
    timing_count: int


def _assert_open_incomplete_cache(cache: StreamingFitCache) -> None:
    handles = _cache_stream_handles(cache)
    assert all(not handle.closed for _, handle in handles)
    root = Path(cache._timing_handle.name).parent
    assert not (root / "header.json").exists()


def _timing_mutation_snapshot(cache: StreamingFitCache) -> _TimingMutationSnapshot:
    _assert_open_incomplete_cache(cache)
    handle = cache._timing_handle
    handle.flush()
    path = Path(handle.name)
    return _TimingMutationSnapshot(
        payload=path.read_bytes(),
        file_size=path.stat().st_size,
        handle_position=handle.tell(),
        timing_count=cache._timing_count,
    )


def _assert_timing_rejected_without_side_effects(
    cache: StreamingFitCache,
    *,
    exclusion_kind: object,
    inventory_index: object,
    cluster_count: object,
    elapsed_ns: object,
    n_iter: object,
) -> None:
    before = _timing_mutation_snapshot(cache)

    with pytest.raises(ValueError):
        cache.append_timing(
            exclusion_kind=exclusion_kind,  # type: ignore[arg-type]
            inventory_index=inventory_index,  # type: ignore[arg-type]
            cluster_count=cluster_count,  # type: ignore[arg-type]
            elapsed_ns=elapsed_ns,  # type: ignore[arg-type]
            n_iter=n_iter,  # type: ignore[arg-type]
        )

    assert _timing_mutation_snapshot(cache) == before
    _assert_open_incomplete_cache(cache)


def _append_synthetic_pair_fit(
    cache: StreamingFitCache, inventory_index: int
) -> CanonicalClusterFit:
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(inventory_index)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=inventory_index,
        excluded_query_ids=excluded_query_ids,
    )
    return fit


def _populate_one_fit_cache(
    root: Path,
    *,
    header: dict[str, Any] | None = None,
    release: bool = False,
) -> tuple[StreamingFitCache, CanonicalClusterFit]:
    cache = StreamingFitCache.create(
        root,
        header=_valid_header() if header is None else header,
    )
    fit = _append_synthetic_pair_fit(cache, 0)
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=1_234_567,
        n_iter=fit.n_iter,
    )
    if release:
        cache.release(fit.key)
    return cache, fit


def _expected_timing_record(
    *,
    sequence: int,
    inventory_index: int,
    elapsed_ns: int,
    n_iter: int,
) -> dict[str, object]:
    return {
        "schema": "trec-mlm-dense-aggregation-fit-timing-v1",
        "sequence": sequence,
        "exclusion_kind": "pair",
        "inventory_index": inventory_index,
        "cluster_count": 2,
        "elapsed_ns": elapsed_ns,
        "n_iter": n_iter,
    }


def _signature_shape(function: object) -> tuple[tuple[str, object, bool], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default is inspect.Parameter.empty)
        for parameter in inspect.signature(  # type: ignore[arg-type]
            function
        ).parameters.values()
    )


def test_stored_fit_record_is_the_exact_frozen_slotted_public_value() -> None:
    assert dataclasses.is_dataclass(StoredFitRecord)
    assert StoredFitRecord.__dataclass_params__.frozen is True
    assert StoredFitRecord.__slots__ == (
        "fit",
        "assignments",
        "exclusion_kind",
        "inventory_index",
        "excluded_query_ids",
    )
    assert tuple(field.name for field in dataclasses.fields(StoredFitRecord)) == (
        "fit",
        "assignments",
        "exclusion_kind",
        "inventory_index",
        "excluded_query_ids",
    )
    assert get_type_hints(StoredFitRecord) == {
        "fit": CanonicalClusterFit,
        "assignments": tuple[TargetAssignment, ...],
        "exclusion_kind": Literal["pair", "single"],
        "inventory_index": int,
        "excluded_query_ids": tuple[str, ...],
    }
    record = StoredFitRecord(
        fit=None,  # type: ignore[arg-type]
        assignments=(),
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=("q1", "q2"),
    )
    with pytest.raises(FrozenInstanceError):
        record.inventory_index = 1  # type: ignore[misc]


def test_streaming_fit_cache_has_the_required_public_call_shapes() -> None:
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    required = True
    expected = {
        "create": (("root", positional, required), ("header", keyword_only, required)),
        "open_readonly": (("root", positional, required),),
        "append_fit": (
            ("self", positional, required),
            ("fit", positional, required),
            ("assignments", positional, required),
            ("exclusion_kind", keyword_only, required),
            ("inventory_index", keyword_only, required),
            ("excluded_query_ids", keyword_only, required),
        ),
        "load_fit": (("self", positional, required), ("key", positional, required)),
        "load_by_inventory": (
            ("self", positional, required),
            ("exclusion_kind", keyword_only, required),
            ("inventory_index", keyword_only, required),
            ("cluster_count", keyword_only, required),
        ),
        "release": (("self", positional, required), ("key", positional, required)),
        "append_timing": (
            ("self", positional, required),
            ("exclusion_kind", keyword_only, required),
            ("inventory_index", keyword_only, required),
            ("cluster_count", keyword_only, required),
            ("elapsed_ns", keyword_only, required),
            ("n_iter", keyword_only, required),
        ),
        "finalize": (("self", positional, required),),
        "finalize_into": (
            ("self", positional, required),
            ("destination", positional, required),
        ),
    }

    assert isinstance(StreamingFitCache.__dict__["create"], classmethod)
    assert isinstance(StreamingFitCache.__dict__["open_readonly"], classmethod)
    for name, shape in expected.items():
        assert _signature_shape(getattr(StreamingFitCache, name)) == shape


def test_literal_header_schema_keys_and_synthetic_pair_fixture() -> None:
    header = _valid_header()

    assert CACHE_HEADER_SCHEMA == _HEADER_SCHEMA
    assert CACHE_HEADER_KEYS == _HEADER_KEYS
    assert set(header) == _HEADER_KEYS
    assert header["schema"] == _HEADER_SCHEMA
    assert header["query_ids"] == list(_QUERY_IDS)
    assert header["shard_schemas"] == _SHARD_SCHEMAS
    assert header["expected_inventory"] == [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2}
    ]
    assert header["expected_counts"] == {
        "pair_fit_count": 1,
        "single_fit_count": 0,
        "fit_count": 1,
        "start_count": 10,
        "timing_record_count": 1,
    }


def test_fit_key_sha256_has_a_literal_golden() -> None:
    key = FitKey(("q1", "q2"), 20, "e" * 64, "c" * 64)

    assert _fit_key_sha256(key) == (
        "d692dce9321c67b521dc56d8a5bce4455ff6797986daee9ee4a2f3fceb2fb7a1"
    )


def test_fit_key_binds_order_c_embedding_and_contract() -> None:
    base = FitKey(("q1", "q2"), 20, "e" * 64, "c" * 64)
    mutations = (
        FitKey(("q2", "q1"), 20, "e" * 64, "c" * 64),
        FitKey(("q1", "q2"), 40, "e" * 64, "c" * 64),
        FitKey(("q1", "q2"), 20, "x" * 64, "c" * 64),
        FitKey(("q1", "q2"), 20, "e" * 64, "x" * 64),
    )

    assert len({_fit_key_sha256(base), *map(_fit_key_sha256, mutations)}) == 5


def _assert_header_rejected(tmp_path: Path, header: object) -> None:
    root = tmp_path / "fit-cache"
    original = copy.deepcopy(header)

    with pytest.raises(ValueError):
        StreamingFitCache.create(root, header=header)  # type: ignore[arg-type]

    assert not root.exists()
    assert header == original


@pytest.mark.parametrize("existing_kind", ("directory", "file"))
def test_create_requires_a_nonexistent_root(tmp_path: Path, existing_kind: str) -> None:
    root = tmp_path / "fit-cache"
    if existing_kind == "directory":
        root.mkdir()
        sentinel = root / "sentinel.bin"
        sentinel.write_bytes(b"authoritative\x00bytes")
    else:
        root.write_bytes(b"authoritative\x00bytes")

    with pytest.raises(FileExistsError):
        StreamingFitCache.create(root, header=_valid_header())

    if existing_kind == "directory":
        assert list(root.iterdir()) == [sentinel]
        assert sentinel.read_bytes() == b"authoritative\x00bytes"
    else:
        assert root.read_bytes() == b"authoritative\x00bytes"


def test_create_deep_copies_header_and_creates_only_empty_streams(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    header = _valid_header()
    expected_header = copy.deepcopy(header)

    cache = StreamingFitCache.create(root, header=header)

    assert root.is_dir()
    entries = {path.name: path for path in root.iterdir()}
    assert set(entries) == _STREAM_PATHS
    assert "header.json" not in entries
    assert all(path.is_file() and path.read_bytes() == b"" for path in entries.values())
    assert cache._header == expected_header
    assert cache._header is not header
    assert cache._header["query_ids"] is not header["query_ids"]
    assert cache._header["shard_schemas"] is not header["shard_schemas"]
    assert cache._header["expected_inventory"] is not header["expected_inventory"]
    assert cache._header["expected_counts"] is not header["expected_counts"]

    header["query_ids"].reverse()
    header["shard_schemas"]["centers"]["path"] = "mutated.f4"
    header["expected_inventory"][0]["cluster_count"] = 40
    header["expected_counts"]["fit_count"] = 99
    assert cache._header == expected_header
    del cache


@pytest.mark.parametrize("missing_key", sorted(_HEADER_KEYS))
def test_create_requires_every_exact_header_key(
    tmp_path: Path, missing_key: str
) -> None:
    header = _valid_header()
    del header[missing_key]

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize(
    "metadata_key",
    (
        "unknown_metadata",
        "absolute_path",
        "cache_root",
        "cache_dir",
        "output_root",
        "output_dir",
        "process_id",
        "worker_pid",
        "timestamp",
        "created_at",
        "elapsed_ns",
        "elapsed_seconds",
    ),
)
def test_create_rejects_unknown_and_forbidden_metadata(
    tmp_path: Path, metadata_key: str
) -> None:
    header = _valid_header()
    header[metadata_key] = "/absolute/metadata" if "path" in metadata_key else 1

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema", "trec-mlm-dense-aggregation-fit-cache-header-v2"),
        ("schema", 1),
        ("query_ids", tuple(_QUERY_IDS)),
        ("shard_schemas", []),
        ("expected_inventory", tuple()),
        ("expected_counts", []),
    ),
)
def test_create_rejects_invalid_top_level_value_types_and_schema(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    header = _valid_header()
    header[field] = invalid_value

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize("field", _HASH_FIELDS)
@pytest.mark.parametrize(
    "invalid_digest",
    ("a" * 63, "A" * 64, "g" * 64, b"0" * 64),
    ids=("short", "uppercase", "nonhex", "bytes"),
)
def test_create_requires_exact_lowercase_sha256_fields(
    tmp_path: Path, field: str, invalid_digest: object
) -> None:
    header = _valid_header()
    header[field] = invalid_digest

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize(
    "invalid_query_ids",
    (
        [],
        ["q1", "q3", "q2", "q4"],
        ["q1", "q2", "q2", "q4"],
        ["", "q2", "q3", "q4"],
        [1, "q2", "q3", "q4"],
    ),
    ids=("empty", "unsorted", "duplicate", "empty-id", "nonstring-id"),
)
def test_create_requires_full_sorted_nonempty_unique_query_ids(
    tmp_path: Path, invalid_query_ids: list[object]
) -> None:
    header = _valid_header()
    header["query_ids"] = invalid_query_ids

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize(("shard_name", "field"), _SHARD_SCHEMA_LEAVES)
def test_create_requires_every_literal_shard_schema_leaf(
    tmp_path: Path, shard_name: str, field: str
) -> None:
    header = _valid_header()
    header["shard_schemas"][shard_name][field] = "mutated"

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-shard",
        "extra-shard",
        "nonobject-shard",
        "missing-field",
        "extra-field",
    ),
)
def test_create_requires_exact_shard_schema_objects(
    tmp_path: Path, mutation: str
) -> None:
    header = _valid_header()
    schemas = header["shard_schemas"]
    if mutation == "missing-shard":
        del schemas["centers"]
    elif mutation == "extra-shard":
        schemas["extra"] = copy.deepcopy(schemas["centers"])
    elif mutation == "nonobject-shard":
        schemas["centers"] = []
    elif mutation == "missing-field":
        del schemas["centers"]["layout"]
    else:
        schemas["centers"]["record_schema"] = "forbidden"

    _assert_header_rejected(tmp_path, header)


def test_create_accepts_canonical_pair_then_single_inventory_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    header = _valid_header(inventory=_CANONICAL_INVENTORY)

    cache = StreamingFitCache.create(root, header=header)

    assert cache._header["expected_inventory"] == _CANONICAL_INVENTORY
    assert cache._header["expected_counts"] == {
        "pair_fit_count": 3,
        "single_fit_count": 3,
        "fit_count": 6,
        "start_count": 60,
        "timing_record_count": 6,
    }
    del cache


@pytest.mark.parametrize(
    "mutation",
    (
        "nonobject-row",
        "missing-key",
        "extra-key",
        "invalid-kind",
        "nonstring-kind",
        "float-index",
        "bool-index",
        "float-cluster-count",
        "bool-cluster-count",
    ),
)
def test_create_requires_exact_inventory_object_keys_and_types(
    tmp_path: Path, mutation: str
) -> None:
    header = _valid_header()
    row = header["expected_inventory"][0]
    assert set(row) == _INVENTORY_KEYS
    if mutation == "nonobject-row":
        header["expected_inventory"][0] = []
    elif mutation == "missing-key":
        del row["cluster_count"]
    elif mutation == "extra-key":
        row["excluded_query_ids"] = ["q1", "q2"]
    elif mutation == "invalid-kind":
        row["exclusion_kind"] = "pairs"
    elif mutation == "nonstring-kind":
        row["exclusion_kind"] = 1
    elif mutation == "float-index":
        row["inventory_index"] = 0.0
    elif mutation == "bool-index":
        row["inventory_index"] = False
    elif mutation == "float-cluster-count":
        row["cluster_count"] = 2.0
    else:
        row["cluster_count"] = True

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize(
    "inventory",
    (
        pytest.param(
            [
                {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
                {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 1},
            ],
            id="cluster-count-descending",
        ),
        pytest.param(
            [
                {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 1},
                {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 1},
            ],
            id="inventory-index-descending",
        ),
        pytest.param(
            [
                {"exclusion_kind": "single", "inventory_index": 0, "cluster_count": 1},
                {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 1},
            ],
            id="single-before-pair",
        ),
    ),
)
def test_create_rejects_noncanonical_inventory_order(
    tmp_path: Path, inventory: list[dict[str, object]]
) -> None:
    _assert_header_rejected(tmp_path, _valid_header(inventory=inventory))


def test_create_rejects_duplicate_expected_inventory_triplet(tmp_path: Path) -> None:
    duplicate = {
        "exclusion_kind": "pair",
        "inventory_index": 0,
        "cluster_count": 2,
    }

    _assert_header_rejected(
        tmp_path,
        _valid_header(inventory=[duplicate, copy.deepcopy(duplicate)]),
    )


@pytest.mark.parametrize(
    ("exclusion_kind", "inventory_index", "cluster_count"),
    (
        ("pair", -1, 2),
        ("pair", 6, 2),
        ("single", -1, 2),
        ("single", 4, 2),
        ("pair", 0, 0),
        ("pair", 0, 65_537),
    ),
    ids=(
        "negative-pair-index",
        "pair-index-upper-bound",
        "negative-single-index",
        "single-index-upper-bound",
        "zero-cluster-count",
        "cluster-count-upper-bound",
    ),
)
def test_create_rejects_inventory_indices_and_cluster_counts_out_of_bounds(
    tmp_path: Path,
    exclusion_kind: str,
    inventory_index: int,
    cluster_count: int,
) -> None:
    row = {
        "exclusion_kind": exclusion_kind,
        "inventory_index": inventory_index,
        "cluster_count": cluster_count,
    }

    _assert_header_rejected(tmp_path, _valid_header(inventory=[row]))


@pytest.mark.parametrize("missing_key", sorted(_COUNT_KEYS))
def test_create_requires_every_exact_expected_count_key(
    tmp_path: Path, missing_key: str
) -> None:
    header = _valid_header()
    del header["expected_counts"][missing_key]

    _assert_header_rejected(tmp_path, header)


def test_create_rejects_unknown_expected_count_key(tmp_path: Path) -> None:
    header = _valid_header()
    header["expected_counts"]["unknown_count"] = 0

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize("field", sorted(_COUNT_KEYS))
@pytest.mark.parametrize(
    "invalid_value",
    (True, 1.0, -1),
    ids=("bool", "float", "negative"),
)
def test_create_requires_exact_nonnegative_integer_counts(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    header = _valid_header()
    header["expected_counts"][field] = invalid_value

    _assert_header_rejected(tmp_path, header)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("pair_fit_count", 0),
        ("single_fit_count", 1),
        ("fit_count", 2),
        ("start_count", 9),
        ("timing_record_count", 0),
    ),
)
def test_create_requires_all_expected_count_redundancies(
    tmp_path: Path, field: str, invalid_value: int
) -> None:
    header = _valid_header()
    header["expected_counts"][field] = invalid_value

    _assert_header_rejected(tmp_path, header)


def test_append_fit_writes_exact_open_shards_receipt_and_resident_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    assert excluded_query_ids == ("q1", "q2")
    assert fit.key.support_query_ids == ("q3", "q4")
    assert fit.centers.dtype == np.dtype("<f4")
    assert fit.centers.flags.c_contiguous
    assert fit.labels.dtype == np.dtype("<u2")
    assert fit.labels.flags.c_contiguous
    assert fit.memberships == (("q3",), ("q4",))
    assert fit.center_sha256 == _independent_center_sha256(fit.centers)
    membership_rows = ((0, ("q3",)), (1, ("q4",)))
    assert fit.fit_membership_sha256 == _independent_membership_sha256(membership_rows)
    assert fit.cluster_membership_sha256 == tuple(
        _independent_membership_sha256((row,)) for row in membership_rows
    )
    assert fit.cluster_inertia == (0.0, 0.0)
    assert fit.sklearn_inertia.hex() == math.fsum(fit.cluster_inertia).hex()
    assert fit.recomputed_total_inertia.hex() == math.fsum(fit.cluster_inertia).hex()
    assert tuple(assignment.query_id for assignment in assignments) == (
        "q1",
        "q2",
    )
    assert tuple(assignment.member_indices for assignment in assignments) == (
        (2,),
        (3,),
    )
    cache = StreamingFitCache.create(root, header=_valid_header())

    result = cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )

    assert result is None
    handles = {
        "centers.f4": cache._center_handle,
        "labels.u2": cache._label_handle,
        "cluster_inertia.f8": cache._inertia_handle,
        "fit_receipts.jsonl": cache._receipt_handle,
        "timings.jsonl": cache._timing_handle,
    }
    assert all(not handle.closed for handle in handles.values())
    assert {Path(handle.name).name for handle in handles.values()} == _STREAM_PATHS

    expected_receipt = _expected_fit_receipt(
        fit=fit,
        assignments=assignments,
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
        sequence=0,
        center_offset=0,
        label_offset=0,
        inertia_offset=0,
    )
    receipt_bytes = (root / "fit_receipts.jsonl").read_bytes()
    assert receipt_bytes == _independent_canonical_json(expected_receipt) + b"\n"
    observed_receipt = json.loads(receipt_bytes)
    assert observed_receipt == expected_receipt
    observed_without_hash = dict(observed_receipt)
    observed_record_sha256 = observed_without_hash.pop("record_sha256")
    assert observed_record_sha256 == _independent_framed_sha256(
        "fit-cache-record",
        _independent_canonical_json(observed_without_hash),
    )
    assignment_records = observed_receipt["assignments"]
    assert isinstance(assignment_records, list)
    assert all(
        set(record)
        == {
            "query_id",
            "canonical_cluster_id",
            "squared_distance_hex",
            "cluster_support",
        }
        and "member_indices" not in record
        for record in assignment_records
    )

    stored = cache.load_fit(fit.key)
    assert isinstance(stored, StoredFitRecord)
    assert stored.fit is fit
    assert stored.assignments == assignments
    assert stored.exclusion_kind == "pair"
    assert stored.inventory_index == 0
    assert stored.excluded_query_ids == excluded_query_ids

    for handle in (
        cache._center_handle,
        cache._label_handle,
        cache._inertia_handle,
    ):
        handle.flush()
    assert (root / "centers.f4").read_bytes() == fit.centers.tobytes(order="C")
    assert (root / "labels.u2").read_bytes() == fit.labels.tobytes(order="C")
    assert (root / "cluster_inertia.f8").read_bytes() == np.asarray(
        fit.cluster_inertia, dtype="<f8"
    ).tobytes(order="C")
    assert (root / "timings.jsonl").read_bytes() == b""
    assert not (root / "header.json").exists()
    assert all(not handle.closed for handle in handles.values())
    del stored, cache


def test_second_pair_fit_concatenates_exact_shards_offsets_and_receipt_order(
    tmp_path: Path,
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(
        root,
        header=_valid_header(inventory=inventory),
    )
    first_fit, first_assignments, first_excluded = _synthetic_pair_fit(0)
    second_fit, second_assignments, second_excluded = _synthetic_pair_fit(1)
    assert first_excluded == tuple(combinations(_QUERY_IDS, 2))[0] == ("q1", "q2")
    assert (
        second_excluded
        == tuple(combinations(_QUERY_IDS, 2))[1]
        == (
            "q1",
            "q3",
        )
    )
    assert (
        first_fit.key.support_query_ids
        == _support_without(first_excluded)
        == (
            "q3",
            "q4",
        )
    )
    assert (
        second_fit.key.support_query_ids
        == _support_without(second_excluded)
        == (
            "q2",
            "q4",
        )
    )
    assert tuple(assignment.member_indices for assignment in second_assignments) == (
        (1,),
        (3,),
    )

    cache.append_fit(
        first_fit,
        first_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=first_excluded,
    )
    cache.append_fit(
        second_fit,
        second_assignments,
        exclusion_kind="pair",
        inventory_index=1,
        excluded_query_ids=second_excluded,
    )

    first_center_bytes = first_fit.centers.tobytes(order="C")
    first_label_bytes = first_fit.labels.tobytes(order="C")
    first_inertia_bytes = np.asarray(first_fit.cluster_inertia, dtype="<f8").tobytes(
        order="C"
    )
    second_center_bytes = second_fit.centers.tobytes(order="C")
    second_label_bytes = second_fit.labels.tobytes(order="C")
    second_inertia_bytes = np.asarray(second_fit.cluster_inertia, dtype="<f8").tobytes(
        order="C"
    )
    expected_receipts = [
        _expected_fit_receipt(
            fit=first_fit,
            assignments=first_assignments,
            inventory_index=0,
            excluded_query_ids=first_excluded,
            sequence=0,
            center_offset=0,
            label_offset=0,
            inertia_offset=0,
        ),
        _expected_fit_receipt(
            fit=second_fit,
            assignments=second_assignments,
            inventory_index=1,
            excluded_query_ids=second_excluded,
            sequence=1,
            center_offset=len(first_center_bytes),
            label_offset=len(first_label_bytes),
            inertia_offset=len(first_inertia_bytes),
        ),
    ]
    expected_receipt_bytes = b"".join(
        _independent_canonical_json(receipt) + b"\n" for receipt in expected_receipts
    )
    observed_receipt_bytes = (root / "fit_receipts.jsonl").read_bytes()
    assert observed_receipt_bytes == expected_receipt_bytes
    observed_receipts = [
        json.loads(line) for line in observed_receipt_bytes.splitlines()
    ]
    assert observed_receipts == expected_receipts
    assert [receipt["sequence"] for receipt in observed_receipts] == [0, 1]
    assert [
        (
            receipt["exclusion_kind"],
            receipt["inventory_index"],
            receipt["cluster_count"],
        )
        for receipt in observed_receipts
    ] == [("pair", 0, 2), ("pair", 1, 2)]

    for handle in (
        cache._center_handle,
        cache._label_handle,
        cache._inertia_handle,
    ):
        handle.flush()
    assert (root / "centers.f4").read_bytes() == (
        first_center_bytes + second_center_bytes
    )
    assert (root / "labels.u2").read_bytes() == (first_label_bytes + second_label_bytes)
    assert (root / "cluster_inertia.f8").read_bytes() == (
        first_inertia_bytes + second_inertia_bytes
    )
    assert (root / "timings.jsonl").read_bytes() == b""
    assert not (root / "header.json").exists()
    del cache


def test_duplicate_full_fit_key_precedes_expected_inventory_validation(
    tmp_path: Path,
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(inventory=inventory),
    )
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )

    _assert_append_rejected_without_side_effects(
        cache,
        fit=fit,
        assignments=assignments,
        exclusion_kind="single",
        inventory_index=3,
        excluded_query_ids=("q4",),
        match="duplicate fit key",
    )

    stored = cache.load_fit(fit.key)
    assert stored is not None
    assert stored.fit is fit
    del stored, cache


@pytest.mark.parametrize("case", ("repeated", "wrong", "exhausted"))
def test_second_append_requires_the_next_exact_inventory_triplet(
    tmp_path: Path, case: str
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
    ]
    if case != "exhausted":
        inventory.append(
            {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2}
        )
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(inventory=inventory),
    )
    first_fit, first_assignments, first_excluded = _synthetic_pair_fit(0)
    cache.append_fit(
        first_fit,
        first_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=first_excluded,
    )
    candidate_index = 2 if case == "wrong" else 1
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(candidate_index)
    submitted_index = 0 if case == "repeated" else candidate_index

    _assert_append_rejected_without_side_effects(
        cache,
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=submitted_index,
        excluded_query_ids=excluded_query_ids,
    )
    del cache


@pytest.mark.parametrize(
    "case",
    (
        "exclusion-kind",
        "inventory-index",
        "excluded-order",
        "excluded-identity",
        "cluster-count",
        "support-order",
    ),
)
def test_append_requires_exact_next_inventory_identity(
    tmp_path: Path, case: str
) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    exclusion_kind = "pair"
    inventory_index = 0
    if case == "exclusion-kind":
        exclusion_kind = "single"
    elif case == "inventory-index":
        inventory_index = 1
    elif case == "excluded-order":
        excluded_query_ids = tuple(reversed(excluded_query_ids))
    elif case == "excluded-identity":
        excluded_query_ids = _pair_excluded_query_ids(1)
    elif case == "cluster-count":
        fit, assignments, excluded_query_ids = _synthetic_c1_pair_fit()
    elif case == "support-order":
        fit, assignments, excluded_query_ids = _synthetic_reordered_support_pair_fit()

    _assert_append_rejected_without_side_effects(
        cache,
        fit=fit,
        assignments=assignments,
        exclusion_kind=exclusion_kind,
        inventory_index=inventory_index,
        excluded_query_ids=excluded_query_ids,
    )
    del cache


@pytest.mark.parametrize(
    "hash_field", ("embedding_sha256", "clustering_contract_sha256")
)
def test_append_requires_fit_hashes_to_equal_header(
    tmp_path: Path, hash_field: str
) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    mutated_key = dataclasses.replace(
        fit.key,
        **{hash_field: "a" * 64},
    )
    fit = dataclasses.replace(fit, key=mutated_key)

    _assert_append_rejected_without_side_effects(
        cache,
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    del cache


@pytest.mark.parametrize(
    "case",
    (
        "centers-type",
        "centers-rank",
        "centers-row-count",
        "centers-dtype",
        "centers-noncontiguous",
        "centers-nonfinite",
        "labels-type",
        "labels-rank",
        "labels-support-length",
        "labels-dtype",
        "labels-noncontiguous",
        "labels-out-of-range",
        "labels-unoccupied-cluster",
        "memberships-not-label-derived",
        "center-hash",
        "fit-membership-hash",
        "cluster-membership-hash",
        "inertia-container-type",
        "inertia-item-type",
        "inertia-length",
        "inertia-nonfinite",
        "inertia-negative",
        "recomputed-total",
        "sklearn-mismatch",
        "n-iter-bool",
        "n-iter-type",
        "n-iter-zero",
        "n-iter-upper-bound",
    ),
)
def test_append_rejects_malformed_canonical_fit_without_side_effects(
    tmp_path: Path, case: str
) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    nonfinite_centers = fit.centers.copy()
    nonfinite_centers[0, 0] = np.nan
    mutations: dict[str, dict[str, object]] = {
        "centers-type": {"centers": fit.centers.tolist()},
        "centers-rank": {"centers": fit.centers.reshape(-1)},
        "centers-row-count": {
            "centers": np.ascontiguousarray(
                np.concatenate((fit.centers, fit.centers[:1]), axis=0),
                dtype="<f4",
            )
        },
        "centers-dtype": {"centers": fit.centers.astype(">f4")},
        "centers-noncontiguous": {"centers": np.asfortranarray(fit.centers)},
        "centers-nonfinite": {"centers": nonfinite_centers},
        "labels-type": {"labels": fit.labels.tolist()},
        "labels-rank": {"labels": fit.labels.reshape(1, -1)},
        "labels-support-length": {
            "labels": np.ascontiguousarray(np.asarray([0, 1, 0], dtype="<u2"))
        },
        "labels-dtype": {"labels": fit.labels.astype(">u2")},
        "labels-noncontiguous": {
            "labels": np.asarray([0, 99, 1, 99], dtype="<u2")[::2]
        },
        "labels-out-of-range": {
            "labels": np.ascontiguousarray(np.asarray([0, 2], dtype="<u2"))
        },
        "labels-unoccupied-cluster": {
            "labels": np.ascontiguousarray(np.asarray([0, 0], dtype="<u2"))
        },
        "memberships-not-label-derived": {
            "memberships": tuple(reversed(fit.memberships))
        },
        "center-hash": {"center_sha256": "a" * 64},
        "fit-membership-hash": {"fit_membership_sha256": "a" * 64},
        "cluster-membership-hash": {
            "cluster_membership_sha256": (
                "a" * 64,
                fit.cluster_membership_sha256[1],
            )
        },
        "inertia-container-type": {"cluster_inertia": [0.0, 0.0]},
        "inertia-item-type": {"cluster_inertia": (0, 0.0)},
        "inertia-length": {"cluster_inertia": (0.0,)},
        "inertia-nonfinite": {"cluster_inertia": (float("nan"), 0.0)},
        "inertia-negative": {"cluster_inertia": (-1.0, 0.0)},
        "recomputed-total": {"recomputed_total_inertia": 1.0},
        "sklearn-mismatch": {"sklearn_inertia": 1.0},
        "n-iter-bool": {"n_iter": True},
        "n-iter-type": {"n_iter": 1.0},
        "n-iter-zero": {"n_iter": 0},
        "n-iter-upper-bound": {"n_iter": 300},
    }
    malformed_fit = dataclasses.replace(fit, **mutations[case])

    _assert_append_rejected_without_side_effects(
        cache,
        fit=malformed_fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    del cache


def test_append_accepts_unequal_sklearn_inertia_within_task5_tolerance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    header = _valid_header()
    cache = StreamingFitCache.create(root, header=header)
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    sklearn_inertia = 5e-9
    assert sklearn_inertia != fit.recomputed_total_inertia
    assert math.isclose(
        sklearn_inertia,
        fit.recomputed_total_inertia,
        rel_tol=1e-5,
        abs_tol=1e-8,
    )
    fit = dataclasses.replace(fit, sklearn_inertia=sklearn_inertia)

    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )

    (receipt,) = [
        json.loads(line)
        for line in (root / "fit_receipts.jsonl").read_bytes().splitlines()
    ]
    assert receipt == _expected_fit_receipt(
        fit=fit,
        assignments=assignments,
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
        sequence=0,
        center_offset=0,
        label_offset=0,
        inertia_offset=0,
    )
    assert receipt["sklearn_inertia_hex"] == sklearn_inertia.hex()
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=123,
        n_iter=fit.n_iter,
    )
    assert cache.finalize() == header
    del cache

    readonly = StreamingFitCache.open_readonly(root)
    loaded = readonly.load_fit(dataclasses.replace(fit.key))

    assert isinstance(loaded, StoredFitRecord)
    assert loaded.fit.sklearn_inertia == sklearn_inertia
    assert loaded.fit.sklearn_inertia.hex() == sklearn_inertia.hex()
    assert loaded.fit.sklearn_inertia != loaded.fit.recomputed_total_inertia
    assert math.isclose(
        loaded.fit.sklearn_inertia,
        loaded.fit.recomputed_total_inertia,
        rel_tol=1e-5,
        abs_tol=1e-8,
    )


@pytest.mark.parametrize(
    "case",
    (
        "missing-assignment",
        "extra-assignment",
        "reversed-assignments",
        "wrong-query-id",
        "cluster-id-out-of-range",
        "cluster-id-bool",
        "cluster-id-float",
        "cluster-membership-mismatch",
        "member-index-not-global",
        "member-index-order",
        "cluster-support",
        "cluster-support-bool",
        "cluster-support-float",
        "distance-int",
        "distance-bool",
        "distance-nan",
        "distance-infinite",
        "distance-negative",
    ),
)
def test_append_requires_exact_semantically_valid_assignments(
    tmp_path: Path, case: str
) -> None:
    header = _valid_header()
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    if case == "missing-assignment":
        assignments = assignments[:1]
    elif case == "extra-assignment":
        assignments = (*assignments, assignments[0])
    elif case == "reversed-assignments":
        assignments = tuple(reversed(assignments))
    elif case == "wrong-query-id":
        assignments = (
            dataclasses.replace(assignments[0], query_id="q4"),
            assignments[1],
        )
    elif case == "cluster-id-out-of-range":
        assignments = (
            dataclasses.replace(assignments[0], canonical_cluster_id=2),
            assignments[1],
        )
    elif case == "cluster-id-bool":
        assignments = (
            dataclasses.replace(assignments[0], canonical_cluster_id=False),
            assignments[1],
        )
    elif case == "cluster-id-float":
        assignments = (
            dataclasses.replace(assignments[0], canonical_cluster_id=0.0),
            assignments[1],
        )
    elif case == "cluster-membership-mismatch":
        assignments = (
            dataclasses.replace(assignments[0], canonical_cluster_id=1),
            assignments[1],
        )
    elif case == "member-index-not-global":
        assignments = (
            dataclasses.replace(assignments[0], member_indices=(0,)),
            assignments[1],
        )
    elif case == "member-index-order":
        fit, assignments, excluded_query_ids = _synthetic_c1_pair_fit()
        header = _valid_header(
            inventory=[
                {
                    "exclusion_kind": "pair",
                    "inventory_index": 0,
                    "cluster_count": 1,
                }
            ]
        )
        assignments = (
            dataclasses.replace(assignments[0], member_indices=(3, 2)),
            assignments[1],
        )
    elif case == "cluster-support":
        assignments = (
            dataclasses.replace(assignments[0], cluster_support=2),
            assignments[1],
        )
    elif case == "cluster-support-bool":
        assignments = (
            dataclasses.replace(assignments[0], cluster_support=True),
            assignments[1],
        )
    elif case == "cluster-support-float":
        assignments = (
            dataclasses.replace(assignments[0], cluster_support=1.0),
            assignments[1],
        )
    elif case == "distance-int":
        assignments = (
            dataclasses.replace(assignments[0], squared_distance=100),
            assignments[1],
        )
    elif case == "distance-bool":
        assignments = (
            dataclasses.replace(assignments[0], squared_distance=True),
            assignments[1],
        )
    elif case == "distance-nan":
        assignments = (
            dataclasses.replace(assignments[0], squared_distance=float("nan")),
            assignments[1],
        )
    elif case == "distance-infinite":
        assignments = (
            dataclasses.replace(assignments[0], squared_distance=float("inf")),
            assignments[1],
        )
    else:
        assignments = (
            dataclasses.replace(assignments[0], squared_distance=-0.25),
            assignments[1],
        )

    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=header,
    )
    _assert_append_rejected_without_side_effects(
        cache,
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    del cache


def test_fit_key_digest_collision_fails_closed_for_append_load_and_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(inventory=inventory),
    )
    first_fit, first_assignments, first_excluded = _synthetic_pair_fit(0)
    second_fit, second_assignments, second_excluded = _synthetic_pair_fit(1)
    assert first_fit.key != second_fit.key
    collision_sha256 = "f" * 64
    monkeypatch.setattr(
        cache_module,
        "_fit_key_sha256",
        lambda _key: collision_sha256,
    )
    cache.append_fit(
        first_fit,
        first_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=first_excluded,
    )
    first_record = cache.load_fit(first_fit.key)
    assert first_record is not None
    assert first_record.fit is first_fit

    _assert_append_rejected_without_side_effects(
        cache,
        fit=second_fit,
        assignments=second_assignments,
        exclusion_kind="pair",
        inventory_index=1,
        excluded_query_ids=second_excluded,
        match="collision",
    )

    before_load = _cache_mutation_snapshot(cache)
    with pytest.raises(ValueError, match="collision"):
        cache.load_fit(second_fit.key)
    assert _cache_mutation_snapshot(cache) == before_load

    before_release = _cache_mutation_snapshot(cache)
    with pytest.raises(ValueError, match="collision"):
        cache.release(second_fit.key)
    assert _cache_mutation_snapshot(cache) == before_release
    retained = cache.load_fit(first_fit.key)
    assert retained is first_record
    assert retained.fit is first_fit
    assert len(cache._index) == len(cache._resident) == cache._fit_count == 1
    del retained, first_record, cache


def test_append_timing_flushes_exact_unhashed_canonical_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(root, header=_valid_header())
    fit = _append_synthetic_pair_fit(cache, 0)
    elapsed_ns = 1_234_567

    result = cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=elapsed_ns,
        n_iter=fit.n_iter,
    )

    assert result is None
    expected = _expected_timing_record(
        sequence=0,
        inventory_index=0,
        elapsed_ns=elapsed_ns,
        n_iter=fit.n_iter,
    )
    timing_bytes = (root / "timings.jsonl").read_bytes()
    assert timing_bytes == _independent_canonical_json(expected) + b"\n"
    observed = json.loads(timing_bytes)
    assert observed == expected
    assert set(observed) == _TIMING_KEYS
    assert not {key for key in observed if "sha256" in key or "hash" in key}
    assert "metadata" not in observed
    assert set(observed) - {"elapsed_ns"} == _TIMING_KEYS - {"elapsed_ns"}
    assert cache._timing_count == 1
    _assert_open_incomplete_cache(cache)
    del cache


def test_two_fit_timings_have_exact_independent_sequence_and_order(
    tmp_path: Path,
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(
        root,
        header=_valid_header(inventory=inventory),
    )
    first_fit = _append_synthetic_pair_fit(cache, 0)
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=101,
        n_iter=first_fit.n_iter,
    )
    second_fit = _append_synthetic_pair_fit(cache, 1)
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=1,
        cluster_count=2,
        elapsed_ns=202,
        n_iter=second_fit.n_iter,
    )

    expected = [
        _expected_timing_record(
            sequence=0,
            inventory_index=0,
            elapsed_ns=101,
            n_iter=first_fit.n_iter,
        ),
        _expected_timing_record(
            sequence=1,
            inventory_index=1,
            elapsed_ns=202,
            n_iter=second_fit.n_iter,
        ),
    ]
    timing_bytes = (root / "timings.jsonl").read_bytes()
    assert timing_bytes == b"".join(
        _independent_canonical_json(record) + b"\n" for record in expected
    )
    observed = [json.loads(line) for line in timing_bytes.splitlines()]
    assert observed == expected
    assert [record["sequence"] for record in observed] == [0, 1]
    assert [
        (
            record["exclusion_kind"],
            record["inventory_index"],
            record["cluster_count"],
        )
        for record in observed
    ] == [("pair", 0, 2), ("pair", 1, 2)]
    assert cache._fit_count == cache._timing_count == 2
    _assert_open_incomplete_cache(cache)
    del cache


def test_append_timing_rejects_next_triplet_before_its_fit(tmp_path: Path) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(inventory=inventory),
    )
    first_fit = _append_synthetic_pair_fit(cache, 0)
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=101,
        n_iter=first_fit.n_iter,
    )
    second_fit, _, _ = _synthetic_pair_fit(1)

    _assert_timing_rejected_without_side_effects(
        cache,
        exclusion_kind="pair",
        inventory_index=1,
        cluster_count=2,
        elapsed_ns=202,
        n_iter=second_fit.n_iter,
    )
    del cache


@pytest.mark.parametrize("case", ("duplicate", "exhausted"))
def test_append_timing_rejects_duplicate_or_exhausted_inventory(
    tmp_path: Path, case: str
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
    ]
    if case == "duplicate":
        inventory.append(
            {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2}
        )
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(inventory=inventory),
    )
    first_fit = _append_synthetic_pair_fit(cache, 0)
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=101,
        n_iter=first_fit.n_iter,
    )
    if case == "duplicate":
        _append_synthetic_pair_fit(cache, 1)

    _assert_timing_rejected_without_side_effects(
        cache,
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=303,
        n_iter=first_fit.n_iter,
    )
    del cache


def test_append_timing_rejects_wrong_next_triplet(tmp_path: Path) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(inventory=inventory),
    )
    _append_synthetic_pair_fit(cache, 0)
    second_fit = _append_synthetic_pair_fit(cache, 1)

    _assert_timing_rejected_without_side_effects(
        cache,
        exclusion_kind="pair",
        inventory_index=1,
        cluster_count=2,
        elapsed_ns=202,
        n_iter=second_fit.n_iter,
    )
    del cache


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("exclusion_kind", "single"),
        ("exclusion_kind", "PAIR"),
        ("exclusion_kind", 1),
        ("inventory_index", True),
        ("inventory_index", 0.0),
        ("inventory_index", -1),
        ("inventory_index", 1),
        ("cluster_count", True),
        ("cluster_count", 2.0),
        ("cluster_count", 0),
        ("cluster_count", 1),
    ),
)
def test_append_timing_requires_exact_kind_index_and_cluster_count(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    fit = _append_synthetic_pair_fit(cache, 0)
    arguments: dict[str, object] = {
        "exclusion_kind": "pair",
        "inventory_index": 0,
        "cluster_count": 2,
        "elapsed_ns": 101,
        "n_iter": fit.n_iter,
    }
    arguments[field] = invalid_value

    _assert_timing_rejected_without_side_effects(cache, **arguments)
    del cache


@pytest.mark.parametrize(
    "invalid_elapsed_ns",
    (True, 1.0, -1),
    ids=("bool", "float", "negative"),
)
def test_append_timing_requires_exact_nonnegative_elapsed_ns(
    tmp_path: Path, invalid_elapsed_ns: object
) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    fit = _append_synthetic_pair_fit(cache, 0)

    _assert_timing_rejected_without_side_effects(
        cache,
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=invalid_elapsed_ns,
        n_iter=fit.n_iter,
    )
    del cache


@pytest.mark.parametrize(
    "invalid_n_iter",
    (True, 1.0, 0, 300),
    ids=("bool", "float", "zero", "upper-bound"),
)
def test_append_timing_requires_exact_n_iter_in_closed_range_1_to_299(
    tmp_path: Path, invalid_n_iter: object
) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    _append_synthetic_pair_fit(cache, 0)

    _assert_timing_rejected_without_side_effects(
        cache,
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=101,
        n_iter=invalid_n_iter,
    )
    del cache


def test_append_timing_n_iter_must_equal_corresponding_fit_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(root, header=_valid_header())
    fit = _append_synthetic_pair_fit(cache, 0)
    (fit_receipt,) = [
        json.loads(line)
        for line in (root / "fit_receipts.jsonl").read_bytes().splitlines()
    ]
    assert fit_receipt["n_iter"] == fit.n_iter == 1

    _assert_timing_rejected_without_side_effects(
        cache,
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=101,
        n_iter=2,
    )
    del cache


def test_append_timing_accepts_zero_elapsed_and_n_iter_299(tmp_path: Path) -> None:
    root = tmp_path / "fit-cache"
    header = _valid_header()
    cache = StreamingFitCache.create(root, header=header)
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    fit = dataclasses.replace(fit, n_iter=299)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )

    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=0,
        n_iter=299,
    )

    expected = _expected_timing_record(
        sequence=0,
        inventory_index=0,
        elapsed_ns=0,
        n_iter=299,
    )
    assert (root / "timings.jsonl").read_bytes() == (
        _independent_canonical_json(expected) + b"\n"
    )
    assert cache._fit_count == cache._timing_count == 1
    assert cache.finalize() == header
    del cache

    readonly = StreamingFitCache.open_readonly(root)
    loaded = readonly.load_fit(dataclasses.replace(fit.key))

    assert isinstance(loaded, StoredFitRecord)
    assert loaded.fit.key == fit.key
    assert loaded.fit.n_iter == 299
    assert (root / "timings.jsonl").read_bytes() == (
        _independent_canonical_json(expected) + b"\n"
    )


def test_worker_can_release_fit_arrays_before_appending_exact_timing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(root, header=_valid_header())
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    fit_key = dataclasses.replace(fit.key)
    captured_n_iter = fit.n_iter
    center_ref = weakref.ref(fit.centers)
    label_ref = weakref.ref(fit.labels)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    appended = _cache_mutation_snapshot(cache)
    index_keys_before_timing = set(cache._index)
    assert index_keys_before_timing == {_independent_fit_key_sha256(fit_key)}

    assert cache.release(fit_key) is None
    del fit, assignments
    gc.collect()

    assert center_ref() is None
    assert label_ref() is None
    released = _cache_mutation_snapshot(cache)
    assert released.streams == appended.streams
    assert released.index_count == released.fit_count == 1
    assert released.resident_count == released.timing_count == 0
    assert set(cache._index) == index_keys_before_timing

    elapsed_ns = 456_789
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=elapsed_ns,
        n_iter=captured_n_iter,
    )

    timed = _cache_mutation_snapshot(cache)
    assert timed.streams[:-1] == released.streams[:-1]
    expected_timing = _expected_timing_record(
        sequence=0,
        inventory_index=0,
        elapsed_ns=elapsed_ns,
        n_iter=captured_n_iter,
    )
    assert timed.streams[-1][0] == "timings.jsonl"
    assert timed.streams[-1][1] == (
        _independent_canonical_json(expected_timing) + b"\n"
    )
    assert timed.index_count == timed.fit_count == timed.timing_count == 1
    assert timed.resident_count == 0
    assert set(cache._index) == index_keys_before_timing
    _assert_open_incomplete_cache(cache)
    del cache


def test_load_fit_returns_none_for_direct_absent_key(tmp_path: Path) -> None:
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(root, header=_valid_header())
    absent_key = FitKey(("q-absent",), 1, "1" * 64, "3" * 64)

    assert cache.load_fit(absent_key) is None
    assert cache._index == {}
    assert cache._resident == {}
    assert cache._fit_count == cache._timing_count == 0
    assert all((root / path).read_bytes() == b"" for path in _STREAM_PATHS)
    assert not (root / "header.json").exists()
    del cache


def test_release_collects_original_arrays_but_preserves_index_and_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(root, header=_valid_header())
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    equal_fit_key = dataclasses.replace(fit.key)
    assert equal_fit_key == fit.key
    assert equal_fit_key is not fit.key
    center_ref = weakref.ref(fit.centers)
    label_ref = weakref.ref(fit.labels)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    receipt_bytes = (root / "fit_receipts.jsonl").read_bytes()
    assert receipt_bytes.endswith(b"\n")
    assert receipt_bytes
    assert len(cache._resident) == 1
    (resident,) = cache._resident.values()
    assert resident.fit is fit
    assert resident.fit.centers is fit.centers
    assert resident.fit.labels is fit.labels
    assert resident.assignments == assignments
    before_release = _cache_mutation_snapshot(cache)
    assert before_release.index_count == before_release.resident_count == 1
    del resident, fit, assignments

    gc.collect()
    assert center_ref() is not None
    assert label_ref() is not None
    assert cache.release(equal_fit_key) is None
    gc.collect()

    assert center_ref() is None
    assert label_ref() is None
    after_release = _cache_mutation_snapshot(cache)
    assert after_release.streams == before_release.streams
    assert after_release.index_count == 1
    assert after_release.resident_count == 0
    assert after_release.fit_count == 1
    assert after_release.timing_count == 0
    assert len(cache._index) == 1
    assert (root / "fit_receipts.jsonl").read_bytes() == receipt_bytes
    assert not (root / "header.json").exists()
    del cache


def test_append_release_then_load_reconstructs_directly(tmp_path: Path) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    fit_key = dataclasses.replace(fit.key)
    expected_semantics = _fit_record_semantics(
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )

    assert cache.release(fit_key) is None
    assert cache._resident == {}
    loaded = cache.load_fit(fit_key)

    assert isinstance(loaded, StoredFitRecord)
    assert loaded.fit is not fit
    assert loaded.fit.centers is not fit.centers
    assert loaded.fit.labels is not fit.labels
    assert (
        _fit_record_semantics(
            fit=loaded.fit,
            assignments=loaded.assignments,
            exclusion_kind=loaded.exclusion_kind,
            inventory_index=loaded.inventory_index,
            excluded_query_ids=loaded.excluded_query_ids,
        )
        == expected_semantics
    )
    del loaded, cache


def test_release_is_idempotent_for_absent_and_nonresident_exact_keys(
    tmp_path: Path,
) -> None:
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(),
    )
    absent_key = FitKey(("q-absent",), 1, "1" * 64, "3" * 64)
    empty = _cache_mutation_snapshot(cache)

    assert cache.release(absent_key) is None
    assert _cache_mutation_snapshot(cache) == empty

    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    equal_fit_key = dataclasses.replace(fit.key)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    assert cache.release(equal_fit_key) is None
    nonresident = _cache_mutation_snapshot(cache)
    assert nonresident.index_count == 1
    assert nonresident.resident_count == 0

    assert cache.release(equal_fit_key) is None
    assert _cache_mutation_snapshot(cache) == nonresident
    del fit, assignments, cache


def test_writing_cache_load_reconstructs_and_releases_independent_arrays(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    cache = StreamingFitCache.create(root, header=_valid_header())
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    equal_fit_key = dataclasses.replace(fit.key)
    expected_semantics = _fit_record_semantics(
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    original_center_ref = weakref.ref(fit.centers)
    original_label_ref = weakref.ref(fit.labels)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    appended = _cache_mutation_snapshot(cache)
    del fit, assignments
    gc.collect()
    assert original_center_ref() is not None
    assert original_label_ref() is not None
    cache.release(equal_fit_key)
    gc.collect()
    assert original_center_ref() is None
    assert original_label_ref() is None
    assert len(cache._resident) == 0

    loaded = cache.load_fit(equal_fit_key)

    assert isinstance(loaded, StoredFitRecord)
    assert (
        _fit_record_semantics(
            fit=loaded.fit,
            assignments=loaded.assignments,
            exclusion_kind=loaded.exclusion_kind,
            inventory_index=loaded.inventory_index,
            excluded_query_ids=loaded.excluded_query_ids,
        )
        == expected_semantics
    )
    assert next(iter(cache._resident.values())) is loaded
    assert len(cache._resident) == len(cache._index) == 1
    assert _cache_mutation_snapshot(cache).streams == appended.streams
    loaded_center_ref = weakref.ref(loaded.fit.centers)
    loaded_label_ref = weakref.ref(loaded.fit.labels)
    del loaded

    gc.collect()
    assert loaded_center_ref() is not None
    assert loaded_label_ref() is not None
    assert cache.release(equal_fit_key) is None
    gc.collect()

    assert loaded_center_ref() is None
    assert loaded_label_ref() is None
    released = _cache_mutation_snapshot(cache)
    assert released.streams == appended.streams
    assert released.index_count == released.fit_count == 1
    assert released.resident_count == released.timing_count == 0
    _assert_open_incomplete_cache(cache)
    del cache


def test_two_fit_releases_bound_residency_independently(
    tmp_path: Path,
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    cache = StreamingFitCache.create(
        tmp_path / "fit-cache",
        header=_valid_header(inventory=inventory),
    )
    first_fit = _append_synthetic_pair_fit(cache, 0)
    second_fit = _append_synthetic_pair_fit(cache, 1)
    first_key = dataclasses.replace(first_fit.key)
    second_key = dataclasses.replace(second_fit.key)
    first_center_ref = weakref.ref(first_fit.centers)
    first_label_ref = weakref.ref(first_fit.labels)
    second_center_ref = weakref.ref(second_fit.centers)
    second_label_ref = weakref.ref(second_fit.labels)
    appended = _cache_mutation_snapshot(cache)
    assert appended.index_count == appended.resident_count == appended.fit_count == 2
    del first_fit, second_fit
    gc.collect()
    assert all(
        reference() is not None
        for reference in (
            first_center_ref,
            first_label_ref,
            second_center_ref,
            second_label_ref,
        )
    )

    assert cache.release(first_key) is None
    gc.collect()

    assert first_center_ref() is None
    assert first_label_ref() is None
    assert second_center_ref() is not None
    assert second_label_ref() is not None
    after_first = _cache_mutation_snapshot(cache)
    assert after_first.streams == appended.streams
    assert after_first.index_count == after_first.fit_count == 2
    assert after_first.resident_count == 1

    assert cache.release(second_key) is None
    gc.collect()

    assert second_center_ref() is None
    assert second_label_ref() is None
    after_second = _cache_mutation_snapshot(cache)
    assert after_second.streams == appended.streams
    assert after_second.index_count == after_second.fit_count == 2
    assert after_second.resident_count == after_second.timing_count == 0
    _assert_open_incomplete_cache(cache)
    del cache


def test_finalize_durably_publishes_header_then_independently_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "fit-cache"
    expected_header = _valid_header()
    cache, _fit = _populate_one_fit_cache(root, header=expected_header)
    stream_handles = dict(_cache_stream_handles(cache))
    stream_fds = {name: handle.fileno() for name, handle in stream_handles.items()}
    header_path = root / "header.json"
    expected_header_bytes = _independent_canonical_json(expected_header) + b"\n"
    assert not header_path.exists()
    assert all(not handle.closed for handle in stream_handles.values())

    real_fsync = os.fsync
    fsync_events: list[tuple[int, Path]] = []

    def fsync_spy(fd: int) -> None:
        proc_fd = Path("/proc/self/fd") / str(fd)
        resolved_path = Path(os.readlink(proc_fd)).resolve()
        if resolved_path.name in _STREAM_PATHS:
            handle = stream_handles[resolved_path.name]
            assert fd == stream_fds[resolved_path.name]
            assert os.fstat(fd).st_size == handle.tell()
            assert not header_path.exists()
        elif resolved_path == header_path.resolve():
            assert all(handle.closed for handle in stream_handles.values())
        fsync_events.append((fd, resolved_path))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync_spy)
    real_open_readonly = StreamingFitCache.__dict__["open_readonly"].__func__
    open_readonly_calls: list[Path] = []

    def validating_open_readonly(
        cls: type[StreamingFitCache], candidate_root: str | Path
    ) -> StreamingFitCache:
        resolved_root = Path(candidate_root).resolve()
        open_readonly_calls.append(resolved_root)
        event_paths = [path for _, path in fsync_events]
        assert len(event_paths) == 7
        assert set(event_paths[:5]) == {
            (root / name).resolve() for name in _STREAM_PATHS
        }
        assert event_paths[5] == header_path.resolve()
        assert event_paths[6] == root.resolve()
        assert header_path.read_bytes() == expected_header_bytes
        validated = real_open_readonly(cls, candidate_root)
        assert validated is not cache
        return validated

    monkeypatch.setattr(
        StreamingFitCache,
        "open_readonly",
        classmethod(validating_open_readonly),
    )

    returned_header = cache.finalize()

    assert returned_header == expected_header
    assert header_path.read_bytes() == expected_header_bytes
    assert open_readonly_calls == [root.resolve()]
    assert len(fsync_events) == 7
    assert {fd for fd, _ in fsync_events[:5]} == set(stream_fds.values())
    assert [path for _, path in fsync_events][5:] == [
        header_path.resolve(),
        root.resolve(),
    ]
    assert all(handle.closed for handle in stream_handles.values())
    assert cache._state == "finalized"
    entries = {path.name: path for path in root.iterdir()}
    assert set(entries) == _STREAM_PATHS | {"header.json"}
    assert all(path.is_file() and not path.is_symlink() for path in entries.values())


def test_finalize_rejects_corrupt_stream_before_header_or_independent_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "fit-cache"
    cache, _fit = _populate_one_fit_cache(root)
    stream_handles = dict(_cache_stream_handles(cache))
    center_handle = stream_handles["centers.f4"]
    center_handle.flush()
    center_path = root / "centers.f4"
    center_bytes = center_path.read_bytes()
    assert len(center_bytes) > 1
    center_path.write_bytes(center_bytes[:-1])
    open_readonly_calls: list[Path] = []

    def forbidden_open_readonly(
        _cls: type[StreamingFitCache], candidate_root: str | Path
    ) -> StreamingFitCache:
        open_readonly_calls.append(Path(candidate_root).resolve())
        raise AssertionError("open_readonly must follow successful final validation")

    monkeypatch.setattr(
        StreamingFitCache,
        "open_readonly",
        classmethod(forbidden_open_readonly),
    )

    with pytest.raises(ValueError):
        cache.finalize()

    assert open_readonly_calls == []
    assert not (root / "header.json").exists()
    assert all(handle.closed for handle in stream_handles.values())


@pytest.mark.parametrize("missing", ("fit", "timing"))
def test_incomplete_finalize_is_side_effect_safe_and_remains_completable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    root = tmp_path / "fit-cache"
    inventory = [{"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2}]
    if missing == "fit":
        inventory.append(
            {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2}
        )
    header = _valid_header(inventory=inventory)
    if missing == "fit":
        cache, fit = _populate_one_fit_cache(root, header=header)
    else:
        cache = StreamingFitCache.create(root, header=header)
        fit = _append_synthetic_pair_fit(cache, 0)
    handles = _cache_stream_handles(cache)
    before = _cache_mutation_snapshot(cache)
    real_fsync = os.fsync
    fsync_calls: list[int] = []

    def tracked_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracked_fsync)

    with pytest.raises(ValueError):
        cache.finalize()

    assert fsync_calls == []
    assert _cache_mutation_snapshot(cache) == before
    assert _cache_stream_handles(cache) == handles
    assert all(not handle.closed for _, handle in handles)
    assert cache._state == "writing"
    assert not (root / "header.json").exists()

    if missing == "fit":
        fit = _append_synthetic_pair_fit(cache, 1)
        timing_inventory_index = 1
    else:
        timing_inventory_index = 0
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=timing_inventory_index,
        cluster_count=2,
        elapsed_ns=2_345_678,
        n_iter=fit.n_iter,
    )

    returned_header = cache.finalize()

    assert returned_header == header
    assert cache._state == "finalized"
    assert (root / "header.json").read_bytes() == (
        _independent_canonical_json(header) + b"\n"
    )


def test_finalized_cache_rejects_writes_but_keeps_exact_reads_and_release(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    cache, fit = _populate_one_fit_cache(root)
    _, expected_assignments, excluded_query_ids = _synthetic_pair_fit(0)
    expected_semantics = _fit_record_semantics(
        fit=fit,
        assignments=expected_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    fit_key = dataclasses.replace(fit.key)
    cache.finalize()

    def file_snapshot() -> tuple[tuple[str, bytes, int, int], ...]:
        rows: list[tuple[str, bytes, int, int]] = []
        for path in sorted(root.iterdir()):
            status = path.stat()
            rows.append(
                (path.name, path.read_bytes(), status.st_size, status.st_mtime_ns)
            )
        return tuple(rows)

    finalized_files = file_snapshot()
    assert len(finalized_files) == 6
    candidate_fit, candidate_assignments, candidate_excluded = _synthetic_pair_fit(0)

    with pytest.raises((RuntimeError, ValueError), match="read.?only|finalized"):
        cache.append_fit(
            candidate_fit,
            candidate_assignments,
            exclusion_kind="pair",
            inventory_index=0,
            excluded_query_ids=candidate_excluded,
        )
    assert file_snapshot() == finalized_files

    with pytest.raises((RuntimeError, ValueError), match="read.?only|finalized"):
        cache.append_timing(
            exclusion_kind="pair",
            inventory_index=0,
            cluster_count=2,
            elapsed_ns=3_456_789,
            n_iter=fit.n_iter,
        )
    assert file_snapshot() == finalized_files

    with pytest.raises((RuntimeError, ValueError), match="read.?only|finalized"):
        cache.finalize()
    assert file_snapshot() == finalized_files
    assert cache._state == "finalized"

    assert cache.release(fit_key) is None
    assert file_snapshot() == finalized_files
    loaded = cache.load_fit(fit_key)
    assert isinstance(loaded, StoredFitRecord)
    assert (
        _fit_record_semantics(
            fit=loaded.fit,
            assignments=loaded.assignments,
            exclusion_kind=loaded.exclusion_kind,
            inventory_index=loaded.inventory_index,
            excluded_query_ids=loaded.excluded_query_ids,
        )
        == expected_semantics
    )
    cache.release(fit_key)
    del loaded
    loaded_by_inventory = cache.load_by_inventory(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
    )
    assert (
        _fit_record_semantics(
            fit=loaded_by_inventory.fit,
            assignments=loaded_by_inventory.assignments,
            exclusion_kind=loaded_by_inventory.exclusion_kind,
            inventory_index=loaded_by_inventory.inventory_index,
            excluded_query_ids=loaded_by_inventory.excluded_query_ids,
        )
        == expected_semantics
    )
    assert file_snapshot() == finalized_files


def test_header_values_remain_owned_by_cache_through_finalize(tmp_path: Path) -> None:
    root = tmp_path / "fit-cache"
    caller_header = _valid_header()
    expected_header = copy.deepcopy(caller_header)
    cache, _fit = _populate_one_fit_cache(root, header=caller_header)

    caller_header["query_ids"][0] = "caller-mutated"
    caller_header["shard_schemas"]["centers"]["path"] = "caller.f4"
    caller_header["expected_inventory"][0]["inventory_index"] = 99
    caller_header["expected_counts"]["fit_count"] = 99
    assert caller_header != expected_header
    assert cache._header == expected_header

    returned_header = cache.finalize()
    expected_bytes = _independent_canonical_json(expected_header) + b"\n"
    assert returned_header == expected_header
    assert cache._header == expected_header
    assert (root / "header.json").read_bytes() == expected_bytes

    returned_header["query_ids"][0] = "return-mutated"
    returned_header["shard_schemas"]["labels"]["path"] = "return.u2"
    returned_header["expected_inventory"][0]["cluster_count"] = 99
    returned_header["expected_counts"]["timing_record_count"] = 99
    assert returned_header != expected_header
    assert cache._header == expected_header
    assert (root / "header.json").read_bytes() == expected_bytes


@pytest.fixture
def populated_cache(tmp_path: Path) -> Path:
    """Build one valid C2 pair cache exclusively through the public write API."""
    root = tmp_path / "populated-fit-cache"
    cache, _fit = _populate_one_fit_cache(root)

    returned_header = cache.finalize()

    assert returned_header == _valid_header()
    return root


@pytest.fixture
def two_fit_cache(tmp_path: Path) -> Path:
    """Build two ordered C2 pair fits exclusively through the public API."""
    root = tmp_path / "two-fit-cache"
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    header = _valid_header(inventory=inventory)
    cache = StreamingFitCache.create(root, header=header)
    for inventory_index, elapsed_ns in ((0, 1_234_567), (1, 2_345_678)):
        fit = _append_synthetic_pair_fit(cache, inventory_index)
        cache.append_timing(
            exclusion_kind="pair",
            inventory_index=inventory_index,
            cluster_count=2,
            elapsed_ns=elapsed_ns,
            n_iter=fit.n_iter,
        )

    assert cache.finalize() == header
    return root


def _finalized_file_snapshot(root: Path) -> tuple[tuple[str, bytes, int, int], ...]:
    return tuple(
        (
            path.name,
            path.read_bytes(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.iterdir())
    )


def _assert_stored_fit_record_exact(
    record: StoredFitRecord,
    *,
    fit: CanonicalClusterFit,
    assignments: tuple[TargetAssignment, ...],
    exclusion_kind: str,
    inventory_index: int,
    excluded_query_ids: tuple[str, ...],
) -> None:
    assert _fit_record_semantics(
        fit=record.fit,
        assignments=record.assignments,
        exclusion_kind=record.exclusion_kind,
        inventory_index=record.inventory_index,
        excluded_query_ids=record.excluded_query_ids,
    ) == _fit_record_semantics(
        fit=fit,
        assignments=assignments,
        exclusion_kind=exclusion_kind,
        inventory_index=inventory_index,
        excluded_query_ids=excluded_query_ids,
    )
    assert record.fit.key.support_query_ids == _support_without(excluded_query_ids)
    assert record.fit.centers.tobytes(order="C") == np.ascontiguousarray(
        fit.centers, dtype="<f4"
    ).tobytes(order="C")
    assert record.fit.labels.tobytes(order="C") == np.ascontiguousarray(
        fit.labels, dtype="<u2"
    ).tobytes(order="C")
    assert record.fit.center_sha256 == _independent_center_sha256(record.fit.centers)

    labels_derived_memberships = tuple(
        tuple(
            record.fit.key.support_query_ids[support_index]
            for support_index, label in enumerate(record.fit.labels)
            if int(label) == cluster_id
        )
        for cluster_id in range(record.fit.key.cluster_count)
    )
    membership_rows = tuple(
        (cluster_id, labels_derived_memberships[cluster_id])
        for cluster_id in range(record.fit.key.cluster_count)
    )
    assert record.fit.memberships == labels_derived_memberships
    assert record.fit.fit_membership_sha256 == _independent_membership_sha256(
        membership_rows
    )
    assert record.fit.cluster_membership_sha256 == tuple(
        _independent_membership_sha256((membership_rows[cluster_id],))
        for cluster_id in range(record.fit.key.cluster_count)
    )

    expected_inertia = np.ascontiguousarray(fit.cluster_inertia, dtype="<f8")
    observed_inertia = np.ascontiguousarray(record.fit.cluster_inertia, dtype="<f8")
    assert observed_inertia.tobytes(order="C") == expected_inertia.tobytes(order="C")
    assert (
        record.fit.recomputed_total_inertia.hex()
        == math.fsum(record.fit.cluster_inertia).hex()
    )
    assert record.fit.sklearn_inertia.hex() == record.fit.recomputed_total_inertia.hex()

    assert len(record.assignments) == len(excluded_query_ids)
    for assignment, excluded_query_id in zip(
        record.assignments, excluded_query_ids, strict=True
    ):
        expected_member_indices = tuple(
            _QUERY_IDS.index(query_id)
            for query_id in labels_derived_memberships[assignment.canonical_cluster_id]
        )
        assert assignment.query_id == excluded_query_id
        assert assignment.member_indices == expected_member_indices
        assert assignment.cluster_support == len(expected_member_indices)


def _different_sha256(value: str) -> str:
    replacement = "0" if value[0] != "0" else "1"
    return replacement + value[1:]


def _mutate_single_fit_receipt(root: Path, case: str) -> None:
    receipt_path = root / "fit_receipts.jsonl"
    lines = receipt_path.read_bytes().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert isinstance(record, dict)
    centers = record["centers"]
    labels = record["labels"]
    inertia = record["cluster_inertia"]
    assert isinstance(centers, dict)
    assert isinstance(labels, dict)
    assert isinstance(inertia, dict)

    if case == "receipt-schema":
        record["schema"] = "trec-mlm-dense-aggregation-fit-receipt-v2"
    elif case == "excluded-query-order":
        record["excluded_query_ids"] = list(reversed(record["excluded_query_ids"]))
    elif case == "cluster-count":
        record["cluster_count"] = 3
    elif case == "fit-key-sha256":
        record["fit_key_sha256"] = _different_sha256(record["fit_key_sha256"])
    elif case == "support-query-ids-sha256":
        record["support_query_ids_sha256"] = _different_sha256(
            record["support_query_ids_sha256"]
        )
    elif case == "embedding-sha256":
        record["embedding_sha256"] = _different_sha256(record["embedding_sha256"])
    elif case == "clustering-contract-sha256":
        record["clustering_contract_sha256"] = _different_sha256(
            record["clustering_contract_sha256"]
        )
    elif case == "centers-offset":
        centers["offset"] = 1
    elif case == "centers-shape":
        centers["shape"] = [3, 2]
    elif case == "centers-dtype":
        centers["dtype"] = ">f4"
    elif case == "centers-bytes":
        centers["bytes"] -= 1
    elif case == "labels-offset":
        labels["offset"] = 1
    elif case == "labels-shape":
        labels["shape"] = [1, 2]
    elif case == "labels-dtype":
        labels["dtype"] = ">u2"
    elif case == "labels-bytes":
        labels["bytes"] -= 1
    elif case == "cluster-inertia-offset":
        inertia["offset"] = 1
    elif case == "cluster-inertia-shape":
        inertia["shape"] = [1, 2]
    elif case == "cluster-inertia-dtype":
        inertia["dtype"] = ">f8"
    elif case == "cluster-inertia-bytes":
        inertia["bytes"] -= 1
    elif case == "center-sha256":
        record["center_sha256"] = _different_sha256(record["center_sha256"])
    elif case == "fit-membership-sha256":
        record["fit_membership_sha256"] = _different_sha256(
            record["fit_membership_sha256"]
        )
    elif case == "cluster-membership-sha256":
        record["cluster_membership_sha256"][0] = _different_sha256(
            record["cluster_membership_sha256"][0]
        )
    elif case == "cluster-inertia-hex":
        record["cluster_inertia_hex"][0] = (
            float.fromhex(record["cluster_inertia_hex"][0]) + 1.0
        ).hex()
    elif case == "sklearn-inertia-hex":
        record["sklearn_inertia_hex"] = (
            float.fromhex(record["sklearn_inertia_hex"]) + 1.0
        ).hex()
    elif case == "recomputed-total-inertia-hex":
        record["recomputed_total_inertia_hex"] = (
            float.fromhex(record["recomputed_total_inertia_hex"]) + 1.0
        ).hex()
    elif case == "assignment-schema":
        record["assignments"][0]["member_indices"] = [2]
    elif case == "assignment-semantics":
        record["assignments"][0]["cluster_support"] += 1
    elif case == "record-sha256":
        record["record_sha256"] = _different_sha256(record["record_sha256"])
    else:
        raise AssertionError(f"unknown semantic mutation: {case}")

    if case != "record-sha256":
        record_without_hash = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        record["record_sha256"] = _independent_framed_sha256(
            "fit-cache-record",
            _independent_canonical_json(record_without_hash),
        )
    receipt_path.write_bytes(_independent_canonical_json(record) + b"\n")


def test_open_readonly_reconstructs_exact_pair_record(
    populated_cache: Path,
) -> None:
    expected_header = _valid_header()
    expected_fit, expected_assignments, expected_excluded = _synthetic_pair_fit(0)
    expected_names = _STREAM_PATHS | {"header.json"}

    entries = {path.name: path for path in populated_cache.iterdir()}
    assert set(entries) == expected_names
    assert all(path.is_file() and not path.is_symlink() for path in entries.values())
    assert entries["header.json"].read_bytes() == (
        _independent_canonical_json(expected_header) + b"\n"
    )
    before = _finalized_file_snapshot(populated_cache)

    readonly = StreamingFitCache.open_readonly(populated_cache)

    assert readonly._state == "finalized"
    assert readonly._header == expected_header
    loaded_by_key = readonly.load_fit(dataclasses.replace(expected_fit.key))
    assert isinstance(loaded_by_key, StoredFitRecord)
    _assert_stored_fit_record_exact(
        loaded_by_key,
        fit=expected_fit,
        assignments=expected_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=expected_excluded,
    )
    assert readonly.release(dataclasses.replace(expected_fit.key)) is None
    loaded_by_inventory = readonly.load_by_inventory(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
    )
    _assert_stored_fit_record_exact(
        loaded_by_inventory,
        fit=expected_fit,
        assignments=expected_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=expected_excluded,
    )
    assert _finalized_file_snapshot(populated_cache) == before


def test_open_readonly_reconstructs_exact_singleton_c1_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "singleton-fit-cache"
    inventory = [{"exclusion_kind": "single", "inventory_index": 1, "cluster_count": 1}]
    header = _valid_header(inventory=inventory)
    fit, assignments, excluded_query_ids = _synthetic_single_c1_fit(1)
    cache = StreamingFitCache.create(root, header=header)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="single",
        inventory_index=1,
        excluded_query_ids=excluded_query_ids,
    )
    cache.append_timing(
        exclusion_kind="single",
        inventory_index=1,
        cluster_count=1,  # type: ignore[arg-type]
        elapsed_ns=7_654_321,
        n_iter=fit.n_iter,
    )
    assert cache.finalize() == header

    readonly = StreamingFitCache.open_readonly(root)
    loaded = readonly.load_by_inventory(
        exclusion_kind="single",
        inventory_index=1,
        cluster_count=1,  # type: ignore[arg-type]
    )

    _assert_stored_fit_record_exact(
        loaded,
        fit=fit,
        assignments=assignments,
        exclusion_kind="single",
        inventory_index=1,
        excluded_query_ids=("q2",),
    )
    assert loaded.fit.key.support_query_ids == ("q1", "q3", "q4")
    assert loaded.fit.memberships == (("q1", "q3", "q4"),)
    assert loaded.assignments[0].query_id == "q2"
    assert loaded.assignments[0].member_indices == (0, 2, 3)
    assert loaded.assignments[0].cluster_support == 3
    assert readonly.release(fit.key) is None
    loaded_by_key = readonly.load_fit(dataclasses.replace(fit.key))
    assert isinstance(loaded_by_key, StoredFitRecord)
    _assert_stored_fit_record_exact(
        loaded_by_key,
        fit=fit,
        assignments=assignments,
        exclusion_kind="single",
        inventory_index=1,
        excluded_query_ids=("q2",),
    )


def test_readonly_cache_rejects_mutation_without_file_changes(
    populated_cache: Path,
) -> None:
    readonly = StreamingFitCache.open_readonly(populated_cache)
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    before = _finalized_file_snapshot(populated_cache)

    with pytest.raises((RuntimeError, ValueError), match="finalized"):
        readonly.append_fit(
            fit,
            assignments,
            exclusion_kind="pair",
            inventory_index=0,
            excluded_query_ids=excluded_query_ids,
        )
    assert _finalized_file_snapshot(populated_cache) == before

    with pytest.raises((RuntimeError, ValueError), match="finalized"):
        readonly.append_timing(
            exclusion_kind="pair",
            inventory_index=0,
            cluster_count=2,
            elapsed_ns=8_765_432,
            n_iter=fit.n_iter,
        )
    assert _finalized_file_snapshot(populated_cache) == before

    with pytest.raises((RuntimeError, ValueError), match="finalized"):
        readonly.finalize()
    assert _finalized_file_snapshot(populated_cache) == before
    assert readonly._state == "finalized"

    loaded = readonly.load_fit(dataclasses.replace(fit.key))
    assert isinstance(loaded, StoredFitRecord)
    assert readonly.release(dataclasses.replace(fit.key)) is None
    loaded_by_inventory = readonly.load_by_inventory(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
    )
    _assert_stored_fit_record_exact(
        loaded_by_inventory,
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    assert _finalized_file_snapshot(populated_cache) == before


def test_readonly_load_and_release_compare_full_key_under_digest(
    populated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readonly = StreamingFitCache.open_readonly(populated_cache)
    fit, _assignments, _excluded_query_ids = _synthetic_pair_fit(0)
    stored_sha256 = _independent_fit_key_sha256(fit.key)
    colliding_key = dataclasses.replace(
        fit.key,
        support_query_ids=tuple(reversed(fit.key.support_query_ids)),
    )
    assert colliding_key != fit.key
    monkeypatch.setattr(cache_module, "_fit_key_sha256", lambda _key: stored_sha256)
    before = _finalized_file_snapshot(populated_cache)

    with pytest.raises(ValueError, match="collision"):
        readonly.load_fit(colliding_key)
    with pytest.raises(ValueError, match="collision"):
        readonly.release(colliding_key)

    loaded = readonly.load_fit(dataclasses.replace(fit.key))
    assert isinstance(loaded, StoredFitRecord)
    assert loaded.fit.key == fit.key
    assert _finalized_file_snapshot(populated_cache) == before


@pytest.mark.parametrize(
    "case",
    (
        "receipt-schema",
        "excluded-query-order",
        "cluster-count",
        "fit-key-sha256",
        "support-query-ids-sha256",
        "embedding-sha256",
        "clustering-contract-sha256",
        "centers-offset",
        "centers-shape",
        "centers-dtype",
        "centers-bytes",
        "labels-offset",
        "labels-shape",
        "labels-dtype",
        "labels-bytes",
        "cluster-inertia-offset",
        "cluster-inertia-shape",
        "cluster-inertia-dtype",
        "cluster-inertia-bytes",
        "center-sha256",
        "fit-membership-sha256",
        "cluster-membership-sha256",
        "cluster-inertia-hex",
        "sklearn-inertia-hex",
        "recomputed-total-inertia-hex",
        "assignment-schema",
        "assignment-semantics",
    ),
)
def test_open_readonly_revalidates_every_receipt_semantic(
    populated_cache: Path,
    case: str,
) -> None:
    _mutate_single_fit_receipt(populated_cache, case)

    with pytest.raises(ValueError):
        StreamingFitCache.open_readonly(populated_cache)


_FINAL_CACHE_PATHS = tuple(sorted(_STREAM_PATHS | {"header.json"}))


def _assert_open_rejected(root: Path, *, match: str | None = None) -> None:
    with pytest.raises(ValueError, match=match):
        StreamingFitCache.open_readonly(root)


def test_cache_rejects_truncated_center_shard(populated_cache: Path) -> None:
    center_path = populated_cache / "centers.f4"
    center_bytes = center_path.read_bytes()
    assert center_bytes
    center_path.write_bytes(center_bytes[:-1])

    _assert_open_rejected(populated_cache, match=r"center.*byte length")


@pytest.mark.parametrize("entry_name", _FINAL_CACHE_PATHS)
def test_cache_rejects_each_missing_required_entry(
    populated_cache: Path,
    entry_name: str,
) -> None:
    entry = populated_cache / entry_name
    entry.unlink()
    assert not entry.exists() and not entry.is_symlink()

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("replacement_kind", ("directory", "symlink"))
@pytest.mark.parametrize("entry_name", _FINAL_CACHE_PATHS)
def test_cache_rejects_each_nonregular_required_entry(
    populated_cache: Path,
    entry_name: str,
    replacement_kind: str,
) -> None:
    entry = populated_cache / entry_name
    original_bytes = entry.read_bytes()
    entry.unlink()

    if replacement_kind == "directory":
        entry.mkdir()
        assert stat.S_ISDIR(os.lstat(entry).st_mode)
    else:
        external = populated_cache.parent / f"external-{entry_name}"
        external.write_bytes(original_bytes)
        entry.symlink_to(external)
        assert stat.S_ISLNK(os.lstat(entry).st_mode)
        assert entry.resolve() == external.resolve()

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("extra_kind", ("regular", "directory", "symlink"))
def test_cache_rejects_each_extra_entry_type(
    populated_cache: Path,
    extra_kind: str,
) -> None:
    extra = populated_cache / "unexpected-entry"
    if extra_kind == "regular":
        extra.write_bytes(b"unexpected\n")
        assert stat.S_ISREG(os.lstat(extra).st_mode)
    elif extra_kind == "directory":
        extra.mkdir()
        assert stat.S_ISDIR(os.lstat(extra).st_mode)
    else:
        external = populated_cache.parent / "external-unexpected-entry"
        external.write_bytes(b"unexpected\n")
        extra.symlink_to(external)
        assert stat.S_ISLNK(os.lstat(extra).st_mode)
        assert extra.resolve() == external.resolve()

    _assert_open_rejected(populated_cache)


def test_cache_rejects_symlinked_root(populated_cache: Path) -> None:
    backing_root = populated_cache.with_name("populated-fit-cache-backing")
    populated_cache.rename(backing_root)
    populated_cache.symlink_to(backing_root, target_is_directory=True)
    assert stat.S_ISLNK(os.lstat(populated_cache).st_mode)
    assert populated_cache.resolve() == backing_root.resolve()

    _assert_open_rejected(populated_cache)


_FIFO_OPEN_SCRIPT = """
from pathlib import Path
import sys

from trec_mlm.dense_aggregation_cache import StreamingFitCache

try:
    StreamingFitCache.open_readonly(Path(sys.argv[1]))
except (OSError, ValueError):
    raise SystemExit(0)
raise SystemExit(3)
"""


def _assert_fifo_open_rejected(root: Path) -> None:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _FIFO_OPEN_SCRIPT, str(root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("open_readonly blocked while validating a FIFO cache entry")

    assert completed.returncode == 0, (
        f"FIFO validation subprocess exited {completed.returncode}: "
        f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )


@pytest.mark.parametrize("entry_name", _FINAL_CACHE_PATHS)
def test_cache_rejects_each_required_fifo_without_blocking(
    populated_cache: Path,
    entry_name: str,
) -> None:
    entry = populated_cache / entry_name
    entry.unlink()
    os.mkfifo(entry)
    assert stat.S_ISFIFO(os.lstat(entry).st_mode)

    _assert_fifo_open_rejected(populated_cache)


def test_cache_rejects_extra_fifo_without_blocking(populated_cache: Path) -> None:
    extra = populated_cache / "unexpected-fifo"
    os.mkfifo(extra)
    assert stat.S_ISFIFO(os.lstat(extra).st_mode)

    _assert_fifo_open_rejected(populated_cache)


def _flip_one_byte(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    assert payload
    index = len(payload) // 2
    payload[index] ^= 0x01
    path.write_bytes(payload)


@pytest.mark.parametrize(
    ("shard_name", "mutation"),
    (
        ("centers.f4", "append"),
        ("centers.f4", "flip"),
        ("labels.u2", "truncate"),
        ("labels.u2", "append"),
        ("labels.u2", "flip"),
        ("cluster_inertia.f8", "truncate"),
        ("cluster_inertia.f8", "append"),
        ("cluster_inertia.f8", "flip"),
    ),
)
def test_cache_rejects_binary_shard_byte_corruption(
    populated_cache: Path,
    shard_name: str,
    mutation: str,
) -> None:
    shard = populated_cache / shard_name
    payload = shard.read_bytes()
    assert payload
    if mutation == "truncate":
        shard.write_bytes(payload[:-1])
    elif mutation == "append":
        shard.write_bytes(payload + b"\x00")
    else:
        assert mutation == "flip"
        _flip_one_byte(shard)

    _assert_open_rejected(populated_cache)


def test_cache_rejects_self_consistent_valid_width_out_of_range_label(
    tmp_path: Path,
) -> None:
    root = tmp_path / "out-of-range-label-cache"
    inventory = [{"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 1}]
    header = _valid_header(inventory=inventory)
    fit, assignments, excluded_query_ids = _synthetic_c1_pair_fit()
    cache = StreamingFitCache.create(root, header=header)
    cache.append_fit(
        fit,
        assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    cache.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=1,  # type: ignore[arg-type]
        elapsed_ns=1_234_567,
        n_iter=fit.n_iter,
    )
    assert cache.finalize() == header

    label_path = root / "labels.u2"
    original = label_path.read_bytes()
    labels = np.frombuffer(original, dtype="<u2").copy()
    assert labels.tolist() == [0, 0]
    labels[1] = fit.key.cluster_count
    corrupted = labels.tobytes(order="C")
    assert len(corrupted) == len(original)
    label_path.write_bytes(corrupted)

    retained_memberships = ((0, (fit.key.support_query_ids[0],)),)

    def mutate(record: dict[str, Any]) -> None:
        assert record["cluster_count"] == fit.key.cluster_count == 1
        record["fit_membership_sha256"] = _independent_membership_sha256(
            retained_memberships
        )
        record["cluster_membership_sha256"] = [
            _independent_membership_sha256(retained_memberships)
        ]
        for assignment in record["assignments"]:
            assert assignment["canonical_cluster_id"] == 0
            assignment["cluster_support"] = 1

    _rewrite_fit_receipt(root, mutate, rehash=True)
    rewritten = _read_jsonl_records(root / "fit_receipts.jsonl")
    assert rewritten[0]["fit_membership_sha256"] == _independent_membership_sha256(
        retained_memberships
    )
    assert np.frombuffer(label_path.read_bytes(), dtype="<u2").tolist() == [0, 1]

    _assert_open_rejected(root)


def test_cache_rejects_self_consistent_wrong_center_shape(
    populated_cache: Path,
) -> None:
    center_path = populated_cache / "centers.f4"
    original = center_path.read_bytes()
    centers = np.frombuffer(original, dtype="<f4").reshape(1, 6)

    def mutate(record: dict[str, Any]) -> None:
        descriptor = record["centers"]
        assert isinstance(descriptor, dict)
        assert descriptor["shape"] == [2, 3]
        descriptor["shape"] = [1, 6]
        record["center_sha256"] = _independent_center_sha256(centers)

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)
    (rewritten,) = _read_jsonl_records(populated_cache / "fit_receipts.jsonl")
    assert rewritten["centers"]["shape"] == [1, 6]
    assert rewritten["center_sha256"] == _independent_center_sha256(centers)
    assert center_path.read_bytes() == original

    _assert_open_rejected(populated_cache)


def _corrupt_json_bytes(path: Path, case: str) -> None:
    original = path.read_bytes()
    assert original.endswith(b"\n")
    assert len(original) > 2

    if case == "truncated":
        corrupted = original[:-2] + b"\n"
    elif case == "malformed":
        corrupted = b"{\xff}\n"
    elif case == "valid-noncanonical":
        parsed = json.loads(original.decode("utf-8"))
        corrupted = (
            json.dumps(
                parsed,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=False,
                separators=(", ", ": "),
            ).encode("utf-8")
            + b"\n"
        )
        assert json.loads(corrupted.decode("utf-8")) == parsed
        assert corrupted != _independent_canonical_json(parsed) + b"\n"
    elif case == "extra-blank-line":
        corrupted = original + b"\n"
    else:
        assert case == "missing-terminal-newline"
        corrupted = original[:-1]

    path.write_bytes(corrupted)


@pytest.mark.parametrize(
    "case",
    (
        "truncated",
        "malformed",
        "valid-noncanonical",
        "extra-blank-line",
        "missing-terminal-newline",
    ),
)
@pytest.mark.parametrize(
    "entry_name",
    ("header.json", "fit_receipts.jsonl", "timings.jsonl"),
)
def test_cache_rejects_noncanonical_or_malformed_json_bytes(
    populated_cache: Path,
    entry_name: str,
    case: str,
) -> None:
    _corrupt_json_bytes(populated_cache / entry_name, case)

    _assert_open_rejected(populated_cache)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    records: list[dict[str, Any]] = []
    for line in payload.splitlines():
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        records.append(parsed)
    return records


def _write_canonical_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_bytes(
        b"".join(_independent_canonical_json(record) + b"\n" for record in records)
    )


def _rehash_fit_receipt(record: dict[str, Any]) -> None:
    record_without_hash = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = _independent_framed_sha256(
        "fit-cache-record",
        _independent_canonical_json(record_without_hash),
    )


def _rewrite_fit_receipt(
    root: Path,
    mutate: Callable[[dict[str, Any]], None],
    *,
    row_index: int = 0,
    rehash: bool = True,
) -> None:
    path = root / "fit_receipts.jsonl"
    records = _read_jsonl_records(path)
    mutate(records[row_index])
    if rehash:
        _rehash_fit_receipt(records[row_index])
    _write_canonical_jsonl(path, records)


def _rewrite_timing(
    root: Path,
    mutate: Callable[[dict[str, Any]], None],
    *,
    row_index: int = 0,
) -> None:
    path = root / "timings.jsonl"
    records = _read_jsonl_records(path)
    mutate(records[row_index])
    _write_canonical_jsonl(path, records)


def _rewrite_header(
    root: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    path = root / "header.json"
    parsed = json.loads(path.read_bytes())
    assert isinstance(parsed, dict)
    mutate(parsed)
    path.write_bytes(_independent_canonical_json(parsed) + b"\n")


_DESCRIPTOR_ITEM_BYTES = {
    "centers": 4,
    "labels": 2,
    "cluster_inertia": 8,
}
_DESCRIPTOR_ALT_SHAPES = {
    "centers": [1, 6],
    "labels": [2, 1],
    "cluster_inertia": [2, 1],
}
_DESCRIPTOR_ALT_DTYPES = {
    "centers": "<i4",
    "labels": "<i2",
    "cluster_inertia": "<u8",
}


@pytest.mark.parametrize(
    ("shard_name", "field"),
    tuple(
        (shard_name, field)
        for shard_name in ("centers", "labels", "cluster_inertia")
        for field in ("offset", "bytes", "shape", "dtype")
    ),
)
def test_cache_rejects_rehashed_receipt_descriptor_corruption(
    populated_cache: Path,
    shard_name: str,
    field: str,
) -> None:
    def mutate(record: dict[str, Any]) -> None:
        descriptor = record[shard_name]
        assert isinstance(descriptor, dict)
        if field in {"offset", "bytes"}:
            descriptor[field] += _DESCRIPTOR_ITEM_BYTES[shard_name]
        elif field == "shape":
            descriptor[field] = _DESCRIPTOR_ALT_SHAPES[shard_name]
        else:
            assert field == "dtype"
            descriptor[field] = _DESCRIPTOR_ALT_DTYPES[shard_name]

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("shard_name", ("centers", "labels", "cluster_inertia"))
@pytest.mark.parametrize("continuity_error", ("gap", "overlap"))
def test_cache_rejects_two_fit_descriptor_gap_or_overlap(
    two_fit_cache: Path,
    shard_name: str,
    continuity_error: str,
) -> None:
    delta = _DESCRIPTOR_ITEM_BYTES[shard_name]
    if continuity_error == "overlap":
        delta = -delta

    def mutate(record: dict[str, Any]) -> None:
        descriptor = record[shard_name]
        assert isinstance(descriptor, dict)
        descriptor["offset"] += delta

    _rewrite_fit_receipt(two_fit_cache, mutate, row_index=1, rehash=True)

    _assert_open_rejected(two_fit_cache)


def test_cache_rejects_balanced_out_of_order_identical_inertia_descriptors(
    two_fit_cache: Path,
) -> None:
    receipt_path = two_fit_cache / "fit_receipts.jsonl"
    inertia_path = two_fit_cache / "cluster_inertia.f8"
    records = _read_jsonl_records(receipt_path)
    assert len(records) == 2
    first_descriptor = records[0]["cluster_inertia"]
    second_descriptor = records[1]["cluster_inertia"]
    assert isinstance(first_descriptor, dict)
    assert isinstance(second_descriptor, dict)
    assert first_descriptor["bytes"] == second_descriptor["bytes"]
    byte_count = first_descriptor["bytes"]
    assert isinstance(byte_count, int) and not isinstance(byte_count, bool)
    assert byte_count > 0

    payload = inertia_path.read_bytes()
    first_offset = first_descriptor["offset"]
    second_offset = second_descriptor["offset"]
    assert first_offset == 0
    assert second_offset == first_offset + byte_count
    assert second_offset + byte_count == len(payload)
    first_payload = payload[first_offset : first_offset + byte_count]
    second_payload = payload[second_offset : second_offset + byte_count]
    assert first_payload == second_payload == b"\x00" * byte_count

    first_descriptor["offset"], second_descriptor["offset"] = (
        second_offset,
        first_offset,
    )
    for record in records:
        _rehash_fit_receipt(record)
    _write_canonical_jsonl(receipt_path, records)

    rewritten = _read_jsonl_records(receipt_path)
    assert [record["sequence"] for record in rewritten] == [0, 1]
    rewritten_descriptors = [record["cluster_inertia"] for record in rewritten]
    assert all(isinstance(descriptor, dict) for descriptor in rewritten_descriptors)
    spans = sorted(
        (
            descriptor["offset"],
            descriptor["offset"] + descriptor["bytes"],
        )
        for descriptor in rewritten_descriptors
    )
    assert spans == [(0, byte_count), (byte_count, len(payload))]
    assert inertia_path.read_bytes() == payload

    _assert_open_rejected(two_fit_cache)


def _corrupt_two_fit_row_inventory(root: Path, stream_name: str, case: str) -> None:
    path = root / stream_name
    records = _read_jsonl_records(path)
    assert len(records) == 2

    if case == "missing-final":
        records.pop()
    elif case == "extra":
        extra = copy.deepcopy(records[-1])
        extra["sequence"] = 2
        if stream_name == "fit_receipts.jsonl":
            _rehash_fit_receipt(extra)
        records.append(extra)
    elif case == "swapped":
        records.reverse()
    else:
        assert case == "logical-duplicate"
        duplicate = copy.deepcopy(records[0])
        duplicate["sequence"] = 1
        if stream_name == "fit_receipts.jsonl":
            shard_paths = {
                "centers": "centers.f4",
                "labels": "labels.u2",
                "cluster_inertia": "cluster_inertia.f8",
            }
            for shard_name, shard_path in shard_paths.items():
                first_descriptor = records[0][shard_name]
                second_descriptor = records[1][shard_name]
                assert isinstance(first_descriptor, dict)
                assert isinstance(second_descriptor, dict)
                assert first_descriptor["bytes"] == second_descriptor["bytes"]
                shard = root / shard_path
                payload = bytearray(shard.read_bytes())
                first_start = first_descriptor["offset"]
                second_start = second_descriptor["offset"]
                byte_count = first_descriptor["bytes"]
                payload[second_start : second_start + byte_count] = payload[
                    first_start : first_start + byte_count
                ]
                shard.write_bytes(payload)
                duplicate[shard_name] = copy.deepcopy(second_descriptor)
            _rehash_fit_receipt(duplicate)
        records[1] = duplicate

    _write_canonical_jsonl(path, records)


@pytest.mark.parametrize(
    "case",
    ("missing-final", "extra", "swapped", "logical-duplicate"),
)
@pytest.mark.parametrize(
    "stream_name",
    ("fit_receipts.jsonl", "timings.jsonl"),
)
def test_cache_rejects_two_fit_row_inventory_corruption(
    two_fit_cache: Path,
    stream_name: str,
    case: str,
) -> None:
    _corrupt_two_fit_row_inventory(two_fit_cache, stream_name, case)

    _assert_open_rejected(two_fit_cache)


@pytest.mark.parametrize(
    "stream_name",
    ("fit_receipts.jsonl", "timings.jsonl"),
)
def test_final_row_corruption_fails_before_return_without_mutating_cache(
    two_fit_cache: Path,
    stream_name: str,
) -> None:
    if stream_name == "fit_receipts.jsonl":
        _rewrite_fit_receipt(
            two_fit_cache,
            lambda record: record.__setitem__(
                "record_sha256", _different_sha256(record["record_sha256"])
            ),
            row_index=1,
            rehash=False,
        )
    else:
        _rewrite_timing(
            two_fit_cache,
            lambda record: record.__setitem__("sequence", 0),
            row_index=1,
        )
    before = _finalized_file_snapshot(two_fit_cache)
    sentinel = object()
    opened: object = sentinel

    with pytest.raises(ValueError):
        opened = StreamingFitCache.open_readonly(two_fit_cache)

    assert opened is sentinel
    assert _finalized_file_snapshot(two_fit_cache) == before


_FIT_RECEIPT_KEYS = {
    "schema",
    "sequence",
    "exclusion_kind",
    "inventory_index",
    "excluded_query_ids",
    "cluster_count",
    "fit_key_sha256",
    "support_query_ids_sha256",
    "embedding_sha256",
    "clustering_contract_sha256",
    "centers",
    "labels",
    "cluster_inertia",
    "center_sha256",
    "fit_membership_sha256",
    "cluster_membership_sha256",
    "cluster_inertia_hex",
    "sklearn_inertia_hex",
    "recomputed_total_inertia_hex",
    "n_iter",
    "assignments",
    "fit_count_increment",
    "start_count_increment",
    "record_sha256",
}
_FIT_DESCRIPTOR_KEYS = {"offset", "bytes", "shape", "dtype"}
_ASSIGNMENT_RECEIPT_KEYS = {
    "query_id",
    "canonical_cluster_id",
    "squared_distance_hex",
    "cluster_support",
}
_RECEIPT_SCALAR_DIGEST_FIELDS = (
    "fit_key_sha256",
    "support_query_ids_sha256",
    "embedding_sha256",
    "clustering_contract_sha256",
    "center_sha256",
    "fit_membership_sha256",
)


@pytest.mark.parametrize("missing_key", tuple(sorted(_FIT_RECEIPT_KEYS)))
def test_cache_rejects_each_missing_receipt_key(
    populated_cache: Path,
    missing_key: str,
) -> None:
    _rewrite_fit_receipt(
        populated_cache,
        lambda record: record.pop(missing_key),
        rehash=missing_key != "record_sha256",
    )

    _assert_open_rejected(populated_cache)


def test_cache_rejects_unknown_receipt_key(populated_cache: Path) -> None:
    _rewrite_fit_receipt(
        populated_cache,
        lambda record: record.__setitem__("unexpected", None),
        rehash=True,
    )

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("shard_name", ("centers", "labels", "cluster_inertia"))
@pytest.mark.parametrize("missing_key", tuple(sorted(_FIT_DESCRIPTOR_KEYS)))
def test_cache_rejects_each_missing_descriptor_key(
    populated_cache: Path,
    shard_name: str,
    missing_key: str,
) -> None:
    def mutate(record: dict[str, Any]) -> None:
        descriptor = record[shard_name]
        assert isinstance(descriptor, dict)
        descriptor.pop(missing_key)

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("shard_name", ("centers", "labels", "cluster_inertia"))
def test_cache_rejects_unknown_descriptor_key(
    populated_cache: Path,
    shard_name: str,
) -> None:
    def mutate(record: dict[str, Any]) -> None:
        descriptor = record[shard_name]
        assert isinstance(descriptor, dict)
        descriptor["unexpected"] = None

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("shard_name", ("centers", "labels", "cluster_inertia"))
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("offset", True),
        ("offset", "same-value-float"),
        ("offset", -1),
        ("bytes", True),
        ("bytes", "same-value-float"),
        ("bytes", -1),
        ("shape", "not-a-shape"),
        ("shape", "first-element-float"),
        ("dtype", 7),
    ),
)
def test_cache_rejects_nonexact_descriptor_field_types(
    populated_cache: Path,
    shard_name: str,
    field: str,
    invalid_value: object,
) -> None:
    def mutate(record: dict[str, Any]) -> None:
        descriptor = record[shard_name]
        assert isinstance(descriptor, dict)
        if invalid_value == "same-value-float":
            descriptor[field] = float(descriptor[field])
        elif invalid_value == "first-element-float":
            shape = descriptor[field]
            assert isinstance(shape, list)
            shape[0] = float(shape[0])
        else:
            descriptor[field] = invalid_value

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)

    _assert_open_rejected(populated_cache)


def _mutate_receipt_contract(record: dict[str, Any], case: str) -> None:
    if case == "schema-type":
        record["schema"] = 7
    elif case == "sequence-bool":
        record["sequence"] = True
    elif case == "sequence-float":
        record["sequence"] = 0.0
    elif case == "sequence-value":
        record["sequence"] = 1
    elif case == "kind-type":
        record["exclusion_kind"] = 7
    elif case == "kind-value":
        record["exclusion_kind"] = "single"
    elif case == "inventory-index-bool":
        record["inventory_index"] = True
    elif case == "inventory-index-float":
        record["inventory_index"] = 0.0
    elif case == "inventory-index-value":
        record["inventory_index"] = 1
    elif case == "excluded-container":
        record["excluded_query_ids"] = "q1,q2"
    elif case == "excluded-item-type":
        record["excluded_query_ids"] = [7, "q2"]
    elif case == "excluded-count":
        record["excluded_query_ids"] = ["q1"]
    elif case == "excluded-duplicate":
        record["excluded_query_ids"] = ["q1", "q1"]
    elif case == "cluster-count-bool":
        record["cluster_count"] = True
    elif case == "cluster-count-float":
        record["cluster_count"] = 2.0
    elif case == "fit-increment-bool":
        record["fit_count_increment"] = True
    elif case == "fit-increment-float":
        record["fit_count_increment"] = 1.0
    elif case == "fit-increment-value":
        record["fit_count_increment"] = 2
    elif case == "start-increment-bool":
        record["start_count_increment"] = True
    elif case == "start-increment-float":
        record["start_count_increment"] = 10.0
    elif case == "start-increment-value":
        record["start_count_increment"] = 9
    elif case.startswith("digest-type:"):
        record[case.partition(":")[2]] = 7
    elif case == "digest-uppercase":
        record["center_sha256"] = "A" * 64
    elif case == "cluster-membership-container":
        record["cluster_membership_sha256"] = "not-a-list"
    elif case == "cluster-membership-item-type":
        record["cluster_membership_sha256"][0] = 7
    elif case == "cluster-membership-count":
        record["cluster_membership_sha256"].pop()
    elif case == "inertia-hex-container":
        record["cluster_inertia_hex"] = "not-a-list"
    elif case == "inertia-hex-item-type":
        record["cluster_inertia_hex"][0] = 7
    elif case == "inertia-hex-count":
        record["cluster_inertia_hex"].pop()
    elif case == "inertia-hex-noncanonical":
        record["cluster_inertia_hex"][0] = "0X0.0P+0"
    elif case == "sklearn-hex-type":
        record["sklearn_inertia_hex"] = 7
    elif case == "sklearn-hex-noncanonical":
        record["sklearn_inertia_hex"] = "0X0.0P+0"
    elif case == "recomputed-hex-type":
        record["recomputed_total_inertia_hex"] = 7
    elif case == "recomputed-hex-noncanonical":
        record["recomputed_total_inertia_hex"] = "0X0.0P+0"
    elif case == "n-iter-bool":
        record["n_iter"] = True
    elif case == "n-iter-float":
        record["n_iter"] = 1.0
    elif case == "n-iter-zero":
        record["n_iter"] = 0
    else:
        assert case == "n-iter-upper-bound"
        record["n_iter"] = 300


_RECEIPT_CONTRACT_CASES = (
    "schema-type",
    "sequence-bool",
    "sequence-float",
    "sequence-value",
    "kind-type",
    "kind-value",
    "inventory-index-bool",
    "inventory-index-float",
    "inventory-index-value",
    "excluded-container",
    "excluded-item-type",
    "excluded-count",
    "excluded-duplicate",
    "cluster-count-bool",
    "cluster-count-float",
    "fit-increment-bool",
    "fit-increment-float",
    "fit-increment-value",
    "start-increment-bool",
    "start-increment-float",
    "start-increment-value",
    *tuple(f"digest-type:{field}" for field in _RECEIPT_SCALAR_DIGEST_FIELDS),
    "digest-uppercase",
    "cluster-membership-container",
    "cluster-membership-item-type",
    "cluster-membership-count",
    "inertia-hex-container",
    "inertia-hex-item-type",
    "inertia-hex-count",
    "inertia-hex-noncanonical",
    "sklearn-hex-type",
    "sklearn-hex-noncanonical",
    "recomputed-hex-type",
    "recomputed-hex-noncanonical",
    "n-iter-bool",
    "n-iter-float",
    "n-iter-zero",
    "n-iter-upper-bound",
)


@pytest.mark.parametrize("case", _RECEIPT_CONTRACT_CASES)
def test_cache_rejects_nonexact_receipt_types_inventory_and_counts(
    populated_cache: Path,
    case: str,
) -> None:
    _rewrite_fit_receipt(
        populated_cache,
        lambda record: _mutate_receipt_contract(record, case),
        rehash=True,
    )

    _assert_open_rejected(populated_cache)


def _mutate_assignment_contract(record: dict[str, Any], case: str) -> None:
    assignments = record["assignments"]
    assert isinstance(assignments, list)
    first = assignments[0]
    assert isinstance(first, dict)
    if case == "assignments-container":
        record["assignments"] = "not-a-list"
    elif case == "assignment-item-type":
        assignments[0] = 7
    elif case == "assignment-count-missing":
        assignments.pop()
    elif case == "assignment-count-extra":
        assignments.append(copy.deepcopy(first))
    elif case == "assignment-order":
        assignments.reverse()
    elif case == "assignment-query-type":
        first["query_id"] = 7
    elif case == "assignment-query-not-excluded":
        first["query_id"] = "q3"
    elif case == "assignment-query-duplicate":
        assignments[1]["query_id"] = first["query_id"]
    elif case == "assignment-cluster-id-bool":
        first["canonical_cluster_id"] = True
    elif case == "assignment-cluster-id-float":
        first["canonical_cluster_id"] = 0.0
    elif case == "assignment-cluster-id-range":
        first["canonical_cluster_id"] = 2
    elif case == "assignment-distance-type":
        first["squared_distance_hex"] = 7
    elif case == "assignment-distance-malformed":
        first["squared_distance_hex"] = "not-a-hex-float"
    elif case == "assignment-distance-noncanonical":
        first["squared_distance_hex"] = "0X1.0000000000000P+0"
    elif case == "assignment-distance-nonfinite":
        first["squared_distance_hex"] = "nan"
    elif case == "assignment-distance-negative":
        first["squared_distance_hex"] = (-1.0).hex()
    elif case == "assignment-support-bool":
        first["cluster_support"] = True
    elif case == "assignment-support-float":
        first["cluster_support"] = 1.0
    else:
        assert case == "assignment-support-zero"
        first["cluster_support"] = 0


_ASSIGNMENT_CONTRACT_CASES = (
    "assignments-container",
    "assignment-item-type",
    "assignment-count-missing",
    "assignment-count-extra",
    "assignment-order",
    "assignment-query-type",
    "assignment-query-not-excluded",
    "assignment-query-duplicate",
    "assignment-cluster-id-bool",
    "assignment-cluster-id-float",
    "assignment-cluster-id-range",
    "assignment-distance-type",
    "assignment-distance-malformed",
    "assignment-distance-noncanonical",
    "assignment-distance-nonfinite",
    "assignment-distance-negative",
    "assignment-support-bool",
    "assignment-support-float",
    "assignment-support-zero",
)


@pytest.mark.parametrize("case", _ASSIGNMENT_CONTRACT_CASES)
def test_cache_rejects_nonexact_assignment_types_and_semantics(
    populated_cache: Path,
    case: str,
) -> None:
    _rewrite_fit_receipt(
        populated_cache,
        lambda record: _mutate_assignment_contract(record, case),
        rehash=True,
    )

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("missing_key", tuple(sorted(_ASSIGNMENT_RECEIPT_KEYS)))
def test_cache_rejects_each_missing_assignment_key(
    populated_cache: Path,
    missing_key: str,
) -> None:
    def mutate(record: dict[str, Any]) -> None:
        assignment = record["assignments"][0]
        assert isinstance(assignment, dict)
        assignment.pop(missing_key)

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)

    _assert_open_rejected(populated_cache)


def test_cache_rejects_unknown_assignment_key(populated_cache: Path) -> None:
    def mutate(record: dict[str, Any]) -> None:
        assignment = record["assignments"][0]
        assert isinstance(assignment, dict)
        assignment["member_indices"] = [2]

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("corruption", ("digest", "type"))
def test_cache_rejects_invalid_receipt_record_hash(
    populated_cache: Path,
    corruption: str,
) -> None:
    def mutate(record: dict[str, Any]) -> None:
        if corruption == "digest":
            record["record_sha256"] = _different_sha256(record["record_sha256"])
        else:
            assert corruption == "type"
            record["record_sha256"] = 7

    _rewrite_fit_receipt(
        populated_cache,
        mutate,
        rehash=False,
    )
    before = _finalized_file_snapshot(populated_cache)

    _assert_open_rejected(populated_cache)

    assert _finalized_file_snapshot(populated_cache) == before


def test_cache_rejects_inertia_total_inconsistent_with_exact_fsum(
    populated_cache: Path,
) -> None:
    inertia_values = np.ascontiguousarray(np.asarray([1.0, 2.0], dtype="<f8"))
    (populated_cache / "cluster_inertia.f8").write_bytes(
        inertia_values.tobytes(order="C")
    )

    def mutate(record: dict[str, Any]) -> None:
        record["cluster_inertia_hex"] = [value.hex() for value in inertia_values]
        record["sklearn_inertia_hex"] = (4.0).hex()
        record["recomputed_total_inertia_hex"] = (4.0).hex()

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)
    assert math.fsum(inertia_values).hex() == (3.0).hex()
    assert math.fsum(inertia_values).hex() != (4.0).hex()

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("invalid_value", (-1.0, float("nan")), ids=("negative", "nan"))
def test_cache_rejects_invalid_inertia_even_when_receipt_matches_shard(
    populated_cache: Path,
    invalid_value: float,
) -> None:
    inertia_values = np.ascontiguousarray(np.asarray([invalid_value, 0.0], dtype="<f8"))
    (populated_cache / "cluster_inertia.f8").write_bytes(
        inertia_values.tobytes(order="C")
    )
    total = math.fsum(inertia_values)

    def mutate(record: dict[str, Any]) -> None:
        record["cluster_inertia_hex"] = [value.hex() for value in inertia_values]
        record["sklearn_inertia_hex"] = total.hex()
        record["recomputed_total_inertia_hex"] = total.hex()

    _rewrite_fit_receipt(populated_cache, mutate, rehash=True)

    _assert_open_rejected(populated_cache)


@pytest.mark.parametrize("missing_key", tuple(sorted(_TIMING_KEYS)))
def test_cache_rejects_each_missing_timing_key(
    populated_cache: Path,
    missing_key: str,
) -> None:
    _rewrite_timing(
        populated_cache,
        lambda record: record.pop(missing_key),
    )

    _assert_open_rejected(populated_cache)


def _mutate_timing_contract(record: dict[str, Any], case: str) -> None:
    if case == "unknown-key":
        record["unexpected"] = None
    elif case == "schema-type":
        record["schema"] = 7
    elif case == "schema-value":
        record["schema"] = "trec-mlm-dense-aggregation-fit-timing-v2"
    elif case == "sequence-bool":
        record["sequence"] = True
    elif case == "sequence-float":
        record["sequence"] = 0.0
    elif case == "sequence-value":
        record["sequence"] = 1
    elif case == "kind-type":
        record["exclusion_kind"] = 7
    elif case == "kind-value":
        record["exclusion_kind"] = "single"
    elif case == "inventory-index-bool":
        record["inventory_index"] = True
    elif case == "inventory-index-float":
        record["inventory_index"] = 0.0
    elif case == "inventory-index-value":
        record["inventory_index"] = 1
    elif case == "cluster-count-bool":
        record["cluster_count"] = True
    elif case == "cluster-count-float":
        record["cluster_count"] = 2.0
    elif case == "cluster-count-value":
        record["cluster_count"] = 1
    elif case == "elapsed-bool":
        record["elapsed_ns"] = True
    elif case == "elapsed-float":
        record["elapsed_ns"] = 1.0
    elif case == "elapsed-negative":
        record["elapsed_ns"] = -1
    elif case == "n-iter-bool":
        record["n_iter"] = True
    elif case == "n-iter-float":
        record["n_iter"] = 1.0
    elif case == "n-iter-zero":
        record["n_iter"] = 0
    elif case == "n-iter-upper-bound":
        record["n_iter"] = 300
    else:
        assert case == "n-iter-fit-mismatch"
        record["n_iter"] = 2


_TIMING_CONTRACT_CASES = (
    "unknown-key",
    "schema-type",
    "schema-value",
    "sequence-bool",
    "sequence-float",
    "sequence-value",
    "kind-type",
    "kind-value",
    "inventory-index-bool",
    "inventory-index-float",
    "inventory-index-value",
    "cluster-count-bool",
    "cluster-count-float",
    "cluster-count-value",
    "elapsed-bool",
    "elapsed-float",
    "elapsed-negative",
    "n-iter-bool",
    "n-iter-float",
    "n-iter-zero",
    "n-iter-upper-bound",
    "n-iter-fit-mismatch",
)


@pytest.mark.parametrize("case", _TIMING_CONTRACT_CASES)
def test_cache_rejects_nonexact_timing_schema_types_and_triplet(
    populated_cache: Path,
    case: str,
) -> None:
    _rewrite_timing(
        populated_cache,
        lambda record: _mutate_timing_contract(record, case),
    )

    _assert_open_rejected(populated_cache)


def _mutate_header_contract(header: dict[str, Any], case: str) -> None:
    if case == "schema":
        header["schema"] = "trec-mlm-dense-aggregation-fit-cache-header-v2"
    elif case == "missing-key":
        header.pop("runtime_inventory_sha256")
    elif case == "unknown-key":
        header["unexpected"] = None
    elif case == "digest":
        header["embedding_sha256"] = "A" * 64
    elif case == "query-order":
        header["query_ids"].reverse()
    elif case == "shard-schema":
        header["shard_schemas"]["labels"]["dtype"] = "<i2"
    elif case == "inventory":
        header["expected_inventory"][0]["inventory_index"] = 1
    elif case == "count":
        header["expected_counts"]["fit_count"] = 2
    else:
        assert case == "count-type"
        header["expected_counts"]["timing_record_count"] = True


@pytest.mark.parametrize(
    "case",
    (
        "schema",
        "missing-key",
        "unknown-key",
        "digest",
        "query-order",
        "shard-schema",
        "inventory",
        "count",
        "count-type",
    ),
)
def test_open_readonly_revalidates_canonical_header_contract(
    populated_cache: Path,
    case: str,
) -> None:
    _rewrite_header(
        populated_cache,
        lambda header: _mutate_header_contract(header, case),
    )

    _assert_open_rejected(populated_cache)


def _finalized_payload_bytes(root: Path) -> tuple[tuple[str, bytes], ...]:
    assert stat.S_ISDIR(os.lstat(root).st_mode)
    entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    assert {path.name for path in entries} == _STREAM_PATHS | {"header.json"}
    assert all(stat.S_ISREG(os.lstat(path).st_mode) for path in entries)
    return tuple((path.name, path.read_bytes()) for path in entries)


def _assert_exact_pair_record_loads(cache: StreamingFitCache) -> None:
    fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    loaded = cache.load_fit(dataclasses.replace(fit.key))
    assert isinstance(loaded, StoredFitRecord)
    _assert_stored_fit_record_exact(
        loaded,
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    assert cache.release(dataclasses.replace(fit.key)) is None


def test_finalize_into_moves_exact_payload_fsyncs_parent_and_updates_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fit-cache"
    destination_parent = tmp_path / "artifact"
    destination_parent.mkdir()
    destination = destination_parent / "fits"
    expected_header = _valid_header()
    cache, fit = _populate_one_fit_cache(
        source,
        header=expected_header,
        release=True,
    )
    assert not cache._resident
    assert cache.finalize() == expected_header
    assert not cache._resident
    expected_payload = _finalized_payload_bytes(source)
    assert not os.path.lexists(destination)

    real_fsync = os.fsync
    resolved_destination_parent = destination_parent.resolve(strict=True)
    fsync_paths: list[Path] = []

    def fsync_spy(fd: int) -> None:
        proc_fd = Path("/proc/self/fd") / str(fd)
        resolved_path = Path(os.readlink(proc_fd)).resolve(strict=True)
        fsync_paths.append(resolved_path)
        if resolved_path == resolved_destination_parent:
            assert not os.path.lexists(source)
            assert stat.S_ISDIR(os.lstat(destination).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync_spy)

    returned_header = cache.finalize_into(destination)

    assert returned_header == expected_header
    assert cache._state == "finalized"
    assert fsync_paths == [resolved_destination_parent]
    assert not os.path.lexists(source)
    assert _finalized_payload_bytes(destination) == expected_payload

    loaded = cache.load_fit(dataclasses.replace(fit.key))
    assert isinstance(loaded, StoredFitRecord)
    _, assignments, excluded_query_ids = _synthetic_pair_fit(0)
    _assert_stored_fit_record_exact(
        loaded,
        fit=fit,
        assignments=assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    assert cache.release(dataclasses.replace(fit.key)) is None
    assert _finalized_payload_bytes(destination) == expected_payload


def test_finalize_into_rejects_writing_cache_without_touching_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fit-cache"
    destination_parent = tmp_path / "artifact"
    destination_parent.mkdir()
    destination = destination_parent / "fits"
    cache, _fit = _populate_one_fit_cache(source, release=True)
    handles = _cache_stream_handles(cache)
    before = _cache_mutation_snapshot(cache)

    def forbidden_rename(_source: str | Path, _destination: str | Path) -> None:
        raise AssertionError("an unfinalized cache must not reach rename_noreplace")

    monkeypatch.setattr(cache_module, "rename_noreplace", forbidden_rename)

    with pytest.raises((RuntimeError, ValueError), match="finaliz"):
        cache.finalize_into(destination)

    assert cache._state == "writing"
    assert _cache_mutation_snapshot(cache) == before
    assert _cache_stream_handles(cache) == handles
    assert all(not handle.closed for _, handle in handles)
    assert not (source / "header.json").exists()
    assert not os.path.lexists(destination)
    assert tuple(destination_parent.iterdir()) == ()


def test_finalize_into_rejects_independently_opened_readonly_cache(
    populated_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination_parent = tmp_path / "artifact"
    destination_parent.mkdir()
    destination = destination_parent / "fits"
    expected_payload = _finalized_payload_bytes(populated_cache)
    readonly = StreamingFitCache.open_readonly(populated_cache)

    def forbidden_rename(_source: str | Path, _destination: str | Path) -> None:
        raise AssertionError("an independently opened cache must not be moved")

    monkeypatch.setattr(cache_module, "rename_noreplace", forbidden_rename)

    with pytest.raises((RuntimeError, ValueError)):
        readonly.finalize_into(destination)

    assert readonly._state == "finalized"
    assert not os.path.lexists(destination)
    assert _finalized_payload_bytes(populated_cache) == expected_payload
    _assert_exact_pair_record_loads(readonly)
    assert _finalized_payload_bytes(populated_cache) == expected_payload


def _assert_existing_move_destination_preserved(
    destination: Path,
    *,
    destination_kind: str,
    dangling_target: Path,
) -> None:
    assert os.path.lexists(destination)
    status = os.lstat(destination)
    if destination_kind == "file":
        assert stat.S_ISREG(status.st_mode)
        assert destination.read_bytes() == b"existing-file\n"
    elif destination_kind == "directory":
        assert stat.S_ISDIR(status.st_mode)
        assert tuple(path.name for path in destination.iterdir()) == ("sentinel.txt",)
        assert (destination / "sentinel.txt").read_bytes() == b"existing-directory\n"
    else:
        assert destination_kind == "dangling-symlink"
        assert stat.S_ISLNK(status.st_mode)
        assert Path(os.readlink(destination)) == dangling_target
        assert not destination.exists()


@pytest.mark.parametrize("destination_kind", ("file", "directory", "dangling-symlink"))
def test_finalize_into_never_replaces_existing_destination_and_can_retry(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    source = tmp_path / "fit-cache"
    destination_parent = tmp_path / "artifact"
    destination_parent.mkdir()
    destination = destination_parent / "fits"
    fallback = destination_parent / "fits-fallback"
    dangling_target = destination_parent / "missing-target"
    cache, _fit = _populate_one_fit_cache(source, release=True)
    expected_header = cache.finalize()
    expected_payload = _finalized_payload_bytes(source)

    if destination_kind == "file":
        destination.write_bytes(b"existing-file\n")
    elif destination_kind == "directory":
        destination.mkdir()
        (destination / "sentinel.txt").write_bytes(b"existing-directory\n")
    else:
        assert destination_kind == "dangling-symlink"
        destination.symlink_to(dangling_target)
    _assert_existing_move_destination_preserved(
        destination,
        destination_kind=destination_kind,
        dangling_target=dangling_target,
    )

    with pytest.raises(OSError):
        cache.finalize_into(destination)

    assert cache._state == "finalized"
    assert _finalized_payload_bytes(source) == expected_payload
    _assert_existing_move_destination_preserved(
        destination,
        destination_kind=destination_kind,
        dangling_target=dangling_target,
    )
    assert not os.path.lexists(fallback)

    assert cache.finalize_into(fallback) == expected_header

    assert not os.path.lexists(source)
    assert _finalized_payload_bytes(fallback) == expected_payload
    _assert_existing_move_destination_preserved(
        destination,
        destination_kind=destination_kind,
        dangling_target=dangling_target,
    )


def test_finalize_into_racing_destination_is_no_clobber_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fit-cache"
    destination_parent = tmp_path / "artifact"
    destination_parent.mkdir()
    destination = destination_parent / "fits"
    fallback = destination_parent / "fits-fallback"
    cache, _fit = _populate_one_fit_cache(source, release=True)
    expected_header = cache.finalize()
    expected_payload = _finalized_payload_bytes(source)
    real_rename_noreplace = cache_module.rename_noreplace
    rename_calls: list[tuple[Path, Path]] = []

    def race_then_real_rename(
        candidate_source: str | Path,
        candidate_destination: str | Path,
    ) -> None:
        source_path = Path(candidate_source)
        destination_path = Path(candidate_destination)
        rename_calls.append((source_path, destination_path))
        assert not os.path.lexists(destination_path)
        destination_path.mkdir()
        (destination_path / "sentinel.txt").write_bytes(b"racing-destination\n")
        real_rename_noreplace(source_path, destination_path)

    monkeypatch.setattr(cache_module, "rename_noreplace", race_then_real_rename)

    with pytest.raises(OSError) as error:
        cache.finalize_into(destination)

    assert error.value.errno == errno.EEXIST
    assert len(rename_calls) == 1
    assert rename_calls[0][1] == destination
    assert cache._state == "finalized"
    assert _finalized_payload_bytes(source) == expected_payload
    assert tuple(path.name for path in destination.iterdir()) == ("sentinel.txt",)
    assert (destination / "sentinel.txt").read_bytes() == b"racing-destination\n"

    monkeypatch.setattr(cache_module, "rename_noreplace", real_rename_noreplace)
    assert cache.finalize_into(fallback) == expected_header

    assert not os.path.lexists(source)
    assert _finalized_payload_bytes(fallback) == expected_payload
    assert (destination / "sentinel.txt").read_bytes() == b"racing-destination\n"


def test_finalize_into_rejects_cross_device_parent_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fit-cache"
    destination_parent = tmp_path / "artifact"
    destination_parent.mkdir()
    destination = destination_parent / "fits"
    fallback = destination_parent / "fits-fallback"
    cache, _fit = _populate_one_fit_cache(source, release=True)
    expected_header = cache.finalize()
    expected_payload = _finalized_payload_bytes(source)
    resolved_destination_parent = destination_parent.resolve(strict=True)
    real_path_stat = Path.stat
    source_device = real_path_stat(source.resolve(strict=True)).st_dev
    destination_stat_calls: list[Path] = []

    def different_device_stat(
        candidate: Path,
        *,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        result = real_path_stat(candidate, follow_symlinks=follow_symlinks)
        if candidate == resolved_destination_parent:
            destination_stat_calls.append(candidate)
            values = list(result)
            values[2] = source_device + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", different_device_stat)

    with pytest.raises(OSError, match="filesystem|device"):
        cache.finalize_into(destination)

    assert destination_stat_calls
    assert cache._state == "finalized"
    assert os.path.lexists(source)
    assert not os.path.lexists(destination)

    monkeypatch.setattr(Path, "stat", real_path_stat)
    assert _finalized_payload_bytes(source) == expected_payload
    assert cache.finalize_into(fallback) == expected_header

    assert not os.path.lexists(source)
    assert not os.path.lexists(destination)
    assert _finalized_payload_bytes(fallback) == expected_payload


def test_finalize_into_allows_only_one_successful_move(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fit-cache"
    destination_parent = tmp_path / "artifact"
    destination_parent.mkdir()
    first_destination = destination_parent / "fits"
    second_destination = destination_parent / "fits-second"
    cache, _fit = _populate_one_fit_cache(source, release=True)
    expected_header = cache.finalize()
    expected_payload = _finalized_payload_bytes(source)

    assert cache.finalize_into(first_destination) == expected_header
    assert not os.path.lexists(source)
    assert _finalized_payload_bytes(first_destination) == expected_payload
    assert not os.path.lexists(second_destination)

    with pytest.raises((RuntimeError, ValueError, OSError)):
        cache.finalize_into(second_destination)

    assert cache._state == "finalized"
    assert _finalized_payload_bytes(first_destination) == expected_payload
    assert not os.path.lexists(second_destination)
    readonly = StreamingFitCache.open_readonly(first_destination)
    _assert_exact_pair_record_loads(readonly)
    assert _finalized_payload_bytes(first_destination) == expected_payload


def test_cache_round_trip_preserves_exact_fit_and_scoring_semantics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fit-cache"
    header = _valid_header()
    naive_fit, naive_assignments, excluded_query_ids = _synthetic_pair_fit(0)
    writer = StreamingFitCache.create(root, header=header)
    writer.append_fit(
        naive_fit,
        naive_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )
    writer.append_timing(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
        elapsed_ns=1_234_567,
        n_iter=naive_fit.n_iter,
    )
    assert writer.release(dataclasses.replace(naive_fit.key)) is None
    assert writer.finalize() == header
    del writer

    finalized_payload = _finalized_payload_bytes(root)
    readonly = StreamingFitCache.open_readonly(root)
    loaded = readonly.load_by_inventory(
        exclusion_kind="pair",
        inventory_index=0,
        cluster_count=2,
    )

    assert _fit_record_semantics(
        fit=loaded.fit,
        assignments=loaded.assignments,
        exclusion_kind=loaded.exclusion_kind,
        inventory_index=loaded.inventory_index,
        excluded_query_ids=loaded.excluded_query_ids,
    ) == _fit_record_semantics(
        fit=naive_fit,
        assignments=naive_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=excluded_query_ids,
    )

    grades = np.ascontiguousarray(
        np.asarray(
            [
                [0.125, 0.25, 0.375, 0.5],
                [1.875, 1.75, 1.625, 1.5],
                [2.0, 0.0, 1.0, 0.5],
                [0.0, 2.0, 0.5, 1.0],
            ],
            dtype="<f8",
        )
    )
    prior = np.ascontiguousarray(np.asarray([1.0, 1.0, 1.25, 0.75], dtype="<f8"))
    local_by_query = {
        "q1": np.ascontiguousarray(np.asarray([0.25, 1.75, 0.5, 1.5], dtype="<f8")),
        "q2": np.ascontiguousarray(np.asarray([1.75, 0.25, 1.5, 0.5], dtype="<f8")),
    }
    model_ids = ("model-0", "model-1", "model-2", "model-3")
    candidate = DenseAggregationCandidate(2, 2, 0.5, 0.1)
    expected_by_query = {
        "q1": {
            "domain_bytes": bytes.fromhex(
                "00000000000000400000000000000000000000000000f03f000000000000e03f"
            ),
            "domain_hex": (
                "0x1.0000000000000p+1",
                "0x0.0p+0",
                "0x1.0000000000000p+0",
                "0x1.0000000000000p-1",
            ),
            "final_bytes": bytes.fromhex(
                "cdccccccccccf13f666666666666ec3f9a9999999999e93f343333333333ef3f"
            ),
            "final_hex": (
                "0x1.1cccccccccccdp+0",
                "0x1.c666666666666p-1",
                "0x1.999999999999ap-1",
                "0x1.f333333333334p-1",
            ),
            "model_order": (0, 3, 1, 2),
            "rank_bytes": b"\x01\x00\x03\x00\x04\x00\x02\x00",
            "ranks": (1, 3, 4, 2),
        },
        "q2": {
            "domain_bytes": bytes.fromhex(
                "00000000000000000000000000000040000000000000e03f000000000000f03f"
            ),
            "domain_hex": (
                "0x0.0p+0",
                "0x1.0000000000000p+1",
                "0x1.0000000000000p-1",
                "0x1.0000000000000p+0",
            ),
            "final_bytes": bytes.fromhex(
                "666666666666ec3fcdccccccccccf13f676666666666f03f000000000000e83f"
            ),
            "final_hex": (
                "0x1.c666666666666p-1",
                "0x1.1cccccccccccdp+0",
                "0x1.0666666666667p+0",
                "0x1.8000000000000p-1",
            ),
            "model_order": (1, 2, 0, 3),
            "rank_bytes": b"\x03\x00\x01\x00\x02\x00\x04\x00",
            "ranks": (3, 1, 2, 4),
        },
    }

    for naive_assignment, loaded_assignment in zip(
        naive_assignments, loaded.assignments, strict=True
    ):
        assert (
            loaded_assignment.query_id,
            loaded_assignment.canonical_cluster_id,
            loaded_assignment.squared_distance.hex(),
            loaded_assignment.member_indices,
            loaded_assignment.cluster_support,
        ) == (
            naive_assignment.query_id,
            naive_assignment.canonical_cluster_id,
            naive_assignment.squared_distance.hex(),
            naive_assignment.member_indices,
            naive_assignment.cluster_support,
        )
        query_id = naive_assignment.query_id
        expected = expected_by_query[query_id]
        naive_domain = domain_scores(
            member_indices=naive_assignment.member_indices,
            grades=grades,
        )
        loaded_domain = domain_scores(
            member_indices=loaded_assignment.member_indices,
            grades=grades,
        )
        assert naive_domain.dtype == loaded_domain.dtype == np.dtype("<f8")
        assert (
            naive_domain.tobytes(order="C")
            == loaded_domain.tobytes(order="C")
            == expected["domain_bytes"]
        )
        assert (
            tuple(float(value).hex() for value in naive_domain)
            == tuple(float(value).hex() for value in loaded_domain)
            == expected["domain_hex"]
        )

        naive_prediction = combine_and_rank(
            local=local_by_query[query_id],
            domain=naive_domain,
            prior=prior,
            model_ids=model_ids,
            candidate=candidate,
        )
        loaded_prediction = combine_and_rank(
            local=local_by_query[query_id],
            domain=loaded_domain,
            prior=prior,
            model_ids=model_ids,
            candidate=candidate,
        )
        assert naive_prediction.final_scores.dtype == np.dtype("<f8")
        assert loaded_prediction.final_scores.dtype == np.dtype("<f8")
        assert (
            naive_prediction.final_scores.tobytes(order="C")
            == loaded_prediction.final_scores.tobytes(order="C")
            == expected["final_bytes"]
        )
        assert (
            tuple(float(value).hex() for value in naive_prediction.final_scores)
            == tuple(float(value).hex() for value in loaded_prediction.final_scores)
            == expected["final_hex"]
        )
        assert (
            naive_prediction.model_order
            == loaded_prediction.model_order
            == expected["model_order"]
        )
        assert naive_prediction.ranks.dtype == np.dtype(np.uint16)
        assert loaded_prediction.ranks.dtype == np.dtype(np.uint16)
        assert (
            naive_prediction.ranks.tobytes(order="C")
            == loaded_prediction.ranks.tobytes(order="C")
            == expected["rank_bytes"]
        )
        assert (
            tuple(int(value) for value in naive_prediction.ranks)
            == tuple(int(value) for value in loaded_prediction.ranks)
            == expected["ranks"]
        )

    assert readonly.release(dataclasses.replace(loaded.fit.key)) is None
    assert _finalized_payload_bytes(root) == finalized_payload
