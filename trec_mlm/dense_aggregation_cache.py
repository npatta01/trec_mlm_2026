from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, BinaryIO, Literal

import numpy as np

from trec_mlm.dense_aggregation_artifacts import rename_noreplace
from trec_mlm.dense_aggregation_clustering import (
    CanonicalClusterFit,
    FitKey,
    TargetAssignment,
)
from trec_mlm.dense_aggregation_hashing import (
    canonical_json,
    canonical_membership_sha256,
    framed_sha256,
    logical_array_sha256,
    support_query_ids_sha256,
)


CACHE_HEADER_SCHEMA = "trec-mlm-dense-aggregation-fit-cache-header-v1"
CACHE_HEADER_KEYS = {
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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
_STREAM_NAMES = (
    "centers.f4",
    "labels.u2",
    "cluster_inertia.f8",
    "fit_receipts.jsonl",
    "timings.jsonl",
)
_FINAL_NAMES = (*_STREAM_NAMES, "header.json")
_COUNT_KEYS = {
    "pair_fit_count",
    "single_fit_count",
    "fit_count",
    "start_count",
    "timing_record_count",
}
_INVENTORY_KEYS = {"exclusion_kind", "inventory_index", "cluster_count"}
_FIT_RECEIPT_SCHEMA = "trec-mlm-dense-aggregation-fit-receipt-v1"
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
_DESCRIPTOR_KEYS = {"offset", "bytes", "shape", "dtype"}
_ASSIGNMENT_KEYS = {
    "query_id",
    "canonical_cluster_id",
    "squared_distance_hex",
    "cluster_support",
}
_TIMING_SCHEMA = "trec-mlm-dense-aggregation-fit-timing-v1"
_TIMING_KEYS = {
    "schema",
    "sequence",
    "exclusion_kind",
    "inventory_index",
    "cluster_count",
    "elapsed_ns",
    "n_iter",
}


@dataclass(frozen=True, slots=True)
class StoredFitRecord:
    fit: CanonicalClusterFit
    assignments: tuple[TargetAssignment, ...]
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_query_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    key: FitKey
    sequence: int
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_query_ids: tuple[str, ...]
    n_iter: int
    receipt_offset: int
    receipt_bytes: int
    receipt_sha256: str
    timing_offset: int | None
    timing_bytes: int | None
    timing_sha256: str | None


@dataclass(frozen=True, slots=True)
class _ScanResult:
    index: dict[str, _IndexEntry]
    inventory_index: dict[tuple[str, int, int], str]
    fit_count: int
    timing_count: int
    center_width: int | None


def _fit_key_sha256(key: FitKey) -> str:
    return framed_sha256(
        "fit-cache-key",
        canonical_json(
            {
                "schema": "trec-mlm-dense-aggregation-fit-key-v1",
                "support_query_ids": list(key.support_query_ids),
                "cluster_count": key.cluster_count,
                "embedding_sha256": key.embedding_sha256,
                "clustering_contract_sha256": key.clustering_contract_sha256,
            }
        ),
    )


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pair_inventory_size(query_ids: Sequence[str]) -> int:
    return math.comb(len(query_ids), 2)


def _pair_at(query_ids: tuple[str, ...], inventory_index: int) -> tuple[str, str]:
    remaining = inventory_index
    for first_index in range(len(query_ids) - 1):
        row_size = len(query_ids) - first_index - 1
        if remaining < row_size:
            return (
                query_ids[first_index],
                query_ids[first_index + 1 + remaining],
            )
        remaining -= row_size
    raise ValueError("pair inventory index is out of range")


def _expected_excluded_query_ids(
    query_ids: tuple[str, ...],
    exclusion_kind: str,
    inventory_index: int,
) -> tuple[str, ...]:
    if exclusion_kind == "pair":
        return _pair_at(query_ids, inventory_index)
    return (query_ids[inventory_index],)


def _validate_header(header: object) -> dict[str, object]:
    if type(header) is not dict or set(header) != CACHE_HEADER_KEYS:
        raise ValueError("cache header has an invalid top-level contract")
    if header["schema"] != CACHE_HEADER_SCHEMA:
        raise ValueError("cache header has an unsupported schema")
    for field in _HASH_FIELDS:
        if not _is_sha256(header[field]):
            raise ValueError(f"cache header {field} must be a lowercase SHA-256")

    query_ids_value = header["query_ids"]
    if type(query_ids_value) is not list:
        raise ValueError("cache header query_ids must be a list")
    query_ids = tuple(query_ids_value)
    if (
        not query_ids
        or any(type(query_id) is not str or not query_id for query_id in query_ids)
        or tuple(sorted(query_ids)) != query_ids
        or len(set(query_ids)) != len(query_ids)
    ):
        raise ValueError("cache header query_ids must be sorted, nonempty, and unique")

    if header["shard_schemas"] != _SHARD_SCHEMAS:
        raise ValueError("cache header shard_schemas do not match the literal schema")
    schemas = header["shard_schemas"]
    if type(schemas) is not dict or any(
        type(schema) is not dict for schema in schemas.values()
    ):
        raise ValueError("cache header shard schemas must be exact objects")

    inventory_value = header["expected_inventory"]
    if type(inventory_value) is not list:
        raise ValueError("cache header expected_inventory must be a list")
    ordering: list[tuple[int, int, int]] = []
    triplets: set[tuple[str, int, int]] = set()
    pair_count = 0
    single_count = 0
    for row in inventory_value:
        if type(row) is not dict or set(row) != _INVENTORY_KEYS:
            raise ValueError("cache header inventory row has an invalid contract")
        exclusion_kind = row["exclusion_kind"]
        inventory_index = row["inventory_index"]
        cluster_count = row["cluster_count"]
        if type(exclusion_kind) is not str or exclusion_kind not in {"pair", "single"}:
            raise ValueError("cache header inventory kind is invalid")
        if type(inventory_index) is not int or inventory_index < 0:
            raise ValueError("cache header inventory index is invalid")
        upper_bound = (
            _pair_inventory_size(query_ids)
            if exclusion_kind == "pair"
            else len(query_ids)
        )
        if inventory_index >= upper_bound:
            raise ValueError("cache header inventory index is out of range")
        if type(cluster_count) is not int or not 1 <= cluster_count <= 65_536:
            raise ValueError("cache header cluster_count is invalid")
        triplet = (exclusion_kind, inventory_index, cluster_count)
        if triplet in triplets:
            raise ValueError("cache header inventory contains a duplicate triplet")
        triplets.add(triplet)
        ordering.append(
            (0 if exclusion_kind == "pair" else 1, inventory_index, cluster_count)
        )
        pair_count += exclusion_kind == "pair"
        single_count += exclusion_kind == "single"
    if ordering != sorted(ordering):
        raise ValueError("cache header inventory is not in canonical order")

    counts = header["expected_counts"]
    if type(counts) is not dict or set(counts) != _COUNT_KEYS:
        raise ValueError("cache header expected_counts has an invalid contract")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("cache header expected counts must be nonnegative ints")
    fit_count = len(inventory_value)
    if (
        counts["pair_fit_count"] != pair_count
        or counts["single_fit_count"] != single_count
        or counts["fit_count"] != fit_count
        or counts["fit_count"] != counts["pair_fit_count"] + counts["single_fit_count"]
        or counts["start_count"] != 10 * fit_count
        or counts["timing_record_count"] != fit_count
    ):
        raise ValueError("cache header expected counts are inconsistent")

    try:
        owned = json.loads(canonical_json(header))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("cache header is not canonical JSON data") from error
    if type(owned) is not dict:
        raise ValueError("cache header is not an object")
    return owned


def _header_query_ids(header: Mapping[str, object]) -> tuple[str, ...]:
    value = header["query_ids"]
    if type(value) is not list:
        raise ValueError("validated cache header query_ids changed type")
    return tuple(value)


def _header_inventory(header: Mapping[str, object]) -> list[dict[str, object]]:
    value = header["expected_inventory"]
    if type(value) is not list:
        raise ValueError("validated cache header inventory changed type")
    return value


def _expected_inventory_row(
    header: Mapping[str, object], sequence: int
) -> dict[str, object]:
    inventory = _header_inventory(header)
    if type(sequence) is not int or not 0 <= sequence < len(inventory):
        raise ValueError("cache fit inventory is exhausted")
    row = inventory[sequence]
    if type(row) is not dict:
        raise ValueError("validated cache header inventory row changed type")
    return row


def _membership_rows(
    memberships: tuple[tuple[str, ...], ...],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple(enumerate(memberships))


def _validate_fit_and_assignments(
    *,
    header: Mapping[str, object],
    fit: CanonicalClusterFit,
    assignments: Sequence[TargetAssignment],
    exclusion_kind: object,
    inventory_index: object,
    excluded_query_ids: Sequence[str],
    expected_row: Mapping[str, object],
) -> tuple[tuple[TargetAssignment, ...], tuple[str, ...]]:
    if type(exclusion_kind) is not str or exclusion_kind not in {"pair", "single"}:
        raise ValueError("fit exclusion_kind is invalid")
    if type(inventory_index) is not int or inventory_index < 0:
        raise ValueError("fit inventory_index is invalid")
    if (
        exclusion_kind != expected_row["exclusion_kind"]
        or inventory_index != expected_row["inventory_index"]
    ):
        raise ValueError("fit does not match the next expected inventory row")
    query_ids = _header_query_ids(header)
    expected_excluded = _expected_excluded_query_ids(
        query_ids, exclusion_kind, inventory_index
    )
    try:
        submitted_excluded = tuple(excluded_query_ids)
    except TypeError as error:
        raise ValueError("excluded_query_ids must be a sequence") from error
    if submitted_excluded != expected_excluded:
        raise ValueError("excluded_query_ids do not match canonical inventory")
    support_query_ids = tuple(
        query_id for query_id in query_ids if query_id not in set(expected_excluded)
    )

    if type(fit) is not CanonicalClusterFit or type(fit.key) is not FitKey:
        raise ValueError("fit must be a canonical cluster fit")
    key = fit.key
    cluster_count = expected_row["cluster_count"]
    if (
        key.support_query_ids != support_query_ids
        or type(key.support_query_ids) is not tuple
        or key.cluster_count != cluster_count
        or type(key.cluster_count) is not int
        or key.embedding_sha256 != header["embedding_sha256"]
        or key.clustering_contract_sha256 != header["clustering_contract_sha256"]
    ):
        raise ValueError("fit key does not match the expected inventory")
    if not _is_sha256(key.embedding_sha256) or not _is_sha256(
        key.clustering_contract_sha256
    ):
        raise ValueError("fit key digests are invalid")
    if type(cluster_count) is not int:
        raise ValueError("validated cluster_count changed type")

    centers = fit.centers
    if (
        type(centers) is not np.ndarray
        or centers.ndim != 2
        or centers.shape[0] != cluster_count
        or centers.shape[1] <= 0
        or centers.dtype.str != "<f4"
        or not centers.flags.c_contiguous
        or not np.isfinite(centers).all()
    ):
        raise ValueError("fit centers violate the canonical array contract")
    labels = fit.labels
    if (
        type(labels) is not np.ndarray
        or labels.ndim != 1
        or labels.shape[0] != len(support_query_ids)
        or labels.dtype.str != "<u2"
        or not labels.flags.c_contiguous
    ):
        raise ValueError("fit labels violate the canonical array contract")
    label_values = tuple(int(label) for label in labels)
    if any(label < 0 or label >= cluster_count for label in label_values) or set(
        label_values
    ) != set(range(cluster_count)):
        raise ValueError("fit labels do not occupy every canonical cluster")
    expected_memberships = tuple(
        tuple(
            support_query_ids[support_index]
            for support_index, label in enumerate(label_values)
            if label == cluster_id
        )
        for cluster_id in range(cluster_count)
    )
    if type(fit.memberships) is not tuple or fit.memberships != expected_memberships:
        raise ValueError("fit memberships are not derived from labels")
    if any(type(membership) is not tuple for membership in fit.memberships):
        raise ValueError("fit memberships must be tuples")
    membership_rows = _membership_rows(expected_memberships)
    expected_fit_membership_sha256 = canonical_membership_sha256(membership_rows)
    expected_cluster_membership_sha256 = tuple(
        canonical_membership_sha256((membership_rows[cluster_id],))
        for cluster_id in range(cluster_count)
    )
    expected_center_sha256 = logical_array_sha256(
        centers,
        tag="canonical-centers",
        dtype="<f4",
        row_inventory=tuple(range(cluster_count)),
        column_inventory=(),
    )
    if fit.center_sha256 != expected_center_sha256:
        raise ValueError("fit center hash does not match centers")
    if fit.fit_membership_sha256 != expected_fit_membership_sha256:
        raise ValueError("fit membership hash does not match labels")
    if (
        type(fit.cluster_membership_sha256) is not tuple
        or fit.cluster_membership_sha256 != expected_cluster_membership_sha256
    ):
        raise ValueError("fit cluster membership hashes do not match labels")

    inertia = fit.cluster_inertia
    if (
        type(inertia) is not tuple
        or len(inertia) != cluster_count
        or any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in inertia
        )
    ):
        raise ValueError("fit cluster inertia is invalid")
    recomputed = math.fsum(inertia)
    if (
        type(fit.recomputed_total_inertia) is not float
        or not math.isfinite(fit.recomputed_total_inertia)
        or fit.recomputed_total_inertia.hex() != recomputed.hex()
    ):
        raise ValueError("fit recomputed inertia does not match exact fsum")
    if (
        type(fit.sklearn_inertia) is not float
        or not math.isfinite(fit.sklearn_inertia)
        or fit.sklearn_inertia < 0.0
        or not math.isclose(
            fit.sklearn_inertia,
            recomputed,
            rel_tol=1e-5,
            abs_tol=1e-8,
        )
    ):
        raise ValueError("fit sklearn inertia differs from canonical recomputation")
    if type(fit.n_iter) is not int or not 1 <= fit.n_iter <= 299:
        raise ValueError("fit n_iter must be an exact int in 1..299")

    try:
        assignment_tuple = tuple(assignments)
    except TypeError as error:
        raise ValueError("assignments must be a sequence") from error
    if len(assignment_tuple) != len(expected_excluded):
        raise ValueError("assignments must match the excluded-query inventory")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    for assignment, expected_query_id in zip(
        assignment_tuple, expected_excluded, strict=True
    ):
        if type(assignment) is not TargetAssignment:
            raise ValueError("assignment must be a TargetAssignment")
        cluster_id = assignment.canonical_cluster_id
        if type(cluster_id) is not int or not 0 <= cluster_id < cluster_count:
            raise ValueError("assignment canonical_cluster_id is invalid")
        expected_member_indices = tuple(
            query_index[query_id] for query_id in expected_memberships[cluster_id]
        )
        if (
            assignment.query_id != expected_query_id
            or type(assignment.query_id) is not str
            or type(assignment.member_indices) is not tuple
            or assignment.member_indices != expected_member_indices
            or any(type(index) is not int for index in assignment.member_indices)
        ):
            raise ValueError("assignment membership does not match canonical labels")
        if type(
            assignment.cluster_support
        ) is not int or assignment.cluster_support != len(expected_member_indices):
            raise ValueError("assignment cluster_support is invalid")
        if (
            type(assignment.squared_distance) is not float
            or not math.isfinite(assignment.squared_distance)
            or assignment.squared_distance < 0.0
        ):
            raise ValueError("assignment squared distance is invalid")
    return assignment_tuple, expected_excluded


def _validate_layout(root: Path, expected_names: Sequence[str]) -> Path:
    try:
        root_status = os.lstat(root)
    except OSError as error:
        raise ValueError(f"cache root is missing or invalid: {root}") from error
    if not stat.S_ISDIR(root_status.st_mode):
        raise ValueError("cache root must be a real directory")
    try:
        with os.scandir(root) as entries:
            names = {entry.name for entry in entries}
    except OSError as error:
        raise ValueError("cache root cannot be inspected") from error
    if names != set(expected_names):
        raise ValueError("cache has missing or extra entries")
    for name in expected_names:
        try:
            status = os.lstat(root / name)
        except OSError as error:
            raise ValueError(f"cache entry is missing: {name}") from error
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"cache entry is not a regular file: {name}")
    try:
        return root.resolve(strict=True)
    except OSError as error:
        raise ValueError("cache root cannot be resolved") from error


def _open_regular_binary(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cache entry cannot be opened safely: {path.name}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"cache entry is not a regular file: {path.name}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        raise


def _best_effort_close(handle: BinaryIO) -> None:
    try:
        handle.close()
    except BaseException:
        pass


def _close_handles(handles: Sequence[BinaryIO], *, suppress_errors: bool) -> None:
    first_error: BaseException | None = None
    for handle in handles:
        try:
            handle.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


def _close_descriptors(descriptors: Sequence[int], *, suppress_errors: bool) -> None:
    first_error: BaseException | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


def _write_all(handle: BinaryIO, payload: bytes, *, label: str) -> None:
    view = memoryview(payload)
    written_total = 0
    while written_total < len(view):
        written = handle.write(view[written_total:])
        remaining = len(view) - written_total
        if type(written) is not int or not 0 < written <= remaining:
            raise OSError(f"{label} write did not report valid forward progress")
        written_total += written


def _runtime_integer_digit_cap() -> int:
    configured = sys.get_int_max_str_digits()
    if configured <= 0:
        configured = sys.int_info.default_max_str_digits
    return max(configured, len(str(sys.maxsize)))


def _validate_elapsed_ns(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative int")
    try:
        decimal_digits = len(str(value))
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} exceeds the runtime integer digit limit") from error
    if decimal_digits > _runtime_integer_digit_cap():
        raise ValueError(f"{label} exceeds the runtime integer digit limit")
    return value


def _jsonl_record_limits(header: Mapping[str, object]) -> tuple[int, int]:
    inventory = _header_inventory(header)
    query_ids = _header_query_ids(header)
    max_query_bytes = max(len(canonical_json(query_id)) for query_id in query_ids)
    max_cluster_count = max(
        (
            row["cluster_count"]
            for row in inventory
            if type(row) is dict and type(row.get("cluster_count")) is int
        ),
        default=1,
    )
    integer_digits = _runtime_integer_digit_cap()

    # A receipt has two cluster-sized scalar lists (one SHA-256 and one
    # hexadecimal float), at most two excluded query IDs/assignments, and a
    # fixed number of integer-valued descriptor fields.  The generous fixed
    # allowance keeps this an upper bound without tying elapsed_ns to uint64.
    receipt_limit = (
        65_536 + 4 * max_query_bytes + 128 * max_cluster_count + 32 * integer_digits
    )
    timing_limit = 4_096 + 16 * integer_digits
    maximum_size_argument = sys.maxsize - 1
    return (
        min(receipt_limit, maximum_size_argument),
        min(timing_limit, maximum_size_argument),
    )


def _read_bounded_jsonl_line(handle: BinaryIO, *, limit: int, label: str) -> bytes:
    try:
        payload = handle.readline(limit + 1)
    except OSError as error:
        raise ValueError(f"{label} cannot be read") from error
    if type(payload) is not bytes:
        raise ValueError(f"{label} read did not return bytes")
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds its conservative byte limit")
    return payload


def _require_jsonl_exhausted(handle: BinaryIO, *, label: str) -> None:
    try:
        extra = handle.read(1)
    except OSError as error:
        raise ValueError(f"{label} cannot be read") from error
    if type(extra) is not bytes:
        raise ValueError(f"{label} read did not return bytes")
    if extra != b"":
        raise ValueError(f"{label} contains extra rows")


def _parse_canonical_json_record(payload: bytes, *, label: str) -> dict[str, object]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} is missing its canonical terminal newline")
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    try:
        expected = canonical_json(value) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if payload != expected:
        raise ValueError(f"{label} bytes are not canonical")
    return value


def _read_header(root: Path) -> dict[str, object]:
    handle = _open_regular_binary(root / "header.json")
    try:
        payload = handle.read()
    except BaseException:
        _close_handles((handle,), suppress_errors=True)
        raise
    else:
        _close_handles((handle,), suppress_errors=False)
    value = _parse_canonical_json_record(payload, label="cache header")
    return _validate_header(value)


def _parse_canonical_hex_float(value: object, *, label: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical hexadecimal float")
    try:
        parsed = float.fromhex(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} is not a hexadecimal float") from error
    if parsed.hex() != value or not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} is not a canonical finite nonnegative float")
    return parsed


def _validate_descriptor(
    value: object,
    *,
    label: str,
    dtype: str,
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, tuple[int, ...]]:
    if type(value) is not dict or set(value) != _DESCRIPTOR_KEYS:
        raise ValueError(f"{label} descriptor has an invalid contract")
    offset = value["offset"]
    byte_count = value["bytes"]
    shape_value = value["shape"]
    if type(offset) is not int or offset < 0:
        raise ValueError(f"{label} descriptor offset is invalid")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{label} descriptor byte length is invalid")
    if type(shape_value) is not list or any(
        type(dimension) is not int or dimension < 0 for dimension in shape_value
    ):
        raise ValueError(f"{label} descriptor shape is invalid")
    shape = tuple(shape_value)
    if value["dtype"] != dtype or type(value["dtype"]) is not str:
        raise ValueError(f"{label} descriptor dtype is invalid")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"{label} descriptor shape does not match fit semantics")
    expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    if byte_count != expected_bytes:
        raise ValueError(f"{label} shard byte length does not match its shape")
    return offset, byte_count, shape


def _read_slice(
    handle: BinaryIO,
    *,
    offset: int,
    byte_count: int,
    label: str,
    expected_offset: int | None,
) -> bytes:
    if expected_offset is not None and offset != expected_offset:
        raise ValueError(f"{label} descriptor offsets are not contiguous in sequence")
    if type(offset) is not int or offset < 0:
        raise ValueError(f"{label} descriptor offset is invalid")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{label} descriptor byte length is invalid")
    try:
        shard_size = os.fstat(handle.fileno()).st_size
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} shard byte length cannot be inspected") from error
    if offset > shard_size or byte_count > shard_size - offset:
        raise ValueError(f"{label} shard byte length is shorter than its receipt")
    try:
        handle.seek(offset)
        payload = handle.read(byte_count)
    except OSError as error:
        raise ValueError(f"{label} shard cannot be read") from error
    if len(payload) != byte_count:
        raise ValueError(f"{label} shard byte length is shorter than its receipt")
    return payload


def _validate_receipt_and_reconstruct(
    *,
    record: dict[str, object],
    header: Mapping[str, object],
    expected_row: Mapping[str, object],
    sequence: int,
    binary_handles: Mapping[str, BinaryIO],
    cursors: dict[str, int] | None,
    expected_center_width: int | None,
) -> tuple[str, StoredFitRecord, int]:
    if set(record) != _FIT_RECEIPT_KEYS:
        raise ValueError("fit receipt has an invalid exact-key contract")
    record_sha256 = record["record_sha256"]
    if not _is_sha256(record_sha256):
        raise ValueError("fit receipt record_sha256 is invalid")
    without_hash = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    if record_sha256 != framed_sha256("fit-cache-record", canonical_json(without_hash)):
        raise ValueError("fit receipt record hash does not match its payload")
    if record["schema"] != _FIT_RECEIPT_SCHEMA or type(record["schema"]) is not str:
        raise ValueError("fit receipt schema is invalid")
    if type(record["sequence"]) is not int or record["sequence"] != sequence:
        raise ValueError("fit receipt sequence is invalid")
    exclusion_kind = record["exclusion_kind"]
    inventory_index = record["inventory_index"]
    cluster_count = record["cluster_count"]
    if (
        type(exclusion_kind) is not str
        or exclusion_kind not in {"pair", "single"}
        or type(inventory_index) is not int
        or inventory_index < 0
        or type(cluster_count) is not int
        or not 1 <= cluster_count <= 65_536
        or exclusion_kind != expected_row["exclusion_kind"]
        or inventory_index != expected_row["inventory_index"]
        or cluster_count != expected_row["cluster_count"]
    ):
        raise ValueError("fit receipt does not match expected inventory")
    query_ids = _header_query_ids(header)
    excluded = _expected_excluded_query_ids(query_ids, exclusion_kind, inventory_index)
    excluded_value = record["excluded_query_ids"]
    if (
        type(excluded_value) is not list
        or any(type(query_id) is not str for query_id in excluded_value)
        or tuple(excluded_value) != excluded
    ):
        raise ValueError("fit receipt excluded queries are invalid")
    support = tuple(query_id for query_id in query_ids if query_id not in set(excluded))
    key = FitKey(
        support,
        cluster_count,
        header["embedding_sha256"],  # type: ignore[arg-type]
        header["clustering_contract_sha256"],  # type: ignore[arg-type]
    )
    fit_key_sha256 = record["fit_key_sha256"]
    scalar_digest_fields = (
        "fit_key_sha256",
        "support_query_ids_sha256",
        "embedding_sha256",
        "clustering_contract_sha256",
        "center_sha256",
        "fit_membership_sha256",
    )
    if any(not _is_sha256(record[field]) for field in scalar_digest_fields):
        raise ValueError("fit receipt contains an invalid digest")
    if (
        fit_key_sha256 != _fit_key_sha256(key)
        or record["support_query_ids_sha256"] != support_query_ids_sha256(support)
        or record["embedding_sha256"] != header["embedding_sha256"]
        or record["clustering_contract_sha256"] != header["clustering_contract_sha256"]
    ):
        raise ValueError("fit receipt key provenance does not match its inventory")

    center_descriptor = record["centers"]
    if type(center_descriptor) is not dict:
        raise ValueError("center descriptor is invalid")
    center_shape_value = center_descriptor.get("shape")
    if (
        type(center_shape_value) is not list
        or len(center_shape_value) != 2
        or any(type(value) is not int for value in center_shape_value)
        or center_shape_value[0] != cluster_count
        or center_shape_value[1] <= 0
    ):
        raise ValueError("center descriptor shape does not match cluster_count")
    center_width = center_shape_value[1]
    if expected_center_width is not None and center_width != expected_center_width:
        raise ValueError("center descriptor width changed between fit receipts")
    center_offset, center_bytes, center_shape = _validate_descriptor(
        center_descriptor,
        label="center",
        dtype="<f4",
        expected_shape=(cluster_count, center_width),
    )
    label_offset, label_bytes, label_shape = _validate_descriptor(
        record["labels"],
        label="label",
        dtype="<u2",
        expected_shape=(len(support),),
    )
    inertia_offset, inertia_bytes, inertia_shape = _validate_descriptor(
        record["cluster_inertia"],
        label="cluster inertia",
        dtype="<f8",
        expected_shape=(cluster_count,),
    )
    center_payload = _read_slice(
        binary_handles["centers"],
        offset=center_offset,
        byte_count=center_bytes,
        label="center",
        expected_offset=None if cursors is None else cursors["centers"],
    )
    label_payload = _read_slice(
        binary_handles["labels"],
        offset=label_offset,
        byte_count=label_bytes,
        label="label",
        expected_offset=None if cursors is None else cursors["labels"],
    )
    inertia_payload = _read_slice(
        binary_handles["cluster_inertia"],
        offset=inertia_offset,
        byte_count=inertia_bytes,
        label="cluster inertia",
        expected_offset=None if cursors is None else cursors["cluster_inertia"],
    )
    if cursors is not None:
        cursors["centers"] = center_offset + center_bytes
        cursors["labels"] = label_offset + label_bytes
        cursors["cluster_inertia"] = inertia_offset + inertia_bytes
    centers = np.frombuffer(center_payload, dtype="<f4").copy().reshape(center_shape)
    labels = np.frombuffer(label_payload, dtype="<u2").copy().reshape(label_shape)
    inertia_array = (
        np.frombuffer(inertia_payload, dtype="<f8").copy().reshape(inertia_shape)
    )
    centers = np.ascontiguousarray(centers, dtype="<f4")
    labels = np.ascontiguousarray(labels, dtype="<u2")
    inertia = tuple(float(value) for value in inertia_array)
    label_values = tuple(int(value) for value in labels)
    if any(value < 0 or value >= cluster_count for value in label_values):
        raise ValueError("fit receipt labels contain an out-of-range cluster ID")
    memberships = tuple(
        tuple(
            support[support_index]
            for support_index, label in enumerate(label_values)
            if label == cluster_id
        )
        for cluster_id in range(cluster_count)
    )
    membership_rows = _membership_rows(memberships)
    expected_center_sha256 = logical_array_sha256(
        centers,
        tag="canonical-centers",
        dtype="<f4",
        row_inventory=tuple(range(cluster_count)),
        column_inventory=(),
    )
    expected_fit_membership_sha256 = canonical_membership_sha256(membership_rows)
    expected_cluster_membership_sha256 = tuple(
        canonical_membership_sha256((membership_rows[cluster_id],))
        for cluster_id in range(cluster_count)
    )
    cluster_membership_value = record["cluster_membership_sha256"]
    if (
        type(cluster_membership_value) is not list
        or len(cluster_membership_value) != cluster_count
        or any(not _is_sha256(value) for value in cluster_membership_value)
        or tuple(cluster_membership_value) != expected_cluster_membership_sha256
        or record["center_sha256"] != expected_center_sha256
        or record["fit_membership_sha256"] != expected_fit_membership_sha256
    ):
        raise ValueError("fit receipt hashes do not match reconstructed arrays")

    inertia_hex_value = record["cluster_inertia_hex"]
    if type(inertia_hex_value) is not list or len(inertia_hex_value) != cluster_count:
        raise ValueError("fit receipt cluster inertia hex list is invalid")
    parsed_inertia = tuple(
        _parse_canonical_hex_float(value, label="cluster inertia")
        for value in inertia_hex_value
    )
    if tuple(value.hex() for value in inertia) != tuple(inertia_hex_value):
        raise ValueError("fit receipt inertia bytes do not match hexadecimal values")
    if parsed_inertia != inertia:
        raise ValueError("fit receipt inertia values do not match shard bytes")
    sklearn_inertia = _parse_canonical_hex_float(
        record["sklearn_inertia_hex"], label="sklearn inertia"
    )
    recomputed_inertia = _parse_canonical_hex_float(
        record["recomputed_total_inertia_hex"], label="recomputed inertia"
    )
    if recomputed_inertia.hex() != math.fsum(inertia).hex():
        raise ValueError("fit receipt recomputed inertia does not equal exact fsum")
    if not math.isclose(
        sklearn_inertia,
        recomputed_inertia,
        rel_tol=1e-5,
        abs_tol=1e-8,
    ):
        raise ValueError("fit receipt sklearn inertia differs from recomputation")
    n_iter = record["n_iter"]
    if type(n_iter) is not int or not 1 <= n_iter <= 299:
        raise ValueError("fit receipt n_iter is invalid")
    if (
        type(record["fit_count_increment"]) is not int
        or record["fit_count_increment"] != 1
        or type(record["start_count_increment"]) is not int
        or record["start_count_increment"] != 10
    ):
        raise ValueError("fit receipt count increments are invalid")

    assignments_value = record["assignments"]
    if type(assignments_value) is not list or len(assignments_value) != len(excluded):
        raise ValueError("fit receipt assignments have an invalid count")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    assignments: list[TargetAssignment] = []
    for assignment_value, expected_query_id in zip(
        assignments_value, excluded, strict=True
    ):
        if (
            type(assignment_value) is not dict
            or set(assignment_value) != _ASSIGNMENT_KEYS
        ):
            raise ValueError("fit receipt assignment has an invalid contract")
        query_id = assignment_value["query_id"]
        cluster_id = assignment_value["canonical_cluster_id"]
        cluster_support = assignment_value["cluster_support"]
        if query_id != expected_query_id or type(query_id) is not str:
            raise ValueError("fit receipt assignment query order is invalid")
        if type(cluster_id) is not int or not 0 <= cluster_id < cluster_count:
            raise ValueError("fit receipt assignment cluster ID is invalid")
        member_indices = tuple(
            query_index[member_id] for member_id in memberships[cluster_id]
        )
        if (
            type(cluster_support) is not int
            or cluster_support <= 0
            or cluster_support != len(member_indices)
        ):
            raise ValueError("fit receipt assignment support is invalid")
        squared_distance = _parse_canonical_hex_float(
            assignment_value["squared_distance_hex"],
            label="assignment squared distance",
        )
        assignments.append(
            TargetAssignment(
                query_id=query_id,
                canonical_cluster_id=cluster_id,
                squared_distance=squared_distance,
                member_indices=member_indices,
                cluster_support=cluster_support,
            )
        )
    fit = CanonicalClusterFit(
        key=key,
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=record["fit_membership_sha256"],  # type: ignore[arg-type]
        cluster_membership_sha256=tuple(cluster_membership_value),
        center_sha256=record["center_sha256"],  # type: ignore[arg-type]
        cluster_inertia=inertia,
        sklearn_inertia=sklearn_inertia,
        recomputed_total_inertia=recomputed_inertia,
        n_iter=n_iter,
    )
    assignment_tuple, validated_excluded = _validate_fit_and_assignments(
        header=header,
        fit=fit,
        assignments=tuple(assignments),
        exclusion_kind=exclusion_kind,
        inventory_index=inventory_index,
        excluded_query_ids=excluded,
        expected_row=expected_row,
    )
    stored = StoredFitRecord(
        fit=fit,
        assignments=assignment_tuple,
        exclusion_kind=exclusion_kind,
        inventory_index=inventory_index,
        excluded_query_ids=validated_excluded,
    )
    if type(fit_key_sha256) is not str:
        raise ValueError("fit receipt key digest has an invalid type")
    return fit_key_sha256, stored, center_width


def _validate_timing_record(
    record: dict[str, object],
    *,
    expected_row: Mapping[str, object],
    sequence: int,
    corresponding_fit: _IndexEntry,
) -> None:
    if set(record) != _TIMING_KEYS:
        raise ValueError("timing receipt has an invalid exact-key contract")
    if record["schema"] != _TIMING_SCHEMA or type(record["schema"]) is not str:
        raise ValueError("timing receipt schema is invalid")
    if type(record["sequence"]) is not int or record["sequence"] != sequence:
        raise ValueError("timing receipt sequence is invalid")
    if (
        type(record["exclusion_kind"]) is not str
        or record["exclusion_kind"] != expected_row["exclusion_kind"]
        or type(record["inventory_index"]) is not int
        or record["inventory_index"] != expected_row["inventory_index"]
        or type(record["cluster_count"]) is not int
        or record["cluster_count"] != expected_row["cluster_count"]
    ):
        raise ValueError("timing receipt does not match expected inventory")
    _validate_elapsed_ns(record["elapsed_ns"], label="timing receipt elapsed_ns")
    if (
        type(record["n_iter"]) is not int
        or not 1 <= record["n_iter"] <= 299
        or record["n_iter"] != corresponding_fit.n_iter
    ):
        raise ValueError("timing receipt n_iter does not match its fit")


def _scan_streams(root: Path, header: Mapping[str, object]) -> _ScanResult:
    inventory = _header_inventory(header)
    receipt_limit, timing_limit = _jsonl_record_limits(header)
    handles: dict[str, BinaryIO] = {}
    with ExitStack() as stack:
        for shard_name, path_name in (
            ("centers", "centers.f4"),
            ("labels", "labels.u2"),
            ("cluster_inertia", "cluster_inertia.f8"),
            ("fit_receipts", "fit_receipts.jsonl"),
            ("timings", "timings.jsonl"),
        ):
            handle = _open_regular_binary(root / path_name)
            stack.callback(_best_effort_close, handle)
            handles[shard_name] = handle
        receipt_handle = handles["fit_receipts"]
        timing_handle = handles["timings"]
        cursors = {"centers": 0, "labels": 0, "cluster_inertia": 0}
        index: dict[str, _IndexEntry] = {}
        inventory_index_map: dict[tuple[str, int, int], str] = {}
        receipt_offset = 0
        center_width: int | None = None
        for fit_count, expected_row in enumerate(inventory):
            line = _read_bounded_jsonl_line(
                receipt_handle,
                limit=receipt_limit,
                label=f"fit receipt row {fit_count}",
            )
            if line == b"":
                raise ValueError("fit receipt stream is incomplete")
            record = _parse_canonical_json_record(
                line, label=f"fit receipt row {fit_count}"
            )
            key_sha256, stored, observed_width = _validate_receipt_and_reconstruct(
                record=record,
                header=header,
                expected_row=expected_row,
                sequence=fit_count,
                binary_handles=handles,
                cursors=cursors,
                expected_center_width=center_width,
            )
            center_width = observed_width
            if key_sha256 in index:
                raise ValueError("fit receipt stream contains a duplicate fit key")
            triplet = (
                stored.exclusion_kind,
                stored.inventory_index,
                stored.fit.key.cluster_count,
            )
            if triplet in inventory_index_map:
                raise ValueError("fit receipt stream contains duplicate inventory")
            entry = _IndexEntry(
                key=stored.fit.key,
                sequence=fit_count,
                exclusion_kind=stored.exclusion_kind,
                inventory_index=stored.inventory_index,
                excluded_query_ids=stored.excluded_query_ids,
                n_iter=stored.fit.n_iter,
                receipt_offset=receipt_offset,
                receipt_bytes=len(line),
                receipt_sha256=_payload_sha256(line),
                timing_offset=None,
                timing_bytes=None,
                timing_sha256=None,
            )
            index[key_sha256] = entry
            inventory_index_map[triplet] = key_sha256
            receipt_offset += len(line)
        _require_jsonl_exhausted(receipt_handle, label="fit receipt stream")
        fit_count = len(inventory)
        size_labels = {
            "centers": "center",
            "labels": "label",
            "cluster_inertia": "cluster inertia",
        }
        for shard_name in ("centers", "labels", "cluster_inertia"):
            actual_size = os.fstat(handles[shard_name].fileno()).st_size
            if actual_size != cursors[shard_name]:
                raise ValueError(
                    f"{size_labels[shard_name]} shard byte length does not match receipts"
                )

        timing_offset = 0
        for timing_count, expected_row in enumerate(inventory):
            line = _read_bounded_jsonl_line(
                timing_handle,
                limit=timing_limit,
                label=f"timing receipt row {timing_count}",
            )
            if line == b"":
                raise ValueError("timing receipt stream is incomplete")
            record = _parse_canonical_json_record(
                line, label=f"timing receipt row {timing_count}"
            )
            triplet = (
                expected_row["exclusion_kind"],
                expected_row["inventory_index"],
                expected_row["cluster_count"],
            )
            key_sha256 = inventory_index_map.get(triplet)  # type: ignore[arg-type]
            if key_sha256 is None:
                raise ValueError("timing receipt has no corresponding fit")
            corresponding_fit = index[key_sha256]
            _validate_timing_record(
                record,
                expected_row=expected_row,
                sequence=timing_count,
                corresponding_fit=corresponding_fit,
            )
            index[key_sha256] = replace(
                corresponding_fit,
                timing_offset=timing_offset,
                timing_bytes=len(line),
                timing_sha256=_payload_sha256(line),
            )
            timing_offset += len(line)
        _require_jsonl_exhausted(timing_handle, label="timing receipt stream")
        timing_count = len(inventory)
        return _ScanResult(
            index=index,
            inventory_index=inventory_index_map,
            fit_count=fit_count,
            timing_count=timing_count,
            center_width=center_width,
        )


class StreamingFitCache:
    @classmethod
    def create(
        cls, root: str | Path, *, header: Mapping[str, object]
    ) -> StreamingFitCache:
        owned_header = _validate_header(header)
        requested_root = Path(root)
        requested_root.mkdir()
        try:
            cache_root = requested_root.resolve(strict=True)
        except BaseException:
            try:
                requested_root.rmdir()
            except BaseException:
                pass
            raise
        opened: list[BinaryIO] = []
        owned_streams: list[tuple[Path, tuple[int, int] | None]] = []
        try:
            for name in _STREAM_NAMES:
                path = cache_root / name
                handle = path.open("xb")
                opened.append(handle)
                owned_streams.append((path, None))
                stream_status = os.fstat(handle.fileno())
                owned_streams[-1] = (
                    path,
                    (stream_status.st_dev, stream_status.st_ino),
                )
        except BaseException:
            for index, ((path, owned_identity), handle) in enumerate(
                zip(owned_streams, opened, strict=True)
            ):
                if owned_identity is not None:
                    continue
                try:
                    stream_status = os.fstat(handle.fileno())
                except BaseException:
                    continue
                owned_streams[index] = (
                    path,
                    (stream_status.st_dev, stream_status.st_ino),
                )
            _close_handles(opened, suppress_errors=True)
            for path, owned_identity in owned_streams:
                try:
                    status = os.lstat(path)
                except BaseException:
                    continue
                if (
                    owned_identity is not None
                    and (
                        status.st_dev,
                        status.st_ino,
                    )
                    == owned_identity
                ):
                    try:
                        path.unlink()
                    except BaseException:
                        pass
            try:
                cache_root.rmdir()
            except BaseException:
                pass
            raise

        self = cls.__new__(cls)
        self.root = cache_root
        self._header = owned_header
        (
            self._center_handle,
            self._label_handle,
            self._inertia_handle,
            self._receipt_handle,
            self._timing_handle,
        ) = opened
        self._index: dict[str, _IndexEntry] = {}
        self._inventory_index: dict[tuple[str, int, int], str] = {}
        self._resident: dict[str, StoredFitRecord] = {}
        self._fit_count = 0
        self._timing_count = 0
        self._center_width: int | None = None
        self._state: Literal["writing", "finalizing", "finalized", "failed"] = "writing"
        self._movable = False
        return self

    @classmethod
    def open_readonly(cls, root: str | Path) -> StreamingFitCache:
        cache_root = _validate_layout(Path(root), _FINAL_NAMES)
        header = _read_header(cache_root)
        scanned = _scan_streams(cache_root, header)
        self = cls.__new__(cls)
        self.root = cache_root
        self._header = header
        self._center_handle = None
        self._label_handle = None
        self._inertia_handle = None
        self._receipt_handle = None
        self._timing_handle = None
        self._index = scanned.index
        self._inventory_index = scanned.inventory_index
        self._resident = {}
        self._fit_count = scanned.fit_count
        self._timing_count = scanned.timing_count
        self._center_width = scanned.center_width
        self._state = "finalized"
        self._movable = False
        return self

    def _require_writing(self) -> None:
        if self._state == "finalized":
            raise RuntimeError("cache is finalized and cannot be mutated")
        if self._state != "writing":
            raise RuntimeError(f"cache is {self._state} and cannot be mutated")

    @staticmethod
    def _append_array(handle: BinaryIO, value: np.ndarray[Any, Any]) -> tuple[int, int]:
        offset = handle.tell()
        payload = value.tobytes(order="C")
        _write_all(handle, payload, label="array shard")
        return offset, len(payload)

    @staticmethod
    def _append_jsonl(
        handle: BinaryIO, record: Mapping[str, object]
    ) -> tuple[int, int]:
        offset = handle.tell()
        payload = canonical_json(record) + b"\n"
        _write_all(handle, payload, label="JSONL record")
        handle.flush()
        return offset, len(payload)

    def append_fit(
        self,
        fit: CanonicalClusterFit,
        assignments: Sequence[TargetAssignment],
        *,
        exclusion_kind: Literal["pair", "single"],
        inventory_index: int,
        excluded_query_ids: Sequence[str],
    ) -> None:
        self._require_writing()
        try:
            key_sha256 = _fit_key_sha256(fit.key)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("fit has an invalid cache key") from error
        existing = self._index.get(key_sha256)
        if existing is not None:
            if existing.key == fit.key:
                raise ValueError(f"duplicate fit key: {key_sha256}")
            raise ValueError(f"fit key SHA-256 collision: {key_sha256}")

        expected_row = _expected_inventory_row(self._header, self._fit_count)
        assignment_tuple, expected_excluded = _validate_fit_and_assignments(
            header=self._header,
            fit=fit,
            assignments=assignments,
            exclusion_kind=exclusion_kind,
            inventory_index=inventory_index,
            excluded_query_ids=excluded_query_ids,
            expected_row=expected_row,
        )
        center_width = fit.centers.shape[1]
        if self._center_width is not None and center_width != self._center_width:
            raise ValueError("fit center width changed between cache rows")
        centers = np.ascontiguousarray(fit.centers, dtype="<f4")
        labels = np.ascontiguousarray(fit.labels, dtype="<u2")
        inertia = np.ascontiguousarray(fit.cluster_inertia, dtype="<f8")
        center_offset, center_bytes = self._append_array(self._center_handle, centers)
        label_offset, label_bytes = self._append_array(self._label_handle, labels)
        inertia_offset, inertia_bytes = self._append_array(
            self._inertia_handle, inertia
        )
        record_without_hash: dict[str, object] = {
            "schema": _FIT_RECEIPT_SCHEMA,
            "sequence": self._fit_count,
            "exclusion_kind": exclusion_kind,
            "inventory_index": inventory_index,
            "excluded_query_ids": list(expected_excluded),
            "cluster_count": fit.key.cluster_count,
            "fit_key_sha256": key_sha256,
            "support_query_ids_sha256": support_query_ids_sha256(
                fit.key.support_query_ids
            ),
            "embedding_sha256": fit.key.embedding_sha256,
            "clustering_contract_sha256": fit.key.clustering_contract_sha256,
            "centers": {
                "offset": center_offset,
                "bytes": center_bytes,
                "shape": list(centers.shape),
                "dtype": "<f4",
            },
            "labels": {
                "offset": label_offset,
                "bytes": label_bytes,
                "shape": list(labels.shape),
                "dtype": "<u2",
            },
            "cluster_inertia": {
                "offset": inertia_offset,
                "bytes": inertia_bytes,
                "shape": [len(fit.cluster_inertia)],
                "dtype": "<f8",
            },
            "center_sha256": fit.center_sha256,
            "fit_membership_sha256": fit.fit_membership_sha256,
            "cluster_membership_sha256": list(fit.cluster_membership_sha256),
            "cluster_inertia_hex": [value.hex() for value in fit.cluster_inertia],
            "sklearn_inertia_hex": fit.sklearn_inertia.hex(),
            "recomputed_total_inertia_hex": fit.recomputed_total_inertia.hex(),
            "n_iter": fit.n_iter,
            "assignments": [
                {
                    "query_id": item.query_id,
                    "canonical_cluster_id": item.canonical_cluster_id,
                    "squared_distance_hex": item.squared_distance.hex(),
                    "cluster_support": item.cluster_support,
                }
                for item in assignment_tuple
            ],
            "fit_count_increment": 1,
            "start_count_increment": 10,
        }
        record = {
            **record_without_hash,
            "record_sha256": framed_sha256(
                "fit-cache-record", canonical_json(record_without_hash)
            ),
        }
        receipt_sha256 = _payload_sha256(canonical_json(record) + b"\n")
        receipt_offset, receipt_bytes = self._append_jsonl(self._receipt_handle, record)
        entry = _IndexEntry(
            key=fit.key,
            sequence=self._fit_count,
            exclusion_kind=exclusion_kind,
            inventory_index=inventory_index,
            excluded_query_ids=expected_excluded,
            n_iter=fit.n_iter,
            receipt_offset=receipt_offset,
            receipt_bytes=receipt_bytes,
            receipt_sha256=receipt_sha256,
            timing_offset=None,
            timing_bytes=None,
            timing_sha256=None,
        )
        self._index[key_sha256] = entry
        self._inventory_index[
            (exclusion_kind, inventory_index, fit.key.cluster_count)
        ] = key_sha256
        self._resident[key_sha256] = StoredFitRecord(
            fit=fit,
            assignments=assignment_tuple,
            exclusion_kind=exclusion_kind,
            inventory_index=inventory_index,
            excluded_query_ids=expected_excluded,
        )
        self._center_width = center_width
        self._fit_count += 1

    def _load_index_entry(self, entry: _IndexEntry) -> StoredFitRecord:
        if self._state == "writing":
            for handle in (
                self._center_handle,
                self._label_handle,
                self._inertia_handle,
                self._receipt_handle,
                self._timing_handle,
            ):
                handle.flush()
        receipt_handle = _open_regular_binary(self.root / "fit_receipts.jsonl")
        try:
            payload = _read_slice(
                receipt_handle,
                offset=entry.receipt_offset,
                byte_count=entry.receipt_bytes,
                label="fit receipt",
                expected_offset=None,
            )
        except BaseException:
            _close_handles((receipt_handle,), suppress_errors=True)
            raise
        else:
            _close_handles((receipt_handle,), suppress_errors=False)
        if _payload_sha256(payload) != entry.receipt_sha256:
            raise ValueError("fit receipt no longer matches its lightweight index")

        timing_fields = (entry.timing_offset, entry.timing_bytes, entry.timing_sha256)
        if any(value is None for value in timing_fields):
            if not all(value is None for value in timing_fields):
                raise ValueError("timing receipt index span is internally inconsistent")
        else:
            timing_offset, timing_bytes, timing_sha256 = timing_fields
            if (
                type(timing_offset) is not int
                or type(timing_bytes) is not int
                or type(timing_sha256) is not str
            ):
                raise ValueError("timing receipt index span is internally inconsistent")
            timing_handle = _open_regular_binary(self.root / "timings.jsonl")
            try:
                timing_payload = _read_slice(
                    timing_handle,
                    offset=timing_offset,
                    byte_count=timing_bytes,
                    label="timing receipt",
                    expected_offset=None,
                )
            except BaseException:
                _close_handles((timing_handle,), suppress_errors=True)
                raise
            else:
                _close_handles((timing_handle,), suppress_errors=False)
            if _payload_sha256(timing_payload) != timing_sha256:
                raise ValueError(
                    "timing receipt no longer matches its lightweight index"
                )
        record = _parse_canonical_json_record(
            payload, label=f"fit receipt row {entry.sequence}"
        )
        binary_handles: dict[str, BinaryIO] = {}
        try:
            binary_handles["centers"] = _open_regular_binary(self.root / "centers.f4")
            binary_handles["labels"] = _open_regular_binary(self.root / "labels.u2")
            binary_handles["cluster_inertia"] = _open_regular_binary(
                self.root / "cluster_inertia.f8"
            )
            key_sha256, stored, _center_width = _validate_receipt_and_reconstruct(
                record=record,
                header=self._header,
                expected_row=_expected_inventory_row(self._header, entry.sequence),
                sequence=entry.sequence,
                binary_handles=binary_handles,
                cursors=None,
                expected_center_width=self._center_width,
            )
        except BaseException:
            _close_handles(tuple(binary_handles.values()), suppress_errors=True)
            raise
        else:
            _close_handles(tuple(binary_handles.values()), suppress_errors=False)
        if (
            key_sha256 != _fit_key_sha256(entry.key)
            or stored.fit.key != entry.key
            or stored.exclusion_kind != entry.exclusion_kind
            or stored.inventory_index != entry.inventory_index
            or stored.excluded_query_ids != entry.excluded_query_ids
            or stored.fit.n_iter != entry.n_iter
        ):
            raise ValueError("fit receipt no longer matches its lightweight index")
        return stored

    def load_fit(self, key: FitKey) -> StoredFitRecord | None:
        try:
            key_sha256 = _fit_key_sha256(key)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("fit key is invalid") from error
        entry = self._index.get(key_sha256)
        if entry is None:
            return None
        if entry.key != key:
            raise ValueError(f"fit key SHA-256 collision: {key_sha256}")
        resident = self._resident.get(key_sha256)
        if resident is not None:
            return resident
        loaded = self._load_index_entry(entry)
        self._resident[key_sha256] = loaded
        return loaded

    def load_by_inventory(
        self,
        *,
        exclusion_kind: Literal["pair", "single"],
        inventory_index: int,
        cluster_count: Literal[20, 40],
    ) -> StoredFitRecord:
        if (
            type(exclusion_kind) is not str
            or exclusion_kind not in {"pair", "single"}
            or type(inventory_index) is not int
            or inventory_index < 0
            or type(cluster_count) is not int
            or not 1 <= cluster_count <= 65_536
        ):
            raise ValueError("inventory lookup is invalid")
        triplet = (exclusion_kind, inventory_index, cluster_count)
        key_sha256 = self._inventory_index.get(triplet)
        if key_sha256 is None:
            raise KeyError(f"fit inventory row is absent: {triplet!r}")
        entry = self._index[key_sha256]
        resident = self._resident.get(key_sha256)
        if resident is not None:
            return resident
        loaded = self._load_index_entry(entry)
        self._resident[key_sha256] = loaded
        return loaded

    def release(self, key: FitKey) -> None:
        try:
            key_sha256 = _fit_key_sha256(key)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("fit key is invalid") from error
        entry = self._index.get(key_sha256)
        if entry is None:
            return
        if entry.key != key:
            raise ValueError(f"fit key SHA-256 collision: {key_sha256}")
        self._resident.pop(key_sha256, None)

    def append_timing(
        self,
        *,
        exclusion_kind: Literal["pair", "single"],
        inventory_index: int,
        cluster_count: Literal[20, 40],
        elapsed_ns: int,
        n_iter: int,
    ) -> None:
        self._require_writing()
        inventory = _header_inventory(self._header)
        if self._timing_count >= len(inventory):
            raise ValueError("cache timing inventory is exhausted")
        expected = inventory[self._timing_count]
        if (
            type(exclusion_kind) is not str
            or exclusion_kind not in {"pair", "single"}
            or type(inventory_index) is not int
            or inventory_index < 0
            or type(cluster_count) is not int
            or not 1 <= cluster_count <= 65_536
            or exclusion_kind != expected["exclusion_kind"]
            or inventory_index != expected["inventory_index"]
            or cluster_count != expected["cluster_count"]
        ):
            raise ValueError("timing does not match the next expected inventory row")
        elapsed_ns = _validate_elapsed_ns(elapsed_ns, label="elapsed_ns")
        if type(n_iter) is not int or not 1 <= n_iter <= 299:
            raise ValueError("n_iter must be an exact int in 1..299")
        key_sha256 = self._inventory_index.get(
            (exclusion_kind, inventory_index, cluster_count)
        )
        if key_sha256 is None:
            raise ValueError("timing cannot precede its corresponding fit")
        entry = self._index[key_sha256]
        if entry.n_iter != n_iter:
            raise ValueError("timing n_iter does not match the fit receipt")
        record: dict[str, object] = {
            "schema": _TIMING_SCHEMA,
            "sequence": self._timing_count,
            "exclusion_kind": exclusion_kind,
            "inventory_index": inventory_index,
            "cluster_count": cluster_count,
            "elapsed_ns": elapsed_ns,
            "n_iter": n_iter,
        }
        timing_sha256 = _payload_sha256(canonical_json(record) + b"\n")
        timing_offset, timing_bytes = self._append_jsonl(self._timing_handle, record)
        self._index[key_sha256] = replace(
            entry,
            timing_offset=timing_offset,
            timing_bytes=timing_bytes,
            timing_sha256=timing_sha256,
        )
        self._timing_count += 1

    def finalize(self) -> dict[str, object]:
        self._require_writing()
        counts = self._header["expected_counts"]
        if type(counts) is not dict:
            raise ValueError("validated expected_counts changed type")
        expected_fit_count = counts["fit_count"]
        expected_timing_count = counts["timing_record_count"]
        if (
            self._fit_count != expected_fit_count
            or self._timing_count != expected_timing_count
        ):
            raise ValueError("cache cannot finalize before all fits and timings exist")

        self._state = "finalizing"
        self._movable = False
        handles = (
            self._center_handle,
            self._label_handle,
            self._inertia_handle,
            self._receipt_handle,
            self._timing_handle,
        )
        header_path = self.root / "header.json"
        header_created = False
        header_identity: tuple[int, int] | None = None
        header_handle: BinaryIO | None = None
        directory_fd: int | None = None
        try:
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
            for handle in handles:
                handle.close()

            _validate_layout(self.root, _STREAM_NAMES)
            scanned = _scan_streams(self.root, self._header)
            if (
                scanned.index != self._index
                or scanned.inventory_index != self._inventory_index
                or scanned.fit_count != self._fit_count
                or scanned.timing_count != self._timing_count
                or scanned.center_width != self._center_width
            ):
                raise ValueError("cache payload differs from the in-memory fit index")

            header_handle = header_path.open("xb")
            try:
                header_created = True
                header_status = os.fstat(header_handle.fileno())
                header_identity = (header_status.st_dev, header_status.st_ino)
                _write_all(
                    header_handle,
                    canonical_json(self._header) + b"\n",
                    label="cache header",
                )
                header_handle.flush()
                os.fsync(header_handle.fileno())
            except BaseException:
                if header_identity is None:
                    try:
                        header_status = os.fstat(header_handle.fileno())
                    except BaseException:
                        pass
                    else:
                        header_identity = (
                            header_status.st_dev,
                            header_status.st_ino,
                        )
                _close_handles((header_handle,), suppress_errors=True)
                raise
            else:
                _close_handles((header_handle,), suppress_errors=False)
            header_handle = None
            directory_fd = os.open(
                self.root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            except BaseException:
                _close_descriptors((directory_fd,), suppress_errors=True)
                raise
            else:
                _close_descriptors((directory_fd,), suppress_errors=False)
            directory_fd = None
            independently_validated = type(self).open_readonly(self.root)
            del independently_validated
        except BaseException:
            _close_handles(handles, suppress_errors=True)
            if header_handle is not None:
                _close_handles((header_handle,), suppress_errors=True)
            if directory_fd is not None:
                _close_descriptors((directory_fd,), suppress_errors=True)
            if header_created:
                try:
                    status = os.lstat(header_path)
                    if (
                        header_identity is not None
                        and (
                            status.st_dev,
                            status.st_ino,
                        )
                        == header_identity
                    ):
                        header_path.unlink()
                except BaseException:
                    pass
                try:
                    directory_fd = os.open(
                        self.root,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                except BaseException:
                    pass
                else:
                    try:
                        os.fsync(directory_fd)
                    except BaseException:
                        pass
                    finally:
                        try:
                            os.close(directory_fd)
                        except BaseException:
                            pass
            self._state = "failed"
            self._movable = False
            raise

        self._state = "finalized"
        self._movable = True
        return copy.deepcopy(self._header)

    def finalize_into(self, destination: str | Path) -> dict[str, object]:
        if self._state != "finalized":
            raise RuntimeError("cache must be finalized before it can be moved")
        if not self._movable:
            raise RuntimeError("cache is not an independently movable finalized writer")
        destination_path = Path(destination)
        try:
            os.lstat(destination_path)
        except FileNotFoundError:
            pass
        except OSError:
            raise
        else:
            raise FileExistsError(f"cache destination exists: {destination_path}")
        destination_parent = destination_path.parent.resolve(strict=True)
        if not stat.S_ISDIR(os.lstat(destination_parent).st_mode):
            raise OSError("cache destination parent is not a directory")
        source_device = self.root.stat().st_dev
        destination_device = destination_parent.stat().st_dev
        if source_device != destination_device:
            raise OSError("cache and destination must share one filesystem")

        rename_noreplace(self.root, destination_path)
        self.root = destination_path.resolve(strict=True)
        self._movable = False
        directory_fd = os.open(
            destination_parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        except BaseException:
            _close_descriptors((directory_fd,), suppress_errors=True)
            raise
        else:
            _close_descriptors((directory_fd,), suppress_errors=False)
        return copy.deepcopy(self._header)
