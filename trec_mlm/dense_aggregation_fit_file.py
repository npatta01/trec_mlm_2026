from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Literal

import h5py
import numpy as np

from trec_mlm.dense_aggregation_clustering import (
    CanonicalClusterFit,
    FitKey,
    TargetAssignment,
)
from trec_mlm.dense_aggregation_hashing import (
    canonical_json,
    canonical_membership_sha256,
    logical_array_sha256,
)


FIT_FILE_SCHEMA = "trec-mlm-dense-aggregation-fit-file-final"
CLUSTER_COUNTS = (20, 40)

_METADATA_KEYS = {
    "query_ids",
    "query_input_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "clustering_contract_sha256",
    "implementation_inventory_sha256",
    "dependency_inventory_sha256",
    "runtime_inventory_sha256",
    "numerical_environment_sha256",
    "protocol_sha256",
}
_HASH_KEYS = _METADATA_KEYS - {"query_ids"}
_LABEL_SENTINEL = np.iinfo(np.uint16).max
_INDEX_SENTINEL = -1
_CLUSTER_SENTINEL = np.iinfo(np.uint16).max


@dataclass(frozen=True, slots=True)
class FitInventoryRow:
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_query_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StoredFitRecord:
    fit: CanonicalClusterFit
    assignments: tuple[TargetAssignment, ...]
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_query_ids: tuple[str, ...]
    elapsed_ns: int


def _validated_metadata(
    metadata: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...]]:
    if set(metadata) != _METADATA_KEYS:
        raise ValueError("metadata must contain exactly the fit-file metadata keys")
    query_value = metadata["query_ids"]
    if isinstance(query_value, (str, bytes)) or not isinstance(query_value, Sequence):
        raise ValueError("metadata query_ids must be a sequence")
    query_ids = tuple(query_value)
    if (
        len(query_ids) != 322
        or any(type(query_id) is not str or not query_id for query_id in query_ids)
        or len(set(query_ids)) != len(query_ids)
    ):
        raise ValueError("metadata query_ids must contain 322 nonempty unique strings")
    for key in _HASH_KEYS:
        value = metadata[key]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"metadata {key} must be lowercase 64-character hex")
    normalized = dict(metadata)
    normalized["query_ids"] = list(query_ids)
    canonical_json(normalized)
    return normalized, query_ids


def _validated_inventory(
    inventory: Sequence[FitInventoryRow], *, query_count: int
) -> tuple[FitInventoryRow, ...]:
    rows = tuple(inventory)
    if not rows:
        raise ValueError("inventory must not be empty")
    seen_indices: dict[str, set[int]] = {"pair": set(), "single": set()}
    seen_single = False
    for row in rows:
        if type(row) is not FitInventoryRow:
            raise ValueError("inventory rows must be FitInventoryRow values")
        if row.exclusion_kind not in ("pair", "single"):
            raise ValueError("inventory exclusion_kind must be pair or single")
        if row.exclusion_kind == "single":
            seen_single = True
        elif seen_single:
            raise ValueError("pair inventory rows must precede single rows")
        expected_length = 2 if row.exclusion_kind == "pair" else 1
        if (
            type(row.inventory_index) is not int
            or not 0 <= row.inventory_index <= 2**32 - 1
            or row.inventory_index in seen_indices[row.exclusion_kind]
        ):
            raise ValueError(
                "inventory_index must be a unique uint32 within exclusion kind"
            )
        excluded = row.excluded_query_indices
        if (
            type(excluded) is not tuple
            or len(excluded) != expected_length
            or any(type(index) is not int for index in excluded)
            or tuple(sorted(excluded)) != excluded
            or len(set(excluded)) != len(excluded)
            or any(index < 0 or index >= query_count for index in excluded)
        ):
            raise ValueError("inventory excluded query indices are invalid")
        seen_indices[row.exclusion_kind].add(row.inventory_index)
    return rows


def initialize_fit_datasets(
    h5: h5py.File,
    *,
    metadata: Mapping[str, object],
    inventory: Sequence[FitInventoryRow],
    embedding_width: int,
) -> None:
    def create_fit_dataset(
        group: h5py.Group, name: str, shape: tuple[int, ...], dtype: str
    ) -> h5py.Dataset:
        return group.create_dataset(
            name,
            shape=shape,
            dtype=np.dtype(dtype),
            chunks=(1, *shape[1:]),
            fletcher32=True,
        )

    if len(h5) or len(h5.attrs):
        raise ValueError("HDF5 fit file must be empty")
    normalized_metadata, query_ids = _validated_metadata(metadata)
    rows = _validated_inventory(inventory, query_count=len(query_ids))
    if type(embedding_width) is not int or not 1 <= embedding_width <= 2**32 - 1:
        raise ValueError("embedding_width must be a positive uint32-sized integer")

    h5.attrs["schema_id"] = FIT_FILE_SCHEMA
    h5.attrs.create("complete", np.uint8(0), dtype="u1")
    h5.attrs.create("inventory_count", np.uint64(len(rows)), dtype="<u8")
    h5.attrs.create("embedding_width", np.uint32(embedding_width), dtype="<u4")
    metadata_group = h5.create_group("metadata")
    metadata_bytes = canonical_json(normalized_metadata)
    metadata_group.create_dataset(
        "canonical_json", data=np.frombuffer(metadata_bytes, dtype="u1"), dtype="u1"
    )

    inventory_group = h5.create_group("inventory")
    inventory_group.create_dataset(
        "exclusion_kind",
        data=np.asarray(
            [0 if row.exclusion_kind == "pair" else 1 for row in rows], dtype="u1"
        ),
        dtype="u1",
    )
    inventory_group.create_dataset(
        "inventory_index",
        data=np.asarray([row.inventory_index for row in rows], dtype="<u4"),
        dtype="<u4",
    )
    excluded_array = np.full((len(rows), 2), _INDEX_SENTINEL, dtype="<i2")
    support_count = np.empty(len(rows), dtype="<u2")
    for index, row in enumerate(rows):
        excluded_array[index, : len(row.excluded_query_indices)] = (
            row.excluded_query_indices
        )
        support_count[index] = len(query_ids) - len(row.excluded_query_indices)
    inventory_group.create_dataset(
        "excluded_query_indices", data=excluded_array, dtype="<i2"
    )
    inventory_group.create_dataset("support_count", data=support_count, dtype="<u2")

    fits_group = h5.create_group("fits")
    for cluster_count in CLUSTER_COUNTS:
        group = fits_group.create_group(f"c{cluster_count}")
        create_fit_dataset(
            group, "centers", (len(rows), cluster_count, embedding_width), "<f4"
        )
        labels = create_fit_dataset(group, "labels", (len(rows), 321), "<u2")
        labels[...] = _LABEL_SENTINEL
        create_fit_dataset(group, "cluster_inertia", (len(rows), cluster_count), "<f8")
        create_fit_dataset(
            group, "cluster_membership_hash", (len(rows), cluster_count, 32), "u1"
        )
        create_fit_dataset(group, "center_hash", (len(rows), 32), "u1")
        create_fit_dataset(group, "fit_membership_hash", (len(rows), 32), "u1")
        create_fit_dataset(group, "sklearn_inertia", (len(rows),), "<f8")
        create_fit_dataset(group, "recomputed_total_inertia", (len(rows),), "<f8")
        create_fit_dataset(group, "n_iter", (len(rows),), "<u4")
        create_fit_dataset(group, "elapsed_ns", (len(rows),), "<u8")
        assignment_indices = create_fit_dataset(
            group, "assignment_query_index", (len(rows), 2), "<i2"
        )
        assignment_indices[...] = _INDEX_SENTINEL
        assignment_clusters = create_fit_dataset(
            group, "assignment_cluster_id", (len(rows), 2), "<u2"
        )
        assignment_clusters[...] = _CLUSTER_SENTINEL
        assignment_distances = create_fit_dataset(
            group, "assignment_distance", (len(rows), 2), "<f8"
        )
        assignment_distances[...] = np.nan
        create_fit_dataset(group, "assignment_support", (len(rows), 2), "<u2")
        create_fit_dataset(group, "complete", (len(rows),), "u1")


def _validate_fit(
    fit: CanonicalClusterFit,
    *,
    metadata: Mapping[str, object],
    query_ids: tuple[str, ...],
    excluded_indices: tuple[int, ...],
    embedding_width: int,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, ...], ...], tuple[float, ...]]:
    expected_support = tuple(
        query_id
        for index, query_id in enumerate(query_ids)
        if index not in excluded_indices
    )
    if fit.key != FitKey(
        expected_support,
        cluster_count,
        str(metadata["embedding_sha256"]),
        str(metadata["clustering_contract_sha256"]),
    ):
        raise ValueError("fit key or complete support tuple does not match inventory")
    centers = np.asarray(fit.centers)
    if centers.dtype != np.dtype("<f4") or centers.shape != (
        cluster_count,
        embedding_width,
    ):
        raise ValueError("fit centers have the wrong dtype or shape")
    if not np.isfinite(centers).all():
        raise ValueError("fit centers contain a nonfinite value")
    labels = np.asarray(fit.labels)
    if labels.dtype != np.dtype("<u2") or labels.shape != (len(expected_support),):
        raise ValueError("fit labels have the wrong dtype or shape")
    if np.any(labels >= cluster_count):
        raise ValueError("fit labels contain an invalid cluster ID")
    memberships = tuple(
        tuple(
            expected_support[index]
            for index, label in enumerate(labels)
            if int(label) == cluster_id
        )
        for cluster_id in range(cluster_count)
    )
    if (
        any(not membership for membership in memberships)
        or fit.memberships != memberships
    ):
        raise ValueError("fit memberships are inconsistent with labels")
    expected_fit_hash = canonical_membership_sha256(tuple(enumerate(memberships)))
    expected_cluster_hashes = tuple(
        canonical_membership_sha256(((cluster_id, membership),))
        for cluster_id, membership in enumerate(memberships)
    )
    expected_center_hash = logical_array_sha256(
        centers,
        tag="canonical-centers",
        dtype="<f4",
        row_inventory=tuple(range(cluster_count)),
        column_inventory=(),
    )
    if fit.fit_membership_sha256 != expected_fit_hash:
        raise ValueError("fit membership hash does not match memberships")
    if fit.cluster_membership_sha256 != expected_cluster_hashes:
        raise ValueError("cluster membership hash does not match memberships")
    if fit.center_sha256 != expected_center_hash:
        raise ValueError("center hash does not match centers")
    inertia = tuple(fit.cluster_inertia)
    if len(inertia) != cluster_count or any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in inertia
    ):
        raise ValueError("cluster inertia values are invalid")
    total = math.fsum(inertia)
    if (
        type(fit.recomputed_total_inertia) is not float
        or not math.isfinite(fit.recomputed_total_inertia)
        or fit.recomputed_total_inertia != total
        or type(fit.sklearn_inertia) is not float
        or not math.isfinite(fit.sklearn_inertia)
        or not math.isclose(fit.sklearn_inertia, total, rel_tol=1e-5, abs_tol=1e-8)
    ):
        raise ValueError("fit inertia relationships are invalid")
    if type(fit.n_iter) is not int or not 1 <= fit.n_iter <= 299:
        raise ValueError("fit n_iter is invalid")
    return centers, labels, memberships, inertia


def _validate_assignments(
    assignments: Sequence[TargetAssignment],
    *,
    query_ids: tuple[str, ...],
    excluded_indices: tuple[int, ...],
    memberships: tuple[tuple[str, ...], ...],
) -> tuple[TargetAssignment, ...]:
    values = tuple(assignments)
    if len(values) != len(excluded_indices):
        raise ValueError("assignment targets do not match inventory")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    for assignment, excluded_index in zip(values, excluded_indices, strict=True):
        if (
            type(assignment) is not TargetAssignment
            or assignment.query_id != query_ids[excluded_index]
        ):
            raise ValueError("assignment targets do not match inventory")
        cluster_id = assignment.canonical_cluster_id
        if type(cluster_id) is not int or not 0 <= cluster_id < len(memberships):
            raise ValueError("assignment cluster ID is invalid")
        expected_members = tuple(
            query_index[query_id] for query_id in memberships[cluster_id]
        )
        if (
            assignment.member_indices != expected_members
            or assignment.cluster_support != len(expected_members)
        ):
            raise ValueError("assignment support does not match its cluster membership")
        if (
            type(assignment.squared_distance) is not float
            or not math.isfinite(assignment.squared_distance)
            or assignment.squared_distance < 0.0
        ):
            raise ValueError("assignment distance is invalid")
    return values


def write_fit_row(
    h5: h5py.File,
    inventory_row: int,
    fit: CanonicalClusterFit,
    assignments: Sequence[TargetAssignment],
    *,
    started_ns: int,
) -> None:
    def read_metadata() -> tuple[dict[str, object], tuple[str, ...]]:
        raw = bytes(np.asarray(h5["metadata/canonical_json"], dtype="u1"))
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("metadata canonical_json is invalid") from error
        if not isinstance(value, dict) or canonical_json(value) != raw:
            raise ValueError("metadata canonical_json is not canonical")
        return _validated_metadata(value)

    def read_inventory_row() -> tuple[int, ...]:
        count = int(h5.attrs["inventory_count"])
        if type(inventory_row) is not int or not 0 <= inventory_row < count:
            raise ValueError("inventory_row is out of range")
        kind_code = int(h5["inventory/exclusion_kind"][inventory_row])
        if kind_code not in (0, 1):
            raise ValueError("stored exclusion kind is invalid")
        length = 2 if kind_code == 0 else 1
        stored_indices = np.asarray(
            h5["inventory/excluded_query_indices"][inventory_row], dtype="<i2"
        )
        if stored_indices.shape != (2,) or (
            length == 1 and int(stored_indices[1]) != _INDEX_SENTINEL
        ):
            raise ValueError("stored excluded query indices are invalid")
        return tuple(int(value) for value in stored_indices[:length])

    cluster_count = fit.key.cluster_count
    if cluster_count not in CLUSTER_COUNTS:
        raise ValueError("fit cluster_count must be 20 or 40")
    group = h5[f"fits/c{cluster_count}"]
    excluded = read_inventory_row()
    if int(group["complete"][inventory_row]) != 0:
        raise ValueError("fit slot is already complete")
    metadata, query_ids = read_metadata()
    centers, labels, memberships, inertia = _validate_fit(
        fit,
        metadata=metadata,
        query_ids=query_ids,
        excluded_indices=excluded,
        embedding_width=int(h5.attrs["embedding_width"]),
        cluster_count=cluster_count,
    )
    assignment_values = _validate_assignments(
        assignments,
        query_ids=query_ids,
        excluded_indices=excluded,
        memberships=memberships,
    )
    if type(started_ns) is not int or started_ns < 0:
        raise ValueError("started_ns must be a nonnegative exact integer")
    if started_ns > time.perf_counter_ns():
        raise ValueError("started_ns is later than the current clock")

    group["centers"][inventory_row] = centers
    group["labels"][inventory_row, : len(labels)] = labels
    group["cluster_inertia"][inventory_row] = inertia
    group["cluster_membership_hash"][inventory_row] = np.asarray(
        [list(bytes.fromhex(value)) for value in fit.cluster_membership_sha256],
        dtype="u1",
    )
    group["center_hash"][inventory_row] = np.frombuffer(
        bytes.fromhex(fit.center_sha256), dtype="u1"
    )
    group["fit_membership_hash"][inventory_row] = np.frombuffer(
        bytes.fromhex(fit.fit_membership_sha256), dtype="u1"
    )
    group["sklearn_inertia"][inventory_row] = fit.sklearn_inertia
    group["recomputed_total_inertia"][inventory_row] = fit.recomputed_total_inertia
    group["n_iter"][inventory_row] = fit.n_iter
    for position, (excluded_index, assignment) in enumerate(
        zip(excluded, assignment_values, strict=True)
    ):
        group["assignment_query_index"][inventory_row, position] = excluded_index
        group["assignment_cluster_id"][inventory_row, position] = (
            assignment.canonical_cluster_id
        )
        group["assignment_distance"][inventory_row, position] = (
            assignment.squared_distance
        )
        group["assignment_support"][inventory_row, position] = (
            assignment.cluster_support
        )
    finished_ns = time.perf_counter_ns()
    group["elapsed_ns"][inventory_row] = finished_ns - started_ns
    group["complete"][inventory_row] = np.uint8(1)
    if all(np.all(h5[f"fits/c{count}/complete"][...] == 1) for count in CLUSTER_COUNTS):
        h5.attrs.modify("complete", np.uint8(1))


def read_fit_row(
    h5: h5py.File,
    inventory_row: int,
    cluster_count: Literal[20, 40],
) -> StoredFitRecord:
    def read_metadata() -> tuple[dict[str, object], tuple[str, ...]]:
        raw = bytes(np.asarray(h5["metadata/canonical_json"], dtype="u1"))
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("metadata canonical_json is invalid") from error
        if not isinstance(value, dict) or canonical_json(value) != raw:
            raise ValueError("metadata canonical_json is not canonical")
        return _validated_metadata(value)

    def read_inventory_row() -> tuple[Literal["pair", "single"], int, tuple[int, ...]]:
        count = int(h5.attrs["inventory_count"])
        if type(inventory_row) is not int or not 0 <= inventory_row < count:
            raise ValueError("inventory_row is out of range")
        kind_code = int(h5["inventory/exclusion_kind"][inventory_row])
        if kind_code not in (0, 1):
            raise ValueError("stored exclusion kind is invalid")
        kind: Literal["pair", "single"] = "pair" if kind_code == 0 else "single"
        length = 2 if kind == "pair" else 1
        stored_indices = np.asarray(
            h5["inventory/excluded_query_indices"][inventory_row], dtype="<i2"
        )
        if stored_indices.shape != (2,) or (
            length == 1 and int(stored_indices[1]) != _INDEX_SENTINEL
        ):
            raise ValueError("stored excluded query indices are invalid")
        return (
            kind,
            int(h5["inventory/inventory_index"][inventory_row]),
            tuple(int(value) for value in stored_indices[:length]),
        )

    if cluster_count not in CLUSTER_COUNTS:
        raise ValueError("cluster_count must be 20 or 40")
    kind, inventory_index, excluded = read_inventory_row()
    group = h5[f"fits/c{cluster_count}"]
    if int(group["complete"][inventory_row]) != 1:
        raise ValueError("fit slot is incomplete")
    metadata, query_ids = read_metadata()
    support_ids = tuple(
        query_id for index, query_id in enumerate(query_ids) if index not in excluded
    )
    centers = np.asarray(group["centers"][inventory_row], dtype="<f4")
    padded_labels = np.asarray(group["labels"][inventory_row], dtype="<u2")
    support_count = len(support_ids)
    if support_count == 320 and int(padded_labels[-1]) != _LABEL_SENTINEL:
        raise ValueError("pair label sentinel is invalid")
    labels = np.ascontiguousarray(padded_labels[:support_count], dtype="<u2")
    memberships = tuple(
        tuple(
            support_ids[index]
            for index, label in enumerate(labels)
            if int(label) == cluster_id
        )
        for cluster_id in range(cluster_count)
    )
    fit = CanonicalClusterFit(
        key=FitKey(
            support_ids,
            cluster_count,
            str(metadata["embedding_sha256"]),
            str(metadata["clustering_contract_sha256"]),
        ),
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=bytes(
            np.asarray(group["fit_membership_hash"][inventory_row], dtype="u1")
        ).hex(),
        cluster_membership_sha256=tuple(
            bytes(row).hex()
            for row in np.asarray(
                group["cluster_membership_hash"][inventory_row], dtype="u1"
            )
        ),
        center_sha256=bytes(
            np.asarray(group["center_hash"][inventory_row], dtype="u1")
        ).hex(),
        cluster_inertia=tuple(
            float(value)
            for value in np.asarray(
                group["cluster_inertia"][inventory_row], dtype="<f8"
            )
        ),
        sklearn_inertia=float(group["sklearn_inertia"][inventory_row]),
        recomputed_total_inertia=float(
            group["recomputed_total_inertia"][inventory_row]
        ),
        n_iter=int(group["n_iter"][inventory_row]),
    )
    _centers, _labels, validated_memberships, _inertia = _validate_fit(
        fit,
        metadata=metadata,
        query_ids=query_ids,
        excluded_indices=excluded,
        embedding_width=int(h5.attrs["embedding_width"]),
        cluster_count=cluster_count,
    )
    query_indices = np.asarray(
        group["assignment_query_index"][inventory_row], dtype="<i2"
    )
    cluster_ids = np.asarray(group["assignment_cluster_id"][inventory_row], dtype="<u2")
    distances = np.asarray(group["assignment_distance"][inventory_row], dtype="<f8")
    supports = np.asarray(group["assignment_support"][inventory_row], dtype="<u2")
    assignment_count = len(excluded)
    if assignment_count == 1 and (
        int(query_indices[1]) != _INDEX_SENTINEL
        or int(cluster_ids[1]) != _CLUSTER_SENTINEL
        or not np.isnan(distances[1])
        or int(supports[1]) != 0
    ):
        raise ValueError("single assignment sentinel is invalid")
    query_index_by_id = {query_id: index for index, query_id in enumerate(query_ids)}
    assignment_values = tuple(
        TargetAssignment(
            query_id=query_ids[int(query_indices[position])],
            canonical_cluster_id=int(cluster_ids[position]),
            squared_distance=float(distances[position]),
            member_indices=tuple(
                query_index_by_id[query_id]
                for query_id in validated_memberships[int(cluster_ids[position])]
            ),
            cluster_support=int(supports[position]),
        )
        for position in range(assignment_count)
    )
    assignments = _validate_assignments(
        assignment_values,
        query_ids=query_ids,
        excluded_indices=excluded,
        memberships=validated_memberships,
    )
    return StoredFitRecord(
        fit=fit,
        assignments=assignments,
        exclusion_kind=kind,
        inventory_index=inventory_index,
        excluded_query_ids=tuple(query_ids[index] for index in excluded),
        elapsed_ns=int(group["elapsed_ns"][inventory_row]),
    )


def validate_fit_file(
    path: Path,
    *,
    expected_metadata: Mapping[str, object],
    expected_inventory: Sequence[FitInventoryRow],
) -> None:
    normalized_metadata, query_ids = _validated_metadata(expected_metadata)
    inventory = _validated_inventory(expected_inventory, query_count=len(query_ids))
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_id") != FIT_FILE_SCHEMA:
            raise ValueError("unknown fit-file schema")
        if int(h5.attrs.get("complete", 0)) != 1:
            raise ValueError("file is incomplete")
        if int(h5.attrs.get("inventory_count", -1)) != len(inventory):
            raise ValueError("inventory count does not match")
        if bytes(
            np.asarray(h5["metadata/canonical_json"], dtype="u1")
        ) != canonical_json(normalized_metadata):
            raise ValueError("metadata does not match expected canonical bytes")
        for row_index, expected in enumerate(inventory):
            expected_kind_code = 0 if expected.exclusion_kind == "pair" else 1
            kind_code = int(h5["inventory/exclusion_kind"][row_index])
            inventory_index = int(h5["inventory/inventory_index"][row_index])
            stored_indices = np.asarray(
                h5["inventory/excluded_query_indices"][row_index], dtype="<i2"
            )
            expected_indices = np.full(2, _INDEX_SENTINEL, dtype="<i2")
            expected_indices[: len(expected.excluded_query_indices)] = (
                expected.excluded_query_indices
            )
            if (
                kind_code != expected_kind_code
                or inventory_index != expected.inventory_index
                or not np.array_equal(stored_indices, expected_indices)
            ):
                raise ValueError("stored inventory does not match expected inventory")
            for cluster_count in CLUSTER_COUNTS:
                read_fit_row(h5, row_index, cluster_count)
