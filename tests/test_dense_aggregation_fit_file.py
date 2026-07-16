from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import time

import h5py
import numpy as np
import pytest

import trec_mlm.dense_aggregation_fit_file as fit_file_module
from trec_mlm.dense_aggregation_clustering import (
    CanonicalClusterFit,
    FitKey,
    TargetAssignment,
)
from trec_mlm.dense_aggregation_fit_file import (
    FitInventoryRow,
    initialize_fit_datasets,
    read_fit_row,
    validate_fit_file,
    write_fit_row,
)
from trec_mlm.dense_aggregation_hashing import (
    canonical_membership_sha256,
    logical_array_sha256,
)


def _metadata(query_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "query_ids": list(query_ids),
        "query_input_sha256": "0" * 64,
        "embedding_sha256": "1" * 64,
        "similarity_sha256": "2" * 64,
        "clustering_contract_sha256": "3" * 64,
        "implementation_inventory_sha256": "4" * 64,
        "dependency_inventory_sha256": "5" * 64,
        "runtime_inventory_sha256": "6" * 64,
        "numerical_environment_sha256": "7" * 64,
        "protocol_sha256": "8" * 64,
    }


def _fit_and_assignments(
    *,
    query_ids: tuple[str, ...],
    excluded: tuple[int, ...],
    cluster_count: int,
) -> tuple[CanonicalClusterFit, tuple[TargetAssignment, ...]]:
    support_indices = tuple(
        index for index in range(len(query_ids)) if index not in excluded
    )
    support_ids = tuple(query_ids[index] for index in support_indices)
    labels = np.asarray(
        [index % cluster_count for index in range(len(support_ids))],
        dtype=np.dtype("<u2"),
    )
    centers = np.asarray(
        [[cluster_id + 0.25, cluster_id + 0.75] for cluster_id in range(cluster_count)],
        dtype=np.dtype("<f4"),
    )
    memberships = tuple(
        tuple(
            query_id
            for query_id, label in zip(support_ids, labels, strict=True)
            if int(label) == cluster_id
        )
        for cluster_id in range(cluster_count)
    )
    cluster_inertia = tuple(float(index + 1) for index in range(cluster_count))
    total_inertia = math.fsum(cluster_inertia)
    fit = CanonicalClusterFit(
        key=FitKey(support_ids, cluster_count, "1" * 64, "3" * 64),
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=canonical_membership_sha256(
            tuple(enumerate(memberships))
        ),
        cluster_membership_sha256=tuple(
            canonical_membership_sha256(((cluster_id, membership),))
            for cluster_id, membership in enumerate(memberships)
        ),
        center_sha256=logical_array_sha256(
            centers,
            tag="canonical-centers",
            dtype="<f4",
            row_inventory=tuple(range(cluster_count)),
            column_inventory=(),
        ),
        cluster_inertia=cluster_inertia,
        sklearn_inertia=total_inertia,
        recomputed_total_inertia=total_inertia,
        n_iter=7,
    )
    support_index_by_id = dict(zip(support_ids, support_indices, strict=True))
    assignments = tuple(
        TargetAssignment(
            query_id=query_ids[target_index],
            canonical_cluster_id=target_index % cluster_count,
            squared_distance=float(target_index) + 0.5,
            member_indices=tuple(
                support_index_by_id[query_id]
                for query_id in memberships[target_index % cluster_count]
            ),
            cluster_support=len(memberships[target_index % cluster_count]),
        )
        for target_index in excluded
    )
    return fit, assignments


def test_direct_hdf5_pair_and_single_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "fits.partial.h5"
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    inventory = (
        FitInventoryRow("pair", 0, (0, 1)),
        FitInventoryRow("single", 0, (0,)),
    )
    metadata = _metadata(query_ids)

    with h5py.File(path, "x") as h5:
        initialize_fit_datasets(
            h5, metadata=metadata, inventory=inventory, embedding_width=2
        )
        pair_fit, pair_assignments = _fit_and_assignments(
            query_ids=query_ids,
            excluded=(0, 1),
            cluster_count=20,
        )
        write_fit_row(
            h5, 0, pair_fit, pair_assignments, started_ns=time.perf_counter_ns()
        )
        single_fit, single_assignments = _fit_and_assignments(
            query_ids=query_ids,
            excluded=(0,),
            cluster_count=40,
        )
        write_fit_row(
            h5,
            1,
            single_fit,
            single_assignments,
            started_ns=time.perf_counter_ns(),
        )

        assert h5["fits/c20/complete"][0] == 1
        assert h5["fits/c40/complete"][1] == 1
        pair_loaded = read_fit_row(h5, 0, 20)
        single_loaded = read_fit_row(h5, 1, 40)

    assert np.array_equal(pair_loaded.fit.centers, pair_fit.centers)
    assert np.array_equal(pair_loaded.fit.labels, pair_fit.labels)
    assert pair_loaded.fit.memberships == pair_fit.memberships
    assert pair_loaded.assignments == pair_assignments
    assert np.array_equal(single_loaded.fit.centers, single_fit.centers)
    assert single_loaded.fit.memberships == single_fit.memberships
    assert single_loaded.assignments == single_assignments


def test_module_has_no_extra_hdf5_helper_surface() -> None:
    assert not hasattr(fit_file_module, "_create_fit_dataset")
    assert not hasattr(fit_file_module, "_metadata_from_file")
    assert not hasattr(fit_file_module, "_inventory_row_from_file")


def _initialized_pair_file(
    tmp_path: Path,
) -> tuple[
    h5py.File,
    tuple[str, ...],
    CanonicalClusterFit,
    tuple[TargetAssignment, ...],
]:
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    h5 = h5py.File(tmp_path / "pair.partial.h5", "x")
    initialize_fit_datasets(
        h5,
        metadata=_metadata(query_ids),
        inventory=(FitInventoryRow("pair", 0, (0, 1)),),
        embedding_width=2,
    )
    fit, assignments = _fit_and_assignments(
        query_ids=query_ids, excluded=(0, 1), cluster_count=20
    )
    return h5, query_ids, fit, assignments


def test_write_fit_row_rejects_duplicate_slot(tmp_path: Path) -> None:
    h5, _query_ids, fit, assignments = _initialized_pair_file(tmp_path)
    with h5:
        write_fit_row(h5, 0, fit, assignments, started_ns=time.perf_counter_ns())
        with pytest.raises(ValueError, match="already complete"):
            write_fit_row(h5, 0, fit, assignments, started_ns=time.perf_counter_ns())


def test_write_fit_row_rejects_wrong_support(tmp_path: Path) -> None:
    h5, _query_ids, fit, assignments = _initialized_pair_file(tmp_path)
    wrong_key = replace(fit.key, support_query_ids=fit.key.support_query_ids[:-1])
    with h5, pytest.raises(ValueError, match="support tuple"):
        write_fit_row(
            h5,
            0,
            replace(fit, key=wrong_key),
            assignments,
            started_ns=time.perf_counter_ns(),
        )


def test_write_fit_row_rejects_wrong_center_shape(tmp_path: Path) -> None:
    h5, _query_ids, fit, assignments = _initialized_pair_file(tmp_path)
    with h5, pytest.raises(ValueError, match="wrong dtype or shape"):
        write_fit_row(
            h5,
            0,
            replace(fit, centers=fit.centers[:, :1]),
            assignments,
            started_ns=time.perf_counter_ns(),
        )


def test_write_fit_row_rejects_nonfinite_center(tmp_path: Path) -> None:
    h5, _query_ids, fit, assignments = _initialized_pair_file(tmp_path)
    centers = fit.centers.copy()
    centers[0, 0] = np.nan
    with h5, pytest.raises(ValueError, match="nonfinite"):
        write_fit_row(
            h5,
            0,
            replace(fit, centers=centers),
            assignments,
            started_ns=time.perf_counter_ns(),
        )


def test_validate_fit_file_accepts_all_complete_slots(tmp_path: Path) -> None:
    path = tmp_path / "complete.h5"
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    metadata = _metadata(query_ids)
    inventory = (FitInventoryRow("pair", 17, (0, 1)),)
    with h5py.File(path, "x") as h5:
        initialize_fit_datasets(
            h5, metadata=metadata, inventory=inventory, embedding_width=2
        )
        for cluster_count in (20, 40):
            fit, assignments = _fit_and_assignments(
                query_ids=query_ids,
                excluded=(0, 1),
                cluster_count=cluster_count,
            )
            write_fit_row(
                h5,
                0,
                fit,
                assignments,
                started_ns=time.perf_counter_ns(),
            )

    validate_fit_file(path, expected_metadata=metadata, expected_inventory=inventory)
