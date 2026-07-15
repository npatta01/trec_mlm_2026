from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import dataclasses
import hashlib
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

import trec_mlm.dense_aggregation_clustering as clustering_module
from trec_mlm.dense_aggregation_artifacts import raw_file_sha256
from trec_mlm.dense_aggregation_clustering import (
    CanonicalClusterFit,
    FitKey,
    NumericalEnvironmentReceipt,
    TargetAssignment,
    assign_targets,
    capture_numerical_environment,
    fit_canonical_clusters,
    numerical_environment_file_bytes,
    require_same_numerical_environment,
)
from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256
from trec_mlm.dense_aggregation_runtime import THREAD_ENVIRONMENT


_THREADPOOL_KEYS = (
    "user_api",
    "internal_api",
    "prefix",
    "filepath",
    "version",
    "num_threads",
)
_UNSORTED_THREADPOOL_ROWS: tuple[Mapping[str, Any], ...] = (
    {
        "user_api": "openmp",
        "internal_api": "openmp",
        "prefix": "libgomp",
        "filepath": "/runtime/libgomp.so.1",
        "version": None,
        "num_threads": 1,
        "ignored_extra_field": "not part of the receipt",
    },
    {
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libscipy_openblas",
        "filepath": "/runtime/libscipy_openblas.so",
        "version": "0.3.30",
        "num_threads": 1,
        "architecture": "Zen",
    },
)
_CANONICAL_THREADPOOL_ROWS: tuple[Mapping[str, str | int | None], ...] = (
    {
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libscipy_openblas",
        "filepath": "/runtime/libscipy_openblas.so",
        "version": "0.3.30",
        "num_threads": 1,
    },
    {
        "user_api": "openmp",
        "internal_api": "openmp",
        "prefix": "libgomp",
        "filepath": "/runtime/libgomp.so.1",
        "version": None,
        "num_threads": 1,
    },
)


@pytest.fixture(autouse=True)
def _pin_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")


def _install_threadpool_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    snapshots: Sequence[Sequence[Mapping[str, Any]]],
) -> list[int]:
    remaining = iter(snapshots)
    active_limits: list[int] = []
    calls: list[int] = []

    @contextmanager
    def fake_threadpool_limits(*, limits: int) -> Iterator[None]:
        active_limits.append(limits)
        try:
            yield
        finally:
            active_limits.pop()

    def fake_threadpool_info() -> list[dict[str, Any]]:
        assert active_limits == [1]
        calls.append(1)
        return [dict(row) for row in next(remaining)]

    monkeypatch.setattr(clustering_module, "threadpool_limits", fake_threadpool_limits)
    monkeypatch.setattr(clustering_module, "threadpool_info", fake_threadpool_info)
    return calls


@pytest.mark.parametrize("name", THREAD_ENVIRONMENT)
@pytest.mark.parametrize("bad_value", [None, "01"])
def test_capture_requires_each_thread_environment_variable_to_be_exactly_one(
    name: str,
    bad_value: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_threadpool_snapshots(monkeypatch, (_UNSORTED_THREADPOOL_ROWS,))
    if bad_value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, bad_value)

    with pytest.raises(RuntimeError, match=name):
        capture_numerical_environment()


def test_capture_canonicalizes_exact_threadpool_keys_types_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_threadpool_snapshots(
        monkeypatch, (_UNSORTED_THREADPOOL_ROWS,)
    )

    receipt = capture_numerical_environment()

    assert calls == [1]
    assert tuple(receipt.thread_environment) == THREAD_ENVIRONMENT
    assert receipt.thread_environment == {
        name: "1" for name in THREAD_ENVIRONMENT
    }
    assert receipt.threadpool_info == _CANONICAL_THREADPOOL_ROWS
    assert tuple(tuple(row) for row in receipt.threadpool_info) == (
        _THREADPOOL_KEYS,
        _THREADPOOL_KEYS,
    )
    for row in receipt.threadpool_info:
        assert all(type(row[key]) is str for key in _THREADPOOL_KEYS[:4])
        assert row["version"] is None or type(row["version"]) is str
        assert type(row["num_threads"]) is int


@pytest.mark.parametrize("bad_num_threads", [0, 2, True, "1"])
def test_capture_requires_exact_integer_single_thread_threadpools(
    bad_num_threads: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = dict(_CANONICAL_THREADPOOL_ROWS[0])
    row["num_threads"] = bad_num_threads
    _install_threadpool_snapshots(monkeypatch, ((row,),))

    with pytest.raises(RuntimeError, match="num_threads"):
        capture_numerical_environment()


def test_threadpool_info_hash_is_framed_from_canonical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_threadpool_snapshots(monkeypatch, (_UNSORTED_THREADPOOL_ROWS,))

    receipt = capture_numerical_environment()

    assert receipt.threadpool_info_sha256 == framed_sha256(
        "threadpool-info", canonical_json(receipt.threadpool_info)
    )


def test_numerical_environment_file_bytes_are_exact_canonical_json_line() -> None:
    receipt = NumericalEnvironmentReceipt(
        thread_environment={name: "1" for name in THREAD_ENVIRONMENT},
        threadpool_info=_CANONICAL_THREADPOOL_ROWS,
        threadpool_info_sha256="a" * 64,
    )

    assert numerical_environment_file_bytes(receipt) == (
        canonical_json(dataclasses.asdict(receipt)) + b"\n"
    )


def test_raw_environment_file_hash_is_distinct_from_inner_threadpool_hash(
    tmp_path: Path,
) -> None:
    receipt = NumericalEnvironmentReceipt(
        thread_environment={name: "1" for name in THREAD_ENVIRONMENT},
        threadpool_info=_CANONICAL_THREADPOOL_ROWS,
        threadpool_info_sha256=framed_sha256(
            "threadpool-info", canonical_json(_CANONICAL_THREADPOOL_ROWS)
        ),
    )
    file_bytes = numerical_environment_file_bytes(receipt)
    path = tmp_path / "environment.json"
    path.write_bytes(file_bytes)

    file_sha256 = raw_file_sha256(path)
    assert file_sha256 == hashlib.sha256(file_bytes).hexdigest()
    assert file_sha256 != receipt.threadpool_info_sha256


def test_require_same_numerical_environment_requires_exact_recapture_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_threadpool_snapshots(
        monkeypatch,
        (_UNSORTED_THREADPOOL_ROWS, _UNSORTED_THREADPOOL_ROWS),
    )
    expected = capture_numerical_environment()

    require_same_numerical_environment(expected)

    assert calls == [1, 1]


def _mutated_threadpool_rows() -> tuple[Mapping[str, Any], ...]:
    changed = [dict(row) for row in _UNSORTED_THREADPOOL_ROWS]
    changed[0]["filepath"] = "/mutated/libgomp.so.1"
    return tuple(changed)


def test_worker_checkpoint_rejects_mutation_before_first_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_threadpool_snapshots(
        monkeypatch,
        (_UNSORTED_THREADPOOL_ROWS, _mutated_threadpool_rows()),
    )
    worker_start = capture_numerical_environment()
    fit_calls = 0

    with pytest.raises(RuntimeError, match="numerical environment"):
        require_same_numerical_environment(worker_start)
        fit_calls += 1

    assert fit_calls == 0


def test_worker_checkpoint_rejects_mutation_after_final_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_threadpool_snapshots(
        monkeypatch,
        (
            _UNSORTED_THREADPOOL_ROWS,
            _UNSORTED_THREADPOOL_ROWS,
            _mutated_threadpool_rows(),
        ),
    )
    worker_start = capture_numerical_environment()
    require_same_numerical_environment(worker_start)
    completed_fit_calls = 1

    with pytest.raises(RuntimeError, match="numerical environment"):
        require_same_numerical_environment(worker_start)

    assert completed_fit_calls == 1


_STATIC_KMEANS_ARGUMENTS = {
    "init": "k-means++",
    "random_state": 2026,
    "n_init": 10,
    "algorithm": "lloyd",
    "max_iter": 300,
    "tol": 1e-4,
}
_EMBEDDING_SHA256 = "e" * 64
_CLUSTERING_CONTRACT_SHA256 = "c" * 64


def test_fit_passes_dynamic_cluster_count_exact_static_arguments_and_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    active_limits: list[int] = []
    events: list[tuple[str, int]] = []

    class FakeKMeans:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs
            self.cluster_centers_ = np.asarray(
                [[20.0, 0.0], [30.0, 0.0], [10.0, 0.0]],
                dtype=np.float64,
            )
            self.inertia_ = 0.0
            self.n_iter_ = 1

        def fit_predict(self, rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            assert active_limits == [1]
            events.append(("fit", active_limits[0]))
            captured["rows"] = rows.copy()
            return np.asarray([2, 0, 1], dtype=np.int64)

    @contextmanager
    def fake_threadpool_limits(*, limits: int) -> Iterator[None]:
        events.append(("enter", limits))
        active_limits.append(limits)
        try:
            yield
        finally:
            active_limits.pop()
            events.append(("exit", limits))

    monkeypatch.setattr(clustering_module, "KMeans", FakeKMeans)
    monkeypatch.setattr(
        clustering_module, "threadpool_limits", fake_threadpool_limits
    )
    embeddings = np.asarray(
        [[30.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32
    )

    fit_canonical_clusters(
        embeddings=embeddings,
        query_ids=("q-c", "q-a", "q-b"),
        support_indices=(0, 2, 1),
        cluster_count=3,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )

    assert type(clustering_module._KMEANS_ARGUMENTS) is dict
    assert clustering_module._KMEANS_ARGUMENTS == _STATIC_KMEANS_ARGUMENTS
    assert captured["kwargs"] == {
        "n_clusters": 3,
        **_STATIC_KMEANS_ARGUMENTS,
    }
    assert events == [("enter", 1), ("fit", 1), ("exit", 1)]
    rows = captured["rows"]
    assert isinstance(rows, np.ndarray)
    assert rows.dtype == np.dtype("<f4")
    assert rows.flags.c_contiguous
    np.testing.assert_array_equal(
        rows,
        np.asarray([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype="<f4"),
    )


def test_pinned_kmeans_fit_has_canonical_golden_outputs() -> None:
    query_ids = ("q30", "q10", "q21", "q11", "q20", "q31")
    embeddings = np.asarray(
        [
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 2.0],
            [-8.0, 0.0],
            [0.0, 4.0],
            [12.0, 0.0],
        ],
        dtype=np.float32,
    )

    fit = fit_canonical_clusters(
        embeddings=embeddings,
        query_ids=query_ids,
        support_indices=(5, 3, 1, 4, 0, 2),
        cluster_count=3,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )

    assert fit.key == FitKey(
        support_query_ids=("q10", "q11", "q20", "q21", "q30", "q31"),
        cluster_count=3,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )
    assert fit.labels.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(
        fit.labels, np.asarray([0, 0, 1, 1, 2, 2], dtype="<u2")
    )
    assert fit.memberships == (
        ("q10", "q11"),
        ("q20", "q21"),
        ("q30", "q31"),
    )
    assert fit.centers.dtype == np.dtype("<f4")
    assert fit.centers.flags.c_contiguous
    np.testing.assert_array_equal(
        fit.centers,
        np.asarray([[-9.0, 0.0], [0.0, 3.0], [11.0, 0.0]], dtype="<f4"),
    )
    assert fit.fit_membership_sha256 == (
        "4c6f252c587b9a2e0801a36b36d9d2e3a88d8f96f5c881403197505509448853"
    )
    assert fit.cluster_membership_sha256 == (
        "b82820fc7dbda8e3840e37f11e8187587aacba55a882f074b2cf9e1a3fd3ff17",
        "63204eb943c5a18c2f4305f10f9193ce76c151f4f48457125fa9b66c45e1f12b",
        "e567341c0933bb2ae610311635085b3868024d396734f4cebf6a8a9bfb071c1a",
    )
    assert fit.center_sha256 == (
        "fdd9b44f18a6462798cf57addcb513ecb68f53ad0c6f81e0251e41d011de20ed"
    )
    assert fit.cluster_inertia == (2.0, 2.0, 2.0)
    assert all(type(value) is float for value in fit.cluster_inertia)
    assert fit.recomputed_total_inertia.hex() == math.fsum(
        fit.cluster_inertia
    ).hex()
    assert fit.recomputed_total_inertia.hex() == float(6.0).hex()
    assert type(fit.sklearn_inertia) is float
    assert fit.sklearn_inertia.hex() == float(6.0).hex()
    assert type(fit.n_iter) is int
    assert fit.n_iter == 2


def test_exact_target_distance_tie_uses_lower_canonical_id() -> None:
    fit = CanonicalClusterFit(
        key=FitKey(("q1", "q2"), 2, "e" * 64, "c" * 64),
        centers=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype="<f4"),
        labels=np.asarray([0, 1], dtype="<u2"),
        memberships=(("q1",), ("q2",)),
        fit_membership_sha256="f" * 64,
        cluster_membership_sha256=("1" * 64, "2" * 64),
        center_sha256="3" * 64,
        cluster_inertia=(0.0, 0.0),
        sklearn_inertia=0.0,
        recomputed_total_inertia=0.0,
        n_iter=1,
    )
    embeddings = np.asarray([[-1, 0], [1, 0], [0, 0]], dtype=np.float32)

    (assignment,) = assign_targets(
        fit=fit,
        embeddings=embeddings,
        query_ids=("q1", "q2", "qt"),
        target_indices=(2,),
        support_index_by_id={"q1": 0, "q2": 1},
    )

    assert assignment.query_id == "qt"
    assert assignment.canonical_cluster_id == 0
    assert assignment.squared_distance.hex() == float(1.0).hex()
    assert assignment.member_indices == (0,)
    assert assignment.cluster_support == 1


def test_target_squared_distance_uses_float64_accumulation() -> None:
    fit = CanonicalClusterFit(
        key=FitKey(("q-support",), 1, "e" * 64, "c" * 64),
        centers=np.asarray([[0.1, 0.2]], dtype="<f4"),
        labels=np.asarray([0], dtype="<u2"),
        memberships=(("q-support",),),
        fit_membership_sha256="f" * 64,
        cluster_membership_sha256=("1" * 64,),
        center_sha256="2" * 64,
        cluster_inertia=(0.0,),
        sklearn_inertia=0.0,
        recomputed_total_inertia=0.0,
        n_iter=1,
    )
    embeddings = np.asarray([[0.1, 0.2], [0.0, 0.0]], dtype=np.float32)

    (assignment,) = assign_targets(
        fit=fit,
        embeddings=embeddings,
        query_ids=("q-support", "q-target"),
        target_indices=(1,),
        support_index_by_id={"q-support": 0},
    )

    assert assignment.query_id == "q-target"
    assert assignment.canonical_cluster_id == 0
    assert assignment.squared_distance.hex() == "0x1.99999a6666668p-5"
    float32_delta = fit.centers[0] - embeddings[1]
    assert float(
        np.sum(float32_delta * float32_delta, dtype=np.float32)
    ).hex() == "0x1.99999c0000000p-5"
    assert assignment.member_indices == (0,)
    assert assignment.cluster_support == 1


def test_pinned_fit_is_invariant_to_query_and_support_input_permutations() -> None:
    query_ids = ("q30", "q10", "q21", "q11", "q20", "q31")
    embeddings = np.asarray(
        [
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 2.0],
            [-8.0, 0.0],
            [0.0, 4.0],
            [12.0, 0.0],
        ],
        dtype=np.float32,
    )
    baseline = fit_canonical_clusters(
        embeddings=embeddings,
        query_ids=query_ids,
        support_indices=(5, 3, 1, 4, 0, 2),
        cluster_count=3,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )
    query_permutation = np.asarray([4, 0, 5, 2, 1, 3], dtype=np.int64)
    permuted = fit_canonical_clusters(
        embeddings=np.ascontiguousarray(embeddings[query_permutation]),
        query_ids=tuple(query_ids[index] for index in query_permutation),
        support_indices=(3, 0, 5, 2, 4, 1),
        cluster_count=3,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )

    assert baseline.key == permuted.key
    assert baseline.key.support_query_ids == (
        "q10",
        "q11",
        "q20",
        "q21",
        "q30",
        "q31",
    )
    np.testing.assert_array_equal(baseline.centers, permuted.centers)
    assert baseline.memberships == permuted.memberships
    assert baseline.fit_membership_sha256 == permuted.fit_membership_sha256
    assert baseline.cluster_membership_sha256 == (
        permuted.cluster_membership_sha256
    )
    assert baseline.center_sha256 == permuted.center_sha256
    assert baseline.cluster_inertia == permuted.cluster_inertia
    assert baseline.sklearn_inertia.hex() == permuted.sklearn_inertia.hex()
    assert baseline.recomputed_total_inertia.hex() == (
        permuted.recomputed_total_inertia.hex()
    )
    assert baseline.n_iter == permuted.n_iter
    expected_canonical_labels = np.asarray([0, 0, 1, 1, 2, 2], dtype="<u2")
    np.testing.assert_array_equal(baseline.labels, expected_canonical_labels)
    np.testing.assert_array_equal(permuted.labels, expected_canonical_labels)


def test_assign_targets_rejects_nonfinite_target_and_distance() -> None:
    fit = CanonicalClusterFit(
        key=FitKey(("q-support",), 1, "e" * 64, "c" * 64),
        centers=np.asarray([[1.0, 0.0]], dtype="<f4"),
        labels=np.asarray([0], dtype="<u2"),
        memberships=(("q-support",),),
        fit_membership_sha256="f" * 64,
        cluster_membership_sha256=("1" * 64,),
        center_sha256="2" * 64,
        cluster_inertia=(0.0,),
        sklearn_inertia=0.0,
        recomputed_total_inertia=0.0,
        n_iter=1,
    )
    query_ids = ("q-support", "q-target")
    support_index_by_id = {"q-support": 0}

    with pytest.raises(ValueError, match="target embedding contains a non-finite"):
        assign_targets(
            fit=fit,
            embeddings=np.asarray([[1.0, 0.0], [np.nan, 0.0]], dtype=np.float32),
            query_ids=query_ids,
            target_indices=(1,),
            support_index_by_id=support_index_by_id,
        )

    nonfinite_center_fit = dataclasses.replace(
        fit, centers=np.asarray([[np.inf, 0.0]], dtype="<f4")
    )
    with pytest.raises(RuntimeError, match="target-to-center distance is non-finite"):
        assign_targets(
            fit=nonfinite_center_fit,
            embeddings=np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            query_ids=query_ids,
            target_indices=(1,),
            support_index_by_id=support_index_by_id,
        )


@pytest.mark.parametrize(
    ("embeddings", "query_ids", "support_indices", "cluster_count", "message"),
    [
        pytest.param(
            np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            ("q1", "q2"),
            (0, 1),
            3,
            "cluster support has fewer rows than C",
            id="fewer-rows-than-clusters",
        ),
        pytest.param(
            np.asarray([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
            ("q1", "q2"),
            (0, 1),
            1,
            "cluster support contains a non-finite or zero row",
            id="nonfinite-row",
        ),
        pytest.param(
            np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            ("q1", "q2"),
            (0, 1),
            1,
            "cluster support contains a non-finite or zero row",
            id="zero-row",
        ),
    ],
)
def test_fit_rejects_invalid_support_before_constructing_kmeans(
    embeddings: np.ndarray[Any, Any],
    query_ids: tuple[str, ...],
    support_indices: tuple[int, ...],
    cluster_count: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedKMeans:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("KMeans must not be constructed for invalid support")

    monkeypatch.setattr(clustering_module, "KMeans", UnexpectedKMeans)

    with pytest.raises(ValueError, match=message):
        fit_canonical_clusters(
            embeddings=embeddings,
            query_ids=query_ids,
            support_indices=support_indices,
            cluster_count=cluster_count,
            embedding_sha256=_EMBEDDING_SHA256,
            clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
        )


def test_fit_counts_distinct_rows_after_float32_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0 + 2.0**-30, 0.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    assert embeddings[0].tobytes() != embeddings[1].tobytes()
    converted = np.ascontiguousarray(embeddings, dtype=np.float32)
    assert converted[0].tobytes() == converted[1].tobytes()

    class UnexpectedKMeans:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("KMeans must not be constructed for invalid support")

    monkeypatch.setattr(clustering_module, "KMeans", UnexpectedKMeans)

    with pytest.raises(
        ValueError, match="cluster support has fewer than C distinct float32 rows"
    ):
        fit_canonical_clusters(
            embeddings=embeddings,
            query_ids=("q1", "q2", "q3"),
            support_indices=(0, 1, 2),
            cluster_count=3,
            embedding_sha256=_EMBEDDING_SHA256,
            clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
        )


def test_fit_rejects_sklearn_convergence_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WarningKMeans:
        def __init__(self, **_kwargs: object) -> None:
            self.cluster_centers_ = np.asarray([[2.0, 0.0]], dtype=np.float64)
            self.inertia_ = 2.0
            self.n_iter_ = 1

        def fit_predict(self, _rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            warnings.warn(
                "synthetic convergence warning",
                ConvergenceWarning,
                stacklevel=1,
            )
            return np.asarray([0, 0], dtype=np.int64)

    monkeypatch.setattr(clustering_module, "KMeans", WarningKMeans)

    with pytest.raises(RuntimeError, match="K-means emitted a convergence warning"):
        fit_canonical_clusters(
            embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            query_ids=("q1", "q2"),
            support_indices=(0, 1),
            cluster_count=1,
            embedding_sha256=_EMBEDDING_SHA256,
            clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
        )


def _fit_with_fake_estimator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings: np.ndarray[Any, Any],
    cluster_count: int,
    labels: Sequence[int],
    centers: Sequence[Sequence[float]],
    inertia: float,
    n_iter: int,
) -> CanonicalClusterFit:
    class FakeKMeans:
        def __init__(self, **_kwargs: object) -> None:
            self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
            self.inertia_ = inertia
            self.n_iter_ = n_iter

        def fit_predict(self, rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            assert rows.shape[0] == len(labels)
            return np.asarray(labels, dtype=np.int64)

    monkeypatch.setattr(clustering_module, "KMeans", FakeKMeans)
    query_ids = tuple(f"q{index}" for index in range(embeddings.shape[0]))
    return fit_canonical_clusters(
        embeddings=embeddings,
        query_ids=query_ids,
        support_indices=tuple(reversed(range(embeddings.shape[0]))),
        cluster_count=cluster_count,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )


def test_fit_rejects_unoccupied_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="K-means did not occupy every cluster"):
        _fit_with_fake_estimator(
            monkeypatch,
            embeddings=np.asarray(
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
            ),
            cluster_count=3,
            labels=(0, 0, 1),
            centers=((1.5, 0.0), (3.0, 0.0), (4.0, 0.0)),
            inertia=0.5,
            n_iter=1,
        )


def test_fit_rejects_nonfinite_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="K-means produced a non-finite center"):
        _fit_with_fake_estimator(
            monkeypatch,
            embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            cluster_count=1,
            labels=(0, 0),
            centers=((np.nan, 0.0),),
            inertia=2.0,
            n_iter=1,
        )


@pytest.mark.parametrize("n_iter", [0, 300])
def test_fit_rejects_n_iter_outside_closed_range_1_to_299(
    n_iter: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match=rf"n_iter_ outside 1\.\.299: {n_iter}"):
        _fit_with_fake_estimator(
            monkeypatch,
            embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            cluster_count=1,
            labels=(0, 0),
            centers=((2.0, 0.0),),
            inertia=2.0,
            n_iter=n_iter,
        )


def test_fit_accepts_n_iter_299(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit_with_fake_estimator(
        monkeypatch,
        embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        cluster_count=1,
        labels=(0, 0),
        centers=((2.0, 0.0),),
        inertia=2.0,
        n_iter=299,
    )

    assert fit.n_iter == 299


def test_fit_rejects_sklearn_inertia_beyond_required_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sklearn_inertia = 2.00003
    assert not math.isclose(
        sklearn_inertia, 2.0, rel_tol=1e-5, abs_tol=1e-8
    )

    with pytest.raises(
        RuntimeError, match="sklearn inertia differs from canonical recomputation"
    ):
        _fit_with_fake_estimator(
            monkeypatch,
            embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            cluster_count=1,
            labels=(0, 0),
            centers=((2.0, 0.0),),
            inertia=sklearn_inertia,
            n_iter=1,
        )


def test_fit_accepts_sklearn_inertia_within_required_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sklearn_inertia = 2.000019
    assert math.isclose(sklearn_inertia, 2.0, rel_tol=1e-5, abs_tol=1e-8)

    fit = _fit_with_fake_estimator(
        monkeypatch,
        embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        cluster_count=1,
        labels=(0, 0),
        centers=((2.0, 0.0),),
        inertia=sklearn_inertia,
        n_iter=1,
    )

    assert fit.sklearn_inertia.hex() == sklearn_inertia.hex()
    assert fit.recomputed_total_inertia.hex() == float(2.0).hex()


def _install_boundary_kmeans(monkeypatch: pytest.MonkeyPatch) -> None:
    class BoundaryKMeans:
        def __init__(self, **_kwargs: object) -> None:
            self.n_iter_ = 1

        def fit_predict(self, rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            flattened = np.asarray(rows, dtype=np.float64).reshape(rows.shape[0], -1)
            center = np.mean(flattened, axis=0, dtype=np.float64)
            delta = flattened - center
            self.cluster_centers_ = np.asarray([center], dtype=np.float64)
            self.inertia_ = float(np.sum(delta * delta, dtype=np.float64))
            return np.zeros(rows.shape[0], dtype=np.int64)

    monkeypatch.setattr(clustering_module, "KMeans", BoundaryKMeans)


def _fit_at_public_boundary(
    *,
    embeddings: np.ndarray[Any, Any],
    query_ids: Sequence[str],
    support_indices: Sequence[int],
    cluster_count: object = 1,
) -> CanonicalClusterFit:
    return fit_canonical_clusters(
        embeddings=embeddings,
        query_ids=query_ids,
        support_indices=support_indices,
        cluster_count=cluster_count,  # type: ignore[arg-type]
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )


@pytest.mark.parametrize(
    "embeddings",
    [
        pytest.param(np.asarray([1.0, 2.0], dtype=np.float32), id="one-dimensional"),
        pytest.param(
            np.asarray([[[1.0, 0.0]], [[3.0, 0.0]]], dtype=np.float32),
            id="three-dimensional",
        ),
    ],
)
def test_fit_requires_embeddings_to_be_exactly_two_dimensional(
    embeddings: np.ndarray[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_boundary_kmeans(monkeypatch)

    with pytest.raises(
        ValueError, match="embeddings must be exactly two-dimensional"
    ):
        _fit_at_public_boundary(
            embeddings=embeddings,
            query_ids=("q1", "q2"),
            support_indices=(0,),
        )


@pytest.mark.parametrize(
    ("query_ids", "support_indices", "message"),
    [
        pytest.param(
            ("q1",),
            (0,),
            "query_ids length must match embedding rows",
            id="too-short",
        ),
        pytest.param(
            ("q1", "q2", "q3"),
            (0, 1),
            "query_ids length must match embedding rows",
            id="too-long",
        ),
        pytest.param(
            ("q1", ""),
            (0,),
            "query_ids must contain nonempty unique IDs",
            id="empty-id",
        ),
        pytest.param(
            ("q1", "q1"),
            (0, 1),
            "query_ids must contain nonempty unique IDs",
            id="duplicate-id",
        ),
    ],
)
def test_fit_requires_complete_nonempty_unique_query_inventory(
    query_ids: tuple[str, ...],
    support_indices: tuple[int, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_boundary_kmeans(monkeypatch)

    with pytest.raises(ValueError, match=message):
        _fit_at_public_boundary(
            embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            query_ids=query_ids,
            support_indices=support_indices,
        )


@pytest.mark.parametrize(
    "cluster_count",
    [
        pytest.param(0, id="zero"),
        pytest.param(65_537, id="above-uint16-capacity"),
        pytest.param(True, id="bool"),
    ],
)
def test_fit_requires_exact_bounded_cluster_count(
    cluster_count: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_boundary_kmeans(monkeypatch)

    with pytest.raises(
        ValueError,
        match=r"cluster_count must be an exact int in 1\.\.65536",
    ):
        _fit_at_public_boundary(
            embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            query_ids=("q1", "q2"),
            support_indices=(0, 1),
            cluster_count=cluster_count,
        )


@pytest.mark.parametrize(
    ("support_indices", "message"),
    [
        pytest.param((-1,), "support index out of range", id="negative"),
        pytest.param((2,), "support index out of range", id="out-of-range"),
        pytest.param((True,), "support indices must be exact ints", id="bool"),
        pytest.param((0, 0), "support indices must be unique", id="duplicate"),
    ],
)
def test_fit_validates_support_indices_before_indexing(
    support_indices: tuple[int, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_boundary_kmeans(monkeypatch)

    with pytest.raises(ValueError, match=message):
        _fit_at_public_boundary(
            embeddings=np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            query_ids=("q1", "q2"),
            support_indices=support_indices,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _AssignmentBoundaryCase:
    fit: CanonicalClusterFit
    embeddings: np.ndarray[Any, Any]
    query_ids: tuple[str, ...]
    target_indices: tuple[int, ...]
    support_index_by_id: Mapping[str, int]


def _assignment_boundary_case() -> _AssignmentBoundaryCase:
    return _AssignmentBoundaryCase(
        fit=CanonicalClusterFit(
            key=FitKey(("q1", "q2", "q3"), 2, "e" * 64, "c" * 64),
            centers=np.asarray([[-1.5, 0.0], [2.0, 0.0]], dtype="<f4"),
            labels=np.asarray([0, 0, 1], dtype="<u2"),
            memberships=(("q1", "q2"), ("q3",)),
            fit_membership_sha256="f" * 64,
            cluster_membership_sha256=("1" * 64, "2" * 64),
            center_sha256="3" * 64,
            cluster_inertia=(0.5, 0.0),
            sklearn_inertia=0.5,
            recomputed_total_inertia=0.5,
            n_iter=1,
        ),
        embeddings=np.asarray(
            [[-2.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
        query_ids=("q1", "q2", "q3", "qt"),
        target_indices=(3,),
        support_index_by_id={"q1": 0, "q2": 1, "q3": 2},
    )


def _assign_boundary_case(
    case: _AssignmentBoundaryCase,
    *,
    fit: CanonicalClusterFit | None = None,
    embeddings: np.ndarray[Any, Any] | None = None,
    query_ids: Sequence[str] | None = None,
    target_indices: Sequence[int] | None = None,
    support_index_by_id: Mapping[str, object] | None = None,
) -> tuple[TargetAssignment, ...]:
    return assign_targets(
        fit=case.fit if fit is None else fit,
        embeddings=case.embeddings if embeddings is None else embeddings,
        query_ids=case.query_ids if query_ids is None else query_ids,
        target_indices=(
            case.target_indices if target_indices is None else target_indices
        ),
        support_index_by_id=(
            case.support_index_by_id
            if support_index_by_id is None
            else support_index_by_id  # type: ignore[arg-type]
        ),
    )


def test_assignment_boundary_case_is_valid() -> None:
    case = _assignment_boundary_case()

    assert _assign_boundary_case(case) == (
        TargetAssignment(
            query_id="qt",
            canonical_cluster_id=0,
            squared_distance=2.25,
            member_indices=(0, 1),
            cluster_support=2,
        ),
    )


@pytest.mark.parametrize(
    "embeddings",
    [
        pytest.param(
            np.asarray([-2.0, -1.0, 2.0, 0.0], dtype=np.float32),
            id="one-dimensional",
        ),
        pytest.param(
            np.asarray(
                [[[-2.0, 0.0]], [[-1.0, 0.0]], [[2.0, 0.0]], [[0.0, 0.0]]],
                dtype=np.float32,
            ),
            id="three-dimensional",
        ),
    ],
)
def test_assignment_boundary_requires_two_dimensional_embeddings(
    embeddings: np.ndarray[Any, Any],
) -> None:
    case = _assignment_boundary_case()

    with pytest.raises(
        ValueError, match="embeddings must be exactly two-dimensional"
    ):
        _assign_boundary_case(case, embeddings=embeddings)


@pytest.mark.parametrize(
    ("query_ids", "target_indices", "message"),
    [
        pytest.param(
            ("q1", "q2", "q3"),
            (2,),
            "query_ids length must match embedding rows",
            id="too-short",
        ),
        pytest.param(
            ("q1", "q2", "q3", "qt", "qx"),
            (3,),
            "query_ids length must match embedding rows",
            id="too-long",
        ),
        pytest.param(
            ("q1", "q2", "q3", ""),
            (3,),
            "query_ids must contain nonempty unique IDs",
            id="empty-id",
        ),
        pytest.param(
            ("q1", "q2", "q3", "q1"),
            (3,),
            "query_ids must contain nonempty unique IDs",
            id="duplicate-id",
        ),
    ],
)
def test_assignment_boundary_requires_complete_unique_query_inventory(
    query_ids: tuple[str, ...],
    target_indices: tuple[int, ...],
    message: str,
) -> None:
    case = _assignment_boundary_case()

    with pytest.raises(ValueError, match=message):
        _assign_boundary_case(
            case,
            query_ids=query_ids,
            target_indices=target_indices,
        )


@pytest.mark.parametrize(
    "centers",
    [
        pytest.param(
            np.asarray([[-1.5, 0.0]], dtype="<f4"),
            id="wrong-cluster-count",
        ),
        pytest.param(
            np.asarray([[-1.5], [2.0]], dtype="<f4"),
            id="wrong-embedding-width",
        ),
    ],
)
def test_assignment_boundary_requires_exact_center_shape(
    centers: np.ndarray[Any, Any],
) -> None:
    case = _assignment_boundary_case()
    fit = dataclasses.replace(case.fit, centers=centers)

    with pytest.raises(ValueError, match=r"fit centers must have shape \(2, 2\)"):
        _assign_boundary_case(case, fit=fit)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        pytest.param(
            np.asarray([[0, 0, 1]], dtype="<u2"),
            "fit labels must be a one-dimensional vector of support length 3",
            id="two-dimensional",
        ),
        pytest.param(
            np.asarray([0, 1], dtype="<u2"),
            "fit labels must be a one-dimensional vector of support length 3",
            id="wrong-length",
        ),
        pytest.param(
            np.asarray([0, -1, 1], dtype=np.int64),
            "fit labels contain invalid canonical cluster IDs",
            id="negative-cluster-id",
        ),
        pytest.param(
            np.asarray([0, 0, 2], dtype="<u2"),
            "fit labels contain invalid canonical cluster IDs",
            id="cluster-id-equals-C",
        ),
    ],
)
def test_assignment_boundary_requires_valid_canonical_labels(
    labels: np.ndarray[Any, Any],
    message: str,
) -> None:
    case = _assignment_boundary_case()
    fit = dataclasses.replace(case.fit, labels=labels)

    with pytest.raises(ValueError, match=message):
        _assign_boundary_case(case, fit=fit)


@pytest.mark.parametrize(
    ("memberships", "message"),
    [
        pytest.param(
            (("q1", "q2", "q3"),),
            "fit memberships count must equal cluster_count",
            id="wrong-cluster-count",
        ),
        pytest.param(
            (("q1", "q2", "q3"), ()),
            "fit memberships must not contain an empty cluster",
            id="empty-cluster",
        ),
        pytest.param(
            (("q1", "q2"), ("q2", "q3")),
            "fit memberships must partition fit support IDs",
            id="duplicate-member",
        ),
        pytest.param(
            (("q1",), ("q3",)),
            "fit memberships must partition fit support IDs",
            id="missing-member",
        ),
        pytest.param(
            (("q1", "q2"), ("q3", "qx")),
            "fit memberships must partition fit support IDs",
            id="extra-member",
        ),
        pytest.param(
            (("q1", "q3"), ("q2",)),
            "fit memberships are inconsistent with labels",
            id="inconsistent-with-labels",
        ),
    ],
)
def test_assignment_boundary_requires_valid_membership_partition(
    memberships: tuple[tuple[str, ...], ...],
    message: str,
) -> None:
    case = _assignment_boundary_case()
    fit = dataclasses.replace(case.fit, memberships=memberships)

    with pytest.raises(ValueError, match=message):
        _assign_boundary_case(case, fit=fit)


@pytest.mark.parametrize(
    ("support_index_by_id", "message"),
    [
        pytest.param(
            {"q1": 0, "q2": 1},
            "support_index_by_id keys must exactly match fit support IDs",
            id="missing-key",
        ),
        pytest.param(
            {"q1": 0, "q2": 1, "q3": 2, "qx": 3},
            "support_index_by_id keys must exactly match fit support IDs",
            id="extra-key",
        ),
        pytest.param(
            {"q1": False, "q2": 1, "q3": 2},
            "support_index_by_id values must be exact ints",
            id="bool-value",
        ),
        pytest.param(
            {"q1": "0", "q2": 1, "q3": 2},
            "support_index_by_id values must be exact ints",
            id="string-value",
        ),
        pytest.param(
            {"q1": 0, "q2": 0, "q3": 2},
            "support_index_by_id values must be unique",
            id="duplicate-value",
        ),
        pytest.param(
            {"q1": -1, "q2": 1, "q3": 2},
            "support_index_by_id index out of range",
            id="negative-value",
        ),
        pytest.param(
            {"q1": 4, "q2": 1, "q3": 2},
            "support_index_by_id index out of range",
            id="out-of-range-value",
        ),
        pytest.param(
            {"q1": 1, "q2": 0, "q3": 2},
            "support_index_by_id must map each support ID to the matching query",
            id="swapped-values",
        ),
    ],
)
def test_assignment_boundary_requires_valid_support_index_mapping(
    support_index_by_id: Mapping[str, object],
    message: str,
) -> None:
    case = _assignment_boundary_case()

    with pytest.raises(ValueError, match=message):
        _assign_boundary_case(
            case,
            support_index_by_id=support_index_by_id,
        )


@pytest.mark.parametrize(
    ("target_indices", "message"),
    [
        pytest.param(
            (True,),
            "target indices must be exact ints",
            id="bool-index",
        ),
        pytest.param(
            ("3",),
            "target indices must be exact ints",
            id="string-index",
        ),
        pytest.param(
            (3, 3),
            "target indices must be unique",
            id="duplicate-index",
        ),
        pytest.param(
            (-1,),
            "target index out of range",
            id="negative-index",
        ),
        pytest.param(
            (4,),
            "target index out of range",
            id="out-of-range-index",
        ),
        pytest.param(
            (0,),
            "target query IDs must be disjoint from fit support IDs",
            id="support-query-as-target",
        ),
    ],
)
def test_assignment_boundary_requires_valid_target_indices(
    target_indices: tuple[Any, ...],
    message: str,
) -> None:
    case = _assignment_boundary_case()

    with pytest.raises(ValueError, match=message):
        _assign_boundary_case(
            case,
            target_indices=target_indices,  # type: ignore[arg-type]
        )


def _fit_with_raw_estimator_output(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_labels: object,
    raw_centers: object,
    sklearn_inertia: float = 4.0,
) -> CanonicalClusterFit:
    class RawKMeans:
        def __init__(self, **_kwargs: object) -> None:
            self.cluster_centers_ = raw_centers
            self.inertia_ = sklearn_inertia
            self.n_iter_ = 1

        def fit_predict(self, _rows: np.ndarray[Any, Any]) -> object:
            return raw_labels

    monkeypatch.setattr(clustering_module, "KMeans", RawKMeans)
    return fit_canonical_clusters(
        embeddings=np.asarray(
            [[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        ),
        query_ids=("q0", "q1", "q2", "q3"),
        support_indices=(3, 1, 2, 0),
        cluster_count=2,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )


@pytest.mark.parametrize(
    ("raw_labels", "sklearn_inertia"),
    [
        pytest.param(
            np.asarray([[0, 0, 1, 1]], dtype=np.int64),
            4.0,
            id="two-dimensional",
        ),
        pytest.param(
            np.asarray([0, 0, 1], dtype=np.int64),
            3.0,
            id="wrong-length",
        ),
        pytest.param(
            np.asarray([0.0, 0.5, 1.0, 1.0], dtype=np.float64),
            4.0,
            id="fractional-float",
        ),
        pytest.param(
            np.asarray([False, False, True, True], dtype=np.bool_),
            4.0,
            id="bool",
        ),
    ],
)
def test_raw_estimator_labels_require_exact_integer_vector(
    raw_labels: object,
    sklearn_inertia: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "K-means labels must be a one-dimensional integer vector "
            "of support length 4"
        ),
    ):
        _fit_with_raw_estimator_output(
            monkeypatch,
            raw_labels=raw_labels,
            raw_centers=np.asarray([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float64),
            sklearn_inertia=sklearn_inertia,
        )


@pytest.mark.parametrize(
    "raw_labels",
    [
        pytest.param(
            np.asarray([0, 0, 1, -1], dtype=np.int64),
            id="negative",
        ),
        pytest.param(
            np.asarray([0, 0, 1, 2], dtype=np.int64),
            id="equal-to-cluster-count",
        ),
    ],
)
def test_raw_estimator_label_set_is_exact_cluster_range(
    raw_labels: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"K-means labels must be exactly range\(cluster_count\)",
    ):
        _fit_with_raw_estimator_output(
            monkeypatch,
            raw_labels=raw_labels,
            raw_centers=np.asarray([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("raw_centers", "sklearn_inertia"),
    [
        pytest.param(
            np.asarray(
                [[-2.0, 0.0], [2.0, 0.0], [100.0, 100.0]],
                dtype=np.float64,
            ),
            4.0,
            id="extra-cluster-row",
        ),
        pytest.param(
            np.asarray([[-2.0, 0.0]], dtype=np.float64),
            4.0,
            id="missing-cluster-row",
        ),
        pytest.param(
            np.asarray([[-2.0], [2.0]], dtype=np.float64),
            20.0,
            id="broadcasting-width",
        ),
        pytest.param(
            np.asarray([-2.0, 2.0], dtype=np.float64),
            20.0,
            id="one-dimensional",
        ),
    ],
)
def test_raw_estimator_centers_require_exact_matrix_shape(
    raw_centers: object,
    sklearn_inertia: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"K-means cluster_centers_ must have shape \(2, 2\)",
    ):
        _fit_with_raw_estimator_output(
            monkeypatch,
            raw_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            raw_centers=raw_centers,
            sklearn_inertia=sklearn_inertia,
        )


def test_characterization_cluster_inertia_uses_math_fsum_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_total = float.fromhex("0x1.1c37937e08001p+53")
    fit = _fit_with_fake_estimator(
        monkeypatch,
        embeddings=np.asarray(
            [[1.0e8, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
        cluster_count=3,
        labels=(0, 1, 2),
        centers=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        inertia=required_total,
        n_iter=1,
    )

    assert tuple(value.hex() for value in fit.cluster_inertia) == (
        "0x1.1c37937e08000p+53",
        "0x1.0000000000000p+0",
        "0x1.0000000000000p+0",
    )
    naive_numpy_total = float(
        np.sum(np.asarray(fit.cluster_inertia, dtype=np.float64), dtype=np.float64)
    )
    assert naive_numpy_total.hex() == "0x1.1c37937e08000p+53"
    assert math.fsum(fit.cluster_inertia).hex() == "0x1.1c37937e08001p+53"
    assert naive_numpy_total.hex() != math.fsum(fit.cluster_inertia).hex()
    assert fit.recomputed_total_inertia.hex() == "0x1.1c37937e08001p+53"
    assert fit.sklearn_inertia.hex() == "0x1.1c37937e08001p+53"


def test_characterization_target_assignment_is_semantically_permutation_invariant(
) -> None:
    query_ids = ("q30", "q10", "q21", "q11", "q20", "q31", "qt")
    embeddings = np.asarray(
        [
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 2.0],
            [-8.0, 0.0],
            [0.0, 4.0],
            [12.0, 0.0],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )
    support_ids = query_ids[:-1]
    support_index_by_id = {
        query_id: query_ids.index(query_id) for query_id in support_ids
    }
    baseline_fit = fit_canonical_clusters(
        embeddings=embeddings,
        query_ids=query_ids,
        support_indices=(5, 3, 1, 4, 0, 2),
        cluster_count=3,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )
    (baseline_assignment,) = assign_targets(
        fit=baseline_fit,
        embeddings=embeddings,
        query_ids=query_ids,
        target_indices=(6,),
        support_index_by_id=support_index_by_id,
    )

    permutation = np.asarray([6, 4, 0, 5, 2, 1, 3], dtype=np.intp)
    permuted_query_ids = tuple(query_ids[index] for index in permutation)
    permuted_embeddings = np.ascontiguousarray(embeddings[permutation])
    permuted_support_index_by_id = {
        query_id: permuted_query_ids.index(query_id) for query_id in support_ids
    }
    permuted_fit = fit_canonical_clusters(
        embeddings=permuted_embeddings,
        query_ids=permuted_query_ids,
        support_indices=tuple(reversed(tuple(permuted_support_index_by_id.values()))),
        cluster_count=3,
        embedding_sha256=_EMBEDDING_SHA256,
        clustering_contract_sha256=_CLUSTERING_CONTRACT_SHA256,
    )
    (permuted_assignment,) = assign_targets(
        fit=permuted_fit,
        embeddings=permuted_embeddings,
        query_ids=permuted_query_ids,
        target_indices=(permuted_query_ids.index("qt"),),
        support_index_by_id=permuted_support_index_by_id,
    )

    assert baseline_assignment.canonical_cluster_id == (
        permuted_assignment.canonical_cluster_id
    ) == 1
    assert baseline_assignment.squared_distance.hex() == (
        permuted_assignment.squared_distance.hex()
    ) == float(1.0).hex()
    assert baseline_assignment.cluster_support == (
        permuted_assignment.cluster_support
    ) == 2
    baseline_member_ids = tuple(
        query_ids[index] for index in baseline_assignment.member_indices
    )
    permuted_member_ids = tuple(
        permuted_query_ids[index] for index in permuted_assignment.member_indices
    )
    assert baseline_member_ids == permuted_member_ids == ("q20", "q21")
    assert baseline_assignment.member_indices != permuted_assignment.member_indices
