from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trec_mlm.dense_aggregation_hashing import (
    StreamingLogicalArrayHasher,
    canonical_json,
    canonical_membership_sha256,
    framed_sha256,
    logical_array_sha256,
    support_query_ids_sha256,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "dense_aggregation"
    / "hash_golden_vectors.json"
)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_canonical_json_unicode_golden_and_nan_rejection(
    golden: dict[str, str],
) -> None:
    payload = canonical_json({"z": "雪", "a": "café"})
    assert payload.hex() == golden["unicode_canonical_json_hex"]
    assert payload == '{"a":"café","z":"雪"}'.encode()

    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"value": float("nan")})


def test_framed_hash_golden_and_payload_boundaries(golden: dict[str, str]) -> None:
    assert (
        framed_sha256("example", b"alpha", b"")
        == golden["framed_example_alpha_empty"]
    )
    assert framed_sha256("example", b"ab", b"c") != framed_sha256(
        "example", b"a", b"bc"
    )
    assert framed_sha256("example", b"alpha", b"") != framed_sha256(
        "example", b"alpha"
    )


@pytest.mark.parametrize(
    ("fixture_key", "tag", "dtype", "value", "rows", "columns"),
    (
        (
            "canonical_centers_2x2",
            "canonical-centers",
            "<f4",
            [[1.0, -0.0], [2.5, 3.0]],
            (0, 1),
            (),
        ),
        (
            "inner_score_2x2",
            "inner-validation-score",
            "<f8",
            [[1.0, -0.0], [2.5, 3.0]],
            ("q1", "q2"),
            ("m1", "m2"),
        ),
        (
            "inner_rank_2x2",
            "inner-validation-rank",
            "<u2",
            [[1, 2], [2, 1]],
            ("q1", "q2"),
            ("m1", "m2"),
        ),
        (
            "full_score_2x2",
            "full-development-score",
            "<f8",
            [[1.0, -0.0], [2.5, 3.0]],
            ("q1", "q2"),
            ("m1", "m2"),
        ),
        (
            "full_rank_2x2",
            "full-development-rank",
            "<u2",
            [[1, 2], [2, 1]],
            ("q1", "q2"),
            ("m1", "m2"),
        ),
    ),
)
def test_all_logical_array_tags_have_literal_golden_vectors(
    golden: dict[str, str],
    fixture_key: str,
    tag: str,
    dtype: str,
    value: list[list[float | int]],
    rows: tuple[str | int, ...],
    columns: tuple[str | int, ...],
) -> None:
    assert (
        logical_array_sha256(
            np.asarray(value),
            tag=tag,
            dtype=dtype,
            row_inventory=rows,
            column_inventory=columns,
        )
        == golden[fixture_key]
    )


def test_logical_hash_converts_endian_dtype_and_layout(
    golden: dict[str, str],
) -> None:
    value = np.asfortranarray(
        np.asarray([[1.0, -0.0], [2.5, 3.0]], dtype=">f4")
    )
    assert value.dtype.str == ">f4"
    assert value.flags.f_contiguous

    assert (
        logical_array_sha256(
            value,
            tag="canonical-centers",
            dtype="<f4",
            row_inventory=(0, 1),
            column_inventory=(),
        )
        == golden["canonical_centers_2x2"]
    )


def test_inner_score_golden_and_streaming_equivalence(
    golden: dict[str, str],
) -> None:
    value = np.asarray([[1.0, -0.0], [2.5, 3.0]], dtype="<f8")
    expected = golden["inner_score_2x2"]
    assert expected == (
        "4af6b8651a99d6de0e405712653e16bae490e61b0473977cb9726147b67d7fed"
    )
    assert (
        logical_array_sha256(
            value,
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1", "q2"),
            column_inventory=("m1", "m2"),
        )
        == expected
    )
    streamed = StreamingLogicalArrayHasher(
        tag="inner-validation-score",
        dtype="<f8",
        shape=(2, 2),
        row_inventory=("q1", "q2"),
        column_inventory=("m1", "m2"),
    )
    streamed.update_row(np.asarray([1.0, -0.0], dtype=">f4"))
    streamed.update_row(np.asarray([2.5, 3.0], dtype=np.float32))
    assert streamed.hexdigest() == expected
    assert streamed.hexdigest() == expected


def test_logical_hash_excludes_receipt_context_and_binds_logical_metadata() -> None:
    value = np.asarray([[1.0, -0.0], [2.5, 3.0]], dtype="<f8")
    contexts = (
        {"outer_query_id": "q17", "config_id": "config-a"},
        {"outer_query_id": "q99", "config_id": "config-b"},
    )
    digests = [
        logical_array_sha256(
            value.copy(),
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1", "q2"),
            column_inventory=("m1", "m2"),
        )
        for _receipt_context in contexts
    ]
    assert digests[0] == digests[1]

    mutations = (
        logical_array_sha256(
            value,
            tag="inner-validation-score",
            dtype="<f4",
            row_inventory=("q1", "q2"),
            column_inventory=("m1", "m2"),
        ),
        logical_array_sha256(
            value.reshape(1, 4),
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1",),
            column_inventory=("m1", "m2", "m3", "m4"),
        ),
        logical_array_sha256(
            value,
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1", "q3"),
            column_inventory=("m1", "m2"),
        ),
        logical_array_sha256(
            value,
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1", "q2"),
            column_inventory=("m1", "m3"),
        ),
        logical_array_sha256(
            np.asarray([[1.0, -0.0], [2.5, 3.25]], dtype="<f8"),
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1", "q2"),
            column_inventory=("m1", "m2"),
        ),
    )
    assert all(digest != digests[0] for digest in mutations)
    assert len(set(mutations)) == len(mutations)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"tag": "not-a-logical-tag", "dtype": "<f8"},
        {"tag": "inner-validation-score", "dtype": ">f8"},
    ),
)
def test_logical_hash_rejects_unsupported_tag_or_non_little_endian_dtype(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        logical_array_sha256(
            np.ones((1, 1)),
            row_inventory=("q1",),
            column_inventory=("m1",),
            **kwargs,
        )


def test_logical_hash_rejects_invalid_shape_inventory_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        logical_array_sha256(
            np.ones(2),
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1", "q2"),
            column_inventory=(),
        )
    with pytest.raises(ValueError, match="row count"):
        logical_array_sha256(
            np.ones((2, 2)),
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1",),
            column_inventory=("m1", "m2"),
        )
    with pytest.raises(ValueError, match="column count"):
        logical_array_sha256(
            np.ones((2, 2)),
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1", "q2"),
            column_inventory=("m1",),
        )
    with pytest.raises(ValueError, match="non-finite"):
        logical_array_sha256(
            np.asarray([[1.0, np.inf]]),
            tag="inner-validation-score",
            dtype="<f8",
            row_inventory=("q1",),
            column_inventory=("m1", "m2"),
        )


@pytest.mark.parametrize(
    ("tag", "dtype"),
    (
        ("inner-validation-score", "<f8"),
        ("inner-validation-rank", "<u2"),
        ("full-development-score", "<f8"),
        ("full-development-rank", "<u2"),
    ),
)
def test_materialized_score_and_rank_hashes_require_complete_inventories(
    tag: str, dtype: str
) -> None:
    value = np.ones((1, 1))
    with pytest.raises(ValueError, match="row count"):
        logical_array_sha256(
            value,
            tag=tag,
            dtype=dtype,
            row_inventory=(),
            column_inventory=("m1",),
        )
    with pytest.raises(ValueError, match="column count"):
        logical_array_sha256(
            value,
            tag=tag,
            dtype=dtype,
            row_inventory=("q1",),
            column_inventory=(),
        )


@pytest.mark.parametrize(
    ("tag", "dtype"),
    (
        ("inner-validation-score", "<f8"),
        ("inner-validation-rank", "<u2"),
        ("full-development-score", "<f8"),
        ("full-development-rank", "<u2"),
    ),
)
def test_streamed_score_and_rank_hashes_require_complete_inventories(
    tag: str, dtype: str
) -> None:
    with pytest.raises(ValueError):
        StreamingLogicalArrayHasher(
            tag=tag,
            dtype=dtype,
            shape=(1, 1),
            row_inventory=(),
            column_inventory=("m1",),
        )
    with pytest.raises(ValueError, match="column count"):
        StreamingLogicalArrayHasher(
            tag=tag,
            dtype=dtype,
            shape=(1, 1),
            row_inventory=("q1",),
            column_inventory=(),
        )


def test_canonical_centers_require_rows_but_allow_empty_columns(
    golden: dict[str, Any],
) -> None:
    value = np.asarray([[1.0, -0.0], [2.5, 3.0]], dtype="<f4")
    streamed = StreamingLogicalArrayHasher(
        tag="canonical-centers",
        dtype="<f4",
        shape=(2, 2),
        row_inventory=(0, 1),
        column_inventory=(),
    )
    streamed.update_row(value[0])
    streamed.update_row(value[1])
    assert streamed.hexdigest() == golden["canonical_centers_2x2"]

    with pytest.raises(ValueError, match="row count"):
        logical_array_sha256(
            value,
            tag="canonical-centers",
            dtype="<f4",
            row_inventory=(),
            column_inventory=(),
        )
    with pytest.raises(ValueError, match="row count"):
        StreamingLogicalArrayHasher(
            tag="canonical-centers",
            dtype="<f4",
            shape=(2, 2),
            row_inventory=(),
            column_inventory=(),
        )


def test_equal_object_arrays_are_rejected_before_hashing() -> None:
    left = np.empty((1, 1), dtype=object)
    right = np.empty((1, 1), dtype=object)
    left[0, 0] = bytearray(b"equal-value")
    right[0, 0] = bytearray(b"equal-value")
    assert np.array_equal(left, right)

    for value in (left, right):
        with pytest.raises(ValueError, match="numeric scalar dtype"):
            logical_array_sha256(
                value,
                tag="inner-validation-score",
                dtype="O",
                row_inventory=("q1",),
                column_inventory=("m1",),
            )
    with pytest.raises(ValueError, match="numeric scalar dtype"):
        StreamingLogicalArrayHasher(
            tag="inner-validation-score",
            dtype="O",
            shape=(1, 1),
            row_inventory=("q1",),
            column_inventory=("m1",),
        )


def test_structured_declared_dtype_is_rejected_by_both_hashers() -> None:
    declared = np.dtype([("left", "<i4"), ("right", "<f4")])
    value = np.zeros((1, 1), dtype=declared)
    with pytest.raises(ValueError, match="numeric scalar dtype"):
        logical_array_sha256(
            value,
            tag="inner-validation-score",
            dtype=declared,  # type: ignore[arg-type]
            row_inventory=("q1",),
            column_inventory=("m1",),
        )
    with pytest.raises(ValueError, match="numeric scalar dtype"):
        StreamingLogicalArrayHasher(
            tag="inner-validation-score",
            dtype=declared,  # type: ignore[arg-type]
            shape=(1, 1),
            row_inventory=("q1",),
            column_inventory=("m1",),
        )


@pytest.mark.parametrize(
    "dtype",
    ("|S4", "<U4", "?", "V8", "datetime64[ns]", "timedelta64[ns]"),
)
def test_nonnumeric_declared_dtype_is_rejected_by_both_hashers(dtype: str) -> None:
    value = np.asarray([[1]])
    with pytest.raises(ValueError, match="numeric scalar dtype"):
        logical_array_sha256(
            value,
            tag="inner-validation-score",
            dtype=dtype,
            row_inventory=("q1",),
            column_inventory=("m1",),
        )
    with pytest.raises(ValueError, match="numeric scalar dtype"):
        StreamingLogicalArrayHasher(
            tag="inner-validation-score",
            dtype=dtype,
            shape=(1, 1),
            row_inventory=("q1",),
            column_inventory=("m1",),
        )


def _streaming_hasher(
    *, shape: tuple[int, int] = (2, 2)
) -> StreamingLogicalArrayHasher:
    return StreamingLogicalArrayHasher(
        tag="inner-validation-score",
        dtype="<f8",
        shape=shape,
        row_inventory=tuple(f"q{index}" for index in range(shape[0])),
        column_inventory=("m1", "m2"),
    )


def test_streaming_hash_rejects_wrong_row_count_width_and_nonfinite_float() -> None:
    with pytest.raises(ValueError, match="row count"):
        StreamingLogicalArrayHasher(
            tag="inner-validation-score",
            dtype="<f8",
            shape=(2, 2),
            row_inventory=("q1",),
            column_inventory=("m1", "m2"),
        )

    incomplete = _streaming_hasher()
    incomplete.update_row(np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="before"):
        incomplete.hexdigest()

    wrong_width = _streaming_hasher()
    with pytest.raises(ValueError, match="width"):
        wrong_width.update_row(np.asarray([1.0]))

    nonfinite = _streaming_hasher()
    with pytest.raises(ValueError, match="non-finite"):
        nonfinite.update_row(np.asarray([1.0, np.nan]))

    too_many = _streaming_hasher(shape=(1, 2))
    too_many.update_row(np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="too many"):
        too_many.update_row(np.asarray([3.0, 4.0]))


def test_support_and_membership_golden_vectors(golden: dict[str, str]) -> None:
    assert support_query_ids_sha256(("q1", "q2")) == golden["support_q1_q2"]
    assert support_query_ids_sha256(("q2", "q1")) != golden["support_q1_q2"]

    memberships = ((0, ("q1", "q3")), (1, ("q2",)))
    assert (
        canonical_membership_sha256(memberships)
        == golden["membership_two_clusters"]
    )
    assert (
        canonical_membership_sha256((memberships[0],))
        == golden["membership_cluster_zero"]
    )
    assert (
        canonical_membership_sha256((memberships[1],))
        == golden["membership_cluster_one"]
    )
    assert canonical_membership_sha256(tuple(reversed(memberships))) != golden[
        "membership_two_clusters"
    ]


def test_complete_clustering_contract_golden_vector(golden: dict[str, Any]) -> None:
    expected_payload = {
        "center": {
            "canonical_order": "minimum_member_query_id_lexical",
            "dtype": "<f4",
            "layout": "C",
        },
        "convergence": {
            "n_iter_max": 299,
            "n_iter_min": 1,
            "warning_action": "reject",
        },
        "embedding_sha256": (
            "6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b"
        ),
        "implementation_files": [
            [
                "trec_mlm/dense_aggregation_artifacts.py",
                "0" * 64,
            ],
            [
                "trec_mlm/dense_aggregation_hashing.py",
                "1" * 64,
            ],
        ],
        "inertia": {
            "atol": 1e-8,
            "cluster_reduction": (
                "numpy.sum(member_delta * member_delta, dtype=numpy.float64)"
            ),
            "dtype": "<f8",
            "rtol": 1e-5,
            "total_reduction": "math.fsum(cluster_inertia)",
        },
        "kmeans": {
            "algorithm": "lloyd",
            "init": "k-means++",
            "max_iter": 300,
            "n_init": 10,
            "random_state": 2026,
            "tol": 1e-4,
        },
        "protocol_file_sha256": (
            "bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49"
        ),
        "schema": "trec-mlm-dense-aggregation-clustering-v1",
        "similarity_sha256": (
            "80c69a8dca29cd5e9a4de08dc4e44553fabf813428dd4956f7ecbeb2852968ed"
        ),
        "target_distance": {
            "dtype": "<f8",
            "reduction": (
                "numpy.sum(delta[z] * delta[z], dtype=numpy.float64)"
            ),
            "tie": "minimum_canonical_cluster_id",
        },
        "threads": {
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "process_count": 1,
            "threadpool_limit": 1,
        },
        "versions": {
            "numpy": "2.5.1",
            "python": "3.13.9",
            "scikit_learn": "1.9.0",
            "scipy": "1.18.0",
        },
    }
    payload = golden["clustering_contract_full_payload"]
    assert payload == expected_payload
    assert set(payload) == {
        "schema",
        "protocol_file_sha256",
        "implementation_files",
        "embedding_sha256",
        "similarity_sha256",
        "versions",
        "threads",
        "kmeans",
        "center",
        "target_distance",
        "inertia",
        "convergence",
    }
    assert framed_sha256("clustering-contract", canonical_json(payload)) == golden[
        "clustering_contract_full_sha256"
    ]
