from __future__ import annotations

import json
from pathlib import Path

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
def golden() -> dict[str, str]:
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
    with pytest.raises(ValueError, match="shape"):
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


def test_clustering_contract_golden_vector(golden: dict[str, str]) -> None:
    payload = {
        "schema": "trec-mlm-dense-aggregation-clustering-v1",
        "threads": {"process_count": 1, "threadpool_limit": 1},
        "versions": {"numpy": "2.5.1", "python": "3.13.9"},
    }
    assert framed_sha256("clustering-contract", canonical_json(payload)) == golden[
        "clustering_contract_minimal"
    ]
