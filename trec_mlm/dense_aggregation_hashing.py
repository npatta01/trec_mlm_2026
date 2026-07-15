from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


_HASH_PREFIX = b"trec-mlm-dense-aggregation-hash-v1\0"
_LOGICAL_TAGS = {
    "canonical-centers",
    "inner-validation-score",
    "inner-validation-rank",
    "full-development-score",
    "full-development-rank",
}
_LOGICAL_DTYPES = {"<f4", "<f8", "<u2"}
_SCORE_RANK_TAGS = _LOGICAL_TAGS - {"canonical-centers"}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _feed_payload(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def framed_sha256(tag: str, *payloads: bytes) -> str:
    tag_bytes = tag.encode("ascii")
    digest = hashlib.sha256()
    digest.update(_HASH_PREFIX)
    for payload in (tag_bytes, *payloads):
        _feed_payload(digest, payload)
    return digest.hexdigest()


def _declared_numeric_dtype(dtype: str) -> np.dtype[Any]:
    if type(dtype) is not str or dtype not in _LOGICAL_DTYPES:
        raise ValueError(
            "declared dtype must be a protocol-safe exact numeric scalar dtype: "
            "<f4, <f8, or <u2"
        )
    return np.dtype(dtype)


def _validate_stream_shape(shape: object) -> tuple[int, int]:
    if (
        type(shape) is not tuple
        or len(shape) != 2
        or type(shape[0]) is not int
        or type(shape[1]) is not int
        or shape[0] < 0
        or shape[1] < 0
    ):
        raise ValueError(
            "streamed shape must be exactly a tuple of two nonnegative ints"
        )
    return shape


def _validate_shape_and_inventories(
    *,
    tag: str,
    shape: tuple[int, int],
    row_inventory: Sequence[str | int],
    column_inventory: Sequence[str | int],
) -> None:
    if shape[0] != len(row_inventory):
        raise ValueError("array row count does not match semantic inventory")
    if tag in _SCORE_RANK_TAGS or column_inventory:
        if shape[1] != len(column_inventory):
            raise ValueError("array column count does not match semantic inventory")


def logical_array_sha256(
    array: NDArray[Any],
    *,
    tag: str,
    dtype: str,
    row_inventory: Sequence[str | int],
    column_inventory: Sequence[str | int],
) -> str:
    if tag not in _LOGICAL_TAGS:
        raise ValueError(f"unsupported logical-array tag: {tag}")
    declared = _declared_numeric_dtype(dtype)
    value = np.ascontiguousarray(array, dtype=declared)
    if value.ndim != 2:
        raise ValueError("logical array must be two-dimensional")
    _validate_shape_and_inventories(
        tag=tag,
        shape=(value.shape[0], value.shape[1]),
        row_inventory=row_inventory,
        column_inventory=column_inventory,
    )
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError("logical float array contains a non-finite value")
    return framed_sha256(
        tag,
        value.dtype.str.encode("ascii"),
        canonical_json(list(value.shape)),
        canonical_json(list(row_inventory)),
        canonical_json(list(column_inventory)),
        value.tobytes(order="C"),
    )


def support_query_ids_sha256(query_ids: Sequence[str]) -> str:
    return framed_sha256("support-query-ids", canonical_json(list(query_ids)))


def canonical_membership_sha256(
    memberships: Sequence[tuple[int, Sequence[str]]],
) -> str:
    rows = [[cluster_id, list(member_ids)] for cluster_id, member_ids in memberships]
    return framed_sha256("canonical-membership", canonical_json(rows))


class StreamingLogicalArrayHasher:
    def __init__(
        self,
        *,
        tag: str,
        dtype: str,
        shape: tuple[int, int],
        row_inventory: Sequence[str | int],
        column_inventory: Sequence[str | int],
    ) -> None:
        validated_shape = _validate_stream_shape(shape)
        if tag not in _LOGICAL_TAGS:
            raise ValueError(f"unsupported logical-array tag: {tag}")
        self._dtype = _declared_numeric_dtype(dtype)
        _validate_shape_and_inventories(
            tag=tag,
            shape=validated_shape,
            row_inventory=row_inventory,
            column_inventory=column_inventory,
        )
        self._shape = validated_shape
        self._row_count = 0
        self._byte_count = 0
        self._expected_bytes = math.prod(validated_shape) * self._dtype.itemsize
        self._digest = hashlib.sha256()
        self._digest.update(_HASH_PREFIX)
        for payload in (
            tag.encode("ascii"),
            self._dtype.str.encode("ascii"),
            canonical_json(list(validated_shape)),
            canonical_json(list(row_inventory)),
            canonical_json(list(column_inventory)),
        ):
            _feed_payload(self._digest, payload)
        self._digest.update(struct.pack("<Q", self._expected_bytes))

    def update_row(self, row: NDArray[Any]) -> None:
        if self._row_count >= self._shape[0]:
            raise ValueError("stream received too many rows")
        value = np.ascontiguousarray(row, dtype=self._dtype)
        if value.shape != (self._shape[1],):
            raise ValueError("stream row width mismatch")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("stream row contains a non-finite value")
        payload = value.tobytes(order="C")
        self._digest.update(payload)
        self._row_count += 1
        self._byte_count += len(payload)

    def hexdigest(self) -> str:
        if (
            self._row_count != self._shape[0]
            or self._byte_count != self._expected_bytes
        ):
            raise ValueError(
                "stream ended before the declared logical array was complete"
            )
        return self._digest.hexdigest()
