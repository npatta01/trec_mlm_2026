from __future__ import annotations

import dataclasses
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from tests.test_dense_aggregation_cache import (
    _cache_mutation_snapshot,
    _independent_canonical_json,
    _independent_center_sha256,
    _independent_framed_sha256,
    _populate_one_fit_cache,
    _synthetic_pair_fit,
    _valid_header,
)
import trec_mlm.dense_aggregation_cache as cache_module
from trec_mlm.dense_aggregation_cache import StreamingFitCache


def _rewrite_single_record(path: Path, record: dict[str, object]) -> None:
    original = path.read_bytes()
    assert original.count(b"\n") == 1
    replacement = _independent_canonical_json(record) + b"\n"
    assert len(replacement) == len(original)
    path.write_bytes(replacement)


def test_finalize_binds_appended_fit_receipt_bytes(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache, _fit = _populate_one_fit_cache(root)
    cache._receipt_handle.flush()
    receipt_path = root / "fit_receipts.jsonl"
    record = json.loads(receipt_path.read_bytes())
    assignment = record["assignments"][0]
    original_hex = assignment["squared_distance_hex"]
    replacement_hex = math.nextafter(float.fromhex(original_hex), math.inf).hex()
    assert replacement_hex != original_hex
    assert len(replacement_hex) == len(original_hex)
    assignment["squared_distance_hex"] = replacement_hex
    without_hash = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = _independent_framed_sha256(
        "fit-cache-record", _independent_canonical_json(without_hash)
    )
    _rewrite_single_record(receipt_path, record)

    with pytest.raises(ValueError):
        cache.finalize()

    assert not (root / "header.json").exists()


def test_finalize_binds_appended_timing_bytes(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache, _fit = _populate_one_fit_cache(root)
    cache._timing_handle.flush()
    timing_path = root / "timings.jsonl"
    record = json.loads(timing_path.read_bytes())
    original_elapsed_ns = record["elapsed_ns"]
    record["elapsed_ns"] = original_elapsed_ns + 1
    assert len(str(record["elapsed_ns"])) == len(str(original_elapsed_ns))
    _rewrite_single_record(timing_path, record)

    with pytest.raises(ValueError):
        cache.finalize()

    assert not (root / "header.json").exists()


def test_append_rejects_cross_fit_center_width_without_mutation(
    tmp_path: Path,
) -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
        {"exclusion_kind": "pair", "inventory_index": 1, "cluster_count": 2},
    ]
    cache = StreamingFitCache.create(
        tmp_path / "cache", header=_valid_header(inventory=inventory)
    )
    first_fit, first_assignments, first_excluded = _synthetic_pair_fit(0)
    cache.append_fit(
        first_fit,
        first_assignments,
        exclusion_kind="pair",
        inventory_index=0,
        excluded_query_ids=first_excluded,
    )
    second_fit, second_assignments, second_excluded = _synthetic_pair_fit(1)
    wide_centers = np.ascontiguousarray(
        np.concatenate((second_fit.centers, np.zeros((2, 1), dtype="<f4")), axis=1),
        dtype="<f4",
    )
    wide_fit = dataclasses.replace(
        second_fit,
        centers=wide_centers,
        center_sha256=_independent_center_sha256(wide_centers),
    )
    before = _cache_mutation_snapshot(cache)

    with pytest.raises(ValueError):
        cache.append_fit(
            wide_fit,
            second_assignments,
            exclusion_kind="pair",
            inventory_index=1,
            excluded_query_ids=second_excluded,
        )

    assert _cache_mutation_snapshot(cache) == before
    assert all(
        not handle.closed
        for handle in (
            cache._center_handle,
            cache._label_handle,
            cache._inertia_handle,
            cache._receipt_handle,
            cache._timing_handle,
        )
    )
    cache.append_fit(
        second_fit,
        second_assignments,
        exclusion_kind="pair",
        inventory_index=1,
        excluded_query_ids=second_excluded,
    )
    assert cache._fit_count == 2


class _InjectedPublicationFailure(OSError):
    pass


@pytest.mark.parametrize(
    "failure_point",
    ("header_fsync", "directory_fsync", "independent_open"),
)
def test_finalize_publication_failure_is_terminal_and_leaves_no_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    root = tmp_path / "cache"
    cache, fit = _populate_one_fit_cache(root)
    if failure_point in {"header_fsync", "directory_fsync"}:
        failure_call = 6 if failure_point == "header_fsync" else 7
        real_fsync = os.fsync
        fsync_calls = 0

        def failing_fsync(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == failure_call:
                raise _InjectedPublicationFailure(
                    f"injected {failure_point} publication failure"
                )
            real_fsync(descriptor)

        monkeypatch.setattr(cache_module.os, "fsync", failing_fsync)
    else:
        real_open_readonly = StreamingFitCache.open_readonly
        open_attempts = 0

        def failing_open_readonly(
            cls: type[StreamingFitCache], candidate_root: str | Path
        ) -> StreamingFitCache:
            nonlocal open_attempts
            open_attempts += 1
            if open_attempts == 1:
                raise _InjectedPublicationFailure(
                    "injected independent_open publication failure"
                )
            return real_open_readonly(candidate_root)

        monkeypatch.setattr(
            StreamingFitCache,
            "open_readonly",
            classmethod(failing_open_readonly),
        )

    with pytest.raises(_InjectedPublicationFailure, match="injected"):
        cache.finalize()

    assert not (root / "header.json").exists()
    with pytest.raises(ValueError):
        StreamingFitCache.open_readonly(root)

    # Publication has started, so every later failure is terminal and fail-closed;
    # only an incomplete preflight remains retryable in the writing state.
    assert cache._state == "failed"

    actions = (
        lambda: cache.append_timing(
            exclusion_kind="pair",
            inventory_index=0,
            cluster_count=2,
            elapsed_ns=1,
            n_iter=fit.n_iter,
        ),
        cache.finalize,
        lambda: cache.finalize_into(tmp_path / "published"),
    )
    for action in actions:
        for _attempt in range(2):
            with pytest.raises(RuntimeError):
                action()
            assert cache._state == "failed"


def test_create_cleanup_preserves_foreign_entry_from_stream_open_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    sentinel = root / "foreign-sentinel"
    real_path_open = Path.open
    cache_stream_open_attempts = 0

    def racing_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        nonlocal cache_stream_open_attempts
        if mode == "xb" and path.parent == root:
            cache_stream_open_attempts += 1
            if cache_stream_open_attempts == 2:
                descriptor = os.open(
                    sentinel,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    os.write(descriptor, b"foreign")
                finally:
                    os.close(descriptor)
                raise FileExistsError("injected stream-open race")
        return real_path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)

    with pytest.raises(FileExistsError, match="stream-open race"):
        StreamingFitCache.create(root, header=_valid_header())

    assert root.is_dir()
    assert sentinel.read_bytes() == b"foreign"
    assert {entry.name for entry in root.iterdir()} == {sentinel.name}


def test_expected_inventory_row_is_constant_time_after_header_validation() -> None:
    inventory = [
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 1},
        {"exclusion_kind": "pair", "inventory_index": 0, "cluster_count": 2},
    ]
    header = cache_module._validate_header(_valid_header(inventory=inventory))
    owned_inventory = header["expected_inventory"]
    assert type(owned_inventory) is list
    expected = owned_inventory[0]
    poison = object()
    owned_inventory[1] = poison

    # This private helper receives the already validated, owned header and must
    # index only the requested row rather than revalidating or scanning later rows.
    assert cache_module._expected_inventory_row(header, 0) == expected
    assert owned_inventory[1] is poison


_MAX_JSONL_READ_BYTES = 16 * 1024 * 1024


class _BoundedJSONLHandle:
    def __init__(
        self,
        wrapped: Any,
        path_name: str,
        calls: list[tuple[str, str, int]],
    ):
        self._wrapped = wrapped
        self._path_name = path_name
        self._calls = calls

    def _validate_size(self, method: str, size: int) -> None:
        self._calls.append((self._path_name, method, size))
        assert type(size) is int, "JSONL read sizes must be exact integers"
        assert 0 <= size <= _MAX_JSONL_READ_BYTES, (
            "JSONL reads must be finite and no larger than 16 MiB"
        )

    def read(self, size: int = -1) -> bytes:
        self._validate_size("read", size)
        return self._wrapped.read(size)

    def readline(self, size: int = -1) -> bytes:
        self._validate_size("readline", size)
        return self._wrapped.readline(size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def test_open_readonly_uses_bounded_jsonl_read_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    cache, _fit = _populate_one_fit_cache(root)
    cache.finalize()
    real_open_regular_binary = cache_module._open_regular_binary
    calls: list[tuple[str, str, int]] = []

    def guarded_open(path: Path) -> Any:
        handle = real_open_regular_binary(path)
        if path.name in {"fit_receipts.jsonl", "timings.jsonl"}:
            return _BoundedJSONLHandle(handle, path.name, calls)
        return handle

    monkeypatch.setattr(cache_module, "_open_regular_binary", guarded_open)

    reopened = StreamingFitCache.open_readonly(root)

    assert reopened._state == "finalized"
    assert {path_name for path_name, _method, _size in calls} == {
        "fit_receipts.jsonl",
        "timings.jsonl",
    }
    assert all(
        0 <= size <= _MAX_JSONL_READ_BYTES for _path_name, _method, size in calls
    )


class _ReadMustNotOccur:
    def __init__(self, wrapped: Any):
        self._wrapped = wrapped
        self.read_called = False

    def read(self, size: int = -1) -> bytes:
        self.read_called = True
        raise AssertionError("descriptor span must be rejected before read")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def test_read_slice_rejects_out_of_file_span_before_read(tmp_path: Path) -> None:
    shard = tmp_path / "tiny.f4"
    shard.write_bytes(b"tiny")
    with shard.open("rb") as raw_handle:
        guarded_handle = _ReadMustNotOccur(raw_handle)
        with pytest.raises(ValueError):
            cache_module._read_slice(
                guarded_handle,
                offset=3,
                byte_count=2,
                label="center",
                expected_offset=None,
            )
        assert guarded_handle.read_called is False


def test_scan_streams_closes_every_acquired_handle_if_fifth_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    cache, _fit = _populate_one_fit_cache(root)
    cache.finalize()
    real_open_regular_binary = cache_module._open_regular_binary
    acquired_handles: list[Any] = []
    acquisition_count = 0

    def failing_fifth_open(path: Path) -> Any:
        nonlocal acquisition_count
        acquisition_count += 1
        if acquisition_count == 5:
            raise _InjectedPublicationFailure("injected fifth stream open failure")
        handle = real_open_regular_binary(path)
        acquired_handles.append(handle)
        return handle

    monkeypatch.setattr(cache_module, "_open_regular_binary", failing_fifth_open)

    try:
        with pytest.raises(_InjectedPublicationFailure, match="fifth stream open"):
            cache_module._scan_streams(root, cache._header)
        assert acquisition_count == 5
        assert len(acquired_handles) == 4
        assert all(handle.closed for handle in acquired_handles)
    finally:
        for handle in acquired_handles:
            if not handle.closed:
                handle.close()


def test_create_normalizes_unhashable_inventory_kind_to_value_error(
    tmp_path: Path,
) -> None:
    header = _valid_header()
    header["expected_inventory"][0]["exclusion_kind"] = []
    root = tmp_path / "cache"

    with pytest.raises(ValueError):
        StreamingFitCache.create(root, header=header)

    assert not root.exists()


def test_open_readonly_normalizes_huge_hex_exponent_to_value_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    cache, _fit = _populate_one_fit_cache(root)
    cache.finalize()
    receipt_path = root / "fit_receipts.jsonl"
    record = json.loads(receipt_path.read_bytes())
    record["assignments"][0]["squared_distance_hex"] = "0x1p+1024"
    without_hash = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = _independent_framed_sha256(
        "fit-cache-record", _independent_canonical_json(without_hash)
    )
    receipt_path.write_bytes(_independent_canonical_json(record) + b"\n")

    with pytest.raises(ValueError):
        StreamingFitCache.open_readonly(root)


class _ShortWritingBytesIO(io.BytesIO):
    def __init__(self, chunk_size: int):
        super().__init__()
        self._chunk_size = chunk_size

    def write(self, payload: Any) -> int:
        chunk = bytes(payload)[: self._chunk_size]
        return super().write(chunk)


@pytest.mark.parametrize("append_kind", ("array", "jsonl"))
def test_append_helpers_never_silently_accept_short_writes(
    append_kind: str,
) -> None:
    handle = _ShortWritingBytesIO(chunk_size=3)
    if append_kind == "array":
        value = np.ascontiguousarray(np.arange(8, dtype="<f4"))
        expected_payload = value.tobytes(order="C")

        def append() -> tuple[int, int]:
            return StreamingFitCache._append_array(handle, value)

    else:
        record = {"schema": "short-write-probe", "value": "0123456789"}
        expected_payload = _independent_canonical_json(record) + b"\n"

        def append() -> tuple[int, int]:
            return StreamingFitCache._append_jsonl(handle, record)

    try:
        result = append()
    except (OSError, ValueError):
        return

    assert result == (0, len(expected_payload))
    assert handle.getvalue() == expected_payload


class _PrimaryFsyncFailure(OSError):
    pass


class _CleanupCloseFailure(OSError):
    pass


class _InjectedFstatFailure(OSError):
    pass


class _FirstCloseFailure:
    def __init__(self, message: str):
        self._message = message
        self._armed = True

    def close(self, handle: Any) -> None:
        was_open = not handle.closed
        handle.close()
        if was_open and self._armed:
            self._armed = False
            raise _CleanupCloseFailure(self._message)


class _CloseFailureHandle:
    def __init__(self, wrapped: Any, controller: _FirstCloseFailure):
        self._wrapped = wrapped
        self._controller = controller

    def close(self) -> None:
        self._controller.close(self._wrapped)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def test_finalize_preserves_primary_fsync_failure_over_close_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    header_path = root / "header.json"
    cache, _fit = _populate_one_fit_cache(root)
    real_path_open = Path.open
    header_descriptor: int | None = None
    close_failure = _FirstCloseFailure("injected header close cleanup failure")

    def wrapping_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        nonlocal header_descriptor
        handle = real_path_open(path, mode, *args, **kwargs)
        if path == header_path and mode == "xb":
            header_descriptor = handle.fileno()
            return _CloseFailureHandle(handle, close_failure)
        return handle

    real_fsync = cache_module.os.fsync

    def failing_header_fsync(descriptor: int) -> None:
        if descriptor == header_descriptor:
            raise _PrimaryFsyncFailure("injected primary header fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(Path, "open", wrapping_open)
    monkeypatch.setattr(cache_module.os, "fsync", failing_header_fsync)

    with pytest.raises((_PrimaryFsyncFailure, _CleanupCloseFailure)) as caught:
        cache.finalize()

    assert cache._state == "failed"
    assert not header_path.exists()
    assert type(caught.value) is _PrimaryFsyncFailure


def test_direct_load_cleanup_closes_all_acquired_binary_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    cache, fit = _populate_one_fit_cache(root)
    cache.finalize()
    cache.release(fit.key)
    real_open_regular_binary = cache_module._open_regular_binary
    binary_names = {"centers.f4", "labels.u2", "cluster_inertia.f8"}
    close_failure = _FirstCloseFailure("injected binary close cleanup failure")
    acquired: list[_CloseFailureHandle] = []

    def wrapping_binary_open(path: Path) -> Any:
        handle = real_open_regular_binary(path)
        if path.name not in binary_names:
            return handle
        wrapped = _CloseFailureHandle(handle, close_failure)
        acquired.append(wrapped)
        return wrapped

    monkeypatch.setattr(cache_module, "_open_regular_binary", wrapping_binary_open)

    try:
        try:
            cache.load_fit(fit.key)
        except _CleanupCloseFailure:
            pass
        assert len(acquired) == len(binary_names)
        assert all(handle.closed for handle in acquired)
    finally:
        for handle in acquired:
            if not handle.closed:
                handle.close()


def test_create_fstat_failure_removes_owned_stream_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    real_fstat = cache_module.os.fstat
    fault_armed = True
    injected_failures = 0

    def failing_first_fstat(descriptor: int) -> os.stat_result:
        nonlocal fault_armed, injected_failures
        if fault_armed:
            fault_armed = False
            injected_failures += 1
            raise _InjectedFstatFailure("injected first stream fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(cache_module.os, "fstat", failing_first_fstat)

    with pytest.raises(_InjectedFstatFailure, match="first stream fstat"):
        StreamingFitCache.create(root, header=_valid_header())

    assert injected_failures == 1
    assert not root.exists()
    retry = StreamingFitCache.create(root, header=_valid_header())
    try:
        assert retry._state == "writing"
        assert all(
            not handle.closed
            for handle in (
                retry._center_handle,
                retry._label_handle,
                retry._inertia_handle,
                retry._receipt_handle,
                retry._timing_handle,
            )
        )
    finally:
        for handle in (
            retry._center_handle,
            retry._label_handle,
            retry._inertia_handle,
            retry._receipt_handle,
            retry._timing_handle,
        ):
            handle.close()


def test_disabled_integer_guard_rejects_timing_larger_than_reader_limit(
    tmp_path: Path,
) -> None:
    if not hasattr(sys, "get_int_max_str_digits"):
        pytest.skip("runtime does not expose the integer digit guard")
    original_limit = sys.get_int_max_str_digits()
    cache: StreamingFitCache | None = None
    try:
        sys.set_int_max_str_digits(0)
        root = tmp_path / "cache"
        cache = StreamingFitCache.create(root, header=_valid_header())
        fit, assignments, excluded_query_ids = _synthetic_pair_fit(0)
        cache.append_fit(
            fit,
            assignments,
            exclusion_kind="pair",
            inventory_index=0,
            excluded_query_ids=excluded_query_ids,
        )
        _receipt_limit, timing_limit = cache_module._jsonl_record_limits(cache._header)
        oversized_elapsed_ns = 10**timing_limit
        assert len(str(oversized_elapsed_ns)) > timing_limit
        before = _cache_mutation_snapshot(cache)
        before_index = cache._index.copy()

        with pytest.raises(ValueError):
            cache.append_timing(
                exclusion_kind="pair",
                inventory_index=0,
                cluster_count=2,
                elapsed_ns=oversized_elapsed_ns,
                n_iter=fit.n_iter,
            )

        assert _cache_mutation_snapshot(cache) == before
        assert cache._index == before_index
        cache.append_timing(
            exclusion_kind="pair",
            inventory_index=0,
            cluster_count=2,
            elapsed_ns=1,
            n_iter=fit.n_iter,
        )
        cache.finalize()
        assert cache._state == "finalized"
    finally:
        sys.set_int_max_str_digits(original_limit)
        if cache is not None and cache._state == "writing":
            for handle in (
                cache._center_handle,
                cache._label_handle,
                cache._inertia_handle,
                cache._receipt_handle,
                cache._timing_handle,
            ):
                if not handle.closed:
                    handle.close()


def test_finalize_identity_failure_preserves_foreign_header_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    header_path = root / "header.json"
    foreign_payload = b"foreign header sentinel\n"
    cache, _fit = _populate_one_fit_cache(root)
    real_path_open = Path.open
    header_descriptor: int | None = None
    identity_fault_injected = False

    def capturing_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        nonlocal header_descriptor
        handle = real_path_open(path, mode, *args, **kwargs)
        if path == header_path and mode == "xb":
            header_descriptor = handle.fileno()
        return handle

    real_fstat = cache_module.os.fstat

    def replacing_header_then_failing(descriptor: int) -> os.stat_result:
        nonlocal identity_fault_injected
        if descriptor == header_descriptor and not identity_fault_injected:
            identity_fault_injected = True
            header_path.unlink()
            foreign_descriptor = os.open(
                header_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                remaining = memoryview(foreign_payload)
                while remaining:
                    written = os.write(foreign_descriptor, remaining)
                    if written <= 0:
                        raise OSError("foreign sentinel write made no progress")
                    remaining = remaining[written:]
            finally:
                os.close(foreign_descriptor)
            raise _InjectedFstatFailure("injected header identity failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(Path, "open", capturing_open)
    monkeypatch.setattr(cache_module.os, "fstat", replacing_header_then_failing)

    with pytest.raises(_InjectedFstatFailure, match="header identity"):
        cache.finalize()

    assert identity_fault_injected
    assert cache._state == "failed"
    assert header_path.exists()
    assert header_path.read_bytes() == foreign_payload
