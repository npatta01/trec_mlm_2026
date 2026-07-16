#!/usr/bin/env python3
"""Build the isolated fixed-MPNet, k=50 legacy-control lock."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import uuid


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in _THREAD_VARIABLES:
    os.environ[_thread_variable] = "1"


_HASH_PREFIX = b"trec-mlm-dense-aggregation-hash-v1\0"
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_MANIFEST_SCHEMA = "trec-mlm-dense-aggregation-manifest-v1"
_GATE_SCHEMA = "trec-mlm-dense-aggregation-feasibility-gate-v1"
_WORKER_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-worker-v1"
_REPEAT_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-repeat-v1"
_LOCK_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-lock-v1"
_COMPACT_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-compact-v1"
_LOCK_FILENAME = "fixed_mpnet_k50_control.lock.json"
_TSV_FILENAME = "fixed_mpnet_k50_control.tsv"
_TREC_FILENAME = "fixed_mpnet_k50_control.trec"
_RUN_ID = "fixed_mpnet_k50_control"
_EXPECTED_NDCG = 0.36045807293576454
_EXPECTED_MRR = 0.4039893654911617
_ROOT_FILES = (
    _LOCK_FILENAME,
    _TREC_FILENAME,
    _TSV_FILENAME,
    "repeat_a.json",
    "repeat_b.json",
)
_FEASIBILITY_FILES = (
    "feasibility_gate.json",
    "repeat_a_receipt.json",
    "repeat_b_receipt.json",
    "repeat_measurements.json",
)
_SOURCE_RELATIVE_PATHS = (
    "scripts/export_fixed_mpnet_k50_control.py",
    "trec_mlm/dense_query_knn.py",
    "trec_mlm/legacy_dense_control.py",
    "trec_mlm/metrics.py",
    "trec_mlm/query_knn.py",
    "trec_mlm/runio.py",
)
_WORKER_KEYS = {
    "schema",
    "source_sha256",
    "query_file_sha256",
    "qrel_file_sha256",
    "query_input_sha256",
    "candidate_inventory_sha256",
    "embedding_file_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "neighbor_count",
    "tsv_sha256",
    "row_count",
    "query_count",
    "candidate_count",
    "mean_ndcg_at_10",
    "mean_reciprocal_rank",
}
_WORKER_DIGEST_KEYS = {
    "query_file_sha256",
    "qrel_file_sha256",
    "query_input_sha256",
    "candidate_inventory_sha256",
    "embedding_file_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "tsv_sha256",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _framed_sha256(tag: str, *payloads: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_PREFIX)
    for payload in (tag.encode("ascii"), *payloads):
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _LOWERCASE_HEX for character in value)
    )


def _raw_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bounded_regular_file(path: Path, *, limit: int, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if before.st_size > limit:
        raise ValueError(f"{label} exceeds its byte limit")
    with path.open("rb") as handle:
        payload = handle.read(before.st_size + 1)
    try:
        after = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    if (
        len(payload) != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload


def _read_canonical_json(path: Path, *, limit: int, label: str) -> object:
    payload = _read_bounded_regular_file(path, limit=limit, label=label)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    try:
        expected = _canonical_json(value) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if payload != expected:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_exact_layout(root: Path, files: tuple[str, ...], *, label: str) -> None:
    try:
        status = root.lstat()
    except OSError as error:
        raise ValueError(f"{label} has an unexpected physical layout") from error
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{label} has an unexpected physical layout")
    expected = {"manifest.json", *files}
    observed: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if (
                    entry.name not in expected
                    or entry.name in observed
                    or not entry.is_file(follow_symlinks=False)
                ):
                    raise ValueError(f"{label} has an unexpected physical layout")
                observed.add(entry.name)
    except OSError as error:
        raise ValueError(f"{label} has an unexpected physical layout") from error
    if observed != expected:
        raise ValueError(f"{label} has an unexpected physical layout")


def _validate_manifest(
    root: Path, files: tuple[str, ...], *, label: str
) -> dict[str, object]:
    _require_exact_layout(root, files, label=label)
    value = _read_canonical_json(
        root / "manifest.json", limit=1024 * 1024, label=f"{label} manifest"
    )
    if type(value) is not dict or set(value) != {
        "schema",
        "files",
        "artifact_inventory_sha256",
    }:
        raise ValueError(f"{label} manifest has an invalid contract")
    if value["schema"] != _MANIFEST_SCHEMA:
        raise ValueError(f"{label} manifest schema mismatch")
    rows = value["files"]
    if type(rows) is not list or len(rows) != len(files):
        raise ValueError(f"{label} manifest inventory is invalid")
    observed_rows: list[list[str | int]] = []
    for index, row in enumerate(rows):
        if type(row) is not list or len(row) != 3:
            raise ValueError(f"{label} manifest inventory is invalid")
        name, size, digest = row
        if (
            type(name) is not str
            or name != files[index]
            or type(size) is not int
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(digest)
        ):
            raise ValueError(f"{label} manifest inventory is invalid")
        payload = _read_bounded_regular_file(
            root / name, limit=32 * 1024 * 1024, label=f"{label} file {name}"
        )
        observed_rows.append([name, len(payload), hashlib.sha256(payload).hexdigest()])
    if rows != observed_rows:
        raise ValueError(f"{label} manifest does not match files")
    if value["artifact_inventory_sha256"] != _framed_sha256(
        "artifact-inventory", _canonical_json(observed_rows)
    ):
        raise ValueError(f"{label} manifest digest mismatch")
    return value


def _validate_feasibility_receipt(path: Path) -> str:
    if path.name != "manifest.json":
        raise ValueError("feasibility receipt must name manifest.json")
    try:
        manifest_path = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("feasibility receipt is missing") from error
    root = manifest_path.parent
    _validate_manifest(root, _FEASIBILITY_FILES, label="feasibility root")
    gate = _read_canonical_json(
        root / "feasibility_gate.json",
        limit=1024 * 1024,
        label="feasibility gate",
    )
    keys = {
        "schema",
        "status",
        "failures",
        "projection_seconds",
        "maximum_rss_bytes",
    }
    if type(gate) is not dict or set(gate) != keys or gate["schema"] != _GATE_SCHEMA:
        raise ValueError("feasibility gate has an invalid contract")
    failures = gate["failures"]
    projection = gate["projection_seconds"]
    rss = gate["maximum_rss_bytes"]
    if (
        type(failures) is not list
        or any(type(item) is not str for item in failures)
        or type(projection) not in {int, float}
        or isinstance(projection, bool)
        or not math.isfinite(float(projection))
        or type(rss) is not int
        or isinstance(rss, bool)
        or rss < 0
    ):
        raise ValueError("feasibility gate values are invalid")
    if gate["status"] != "pass" or failures:
        raise ValueError("feasibility gate did not pass")
    return hashlib.sha256(
        _read_bounded_regular_file(
            manifest_path, limit=1024 * 1024, label="feasibility manifest"
        )
    ).hexdigest()


def _require_unchanged_feasibility_manifest(path: Path, expected: str) -> None:
    try:
        observed = _validate_feasibility_receipt(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "feasibility artifact drifted during legacy export; "
            "feasibility manifest drifted or a manifested leaf changed"
        ) from error
    if observed != expected:
        raise RuntimeError(
            "feasibility artifact drifted during legacy export; "
            "feasibility manifest drifted or a manifested leaf changed"
        )


def _repo_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[1]


def _source_sha256() -> dict[str, str]:
    root = _repo_root()
    return {
        relative: _raw_file_sha256(root / relative)
        for relative in _SOURCE_RELATIVE_PATHS
    }


def _preflight_namespaces(
    *,
    query_path: Path,
    qrel_path: Path,
    embedding_path: Path,
    feasibility_receipt: Path,
    output_root: Path,
    compact_receipt: Path,
    scratch_root: Path,
) -> None:
    inputs = (query_path, qrel_path, embedding_path, feasibility_receipt)
    sources = tuple(_repo_root() / relative for relative in _SOURCE_RELATIVE_PATHS)
    for path in (*inputs, *sources):
        try:
            status = path.lstat()
        except OSError as error:
            raise ValueError(f"required input is missing: {path}") from error
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"required input is not a regular file: {path}")
    for path, label in (
        (output_root, "output root"),
        (compact_receipt, "compact receipt"),
    ):
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"{label} exists: {path}")

    paths = (*inputs, *sources, output_root, compact_receipt, scratch_root)
    lexical = [Path(os.path.abspath(path)) for path in paths]
    resolved = [path.resolve(strict=False) for path in paths]
    for index, left in enumerate(paths):
        for right_index in range(index + 1, len(paths)):
            right = paths[right_index]
            for left_value, right_value in (
                (lexical[index], lexical[right_index]),
                (resolved[index], resolved[right_index]),
            ):
                if (
                    left_value == right_value
                    or left_value in right_value.parents
                    or right_value in left_value.parents
                ):
                    raise ValueError(f"artifact namespaces overlap: {left} and {right}")
    feasibility_root = feasibility_receipt.parent.resolve(strict=True)
    for destination in (output_root, compact_receipt, scratch_root):
        for candidate in (
            Path(os.path.abspath(destination)),
            destination.resolve(strict=False),
        ):
            if candidate == feasibility_root or candidate.is_relative_to(
                feasibility_root
            ):
                raise ValueError(
                    f"artifact namespaces overlap: feasibility root and {destination}"
                )


def _preflight_worker_namespaces(staging: Path) -> dict[str, Path]:
    try:
        status = staging.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(status.st_mode):
            with os.scandir(staging) as entries:
                if next(entries, None) is not None:
                    raise ValueError("worker scratch root is not empty")
        raise FileExistsError(f"worker scratch root exists: {staging}")
    repeat_dirs = {repeat: staging / f"repeat-{repeat}" for repeat in ("a", "b")}
    planned = tuple(
        path
        for directory in repeat_dirs.values()
        for path in (directory, directory / "control.tsv", directory / "worker.json")
    )
    lexical = [Path(os.path.abspath(path)) for path in planned]
    resolved = [path.resolve(strict=False) for path in planned]
    if len(set(lexical)) != len(lexical) or len(set(resolved)) != len(resolved):
        raise ValueError("worker scratch namespaces overlap")
    for path in planned:
        if not path.resolve(strict=False).is_relative_to(staging.resolve(strict=False)):
            raise ValueError("worker scratch namespace escapes staging")
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise FileExistsError(f"worker scratch destination exists: {path}")
    return repeat_dirs


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_exclusive_json(path: Path, value: object) -> None:
    _write_exclusive_bytes(path, _canonical_json(value) + b"\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), destination)


def _atomic_publish_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"destination exists: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_exclusive_bytes(temporary, payload)
        _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(root: Path) -> dict[str, object]:
    rows = [
        [name, (root / name).stat().st_size, _raw_file_sha256(root / name)]
        for name in _ROOT_FILES
    ]
    value = {
        "schema": _MANIFEST_SCHEMA,
        "files": rows,
        "artifact_inventory_sha256": _framed_sha256(
            "artifact-inventory", _canonical_json(rows)
        ),
    }
    _write_exclusive_json(root / "manifest.json", value)
    return value


def _worker_command(
    *,
    repeat: str,
    query_path: Path,
    qrel_path: Path,
    embedding_path: Path,
    output_tsv: Path,
    result_json: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "_worker",
        "--queries",
        str(query_path),
        "--qrels",
        str(qrel_path),
        "--embeddings",
        str(embedding_path),
        "--output-tsv",
        str(output_tsv),
        "--result-json",
        str(result_json),
        "--repeat",
        repeat,
    ]


def _run_fresh_worker(
    *,
    repeat: str,
    query_path: Path,
    qrel_path: Path,
    embedding_path: Path,
    output_tsv: Path,
    result_json: Path,
) -> int:
    command = _worker_command(
        repeat=repeat,
        query_path=query_path,
        qrel_path=qrel_path,
        embedding_path=embedding_path,
        output_tsv=output_tsv,
        result_json=result_json,
    )
    environment = os.environ.copy()
    for name in _THREAD_VARIABLES:
        environment[name] = "1"
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"legacy control repeat {repeat} failed with return code "
            f"{completed.returncode}"
        )
    return completed.returncode


def _validate_worker_result(
    path: Path, *, output_tsv: Path
) -> tuple[dict[str, object], bytes]:
    value = _read_canonical_json(path, limit=256 * 1024, label="legacy worker result")
    tsv_payload = _read_bounded_regular_file(
        output_tsv, limit=32 * 1024 * 1024, label="legacy worker TSV"
    )
    if type(value) is not dict or set(value) != _WORKER_KEYS:
        raise ValueError("legacy worker result has an invalid contract")
    if value["schema"] != _WORKER_SCHEMA:
        raise ValueError("legacy worker result schema mismatch")
    if any(not _is_sha256(value[field]) for field in _WORKER_DIGEST_KEYS):
        raise ValueError("legacy worker result digest is invalid")
    source = value["source_sha256"]
    if (
        type(source) is not dict
        or tuple(source) != _SOURCE_RELATIVE_PATHS
        or any(not _is_sha256(digest) for digest in source.values())
        or source != _source_sha256()
    ):
        raise ValueError("legacy worker source attestation is invalid")
    for field, expected in (
        ("neighbor_count", 50),
        ("row_count", 48_300),
        ("query_count", 322),
        ("candidate_count", 150),
    ):
        if type(value[field]) is not int or value[field] != expected:
            raise ValueError(f"legacy worker {field} mismatch")
    for field, expected in (
        ("mean_ndcg_at_10", _EXPECTED_NDCG),
        ("mean_reciprocal_rank", _EXPECTED_MRR),
    ):
        metric = value[field]
        if type(metric) is not float or not math.isfinite(metric) or metric != expected:
            raise ValueError(f"legacy worker {field} mismatch")
    if value["tsv_sha256"] != hashlib.sha256(tsv_payload).hexdigest():
        raise ValueError("legacy worker TSV attestation mismatch")
    return dict(value), tsv_payload


def _repeat_receipt(
    result: dict[str, object],
    *,
    repeat: str,
    return_code: int,
    feasibility_manifest_sha256: str,
) -> dict[str, object]:
    excluded = {"schema", "embedding_file_sha256"}
    return {
        "schema": _REPEAT_SCHEMA,
        "repeat": repeat,
        "return_code": return_code,
        "feasibility_manifest_sha256": feasibility_manifest_sha256,
        **{key: value for key, value in result.items() if key not in excluded},
    }


def _coordinate(
    *,
    query_path: Path,
    qrel_path: Path,
    embedding_path: Path,
    feasibility_receipt: Path,
    output_root: Path,
    compact_receipt: Path,
) -> int:
    query_path = Path(query_path)
    qrel_path = Path(qrel_path)
    embedding_path = Path(embedding_path)
    feasibility_receipt = Path(feasibility_receipt)
    output_root = Path(output_root)
    compact_receipt = Path(compact_receipt)

    feasibility_sha256 = _validate_feasibility_receipt(feasibility_receipt)
    query_path = query_path.resolve(strict=True)
    qrel_path = qrel_path.resolve(strict=True)
    embedding_path = embedding_path.resolve(strict=True)
    feasibility_receipt = feasibility_receipt.resolve(strict=True)
    output_root = Path(os.path.abspath(output_root))
    compact_receipt = Path(os.path.abspath(compact_receipt))
    staging = output_root.parent / f".{output_root.name}.legacy-control-scratch"
    _preflight_namespaces(
        query_path=query_path,
        qrel_path=qrel_path,
        embedding_path=embedding_path,
        feasibility_receipt=feasibility_receipt,
        output_root=output_root,
        compact_receipt=compact_receipt,
        scratch_root=staging,
    )
    repeat_dirs = _preflight_worker_namespaces(staging)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    compact_receipt.parent.mkdir(parents=True, exist_ok=True)
    published = False
    staging_created = False
    try:
        staging.mkdir()
        staging_created = True
        for directory in repeat_dirs.values():
            directory.mkdir()
        results: dict[str, dict[str, object]] = {}
        return_codes: dict[str, int] = {}
        tsv_payloads: dict[str, bytes] = {}
        for repeat in ("a", "b"):
            _require_unchanged_feasibility_manifest(
                feasibility_receipt, feasibility_sha256
            )
            directory = repeat_dirs[repeat]
            output_tsv = directory / "control.tsv"
            result_json = directory / "worker.json"
            if output_tsv.exists() or result_json.exists():
                raise FileExistsError("repeat destination exists")
            return_codes[repeat] = _run_fresh_worker(
                repeat=repeat,
                query_path=query_path,
                qrel_path=qrel_path,
                embedding_path=embedding_path,
                output_tsv=output_tsv,
                result_json=result_json,
            )
            _require_unchanged_feasibility_manifest(
                feasibility_receipt, feasibility_sha256
            )
            results[repeat], tsv_payloads[repeat] = _validate_worker_result(
                result_json, output_tsv=output_tsv
            )
        if tsv_payloads["a"] != tsv_payloads["b"]:
            raise RuntimeError("legacy control repeat TSV bytes differ")
        if results["a"] != results["b"]:
            raise RuntimeError("legacy control worker attestations differ")

        _write_exclusive_bytes(staging / _TSV_FILENAME, tsv_payloads["a"])
        for directory in repeat_dirs.values():
            shutil.rmtree(directory)
        repeat_receipts = {
            repeat: _repeat_receipt(
                results[repeat],
                repeat=repeat,
                return_code=return_codes[repeat],
                feasibility_manifest_sha256=feasibility_sha256,
            )
            for repeat in ("a", "b")
        }
        for repeat in ("a", "b"):
            _write_exclusive_json(
                staging / f"repeat_{repeat}.json", repeat_receipts[repeat]
            )

        from trec_mlm.legacy_dense_control import (
            _parse_legacy_tsv,
            _trec_bytes_from_rankings,
            validate_legacy_control_lock_bindings,
        )

        rankings, _candidate_ids, row_count = _parse_legacy_tsv(staging / _TSV_FILENAME)
        _write_exclusive_bytes(
            staging / _TREC_FILENAME, _trec_bytes_from_rankings(rankings)
        )
        common = results["a"]
        lock = {
            "schema": _LOCK_SCHEMA,
            "feasibility_manifest_sha256": feasibility_sha256,
            "source_sha256": common["source_sha256"],
            "query_file_sha256": common["query_file_sha256"],
            "qrel_file_sha256": common["qrel_file_sha256"],
            "query_input_sha256": common["query_input_sha256"],
            "candidate_inventory_sha256": common["candidate_inventory_sha256"],
            "embedding_sha256": common["embedding_sha256"],
            "similarity_sha256": common["similarity_sha256"],
            "neighbor_count": common["neighbor_count"],
            "tsv_filename": _TSV_FILENAME,
            "tsv_sha256": common["tsv_sha256"],
            "trec_filename": _TREC_FILENAME,
            "trec_sha256": _raw_file_sha256(staging / _TREC_FILENAME),
            "run_id": _RUN_ID,
            "row_count": row_count,
            "query_count": common["query_count"],
            "candidate_count": common["candidate_count"],
            "mean_ndcg_at_10": common["mean_ndcg_at_10"],
            "mean_reciprocal_rank": common["mean_reciprocal_rank"],
            "repeat_a_receipt_sha256": _raw_file_sha256(staging / "repeat_a.json"),
            "repeat_b_receipt_sha256": _raw_file_sha256(staging / "repeat_b.json"),
        }
        _write_exclusive_json(staging / _LOCK_FILENAME, lock)
        expected_lock_sha256 = _raw_file_sha256(staging / _LOCK_FILENAME)
        _write_manifest(staging)
        for path in staging.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        validate_legacy_control_lock_bindings(
            staging / _LOCK_FILENAME,
            expected_lock_sha256=expected_lock_sha256,
            expected_tsv_sha256=str(common["tsv_sha256"]),
        )
        _require_unchanged_feasibility_manifest(feasibility_receipt, feasibility_sha256)
        _fsync_directory(staging)
        _rename_noreplace(staging, output_root)
        published = True
        _fsync_directory(output_root.parent)

        validation = validate_legacy_control_lock_bindings(
            output_root / _LOCK_FILENAME,
            expected_lock_sha256=expected_lock_sha256,
            expected_tsv_sha256=str(common["tsv_sha256"]),
        )
        compact = {
            "schema": _COMPACT_SCHEMA,
            "feasibility_manifest_sha256": feasibility_sha256,
            "legacy_control_manifest_sha256": _raw_file_sha256(
                output_root / "manifest.json"
            ),
            "legacy_control_lock_sha256": expected_lock_sha256,
            "legacy_control_tsv_sha256": validation.tsv_sha256,
            "legacy_control_trec_sha256": _raw_file_sha256(
                output_root / _TREC_FILENAME
            ),
            "row_count": validation.row_count,
            "query_count": 322,
            "candidate_count": 150,
            "mean_ndcg_at_10": validation.mean_ndcg_at_10,
            "mean_reciprocal_rank": validation.mean_reciprocal_rank,
        }
        _atomic_publish_bytes(compact_receipt, _canonical_json(compact) + b"\n")
        return 0
    finally:
        if staging_created and not published:
            shutil.rmtree(staging, ignore_errors=True)


def _run_private_worker(arguments: argparse.Namespace) -> int:
    output_tsv = Path(arguments.output_tsv)
    result_json = Path(arguments.result_json)
    if arguments.repeat not in {"a", "b"}:
        raise ValueError("private worker repeat must be a or b")
    if output_tsv.exists() or result_json.exists():
        raise FileExistsError("private worker destination exists")
    if output_tsv.parent != result_json.parent or not output_tsv.parent.is_dir():
        raise ValueError("private worker destinations must share an existing directory")

    from trec_mlm.legacy_dense_control import (
        _export_fixed_mpnet_k50_control_attested,
    )

    result = _export_fixed_mpnet_k50_control_attested(
        query_path=Path(arguments.queries),
        qrel_path=Path(arguments.qrels),
        embedding_path=Path(arguments.embeddings),
        output_tsv=output_tsv,
    )
    value = {
        "schema": _WORKER_SCHEMA,
        "source_sha256": result.source_sha256,
        "query_file_sha256": result.query_file_sha256,
        "qrel_file_sha256": result.qrel_file_sha256,
        "query_input_sha256": result.query_input_sha256,
        "candidate_inventory_sha256": result.candidate_inventory_sha256,
        "embedding_file_sha256": result.embedding_file_sha256,
        "embedding_sha256": result.embedding_sha256,
        "similarity_sha256": result.similarity_sha256,
        "neighbor_count": result.neighbor_count,
        "tsv_sha256": result.validation.tsv_sha256,
        "row_count": result.validation.row_count,
        "query_count": result.query_count,
        "candidate_count": result.candidate_count,
        "mean_ndcg_at_10": result.validation.mean_ndcg_at_10,
        "mean_reciprocal_rank": result.validation.mean_reciprocal_rank,
    }
    _write_exclusive_json(result_json, value)
    return 0


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--repeat", required=True, choices=("a", "b"))
    return parser


def _coordinator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--feasibility-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--compact-receipt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the private worker or public legacy-control coordinator."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["_worker"]:
        return _run_private_worker(_worker_parser().parse_args(arguments[1:]))
    parsed = _coordinator_parser().parse_args(arguments)
    return _coordinate(
        query_path=Path(parsed.queries),
        qrel_path=Path(parsed.qrels),
        embedding_path=Path(parsed.embeddings),
        feasibility_receipt=Path(parsed.feasibility_receipt),
        output_root=Path(parsed.output_root),
        compact_receipt=Path(parsed.compact_receipt),
    )


if __name__ == "__main__":
    raise SystemExit(main())
