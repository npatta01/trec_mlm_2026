from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256


_MANIFEST_KEYS = {"schema", "files", "artifact_inventory_sha256"}
_MANIFEST_SCHEMA = "trec-mlm-dense-aggregation-manifest-v1"
_QUARANTINE_NAME = re.compile(r".+\.aborted-[0-9a-f]{32}\Z")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def atomic_artifact_directory(
    destination: str | Path, *, quarantine_on_error: bool = False
) -> Iterator[Path]:
    final = Path(destination)
    if final.exists():
        raise FileExistsError(f"artifact destination exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        yield staging
        _fsync_regular_files(staging)
        write_manifest(staging)
        validate_manifest(staging)
        rename_noreplace(staging, final)
        _fsync_directory(final.parent)
    except BaseException:
        if quarantine_on_error and staging.exists():
            quarantine = final.parent / f"{final.name}.aborted-{uuid.uuid4().hex}"
            rename_noreplace(staging, quarantine)
            _fsync_directory(final.parent)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def rename_noreplace(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("authoritative artifact publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source_path), -100, os.fsencode(destination_path), 1
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination_path)


def raw_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_rows(root: Path) -> list[list[str | int]]:
    if root.is_symlink():
        raise ValueError(f"artifact symlink is forbidden: {root}")
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    rows: list[list[str | int]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact entry is not a regular file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        rows.append([relative, path.stat().st_size, raw_file_sha256(path)])
    return rows


def _write_json_fsync(path: Path, value: object) -> None:
    payload = canonical_json(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_regular_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {path}")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())


def write_manifest(root: str | Path) -> dict[str, object]:
    artifact_root = Path(root)
    rows = _inventory_rows(artifact_root)
    payload: dict[str, object] = {
        "schema": _MANIFEST_SCHEMA,
        "files": rows,
        "artifact_inventory_sha256": framed_sha256(
            "artifact-inventory", canonical_json(rows)
        ),
    }
    _write_json_fsync(artifact_root / "manifest.json", payload)
    return payload


def _validate_manifest_types(value: Any) -> dict[str, object]:
    if type(value) is not dict or set(value) != _MANIFEST_KEYS:
        raise ValueError("artifact manifest has an invalid top-level contract")
    if type(value["schema"]) is not str:
        raise ValueError("artifact manifest schema has an invalid type")
    if type(value["files"]) is not list:
        raise ValueError("artifact manifest files have an invalid type")
    if type(value["artifact_inventory_sha256"]) is not str:
        raise ValueError("artifact manifest digest has an invalid type")
    for index, row in enumerate(value["files"]):
        if type(row) is not list or len(row) != 3:
            raise ValueError(f"artifact manifest row {index} has an invalid type")
        relative_path, byte_size, raw_sha256 = row
        if type(relative_path) is not str:
            raise ValueError(
                f"artifact manifest row {index} path has an invalid type"
            )
        if type(byte_size) is not int:
            raise ValueError(
                f"artifact manifest row {index} size has an invalid type"
            )
        if type(raw_sha256) is not str:
            raise ValueError(
                f"artifact manifest row {index} digest has an invalid type"
            )
    return value


def validate_manifest(root: str | Path) -> dict[str, object]:
    artifact_root = Path(root)
    if _QUARANTINE_NAME.fullmatch(artifact_root.name):
        raise ValueError(f"quarantined artifact is not authoritative: {artifact_root}")
    if artifact_root.is_symlink():
        raise ValueError(f"artifact symlink is forbidden: {artifact_root}")
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {artifact_root}")

    manifest_path = artifact_root / "manifest.json"
    try:
        manifest_stat = manifest_path.lstat()
    except OSError as error:
        raise ValueError(f"artifact manifest is missing: {manifest_path}") from error
    if stat.S_ISLNK(manifest_stat.st_mode):
        raise ValueError(f"artifact symlink is forbidden: {manifest_path}")
    if not stat.S_ISREG(manifest_stat.st_mode):
        raise ValueError(f"artifact manifest is not a regular file: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_value: Any = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"artifact manifest is missing or invalid: {manifest_path}"
        ) from error
    manifest_value = _validate_manifest_types(manifest_value)
    try:
        expected_manifest_bytes = canonical_json(manifest_value) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("artifact manifest is not canonical JSON") from error
    if manifest_bytes != expected_manifest_bytes:
        raise ValueError("artifact manifest bytes are not canonical")
    if manifest_value["schema"] != _MANIFEST_SCHEMA:
        raise ValueError("artifact manifest has an unsupported schema")

    rows = _inventory_rows(artifact_root)
    if manifest_value["files"] != rows:
        raise ValueError("artifact manifest inventory does not match files")
    expected_digest = framed_sha256("artifact-inventory", canonical_json(rows))
    if manifest_value["artifact_inventory_sha256"] != expected_digest:
        raise ValueError("artifact manifest digest does not match inventory")
    return manifest_value


def semantic_artifact_sha256(
    root: str | Path, *, normalized_files: Mapping[str, bytes], tag: str
) -> str:
    manifest = validate_manifest(root)
    rows = manifest["files"]
    if not isinstance(rows, list):
        raise ValueError("artifact manifest inventory is invalid")
    manifested_paths = {row[0] for row in rows}
    normalized_paths = set(normalized_files)
    if not normalized_paths <= manifested_paths:
        raise ValueError(
            "normalized_files keys must be a subset of manifested regular files"
        )

    semantic_rows: list[list[str]] = []
    for path, _size, raw_sha256 in sorted(rows, key=lambda row: row[0]):
        semantic_sha256 = raw_sha256
        if path in normalized_files:
            semantic_sha256 = hashlib.sha256(normalized_files[path]).hexdigest()
        semantic_rows.append([path, semantic_sha256])
    return framed_sha256(tag, canonical_json(semantic_rows))
