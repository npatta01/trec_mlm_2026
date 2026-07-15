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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256


_MANIFEST_KEYS = {"schema", "files", "artifact_inventory_sha256"}
_MANIFEST_SCHEMA = "trec-mlm-dense-aggregation-manifest-v1"
_QUARANTINE_NAME = re.compile(r".+\.aborted-[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_BYTE_LIMIT = 16 * 1024 * 1024


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def preflight_atomic_roots(*, cache_dir: Path, output_dir: Path) -> None:
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    cache_root = cache_path.resolve()
    output_root = output_path.resolve()
    if (
        cache_root == output_root
        or cache_root in output_root.parents
        or output_root in cache_root.parents
    ):
        raise ValueError("cache and output destinations overlap")
    for destination_kind, destination_path, destination_root in (
        ("cache", cache_path, cache_root),
        ("output", output_path, output_root),
    ):
        for candidate in (destination_path, destination_root):
            try:
                candidate.lstat()
            except FileNotFoundError:
                continue
            raise FileExistsError(
                f"{destination_kind} destination exists: {destination_path}"
            )

    cache_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if cache_root.parent.stat().st_dev != output_root.parent.stat().st_dev:
        raise OSError("cache and output must share one filesystem")


@contextmanager
def atomic_artifact_directory(
    destination: str | Path,
    *,
    quarantine_on_error: bool = False,
    post_publish_validator: Callable[[Path], object] | None = None,
) -> Iterator[Path]:
    final = Path(destination)
    try:
        final.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"artifact destination exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    staging_status = staging.lstat()
    owned_identity = (staging_status.st_dev, staging_status.st_ino)
    published = False

    def is_owned(path: Path) -> bool:
        try:
            status = path.lstat()
        except OSError:
            return False
        return (
            stat.S_ISDIR(status.st_mode)
            and (
                status.st_dev,
                status.st_ino,
            )
            == owned_identity
        )

    def quarantine_owned(path: Path, primary_error: BaseException) -> None:
        if not is_owned(path):
            return
        quarantine = final.parent / f"{final.name}.aborted-{uuid.uuid4().hex}"
        try:
            rename_noreplace(path, quarantine)
            if not is_owned(quarantine):
                try:
                    rename_noreplace(quarantine, path)
                except BaseException as restore_error:
                    primary_error.add_note(
                        "artifact pathname ownership changed during quarantine; "
                        "foreign-path restoration also failed: "
                        f"{type(restore_error).__name__}: {restore_error}"
                    )
                else:
                    primary_error.add_note(
                        "artifact pathname ownership changed during quarantine; "
                        "the replacement path was restored"
                    )
                return
            _fsync_directory(final.parent)
        except BaseException as quarantine_error:
            primary_error.add_note(
                "owned artifact quarantine failed: "
                f"{type(quarantine_error).__name__}: {quarantine_error}"
            )

    try:
        yield staging
        _fsync_regular_files(staging)
        write_manifest(staging)
        validate_manifest(staging)
        rename_noreplace(staging, final)
        published = True
        if not is_owned(final):
            raise RuntimeError("published artifact identity changed")
        _fsync_directory(final.parent)
        if post_publish_validator is not None:
            post_publish_validator(final)
        if not is_owned(final):
            raise RuntimeError("published artifact identity changed during validation")
    except BaseException as error:
        owned_path = final if published else staging
        if quarantine_on_error:
            quarantine_owned(owned_path, error)
        elif not published and is_owned(staging):
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
        byte_size, digest = _stable_regular_file_evidence(path)
        rows.append([relative, byte_size, digest])
    return rows


def _stable_regular_file_evidence(path: Path) -> tuple[int, str]:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"artifact file is missing: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"artifact entry is not a regular file: {path}")
    digest = hashlib.sha256()
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(f"artifact file changed while opening: {path}")
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"artifact file changed while hashing: {path}")
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1) != b"":
            raise ValueError(f"artifact file grew while hashing: {path}")
        after_handle = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"artifact file could not be hashed: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as error:
        raise ValueError(f"artifact file changed while hashing: {path}") from error
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after_handle, field)
        or getattr(before, field) != getattr(after_path, field)
        for field in stable_fields
    ):
        raise ValueError(f"artifact file changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


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
    manifest_bytes = canonical_json(payload) + b"\n"
    if len(manifest_bytes) > _MANIFEST_BYTE_LIMIT:
        raise ValueError("artifact manifest exceeds its byte limit")
    with (artifact_root / "manifest.json").open("xb") as handle:
        handle.write(manifest_bytes)
        handle.flush()
        os.fsync(handle.fileno())
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
    paths: list[str] = []
    for index, row in enumerate(value["files"]):
        if type(row) is not list or len(row) != 3:
            raise ValueError(f"artifact manifest row {index} has an invalid type")
        relative_path, byte_size, raw_sha256 = row
        if type(relative_path) is not str:
            raise ValueError(f"artifact manifest row {index} path has an invalid type")
        canonical_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or canonical_path.is_absolute()
            or canonical_path.as_posix() != relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
            or relative_path == "manifest.json"
        ):
            raise ValueError(f"artifact manifest row {index} path is invalid")
        if type(byte_size) is not int:
            raise ValueError(f"artifact manifest row {index} size has an invalid type")
        if byte_size < 0:
            raise ValueError(f"artifact manifest row {index} size is invalid")
        if type(raw_sha256) is not str:
            raise ValueError(
                f"artifact manifest row {index} digest has an invalid type"
            )
        if _SHA256.fullmatch(raw_sha256) is None:
            raise ValueError(f"artifact manifest row {index} digest is invalid")
        paths.append(relative_path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("artifact manifest inventory paths are not sorted and unique")
    if _SHA256.fullmatch(value["artifact_inventory_sha256"]) is None:
        raise ValueError("artifact manifest digest is invalid")
    return value


def _read_bounded_manifest(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"artifact manifest is missing: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(f"artifact symlink is forbidden: {path}")
        raise ValueError(f"artifact manifest is not a regular file: {path}")
    if before.st_size > _MANIFEST_BYTE_LIMIT:
        raise ValueError("artifact manifest exceeds its byte limit")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError("artifact manifest changed while opening")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError("artifact manifest changed while it was read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1) != b"":
            raise ValueError("artifact manifest grew while it was read")
        payload = b"".join(chunks)
        after_handle = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"artifact manifest is missing or invalid: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as error:
        raise ValueError("artifact manifest changed while it was read") from error
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if len(payload) != before.st_size or any(
        getattr(before, field) != getattr(after_handle, field)
        or getattr(before, field) != getattr(after_path, field)
        for field in stable_fields
    ):
        raise ValueError("artifact manifest changed while it was read")
    return payload


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
        manifest_bytes = _read_bounded_manifest(manifest_path)
        manifest_value: Any = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
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
