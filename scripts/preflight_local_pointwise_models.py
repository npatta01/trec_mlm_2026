#!/usr/bin/env python3
"""Fail-closed preflight for one frozen local pointwise-judge model.

For a multi-file Hugging Face snapshot, ``checkpoint.sha256`` is a tree
commitment.  Each regular file is represented by its snapshot-relative POSIX
path, exact byte size, and SHA-256.  Records are sorted by path and encoded as
canonical JSON (sorted keys, compact separators, UTF-8); the commitment is the
SHA-256 of those bytes and ``expected_bytes`` is the sum of record sizes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
from typing import Any, Protocol
import urllib.error
import urllib.request

from trec_mlm.pointwise_judge import (
    decision_to_dict,
    load_prompt_contract,
    parse_judge_decision,
)
from trec_mlm.pointwise_runner import (
    FROZEN_DECISION_SCHEMA_SHA256,
    validate_local_model_registry,
    validate_loopback_endpoint,
    validate_runtime_receipt,
)


GIB = 1024**3
MEMORY_HEADROOM_BYTES = 8 * GIB
MIN_AVAILABLE_MEMORY_BYTES = 100 * GIB
MIN_FREE_DISK_BYTES = 50 * GIB
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_HTTP_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourcePins:
    lemonade_models_path: Path
    lemonade_models_sha256: str
    backend_versions_path: Path
    backend_versions_sha256: str
    lemonade_version: str
    llamacpp_backend_version: str
    vllm_backend_version: str


FROZEN_SOURCE_PINS = SourcePins(
    lemonade_models_path=Path(
        "/usr/share/lemonade-server/resources/server_models.json"
    ),
    lemonade_models_sha256=(
        "da2a602ad6fa21ea0cd16017076937dea4aaf0391b32eef08229a457aff21747"
    ),
    backend_versions_path=Path(
        "/usr/share/lemonade-server/resources/backend_versions.json"
    ),
    backend_versions_sha256=(
        "bb40c70709fdf9065123d24c8ed2a8f05e0d16a5e8e08cc69ba6a7d381df75b6"
    ),
    lemonade_version="11.5.0",
    llamacpp_backend_version="b9752",
    vllm_backend_version="vllm0.20.1-rocm7.12.0",
)


class PreflightDependencies(Protocol):
    def run_command(self, argv: tuple[str, ...]) -> str: ...

    def read_bytes(self, path: Path) -> bytes: ...

    def inspect_checkpoint(
        self, path: Path, checkpoint: dict[str, object]
    ) -> dict[str, object]: ...

    def available_memory_bytes(self) -> int: ...

    def available_disk_bytes(self, path: Path) -> int: ...

    def process_observation(self, pid: int) -> dict[str, object]: ...

    def request_json(
        self, method: str, url: str, payload: object | None
    ) -> dict[str, object]: ...


class SystemDependencies:
    """Read-only host observations used by the CLI."""

    def run_command(self, argv: tuple[str, ...]) -> str:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            },
        )
        return completed.stdout + completed.stderr

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def inspect_checkpoint(
        self, path: Path, checkpoint: dict[str, object]
    ) -> dict[str, object]:
        path = path.absolute()
        path_mode = path.lstat().st_mode
        if stat.S_ISREG(path_mode) or stat.S_ISLNK(path_mode):
            blob_root = _snapshot_blob_root(path) if stat.S_ISLNK(path_mode) else None
            size_bytes, digest = _secure_hash_logical_file(path, blob_root=blob_root)
            return {
                "path": str(path),
                "revision": "not-applicable",
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        if not stat.S_ISDIR(path_mode):
            raise ValueError("checkpoint path must be a file or snapshot directory")
        revision = checkpoint.get("revision")
        if type(revision) is not str or path.name != revision:
            raise ValueError("checkpoint snapshot path does not match pinned revision")
        blob_root = _snapshot_blob_root(path / "placeholder")
        files: list[dict[str, object]] = []
        for candidate in sorted(path.rglob("*")):
            candidate_mode = candidate.lstat().st_mode
            if stat.S_ISDIR(candidate_mode):
                continue
            size_bytes, digest = _secure_hash_logical_file(
                candidate, blob_root=blob_root
            )
            files.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "size_bytes": size_bytes,
                    "sha256": digest,
                }
            )
        size_bytes, tree_sha256 = checkpoint_tree_commitment(files)
        return {
            "path": str(path),
            "revision": revision,
            "size_bytes": size_bytes,
            "sha256": tree_sha256,
        }

    def available_memory_bytes(self) -> int:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
        raise ValueError("available memory could not be observed")

    def available_disk_bytes(self, path: Path) -> int:
        probe = path if path.exists() else path.parent
        return shutil.disk_usage(probe).free

    def process_observation(self, pid: int) -> dict[str, object]:
        if type(pid) is not int or pid <= 0:
            raise ValueError("launch process PID is invalid")
        start_ticks = _process_start_ticks(pid)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = [item.decode("utf-8") for item in cmdline.split(b"\0") if item]
        if not argv:
            raise ValueError("launch process argv is unavailable")
        if _process_start_ticks(pid) != start_ticks:
            raise ValueError("launch process changed during observation")
        return {"start_ticks": start_ticks, "argv": argv}

    def request_json(
        self, method: str, url: str, payload: object | None
    ) -> dict[str, object]:
        encoded = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            encoded = _canonical_json(payload)
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=encoded, headers=headers, method=method
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _RejectRedirects()
        )
        try:
            with opener.open(request, timeout=30) as response:
                if response.geturl() != url:
                    raise ValueError("preflight endpoint redirected")
                raw = response.read(MAX_HTTP_BYTES + 1)
        except (TimeoutError, urllib.error.URLError) as error:
            raise ValueError("preflight endpoint request failed") from error
        if len(raw) > MAX_HTTP_BYTES:
            raise ValueError("preflight endpoint response is too large")
        return _json_object(raw, "preflight endpoint response")


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def checkpoint_tree_commitment(
    files: Sequence[Mapping[str, object]],
) -> tuple[int, str]:
    """Return exact bytes and the deterministic multi-file tree commitment."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in files:
        if type(value) is not dict or set(value) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("checkpoint tree record must contain exact keys")
        path = value["path"]
        size = value["size_bytes"]
        digest = value["sha256"]
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or path in seen
        ):
            raise ValueError("checkpoint tree path is invalid or duplicate")
        if type(size) is not int or size < 0:
            raise ValueError("checkpoint tree size is invalid")
        if not _is_sha256(digest):
            raise ValueError("checkpoint tree hash is invalid")
        seen.add(path)
        rows.append({"path": path, "size_bytes": size, "sha256": digest})
    if not rows:
        raise ValueError("checkpoint tree must contain files")
    rows.sort(key=lambda row: str(row["path"]))
    return sum(int(row["size_bytes"]) for row in rows), hashlib.sha256(
        _canonical_json(rows)
    ).hexdigest()


def _snapshot_blob_root(logical_path: Path) -> Path:
    parts = logical_path.parts
    indexes = [index for index, part in enumerate(parts) if part == "snapshots"]
    if not indexes:
        raise ValueError("cache symlink is not under a snapshots directory")
    snapshots_index = indexes[-1]
    if snapshots_index + 2 >= len(parts):
        raise ValueError("cache symlink snapshot layout is invalid")
    repository = Path(*parts[:snapshots_index])
    blob_root = repository / "blobs"
    mode = blob_root.lstat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ValueError("permitted model-cache blob root must be a real directory")
    return blob_root.resolve(strict=True)


def _secure_hash_logical_file(
    logical_path: Path, *, blob_root: Path | None
) -> tuple[int, str]:
    logical_before = logical_path.lstat()
    if stat.S_ISLNK(logical_before.st_mode):
        if blob_root is None:
            raise ValueError("cache symlink has no permitted model-cache blob root")
        target = logical_path.resolve(strict=True)
        if not target.is_relative_to(blob_root):
            raise ValueError("cache symlink target escapes permitted blob root")
    elif stat.S_ISREG(logical_before.st_mode):
        target = logical_path
    else:
        raise ValueError("checkpoint tree contains a nonregular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        target_before = os.fstat(descriptor)
        if not stat.S_ISREG(target_before.st_mode):
            raise ValueError("checkpoint target must be a regular file")
        if target_before.st_nlink != 1:
            raise ValueError("checkpoint target has hardlink ambiguity")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        target_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        logical_after = logical_path.lstat()
        target_path_after = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("checkpoint changed during hash") from error
    if (
        _stat_identity(logical_before) != _stat_identity(logical_after)
        or _stat_identity(target_before) != _stat_identity(target_after)
        or _stat_identity(target_after) != _stat_identity(target_path_after)
    ):
        raise ValueError("checkpoint changed during hash")
    return target_after.st_size, digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def load_registry(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, object]:
    """Load a bounded regular registry file and reject duplicate JSON keys."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON_BYTES:
            raise ValueError("local model registry must be a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_JSON_BYTES:
            chunk = os.read(descriptor, min(65536, MAX_JSON_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    raw = b"".join(chunks)
    try:
        path_after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("local model registry changed during read") from error
    if (
        len(raw) > MAX_JSON_BYTES
        or identity_before != identity_after
        or identity_after
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
    ):
        raise ValueError("local model registry changed during read")
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError("registry expected SHA-256 is invalid")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("registry does not match expected SHA-256")
    return _json_object(raw, "local model registry")


def run_preflight(
    registry: Mapping[str, object],
    *,
    model_alias: str,
    base_url: str,
    registry_sha256: str,
    checkpoint_path: Path,
    dependencies: PreflightDependencies,
    source_pins: SourcePins = FROZEN_SOURCE_PINS,
    schema: Mapping[str, object],
    schema_sha256: str,
    launch_evidence: Mapping[str, object],
    launch_evidence_sha256: str,
) -> dict[str, object]:
    """Return an accepted v1 receipt, or the minimal fail-closed result."""
    try:
        if not _is_sha256(registry_sha256):
            raise ValueError("registry SHA-256 is invalid")
        models = registry.get("models")
        if type(models) is not dict or model_alias not in models:
            raise ValueError("requested model alias is absent")
        untrusted = models[model_alias]
        if type(untrusted) is not dict or type(untrusted.get("model_id")) is not str:
            raise ValueError("requested model registry entry is invalid")
        model_id = untrusted["model_id"]
        selected = validate_local_model_registry(
            registry, model_alias=model_alias, model_id=model_id
        )
        endpoint = validate_loopback_endpoint(base_url)
        if (
            schema_sha256 != FROZEN_DECISION_SCHEMA_SHA256
            or _sha256_json(schema) != FROZEN_DECISION_SCHEMA_SHA256
        ):
            raise ValueError("canary schema is not the exact frozen Task 1 schema")
        launch = _validate_launch_evidence(
            launch_evidence,
            expected_sha256=launch_evidence_sha256,
            model_alias=model_alias,
            model_id=model_id,
            endpoint=endpoint,
            runtime=selected["runtime"],
            dependencies=dependencies,
        )
        _validate_source_files(dependencies, source_pins)
        runtime_version = _parse_lemonade_version(
            dependencies.run_command(("lemonade", "--version"))
        )
        if runtime_version != source_pins.lemonade_version:
            raise ValueError("wrong Lemonade runtime version")
        backend_version = _backend_version(
            dependencies.read_bytes(source_pins.backend_versions_path),
            selected["runtime"],
        )
        expected_backend = (
            source_pins.vllm_backend_version
            if selected["runtime"] == "vllm-rocm"
            else source_pins.llamacpp_backend_version
        )
        if backend_version != expected_backend:
            raise ValueError("wrong runtime backend version")
        if not re.search(
            r"(?<![A-Za-z0-9_])gfx1151(?![A-Za-z0-9_])",
            dependencies.run_command(("rocminfo",)),
        ):
            raise ValueError("gfx1151 is not visible")
        checkpoint_pin = selected["checkpoint"]
        observed_checkpoint = dependencies.inspect_checkpoint(
            checkpoint_path, dict(checkpoint_pin)
        )
        expected_bytes = checkpoint_pin["expected_bytes"]
        if (
            observed_checkpoint.get("size_bytes") != expected_bytes
            or observed_checkpoint.get("sha256") != checkpoint_pin["sha256"]
        ):
            raise ValueError("checkpoint size or commitment drift")
        revision = checkpoint_pin["revision"]
        if revision is not None and observed_checkpoint.get("revision") != revision:
            raise ValueError("checkpoint revision drift")
        filename = checkpoint_pin["gguf_filename"]
        if (
            filename is not None
            and Path(str(observed_checkpoint.get("path"))).name != filename
        ):
            raise ValueError("checkpoint filename drift")
        if not _json_values_equal(launch["checkpoint_after"], observed_checkpoint):
            raise ValueError("launch checkpoint-after evidence drift")
        _validate_model_argv(
            launch["process"]["argv"],
            runtime=selected["runtime"],
            model_id=model_id,
            checkpoint_path=str(observed_checkpoint["path"]),
        )
        download_occurred = _derive_download_observation(
            launch, observed_checkpoint, model_alias=model_alias
        )
        visible_memory = dependencies.available_memory_bytes()
        free_disk = dependencies.available_disk_bytes(checkpoint_path)
        if type(visible_memory) is not int or visible_memory < max(
            MIN_AVAILABLE_MEMORY_BYTES, expected_bytes + MEMORY_HEADROOM_BYTES
        ):
            raise ValueError("insufficient visible memory")
        if type(free_disk) is not int or free_disk < MIN_FREE_DISK_BYTES:
            raise ValueError("insufficient free disk")
        if download_occurred and not selected["new_download_authorized"]:
            raise ValueError("unauthorized download")
        observed_configuration = _configuration_from_argv(
            launch["process"]["argv"], selected["runtime"]
        )
        if not _configuration_matches(
            observed_configuration, selected["configuration"]
        ):
            raise ValueError("observed runtime configuration is not frozen")
        endpoint_models = _endpoint_models(
            dependencies.request_json("GET", f"{endpoint}/models", None)
        )
        if endpoint_models != [model_id]:
            raise ValueError("endpoint does not expose the exact requested model")
        canary_raw = _canary(
            dependencies,
            endpoint=endpoint,
            model_id=model_id,
            configuration=observed_configuration,
            schema=schema,
        )
        parsed_canary = decision_to_dict(parse_judge_decision(canary_raw))
        body: dict[str, object] = {
            "version": 1,
            "passed": True,
            "model_alias": model_alias,
            "model_id": model_id,
            "registry_sha256": registry_sha256,
            "schema_sha256": schema_sha256,
            "launch_evidence_sha256": launch_evidence_sha256,
            "runtime": {
                "name": selected["runtime"],
                "version": runtime_version,
                "backend_version": backend_version,
            },
            "hardware": {
                "gpu_arch": "gfx1151",
                "visible_memory_bytes": visible_memory,
                "free_disk_bytes": free_disk,
            },
            "checkpoint": observed_checkpoint,
            "endpoint": {"base_url": endpoint, "models": endpoint_models},
            "configuration": observed_configuration,
            "canary": {"passed": True, "raw": canary_raw, "parsed": parsed_canary},
            "startup_seconds": (
                launch["ready_monotonic_ns"] - launch["started_monotonic_ns"]
            )
            / 1_000_000_000,
            "download_occurred": download_occurred,
        }
        receipt = {**body, "receipt_sha256": _sha256_json(body)}
        validate_runtime_receipt(
            receipt,
            model_alias=model_alias,
            model_id=model_id,
            endpoint=endpoint,
            registry_sha256=registry_sha256,
            registry_entry=selected,
        )
        return receipt
    except (KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return {"passed": False}


def write_runtime_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    """Write one successful receipt atomically under exact private modes."""
    if receipt.get("passed") is not True:
        raise ValueError("only a successful runtime receipt may be written")
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, mode=0o700)
        path.parent.chmod(0o700)
    parent_lstat = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_lstat.st_mode)
        or stat.S_ISLNK(parent_lstat.st_mode)
        or stat.S_IMODE(parent_lstat.st_mode) != 0o700
    ):
        raise ValueError("runtime receipt parent must be a real directory mode 0700")
    encoded = (
        json.dumps(
            receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory = os.open(path.parent, directory_flags)
    try:
        directory_stat = os.fstat(directory)
        if (
            _stat_identity(directory_stat) != _stat_identity(parent_lstat)
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            raise ValueError("runtime receipt parent changed before write")
        temporary_name = _create_private_temporary(directory, path.name)
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            try:
                _write_all(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                    follow_symlinks=False,
                )
            except FileExistsError:
                _validate_identical_existing_receipt(directory, path.name, encoded)
            else:
                os.unlink(temporary_name, dir_fd=directory)
                temporary_name = ""
            os.fsync(directory)
        finally:
            if temporary_name:
                os.unlink(temporary_name, dir_fd=directory)
    finally:
        os.close(directory)


def _create_private_temporary(directory: int, final_name: str) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(10):
        name = f".{final_name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=directory)
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        return name
    raise ValueError("could not create an exclusive runtime receipt temporary")


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("runtime receipt write made no progress")
        offset += written


def _validate_identical_existing_receipt(
    directory: int, name: str, expected: bytes
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOENT}:
            raise ValueError(
                "existing runtime receipt is a symlink or vanished"
            ) from error
        raise
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
        ):
            raise ValueError(
                "existing runtime receipt must be a regular file mode 0600"
            )
        raw = b""
        while len(raw) <= len(expected):
            chunk = os.read(descriptor, len(expected) + 1 - len(raw))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(descriptor)
        path_after = os.stat(name, dir_fd=directory, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        after
    ) != _stat_identity(path_after):
        raise ValueError("existing runtime receipt changed during validation")
    if raw != expected:
        raise ValueError("refusing to overwrite a different runtime receipt")


_LAUNCH_EVIDENCE_KEYS = {
    "version",
    "model_alias",
    "model_id",
    "base_url",
    "runtime",
    "process",
    "started_monotonic_ns",
    "ready_monotonic_ns",
    "checkpoint_before",
    "checkpoint_after",
    "download_events",
    "evidence_sha256",
}


def _validate_launch_evidence(
    value: Mapping[str, object],
    *,
    expected_sha256: str,
    model_alias: str,
    model_id: str,
    endpoint: str,
    runtime: object,
    dependencies: PreflightDependencies,
) -> dict[str, object]:
    launch = dict(value)
    if set(launch) != _LAUNCH_EVIDENCE_KEYS:
        raise ValueError("launch evidence must contain exact version-1 keys")
    if not _is_sha256(expected_sha256) or _sha256_json(launch) != expected_sha256:
        raise ValueError("launch evidence is not externally anchored")
    body = dict(launch)
    stored_hash = body.pop("evidence_sha256")
    if not _is_sha256(stored_hash) or _sha256_json(body) != stored_hash:
        raise ValueError("launch evidence self-hash is invalid")
    if type(launch["version"]) is not int or launch["version"] != 1:
        raise ValueError("launch evidence version must be 1")
    if (
        launch["model_alias"] != model_alias
        or launch["model_id"] != model_id
        or launch["runtime"] != runtime
    ):
        raise ValueError("launch evidence model identity drift")
    if validate_loopback_endpoint(launch["base_url"]) != endpoint:
        raise ValueError("launch evidence endpoint drift")
    started = launch["started_monotonic_ns"]
    ready = launch["ready_monotonic_ns"]
    if (
        type(started) is not int
        or type(ready) is not int
        or started < 0
        or ready < started
    ):
        raise ValueError("launch evidence timestamps are invalid")
    process = launch["process"]
    if type(process) is not dict or set(process) != {"pid", "start_ticks", "argv"}:
        raise ValueError("launch evidence process identity is invalid")
    pid = process["pid"]
    start_ticks = process["start_ticks"]
    argv = process["argv"]
    if (
        type(pid) is not int
        or pid <= 0
        or type(start_ticks) is not int
        or start_ticks < 0
        or type(argv) is not list
        or not argv
        or any(type(item) is not str or not item for item in argv)
    ):
        raise ValueError("launch evidence process fields are invalid")
    observed_process = dependencies.process_observation(pid)
    if not _json_values_equal(
        observed_process, {"start_ticks": start_ticks, "argv": argv}
    ):
        raise ValueError("live launch process does not match anchored evidence")
    _checkpoint_evidence(launch["checkpoint_after"], allow_none=False)
    _checkpoint_evidence(launch["checkpoint_before"], allow_none=True)
    if type(launch["download_events"]) is not list:
        raise ValueError("launch download journal must be a list")
    return launch


def _checkpoint_evidence(value: object, *, allow_none: bool) -> None:
    if allow_none and value is None:
        return
    if type(value) is not dict or set(value) != {
        "path",
        "revision",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("launch checkpoint evidence is invalid")
    if (
        type(value["path"]) is not str
        or not value["path"]
        or type(value["revision"]) is not str
        or not value["revision"]
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] <= 0
        or not _is_sha256(value["sha256"])
    ):
        raise ValueError("launch checkpoint evidence fields are invalid")


def _derive_download_observation(
    launch: Mapping[str, object],
    checkpoint_after: Mapping[str, object],
    *,
    model_alias: str,
) -> bool:
    before = launch["checkpoint_before"]
    events = launch["download_events"]
    if before is not None:
        if not _json_values_equal(before, checkpoint_after) or events != []:
            raise ValueError(
                "checkpoint changed without an authorized download journal"
            )
        return False
    if type(events) is not list or len(events) != 1:
        raise ValueError("new checkpoint requires exactly one download event")
    event = events[0]
    if type(event) is not dict or set(event) != {
        "model_alias",
        "started_monotonic_ns",
        "completed_monotonic_ns",
        "expected_bytes",
        "sha256",
    }:
        raise ValueError("download event schema is invalid")
    started = event["started_monotonic_ns"]
    completed = event["completed_monotonic_ns"]
    if (
        event["model_alias"] != model_alias
        or type(started) is not int
        or type(completed) is not int
        or not launch["started_monotonic_ns"]
        <= started
        <= completed
        <= launch["ready_monotonic_ns"]
        or event["expected_bytes"] != checkpoint_after["size_bytes"]
        or event["sha256"] != checkpoint_after["sha256"]
    ):
        raise ValueError("download event does not bind the observed checkpoint")
    return True


def _configuration_from_argv(argv: object, runtime: object) -> dict[str, object]:
    if type(argv) is not list or any(type(item) is not str for item in argv):
        raise ValueError("runtime argv is invalid")
    if runtime == "vllm-rocm":
        context = _single_integer_flag(argv, "--max-model-len")
        concurrency = _single_integer_flag(argv, "--max-num-seqs")
    else:
        context = _single_integer_flag(argv, "--ctx-size")
        concurrency = _single_integer_flag(argv, "--parallel")
    return {
        "context_tokens": context,
        "temperature": 0,
        "max_output_tokens": 384,
        "max_concurrency": concurrency,
    }


def _validate_model_argv(
    argv: object, *, runtime: object, model_id: str, checkpoint_path: str
) -> None:
    if type(argv) is not list or any(type(item) is not str for item in argv):
        raise ValueError("runtime argv is invalid")
    if runtime == "vllm-rocm":
        serve_indexes = [index for index, item in enumerate(argv) if item == "serve"]
        if (
            len(serve_indexes) != 1
            or serve_indexes[0] + 1 >= len(argv)
            or argv[serve_indexes[0] + 1] != model_id
        ):
            raise ValueError("vLLM argv does not bind the selected model")
        return
    if _single_string_flag(argv, "--model") != checkpoint_path:
        raise ValueError("llama.cpp argv does not bind the observed checkpoint")


def _single_integer_flag(argv: list[str], name: str) -> int:
    raw = _single_string_flag(argv, name)
    if not raw.isdecimal():
        raise ValueError(f"runtime argv {name} value is invalid")
    return int(raw)


def _single_string_flag(argv: list[str], name: str) -> str:
    indexes = [index for index, item in enumerate(argv) if item == name]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        raise ValueError(f"runtime argv must contain exactly one {name}")
    return argv[indexes[0] + 1]


def _json_values_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            type(key) is str and _json_values_equal(left[key], right[key])
            for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _validate_source_files(
    dependencies: PreflightDependencies, pins: SourcePins
) -> None:
    for path, expected in (
        (pins.lemonade_models_path, pins.lemonade_models_sha256),
        (pins.backend_versions_path, pins.backend_versions_sha256),
    ):
        if hashlib.sha256(dependencies.read_bytes(path)).hexdigest() != expected:
            raise ValueError("pinned Lemonade source metadata drift")


def _backend_version(raw: bytes, runtime: object) -> str:
    value = _json_object(raw, "backend version metadata")
    if runtime == "vllm-rocm":
        backend = value.get("vllm")
        key = "rocm"
    else:
        backend = value.get("llamacpp")
        key = "rocm-stable"
    if type(backend) is not dict or type(backend.get(key)) is not str:
        raise ValueError("backend version metadata is incomplete")
    return backend[key]


def _parse_lemonade_version(output: str) -> str:
    match = re.fullmatch(r"lemonade version ([0-9]+\.[0-9]+\.[0-9]+)\s*", output)
    if match is None:
        raise ValueError("Lemonade version output is invalid")
    return match.group(1)


def _endpoint_models(value: Mapping[str, object]) -> list[str]:
    if set(value) != {"object", "data"} or value["object"] != "list":
        raise ValueError("endpoint model list schema is invalid")
    data = value["data"]
    if type(data) is not list:
        raise ValueError("endpoint model list data is invalid")
    models: list[str] = []
    for row in data:
        if type(row) is not dict or type(row.get("id")) is not str:
            raise ValueError("endpoint model list row is invalid")
        models.append(row["id"])
    return models


def _canary(
    dependencies: PreflightDependencies,
    *,
    endpoint: str,
    model_id: str,
    configuration: Mapping[str, object],
    schema: Mapping[str, object],
) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Question: What is the capital of France?\n"
                    "PROVISIONAL REFERENCE: Paris.\nCANDIDATE RESPONSE: Paris."
                ),
            }
        ],
        "temperature": configuration["temperature"],
        "max_tokens": configuration["max_output_tokens"],
        "seed": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_canary",
                "strict": True,
                "schema": schema,
            },
        },
    }
    value = dependencies.request_json("POST", f"{endpoint}/chat/completions", payload)
    if value.get("model") != model_id:
        raise ValueError("canary response model is wrong")
    choices = value.get("choices")
    if type(choices) is not list or len(choices) != 1:
        raise ValueError("canary response choices are invalid")
    choice = choices[0]
    if type(choice) is not dict or type(choice.get("message")) is not dict:
        raise ValueError("canary response message is invalid")
    raw = choice["message"].get("content")
    if type(raw) is not str or not raw:
        raise ValueError("canary response content is invalid")
    usage = value.get("usage")
    if type(usage) is not dict or any(
        type(amount) not in {int, float} or amount < 0 for amount in usage.values()
    ):
        raise ValueError("canary response usage is invalid")
    return raw


def _json_object(raw: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _process_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    command_end = raw.rfind(")")
    if command_end < 0:
        raise ValueError("launch process stat is invalid")
    fields = raw[command_end + 2 :].split()
    if len(fields) <= 19 or not fields[19].isdecimal():
        raise ValueError("launch process start time is unavailable")
    return int(fields[19])


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _configuration_matches(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    integer_keys = {"context_tokens", "max_output_tokens", "max_concurrency"}
    return (
        set(observed) == set(expected)
        and observed == expected
        and all(type(observed[key]) is int for key in integer_keys)
        and type(observed["temperature"]) in {int, float}
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--launch-evidence", type=Path, required=True)
    parser.add_argument("--launch-evidence-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: PreflightDependencies | None = None,
) -> int:
    args = _parser().parse_args(argv)
    registry = load_registry(args.registry, expected_sha256=args.registry_sha256)
    contract = load_prompt_contract(args.protocol_dir)
    schema_sha256 = _sha256_json(contract.schema)
    launch_evidence = load_registry(
        args.launch_evidence, expected_sha256=args.launch_evidence_sha256
    )
    receipt = run_preflight(
        registry,
        model_alias=args.model_alias,
        base_url=args.base_url,
        registry_sha256=args.registry_sha256,
        checkpoint_path=args.checkpoint_path,
        dependencies=SystemDependencies() if dependencies is None else dependencies,
        schema=contract.schema,
        schema_sha256=schema_sha256,
        launch_evidence=launch_evidence,
        launch_evidence_sha256=args.launch_evidence_sha256,
    )
    if receipt.get("passed") is not True:
        return 1
    write_runtime_receipt(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
