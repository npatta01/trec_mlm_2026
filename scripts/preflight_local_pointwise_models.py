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
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Protocol
import urllib.error
import urllib.request

from trec_mlm.pointwise_judge import decision_to_dict, parse_judge_decision
from trec_mlm.pointwise_runner import (
    validate_local_model_registry,
    validate_loopback_endpoint,
    validate_runtime_receipt,
)


GIB = 1024**3
MEMORY_HEADROOM_BYTES = 8 * GIB
MIN_FREE_DISK_BYTES = 8 * GIB
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

    def runtime_configuration(self) -> dict[str, object]: ...

    def request_json(
        self, method: str, url: str, payload: object | None
    ) -> dict[str, object]: ...


class SystemDependencies:
    """Read-only host observations used by the CLI."""

    def __init__(self, configuration: Mapping[str, object]) -> None:
        self._configuration = dict(configuration)

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
        path = path.resolve(strict=True)
        if path.is_file():
            return {
                "path": str(path),
                "revision": "not-applicable",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        if not path.is_dir():
            raise ValueError("checkpoint path must be a file or snapshot directory")
        revision = checkpoint.get("revision")
        if type(revision) is not str or path.name != revision:
            raise ValueError("checkpoint snapshot path does not match pinned revision")
        files: list[dict[str, object]] = []
        for candidate in sorted(path.rglob("*")):
            if candidate.is_dir():
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError("checkpoint tree contains a non-file entry")
            files.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "size_bytes": resolved.stat().st_size,
                    "sha256": _sha256_file(resolved),
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

    def runtime_configuration(self) -> dict[str, object]:
        return dict(self._configuration)

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
        raw = os.read(descriptor, MAX_JSON_BYTES + 1)
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
    if len(raw) > MAX_JSON_BYTES or identity_before != identity_after:
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
    download_occurred: bool,
    startup_seconds: float,
    schema: Mapping[str, object] | None = None,
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
        visible_memory = dependencies.available_memory_bytes()
        free_disk = dependencies.available_disk_bytes(checkpoint_path)
        if (
            type(visible_memory) is not int
            or visible_memory < expected_bytes + MEMORY_HEADROOM_BYTES
        ):
            raise ValueError("insufficient visible memory")
        if type(free_disk) is not int or free_disk < MIN_FREE_DISK_BYTES:
            raise ValueError("insufficient free disk")
        if type(download_occurred) is not bool:
            raise ValueError("download observation must be boolean")
        if download_occurred and not selected["new_download_authorized"]:
            raise ValueError("unauthorized download")
        if type(startup_seconds) not in {int, float} or startup_seconds < 0:
            raise ValueError("startup seconds is invalid")
        observed_configuration = dependencies.runtime_configuration()
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
            schema={} if schema is None else schema,
        )
        parsed_canary = decision_to_dict(parse_judge_decision(canary_raw))
        body: dict[str, object] = {
            "version": 1,
            "passed": True,
            "model_alias": model_alias,
            "model_id": model_id,
            "registry_sha256": registry_sha256,
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
            "startup_seconds": startup_seconds,
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
    if path.parent.exists():
        mode = stat.S_IMODE(path.parent.stat().st_mode)
        if not path.parent.is_dir() or mode != 0o700:
            raise ValueError("runtime receipt parent must have mode 0700")
    else:
        path.parent.mkdir(parents=True, mode=0o700)
        path.parent.chmod(0o700)
    encoded = (
        json.dumps(
            receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    if path.exists():
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError("existing runtime receipt must have mode 0600")
        if path.read_bytes() == encoded:
            return
        raise ValueError("refusing to overwrite a different runtime receipt")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


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
            raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-seconds", type=float, required=True)
    parser.add_argument("--download-occurred", action="store_true")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--max-concurrency", type=int, default=1)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: PreflightDependencies | None = None,
) -> int:
    args = _parser().parse_args(argv)
    registry = load_registry(args.registry, expected_sha256=args.registry_sha256)
    schema = load_registry(args.protocol_dir / "decision-v1.schema.json")
    configuration = {
        "context_tokens": args.context_tokens,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_concurrency": args.max_concurrency,
    }
    receipt = run_preflight(
        registry,
        model_alias=args.model_alias,
        base_url=args.base_url,
        registry_sha256=args.registry_sha256,
        checkpoint_path=args.checkpoint_path,
        dependencies=(
            SystemDependencies(configuration) if dependencies is None else dependencies
        ),
        download_occurred=args.download_occurred,
        startup_seconds=args.startup_seconds,
        schema=schema,
    )
    if receipt.get("passed") is not True:
        return 1
    write_runtime_receipt(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
