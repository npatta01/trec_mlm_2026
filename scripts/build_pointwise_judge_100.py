#!/usr/bin/env python3
"""Build the private, frozen 100-case pointwise judge manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence

from scripts.build_pointwise_judge_smoke import (
    _load_json_object,
    _manifest_hash,
    _object_without_duplicate_keys,
    _snapshot_source,
    _write_json_atomically,
)
from scripts.run_local_pointwise_judge import _load_cases
from trec_mlm.pointwise_sampling import (
    DIAGNOSTIC_REASONS,
    iter_discovery_responses,
    load_discovery_questions,
    load_grounded_references,
    select_diagnostic_case_set,
)


_CASES_PER_REASON = 10
_MAX_EXISTING_BYTES = 4 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--grounded", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _validate_existing_output(
    output_dir: Path,
    manifest_path: Path,
    sealed_path: Path,
    manifest: dict[str, Any],
    sealed: dict[str, str],
) -> None:
    output_status = output_dir.stat(follow_symlinks=False)
    if not stat.S_ISDIR(output_status.st_mode):
        raise ValueError("output path must be a private directory")
    if stat.S_IMODE(output_status.st_mode) != 0o700:
        raise ValueError("output directory must have mode 0700")
    existing = _load_existing_private_json(manifest_path, "existing 100-case manifest")
    existing_body = dict(existing)
    stored_hash = existing_body.pop("manifest_hash", None)
    if type(stored_hash) is not str or stored_hash != _manifest_hash(existing_body):
        raise ValueError("invalid existing manifest hash")
    if stored_hash != manifest["manifest_hash"]:
        raise ValueError("output already exists with a different manifest hash")
    existing_sealed = _load_existing_private_json(
        sealed_path, "existing sealed candidates"
    )
    if existing_sealed != sealed:
        raise ValueError("sealed candidates do not match regenerated selection")


def _load_existing_private_json(path: Path, description: str) -> dict[str, Any]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} could not be opened safely") from error
    try:
        os.set_inheritable(descriptor, False)
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode) or opened_before.st_nlink != 1:
            raise ValueError(f"{description} must be a single-link regular file")
        if stat.S_IMODE(opened_before.st_mode) != 0o600:
            raise ValueError(f"{description} must have mode 0600")
        if opened_before.st_size > _MAX_EXISTING_BYTES:
            raise ValueError(f"{description} is too large")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_EXISTING_BYTES:
            chunk = os.read(descriptor, min(65536, _MAX_EXISTING_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_EXISTING_BYTES:
            raise ValueError(f"{description} is too large")
        opened_after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{description} changed during read") from error
        identity_before = (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_size,
            opened_before.st_mtime_ns,
            opened_before.st_ctime_ns,
        )
        identity_after = (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
            opened_after.st_ctime_ns,
        )
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(path_after.st_mode)
            or path_after.st_nlink != 1
            or identity_before != identity_after
            or identity_after != path_identity
        ):
            raise ValueError(f"{description} changed during read")
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON") from error
    finally:
        os.close(descriptor)
    if type(value) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return value


def _serialized_json(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _publish_output(
    output_dir: Path, manifest: dict[str, Any], sealed: dict[str, str]
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    staging.chmod(0o700)
    manifest_path = staging / "manifest.json"
    sealed_path = staging / "sealed_candidates.json"
    published = False
    try:
        _write_json_atomically(sealed_path, sealed, mode=0o600)
        _write_json_atomically(manifest_path, manifest, mode=0o600)
        _fsync_directory(staging)
        os.replace(staging, output_dir)
        published = True
        _fsync_directory(output_dir.parent)
    finally:
        if not published:
            sealed_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            try:
                staging.rmdir()
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_smoke_sources(
    smoke_manifest: dict[str, Any], source_hashes: dict[str, str]
) -> None:
    expected = {
        "discovery": {"sha256": source_hashes["discovery"]},
        "grounded": {"sha256": source_hashes["grounded"]},
    }
    if smoke_manifest.get("sources") != expected:
        raise ValueError("smoke manifest sources do not match current inputs")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="pointwise-100-sources-") as temporary:
        snapshot_dir = Path(temporary)
        snapshot_dir.chmod(0o700)
        discovery, discovery_hash = _snapshot_source(
            args.discovery, snapshot_dir, "discovery"
        )
        grounded, grounded_hash = _snapshot_source(
            args.grounded, snapshot_dir, "grounded"
        )
        smoke_snapshot, smoke_hash = _snapshot_source(
            args.smoke_manifest, snapshot_dir, "smoke-manifest"
        )
        source_hashes = {
            "discovery": discovery_hash,
            "grounded": grounded_hash,
            "smoke_manifest": smoke_hash,
        }
        smoke_manifest = _load_json_object(smoke_snapshot, "smoke manifest")
        smoke_cases = _load_cases(smoke_manifest)
        if smoke_manifest["version"] != 1:
            raise ValueError("required smoke manifest must have version 1")
        _validate_smoke_sources(smoke_manifest, source_hashes)

        questions = load_discovery_questions(discovery)
        references = load_grounded_references(grounded, questions=questions)
        excluded_candidate_ids: set[str] = set()
        reused_response_ids = [case.response_id for case in smoke_cases]
        while True:
            cases, sealed = select_diagnostic_case_set(
                references,
                (
                    response
                    for response in iter_discovery_responses(discovery)
                    if response.candidate_id not in excluded_candidate_ids
                ),
                cases_per_reason=_CASES_PER_REASON,
                required_cases=smoke_cases,
                seed=args.seed,
            )
            manifest_body: dict[str, Any] = {
                "version": 2,
                "seed": args.seed,
                "sources": {
                    name: {"sha256": digest} for name, digest in source_hashes.items()
                },
                "selection_reasons": list(DIAGNOSTIC_REASONS),
                "cases_per_reason": _CASES_PER_REASON,
                "reused_smoke_response_ids": reused_response_ids,
                "cases": [asdict(case) for case in cases],
            }
            manifest = {
                **manifest_body,
                "manifest_hash": _manifest_hash(manifest_body),
            }
            manifest_bytes = _serialized_json(manifest)
            leaking_candidate_ids = {
                candidate_id
                for candidate_id in sealed.values()
                if candidate_id.encode("utf-8") in manifest_bytes
            }
            if not leaking_candidate_ids:
                break
            required_candidate_ids = {
                sealed[response_id] for response_id in reused_response_ids
            }
            if leaking_candidate_ids & required_candidate_ids:
                raise ValueError(
                    "required smoke candidate identity occurs in manifest content"
                )
            excluded_candidate_ids.update(leaking_candidate_ids)

        output_dir: Path = args.output_dir
        manifest_path = output_dir / "manifest.json"
        sealed_path = output_dir / "sealed_candidates.json"
        try:
            output_dir.stat(follow_symlinks=False)
        except FileNotFoundError:
            _publish_output(output_dir, manifest, sealed)
        else:
            _validate_existing_output(
                output_dir, manifest_path, sealed_path, manifest, sealed
            )
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
