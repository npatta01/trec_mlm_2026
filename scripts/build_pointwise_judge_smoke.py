#!/usr/bin/env python3
"""Build the private, identity-blind pointwise judge smoke sample."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence

from trec_mlm.pointwise_sampling import (
    DIAGNOSTIC_REASONS,
    iter_discovery_responses,
    load_discovery_questions,
    load_grounded_references,
    select_diagnostic_cases,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--grounded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json_atomically(path: Path, payload: object, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in existing artifact: {key}")
        result[key] = value
    return result


def _validate_existing_output(
    manifest_path: Path,
    sealed_path: Path,
    manifest: dict[str, Any],
    sealed: dict[str, str],
) -> None:
    existing = _load_json_object(manifest_path, "existing smoke manifest")
    existing_body = dict(existing)
    stored_hash = existing_body.pop("manifest_hash", None)
    if type(stored_hash) is not str or stored_hash != _manifest_hash(existing_body):
        raise ValueError("invalid existing manifest hash")
    if stored_hash != manifest["manifest_hash"]:
        raise ValueError("output already exists with a different manifest hash")
    if not sealed_path.exists():
        raise ValueError("output already exists without sealed candidates")
    existing_sealed = _load_json_object(sealed_path, "existing sealed candidates")
    if existing_sealed != sealed:
        raise ValueError("sealed candidates do not match regenerated selection")
    if stat.S_IMODE(sealed_path.stat().st_mode) != 0o600:
        raise ValueError("sealed candidates must have mode 0600")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_hashes = {
        "discovery": _sha256_file(args.discovery),
        "grounded": _sha256_file(args.grounded),
    }
    questions = load_discovery_questions(args.discovery)
    references = load_grounded_references(args.grounded, questions=questions)
    cases, sealed = select_diagnostic_cases(
        references,
        iter_discovery_responses(args.discovery),
        seed=args.seed,
    )
    if source_hashes != {
        "discovery": _sha256_file(args.discovery),
        "grounded": _sha256_file(args.grounded),
    }:
        raise ValueError("source changed during selection")
    manifest_body: dict[str, Any] = {
        "version": 1,
        "seed": args.seed,
        "sources": {name: {"sha256": digest} for name, digest in source_hashes.items()},
        "selection_reasons": list(DIAGNOSTIC_REASONS),
        "cases": [asdict(case) for case in cases],
    }
    manifest = {**manifest_body, "manifest_hash": _manifest_hash(manifest_body)}

    output_dir: Path = args.output_dir
    manifest_path = output_dir / "smoke_manifest.json"
    sealed_path = output_dir / "sealed_candidates.json"
    if output_dir.exists():
        if not output_dir.is_dir() or not manifest_path.exists():
            raise ValueError("output already exists without a smoke manifest")
        _validate_existing_output(manifest_path, sealed_path, manifest, sealed)
        return 0

    output_dir.mkdir(parents=True)
    try:
        _write_json_atomically(sealed_path, sealed, mode=0o600)
        _write_json_atomically(manifest_path, manifest, mode=0o600)
    except BaseException:
        sealed_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
