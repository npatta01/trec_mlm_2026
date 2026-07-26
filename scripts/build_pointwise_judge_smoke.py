#!/usr/bin/env python3
"""Build the private, identity-blind pointwise judge smoke sample."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from trec_mlm.pointwise_sampling import (
    DIAGNOSTIC_REASONS,
    iter_discovery_responses,
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json_atomically(path: Path, payload: object, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    references = load_grounded_references(args.grounded)
    cases, sealed = select_diagnostic_cases(
        references,
        iter_discovery_responses(args.discovery),
        seed=args.seed,
    )
    manifest_body: dict[str, Any] = {
        "version": 1,
        "seed": args.seed,
        "sources": {
            "discovery": {"sha256": _sha256_file(args.discovery)},
            "grounded": {"sha256": _sha256_file(args.grounded)},
        },
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
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("output already exists with an invalid smoke manifest") from error
        if existing.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("output already exists with a different manifest hash")
        if not sealed_path.exists():
            raise ValueError("output already exists without sealed candidates")
        sealed_path.chmod(0o600)
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
