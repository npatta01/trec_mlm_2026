#!/usr/bin/env python3
"""Build the private, frozen 100-case pointwise judge manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence

from scripts.build_pointwise_judge_smoke import (
    _load_json_object,
    _manifest_hash,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--grounded", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _validate_existing_output(
    manifest_path: Path,
    sealed_path: Path,
    manifest: dict[str, Any],
    sealed: dict[str, str],
) -> None:
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o600:
        raise ValueError("100-case manifest must have mode 0600")
    if not sealed_path.exists():
        raise ValueError("output already exists without sealed candidates")
    if stat.S_IMODE(sealed_path.stat().st_mode) != 0o600:
        raise ValueError("sealed candidates must have mode 0600")
    existing = _load_json_object(manifest_path, "existing 100-case manifest")
    existing_body = dict(existing)
    stored_hash = existing_body.pop("manifest_hash", None)
    if type(stored_hash) is not str or stored_hash != _manifest_hash(existing_body):
        raise ValueError("invalid existing manifest hash")
    if stored_hash != manifest["manifest_hash"]:
        raise ValueError("output already exists with a different manifest hash")
    existing_sealed = _load_json_object(sealed_path, "existing sealed candidates")
    if existing_sealed != sealed:
        raise ValueError("sealed candidates do not match regenerated selection")


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
        cases, sealed = select_diagnostic_case_set(
            references,
            iter_discovery_responses(discovery),
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
            "reused_smoke_response_ids": [case.response_id for case in smoke_cases],
            "cases": [asdict(case) for case in cases],
        }
        manifest = {**manifest_body, "manifest_hash": _manifest_hash(manifest_body)}

        output_dir: Path = args.output_dir
        manifest_path = output_dir / "manifest.json"
        sealed_path = output_dir / "sealed_candidates.json"
        if output_dir.exists():
            if not output_dir.is_dir() or not manifest_path.exists():
                raise ValueError("output already exists without a 100-case manifest")
            _validate_existing_output(manifest_path, sealed_path, manifest, sealed)
            return 0

        output_dir.mkdir(parents=True, mode=0o700)
        output_dir.chmod(0o700)
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
