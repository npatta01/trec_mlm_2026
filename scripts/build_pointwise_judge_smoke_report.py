#!/usr/bin/env python3
"""Build private pointwise smoke reports and a whitelist-sanitized summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence

from trec_mlm.pointwise_judge import CandidateCase
from trec_mlm.pointwise_report import (
    build_smoke_summary,
    render_private_report,
    sanitize_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--sanitized-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = _load_artifact_dir(args.artifact_dir)
    summary = build_smoke_summary(**inputs)
    _write_private(args.artifact_dir / "smoke-summary.json", summary)
    _write_private_text(
        args.artifact_dir / "smoke-review.md", render_private_report(summary)
    )
    _write_public(args.sanitized_output, sanitize_summary(summary))
    return 0


def _load_artifact_dir(directory: Path) -> dict[str, object]:
    if not directory.is_dir() or stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise ValueError("artifact directory must be a private mode-0700 directory")
    manifest = _read_private(directory / "smoke_manifest.json")
    rows = manifest.get("cases")
    if type(rows) is not list:
        raise ValueError("smoke manifest cases must be a list")
    cases = [
        CandidateCase(**{**row, "strata": tuple(row["strata"])})
        for row in rows
        if type(row) is dict
    ]
    if len(cases) != len(rows):
        raise ValueError("smoke manifest case must be an object")
    sealed = _read_private(directory / "reviews" / "sealed-judge-aliases.json")
    aliases = sealed.get("judge_aliases")
    if type(aliases) is not dict:
        raise ValueError("sealed judge aliases must be an object")
    runs = {
        alias: _read_private(directory / "runs" / f"{alias}.json")
        for alias in aliases.values()
        if type(alias) is str
    }
    if len(runs) != len(aliases):
        raise ValueError("sealed judge alias must be a string")
    receipts = {
        alias: _read_private(directory / f"runtime-{alias}.json")
        for alias in runs
        if (directory / f"runtime-{alias}.json").exists()
    }
    return {
        "cases": cases,
        "runs": runs,
        "advisor_review": _read_private(directory / "reviews" / "advisor.json"),
        "claude_review": _read_private(directory / "reviews" / "claude.json"),
        "sealed_judge_aliases": aliases,
        "runtime_receipts": receipts,
        "manifest_hash": manifest.get("manifest_hash"),
    }


def _read_private(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError("private report input must be a mode-0600 regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("private report input is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("private report input must be an object")
    return value


def _write_private(path: Path, value: object) -> None:
    _write_private_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _write_private_text(path: Path, text: str) -> None:
    if path.exists():
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError("private report output must be a mode-0600 regular file")
    _atomic_write(path, text.encode("utf-8"), mode=0o600)


def _write_public(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        path,
        (
            json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        ).encode("utf-8"),
        mode=0o644,
    )


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
