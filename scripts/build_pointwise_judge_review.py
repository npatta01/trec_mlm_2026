#!/usr/bin/env python3
"""Build private blind two-pass pointwise review packets."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence

from scripts.run_local_pointwise_judge import (
    _load_cases,
    _load_json_from_opened_file,
)
from trec_mlm.pointwise_review import (
    build_critique_packet,
    build_independent_review_packet,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    packets = subparsers.add_parser("packets")
    packets.add_argument("--manifest", type=Path, required=True)
    packets.add_argument("--manifest-sha256", required=True)
    packets.add_argument("--runs-dir", type=Path, required=True)
    packets.add_argument("--run-sha256", action="append", default=[], required=True)
    packets.add_argument("--forbidden-identity", action="append", default=[])
    packets.add_argument("--output-dir", type=Path, required=True)
    packets.add_argument("--seed", type=int, default=2026)
    return parser


def _parse_run_hashes(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        alias, separator, digest = value.partition("=")
        if (
            not separator
            or not alias
            or "/" in alias
            or alias in {".", ".."}
            or alias in result
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("run SHA-256 anchors must be unique alias=sha256 values")
        result[alias] = digest
    return result


def _private_output_dir(path: Path) -> None:
    if path.exists():
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) != 0o700:
            raise ValueError(
                "review output directory must be an exact mode-0700 directory"
            )
        return
    path.mkdir(parents=False, mode=0o700)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise ValueError("review output directory must have mode 0700")


def _encoded_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _publish_private(path: Path, payload: object) -> None:
    content = _encoded_json(payload)
    if path.exists():
        _require_identical_private(path, content)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _require_identical_private(path, content)
        else:
            temporary.unlink()
            _require_identical_private(path, content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_identical_private(path: Path, content: bytes) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOENT}:
            raise ValueError("review output could not be opened safely") from error
        raise
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size != len(content)
        ):
            raise ValueError("existing review output is not a private regular file")
        chunks: list[bytes] = []
        remaining = len(content) + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if b"".join(chunks) != content:
            raise ValueError("existing review output differs; refusing overwrite")
    finally:
        os.close(descriptor)


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    manifest = _load_json_from_opened_file(
        args.manifest,
        "smoke manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    anchors = _parse_run_hashes(args.run_sha256)
    runs: dict[str, dict[str, Any]] = {}
    for alias in sorted(anchors):
        runs[alias] = _load_json_from_opened_file(
            args.runs_dir / f"{alias}.json",
            f"run {alias}",
            expected_sha256=anchors[alias],
            private=True,
        )
    return cases, runs


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases, runs = _load_inputs(args)
    independent = build_independent_review_packet(cases)
    critique, sealed = build_critique_packet(
        cases,
        runs,
        seed=args.seed,
        forbidden_identity_tokens=args.forbidden_identity,
    )

    _private_output_dir(args.output_dir)
    _publish_private(args.output_dir / "independent-review-packet.json", independent)
    _publish_private(args.output_dir / "critique-review-packet.json", critique)
    _publish_private(args.output_dir / "sealed-judge-aliases.json", sealed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
