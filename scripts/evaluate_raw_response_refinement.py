#!/usr/bin/env python3
"""Evaluate one sealed raw-response refinement artifact after repeat validation."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from trec_mlm.raw_response_refinement_evaluation import (
    RawRefinementEvaluationConfig,
    evaluate_raw_response_refinement,
)


def _rename_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), str(destination)
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Run with the locked numerical dependencies: uv run --extra dense "
            "python scripts/evaluate_raw_response_refinement.py ..."
        ),
    )
    parser.add_argument("--construction", required=True)
    parser.add_argument("--repeat-receipt", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _write_evaluation(payload: object, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        output = staging / "evaluation.json"
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "evaluation.json"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = evaluate_raw_response_refinement(
        RawRefinementEvaluationConfig(
            artifact_path=args.construction,
            repeat_receipt_path=args.repeat_receipt,
            qrels_path=args.qrels,
        )
    )
    if payload.get("status") == "invalid":
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    try:
        output = _write_evaluation(payload, args.output_dir)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
