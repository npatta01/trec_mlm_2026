#!/usr/bin/env python3
"""Construct, validate, or compare frozen raw-response refinement artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from trec_mlm.raw_response_refinement import (
    RawRefinementConfig,
    compare_raw_response_refinement_repeats,
    construct_raw_response_refinement,
    load_raw_response_refinement_artifact,
    write_raw_response_refinement_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Run with the locked numerical dependencies, for example: "
            "uv run --extra dense python scripts/run_raw_response_refinement.py ..."
        ),
    )
    operations = parser.add_subparsers(dest="operation", required=True)
    construct = operations.add_parser("construct", help="build one frozen artifact")
    construct.add_argument("--inference", required=True)
    construct.add_argument("--profiles", required=True)
    construct.add_argument("--dense", required=True)
    construct.add_argument("--output-dir", required=True)
    validate = operations.add_parser("validate", help="strictly reload an artifact")
    validate.add_argument("--construction", required=True)
    compare = operations.add_parser("compare", help="compare two fresh constructions")
    compare.add_argument("--first", required=True)
    compare.add_argument("--second", required=True)
    compare.add_argument("--receipt", required=True)
    return parser


def _write_json_noreplace(path: str | Path, payload: object) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "construct":
            result = construct_raw_response_refinement(
                RawRefinementConfig(
                    inference_path=args.inference,
                    construction_path=args.profiles,
                    dense_artifact_path=args.dense,
                )
            )
            payload: object = write_raw_response_refinement_artifact(
                result, args.output_dir
            )
        elif args.operation == "validate":
            artifact = load_raw_response_refinement_artifact(args.construction)
            payload = {
                "status": "passed",
                "aggregate_sha256": artifact.manifest["aggregate_sha256"],
                "query_count": len(artifact.query_ids),
                "candidate_count": len(artifact.candidate_ids),
            }
        else:
            payload = compare_raw_response_refinement_repeats(
                args.first, args.second
            )
            _write_json_noreplace(args.receipt, payload)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
