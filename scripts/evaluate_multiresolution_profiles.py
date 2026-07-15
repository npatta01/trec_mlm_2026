#!/usr/bin/env python3
"""Evaluate frozen issue-18 rankings after the qrel-free viability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trec_mlm.profile_multiresolution import load_multiresolution_artifact
from trec_mlm.profile_multiresolution_evaluation import (
    evaluate_multiresolution,
    validate_viability_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="frozen issue-18 artifact")
    parser.add_argument("--qrels", required=True, help="sealed complete qrels")
    parser.add_argument(
        "--viability-receipt",
        required=True,
        help="qrel-free receipt authorizing diagnostic evaluation",
    )
    parser.add_argument("--output", required=True, help="new evaluation JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_multiresolution_artifact(args.artifact)
    try:
        receipt = json.loads(Path(args.viability_receipt).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid viability receipt") from error
    manifest = json.loads(
        (Path(args.artifact) / "manifest.json").read_text(encoding="utf-8")
    )
    validate_viability_receipt(receipt, manifest.get("aggregate_sha256"))
    payload = evaluate_multiresolution(
        args.artifact, args.viability_receipt, args.qrels, args.output
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
