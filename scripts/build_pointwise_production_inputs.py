#!/usr/bin/env python3
"""Seal the private full-corpus pointwise production inventory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from trec_mlm.pointwise_production import build_production_inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--grounded", type=Path, required=True)
    parser.add_argument("--refinement-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_production_inventory(
        discovery=args.discovery,
        grounded=args.grounded,
        refinement_records=args.refinement_records,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
