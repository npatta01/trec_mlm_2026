#!/usr/bin/env python3
"""Build the private identity-blind review packet for the pointwise v2 contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from trec_mlm.pointwise_production import build_opus_prompt_review_packet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_opus_prompt_review_packet(
        protocol_dir=args.protocol_dir,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
