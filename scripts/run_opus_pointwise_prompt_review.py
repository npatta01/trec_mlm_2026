#!/usr/bin/env python3
"""Run or preflight the sealed Claude Opus pointwise prompt-review gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from trec_mlm.pointwise_opus_review import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_CAP_USD,
    OPUS_MODEL_ALIAS,
    OpusReviewCall,
    SpendBudget,
    build_review_receipt,
    load_api_key,
    read_private_json,
    validate_opus_review_packet,
    write_private_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet = validate_opus_review_packet(
        read_private_json(args.packet), protocol_dir=args.protocol_dir
    )
    if args.dry_run:
        write_private_json(
            args.output,
            {
                "version": 1,
                "mode": "dry-run",
                "packet_sha256": packet["packet_sha256"],
                "model_alias": OPUS_MODEL_ALIAS,
            },
        )
        return 0
    budget = SpendBudget(args.cap_usd)
    review, provider = OpusReviewCall(
        api_key=load_api_key(args.api_key_file),
        budget=budget,
        timeout_seconds=args.timeout_seconds,
    )(packet, protocol_dir=args.protocol_dir)
    write_private_json(
        args.output,
        build_review_receipt(packet=packet, review=review, provider=provider),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
