#!/usr/bin/env python3
"""Run and adjudicate the sealed Claude Opus pointwise prompt-review gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from trec_mlm.pointwise_opus_review import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_CAP_USD,
    OPUS_MODEL_ALIAS,
    ClaudeCliReviewCall,
    OpusReviewCall,
    SpendBudget,
    _sha256_json,
    adjudicate_opus_review,
    build_review_receipt,
    load_api_key,
    parse_opus_review,
    read_private_json,
    validate_opus_review_packet,
    write_private_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", "--review-output", dest="output", type=Path)
    parser.add_argument("--review-input", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--adjudication-output", type=Path)
    parser.add_argument("--adjudicate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--transport", choices=("claude-cli", "openrouter"), default="claude-cli"
    )
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
        if args.output is None:
            raise ValueError("--output is required for a dry run")
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
    if args.adjudicate_only:
        _require_adjudication_args(args)
        review, review_receipt = _load_review_receipt(
            args.review_input, packet=packet, protocol_dir=args.protocol_dir
        )
        decisions = read_private_json(args.decisions)
        receipt = adjudicate_opus_review(
            packet=packet, review=review, decisions=decisions
        )
        write_private_json(args.adjudication_output, receipt)
        return 0
    if args.output is None:
        raise ValueError("--output/--review-output is required for a live review")
    if (args.decisions is None) != (args.adjudication_output is None):
        raise ValueError(
            "--decisions and --adjudication-output must be supplied together"
        )
    budget = SpendBudget(args.cap_usd)
    if args.transport == "claude-cli":
        review, provider = ClaudeCliReviewCall(
            budget=budget, timeout_seconds=args.timeout_seconds
        )(packet, protocol_dir=args.protocol_dir)
    else:
        review, provider = OpusReviewCall(
            api_key=load_api_key(args.api_key_file),
            budget=budget,
            timeout_seconds=args.timeout_seconds,
        )(packet, protocol_dir=args.protocol_dir)
    write_private_json(
        args.output,
        build_review_receipt(packet=packet, review=review, provider=provider),
    )
    if args.decisions is not None and args.adjudication_output is not None:
        receipt = adjudicate_opus_review(
            packet=packet,
            review=review,
            decisions=read_private_json(args.decisions),
        )
        write_private_json(args.adjudication_output, receipt)
    return 0


def _require_adjudication_args(args: argparse.Namespace) -> None:
    if (
        args.review_input is None
        or args.decisions is None
        or args.adjudication_output is None
    ):
        raise ValueError(
            "--adjudicate-only requires --review-input, --decisions, and "
            "--adjudication-output"
        )


def _load_review_receipt(
    path: Path | None,
    *,
    packet: dict[str, object],
    protocol_dir: Path,
) -> tuple[object, dict[str, object]]:
    if path is None:
        raise ValueError("review receipt input is required")
    receipt = read_private_json(path)
    if set(receipt) != {
        "version",
        "packet_sha256",
        "model_alias",
        "provider",
        "review",
        "review_sha256",
    }:
        raise ValueError("review receipt has invalid exact keys")
    body = {key: receipt[key] for key in receipt if key != "review_sha256"}
    if receipt["review_sha256"] != _sha256_json(body):
        raise ValueError("review receipt hash is invalid")
    if receipt["packet_sha256"] != packet["packet_sha256"]:
        raise ValueError("review receipt packet identity is invalid")
    review = parse_opus_review(
        json.dumps(receipt["review"], ensure_ascii=False),
        packet=packet,
        protocol_dir=protocol_dir,
    )
    # Rebuild the receipt through the allowlisted provider/cost validator so a
    # preexisting artifact cannot smuggle arbitrary usage or credential text
    # into the adjudication boundary.
    provider = receipt["provider"]
    if type(provider) is not dict:
        raise ValueError("review receipt provider is invalid")
    provider_for_validation = {**provider, "model_alias": receipt["model_alias"]}
    if (
        build_review_receipt(
            packet=packet, review=review, provider=provider_for_validation
        )
        != receipt
    ):
        raise ValueError("review receipt provider or hash binding is invalid")
    return review, receipt


if __name__ == "__main__":
    raise SystemExit(main())
