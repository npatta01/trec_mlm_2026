#!/usr/bin/env python3
"""Inspect unreviewed blind packets and append validated Codex decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from trec_mlm.reference_adjudication import merge_reviews, validate_review
from trec_mlm.reference_data import read_jsonl, write_jsonl_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("next", "status"):
        item = subparsers.add_parser(command)
        item.add_argument("--packet", type=Path, required=True)
        item.add_argument("--reviews", type=Path, required=True)
        if command == "next":
            item.add_argument("--count", type=int, default=20)
    record = subparsers.add_parser("record")
    record.add_argument("--packet", type=Path, required=True)
    record.add_argument("--reviews", type=Path, required=True)
    record.add_argument("--decision-file", type=Path, required=True)
    record.add_argument("--replace", action="store_true")
    return parser


def _existing(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet = read_jsonl(args.packet)
    existing = _existing(args.reviews)
    reviewed_ids = {str(row["question_id"]) for row in existing}
    if args.command == "status":
        print(
            json.dumps(
                {
                    "total": len(packet),
                    "reviewed": len(reviewed_ids),
                    "remaining": len(packet) - len(reviewed_ids),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "next":
        if args.count < 1:
            raise ValueError("count must be at least one")
        pending = [
            row for row in packet if str(row["question_id"]) not in reviewed_ids
        ][: args.count]
        for row in pending:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return 0

    additions = read_jsonl(args.decision_file)
    packet_ids = {str(row["question_id"]) for row in packet}
    for review in additions:
        validate_review(review)
        if str(review["question_id"]) not in packet_ids:
            raise ValueError(f"decision ID absent from packet: {review['question_id']}")
    merged = merge_reviews(existing, additions, replace=args.replace)
    write_jsonl_atomic(args.reviews, merged)
    print(json.dumps({"recorded": len(additions), "reviewed": len(merged)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
