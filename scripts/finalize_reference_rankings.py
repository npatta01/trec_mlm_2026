#!/usr/bin/env python3
"""Unblind locked four-model reviews and calculate ranking summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from trec_mlm.reference_adjudication import (
    aggregate_rankings,
    apply_review_corrections,
    unblind_review,
    validate_review_coverage,
    validate_sealed_map,
)
from trec_mlm.reference_data import read_jsonl, write_jsonl_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRECTIONS = (
    ROOT / "protocols/trec_2026_reference_review_corrections.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--sealed-map", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--corrections",
        type=Path,
        default=DEFAULT_CORRECTIONS,
        help="Drift-checked corrections applied to the immutable blind ledger.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet = read_jsonl(args.packet)
    reviews = read_jsonl(args.reviews)
    expected_ids = [str(row["question_id"]) for row in packet]
    correction_payload = json.loads(args.corrections.read_text(encoding="utf-8"))
    if (
        not isinstance(correction_payload, Mapping)
        or correction_payload.get("schema_version")
        != "reference-review-corrections-v1"
        or not isinstance(correction_payload.get("corrections"), list)
    ):
        raise ValueError("invalid reference review correction file")
    reviews = apply_review_corrections(
        reviews, correction_payload["corrections"]
    )
    validate_review_coverage(expected_ids, reviews)
    sealed = json.loads(args.sealed_map.read_text(encoding="utf-8"))
    if not isinstance(sealed, Mapping):
        raise ValueError("sealed map must be an object")
    maps = validate_sealed_map(sealed, expected_ids)
    rankings = [
        unblind_review(review, maps[str(review["question_id"])])
        for review in reviews
    ]
    write_jsonl_atomic(args.rankings, rankings)
    summary = aggregate_rankings(rankings)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.summary)
    print(json.dumps({"questions": len(rankings), "summary": str(args.summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
