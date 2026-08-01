#!/usr/bin/env python3
"""Independently validate and finalize the frozen pointwise Phase 0 result."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from trec_mlm.pointwise_ranking_validation import (
    ValidationError,
    validate_phase0_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--check",
        type=Path,
        help="require the tracked validated result to equal independently recreated bytes",
    )
    destination.add_argument(
        "--output",
        type=Path,
        help="write the independently finalized result JSON",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("data/dev/llm_dev_qrels.txt"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/pointwise_ranking_phase0"),
    )
    parser.add_argument(
        "--dense-results",
        type=Path,
        default=Path("reports/trec_2026_dense_query_knn_results.json"),
    )
    parser.add_argument(
        "--dense-per-query",
        type=Path,
        default=Path("reports/trec_2026_dense_query_knn_per_query.tsv"),
    )
    parser.add_argument(
        "--tracked-per-query",
        type=Path,
        default=Path("reports/trec_2026_pointwise_ranking_phase0_per_query.tsv"),
    )
    parser.add_argument(
        "--tracked-candidate-slices",
        type=Path,
        default=Path(
            "reports/trec_2026_pointwise_ranking_phase0_candidate_slices.tsv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finalized = validate_phase0_bundle(
        qrels_path=args.qrels,
        artifact_dir=args.artifact_dir,
        dense_results_path=args.dense_results,
        dense_per_query_path=args.dense_per_query,
        tracked_per_query_path=args.tracked_per_query,
        tracked_candidate_slices_path=args.tracked_candidate_slices,
    )
    digest = hashlib.sha256(finalized).hexdigest()
    if args.check is not None:
        if args.check.read_bytes() != finalized:
            raise ValidationError(
                f"tracked result differs from independently recreated bytes: {args.check}"
            )
        print(f"independent validation passed: SHA-256 {digest}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(finalized)
    print(f"wrote independently validated result: SHA-256 {digest}")


if __name__ == "__main__":
    main()
