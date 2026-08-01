#!/usr/bin/env python3
"""Run the frozen, zero-call TREC 2026 pointwise Phase 0 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trec_mlm.pointwise_ranking_evaluation import run_pointwise_phase0_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen pointwise rankings against development qrels without "
            "API, judge, embedding, pairwise, or pointwise-regeneration calls."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--refinement-root", type=Path, required=True)
    parser.add_argument("--development-queries", type=Path, default=Path("data/dev/llm_dev_data.tsv"))
    parser.add_argument("--qrels", type=Path, default=Path("data/dev/llm_dev_qrels.txt"))
    parser.add_argument(
        "--dense-rankings",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dense-run",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dense-result-receipt",
        type=Path,
        default=Path("reports/trec_2026_dense_query_knn_results.json"),
    )
    parser.add_argument(
        "--dense-per-query-receipt",
        type=Path,
        default=Path("reports/trec_2026_dense_query_knn_per_query.tsv"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/pointwise_ranking_phase0"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense_rankings = args.dense_rankings or (
        args.refinement_root / "rankings/dense_reference.json"
    )
    dense_run = args.dense_run or (args.refinement_root / "runs/dense_reference.trec")
    receipt = run_pointwise_phase0_evaluation(
        labels_path=args.labels,
        records_path=args.records,
        refinement_root=args.refinement_root,
        development_queries_path=args.development_queries,
        qrels_path=args.qrels,
        dense_rankings_path=dense_rankings,
        dense_run_path=dense_run,
        dense_result_receipt_path=args.dense_result_receipt,
        dense_per_query_receipt_path=args.dense_per_query_receipt,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
