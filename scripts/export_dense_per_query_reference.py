"""Export the reviewed Baseline 2 per-query metrics as a compact control."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/dense_query_knn_baseline/per_query.tsv"
DEFAULT_OUTPUT = ROOT / "reports/trec_2026_dense_query_knn_per_query.tsv"
EXPECTED_QUERY_COUNT = 322
EXPECTED_NDCG_AT_10 = 0.35985474409504775
EXPECTED_MRR = 0.40406330811168817


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    with args.source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "query_id",
            "dense_ndcg_at_10",
            "dense_reciprocal_rank",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"dense source is missing columns: {sorted(missing)!r}")
        for row in reader:
            rows.append({name: row[name] for name in sorted(required)})

    rows.sort(key=lambda row: row["query_id"])
    query_ids = [row["query_id"] for row in rows]
    if len(rows) != EXPECTED_QUERY_COUNT or len(set(query_ids)) != len(rows):
        raise ValueError("dense source must contain 322 unique development queries")
    ndcg = statistics.fmean(float(row["dense_ndcg_at_10"]) for row in rows)
    mrr = statistics.fmean(float(row["dense_reciprocal_rank"]) for row in rows)
    if abs(ndcg - EXPECTED_NDCG_AT_10) > 5e-12:
        raise ValueError(f"dense nDCG@10 mismatch: {ndcg!r}")
    if abs(mrr - EXPECTED_MRR) > 5e-12:
        raise ValueError(f"dense MRR mismatch: {mrr!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "query_id",
        "dense_ndcg_at_10",
        "dense_reciprocal_rank",
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
