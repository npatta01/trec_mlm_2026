"""Command-line interface for baseline experiments and run validation."""

from __future__ import annotations

import argparse
import json

from trec_mlm.experiment import (
    ExperimentConfig,
    run_leave_one_out_experiment,
    write_experiment_artifacts,
)
from trec_mlm.metrics import evaluate_rankings
from trec_mlm.query_knn import load_complete_qrels, load_development_data
from trec_mlm.runio import rankings_from_run, validate_trec_run


def _neighbor_grid(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "neighbor grid must be comma-separated integers"
        ) from error
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("neighbor grid values must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trec-mlm")
    commands = parser.add_subparsers(dest="command", required=True)

    experiment = commands.add_parser("query-knn", help="run strict LOO query-kNN")
    experiment.add_argument("--queries", default="data/dev/llm_dev_data.tsv")
    experiment.add_argument("--qrels", default="data/dev/llm_dev_qrels.txt")
    experiment.add_argument(
        "--neighbors-grid",
        type=_neighbor_grid,
        default=(1, 3, 5, 10, 20, 50, 100),
    )
    experiment.add_argument("--primary-neighbors", type=int, default=10)
    experiment.add_argument(
        "--primary-aggregation",
        choices=("weighted_mean", "weighted_sum", "max"),
        default="weighted_mean",
    )
    experiment.add_argument("--run-id", default="qknn_tfidf_w12_k10_loo")
    experiment.add_argument("--bootstrap-resamples", type=int, default=10_000)
    experiment.add_argument("--seed", type=int, default=2026)
    experiment.add_argument(
        "--primary-selection-provenance",
        default="not_recorded",
        help="declared selection history, e.g. predeclared_before_current_run",
    )
    experiment.add_argument("--output-dir", default="artifacts/query_knn_baseline")

    validate = commands.add_parser("validate-run", help="validate a TREC run")
    validate.add_argument("run")
    validate.add_argument("--queries", default="data/dev/llm_dev_data.tsv")
    validate.add_argument("--qrels", default="data/dev/llm_dev_qrels.txt")

    score = commands.add_parser("score-run", help="score nDCG@10 and MRR")
    score.add_argument("--qrels", required=True)
    score.add_argument("--run", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "query-knn":
        config = ExperimentConfig(
            query_path=args.queries,
            qrels_path=args.qrels,
            neighbor_counts=args.neighbors_grid,
            primary_neighbors=args.primary_neighbors,
            primary_aggregation=args.primary_aggregation,
            run_id=args.run_id,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            primary_selection_provenance=args.primary_selection_provenance,
        )
        result = run_leave_one_out_experiment(config)
        paths = write_experiment_artifacts(result, args.output_dir)
        print(json.dumps({"summary": result.summary, "artifacts": paths}, indent=2))
        return 0

    if args.command == "validate-run":
        expected_queries = None
        expected_models = None
        data = load_development_data(args.queries, args.qrels)
        expected_queries = tuple(data.queries)
        expected_models = data.models
        print(
            json.dumps(
                validate_trec_run(
                    args.run,
                    expected_query_ids=expected_queries,
                    expected_model_ids=expected_models,
                ),
                indent=2,
            )
        )
        return 0

    qrels = load_complete_qrels(args.qrels)
    validate_trec_run(
        args.run,
        expected_query_ids=tuple(qrels),
        expected_model_ids=tuple(next(iter(qrels.values()))),
    )
    rankings = rankings_from_run(args.run)
    result = evaluate_rankings(rankings, qrels, k=10)
    print(
        json.dumps(
            {
                "ndcg_at_10": result.macro_ndcg_at_k,
                "mrr": result.mean_reciprocal_rank,
                "query_count": len(result.per_query),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
