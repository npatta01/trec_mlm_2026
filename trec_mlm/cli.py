"""Command-line interface for baseline experiments and run validation."""

from __future__ import annotations

import argparse
import json

from trec_mlm.dense_query_knn import (
    DEFAULT_ENCODERS,
    EncoderSpec,
    encoder_specs_by_key,
)
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


def _encoder_grid(value: str) -> tuple[EncoderSpec, ...]:
    available = encoder_specs_by_key()
    keys = tuple(item.strip() for item in value.split(",") if item.strip())
    if not keys:
        raise argparse.ArgumentTypeError("encoder grid cannot be empty")
    invalid = sorted(set(keys) - set(available))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown encoders {invalid!r}; choose from {sorted(available)!r}"
        )
    if len(set(keys)) != len(keys):
        raise argparse.ArgumentTypeError("encoder grid values must be unique")
    return tuple(available[key] for key in keys)


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

    dense = commands.add_parser(
        "dense-query-knn",
        help="run exact nested-LOO frozen dense query-kNN",
    )
    dense.add_argument("--queries", default="data/dev/llm_dev_data.tsv")
    dense.add_argument("--qrels", default="data/dev/llm_dev_qrels.txt")
    dense.add_argument(
        "--encoders",
        type=_encoder_grid,
        default=DEFAULT_ENCODERS,
        help=(
            "comma-separated pinned encoder keys: "
            + ",".join(spec.key for spec in DEFAULT_ENCODERS)
        ),
    )
    dense.add_argument(
        "--neighbors-grid",
        type=_neighbor_grid,
        default=(1, 3, 5, 10, 20, 50, 100),
    )
    dense.add_argument(
        "--device",
        default="cpu",
        help="cpu by default; another backend requires an external compatible torch build",
    )
    dense.add_argument("--batch-size", type=int, default=32)
    dense.add_argument("--torch-threads", type=int, default=16)
    dense.add_argument("--bootstrap-resamples", type=int, default=10_000)
    dense.add_argument("--seed", type=int, default=2026)
    dense.add_argument("--show-progress", action="store_true")
    dense.add_argument(
        "--grid-selection-provenance",
        default="advisor_grid_locked_before_dense_results",
    )
    dense.add_argument(
        "--output-dir",
        default="artifacts/dense_query_knn_baseline",
    )

    response = commands.add_parser(
        "response-retrieval",
        help="run nested-LOO BM25/RM3 retrieval over discovery responses",
    )
    response.add_argument("--queries", default="data/dev/llm_dev_data.tsv")
    response.add_argument("--qrels", default="data/dev/llm_dev_qrels.txt")
    response.add_argument(
        "--discovery",
        default="data/dev/llm_discovery_data.json",
    )
    response.add_argument(
        "--dense-per-query",
        default="reports/trec_2026_dense_query_knn_per_query.tsv",
        help="validated Baseline 2 per-query metrics used for paired comparisons",
    )
    response.add_argument("--bootstrap-resamples", type=int, default=10_000)
    response.add_argument("--seed", type=int, default=2026)
    response.add_argument("--show-progress", action="store_true")
    response.add_argument(
        "--grid-selection-provenance",
        default="advisor_reviewed_preregistration_locked_before_response_scores",
    )
    response.add_argument(
        "--output-dir",
        default="artifacts/response_retrieval_baseline",
    )

    normalization = commands.add_parser(
        "response-volume-normalization",
        help=(
            "run the nested 144-candidate volume-normalized response retriever "
            "and independent 24-candidate SumScore control"
        ),
    )
    normalization.add_argument("--queries", default="data/dev/llm_dev_data.tsv")
    normalization.add_argument("--qrels", default="data/dev/llm_dev_qrels.txt")
    normalization.add_argument(
        "--discovery",
        default="data/dev/llm_discovery_data.json",
    )
    normalization.add_argument(
        "--dense-per-query",
        default="reports/trec_2026_dense_query_knn_per_query.tsv",
        help="validated Baseline 2 per-query metrics used for paired comparisons",
    )
    normalization.add_argument("--bootstrap-resamples", type=int, default=10_000)
    normalization.add_argument("--seed", type=int, default=2026)
    normalization.add_argument("--show-progress", action="store_true")
    normalization.add_argument(
        "--grid-selection-provenance",
        default=(
            "advisor_approved_protocol_b6cfa6c_locked_before_effectiveness_scores"
        ),
    )
    normalization.add_argument(
        "--output-dir",
        default="artifacts/response_volume_normalization",
    )

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

    if args.command == "dense-query-knn":
        from trec_mlm.dense_experiment import (
            DenseExperimentConfig,
            run_nested_dense_experiment,
            write_dense_experiment_artifacts,
        )

        config = DenseExperimentConfig(
            query_path=args.queries,
            qrels_path=args.qrels,
            encoder_specs=args.encoders,
            neighbor_counts=args.neighbors_grid,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            torch_threads=args.torch_threads,
            show_progress_bar=args.show_progress,
            grid_selection_provenance=args.grid_selection_provenance,
        )
        result = run_nested_dense_experiment(config)
        paths = write_dense_experiment_artifacts(result, args.output_dir)
        print(json.dumps({"summary": result.summary, "artifacts": paths}, indent=2))
        return 0

    if args.command == "response-retrieval":
        from trec_mlm.response_experiment import (
            ResponseExperimentConfig,
            run_nested_response_experiment,
            write_response_experiment_artifacts,
        )

        config = ResponseExperimentConfig(
            query_path=args.queries,
            qrels_path=args.qrels,
            discovery_path=args.discovery,
            dense_per_query_path=args.dense_per_query,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            show_progress=args.show_progress,
            grid_selection_provenance=args.grid_selection_provenance,
        )
        result = run_nested_response_experiment(config)
        paths = write_response_experiment_artifacts(result, args.output_dir)
        print(json.dumps({"summary": result.summary, "artifacts": paths}, indent=2))
        return 0

    if args.command == "response-volume-normalization":
        from trec_mlm.response_normalization import (
            ResponseNormalizationConfig,
            run_response_normalization_experiment,
            write_response_normalization_artifacts,
        )

        config = ResponseNormalizationConfig(
            query_path=args.queries,
            qrels_path=args.qrels,
            discovery_path=args.discovery,
            dense_per_query_path=args.dense_per_query,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            show_progress=args.show_progress,
            grid_selection_provenance=args.grid_selection_provenance,
        )
        result = run_response_normalization_experiment(config)
        paths = write_response_normalization_artifacts(result, args.output_dir)
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
