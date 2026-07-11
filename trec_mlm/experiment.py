"""Single-pass leave-one-out experiment for the query-kNN baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from trec_mlm.metrics import EvaluationResult, evaluate_rankings, paired_bootstrap_ci
from trec_mlm.query_knn import DevelopmentData, QueryKNNRanker, load_development_data
from trec_mlm.runio import validate_trec_run, write_trec_run


DEFAULT_NEIGHBOR_COUNTS = (1, 3, 5, 10, 20, 50, 100)
DEFAULT_AGGREGATIONS = ("weighted_mean", "weighted_sum", "max")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Resolved experiment parameters."""

    query_path: str
    qrels_path: str
    neighbor_counts: tuple[int, ...] = DEFAULT_NEIGHBOR_COUNTS
    aggregations: tuple[str, ...] = DEFAULT_AGGREGATIONS
    primary_neighbors: int = 10
    primary_aggregation: str = "weighted_mean"
    ngram_range: tuple[int, int] = (1, 2)
    cutoff: int = 10
    run_id: str = "qknn_tfidf_w12_k10_loo"
    bootstrap_resamples: int = 10_000
    seed: int = 2026
    primary_selection_provenance: str = "not_recorded"


@dataclass(slots=True)
class ExperimentResult:
    """In-memory predictions, diagnostics, and evaluations."""

    config: ExperimentConfig
    data: DevelopmentData
    rankings: dict[str, dict[str, list[str]]]
    raw_scores: dict[str, dict[str, float]]
    priors: dict[str, dict[str, float]]
    neighbor_mass: dict[str, float]
    neighbors: dict[str, list[tuple[str, float]]]
    evaluations: dict[str, EvaluationResult]
    summary: dict[str, Any]


def configuration_key(aggregation: str, k: int) -> str:
    return f"{aggregation}@{k}"


def _validate_config(config: ExperimentConfig) -> None:
    if not config.neighbor_counts:
        raise ValueError("neighbor_counts cannot be empty")
    if any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1
        for k in config.neighbor_counts
    ):
        raise ValueError("every neighbor count must be a positive integer")
    if len(set(config.neighbor_counts)) != len(config.neighbor_counts):
        raise ValueError("neighbor_counts must be unique")
    allowed = {"weighted_mean", "weighted_sum", "max"}
    invalid = set(config.aggregations) - allowed
    if invalid:
        raise ValueError(f"unsupported aggregations: {sorted(invalid)!r}")
    if config.primary_neighbors not in config.neighbor_counts:
        raise ValueError("primary_neighbors must appear in neighbor_counts")
    if config.primary_aggregation not in config.aggregations:
        raise ValueError("primary_aggregation must appear in aggregations")
    if config.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if config.cutoff != 10:
        raise ValueError("this TREC baseline currently supports cutoff=10 only")


def run_leave_one_out_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Evaluate every configured rule while fitting each LOO fold only once."""

    _validate_config(config)
    data = load_development_data(config.query_path, config.qrels_path)
    observed_grades = {
        grade for grades in data.qrels.values() for grade in grades.values()
    }
    if not observed_grades.issubset({0.0, 1.0, 2.0}):
        raise ValueError(
            "the TREC development experiment expects qrel grades in {0, 1, 2}"
        )
    if len(data.queries) < 2:
        raise ValueError("leave-one-out evaluation requires at least two queries")

    neighbor_counts = tuple(sorted(config.neighbor_counts))
    maximum_k = max(neighbor_counts)
    keys = [
        configuration_key(aggregation, k)
        for aggregation in config.aggregations
        for k in neighbor_counts
    ]
    rankings: dict[str, dict[str, list[str]]] = {key: {} for key in keys}
    rankings["global_prior"] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    priors: dict[str, dict[str, float]] = {}
    neighbor_mass: dict[str, float] = {}
    neighbors_by_target: dict[str, list[tuple[str, float]]] = {}

    for held_out_id in sorted(data.queries):
        training_queries = {
            query_id: text
            for query_id, text in data.queries.items()
            if query_id != held_out_id
        }
        training_qrels = {
            query_id: grades
            for query_id, grades in data.qrels.items()
            if query_id != held_out_id
        }
        ranker = QueryKNNRanker(
            k=maximum_k,
            aggregation=config.primary_aggregation,
            ngram_range=config.ngram_range,
        ).fit(training_queries, training_qrels)
        query_neighbors = ranker.nearest_queries(
            data.queries[held_out_id], k=maximum_k
        )
        neighbors_by_target[held_out_id] = [
            (neighbor.query_id, neighbor.similarity) for neighbor in query_neighbors
        ]
        query_priors = ranker.global_priors
        priors[held_out_id] = query_priors
        prior_order = sorted(data.models, key=lambda model: (-query_priors[model], model))
        rankings["global_prior"][held_out_id] = prior_order

        similarities = [neighbor.similarity for neighbor in query_neighbors]
        weight_by_count = {
            count: math.fsum(similarities[:count])
            for count in {min(k, len(query_neighbors)) for k in neighbor_counts}
        }

        snapshots = {min(k, len(query_neighbors)) for k in neighbor_counts}
        sums_by_k: dict[int, dict[str, float]] = {k: {} for k in neighbor_counts}
        maxima_by_k: dict[int, dict[str, float]] = {k: {} for k in neighbor_counts}
        for model_id in data.models:
            evidence_values: list[float] = []
            running_max = -math.inf
            snapshot_values: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
            for position, neighbor in enumerate(query_neighbors, start=1):
                evidence = neighbor.similarity * training_qrels[neighbor.query_id][model_id]
                evidence_values.append(evidence)
                running_max = max(running_max, evidence)
                if position in snapshots:
                    snapshot_values[position] = (
                        math.fsum(evidence_values),
                        running_max,
                    )
            for k in neighbor_counts:
                actual_k = min(k, len(query_neighbors))
                summed, maximum = snapshot_values[actual_k]
                sums_by_k[k][model_id] = summed
                maxima_by_k[k][model_id] = maximum

        per_rule_scores: dict[str, dict[str, float]] = {}
        for k in neighbor_counts:
            actual_k = min(k, len(query_neighbors))
            mass = weight_by_count[actual_k]
            for aggregation in config.aggregations:
                key = configuration_key(aggregation, k)
                if mass <= 0.0:
                    scores = dict(query_priors)
                elif aggregation == "weighted_sum":
                    scores = sums_by_k[k]
                elif aggregation == "weighted_mean":
                    scores = {
                        model_id: sums_by_k[k][model_id] / mass
                        for model_id in data.models
                    }
                else:
                    scores = maxima_by_k[k]
                per_rule_scores[key] = scores
                rankings[key][held_out_id] = sorted(
                    data.models,
                    key=lambda model: (-scores[model], -query_priors[model], model),
                )

        primary_key = configuration_key(
            config.primary_aggregation, config.primary_neighbors
        )
        raw_scores[held_out_id] = per_rule_scores[primary_key]
        primary_actual_k = min(config.primary_neighbors, len(query_neighbors))
        neighbor_mass[held_out_id] = weight_by_count[primary_actual_k]

    evaluations = {
        key: evaluate_rankings(system_rankings, data.qrels, k=config.cutoff)
        for key, system_rankings in rankings.items()
    }
    summary = _build_summary(
        config=config,
        data=data,
        rankings=rankings,
        evaluations=evaluations,
        neighbors=neighbors_by_target,
        neighbor_mass=neighbor_mass,
    )
    return ExperimentResult(
        config=config,
        data=data,
        rankings=rankings,
        raw_scores=raw_scores,
        priors=priors,
        neighbor_mass=neighbor_mass,
        neighbors=neighbors_by_target,
        evaluations=evaluations,
        summary=summary,
    )


def _build_summary(
    *,
    config: ExperimentConfig,
    data: DevelopmentData,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    evaluations: Mapping[str, EvaluationResult],
    neighbors: Mapping[str, Sequence[tuple[str, float]]],
    neighbor_mass: Mapping[str, float],
) -> dict[str, Any]:
    primary_key = configuration_key(config.primary_aggregation, config.primary_neighbors)
    primary = evaluations[primary_key]
    prior = evaluations["global_prior"]
    primary_ndcg = {q: value.ndcg_at_k for q, value in primary.per_query.items()}
    prior_ndcg = {q: value.ndcg_at_k for q, value in prior.per_query.items()}
    primary_rr = {q: value.reciprocal_rank for q, value in primary.per_query.items()}
    prior_rr = {q: value.reciprocal_rank for q, value in prior.per_query.items()}
    ndcg_ci = paired_bootstrap_ci(
        prior_ndcg,
        primary_ndcg,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    rr_ci = paired_bootstrap_ci(
        prior_rr,
        primary_rr,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed + 1,
    )
    deltas = {q: primary_ndcg[q] - prior_ndcg[q] for q in primary_ndcg}
    tolerance = 1e-12
    wins = sum(delta > tolerance for delta in deltas.values())
    losses = sum(delta < -tolerance for delta in deltas.values())
    ties = len(deltas) - wins - losses

    top_similarities = [items[0][1] for items in neighbors.values() if items]
    positive_neighbor_counts = [len(items) for items in neighbors.values()]
    all_similarity_values = [score for items in neighbors.values() for _, score in items]
    density_slices: dict[str, dict[str, float | int]] = {}
    for label, lower, upper in (
        ("exactly_1_positive", 1, 1),
        ("2_to_5_positives", 2, 5),
        ("6_to_20_positives", 6, 20),
        ("more_than_20_positives", 21, math.inf),
    ):
        query_ids = [
            query_id
            for query_id, grades in data.qrels.items()
            if lower <= sum(grade > 0 for grade in grades.values()) <= upper
        ]
        density_slices[label] = {
            "query_count": len(query_ids),
            "primary_ndcg_at_10": _mean_for_query_ids(primary_ndcg, query_ids),
            "prior_ndcg_at_10": _mean_for_query_ids(prior_ndcg, query_ids),
            "primary_mrr": _mean_for_query_ids(primary_rr, query_ids),
            "prior_mrr": _mean_for_query_ids(prior_rr, query_ids),
        }

    comparable_k = (
        config.neighbor_counts
        if {"weighted_sum", "weighted_mean"}.issubset(config.aggregations)
        else ()
    )
    sum_mean_identical = (
        all(
            rankings[configuration_key("weighted_sum", k)]
            == rankings[configuration_key("weighted_mean", k)]
            for k in comparable_k
        )
        if comparable_k
        else None
    )
    return {
        "protocol": {
            "split": "strict_leave_one_query_out",
            "representation": "word_tfidf",
            "ngram_range": list(config.ngram_range),
            "primary_selection_provenance_declared": (
                config.primary_selection_provenance
            ),
            "sensitivity_grid_is_exploratory": True,
            "metric_gain": "linear_qrel_grade",
            "evidence_source": "neighbor_query_qrels_not_discovery_responses",
            "tie_break": "training_global_prior_then_model_id",
        },
        "data": {
            "query_count": len(data.queries),
            "candidate_count": len(data.models),
            "qrel_count": sum(len(grades) for grades in data.qrels.values()),
            "grade_counts": {
                str(int(grade)): sum(
                    value == grade
                    for grades in data.qrels.values()
                    for value in grades.values()
                )
                for grade in (0.0, 1.0, 2.0)
            },
        },
        "primary_key": primary_key,
        "metrics": {
            key: {
                "ndcg_at_10": value.macro_ndcg_at_k,
                "mrr": value.mean_reciprocal_rank,
            }
            for key, value in evaluations.items()
        },
        "primary_vs_global_prior": {
            "ndcg_at_10_delta": asdict(ndcg_ci),
            "mrr_delta": asdict(rr_ci),
            "ndcg_win_tie_loss": {"wins": wins, "ties": ties, "losses": losses},
        },
        "density_slices": density_slices,
        "neighbor_diagnostics": {
            "zero_weight_fallback_count": sum(
                mass <= 0.0 for mass in neighbor_mass.values()
            ),
            "median_top1_similarity": (
                statistics.median(top_similarities) if top_similarities else 0.0
            ),
            "top1_similarity_q75": (
                _quantile(top_similarities, 0.75) if top_similarities else 0.0
            ),
            "median_positive_neighbors_retained_at_max_k": statistics.median(
                positive_neighbor_counts
            ),
            "mean_similarity_across_retained_neighbors": (
                statistics.fmean(all_similarity_values) if all_similarity_values else 0.0
            ),
        },
        "weighted_sum_mean_rankings_identical": sum_mean_identical,
    }


def write_experiment_artifacts(
    result: ExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the reproducibility, diagnostic, run, and score artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    primary_key = configuration_key(
        result.config.primary_aggregation, result.config.primary_neighbors
    )
    input_paths = [Path(result.config.query_path), Path(result.config.qrels_path)]
    resolved_config = asdict(result.config)
    resolved_config.update(
        {
            "input_sha256": {str(path): _sha256(path) for path in input_paths},
            "code_revision": _git_revision(),
            "git_worktree_status": _git_worktree_status(),
            "python_version": sys.version,
            "lockfile_sha256": _sha256(
                Path(__file__).resolve().parents[1] / "uv.lock"
            ),
            "code_sha256": {
                path.name: _sha256(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("query_knn.py"),
                    Path(__file__).with_name("metrics.py"),
                    Path(__file__).with_name("runio.py"),
                    Path(__file__).with_name("cli.py"),
                )
            },
        }
    )
    _write_json(destination / "config.json", resolved_config)

    with (destination / "folds.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("query_id", "split", "held_out_fold"))
        for query_id in sorted(result.data.queries):
            writer.writerow((query_id, "test", query_id))

    with (destination / "neighbors.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ("target_query_id", "neighbor_query_id", "similarity", "neighbor_rank")
        )
        for target_id in sorted(result.neighbors):
            for rank, (neighbor_id, similarity) in enumerate(
                result.neighbors[target_id], start=1
            ):
                writer.writerow((target_id, neighbor_id, f"{similarity:.12g}", rank))

    with (destination / "predictions.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "target_query_id",
                "model_id",
                "raw_score",
                "global_prior",
                "neighbor_mass",
                "final_rank",
                "configuration",
            )
        )
        for query_id in sorted(result.data.queries):
            for rank, model_id in enumerate(
                result.rankings[primary_key][query_id], start=1
            ):
                writer.writerow(
                    (
                        query_id,
                        model_id,
                        f"{result.raw_scores[query_id][model_id]:.12g}",
                        f"{result.priors[query_id][model_id]:.12g}",
                        f"{result.neighbor_mass[query_id]:.12g}",
                        rank,
                        primary_key,
                    )
                )

    run_path = destination / "run.trec"
    write_trec_run(
        run_path,
        result.rankings[primary_key],
        run_id=result.config.run_id,
    )

    primary_metrics = result.evaluations[primary_key].per_query
    prior_metrics = result.evaluations["global_prior"].per_query
    with (destination / "per_query.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "ndcg_at_10",
                "reciprocal_rank",
                "prior_ndcg_at_10",
                "prior_reciprocal_rank",
                "positive_count",
                "grade_2_count",
            )
        )
        for query_id in sorted(result.data.queries):
            grades = result.data.qrels[query_id]
            writer.writerow(
                (
                    query_id,
                    f"{primary_metrics[query_id].ndcg_at_k:.12g}",
                    f"{primary_metrics[query_id].reciprocal_rank:.12g}",
                    f"{prior_metrics[query_id].ndcg_at_k:.12g}",
                    f"{prior_metrics[query_id].reciprocal_rank:.12g}",
                    sum(value > 0 for value in grades.values()),
                    sum(value == 2 for value in grades.values()),
                )
            )

    _write_json(destination / "summary.json", result.summary)
    output_names = (
        "config.json",
        "folds.tsv",
        "neighbors.tsv",
        "predictions.tsv",
        "run.trec",
        "per_query.tsv",
        "summary.json",
    )
    validation = {
        "inputs": {
            "complete_qrel_matrix": True,
            "query_count": len(result.data.queries),
            "candidate_count": len(result.data.models),
        },
        "run": validate_trec_run(
            run_path,
            expected_query_ids=tuple(result.data.queries),
            expected_model_ids=result.data.models,
        ),
        "weighted_sum_mean_rankings_identical": result.summary[
            "weighted_sum_mean_rankings_identical"
        ],
        "artifact_sha256": {
            name: _sha256(destination / name) for name in output_names
        },
    }
    _write_json(destination / "validation.json", validation)
    return {
        name: str(destination / name)
        for name in (
            "config.json",
            "folds.tsv",
            "neighbors.tsv",
            "predictions.tsv",
            "run.trec",
            "per_query.tsv",
            "summary.json",
            "validation.json",
        )
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean_for_query_ids(
    values: Mapping[str, float], query_ids: Sequence[str]
) -> float | None:
    return statistics.fmean(values[query_id] for query_id in query_ids) if query_ids else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    repository_root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_worktree_status() -> str | None:
    repository_root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
