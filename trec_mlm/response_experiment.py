"""Leakage-safe nested evaluation for response-level lexical retrieval.

The retrieval tensor consumed here is deliberately qrel-free.  Development
qrels are used only for fold-local tie priors, inner configuration selection,
and final effectiveness measurement.  Keeping retrieval and evaluation as two
separate stages makes the exclusion rules in the nested protocol auditable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.experiment import ExperimentConfig, run_leave_one_out_experiment
from trec_mlm.metrics import (
    EvaluationResult,
    QueryMetrics,
    evaluate_rankings,
    paired_bootstrap_ci,
)
from trec_mlm.query_knn import DevelopmentData, load_development_data
from trec_mlm.response_retrieval import (
    DEFAULT_RESPONSE_CANDIDATES,
    ResponseCandidate,
)
from trec_mlm.runio import validate_trec_run, write_trec_run


TFIDF_REFERENCE_NDCG_AT_10 = 0.2177
TFIDF_REFERENCE_MRR = 0.2777
PRIOR_REFERENCE_NDCG_AT_10 = 0.1385
PRIOR_REFERENCE_MRR = 0.2067
DENSE_REFERENCE_NDCG_AT_10 = 0.35985474409504775
DENSE_REFERENCE_MRR = 0.40406330811168817
DEFAULT_DENSE_PER_QUERY_PATH = "reports/trec_2026_dense_query_knn_per_query.tsv"


@dataclass(frozen=True, slots=True)
class ResponseExperimentConfig:
    """Resolved corpus, candidate-grid, and nested-evaluation parameters."""

    query_path: str
    qrels_path: str
    discovery_path: str
    candidates: tuple[ResponseCandidate, ...] = DEFAULT_RESPONSE_CANDIDATES
    dense_per_query_path: str | None = DEFAULT_DENSE_PER_QUERY_PATH
    cutoff: int = 10
    run_id: str = "response_bm25_nested_loo"
    tfidf_run_id: str = "qknn_tfidf_w12_k10_loo_control"
    prior_run_id: str = "global_prior_loo_control"
    bootstrap_resamples: int = 10_000
    seed: int = 2026
    show_progress: bool = False
    grid_selection_provenance: str = (
        "advisor_reviewed_preregistration_locked_before_response_scores"
    )


@dataclass(slots=True)
class ResponseExperimentResult:
    """Nested predictions, controls, diagnostics, and reproducibility state."""

    config: ResponseExperimentConfig
    data: DevelopmentData
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    candidates: tuple[ResponseCandidate, ...]
    model_scores: NDArray[np.float64]
    hit_counts: NDArray[np.int32] | None
    mean_hit_scores: NDArray[np.float64] | None
    overlap_document_counts: NDArray[np.int32] | None
    retrieval_metadata: dict[str, Any]
    inner_ndcg_by_candidate: dict[str, NDArray[np.float64]]
    inner_mrr_by_candidate: dict[str, NDArray[np.float64]]
    selected_config_by_outer: dict[str, str]
    selected_raw_scores: dict[str, dict[str, float]]
    selected_priors: dict[str, dict[str, float]]
    rankings: dict[str, dict[str, list[str]]]
    evaluations: dict[str, EvaluationResult]
    fixed_candidate_metrics: dict[str, dict[str, float]]
    deployment_config: str
    diagnostics: dict[str, Any]
    summary: dict[str, Any]


def candidate_key(candidate: ResponseCandidate) -> str:
    """Return the stable, preregistered key for a response candidate."""

    key = getattr(candidate, "key", None)
    if not isinstance(key, str) or not key:
        raise TypeError("every response candidate must expose a non-empty .key")
    return key


def fold_local_prior(
    grades: NDArray[np.floating[Any]],
    excluded_indices: Sequence[int],
) -> NDArray[np.float64]:
    """Return mean model grades after excluding the supplied query rows."""

    matrix = np.asarray(grades, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("grades must be a two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("grades must contain only finite values")
    excluded = tuple(excluded_indices)
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded query indices must be unique")
    if any(
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= matrix.shape[0]
        for index in excluded
    ):
        raise IndexError("excluded query index is out of range")
    denominator = matrix.shape[0] - len(excluded)
    if denominator < 1:
        raise ValueError("at least one qrel row must remain for the fold prior")
    total = matrix.sum(axis=0, dtype=np.float64)
    if excluded:
        total -= matrix[np.asarray(excluded, dtype=np.int64)].sum(
            axis=0,
            dtype=np.float64,
        )
    return total / denominator


def rank_model_scores(
    raw_scores: NDArray[np.floating[Any]],
    prior: NDArray[np.floating[Any]],
    model_ids: Sequence[str],
) -> NDArray[np.int64]:
    """Rank models by ``(-raw_score, -prior, model_id)`` exactly."""

    scores = np.asarray(raw_scores, dtype=np.float64)
    priors = np.asarray(prior, dtype=np.float64)
    if scores.ndim != 1 or priors.ndim != 1:
        raise ValueError("raw_scores and prior must be one-dimensional")
    if scores.shape != priors.shape or scores.shape[0] != len(model_ids):
        raise ValueError("score, prior, and model-ID shapes do not match")
    if not np.isfinite(scores).all() or not np.isfinite(priors).all():
        raise ValueError("raw scores and priors must be finite")
    lexical_ranks = _lexical_ranks(model_ids)
    return np.lexsort((lexical_ranks, -priors, -scores)).astype(
        np.int64,
        copy=False,
    )


def exact_nested_response_inner_scores(
    *,
    raw_scores: NDArray[np.floating[Any]],
    grades: NDArray[np.floating[Any]],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    cutoff: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact inner-LOO macro metrics for every outer exclusion.

    Retrieval scores for an inner query are fixed and qrel-free.  The only
    outer-dependent ranking component is the tie prior, which excludes both
    the outer and inner rows.  The implementation vectorizes those priors and
    rankings over every possible outer exclusion.
    """

    score_matrix, grade_matrix = _validate_score_and_grade_matrices(
        raw_scores,
        grades,
        query_ids=query_ids,
        model_ids=model_ids,
        minimum_queries=3,
    )
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")

    query_count, model_count = grade_matrix.shape
    model_lexical_rank = _lexical_ranks(model_ids)
    model_tie_matrix = np.broadcast_to(
        model_lexical_rank,
        (query_count, model_count),
    )
    grade_total = grade_matrix.sum(axis=0, dtype=np.float64)
    ndcg_sums = np.zeros(query_count, dtype=np.float64)
    mrr_sums = np.zeros(query_count, dtype=np.float64)
    top_count = min(cutoff, model_count)
    discounts = 1.0 / np.log2(np.arange(2, top_count + 2, dtype=np.float64))

    for inner_index in range(query_count):
        priors = (
            grade_total[None, :] - grade_matrix[inner_index][None, :] - grade_matrix
        ) / (query_count - 2)
        scores = np.broadcast_to(
            score_matrix[inner_index],
            (query_count, model_count),
        )
        orders = np.lexsort(
            (model_tie_matrix, -priors, -scores),
            axis=1,
        )

        target_grades = grade_matrix[inner_index]
        gains = np.maximum(target_grades[orders[:, :top_count]], 0.0)
        ideal_gains = np.sort(np.maximum(target_grades, 0.0))[::-1][:top_count]
        ideal_dcg = float(np.sum(ideal_gains * discounts, dtype=np.float64))
        if ideal_dcg > 0.0:
            ndcg_values = (
                np.sum(gains * discounts, axis=1, dtype=np.float64) / ideal_dcg
            )
        else:
            ndcg_values = np.zeros(query_count, dtype=np.float64)

        relevant = target_grades[orders] > 0.0
        has_relevant = np.any(relevant, axis=1)
        reciprocal_ranks = np.zeros(query_count, dtype=np.float64)
        reciprocal_ranks[has_relevant] = 1.0 / (
            np.argmax(relevant[has_relevant], axis=1) + 1
        )

        # outer == inner is not an inner validation observation for that fold.
        ndcg_values[inner_index] = 0.0
        reciprocal_ranks[inner_index] = 0.0
        ndcg_sums += ndcg_values
        mrr_sums += reciprocal_ranks

    denominator = query_count - 1
    return ndcg_sums / denominator, mrr_sums / denominator


def run_nested_response_experiment(
    config: ResponseExperimentConfig,
    *,
    score_result: Any | None = None,
    dense_per_query_reference: Mapping[str, Any] | EvaluationResult | None = None,
) -> ResponseExperimentResult:
    """Run strict nested outer-LOO selection over response retrieval scores."""

    _validate_config(config)
    started = time.perf_counter()
    data = load_development_data(config.query_path, config.qrels_path)
    if len(data.queries) < 3:
        raise ValueError("nested leave-one-out evaluation needs at least three queries")
    observed_grades = {
        grade for query_grades in data.qrels.values() for grade in query_grades.values()
    }
    if not observed_grades.issubset({0.0, 1.0, 2.0}):
        raise ValueError("the TREC experiment expects qrel grades in {0, 1, 2}")

    query_ids = tuple(sorted(data.queries))
    model_ids = data.models
    grades = np.asarray(
        [
            [data.qrels[query_id][model_id] for model_id in model_ids]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )

    if score_result is None:
        score_result = _score_from_discovery(config, data, query_ids, model_ids)
    score_bundle = _coerce_score_result(
        score_result,
        candidates=config.candidates,
        query_ids=query_ids,
        model_ids=model_ids,
    )
    candidates = score_bundle["candidates"]
    model_scores = score_bundle["model_scores"]
    hit_counts = score_bundle["hit_counts"]
    mean_hit_scores = score_bundle["mean_hit_scores"]
    overlap_document_counts = score_bundle["overlap_document_counts"]
    retrieval_metadata = score_bundle["metadata"]

    inner_ndcg_by_candidate: dict[str, NDArray[np.float64]] = {}
    inner_mrr_by_candidate: dict[str, NDArray[np.float64]] = {}
    for candidate_index, candidate in enumerate(candidates):
        key = candidate_key(candidate)
        inner_ndcg, inner_mrr = exact_nested_response_inner_scores(
            raw_scores=model_scores[candidate_index],
            grades=grades,
            query_ids=query_ids,
            model_ids=model_ids,
            cutoff=config.cutoff,
        )
        inner_ndcg_by_candidate[key] = inner_ndcg
        inner_mrr_by_candidate[key] = inner_mrr

    # np.argmax keeps the first declared candidate on exact metric ties.
    inner_score_matrix = np.column_stack(
        [inner_ndcg_by_candidate[candidate_key(candidate)] for candidate in candidates]
    )
    selected_indices = np.argmax(inner_score_matrix, axis=1)
    selected_config_by_outer = {
        query_id: candidate_key(candidates[int(selected_indices[index])])
        for index, query_id in enumerate(query_ids)
    }

    response_rankings: dict[str, list[str]] = {}
    prior_rankings: dict[str, list[str]] = {}
    selected_raw_scores: dict[str, dict[str, float]] = {}
    selected_priors: dict[str, dict[str, float]] = {}
    for outer_index, query_id in enumerate(query_ids):
        candidate_index = int(selected_indices[outer_index])
        raw = model_scores[candidate_index, outer_index]
        prior = fold_local_prior(grades, (outer_index,))
        order = rank_model_scores(raw, prior, model_ids)
        response_rankings[query_id] = [model_ids[int(index)] for index in order]
        prior_order = rank_model_scores(np.zeros_like(prior), prior, model_ids)
        prior_rankings[query_id] = [model_ids[int(index)] for index in prior_order]
        selected_raw_scores[query_id] = {
            model_id: float(raw[index]) for index, model_id in enumerate(model_ids)
        }
        selected_priors[query_id] = {
            model_id: float(prior[index]) for index, model_id in enumerate(model_ids)
        }

    tfidf_result = run_leave_one_out_experiment(
        ExperimentConfig(
            query_path=config.query_path,
            qrels_path=config.qrels_path,
            neighbor_counts=(10,),
            aggregations=("weighted_mean",),
            primary_neighbors=10,
            primary_aggregation="weighted_mean",
            ngram_range=(1, 2),
            cutoff=config.cutoff,
            run_id=config.tfidf_run_id,
            bootstrap_resamples=config.bootstrap_resamples,
            seed=config.seed,
            primary_selection_provenance="fixed_merged_baseline_k10",
        )
    )
    tfidf_rankings = tfidf_result.rankings["weighted_mean@10"]
    if prior_rankings != tfidf_result.rankings["global_prior"]:
        raise AssertionError(
            "response and TF-IDF pipelines produced different fold-local priors"
        )

    rankings = {
        "response_nested": response_rankings,
        "tfidf_fixed_k10": tfidf_rankings,
        "global_prior": prior_rankings,
    }
    evaluations = {
        key: evaluate_rankings(system_rankings, data.qrels, k=config.cutoff)
        for key, system_rankings in rankings.items()
    }
    dense_reference = _resolve_dense_reference(
        config,
        query_ids=query_ids,
        supplied=dense_per_query_reference,
    )
    evaluations["dense_nested_reference"] = dense_reference

    fixed_candidate_metrics: dict[str, dict[str, float]] = {}
    fixed_rankings_by_candidate: dict[str, dict[str, list[str]]] = {}
    for candidate_index, candidate in enumerate(candidates):
        key = candidate_key(candidate)
        candidate_rankings: dict[str, list[str]] = {}
        for query_index, query_id in enumerate(query_ids):
            prior = fold_local_prior(grades, (query_index,))
            order = rank_model_scores(
                model_scores[candidate_index, query_index],
                prior,
                model_ids,
            )
            candidate_rankings[query_id] = [model_ids[int(index)] for index in order]
        fixed_rankings_by_candidate[key] = candidate_rankings
        evaluation = evaluate_rankings(candidate_rankings, data.qrels, k=config.cutoff)
        fixed_candidate_metrics[key] = {
            "ndcg_at_10": evaluation.macro_ndcg_at_k,
            "mrr": evaluation.mean_reciprocal_rank,
        }
    deployment_config = max(
        range(len(candidates)),
        key=lambda index: fixed_candidate_metrics[candidate_key(candidates[index])][
            "ndcg_at_10"
        ],
    )
    deployment_key = candidate_key(candidates[deployment_config])

    diagnostics = _build_diagnostics(
        candidates=candidates,
        query_ids=query_ids,
        model_ids=model_ids,
        model_scores=model_scores,
        hit_counts=hit_counts,
        mean_hit_scores=mean_hit_scores,
        overlap_document_counts=overlap_document_counts,
        selected_indices=selected_indices,
        fixed_rankings_by_candidate=fixed_rankings_by_candidate,
        retrieval_metadata=retrieval_metadata,
    )
    core_elapsed = time.perf_counter() - started
    summary_started = time.perf_counter()
    summary = _build_summary(
        config=config,
        data=data,
        query_ids=query_ids,
        candidates=candidates,
        retrieval_metadata=retrieval_metadata,
        inner_ndcg_by_candidate=inner_ndcg_by_candidate,
        inner_mrr_by_candidate=inner_mrr_by_candidate,
        selected_config_by_outer=selected_config_by_outer,
        evaluations=evaluations,
        fixed_candidate_metrics=fixed_candidate_metrics,
        deployment_config=deployment_key,
        diagnostics=diagnostics,
        core_elapsed_seconds=core_elapsed,
    )
    summary["runtime"].update(
        {
            "summary_seconds_including_bootstrap": (
                time.perf_counter() - summary_started
            ),
            "experiment_seconds_through_summary": time.perf_counter() - started,
        }
    )
    return ResponseExperimentResult(
        config=config,
        data=data,
        query_ids=query_ids,
        model_ids=model_ids,
        candidates=candidates,
        model_scores=model_scores,
        hit_counts=hit_counts,
        mean_hit_scores=mean_hit_scores,
        overlap_document_counts=overlap_document_counts,
        retrieval_metadata=retrieval_metadata,
        inner_ndcg_by_candidate=inner_ndcg_by_candidate,
        inner_mrr_by_candidate=inner_mrr_by_candidate,
        selected_config_by_outer=selected_config_by_outer,
        selected_raw_scores=selected_raw_scores,
        selected_priors=selected_priors,
        rankings=rankings,
        evaluations=evaluations,
        fixed_candidate_metrics=fixed_candidate_metrics,
        deployment_config=deployment_key,
        diagnostics=diagnostics,
        summary=summary,
    )


def load_dense_per_query_reference(
    path: str | Path,
    *,
    expected_query_ids: Sequence[str] | None = None,
    expected_ndcg_at_10: float | None = None,
    expected_mrr: float | None = None,
) -> EvaluationResult:
    """Load and strictly validate Baseline 2 per-query metrics."""

    source = Path(path)
    per_query: dict[str, QueryMetrics] = {}
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "query_id",
            "dense_ndcg_at_10",
            "dense_reciprocal_rank",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{source}: dense reference is missing columns {sorted(missing)!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            if not query_id:
                raise ValueError(f"{source}:{line_number}: empty query ID")
            if query_id in per_query:
                raise ValueError(
                    f"{source}:{line_number}: duplicate query ID {query_id!r}"
                )
            try:
                ndcg = float(row["dense_ndcg_at_10"])
                reciprocal = float(row["dense_reciprocal_rank"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid dense metric"
                ) from exc
            _validate_reference_metric(ndcg, "nDCG@10", query_id)
            _validate_reference_metric(reciprocal, "reciprocal rank", query_id)
            per_query[query_id] = QueryMetrics(ndcg, reciprocal)

    evaluation = _evaluation_from_per_query(
        per_query,
        expected_query_ids=expected_query_ids,
    )
    if expected_ndcg_at_10 is not None and not math.isclose(
        evaluation.macro_ndcg_at_k,
        expected_ndcg_at_10,
        rel_tol=0.0,
        abs_tol=5e-12,
    ):
        raise ValueError(
            "dense reference nDCG@10 mean does not match Baseline 2: "
            f"observed={evaluation.macro_ndcg_at_k:.15g}, "
            f"expected={expected_ndcg_at_10:.15g}"
        )
    if expected_mrr is not None and not math.isclose(
        evaluation.mean_reciprocal_rank,
        expected_mrr,
        rel_tol=0.0,
        abs_tol=5e-12,
    ):
        raise ValueError(
            "dense reference MRR mean does not match Baseline 2: "
            f"observed={evaluation.mean_reciprocal_rank:.15g}, "
            f"expected={expected_mrr:.15g}"
        )
    return evaluation


def write_response_experiment_artifacts(
    result: ResponseExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write nested evidence, complete runs, diagnostics, and QA receipts."""

    artifact_started = time.perf_counter()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_config = _jsonable(asdict(result.config))
    input_paths = [
        Path(result.config.query_path),
        Path(result.config.qrels_path),
        Path(result.config.discovery_path),
    ]
    if result.config.dense_per_query_path:
        input_paths.append(Path(result.config.dense_per_query_path))
    resolved_config.update(
        {
            "input_sha256": {
                str(path): _sha256(path) for path in input_paths if path.is_file()
            },
            "code_revision": _git_revision(),
            "git_worktree_status": _git_worktree_status(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "code_sha256": {
                path.name: _sha256(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("response_retrieval.py"),
                    Path(__file__).with_name("experiment.py"),
                    Path(__file__).with_name("metrics.py"),
                    Path(__file__).with_name("runio.py"),
                )
                if path.is_file()
            },
        }
    )
    _write_json(destination / "config.json", resolved_config)
    _write_json(destination / "retrieval_metadata.json", result.retrieval_metadata)
    _write_json(destination / "diagnostics.json", result.diagnostics)
    score_tensors: dict[str, NDArray[Any]] = {
        "model_scores": result.model_scores,
    }
    if result.hit_counts is not None:
        score_tensors["hit_counts"] = result.hit_counts
    if result.mean_hit_scores is not None:
        score_tensors["mean_hit_scores"] = result.mean_hit_scores
    if result.overlap_document_counts is not None:
        score_tensors["overlap_document_counts"] = result.overlap_document_counts
    np.savez_compressed(destination / "score_tensors.npz", **score_tensors)

    query_index = {query_id: index for index, query_id in enumerate(result.query_ids)}
    candidate_index = {
        candidate_key(candidate): index
        for index, candidate in enumerate(result.candidates)
    }
    with (destination / "inner_scores.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "outer_query_id",
                "configuration",
                "inner_macro_ndcg_at_10",
                "inner_mrr",
                "selected",
            )
        )
        for query_id in result.query_ids:
            outer_index = query_index[query_id]
            for candidate in result.candidates:
                key = candidate_key(candidate)
                writer.writerow(
                    (
                        query_id,
                        key,
                        f"{result.inner_ndcg_by_candidate[key][outer_index]:.12g}",
                        f"{result.inner_mrr_by_candidate[key][outer_index]:.12g}",
                        int(result.selected_config_by_outer[query_id] == key),
                    )
                )

    with (destination / "selections.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "outer_query_id",
                "configuration",
                "selected_inner_ndcg_at_10",
                "selected_inner_mrr",
            )
        )
        for query_id in result.query_ids:
            index = query_index[query_id]
            key = result.selected_config_by_outer[query_id]
            writer.writerow(
                (
                    query_id,
                    key,
                    f"{result.inner_ndcg_by_candidate[key][index]:.12g}",
                    f"{result.inner_mrr_by_candidate[key][index]:.12g}",
                )
            )

    with (destination / "predictions.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "target_query_id",
                "configuration",
                "model_id",
                "sum_score",
                "global_prior",
                "hit_count",
                "mean_hit_score",
                "final_rank",
            )
        )
        for query_id in result.query_ids:
            query_position = query_index[query_id]
            key = result.selected_config_by_outer[query_id]
            config_position = candidate_index[key]
            raw_by_model = result.selected_raw_scores[query_id]
            prior_by_model = result.selected_priors[query_id]
            ranking = result.rankings["response_nested"][query_id]
            for rank, model_id in enumerate(ranking, start=1):
                model_position = result.model_ids.index(model_id)
                hit_count = (
                    int(
                        result.hit_counts[
                            config_position, query_position, model_position
                        ]
                    )
                    if result.hit_counts is not None
                    else ""
                )
                mean_hit = (
                    f"{result.mean_hit_scores[config_position, query_position, model_position]:.12g}"
                    if result.mean_hit_scores is not None
                    else ""
                )
                writer.writerow(
                    (
                        query_id,
                        key,
                        model_id,
                        f"{raw_by_model[model_id]:.12g}",
                        f"{prior_by_model[model_id]:.12g}",
                        hit_count,
                        mean_hit,
                        rank,
                    )
                )

    with (destination / "query_diagnostics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "configuration",
                "zero_score_model_count",
                "total_hit_count",
                "overlap_document_count",
            )
        )
        for query_id in result.query_ids:
            q = query_index[query_id]
            key = result.selected_config_by_outer[query_id]
            c = candidate_index[key]
            writer.writerow(
                (
                    query_id,
                    key,
                    int(np.count_nonzero(result.model_scores[c, q] == 0.0)),
                    int(result.hit_counts[c, q].sum())
                    if result.hit_counts is not None
                    else "",
                    int(result.overlap_document_counts[c, q])
                    if result.overlap_document_counts is not None
                    else "",
                )
            )

    with (destination / "per_query.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "selected_configuration",
                "response_ndcg_at_10",
                "response_reciprocal_rank",
                "dense_ndcg_at_10",
                "dense_reciprocal_rank",
                "tfidf_ndcg_at_10",
                "tfidf_reciprocal_rank",
                "prior_ndcg_at_10",
                "prior_reciprocal_rank",
            )
        )
        for query_id in result.query_ids:
            values = {
                key: result.evaluations[key].per_query[query_id]
                for key in (
                    "response_nested",
                    "dense_nested_reference",
                    "tfidf_fixed_k10",
                    "global_prior",
                )
            }
            writer.writerow(
                (
                    query_id,
                    result.selected_config_by_outer[query_id],
                    f"{values['response_nested'].ndcg_at_k:.12g}",
                    f"{values['response_nested'].reciprocal_rank:.12g}",
                    f"{values['dense_nested_reference'].ndcg_at_k:.12g}",
                    f"{values['dense_nested_reference'].reciprocal_rank:.12g}",
                    f"{values['tfidf_fixed_k10'].ndcg_at_k:.12g}",
                    f"{values['tfidf_fixed_k10'].reciprocal_rank:.12g}",
                    f"{values['global_prior'].ndcg_at_k:.12g}",
                    f"{values['global_prior'].reciprocal_rank:.12g}",
                )
            )

    run_names = {
        "response_nested": ("run.trec", result.config.run_id),
        "tfidf_fixed_k10": ("tfidf_control.trec", result.config.tfidf_run_id),
        "global_prior": ("global_prior_control.trec", result.config.prior_run_id),
    }
    for system, (filename, run_id) in run_names.items():
        write_trec_run(
            destination / filename,
            result.rankings[system],
            run_id=run_id,
        )
    _write_json(destination / "summary.json", result.summary)

    output_names = (
        "config.json",
        "retrieval_metadata.json",
        "diagnostics.json",
        "score_tensors.npz",
        "inner_scores.tsv",
        "selections.tsv",
        "predictions.tsv",
        "query_diagnostics.tsv",
        "per_query.tsv",
        "run.trec",
        "tfidf_control.trec",
        "global_prior_control.trec",
        "summary.json",
    )
    validation = {
        "inputs": {
            "complete_qrel_matrix": True,
            "query_count": len(result.query_ids),
            "candidate_model_count": len(result.model_ids),
            "configuration_count": len(result.candidates),
        },
        "protocol": {
            "outer_prediction_count": len(result.query_ids),
            "inner_targets_per_outer": len(result.query_ids) - 1,
            "outer_qrel_excluded_from_selection": True,
            "inner_prior_excludes_outer_and_inner": True,
            "response_prior_matches_tfidf_prior": True,
            "score_tensor_shape": list(result.model_scores.shape),
            "score_tensor_sha256": _array_sha256(result.model_scores),
        },
        "runs": {
            system: validate_trec_run(
                destination / filename,
                expected_query_ids=result.query_ids,
                expected_model_ids=result.model_ids,
            )
            for system, (filename, _run_id) in run_names.items()
        },
        "artifact_sha256": {name: _sha256(destination / name) for name in output_names},
        "runtime": {
            "artifact_generation_and_validation_seconds_before_receipt": (
                time.perf_counter() - artifact_started
            )
        },
    }
    _write_json(destination / "validation.json", validation)
    return {
        name: str(destination / name) for name in (*output_names, "validation.json")
    }


def _score_from_discovery(
    config: ResponseExperimentConfig,
    data: DevelopmentData,
    query_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
) -> Any:
    # The lazy import keeps unit tests and score-tensor-only audits independent
    # of the optional response-retrieval dependencies.
    from trec_mlm.response_retrieval import (  # pylint: disable=import-outside-toplevel
        load_discovery_records,
        score_response_candidates,
    )

    corpus = load_discovery_records(config.discovery_path, model_ids)
    if tuple(corpus.model_ids) != model_ids:
        raise ValueError("discovery and development model universes do not match")
    return score_response_candidates(
        corpus,
        query_ids=query_ids,
        query_texts=[data.queries[query_id] for query_id in query_ids],
        expected_model_ids=model_ids,
        candidates=config.candidates,
        show_progress=config.show_progress,
    )


def _coerce_score_result(
    score_result: Any,
    *,
    candidates: Sequence[ResponseCandidate],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    actual_candidates = tuple(score_result.candidates)
    expected_keys = tuple(candidate_key(candidate) for candidate in candidates)
    actual_keys = tuple(candidate_key(candidate) for candidate in actual_candidates)
    if actual_keys != expected_keys:
        raise ValueError(
            "score-result candidate order does not match config: "
            f"actual={actual_keys!r}, expected={expected_keys!r}"
        )
    if tuple(score_result.query_ids) != tuple(query_ids):
        raise ValueError(
            "score-result query order does not match sorted development IDs"
        )
    if tuple(score_result.model_ids) != tuple(model_ids):
        raise ValueError("score-result model order does not match development models")

    scores = np.asarray(score_result.model_scores, dtype=np.float64)
    expected_shape = (len(candidates), len(query_ids), len(model_ids))
    if scores.shape != expected_shape:
        raise ValueError(
            f"model score tensor has shape {scores.shape}; expected {expected_shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("model score tensor must contain only finite values")
    if np.any(scores < 0.0):
        raise ValueError("SumScore values must be non-negative")

    hit_counts = _optional_tensor(
        score_result,
        "hit_counts",
        expected_shape,
        dtype=np.int32,
        nonnegative=True,
    )
    mean_hit_scores = _optional_tensor(
        score_result,
        "mean_hit_scores",
        expected_shape,
        dtype=np.float64,
        nonnegative=True,
    )
    overlap_counts = _optional_tensor(
        score_result,
        "overlap_document_counts",
        (len(candidates), len(query_ids)),
        dtype=np.int32,
        nonnegative=True,
    )
    metadata_names = (
        "filter_audits",
        "inventory_filter_names",
        "retained_document_counts",
        "retained_character_counts",
        "retained_token_counts",
        "index_diagnostics",
        "analyzer_diagnostics",
        "source_sha256",
        "corpus_sha256",
        "configuration_sha256",
        "score_tensor_sha256",
    )
    metadata = {
        name: _jsonable(getattr(score_result, name))
        for name in metadata_names
        if hasattr(score_result, name)
    }
    # The retrieval-layer score hash intentionally covers scores *and* the
    # associated hit/overlap diagnostic arrays.  Record a separate hash for
    # the C×Q×M SumScore tensor instead of incorrectly comparing the two.
    metadata["model_score_tensor_sha256"] = _array_sha256(scores)
    return {
        "candidates": actual_candidates,
        "model_scores": scores,
        "hit_counts": hit_counts,
        "mean_hit_scores": mean_hit_scores,
        "overlap_document_counts": overlap_counts,
        "metadata": metadata,
    }


def _resolve_dense_reference(
    config: ResponseExperimentConfig,
    *,
    query_ids: Sequence[str],
    supplied: Mapping[str, Any] | EvaluationResult | None,
) -> EvaluationResult:
    if supplied is not None:
        return _coerce_dense_reference(supplied, expected_query_ids=query_ids)
    if config.dense_per_query_path is None:
        raise ValueError(
            "dense_per_query_reference or config.dense_per_query_path is required"
        )
    return load_dense_per_query_reference(
        config.dense_per_query_path,
        expected_query_ids=query_ids,
        expected_ndcg_at_10=DENSE_REFERENCE_NDCG_AT_10,
        expected_mrr=DENSE_REFERENCE_MRR,
    )


def _coerce_dense_reference(
    supplied: Mapping[str, Any] | EvaluationResult,
    *,
    expected_query_ids: Sequence[str],
) -> EvaluationResult:
    if isinstance(supplied, EvaluationResult):
        if supplied.cutoff != 10:
            raise ValueError("dense reference must use nDCG cutoff 10")
        evaluation = _evaluation_from_per_query(
            supplied.per_query,
            expected_query_ids=expected_query_ids,
        )
        if not math.isclose(
            evaluation.macro_ndcg_at_k,
            supplied.macro_ndcg_at_k,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            evaluation.mean_reciprocal_rank,
            supplied.mean_reciprocal_rank,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "dense reference macro values do not match per-query values"
            )
        return evaluation

    per_query: dict[str, QueryMetrics] = {}
    for query_id, value in supplied.items():
        if isinstance(value, QueryMetrics):
            metric = value
        elif isinstance(value, Mapping):
            ndcg_value = value.get("ndcg_at_k", value.get("ndcg_at_10"))
            rr_value = value.get("reciprocal_rank", value.get("mrr"))
            if ndcg_value is None or rr_value is None:
                raise ValueError(
                    f"dense reference for {query_id!r} lacks nDCG/RR fields"
                )
            metric = QueryMetrics(float(ndcg_value), float(rr_value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError(
                    f"dense reference tuple for {query_id!r} must have two values"
                )
            metric = QueryMetrics(float(value[0]), float(value[1]))
        else:
            raise TypeError(f"unsupported dense reference value for {query_id!r}")
        _validate_reference_metric(metric.ndcg_at_k, "nDCG@10", query_id)
        _validate_reference_metric(metric.reciprocal_rank, "reciprocal rank", query_id)
        per_query[str(query_id)] = metric
    return _evaluation_from_per_query(
        per_query,
        expected_query_ids=expected_query_ids,
    )


def _evaluation_from_per_query(
    per_query: Mapping[str, QueryMetrics],
    *,
    expected_query_ids: Sequence[str] | None,
) -> EvaluationResult:
    if not per_query:
        raise ValueError("dense per-query reference is empty")
    if expected_query_ids is not None and set(per_query) != set(expected_query_ids):
        missing = sorted(set(expected_query_ids) - set(per_query))
        extra = sorted(set(per_query) - set(expected_query_ids))
        raise ValueError(
            f"dense reference query mismatch: missing={missing!r}, extra={extra!r}"
        )
    metrics = dict(per_query)
    return EvaluationResult(
        cutoff=10,
        per_query=metrics,
        macro_ndcg_at_k=statistics.fmean(value.ndcg_at_k for value in metrics.values()),
        mean_reciprocal_rank=statistics.fmean(
            value.reciprocal_rank for value in metrics.values()
        ),
    )


def _build_diagnostics(
    *,
    candidates: Sequence[ResponseCandidate],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    model_scores: NDArray[np.float64],
    hit_counts: NDArray[np.int32] | None,
    mean_hit_scores: NDArray[np.float64] | None,
    overlap_document_counts: NDArray[np.int32] | None,
    selected_indices: NDArray[np.int64],
    fixed_rankings_by_candidate: Mapping[str, Mapping[str, Sequence[str]]],
    retrieval_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    selected_zero_counts = [
        int(np.count_nonzero(model_scores[int(selected_indices[q]), q] == 0.0))
        for q in range(len(query_ids))
    ]
    diagnostics: dict[str, Any] = {
        "selected_zero_score_model_count": {
            "mean": statistics.fmean(selected_zero_counts),
            "min": min(selected_zero_counts),
            "max": max(selected_zero_counts),
        },
        "overlap_document_count_max_per_query_total": (
            int(overlap_document_counts.max(axis=0).sum())
            if overlap_document_counts is not None
            else None
        ),
        "overlap_candidate_query_pairs": (
            int(np.count_nonzero(overlap_document_counts))
            if overlap_document_counts is not None
            else None
        ),
        "mean_hit_score_tensor_available": mean_hit_scores is not None,
        "hit_count_tensor_available": hit_counts is not None,
        "volume_bias_by_candidate": {},
    }

    inventory_names = tuple(retrieval_metadata.get("inventory_filter_names", ()))
    inventory_counts_value = retrieval_metadata.get("retained_document_counts")
    if hit_counts is None or inventory_counts_value is None or not inventory_names:
        return diagnostics
    inventory_counts = np.asarray(inventory_counts_value, dtype=np.float64)
    if inventory_counts.ndim != 2 or inventory_counts.shape[1] != len(model_ids):
        diagnostics["volume_bias_unavailable_reason"] = (
            "retained_document_counts shape does not match model order"
        )
        return diagnostics

    inventory_by_name = {
        str(name): inventory_counts[index] for index, name in enumerate(inventory_names)
    }
    model_position = {model_id: index for index, model_id in enumerate(model_ids)}
    for candidate_index, candidate in enumerate(candidates):
        key = candidate_key(candidate)
        filter_name = _candidate_filter_name(candidate)
        inventory = inventory_by_name.get(filter_name)
        if inventory is None:
            continue
        hit_totals = hit_counts[candidate_index].sum(axis=0, dtype=np.float64)
        score_totals = model_scores[candidate_index].sum(axis=0, dtype=np.float64)
        rank_sums = np.zeros(len(model_ids), dtype=np.float64)
        top10_counts = np.zeros(len(model_ids), dtype=np.float64)
        for ranking in fixed_rankings_by_candidate[key].values():
            for rank, model_id in enumerate(ranking, start=1):
                position = model_position[model_id]
                rank_sums[position] += rank
                if rank <= 10:
                    top10_counts[position] += 1
        mean_ranks = rank_sums / len(query_ids)
        diagnostics["volume_bias_by_candidate"][key] = {
            "filter": filter_name,
            "spearman": {
                "inventory_vs_hit_count": _spearman(inventory, hit_totals),
                "inventory_vs_sum_score": _spearman(inventory, score_totals),
                "inventory_vs_mean_rank": _spearman(inventory, mean_ranks),
                "inventory_vs_top10_frequency": _spearman(
                    inventory,
                    top10_counts,
                ),
            },
            "top10_representation_by_inventory_quartile": _quartile_diagnostics(
                inventory,
                top10_counts,
                model_ids,
            ),
        }
    return diagnostics


def _build_summary(
    *,
    config: ResponseExperimentConfig,
    data: DevelopmentData,
    query_ids: Sequence[str],
    candidates: Sequence[ResponseCandidate],
    retrieval_metadata: Mapping[str, Any],
    inner_ndcg_by_candidate: Mapping[str, NDArray[np.float64]],
    inner_mrr_by_candidate: Mapping[str, NDArray[np.float64]],
    selected_config_by_outer: Mapping[str, str],
    evaluations: Mapping[str, EvaluationResult],
    fixed_candidate_metrics: Mapping[str, Mapping[str, float]],
    deployment_config: str,
    diagnostics: Mapping[str, Any],
    core_elapsed_seconds: float,
) -> dict[str, Any]:
    selected_counts = Counter(selected_config_by_outer.values())
    comparisons = {
        baseline: _paired_comparison(
            baseline,
            evaluations[baseline],
            evaluations["response_nested"],
            n_resamples=config.bootstrap_resamples,
            seed=config.seed + offset,
        )
        for offset, baseline in enumerate(
            ("dense_nested_reference", "tfidf_fixed_k10", "global_prior")
        )
    }
    selected_inner_ndcg = [
        float(inner_ndcg_by_candidate[selected_config_by_outer[query_id]][index])
        for index, query_id in enumerate(query_ids)
    ]
    selected_inner_mrr = [
        float(inner_mrr_by_candidate[selected_config_by_outer[query_id]][index])
        for index, query_id in enumerate(query_ids)
    ]
    metrics = {
        key: {
            "ndcg_at_10": evaluation.macro_ndcg_at_k,
            "mrr": evaluation.mean_reciprocal_rank,
        }
        for key, evaluation in evaluations.items()
    }
    return {
        "protocol": {
            "outer_split": "strict_leave_one_query_out",
            "inner_split": "strict_leave_one_query_out_within_outer_training_rows",
            "outer_query_count": len(query_ids),
            "inner_validation_query_count_per_outer": len(query_ids) - 1,
            "retrieval_representation": "response_level_bm25_or_explicit_rm3",
            "retrieval_scores_are_qrel_free": True,
            "selection_metric": "inner_macro_ndcg_at_10",
            "selection_tie_break": "declared_24_candidate_order",
            "mrr_used_for_selection": False,
            "ranking_tie_break": "fold_local_global_prior_then_model_id",
            "inner_prior_exclusions": "outer_and_inner_query",
            "outer_prior_exclusions": "outer_query",
            "grid_selection_provenance": config.grid_selection_provenance,
            "bootstrap": (
                "paired_query_resampling_conditioned_on_stitched_out_of_fold_selections"
            ),
            "run_classification": "internal_automatic",
            "logprob_features": False,
            "learned_retrieval_features": False,
        },
        "data": {
            "query_count": len(query_ids),
            "candidate_model_count": len(data.models),
            "qrel_count": sum(len(values) for values in data.qrels.values()),
            "grade_counts": {
                str(grade): sum(
                    value == grade
                    for values in data.qrels.values()
                    for value in values.values()
                )
                for grade in (0.0, 1.0, 2.0)
            },
        },
        "candidate_grid": [
            {"key": candidate_key(candidate), **_candidate_payload(candidate)}
            for candidate in candidates
        ],
        "retrieval_reproducibility": dict(retrieval_metadata),
        "metrics": metrics,
        "comparisons": comparisons,
        "selection": {
            "configuration_frequency": dict(sorted(selected_counts.items())),
            "selected_inner_ndcg_mean": statistics.fmean(selected_inner_ndcg),
            "selected_inner_ndcg_min": min(selected_inner_ndcg),
            "selected_inner_ndcg_max": max(selected_inner_ndcg),
            "selected_inner_mrr_mean": statistics.fmean(selected_inner_mrr),
        },
        "deployment_selection": {
            "configuration": deployment_config,
            "criterion": "full_development_fixed_candidate_ndcg_at_10",
            "tie_break": "declared_24_candidate_order",
            "purpose": "future_hidden_test_queries_only",
            "development_score_is_non_nested_and_optimistic": True,
        },
        "fixed_candidate_development_metrics": dict(fixed_candidate_metrics),
        "diagnostics": dict(diagnostics),
        "control_regression_reference": {
            "scope": "official_322_query_development_files_only",
            "tfidf_fixed_k10": {
                "expected_rounded_ndcg_at_10": TFIDF_REFERENCE_NDCG_AT_10,
                "expected_rounded_mrr": TFIDF_REFERENCE_MRR,
                "observed_rounded_ndcg_at_10": round(
                    metrics["tfidf_fixed_k10"]["ndcg_at_10"], 4
                ),
                "observed_rounded_mrr": round(metrics["tfidf_fixed_k10"]["mrr"], 4),
            },
            "global_prior": {
                "expected_rounded_ndcg_at_10": PRIOR_REFERENCE_NDCG_AT_10,
                "expected_rounded_mrr": PRIOR_REFERENCE_MRR,
                "observed_rounded_ndcg_at_10": round(
                    metrics["global_prior"]["ndcg_at_10"], 4
                ),
                "observed_rounded_mrr": round(metrics["global_prior"]["mrr"], 4),
            },
            "dense_nested_reference": {
                "expected_ndcg_at_10": DENSE_REFERENCE_NDCG_AT_10,
                "expected_mrr": DENSE_REFERENCE_MRR,
                "observed_ndcg_at_10": metrics["dense_nested_reference"]["ndcg_at_10"],
                "observed_mrr": metrics["dense_nested_reference"]["mrr"],
            },
        },
        "runtime": {
            "experiment_core_seconds_excluding_summary_and_artifact_writes": (
                core_elapsed_seconds
            )
        },
    }


def _paired_comparison(
    baseline_name: str,
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_ndcg = {
        query_id: values.ndcg_at_k for query_id, values in baseline.per_query.items()
    }
    candidate_ndcg = {
        query_id: values.ndcg_at_k for query_id, values in candidate.per_query.items()
    }
    baseline_rr = {
        query_id: values.reciprocal_rank
        for query_id, values in baseline.per_query.items()
    }
    candidate_rr = {
        query_id: values.reciprocal_rank
        for query_id, values in candidate.per_query.items()
    }
    return {
        "candidate": "response_nested",
        "baseline": baseline_name,
        "ndcg_at_10_delta": asdict(
            paired_bootstrap_ci(
                baseline_ndcg,
                candidate_ndcg,
                n_resamples=n_resamples,
                seed=seed,
            )
        ),
        "mrr_delta": asdict(
            paired_bootstrap_ci(
                baseline_rr,
                candidate_rr,
                n_resamples=n_resamples,
                seed=seed + 10_000,
            )
        ),
        "ndcg_at_10_win_tie_loss": _win_tie_loss(
            baseline_ndcg,
            candidate_ndcg,
        ),
        "mrr_win_tie_loss": _win_tie_loss(baseline_rr, candidate_rr),
    }


def _win_tie_loss(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, int]:
    differences = [candidate[key] - baseline[key] for key in baseline]
    wins = sum(value > tolerance for value in differences)
    losses = sum(value < -tolerance for value in differences)
    return {"wins": wins, "ties": len(differences) - wins - losses, "losses": losses}


def _validate_config(config: ResponseExperimentConfig) -> None:
    if not config.candidates:
        raise ValueError("candidates cannot be empty")
    keys = [candidate_key(candidate) for candidate in config.candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("response candidate keys must be unique")
    if config.cutoff != 10:
        raise ValueError("the TREC response baseline currently supports cutoff=10 only")
    if (
        isinstance(config.bootstrap_resamples, bool)
        or not isinstance(config.bootstrap_resamples, int)
        or config.bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")


def _validate_score_and_grade_matrices(
    raw_scores: NDArray[np.floating[Any]],
    grades: NDArray[np.floating[Any]],
    *,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    minimum_queries: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    score_matrix = np.asarray(raw_scores, dtype=np.float64)
    grade_matrix = np.asarray(grades, dtype=np.float64)
    if score_matrix.ndim != 2 or grade_matrix.ndim != 2:
        raise ValueError("raw_scores and grades must be two-dimensional")
    if score_matrix.shape != grade_matrix.shape:
        raise ValueError("raw score and grade matrices must have the same shape")
    if score_matrix.shape[0] < minimum_queries:
        raise ValueError(
            f"nested leave-one-out requires at least {minimum_queries} queries"
        )
    if score_matrix.shape != (len(query_ids), len(model_ids)):
        raise ValueError("query/model IDs do not match matrix shapes")
    if not np.isfinite(score_matrix).all() or not np.isfinite(grade_matrix).all():
        raise ValueError("scores and grades must contain only finite values")
    return score_matrix, grade_matrix


def _optional_tensor(
    result: Any,
    name: str,
    expected_shape: tuple[int, ...],
    *,
    dtype: Any,
    nonnegative: bool,
) -> NDArray[Any] | None:
    if not hasattr(result, name) or getattr(result, name) is None:
        return None
    value = np.asarray(getattr(result, name), dtype=dtype)
    if value.shape != expected_shape:
        raise ValueError(
            f"{name} tensor has shape {value.shape}; expected {expected_shape}"
        )
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(value < 0):
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_reference_metric(value: float, name: str, query_id: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"dense {name} for query {query_id!r} must be finite in [0, 1]"
        )


def _candidate_filter_name(candidate: ResponseCandidate) -> str:
    for name in ("filter_name", "response_filter", "filter"):
        value = getattr(candidate, name, None)
        if isinstance(value, str):
            return value
    return "unknown"


def _candidate_payload(candidate: ResponseCandidate) -> dict[str, Any]:
    if is_dataclass(candidate):
        return {
            field.name: _jsonable(getattr(candidate, field.name))
            for field in fields(candidate)
            if field.name != "key"
        }
    return {"repr": repr(candidate)}


def _lexical_ranks(values: Sequence[str]) -> NDArray[np.int64]:
    names = np.asarray(values)
    order = np.argsort(names, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


def _rankdata_average(values: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(
    left: NDArray[np.floating[Any]],
    right: NDArray[np.floating[Any]],
) -> float | None:
    left_ranks = _rankdata_average(left)
    right_ranks = _rankdata_average(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = math.sqrt(
        float(left_centered @ left_centered) * float(right_centered @ right_centered)
    )
    if denominator == 0.0:
        return None
    return float((left_centered @ right_centered) / denominator)


def _quartile_diagnostics(
    inventory: NDArray[np.floating[Any]],
    top10_counts: NDArray[np.floating[Any]],
    model_ids: Sequence[str],
) -> list[dict[str, Any]]:
    order = sorted(
        range(len(model_ids)),
        key=lambda index: (float(inventory[index]), model_ids[index]),
    )
    groups = np.array_split(
        np.asarray(order, dtype=np.int64),
        min(4, len(order)),
    )
    total_top10 = float(np.sum(top10_counts, dtype=np.float64))
    output: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        group_top10 = float(np.sum(top10_counts[group], dtype=np.float64))
        output.append(
            {
                "quartile": index,
                "model_count": int(len(group)),
                "inventory_min": float(np.min(inventory[group])),
                "inventory_max": float(np.max(inventory[group])),
                "top10_appearances": int(group_top10),
                "top10_appearance_share": (
                    group_top10 / total_top10 if total_top10 > 0.0 else 0.0
                ),
            }
        )
    return output


def _array_sha256(value: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_worktree_status() -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
