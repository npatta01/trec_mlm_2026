"""Exact nested leave-one-out evaluation for frozen dense query retrieval."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.dense_query_knn import (
    DEFAULT_ENCODERS,
    DenseTargetRanking,
    EncoderSpec,
    array_sha256,
    cosine_similarity_matrix,
    encode_frozen_queries,
    normalize_embeddings,
    rank_target_from_similarity,
    validate_similarity_matrix,
)
from trec_mlm.experiment import (
    ExperimentConfig,
    configuration_key as tfidf_configuration_key,
    run_leave_one_out_experiment,
)
from trec_mlm.metrics import EvaluationResult, evaluate_rankings, paired_bootstrap_ci
from trec_mlm.query_knn import DevelopmentData, load_development_data
from trec_mlm.runio import validate_trec_run, write_trec_run


DEFAULT_DENSE_NEIGHBOR_COUNTS = (1, 3, 5, 10, 20, 50, 100)
TFIDF_REFERENCE_NDCG_AT_10 = 0.2177
TFIDF_REFERENCE_MRR = 0.2777
PRIOR_REFERENCE_NDCG_AT_10 = 0.1385
PRIOR_REFERENCE_MRR = 0.2067


@dataclass(frozen=True, slots=True)
class DenseCandidate:
    """One encoder and neighbor-count option in the inner grid."""

    encoder_key: str
    neighbors: int

    @property
    def key(self) -> str:
        return f"{self.encoder_key}@{self.neighbors}"


@dataclass(frozen=True, slots=True)
class DenseExperimentConfig:
    """Resolved frozen-encoder and exact nested-validation parameters."""

    query_path: str
    qrels_path: str
    encoder_specs: tuple[EncoderSpec, ...] = DEFAULT_ENCODERS
    neighbor_counts: tuple[int, ...] = DEFAULT_DENSE_NEIGHBOR_COUNTS
    cutoff: int = 10
    run_id: str = "qknn_dense_nested_loo"
    tfidf_run_id: str = "qknn_tfidf_w12_k10_loo_control"
    prior_run_id: str = "qknn_global_prior_loo_control"
    bootstrap_resamples: int = 10_000
    seed: int = 2026
    device: str = "cpu"
    batch_size: int = 32
    torch_threads: int | None = 16
    show_progress_bar: bool = False
    grid_selection_provenance: str = "advisor_grid_locked_before_dense_results"


@dataclass(slots=True)
class DenseExperimentResult:
    """Predictions, nested selection evidence, and reproducibility state."""

    config: DenseExperimentConfig
    data: DevelopmentData
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    candidates: tuple[DenseCandidate, ...]
    embeddings: dict[str, NDArray[np.float32]]
    similarities: dict[str, NDArray[np.float32]]
    encoder_metadata: dict[str, dict[str, Any]]
    inner_ndcg_by_outer: dict[str, NDArray[np.float64]]
    inner_mrr_by_outer: dict[str, NDArray[np.float64]]
    selected_config_by_outer: dict[str, str]
    selected_rankings: dict[str, DenseTargetRanking]
    rankings: dict[str, dict[str, list[str]]]
    evaluations: dict[str, EvaluationResult]
    fixed_config_metrics: dict[str, dict[str, float]]
    deployment_config: str
    summary: dict[str, Any]


def run_nested_dense_experiment(
    config: DenseExperimentConfig,
    *,
    precomputed_embeddings: Mapping[str, NDArray[np.floating[Any]]] | None = None,
    precomputed_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> DenseExperimentResult:
    """Run exact outer-LOO evaluation with encoder+k chosen by inner LOO.

    Frozen, qrel-free query embeddings are computed once.  For outer target
    ``o``, each inner target ``i`` is predicted only from rows other than
    ``o`` and ``i``.  The outer target's qrel cannot affect configuration
    selection, model scores, the fold prior, or deterministic tie breaks.
    """

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

    if precomputed_embeddings is None:
        embeddings, encoder_metadata = encode_frozen_queries(
            query_ids=query_ids,
            query_texts=[data.queries[query_id] for query_id in query_ids],
            specs=config.encoder_specs,
            device=config.device,
            batch_size=config.batch_size,
            torch_threads=config.torch_threads,
            show_progress_bar=config.show_progress_bar,
        )
    else:
        expected_keys = {spec.key for spec in config.encoder_specs}
        actual_keys = set(precomputed_embeddings)
        if actual_keys != expected_keys:
            raise ValueError(
                "precomputed encoder keys do not match config: "
                f"missing={sorted(expected_keys - actual_keys)!r}, "
                f"extra={sorted(actual_keys - expected_keys)!r}"
            )
        embeddings = {}
        encoder_metadata = {}
        supplied_metadata = precomputed_metadata or {}
        for spec in config.encoder_specs:
            embedding = normalize_embeddings(precomputed_embeddings[spec.key])
            if embedding.shape[0] != len(query_ids):
                raise ValueError(
                    f"encoder {spec.key!r} has {embedding.shape[0]} rows; "
                    f"expected {len(query_ids)}"
                )
            embeddings[spec.key] = embedding
            encoder_metadata[spec.key] = {
                **dict(supplied_metadata.get(spec.key, {})),
                "key": spec.key,
                "model_id": spec.model_id,
                "requested_revision": spec.revision,
                "resolved_revision": spec.revision,
                "license": spec.license,
                "query_prefix": spec.query_prefix,
                "device": config.device,
                "dtype": str(embedding.dtype),
                "normalized": True,
                "embedding_dimension": int(embedding.shape[1]),
                "query_count": int(embedding.shape[0]),
                "embedding_sha256": array_sha256(embedding),
                "source": "precomputed",
            }

    similarities: dict[str, NDArray[np.float32]] = {}
    for spec in config.encoder_specs:
        similarity = cosine_similarity_matrix(embeddings[spec.key])
        similarities[spec.key] = similarity
        encoder_metadata[spec.key]["similarity_sha256"] = array_sha256(similarity)
        encoder_metadata[spec.key]["zero_embedding_count"] = int(
            np.count_nonzero(np.linalg.norm(embeddings[spec.key], axis=1) == 0.0)
        )

    candidates = tuple(
        DenseCandidate(encoder_key=spec.key, neighbors=k)
        for spec in config.encoder_specs
        for k in sorted(config.neighbor_counts)
    )
    inner_ndcg_by_outer: dict[str, NDArray[np.float64]] = {}
    inner_mrr_by_outer: dict[str, NDArray[np.float64]] = {}
    for candidate in candidates:
        inner_ndcg, inner_mrr = exact_nested_loo_inner_scores(
            similarities=similarities[candidate.encoder_key],
            grades=grades,
            query_ids=query_ids,
            model_ids=model_ids,
            k=candidate.neighbors,
            cutoff=config.cutoff,
        )
        inner_ndcg_by_outer[candidate.key] = inner_ndcg
        inner_mrr_by_outer[candidate.key] = inner_mrr

    # Candidate order is the declared encoder order followed by ascending k.
    # np.argmax therefore supplies the preregistered exact-tie rule without
    # consulting outer labels or outer performance.
    inner_score_matrix = np.column_stack(
        [inner_ndcg_by_outer[candidate.key] for candidate in candidates]
    )
    selected_indices = np.argmax(inner_score_matrix, axis=1)
    selected_config_by_outer = {
        query_id: candidates[int(selected_indices[index])].key
        for index, query_id in enumerate(query_ids)
    }
    candidates_by_key = {candidate.key: candidate for candidate in candidates}

    selected_rankings: dict[str, DenseTargetRanking] = {}
    dense_rankings: dict[str, list[str]] = {}
    prior_rankings: dict[str, list[str]] = {}
    for target_index, query_id in enumerate(query_ids):
        candidate = candidates_by_key[selected_config_by_outer[query_id]]
        prediction = rank_target_from_similarity(
            target_index=target_index,
            additionally_excluded_indices=(),
            similarities=similarities[candidate.encoder_key],
            grades=grades,
            query_ids=query_ids,
            model_ids=model_ids,
            k=candidate.neighbors,
        )
        selected_rankings[query_id] = prediction
        dense_rankings[query_id] = [
            model_ids[index] for index in prediction.model_order
        ]
        prior_order = np.lexsort((np.asarray(model_ids), -prediction.global_prior))
        prior_rankings[query_id] = [model_ids[int(index)] for index in prior_order]

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
    tfidf_key = tfidf_configuration_key("weighted_mean", 10)
    tfidf_rankings = tfidf_result.rankings[tfidf_key]
    if prior_rankings != tfidf_result.rankings["global_prior"]:
        raise AssertionError(
            "dense and TF-IDF pipelines produced different fold-local prior controls"
        )

    rankings = {
        "dense_nested": dense_rankings,
        "tfidf_fixed_k10": tfidf_rankings,
        "global_prior": prior_rankings,
    }
    evaluations = {
        key: evaluate_rankings(system_rankings, data.qrels, k=config.cutoff)
        for key, system_rankings in rankings.items()
    }

    fixed_config_metrics: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        fixed_rankings = _fixed_config_leave_one_out_rankings(
            similarities=similarities[candidate.encoder_key],
            grades=grades,
            query_ids=query_ids,
            model_ids=model_ids,
            k=candidate.neighbors,
        )
        fixed_evaluation = evaluate_rankings(
            fixed_rankings,
            data.qrels,
            k=config.cutoff,
        )
        fixed_config_metrics[candidate.key] = {
            "ndcg_at_10": fixed_evaluation.macro_ndcg_at_k,
            "mrr": fixed_evaluation.mean_reciprocal_rank,
        }
    deployment_config = max(
        candidates,
        key=lambda candidate: (
            fixed_config_metrics[candidate.key]["ndcg_at_10"],
            -candidates.index(candidate),
        ),
    ).key

    core_elapsed_seconds = time.perf_counter() - started
    summary_started = time.perf_counter()
    summary = _build_summary(
        config=config,
        data=data,
        query_ids=query_ids,
        candidates=candidates,
        encoder_metadata=encoder_metadata,
        inner_ndcg_by_outer=inner_ndcg_by_outer,
        inner_mrr_by_outer=inner_mrr_by_outer,
        selected_config_by_outer=selected_config_by_outer,
        evaluations=evaluations,
        fixed_config_metrics=fixed_config_metrics,
        deployment_config=deployment_config,
        core_elapsed_seconds=core_elapsed_seconds,
    )
    summary["runtime"].update(
        {
            "summary_seconds_including_bootstrap": (
                time.perf_counter() - summary_started
            ),
            "experiment_seconds_through_summary": time.perf_counter() - started,
        }
    )
    return DenseExperimentResult(
        config=config,
        data=data,
        query_ids=query_ids,
        model_ids=model_ids,
        candidates=candidates,
        embeddings=embeddings,
        similarities=similarities,
        encoder_metadata=encoder_metadata,
        inner_ndcg_by_outer=inner_ndcg_by_outer,
        inner_mrr_by_outer=inner_mrr_by_outer,
        selected_config_by_outer=selected_config_by_outer,
        selected_rankings=selected_rankings,
        rankings=rankings,
        evaluations=evaluations,
        fixed_config_metrics=fixed_config_metrics,
        deployment_config=deployment_config,
        summary=summary,
    )


def exact_nested_loo_inner_scores(
    *,
    similarities: NDArray[np.floating[Any]],
    grades: NDArray[np.floating[Any]],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    k: int,
    cutoff: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact inner-LOO macro metrics for every outer exclusion.

    The implementation vectorizes over all possible outer exclusions.  For an
    inner target, excluding one outer row either leaves its top-k prefix intact
    or removes that row and promotes the old k+1 neighbor.  This is exactly
    equivalent to materializing every ``all_queries - {outer, inner}`` support
    set, while avoiding millions of Python-level ranking loops.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    grade_matrix = np.asarray(grades, dtype=np.float64)
    if grade_matrix.ndim != 2:
        raise ValueError("grades must be a two-dimensional array")
    query_count, candidate_count = grade_matrix.shape
    if query_count < 3:
        raise ValueError("nested leave-one-out requires at least three queries")
    if len(query_ids) != query_count or len(model_ids) != candidate_count:
        raise ValueError("query/model IDs do not match grade matrix shape")
    if not np.isfinite(grade_matrix).all():
        raise ValueError("grades must contain only finite values")
    matrix = validate_similarity_matrix(
        similarities,
        expected_queries=query_count,
    ).astype(np.float64, copy=False)

    query_lexical_rank = _lexical_ranks(query_ids)
    model_lexical_rank = _lexical_ranks(model_ids)
    model_tie_matrix = np.broadcast_to(
        model_lexical_rank,
        (query_count, candidate_count),
    )
    grade_total = grade_matrix.sum(axis=0, dtype=np.float64)
    ndcg_sums = np.zeros(query_count, dtype=np.float64)
    mrr_sums = np.zeros(query_count, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))

    for inner_index in range(query_count):
        inner_similarities = matrix[inner_index]
        possible = np.asarray(
            [
                index
                for index in range(query_count)
                if index != inner_index and inner_similarities[index] > 0.0
            ],
            dtype=np.int64,
        )
        if possible.size:
            neighbor_order = np.lexsort(
                (
                    query_lexical_rank[possible],
                    -inner_similarities[possible],
                )
            )
            ordered_neighbors = possible[neighbor_order]
        else:
            ordered_neighbors = possible

        selected_count = min(k, len(ordered_neighbors))
        selected = ordered_neighbors[:selected_count]
        selected_weights = inner_similarities[selected]
        base_mass = float(selected_weights.sum(dtype=np.float64))

        # Row o is the prior over all qrels except inner i and outer o.
        priors = (
            grade_total[None, :] - grade_matrix[inner_index][None, :] - grade_matrix
        ) / (query_count - 2)
        if selected_count == 0 or base_mass <= 0.0:
            scores = priors.copy()
        else:
            base_numerator = selected_weights @ grade_matrix[selected]
            scores = np.broadcast_to(
                base_numerator / base_mass,
                (query_count, candidate_count),
            ).copy()
            replacement_index = (
                int(ordered_neighbors[k]) if len(ordered_neighbors) > k else None
            )
            adjusted_numerators = (
                base_numerator[None, :]
                - selected_weights[:, None] * grade_matrix[selected]
            )
            adjusted_masses = base_mass - selected_weights
            if replacement_index is not None:
                replacement_weight = float(inner_similarities[replacement_index])
                adjusted_numerators += (
                    replacement_weight * grade_matrix[replacement_index]
                )
                adjusted_masses += replacement_weight
            positive_mass = adjusted_masses > 0.0
            if np.any(positive_mass):
                rows = selected[positive_mass]
                scores[rows] = (
                    adjusted_numerators[positive_mass]
                    / adjusted_masses[positive_mass, None]
                )
            if np.any(~positive_mass):
                rows = selected[~positive_mass]
                scores[rows] = priors[rows]

        model_orders = np.lexsort(
            (model_tie_matrix, -priors, -scores),
            axis=1,
        )
        target_grades = grade_matrix[inner_index]
        top_count = min(cutoff, candidate_count)
        gains = target_grades[model_orders[:, :top_count]]
        ideal_gains = np.sort(np.maximum(target_grades, 0.0))[::-1][:top_count]
        ideal_dcg = float(np.sum(ideal_gains * discounts[:top_count]))
        if ideal_dcg > 0.0:
            ndcg_values = np.sum(gains * discounts[:top_count], axis=1) / ideal_dcg
        else:
            ndcg_values = np.zeros(query_count, dtype=np.float64)

        relevant = target_grades[model_orders] > 0.0
        has_relevant = np.any(relevant, axis=1)
        reciprocal_ranks = np.zeros(query_count, dtype=np.float64)
        reciprocal_ranks[has_relevant] = 1.0 / (
            np.argmax(relevant[has_relevant], axis=1) + 1
        )

        # outer == inner is not a real inner validation target for that outer
        # fold and must not enter its n-1 denominator.
        ndcg_values[inner_index] = 0.0
        reciprocal_ranks[inner_index] = 0.0
        ndcg_sums += ndcg_values
        mrr_sums += reciprocal_ranks

    return ndcg_sums / (query_count - 1), mrr_sums / (query_count - 1)


def write_dense_experiment_artifacts(
    result: DenseExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write nested-selection, embedding, prediction, run, and QA artifacts."""

    artifact_started = time.perf_counter()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    embedding_dir = destination / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    expected_embedding_files = {
        f"{encoder_key}.npy" for encoder_key in result.embeddings
    }
    for stale_path in embedding_dir.glob("*.npy"):
        if stale_path.name not in expected_embedding_files:
            stale_path.unlink()

    input_paths = [Path(result.config.query_path), Path(result.config.qrels_path)]
    resolved_config = asdict(result.config)
    resolved_config.update(
        {
            "input_sha256": {str(path): _sha256(path) for path in input_paths},
            "code_revision": _git_revision(),
            "git_worktree_status": _git_worktree_status(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "lockfile_sha256": _sha256(Path(__file__).resolve().parents[1] / "uv.lock"),
            "code_sha256": {
                path.name: _sha256(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("dense_query_knn.py"),
                    Path(__file__).with_name("experiment.py"),
                    Path(__file__).with_name("query_knn.py"),
                    Path(__file__).with_name("metrics.py"),
                    Path(__file__).with_name("runio.py"),
                    Path(__file__).with_name("cli.py"),
                )
            },
        }
    )
    _write_json(destination / "config.json", resolved_config)
    _write_json(destination / "encoder_metadata.json", result.encoder_metadata)

    for encoder_key, embeddings in result.embeddings.items():
        np.save(embedding_dir / f"{encoder_key}.npy", embeddings, allow_pickle=False)

    candidates_by_key = {candidate.key: candidate for candidate in result.candidates}
    query_index = {query_id: index for index, query_id in enumerate(result.query_ids)}
    with (destination / "inner_scores.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "outer_query_id",
                "configuration",
                "encoder_key",
                "neighbors",
                "inner_macro_ndcg_at_10",
                "inner_mrr",
                "selected",
            )
        )
        for query_id in result.query_ids:
            index = query_index[query_id]
            for candidate in result.candidates:
                writer.writerow(
                    (
                        query_id,
                        candidate.key,
                        candidate.encoder_key,
                        candidate.neighbors,
                        f"{result.inner_ndcg_by_outer[candidate.key][index]:.12g}",
                        f"{result.inner_mrr_by_outer[candidate.key][index]:.12g}",
                        int(result.selected_config_by_outer[query_id] == candidate.key),
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
                "encoder_key",
                "neighbors",
                "selected_inner_ndcg_at_10",
                "selected_inner_mrr",
            )
        )
        for query_id in result.query_ids:
            index = query_index[query_id]
            selected_key = result.selected_config_by_outer[query_id]
            candidate = candidates_by_key[selected_key]
            writer.writerow(
                (
                    query_id,
                    selected_key,
                    candidate.encoder_key,
                    candidate.neighbors,
                    f"{result.inner_ndcg_by_outer[selected_key][index]:.12g}",
                    f"{result.inner_mrr_by_outer[selected_key][index]:.12g}",
                )
            )

    with (destination / "neighbors.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "target_query_id",
                "configuration",
                "neighbor_query_id",
                "similarity",
                "neighbor_rank",
            )
        )
        for query_id in result.query_ids:
            prediction = result.selected_rankings[query_id]
            selected_key = result.selected_config_by_outer[query_id]
            for neighbor_rank, (neighbor_index, similarity) in enumerate(
                zip(
                    prediction.neighbor_indices,
                    prediction.neighbor_similarities,
                    strict=True,
                ),
                start=1,
            ):
                writer.writerow(
                    (
                        query_id,
                        selected_key,
                        result.query_ids[neighbor_index],
                        f"{similarity:.12g}",
                        neighbor_rank,
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
                "raw_score",
                "global_prior",
                "neighbor_mass",
                "final_rank",
            )
        )
        for query_id in result.query_ids:
            prediction = result.selected_rankings[query_id]
            selected_key = result.selected_config_by_outer[query_id]
            for rank, model_index in enumerate(prediction.model_order, start=1):
                writer.writerow(
                    (
                        query_id,
                        selected_key,
                        result.model_ids[model_index],
                        f"{prediction.scores[model_index]:.12g}",
                        f"{prediction.global_prior[model_index]:.12g}",
                        f"{prediction.neighbor_mass:.12g}",
                        rank,
                    )
                )

    run_names = {
        "dense_nested": ("run.trec", result.config.run_id),
        "tfidf_fixed_k10": ("tfidf_control.trec", result.config.tfidf_run_id),
        "global_prior": ("global_prior_control.trec", result.config.prior_run_id),
    }
    for system, (filename, run_id) in run_names.items():
        write_trec_run(
            destination / filename,
            result.rankings[system],
            run_id=run_id,
        )

    with (destination / "per_query.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "selected_configuration",
                "dense_ndcg_at_10",
                "dense_reciprocal_rank",
                "tfidf_ndcg_at_10",
                "tfidf_reciprocal_rank",
                "prior_ndcg_at_10",
                "prior_reciprocal_rank",
                "positive_count",
                "grade_2_count",
            )
        )
        for query_id in result.query_ids:
            dense = result.evaluations["dense_nested"].per_query[query_id]
            tfidf = result.evaluations["tfidf_fixed_k10"].per_query[query_id]
            prior = result.evaluations["global_prior"].per_query[query_id]
            query_grades = result.data.qrels[query_id]
            writer.writerow(
                (
                    query_id,
                    result.selected_config_by_outer[query_id],
                    f"{dense.ndcg_at_k:.12g}",
                    f"{dense.reciprocal_rank:.12g}",
                    f"{tfidf.ndcg_at_k:.12g}",
                    f"{tfidf.reciprocal_rank:.12g}",
                    f"{prior.ndcg_at_k:.12g}",
                    f"{prior.reciprocal_rank:.12g}",
                    sum(value > 0 for value in query_grades.values()),
                    sum(value == 2 for value in query_grades.values()),
                )
            )

    _write_json(destination / "summary.json", result.summary)
    relative_outputs = (
        "config.json",
        "encoder_metadata.json",
        "inner_scores.tsv",
        "selections.tsv",
        "neighbors.tsv",
        "predictions.tsv",
        "run.trec",
        "tfidf_control.trec",
        "global_prior_control.trec",
        "per_query.tsv",
        "summary.json",
    )
    validation = {
        "inputs": {
            "complete_qrel_matrix": True,
            "query_count": len(result.query_ids),
            "candidate_count": len(result.model_ids),
        },
        "protocol": {
            "outer_prediction_count": len(result.query_ids),
            "inner_targets_per_outer": len(result.query_ids) - 1,
            "outer_qrel_excluded_from_selection": True,
            "dense_prior_matches_tfidf_prior": True,
            "prior_identity_checked_during_experiment": True,
        },
        "runs": {
            system: validate_trec_run(
                destination / filename,
                expected_query_ids=result.query_ids,
                expected_model_ids=result.model_ids,
            )
            for system, (filename, _run_id) in run_names.items()
        },
        "embeddings": {
            key: {
                "shape": list(value.shape),
                "sha256": array_sha256(value),
                "finite": bool(np.isfinite(value).all()),
                "normalized_or_zero": bool(
                    np.all(
                        np.isclose(np.linalg.norm(value, axis=1), 1.0, atol=1e-5)
                        | np.isclose(np.linalg.norm(value, axis=1), 0.0, atol=1e-8)
                    )
                ),
            }
            for key, value in result.embeddings.items()
        },
        "artifact_sha256": {
            name: _sha256(destination / name) for name in relative_outputs
        },
    }
    validation["runtime"] = {
        "artifact_generation_and_validation_seconds_before_receipt": (
            time.perf_counter() - artifact_started
        )
    }
    _write_json(destination / "validation.json", validation)

    paths = {
        name: str(destination / name) for name in (*relative_outputs, "validation.json")
    }
    paths.update(
        {
            f"embeddings/{key}.npy": str(embedding_dir / f"{key}.npy")
            for key in result.embeddings
        }
    )
    return paths


def _fixed_config_leave_one_out_rankings(
    *,
    similarities: NDArray[np.floating[Any]],
    grades: NDArray[np.floating[Any]],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    k: int,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for target_index, query_id in enumerate(query_ids):
        prediction = rank_target_from_similarity(
            target_index=target_index,
            additionally_excluded_indices=(),
            similarities=similarities,
            grades=grades,
            query_ids=query_ids,
            model_ids=model_ids,
            k=k,
        )
        rankings[query_id] = [model_ids[index] for index in prediction.model_order]
    return rankings


def _build_summary(
    *,
    config: DenseExperimentConfig,
    data: DevelopmentData,
    query_ids: Sequence[str],
    candidates: Sequence[DenseCandidate],
    encoder_metadata: Mapping[str, Mapping[str, Any]],
    inner_ndcg_by_outer: Mapping[str, NDArray[np.float64]],
    inner_mrr_by_outer: Mapping[str, NDArray[np.float64]],
    selected_config_by_outer: Mapping[str, str],
    evaluations: Mapping[str, EvaluationResult],
    fixed_config_metrics: Mapping[str, Mapping[str, float]],
    deployment_config: str,
    core_elapsed_seconds: float,
) -> dict[str, Any]:
    selected_counts = Counter(selected_config_by_outer.values())
    encoder_counts = Counter(
        key.rsplit("@", maxsplit=1)[0] for key in selected_config_by_outer.values()
    )
    k_counts = Counter(
        int(key.rsplit("@", maxsplit=1)[1]) for key in selected_config_by_outer.values()
    )
    comparisons = {
        baseline: _paired_comparison(
            baseline,
            evaluations[baseline],
            evaluations["dense_nested"],
            n_resamples=config.bootstrap_resamples,
            seed=config.seed + offset,
        )
        for offset, baseline in enumerate(("tfidf_fixed_k10", "global_prior"))
    }
    selected_inner_ndcg = [
        float(inner_ndcg_by_outer[selected_config_by_outer[query_id]][index])
        for index, query_id in enumerate(query_ids)
    ]
    selected_inner_mrr = [
        float(inner_mrr_by_outer[selected_config_by_outer[query_id]][index])
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
            "representation": "frozen_pretrained_dense_sentence_embedding",
            "embeddings_are_qrel_free_and_not_corpus_fitted": True,
            "embedding_normalization": "l2_float32",
            "similarity": "cosine_dot_product",
            "positive_neighbor_rule": "similarity_strictly_greater_than_zero",
            "aggregation": "similarity_weighted_mean",
            "selection_metric": "inner_macro_ndcg_at_10",
            "selection_tie_break": "declared_encoder_order_then_smaller_k",
            "grid_selection_provenance": config.grid_selection_provenance,
            "metric_gain": "linear_qrel_grade",
            "tie_break": "training_global_prior_then_model_id",
            "external_run": True,
            "external_run_reason": "pretrained encoders contain outside training data",
            "authoritative_device": config.device,
        },
        "data": {
            "query_count": len(query_ids),
            "candidate_count": len(data.models),
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
            {
                "key": candidate.key,
                "encoder_key": candidate.encoder_key,
                "neighbors": candidate.neighbors,
            }
            for candidate in candidates
        ],
        "encoder_metadata": dict(encoder_metadata),
        "metrics": metrics,
        "comparisons": comparisons,
        "selection": {
            "configuration_frequency": dict(sorted(selected_counts.items())),
            "encoder_frequency": dict(sorted(encoder_counts.items())),
            "neighbor_count_frequency": {
                str(key): value for key, value in sorted(k_counts.items())
            },
            "selected_inner_ndcg_mean": statistics.fmean(selected_inner_ndcg),
            "selected_inner_ndcg_min": min(selected_inner_ndcg),
            "selected_inner_ndcg_max": max(selected_inner_ndcg),
            "selected_inner_mrr_mean": statistics.fmean(selected_inner_mrr),
        },
        "deployment_selection": {
            "configuration": deployment_config,
            "criterion": "full_development_leave_one_out_ndcg_at_10",
            "purpose": "future_test_queries_only",
            "same_development_metrics_are_not_an_unbiased_evaluation": True,
        },
        "fixed_config_development_metrics": dict(fixed_config_metrics),
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
        "candidate": "dense_nested",
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


def _validate_config(config: DenseExperimentConfig) -> None:
    if not config.encoder_specs:
        raise ValueError("encoder_specs cannot be empty")
    encoder_keys = [spec.key for spec in config.encoder_specs]
    if len(set(encoder_keys)) != len(encoder_keys):
        raise ValueError("encoder keys must be unique")
    if not config.neighbor_counts:
        raise ValueError("neighbor_counts cannot be empty")
    if any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1
        for k in config.neighbor_counts
    ):
        raise ValueError("every neighbor count must be a positive integer")
    if len(set(config.neighbor_counts)) != len(config.neighbor_counts):
        raise ValueError("neighbor_counts must be unique")
    if config.cutoff != 10:
        raise ValueError("the TREC dense baseline currently supports cutoff=10 only")
    if config.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.torch_threads is not None and config.torch_threads < 1:
        raise ValueError("torch_threads must be positive or None")


def _lexical_ranks(values: Sequence[str]) -> NDArray[np.int64]:
    names = np.asarray(values)
    order = np.argsort(names, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
