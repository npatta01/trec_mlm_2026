"""Leakage-safe evaluation for source-query-first response alignment.

The scorer consumed by this module is qrel-free.  Development qrels enter only
the exact nested leave-one-query-out selector, fold-local ranking priors, and
effectiveness evaluation.  The primary selector sees answer-alignment scores;
the target-to-response signal is ranked only at the primary-selected ``K`` and
cannot affect selection, deployment choice, or the preregistered gates.
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
import resource
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
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
from trec_mlm.response_experiment import (
    DEFAULT_DENSE_PER_QUERY_PATH,
    DENSE_REFERENCE_MRR,
    DENSE_REFERENCE_NDCG_AT_10,
    ResponseExperimentConfig,
    ResponseExperimentResult,
    exact_nested_response_inner_scores,
    fold_local_prior,
    load_dense_per_query_reference,
    rank_model_scores,
    run_nested_response_experiment,
)
from trec_mlm.response_retrieval import (
    DEFAULT_RESPONSE_CANDIDATES,
    NORMALIZED_RESPONSE_CANDIDATES,
    OFFICIAL_DISCOVERY_SHA256,
)
from trec_mlm.runio import validate_trec_run, write_trec_run


SOURCE_QUERY_K_VALUES = (10, 50, 100)
TOP10SUM_RESPONSE_CANDIDATES = tuple(
    candidate
    for candidate in NORMALIZED_RESPONSE_CANDIDATES
    if candidate.aggregation == "top10_sum"
)
RESPONSE_CONTROL_CANDIDATES = (
    *DEFAULT_RESPONSE_CANDIDATES,
    *TOP10SUM_RESPONSE_CANDIDATES,
)

TFIDF_REFERENCE_NDCG_AT_10 = 0.2176542461618029
TFIDF_REFERENCE_MRR = 0.2776956693601721
PRIOR_REFERENCE_NDCG_AT_10 = 0.1385241823188521
PRIOR_REFERENCE_MRR = 0.20665310801034087
SUMSCORE_REFERENCE_NDCG_AT_10 = 0.2773577839029814
SUMSCORE_REFERENCE_MRR = 0.31753289707886323
TOP10SUM_REFERENCE_NDCG_AT_10 = 0.2892254323622026
TOP10SUM_REFERENCE_MRR = 0.3282546454493165
SUMSCORE_REFERENCE_RUN_SHA256 = (
    "51a30d186739b4ea25de81646203d5e371227cfccfa0b013a03bf9500efd5ed6"
)
TOP10SUM_REFERENCE_RUN_SHA256 = (
    "96c33b4cf9e3a955031297abd2079f16acd1a633e9447c2d9a2d37eb2c5dc24e"
)
DENSE_PER_QUERY_REFERENCE_SHA256 = (
    "ae4a0cf8bc7eb0642e1c0d315cafa3a56cf60b3d6bfb6ab63e80fb2d0de56437"
)
SUMSCORE_REFERENCE_SELECTION = {
    "bm25_rm3__query_response__basic__d500": 6,
    "bm25_rm3__response_only__aggressive__d500": 2,
    "bm25_rm3__response_only__basic__d500": 314,
}
TOP10SUM_REFERENCE_SELECTION = {
    "bm25_rm3__query_response__aggressive__d500__agg_top10_sum": 6,
    "bm25_rm3__query_response__basic__d500__agg_top10_sum": 316,
}

INVENTORY_RHO_REFERENCE = 0.8388
INVENTORY_RHO_GATE = 0.7388
LARGEST_QUARTILE_REFERENCE = 0.466
LARGEST_QUARTILE_MODEL_SHARE = 0.247
LARGEST_QUARTILE_GATE = 0.411
MAX_NDCG_DROP_FROM_SUMSCORE = 0.010

HEADLINE_BASELINES = (
    "response_sumscore_nested",
    "response_top10sum_nested",
    "dense_nested_reference",
    "tfidf_fixed_k10",
    "global_prior",
)


@dataclass(frozen=True, slots=True)
class SourceQueryExperimentConfig:
    """Resolved inputs and frozen evaluation settings."""

    query_path: str
    qrels_path: str
    discovery_path: str
    source_query_ks: tuple[int, ...] = SOURCE_QUERY_K_VALUES
    dense_per_query_path: str | None = DEFAULT_DENSE_PER_QUERY_PATH
    cutoff: int = 10
    run_id: str = "source_query_alignment_nested_loo"
    diagnostic_run_id: str = "target_response_same_k_diagnostic"
    sumscore_run_id: str = "response_sumscore_nested_loo_control"
    # Preserve the PR #8 run ID so the official control can be compared byte for byte.
    top10sum_run_id: str = "response_volume_normalized_nested_loo"
    tfidf_run_id: str = "qknn_tfidf_w12_k10_loo_control"
    prior_run_id: str = "global_prior_loo_control"
    bootstrap_resamples: int = 10_000
    seed: int = 2026
    show_progress: bool = False
    grid_selection_provenance: str = (
        "advisor_approved_source_query_first_protocol_locked_at_459e545"
    )


@dataclass(slots=True)
class SourceQueryExperimentResult:
    """Nested source-alignment predictions, controls, evidence, and gates."""

    config: SourceQueryExperimentConfig
    data: DevelopmentData
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    candidates: tuple[Any, ...]
    model_scores: NDArray[np.float64]
    diagnostic_model_scores: NDArray[np.float64]
    source_score_metadata: dict[str, Any]
    source_evidence_tensors: dict[str, NDArray[Any]]
    selected_source_ids: tuple[tuple[str, ...], ...]
    selected_source_keys: tuple[tuple[str, ...], ...]
    inner_ndcg_by_candidate: dict[str, NDArray[np.float64]]
    inner_mrr_by_candidate: dict[str, NDArray[np.float64]]
    selected_config_by_outer: dict[str, str]
    selected_primary_scores: dict[str, dict[str, float]]
    selected_diagnostic_scores: dict[str, dict[str, float]]
    selected_priors: dict[str, dict[str, float]]
    rankings: dict[str, dict[str, list[str]]]
    evaluations: dict[str, EvaluationResult]
    fixed_candidate_metrics: dict[str, dict[str, float]]
    deployment_config: str
    sumscore_control: ResponseExperimentResult
    top10sum_control: ResponseExperimentResult
    comparisons: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    decision: dict[str, Any]
    summary: dict[str, Any]


def run_source_query_experiment(
    config: SourceQueryExperimentConfig,
    *,
    score_result: Any | None = None,
    response_control_score_result: Any | None = None,
    dense_per_query_reference: Mapping[str, Any] | EvaluationResult | None = None,
) -> SourceQueryExperimentResult:
    """Run the locked source-query-first experiment without post-hoc variants.

    ``score_result`` and ``response_control_score_result`` are explicit fixture
    seams.  When omitted, both qrel-free tensors are computed from the official
    discovery corpus.  Supplying one does not weaken validation of its candidate,
    query, model, or tensor axes.
    """

    _validate_config(config)
    started = time.perf_counter()
    cpu_started = time.process_time()
    data = load_development_data(config.query_path, config.qrels_path)
    if len(data.queries) < 3:
        raise ValueError("nested leave-one-out evaluation needs at least three queries")
    observed_grades = {
        grade for grades in data.qrels.values() for grade in grades.values()
    }
    if not observed_grades.issubset({0.0, 1.0, 2.0}):
        raise ValueError("the TREC experiment expects qrel grades in {0, 1, 2}")

    query_ids = tuple(sorted(data.queries))
    model_ids = data.models
    _validate_official_config(config, len(query_ids), len(model_ids))
    grades = np.asarray(
        [
            [data.qrels[query_id][model_id] for model_id in model_ids]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )

    corpus: Any | None = None
    if score_result is None or response_control_score_result is None:
        from trec_mlm.response_retrieval import (  # pylint: disable=import-outside-toplevel
            load_discovery_records,
        )

        corpus = load_discovery_records(config.discovery_path, model_ids)
        if tuple(corpus.model_ids) != model_ids:
            raise ValueError("discovery and development model universes do not match")

    if score_result is None:
        from trec_mlm.source_query_first import (  # pylint: disable=import-outside-toplevel
            score_source_query_first,
        )

        score_result = score_source_query_first(
            corpus,
            query_ids=query_ids,
            query_texts=[data.queries[query_id] for query_id in query_ids],
            expected_model_ids=model_ids,
            show_progress=config.show_progress,
        )
    source_bundle = _coerce_source_score_result(
        score_result,
        expected_ks=config.source_query_ks,
        query_ids=query_ids,
        model_ids=model_ids,
    )
    candidates = source_bundle["candidates"]
    model_scores = source_bundle["model_scores"]
    diagnostic_model_scores = source_bundle["diagnostic_model_scores"]

    inner_ndcg_by_candidate: dict[str, NDArray[np.float64]] = {}
    inner_mrr_by_candidate: dict[str, NDArray[np.float64]] = {}
    for candidate_index, candidate in enumerate(candidates):
        key = source_candidate_key(candidate)
        inner_ndcg, inner_mrr = exact_nested_response_inner_scores(
            raw_scores=model_scores[candidate_index],
            grades=grades,
            query_ids=query_ids,
            model_ids=model_ids,
            cutoff=config.cutoff,
        )
        inner_ndcg_by_candidate[key] = inner_ndcg
        inner_mrr_by_candidate[key] = inner_mrr

    inner_matrix = np.column_stack(
        [inner_ndcg_by_candidate[source_candidate_key(value)] for value in candidates]
    )
    selected_indices = np.argmax(inner_matrix, axis=1)
    selected_config_by_outer = {
        query_id: source_candidate_key(candidates[int(selected_indices[index])])
        for index, query_id in enumerate(query_ids)
    }

    primary_rankings: dict[str, list[str]] = {}
    diagnostic_rankings: dict[str, list[str]] = {}
    prior_rankings: dict[str, list[str]] = {}
    selected_primary_scores: dict[str, dict[str, float]] = {}
    selected_diagnostic_scores: dict[str, dict[str, float]] = {}
    selected_priors: dict[str, dict[str, float]] = {}
    for query_index, query_id in enumerate(query_ids):
        candidate_index = int(selected_indices[query_index])
        primary = model_scores[candidate_index, query_index]
        diagnostic = diagnostic_model_scores[candidate_index, query_index]
        prior = fold_local_prior(grades, (query_index,))
        primary_order = rank_model_scores(primary, prior, model_ids)
        diagnostic_order = rank_model_scores(diagnostic, prior, model_ids)
        prior_order = rank_model_scores(np.zeros_like(prior), prior, model_ids)
        primary_rankings[query_id] = [model_ids[int(i)] for i in primary_order]
        diagnostic_rankings[query_id] = [model_ids[int(i)] for i in diagnostic_order]
        prior_rankings[query_id] = [model_ids[int(i)] for i in prior_order]
        selected_primary_scores[query_id] = {
            model_id: float(primary[index]) for index, model_id in enumerate(model_ids)
        }
        selected_diagnostic_scores[query_id] = {
            model_id: float(diagnostic[index])
            for index, model_id in enumerate(model_ids)
        }
        selected_priors[query_id] = {
            model_id: float(prior[index]) for index, model_id in enumerate(model_ids)
        }

    fixed_candidate_metrics: dict[str, dict[str, float]] = {}
    fixed_rankings: dict[str, dict[str, list[str]]] = {}
    for candidate_index, candidate in enumerate(candidates):
        key = source_candidate_key(candidate)
        rankings: dict[str, list[str]] = {}
        for query_index, query_id in enumerate(query_ids):
            prior = fold_local_prior(grades, (query_index,))
            order = rank_model_scores(
                model_scores[candidate_index, query_index], prior, model_ids
            )
            rankings[query_id] = [model_ids[int(i)] for i in order]
        fixed_rankings[key] = rankings
        evaluation = evaluate_rankings(rankings, data.qrels, k=config.cutoff)
        fixed_candidate_metrics[key] = {
            "ndcg_at_10": evaluation.macro_ndcg_at_k,
            "mrr": evaluation.mean_reciprocal_rank,
        }
    deployment_index = max(
        range(len(candidates)),
        key=lambda index: fixed_candidate_metrics[
            source_candidate_key(candidates[index])
        ]["ndcg_at_10"],
    )
    deployment_config = source_candidate_key(candidates[deployment_index])

    tfidf = run_leave_one_out_experiment(
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
    if prior_rankings != tfidf.rankings["global_prior"]:
        raise AssertionError(
            "source and TF-IDF pipelines produced different fold priors"
        )

    dense_reference = _resolve_dense_reference(
        config,
        query_ids=query_ids,
        supplied=dense_per_query_reference,
    )

    if response_control_score_result is None:
        from trec_mlm.response_retrieval import (  # pylint: disable=import-outside-toplevel
            score_response_candidates,
        )

        response_control_score_result = score_response_candidates(
            corpus,
            query_ids=query_ids,
            query_texts=[data.queries[query_id] for query_id in query_ids],
            expected_model_ids=model_ids,
            candidates=RESPONSE_CONTROL_CANDIDATES,
            show_progress=config.show_progress,
        )
    sumscore_scores, top10sum_scores = _split_response_control_scores(
        response_control_score_result,
        query_ids=query_ids,
        model_ids=model_ids,
        source_metadata=source_bundle["metadata"],
    )
    common_response_config = {
        "query_path": config.query_path,
        "qrels_path": config.qrels_path,
        "discovery_path": config.discovery_path,
        "dense_per_query_path": None,
        "cutoff": config.cutoff,
        "tfidf_run_id": config.tfidf_run_id,
        "prior_run_id": config.prior_run_id,
        "bootstrap_resamples": config.bootstrap_resamples,
        "seed": config.seed,
        "show_progress": config.show_progress,
    }
    sumscore_control = run_nested_response_experiment(
        ResponseExperimentConfig(
            **common_response_config,
            candidates=DEFAULT_RESPONSE_CANDIDATES,
            run_id=config.sumscore_run_id,
            grid_selection_provenance="frozen_pr5_sumscore_control",
        ),
        score_result=sumscore_scores,
        dense_per_query_reference=dense_reference,
    )
    top10sum_control = run_nested_response_experiment(
        ResponseExperimentConfig(
            **common_response_config,
            candidates=TOP10SUM_RESPONSE_CANDIDATES,
            run_id=config.top10sum_run_id,
            grid_selection_provenance="frozen_pr8_top10sum_control",
        ),
        score_result=top10sum_scores,
        dense_per_query_reference=dense_reference,
    )
    _assert_shared_controls(
        prior_rankings=prior_rankings,
        tfidf_rankings=tfidf.rankings["weighted_mean@10"],
        sumscore=sumscore_control,
        top10sum=top10sum_control,
    )

    rankings = {
        "source_alignment_nested": primary_rankings,
        "target_response_same_k": diagnostic_rankings,
        "response_sumscore_nested": sumscore_control.rankings["response_nested"],
        "response_top10sum_nested": top10sum_control.rankings["response_nested"],
        "tfidf_fixed_k10": tfidf.rankings["weighted_mean@10"],
        "global_prior": prior_rankings,
    }
    evaluations = {
        key: evaluate_rankings(value, data.qrels, k=config.cutoff)
        for key, value in rankings.items()
    }
    evaluations["dense_nested_reference"] = dense_reference

    comparisons = {
        baseline: _paired_comparison(
            baseline_name=baseline,
            candidate_name="source_alignment_nested",
            baseline=evaluations[baseline],
            candidate=evaluations["source_alignment_nested"],
            n_resamples=config.bootstrap_resamples,
            seed=config.seed + offset,
        )
        for offset, baseline in enumerate(HEADLINE_BASELINES)
    }
    diagnostic_comparison = _paired_comparison(
        baseline_name="source_alignment_nested",
        candidate_name="target_response_same_k",
        baseline=evaluations["source_alignment_nested"],
        candidate=evaluations["target_response_same_k"],
        n_resamples=config.bootstrap_resamples,
        seed=config.seed + 100,
    )

    diagnostics = _build_diagnostics(
        query_ids=query_ids,
        model_ids=model_ids,
        candidates=candidates,
        model_scores=model_scores,
        diagnostic_model_scores=diagnostic_model_scores,
        selected_indices=selected_indices,
        selected_config_by_outer=selected_config_by_outer,
        inner_ndcg_by_candidate=inner_ndcg_by_candidate,
        inner_mrr_by_candidate=inner_mrr_by_candidate,
        selected_priors=selected_priors,
        primary_rankings=primary_rankings,
        diagnostic_rankings=diagnostic_rankings,
        fixed_rankings=fixed_rankings,
        deployment_index=deployment_index,
        source_bundle=source_bundle,
        sumscore=sumscore_control,
        top10sum=top10sum_control,
        diagnostic_comparison=diagnostic_comparison,
    )
    decision = _build_decision(evaluations, comparisons, diagnostics)
    controls = _validate_control_references(
        query_ids=query_ids,
        model_ids=model_ids,
        evaluations=evaluations,
        sumscore=sumscore_control,
        top10sum=top10sum_control,
    )
    summary = _build_summary(
        config=config,
        data=data,
        query_ids=query_ids,
        candidates=candidates,
        source_metadata=source_bundle["metadata"],
        inner_ndcg_by_candidate=inner_ndcg_by_candidate,
        inner_mrr_by_candidate=inner_mrr_by_candidate,
        selected_config_by_outer=selected_config_by_outer,
        evaluations=evaluations,
        comparisons=comparisons,
        fixed_candidate_metrics=fixed_candidate_metrics,
        deployment_config=deployment_config,
        diagnostics=diagnostics,
        decision=decision,
        controls=controls,
        elapsed_seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu_started,
    )
    return SourceQueryExperimentResult(
        config=config,
        data=data,
        query_ids=query_ids,
        model_ids=model_ids,
        candidates=candidates,
        model_scores=model_scores,
        diagnostic_model_scores=diagnostic_model_scores,
        source_score_metadata=source_bundle["metadata"],
        source_evidence_tensors=source_bundle["evidence_tensors"],
        selected_source_ids=source_bundle["selected_source_ids"],
        selected_source_keys=source_bundle["selected_source_keys"],
        inner_ndcg_by_candidate=inner_ndcg_by_candidate,
        inner_mrr_by_candidate=inner_mrr_by_candidate,
        selected_config_by_outer=selected_config_by_outer,
        selected_primary_scores=selected_primary_scores,
        selected_diagnostic_scores=selected_diagnostic_scores,
        selected_priors=selected_priors,
        rankings=rankings,
        evaluations=evaluations,
        fixed_candidate_metrics=fixed_candidate_metrics,
        deployment_config=deployment_config,
        sumscore_control=sumscore_control,
        top10sum_control=top10sum_control,
        comparisons=comparisons,
        diagnostics=diagnostics,
        decision=decision,
        summary=summary,
    )


def write_source_query_experiment_artifacts(
    result: SourceQueryExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write deterministic evidence, six complete runs, and validation receipts."""

    started = time.perf_counter()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    execution_provenance = _execution_provenance(result)
    resolved_config = {
        **asdict(result.config),
        "candidate_keys": [source_candidate_key(value) for value in result.candidates],
        "execution_provenance": execution_provenance,
    }
    _write_json(destination / "config.json", resolved_config)
    _write_json(
        destination / "source_score_metadata.json", result.source_score_metadata
    )
    _write_json(destination / "diagnostics.json", result.diagnostics)

    tensor_payload = {
        "model_scores": result.model_scores,
        "diagnostic_model_scores": result.diagnostic_model_scores,
        **result.source_evidence_tensors,
    }
    np.savez_compressed(destination / "score_tensors.npz", **tensor_payload)
    _write_selected_sources(result, destination / "selected_sources.tsv")
    _write_inner_scores(result, destination / "inner_scores.tsv")
    _write_selections(result, destination / "selections.tsv")
    _write_predictions(result, destination / "predictions.tsv")
    _write_per_query(result, destination / "per_query.tsv")
    _write_query_diagnostics(result, destination / "query_diagnostics.tsv")

    run_specs = {
        "source_alignment_nested": (
            "source_alignment_run.trec",
            result.config.run_id,
        ),
        "target_response_same_k": (
            "target_response_diagnostic.trec",
            result.config.diagnostic_run_id,
        ),
        "response_sumscore_nested": (
            "sumscore_control.trec",
            result.config.sumscore_run_id,
        ),
        "response_top10sum_nested": (
            "top10sum_control.trec",
            result.config.top10sum_run_id,
        ),
        "tfidf_fixed_k10": ("tfidf_control.trec", result.config.tfidf_run_id),
        "global_prior": ("global_prior_control.trec", result.config.prior_run_id),
    }
    for system, (filename, run_id) in run_specs.items():
        write_trec_run(destination / filename, result.rankings[system], run_id=run_id)

    _write_json(destination / "summary.json", result.summary)
    artifact_names = (
        "config.json",
        "source_score_metadata.json",
        "diagnostics.json",
        "score_tensors.npz",
        "selected_sources.tsv",
        "inner_scores.tsv",
        "selections.tsv",
        "predictions.tsv",
        "per_query.tsv",
        "query_diagnostics.tsv",
        *(filename for filename, _run_id in run_specs.values()),
        "summary.json",
    )
    run_validation = {
        system: validate_trec_run(
            destination / filename,
            expected_query_ids=result.query_ids,
            expected_model_ids=result.model_ids,
        )
        for system, (filename, _run_id) in run_specs.items()
    }
    control_run_parity = _control_run_parity(result, destination)
    decision_receipt = _decision_receipt(result)
    validation = {
        "inputs": {
            "query_count": len(result.query_ids),
            "candidate_model_count": len(result.model_ids),
            "source_candidate_count": len(result.candidates),
            "sumscore_candidate_count": len(result.sumscore_control.candidates),
            "top10sum_candidate_count": len(result.top10sum_control.candidates),
            "complete_qrel_matrix": True,
        },
        "protocol": {
            "outer_prediction_count": len(result.query_ids),
            "inner_targets_per_outer": len(result.query_ids) - 1,
            "outer_qrel_excluded_from_selection": True,
            "inner_prior_excludes_outer_and_inner": True,
            "candidate_keys": [
                source_candidate_key(value) for value in result.candidates
            ],
            "target_response_excluded_from_selection": True,
            "target_response_uses_primary_selected_k": True,
            "sumscore_selected_independently": True,
            "top10sum_selected_independently": True,
            "primary_score_tensor_shape": list(result.model_scores.shape),
            "diagnostic_score_tensor_shape": list(result.diagnostic_model_scores.shape),
            "primary_score_tensor_sha256": _array_sha256(result.model_scores),
            "diagnostic_score_tensor_sha256": _array_sha256(
                result.diagnostic_model_scores
            ),
            "four_gate_count": len(result.decision["gates"]),
            "bootstrap_resamples": result.config.bootstrap_resamples,
            "bootstrap_base_seed": result.config.seed,
            "bootstrap_baseline_order": list(HEADLINE_BASELINES),
            "dense_per_query_reference_sha256": (
                DENSE_PER_QUERY_REFERENCE_SHA256
                if len(result.query_ids) == 322
                else None
            ),
        },
        "controls": {
            **result.summary["controls"],
            "run_parity": control_run_parity,
        },
        "runs": run_validation,
        "execution_provenance": execution_provenance,
        "determinism": {
            "decision_receipt_sha256": decision_receipt,
            "source_signal_tensor_sha256": result.source_score_metadata.get(
                "signal_tensor_sha256"
            ),
            "source_score_tensor_sha256": result.source_score_metadata.get(
                "score_tensor_sha256"
            ),
            "all_runs_complete": all(
                value["valid"] for value in run_validation.values()
            ),
            "decision_relevant_outputs_are_canonically_hashed": True,
        },
        "artifact_sha256": {
            name: _sha256(destination / name) for name in artifact_names
        },
        "runtime": {
            "artifact_generation_and_validation_seconds_before_receipt": (
                time.perf_counter() - started
            )
        },
    }
    _write_json(destination / "validation.json", validation)
    return {
        name: str(destination / name) for name in (*artifact_names, "validation.json")
    }


def source_candidate_key(candidate: Any) -> str:
    """Return and validate the locked source-alignment candidate key."""

    k = source_candidate_k(candidate)
    expected = f"source_alignment__k{k}"
    observed = getattr(candidate, "key", expected)
    if observed != expected:
        raise ValueError(
            f"source candidate key {observed!r} does not match K={k}: {expected!r}"
        )
    return expected


def source_candidate_k(candidate: Any) -> int:
    value = getattr(candidate, "k", None)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise TypeError("source candidate must expose a positive integer .k")
    return int(value)


def _validate_config(config: SourceQueryExperimentConfig) -> None:
    if tuple(config.source_query_ks) != SOURCE_QUERY_K_VALUES:
        raise ValueError("source_query_ks must equal the frozen (10, 50, 100) grid")
    if config.cutoff != 10:
        raise ValueError("the source-query experiment supports cutoff=10 only")
    if (
        isinstance(config.bootstrap_resamples, bool)
        or not isinstance(config.bootstrap_resamples, int)
        or config.bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise ValueError("seed must be an integer")


def _validate_official_config(
    config: SourceQueryExperimentConfig, query_count: int, model_count: int
) -> None:
    if (query_count, model_count) != (322, 150):
        return
    if config.bootstrap_resamples != 10_000:
        raise ValueError("official scoring requires exactly 10,000 bootstrap resamples")
    if config.seed != 2026:
        raise ValueError("official scoring requires the frozen seed 2026")


def _coerce_source_score_result(
    score_result: Any,
    *,
    expected_ks: Sequence[int],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    candidates = tuple(score_result.candidates)
    observed_ks = tuple(source_candidate_k(value) for value in candidates)
    if observed_ks != tuple(expected_ks):
        raise ValueError(
            f"source score candidate order is {observed_ks!r}; expected {tuple(expected_ks)!r}"
        )
    # Also validates every exposed key.
    tuple(source_candidate_key(value) for value in candidates)
    if tuple(score_result.query_ids) != tuple(query_ids):
        raise ValueError("source score query order must match sorted development IDs")
    if tuple(score_result.model_ids) != tuple(model_ids):
        raise ValueError("source score model order must match development models")

    shape = (len(candidates), len(query_ids), len(model_ids))
    model_scores = _required_array(
        score_result, "model_scores", shape, np.float64, nonnegative=True
    )
    diagnostic_scores = _required_array(
        score_result,
        "diagnostic_model_scores",
        shape,
        np.float64,
        nonnegative=True,
    )
    maximum_k = max(expected_ks)
    selected_source_ids = _coerce_source_rows(
        getattr(score_result, "selected_source_ids"),
        query_count=len(query_ids),
        maximum_k=maximum_k,
        name="selected_source_ids",
    )
    selected_source_keys = _coerce_source_rows(
        getattr(score_result, "selected_source_keys"),
        query_count=len(query_ids),
        maximum_k=maximum_k,
        name="selected_source_keys",
    )

    evidence_specs: tuple[tuple[str, tuple[int, ...], Any, bool], ...] = (
        ("source_relevance_raw", (len(query_ids), maximum_k), np.float64, True),
        (
            "source_weights",
            (len(candidates), len(query_ids), maximum_k),
            np.float64,
            True,
        ),
        (
            "answer_alignment_raw",
            (len(query_ids), maximum_k, len(model_ids)),
            np.float64,
            True,
        ),
        (
            "answer_alignment_normalized",
            (len(query_ids), maximum_k, len(model_ids)),
            np.float64,
            True,
        ),
        (
            "target_response_raw",
            (len(query_ids), maximum_k, len(model_ids)),
            np.float64,
            True,
        ),
        (
            "target_response_normalized",
            (len(query_ids), maximum_k, len(model_ids)),
            np.float64,
            True,
        ),
        ("opportunity_counts", shape, np.int32, True),
        ("valid_opportunity_counts", shape, np.int32, True),
        (
            "validity_codes",
            (len(query_ids), maximum_k, len(model_ids)),
            np.uint8,
            True,
        ),
        ("exact_mask_canonical_counts", (len(query_ids),), np.int32, True),
        ("exact_mask_source_id_counts", (len(query_ids),), np.int32, True),
        ("positive_retrieval_counts", (len(query_ids),), np.int32, True),
        (
            "source_weight_sums",
            (len(candidates), len(query_ids)),
            np.float64,
            True,
        ),
    )
    evidence_tensors: dict[str, NDArray[Any]] = {}
    for name, expected_shape, dtype, nonnegative in evidence_specs:
        if not hasattr(score_result, name):
            raise ValueError(f"source score result is missing required tensor {name!r}")
        evidence_tensors[name] = _required_array(
            score_result,
            name,
            expected_shape,
            dtype,
            nonnegative=nonnegative,
        )
    source_relevance = evidence_tensors.get("source_relevance_raw")
    if source_relevance is not None:
        if np.any(source_relevance <= 0.0):
            raise ValueError("all selected source-query BM25 scores must be positive")
        if np.any(np.diff(source_relevance, axis=1) > 0.0):
            raise ValueError("selected source-query scores must be nonincreasing")
        for query_index, ids in enumerate(selected_source_ids):
            scores = source_relevance[query_index]
            if any(
                scores[index] == scores[index + 1] and ids[index] > ids[index + 1]
                for index in range(len(ids) - 1)
            ):
                raise ValueError("tied source-query scores must use source ID order")
    source_weights = evidence_tensors.get("source_weights")
    source_weight_sums = evidence_tensors.get("source_weight_sums")
    if source_weights is not None:
        for index, k in enumerate(expected_ks):
            if not np.all(source_weights[index, :, k:] == 0.0):
                raise ValueError(f"source weights beyond K={k} must be zero")
        observed_sums = source_weights.sum(axis=2, dtype=np.float64)
        if not np.allclose(observed_sums, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                "each candidate/query source-weight vector must sum to one"
            )
        if source_weight_sums is not None and not np.allclose(
            source_weight_sums, observed_sums, rtol=0.0, atol=1e-12
        ):
            raise ValueError("source_weight_sums disagree with source_weights")
    answer_normalized = evidence_tensors.get("answer_alignment_normalized")
    target_normalized = evidence_tensors.get("target_response_normalized")
    for name, values in (
        ("answer_alignment_normalized", answer_normalized),
        ("target_response_normalized", target_normalized),
    ):
        if values is None:
            continue
        masses = values.sum(axis=2, dtype=np.float64)
        if not np.all(
            np.isclose(masses, 0.0, rtol=0.0, atol=1e-12)
            | np.isclose(masses, 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{name} rows must have L1 mass zero or one")
    if source_weights is not None and answer_normalized is not None:
        reconstructed = np.einsum(
            "cqk,qkm->cqm", source_weights, answer_normalized, dtype=np.float64
        )
        if not np.allclose(reconstructed, model_scores, rtol=0.0, atol=1e-12):
            raise ValueError("primary model scores do not reconstruct from w and a")
    if source_weights is not None and target_normalized is not None:
        reconstructed = np.einsum(
            "cqk,qkm->cqm", source_weights, target_normalized, dtype=np.float64
        )
        if not np.allclose(reconstructed, diagnostic_scores, rtol=0.0, atol=1e-12):
            raise ValueError("diagnostic scores do not reconstruct from w and t")
    opportunity_counts = evidence_tensors.get("opportunity_counts")
    if opportunity_counts is not None:
        for index, k in enumerate(expected_ks):
            if not np.all(opportunity_counts[index] == k):
                raise ValueError(f"opportunity_counts for K={k} are not uniformly {k}")
    valid_opportunity_counts = evidence_tensors.get("valid_opportunity_counts")
    if valid_opportunity_counts is not None:
        if opportunity_counts is not None and np.any(
            valid_opportunity_counts > opportunity_counts
        ):
            raise ValueError("valid opportunity counts exceed total opportunities")
    validity_codes = evidence_tensors.get("validity_codes")
    if validity_codes is not None:
        invalid = validity_codes != 0
        for name in (
            "answer_alignment_raw",
            "answer_alignment_normalized",
            "target_response_raw",
            "target_response_normalized",
        ):
            values = evidence_tensors.get(name)
            if values is not None and np.any(values[invalid] != 0.0):
                raise ValueError(f"invalid response cells must be zero in {name}")

    metadata_names = (
        "validity_reason_names",
        "basic_retained_document_counts",
        "basic_retained_character_counts",
        "basic_retained_token_counts",
        "basic_plus_prefix_retained_document_counts",
        "basic_plus_prefix_retained_character_counts",
        "basic_plus_prefix_retained_token_counts",
        "inventory_filter_names",
        "retained_document_counts",
        "retained_character_counts",
        "retained_token_counts",
        "canonical_audit",
        "filter_audit",
        "filter_audits",
        "index_diagnostics",
        "analyzer_diagnostics",
        "source_sha256",
        "corpus_sha256",
        "canonical_query_sha256",
        "response_cell_sha256",
        "configuration_sha256",
        "signal_tensor_sha256",
        "score_tensor_sha256",
    )
    missing_metadata = [
        name for name in metadata_names if not hasattr(score_result, name)
    ]
    if missing_metadata:
        raise ValueError(
            f"source score result lacks required metadata: {missing_metadata!r}"
        )
    metadata = {name: _jsonable(getattr(score_result, name)) for name in metadata_names}
    for name in (
        "source_sha256",
        "corpus_sha256",
        "canonical_query_sha256",
        "response_cell_sha256",
        "configuration_sha256",
        "signal_tensor_sha256",
        "score_tensor_sha256",
    ):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"source score metadata {name!r} must be non-empty")
    metadata["primary_model_score_tensor_sha256"] = _array_sha256(model_scores)
    metadata["diagnostic_model_score_tensor_sha256"] = _array_sha256(diagnostic_scores)
    return {
        "candidates": candidates,
        "model_scores": model_scores,
        "diagnostic_model_scores": diagnostic_scores,
        "selected_source_ids": selected_source_ids,
        "selected_source_keys": selected_source_keys,
        "evidence_tensors": evidence_tensors,
        "metadata": metadata,
    }


def _coerce_source_rows(
    values: Any,
    *,
    query_count: int,
    maximum_k: int,
    name: str,
) -> tuple[tuple[str, ...], ...]:
    rows = tuple(tuple(str(value) for value in row) for row in values)
    if len(rows) != query_count or any(len(row) != maximum_k for row in rows):
        raise ValueError(
            f"{name} must contain {query_count} rows of exactly {maximum_k} values"
        )
    if any(not value for row in rows for value in row):
        raise ValueError(f"{name} values must be non-empty strings")
    if any(len(set(row)) != len(row) for row in rows):
        raise ValueError(f"{name} must be unique within each target query")
    return rows


def _required_array(
    source: Any,
    name: str,
    expected_shape: tuple[int, ...],
    dtype: Any,
    *,
    nonnegative: bool,
) -> NDArray[Any]:
    array = np.asarray(getattr(source, name), dtype=dtype)
    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {expected_shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _split_response_control_scores(
    score_result: Any,
    *,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    source_metadata: Mapping[str, Any],
) -> tuple[Any, Any]:
    candidates = tuple(score_result.candidates)
    actual_keys = tuple(candidate.key for candidate in candidates)
    expected_keys = tuple(candidate.key for candidate in RESPONSE_CONTROL_CANDIDATES)
    if actual_keys != expected_keys:
        raise ValueError(
            "response control score candidates do not match frozen SumScore+Top10Sum order"
        )
    if tuple(score_result.query_ids) != tuple(query_ids):
        raise ValueError("response control query order does not match development IDs")
    if tuple(score_result.model_ids) != tuple(model_ids):
        raise ValueError(
            "response control model order does not match development models"
        )
    if (len(query_ids), len(model_ids)) == (322, 150):
        for name in ("source_sha256", "corpus_sha256", "analyzer_diagnostics"):
            if not hasattr(score_result, name):
                raise ValueError(f"official response control tensor lacks {name!r}")
        if (
            source_metadata["source_sha256"] != OFFICIAL_DISCOVERY_SHA256
            or str(score_result.source_sha256) != OFFICIAL_DISCOVERY_SHA256
        ):
            raise ValueError(
                "official score tensors do not use the locked discovery file"
            )
        if source_metadata["corpus_sha256"] != str(score_result.corpus_sha256):
            raise ValueError(
                "source and response controls use different corpus records"
            )
        source_analyzer = source_metadata["analyzer_diagnostics"]
        response_analyzer = _jsonable(score_result.analyzer_diagnostics)
        if source_analyzer.get("analyzer_sha256") != response_analyzer.get(
            "analyzer_sha256"
        ):
            raise ValueError("source and response controls use different analyzers")
    split = len(DEFAULT_RESPONSE_CANDIDATES)
    return (
        _slice_score_result(score_result, range(split)),
        _slice_score_result(score_result, range(split, len(candidates))),
    )


def _slice_score_result(score_result: Any, indices: Sequence[int]) -> Any:
    positions = np.asarray(tuple(indices), dtype=np.int64)
    candidates = tuple(score_result.candidates[int(index)] for index in positions)
    payload: dict[str, Any] = {
        "candidates": candidates,
        "query_ids": tuple(score_result.query_ids),
        "model_ids": tuple(score_result.model_ids),
        "model_scores": np.asarray(score_result.model_scores)[positions],
    }
    for name in ("hit_counts", "mean_hit_scores", "overlap_document_counts"):
        if hasattr(score_result, name):
            payload[name] = np.asarray(getattr(score_result, name))[positions]
    for name in (
        "filter_audits",
        "inventory_filter_names",
        "retained_document_counts",
        "retained_character_counts",
        "retained_token_counts",
        "index_diagnostics",
        "analyzer_diagnostics",
        "source_sha256",
        "corpus_sha256",
    ):
        if hasattr(score_result, name):
            payload[name] = getattr(score_result, name)
    parent_configuration = getattr(score_result, "configuration_sha256", None)
    payload["configuration_sha256"] = _hash_json(
        {
            "parent_configuration_sha256": parent_configuration,
            "candidate_keys": [candidate.key for candidate in candidates],
        }
    )
    score_arrays = [payload["model_scores"]]
    score_arrays.extend(
        payload[name]
        for name in ("hit_counts", "mean_hit_scores", "overlap_document_counts")
        if name in payload
    )
    payload["score_tensor_sha256"] = _raw_arrays_sha256(*score_arrays)
    return SimpleNamespace(**payload)


def _resolve_dense_reference(
    config: SourceQueryExperimentConfig,
    *,
    query_ids: Sequence[str],
    supplied: Mapping[str, Any] | EvaluationResult | None,
) -> EvaluationResult:
    if supplied is not None:
        evaluation = _coerce_dense_reference(supplied, expected_query_ids=query_ids)
        if len(query_ids) == 322:
            reference_path = Path(
                config.dense_per_query_path or DEFAULT_DENSE_PER_QUERY_PATH
            )
            reference = _load_locked_dense_reference(reference_path, query_ids)
            if evaluation.per_query != reference.per_query:
                raise ValueError(
                    "supplied dense per-query vector differs from the frozen reference"
                )
        return evaluation
    if config.dense_per_query_path is None:
        raise ValueError("dense reference or dense_per_query_path is required")
    path = Path(config.dense_per_query_path)
    if len(query_ids) == 322:
        return _load_locked_dense_reference(path, query_ids)
    return load_dense_per_query_reference(
        path,
        expected_query_ids=query_ids,
    )


def _load_locked_dense_reference(
    path: Path, query_ids: Sequence[str]
) -> EvaluationResult:
    if _sha256(path) != DENSE_PER_QUERY_REFERENCE_SHA256:
        raise ValueError(
            "dense per-query reference does not match the frozen file hash"
        )
    return load_dense_per_query_reference(
        path,
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
        per_query = supplied.per_query
    else:
        per_query: dict[str, QueryMetrics] = {}
        for query_id, value in supplied.items():
            if isinstance(value, QueryMetrics):
                metric = value
            elif isinstance(value, Mapping):
                ndcg = value.get("ndcg_at_k", value.get("ndcg_at_10"))
                rr = value.get("reciprocal_rank", value.get("mrr"))
                if ndcg is None or rr is None:
                    raise ValueError(f"dense reference for {query_id!r} lacks metrics")
                metric = QueryMetrics(float(ndcg), float(rr))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) != 2:
                    raise ValueError("dense metric tuples must contain two values")
                metric = QueryMetrics(float(value[0]), float(value[1]))
            else:
                raise TypeError(f"unsupported dense reference for {query_id!r}")
            per_query[str(query_id)] = metric
    if set(per_query) != set(expected_query_ids):
        raise ValueError("dense reference query IDs do not match development queries")
    ordered = {query_id: per_query[query_id] for query_id in expected_query_ids}
    if any(
        not math.isfinite(value.ndcg_at_k)
        or not 0.0 <= value.ndcg_at_k <= 1.0
        or not math.isfinite(value.reciprocal_rank)
        or not 0.0 <= value.reciprocal_rank <= 1.0
        for value in ordered.values()
    ):
        raise ValueError("dense reference metrics must be finite values in [0, 1]")
    evaluation = EvaluationResult(
        cutoff=10,
        per_query=ordered,
        macro_ndcg_at_k=statistics.fmean(value.ndcg_at_k for value in ordered.values()),
        mean_reciprocal_rank=statistics.fmean(
            value.reciprocal_rank for value in ordered.values()
        ),
    )
    if isinstance(supplied, EvaluationResult) and (
        not math.isclose(
            supplied.macro_ndcg_at_k,
            evaluation.macro_ndcg_at_k,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            supplied.mean_reciprocal_rank,
            evaluation.mean_reciprocal_rank,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("dense macro metrics do not match its per-query metrics")
    return evaluation


def _assert_shared_controls(
    *,
    prior_rankings: Mapping[str, Sequence[str]],
    tfidf_rankings: Mapping[str, Sequence[str]],
    sumscore: ResponseExperimentResult,
    top10sum: ResponseExperimentResult,
) -> None:
    for name, control in (("SumScore", sumscore), ("Top10Sum", top10sum)):
        if control.rankings["global_prior"] != prior_rankings:
            raise AssertionError(f"{name} global prior differs from primary prior")
        if control.rankings["tfidf_fixed_k10"] != tfidf_rankings:
            raise AssertionError(f"{name} TF-IDF control differs from primary TF-IDF")
    if sumscore.rankings["global_prior"] != top10sum.rankings["global_prior"]:
        raise AssertionError("response controls disagree on global prior")


def _paired_comparison(
    *,
    baseline_name: str,
    candidate_name: str,
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_ndcg = {
        query_id: baseline.per_query[query_id].ndcg_at_k
        for query_id in sorted(baseline.per_query)
    }
    candidate_ndcg = {
        query_id: candidate.per_query[query_id].ndcg_at_k
        for query_id in sorted(candidate.per_query)
    }
    baseline_mrr = {
        query_id: baseline.per_query[query_id].reciprocal_rank
        for query_id in sorted(baseline.per_query)
    }
    candidate_mrr = {
        query_id: candidate.per_query[query_id].reciprocal_rank
        for query_id in sorted(candidate.per_query)
    }
    return {
        "candidate": candidate_name,
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
                baseline_mrr,
                candidate_mrr,
                n_resamples=n_resamples,
                seed=seed + 10_000,
            )
        ),
        "ndcg_at_10_win_tie_loss": _win_tie_loss(baseline_ndcg, candidate_ndcg),
        "mrr_win_tie_loss": _win_tie_loss(baseline_mrr, candidate_mrr),
    }


def _win_tie_loss(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, int]:
    if set(baseline) != set(candidate):
        raise ValueError("win/tie/loss inputs must have identical query IDs")
    differences = [candidate[key] - baseline[key] for key in sorted(baseline)]
    wins = sum(value > tolerance for value in differences)
    losses = sum(value < -tolerance for value in differences)
    return {"wins": wins, "ties": len(differences) - wins - losses, "losses": losses}


def _build_diagnostics(
    *,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    candidates: Sequence[Any],
    model_scores: NDArray[np.float64],
    diagnostic_model_scores: NDArray[np.float64],
    selected_indices: NDArray[np.int64],
    selected_config_by_outer: Mapping[str, str],
    inner_ndcg_by_candidate: Mapping[str, NDArray[np.float64]],
    inner_mrr_by_candidate: Mapping[str, NDArray[np.float64]],
    selected_priors: Mapping[str, Mapping[str, float]],
    primary_rankings: Mapping[str, Sequence[str]],
    diagnostic_rankings: Mapping[str, Sequence[str]],
    fixed_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    deployment_index: int,
    source_bundle: Mapping[str, Any],
    sumscore: ResponseExperimentResult,
    top10sum: ResponseExperimentResult,
    diagnostic_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = source_bundle["evidence_tensors"]
    metadata = source_bundle["metadata"]
    inventory = _basic_inventory(metadata, len(model_ids))
    deployment_key = source_candidate_key(candidates[deployment_index])
    deployment = _deployment_behavior(
        inventory=inventory,
        scores=model_scores[deployment_index],
        rankings=fixed_rankings[deployment_key],
        model_ids=model_ids,
        configuration=deployment_key,
    )
    outer = _outer_fold_behavior(
        inventory=inventory,
        query_ids=query_ids,
        model_ids=model_ids,
        model_scores=model_scores,
        diagnostic_scores=diagnostic_model_scores,
        selected_indices=selected_indices,
        selected_priors=selected_priors,
        rankings=primary_rankings,
    )

    opportunity_counts = evidence.get("opportunity_counts")
    valid_opportunity_counts = evidence.get("valid_opportunity_counts")
    validity_codes = evidence.get("validity_codes")
    validity_names = tuple(metadata.get("validity_reason_names", ()))
    valid_code = validity_names.index("valid") if "valid" in validity_names else 0
    selected_invalid_shares: list[float] = []
    validity_counts: Counter[str] = Counter()
    if validity_codes is not None:
        for query_index, selected_index in enumerate(selected_indices):
            k = source_candidate_k(candidates[int(selected_index)])
            cells = validity_codes[query_index, :k]
            selected_invalid_shares.append(float(np.mean(cells != valid_code)))
            for code, count in zip(*np.unique(cells, return_counts=True), strict=True):
                name = (
                    validity_names[int(code)]
                    if int(code) < len(validity_names)
                    else f"code_{int(code)}"
                )
                validity_counts[name] += int(count)

    source_weight_sums = evidence.get("source_weight_sums")
    opportunities = {
        "candidate_k_values": [source_candidate_k(value) for value in candidates],
        "every_model_receives_same_k": (
            bool(
                opportunity_counts is not None
                and all(
                    np.all(opportunity_counts[index] == source_candidate_k(candidate))
                    for index, candidate in enumerate(candidates)
                )
            )
            if opportunity_counts is not None
            else None
        ),
        "selected_invalid_cell_share": _mean_and_range(selected_invalid_shares),
        "selected_validity_reason_counts": dict(sorted(validity_counts.items())),
        "valid_opportunity_count": _array_summary(valid_opportunity_counts),
        "source_weight_sum": (
            {
                "min": float(np.min(source_weight_sums)),
                "max": float(np.max(source_weight_sums)),
                "mean": float(np.mean(source_weight_sums)),
            }
            if source_weight_sums is not None
            else None
        ),
        "positive_retrieval_count": _array_summary(
            evidence.get("positive_retrieval_counts")
        ),
        "exact_mask_canonical_count": _array_summary(
            evidence.get("exact_mask_canonical_counts")
        ),
        "exact_mask_source_id_count": _array_summary(
            evidence.get("exact_mask_source_id_counts")
        ),
        "canonical_audit": metadata.get("canonical_audit"),
        "filter_audit": metadata.get("filter_audit"),
        "basic_inventory": inventory.tolist(),
        "basic_plus_prefix_inventory": metadata.get(
            "basic_plus_prefix_retained_document_counts"
        ),
    }
    selected_counts = Counter(selected_config_by_outer.values())
    per_query_selected_k, inner_margins = _per_query_selected_k_diagnostics(
        query_ids=query_ids,
        model_ids=model_ids,
        candidates=candidates,
        model_scores=model_scores,
        diagnostic_scores=diagnostic_model_scores,
        selected_indices=selected_indices,
        inner_ndcg_by_candidate=inner_ndcg_by_candidate,
        inner_mrr_by_candidate=inner_mrr_by_candidate,
        selected_priors=selected_priors,
        rankings=primary_rankings,
        evidence=evidence,
        validity_names=validity_names,
        valid_code=valid_code,
    )
    selection = {
        "configuration_frequency": dict(sorted(selected_counts.items())),
        "inner_ndcg_winner_margin": _mean_and_range(inner_margins),
        "sumscore_configuration_frequency": dict(
            sorted(Counter(sumscore.selected_config_by_outer.values()).items())
        ),
        "top10sum_configuration_frequency": dict(
            sorted(Counter(top10sum.selected_config_by_outer.values()).items())
        ),
    }
    target_response = {
        "selection_role": "same_primary_selected_k_nonselecting_non_gating",
        "uses_primary_selected_k": True,
        "can_change_primary_selection": False,
        "zero_score_model_count": _selected_zero_summary(
            diagnostic_model_scores, selected_indices
        ),
        "comparison_to_primary": dict(diagnostic_comparison),
        "ranking_hash": _hash_json(diagnostic_rankings),
    }
    return {
        "opportunities": opportunities,
        "selection": selection,
        "deployment_behavior": deployment,
        "outer_fold_behavior": outer,
        "target_response_diagnostic": target_response,
        "per_query_selected_k": per_query_selected_k,
        "primary_zero_score_model_count": _selected_zero_summary(
            model_scores, selected_indices
        ),
        "control_behavior": {
            "sumscore_deployment_configuration": sumscore.deployment_config,
            "top10sum_deployment_configuration": top10sum.deployment_config,
        },
    }


def _per_query_selected_k_diagnostics(
    *,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    candidates: Sequence[Any],
    model_scores: NDArray[np.float64],
    diagnostic_scores: NDArray[np.float64],
    selected_indices: NDArray[np.int64],
    inner_ndcg_by_candidate: Mapping[str, NDArray[np.float64]],
    inner_mrr_by_candidate: Mapping[str, NDArray[np.float64]],
    selected_priors: Mapping[str, Mapping[str, float]],
    rankings: Mapping[str, Sequence[str]],
    evidence: Mapping[str, NDArray[Any]],
    validity_names: Sequence[str],
    valid_code: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    validity = evidence["validity_codes"]
    weight_sums = evidence["source_weight_sums"]
    positive = evidence["positive_retrieval_counts"]
    canonical_masks = evidence["exact_mask_canonical_counts"]
    source_masks = evidence["exact_mask_source_id_counts"]
    model_position = {model_id: index for index, model_id in enumerate(model_ids)}
    rows: list[dict[str, Any]] = []
    margins: list[float] = []
    for query_index, query_id in enumerate(query_ids):
        selected = int(selected_indices[query_index])
        candidate = candidates[selected]
        key = source_candidate_key(candidate)
        k = source_candidate_k(candidate)
        cells = validity[query_index, :k]
        reason_counts: Counter[str] = Counter()
        for code, count in zip(*np.unique(cells, return_counts=True), strict=True):
            reason = (
                validity_names[int(code)]
                if int(code) < len(validity_names)
                else f"code_{int(code)}"
            )
            reason_counts[reason] += int(count)
        primary = model_scores[selected, query_index]
        diagnostic = diagnostic_scores[selected, query_index]
        priors = np.asarray(
            [selected_priors[query_id][model_id] for model_id in model_ids],
            dtype=np.float64,
        )
        top = list(rankings[query_id])[: min(10, len(model_ids))]
        prior_ties = 0
        zero_top = 0
        for model_id in top:
            position = model_position[model_id]
            tied = np.flatnonzero(primary == primary[position])
            if len(tied) > 1 and np.any(priors[tied] != priors[position]):
                prior_ties += 1
            zero_top += int(primary[position] == 0.0)
        inner_values = sorted(
            (
                float(
                    inner_ndcg_by_candidate[source_candidate_key(value)][query_index]
                ),
                source_candidate_key(value),
            )
            for value in candidates
        )
        winner = inner_values[-1][0]
        runner_up = inner_values[-2][0]
        margin = winner - runner_up
        margins.append(margin)
        rows.append(
            {
                "query_id": query_id,
                "selected_configuration": key,
                "selected_k": k,
                "selected_inner_ndcg_at_10": float(
                    inner_ndcg_by_candidate[key][query_index]
                ),
                "selected_inner_mrr": float(inner_mrr_by_candidate[key][query_index]),
                "inner_ndcg_winner_margin": margin,
                "invalid_cell_count": int(np.count_nonzero(cells != valid_code)),
                "invalid_cell_share": float(np.mean(cells != valid_code)),
                "validity_reason_counts": dict(sorted(reason_counts.items())),
                "primary_zero_score_model_count": int(np.count_nonzero(primary == 0.0)),
                "diagnostic_zero_score_model_count": int(
                    np.count_nonzero(diagnostic == 0.0)
                ),
                "primary_top10_prior_tie_position_count": prior_ties,
                "primary_top10_zero_score_position_count": zero_top,
                "source_weight_sum": float(weight_sums[selected, query_index]),
                "positive_retrieval_count": int(positive[query_index]),
                "exact_mask_canonical_count": int(canonical_masks[query_index]),
                "exact_mask_source_id_count": int(source_masks[query_index]),
            }
        )
    return rows, margins


def _basic_inventory(
    metadata: Mapping[str, Any], model_count: int
) -> NDArray[np.float64]:
    direct = metadata.get("basic_retained_document_counts")
    if direct is not None:
        inventory = np.asarray(direct, dtype=np.float64)
    else:
        names = tuple(metadata.get("inventory_filter_names", ()))
        values = np.asarray(
            metadata.get("retained_document_counts", ()), dtype=np.float64
        )
        if "basic" not in names or values.ndim != 2:
            raise ValueError("source score metadata lacks frozen BASIC inventory")
        inventory = values[names.index("basic")]
    if inventory.shape != (model_count,) or not np.isfinite(inventory).all():
        raise ValueError("BASIC inventory does not match the model universe")
    return inventory


def _deployment_behavior(
    *,
    inventory: NDArray[np.float64],
    scores: NDArray[np.float64],
    rankings: Mapping[str, Sequence[str]],
    model_ids: Sequence[str],
    configuration: str,
) -> dict[str, Any]:
    total_scores = scores.sum(axis=0, dtype=np.float64)
    rank_sums = np.zeros(len(model_ids), dtype=np.float64)
    top10_counts = np.zeros(len(model_ids), dtype=np.float64)
    position = {model_id: index for index, model_id in enumerate(model_ids)}
    for ranking in rankings.values():
        for rank, model_id in enumerate(ranking, start=1):
            index = position[model_id]
            rank_sums[index] += rank
            if rank <= 10:
                top10_counts[index] += 1.0
    mean_ranks = rank_sums / len(rankings)
    quartiles = _quartile_diagnostics(inventory, top10_counts, model_ids)
    return {
        "configuration": configuration,
        "effectiveness_estimate": "optimistic_not_oof",
        "gating_scope": "fixed_full_development_deployment_configuration",
        "inventory_definition": "frozen_pr8_basic_retained_response_count",
        "inventory_vs_total_primary_score_spearman": _spearman(inventory, total_scores),
        "inventory_vs_mean_rank_spearman": _spearman(inventory, mean_ranks),
        "inventory_vs_top10_frequency_spearman": _spearman(inventory, top10_counts),
        "largest_inventory_quartile_top10_share": quartiles[-1][
            "top10_appearance_share"
        ],
        "quartiles": quartiles,
    }


def _outer_fold_behavior(
    *,
    inventory: NDArray[np.float64],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    model_scores: NDArray[np.float64],
    diagnostic_scores: NDArray[np.float64],
    selected_indices: NDArray[np.int64],
    selected_priors: Mapping[str, Mapping[str, float]],
    rankings: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    position = {model_id: index for index, model_id in enumerate(model_ids)}
    rho_values: list[float] = []
    quartile_shares: list[float] = []
    prior_tie_positions = 0
    zero_positions = 0
    diagnostic_zero_positions = 0
    top_positions = 0
    for query_index, query_id in enumerate(query_ids):
        selected = int(selected_indices[query_index])
        raw = model_scores[selected, query_index]
        diagnostic = diagnostic_scores[selected, query_index]
        rho = _spearman(inventory, raw)
        if rho is not None:
            rho_values.append(float(rho))
        top = list(rankings[query_id])[: min(10, len(model_ids))]
        counts = np.zeros(len(model_ids), dtype=np.float64)
        for model_id in top:
            counts[position[model_id]] += 1.0
        quartile_shares.append(
            _quartile_diagnostics(inventory, counts, model_ids)[-1][
                "top10_appearance_share"
            ]
        )
        priors = np.asarray(
            [selected_priors[query_id][model_id] for model_id in model_ids],
            dtype=np.float64,
        )
        for model_id in top:
            index = position[model_id]
            tied = np.flatnonzero(raw == raw[index])
            if len(tied) > 1 and np.any(priors[tied] != priors[index]):
                prior_tie_positions += 1
            zero_positions += int(raw[index] == 0.0)
            diagnostic_zero_positions += int(diagnostic[index] == 0.0)
            top_positions += 1
    return {
        "fold_inventory_vs_primary_score_spearman": _mean_and_range(rho_values),
        "fold_largest_quartile_top10_share": _mean_and_range(quartile_shares),
        "top10_position_count": top_positions,
        "primary_tie_resolved_by_prior_fraction": (
            prior_tie_positions / top_positions if top_positions else 0.0
        ),
        "primary_zero_score_top10_fraction": (
            zero_positions / top_positions if top_positions else 0.0
        ),
        "diagnostic_zero_score_at_primary_top10_fraction": (
            diagnostic_zero_positions / top_positions if top_positions else 0.0
        ),
        "prior_tie_position_count": prior_tie_positions,
        "primary_zero_score_position_count": zero_positions,
    }


def _build_decision(
    evaluations: Mapping[str, EvaluationResult],
    comparisons: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    primary_ndcg = evaluations["source_alignment_nested"].macro_ndcg_at_k
    sumscore_ndcg = evaluations["response_sumscore_nested"].macro_ndcg_at_k
    tfidf_interval = comparisons["tfidf_fixed_k10"]["ndcg_at_10_delta"]
    deployment = diagnostics["deployment_behavior"]
    rho = deployment["inventory_vs_total_primary_score_spearman"]
    quartile_share = deployment["largest_inventory_quartile_top10_share"]
    gates = {
        "ndcg_ci_above_tfidf": {
            "passed": bool(tfidf_interval["lower"] > 0.0),
            "observed": tfidf_interval["lower"],
            "operator": ">",
            "threshold": 0.0,
            "measure": "paired_95pct_ci_lower_primary_minus_tfidf_ndcg_at_10",
        },
        "ndcg_within_0_010_of_sumscore": {
            "passed": bool(primary_ndcg >= sumscore_ndcg - MAX_NDCG_DROP_FROM_SUMSCORE),
            "observed": primary_ndcg - sumscore_ndcg,
            "operator": ">=",
            "threshold": -MAX_NDCG_DROP_FROM_SUMSCORE,
            "measure": "primary_minus_sumscore_oof_ndcg_at_10",
        },
        "deployment_inventory_rho_at_most_0_7388": {
            "passed": bool(rho is not None and abs(float(rho)) <= INVENTORY_RHO_GATE),
            "observed": abs(float(rho)) if rho is not None else None,
            "operator": "<=",
            "threshold": INVENTORY_RHO_GATE,
            "reference": INVENTORY_RHO_REFERENCE,
        },
        "deployment_largest_quartile_share_at_most_0_411": {
            "passed": bool(float(quartile_share) <= LARGEST_QUARTILE_GATE),
            "observed": float(quartile_share),
            "operator": "<=",
            "threshold": LARGEST_QUARTILE_GATE,
            "reference": LARGEST_QUARTILE_REFERENCE,
            "equal_model_share_reference": LARGEST_QUARTILE_MODEL_SHARE,
        },
    }
    passed = all(value["passed"] for value in gates.values())
    return {
        "all_four_gates_passed": passed,
        "recommendation": (
            "advance_source_query_first_for_owner_consideration"
            if passed
            else "retain_sumscore_and_stop_source_query_first_branch"
        ),
        "strong_success": primary_ndcg >= sumscore_ndcg,
        "gates": gates,
        "mrr_is_non_gating": True,
        "dense_comparison_is_non_gating": True,
        "top10sum_comparison_is_non_gating": True,
        "target_response_diagnostic_is_non_gating": True,
        "owner_decision_still_required": True,
    }


def _validate_control_references(
    *,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    evaluations: Mapping[str, EvaluationResult],
    sumscore: ResponseExperimentResult,
    top10sum: ResponseExperimentResult,
) -> dict[str, Any]:
    if (len(query_ids), len(model_ids)) != (322, 150):
        return {
            "official_reference_applicable": False,
            "reason": "non_official_fixture_shape",
            "run_parity_validation": "writer_checks_exact_official_run_sha256",
        }
    references = {
        "response_sumscore_nested": (
            SUMSCORE_REFERENCE_NDCG_AT_10,
            SUMSCORE_REFERENCE_MRR,
            SUMSCORE_REFERENCE_SELECTION,
            sumscore,
        ),
        "response_top10sum_nested": (
            TOP10SUM_REFERENCE_NDCG_AT_10,
            TOP10SUM_REFERENCE_MRR,
            TOP10SUM_REFERENCE_SELECTION,
            top10sum,
        ),
        "tfidf_fixed_k10": (
            TFIDF_REFERENCE_NDCG_AT_10,
            TFIDF_REFERENCE_MRR,
            None,
            None,
        ),
        "global_prior": (
            PRIOR_REFERENCE_NDCG_AT_10,
            PRIOR_REFERENCE_MRR,
            None,
            None,
        ),
        "dense_nested_reference": (
            DENSE_REFERENCE_NDCG_AT_10,
            DENSE_REFERENCE_MRR,
            None,
            None,
        ),
    }
    output: dict[str, Any] = {
        "official_reference_applicable": True,
        "run_parity_validation": "writer_checks_exact_official_run_sha256",
    }
    for system, (ndcg, mrr, selection, control) in references.items():
        observed = evaluations[system]
        metrics_match = math.isclose(
            observed.macro_ndcg_at_k, ndcg, rel_tol=0.0, abs_tol=1e-12
        ) and math.isclose(
            observed.mean_reciprocal_rank, mrr, rel_tol=0.0, abs_tol=1e-12
        )
        selection_match = (
            Counter(control.selected_config_by_outer.values()) == Counter(selection)
            if selection is not None and control is not None
            else None
        )
        output[system] = {
            "metrics_match": metrics_match,
            "selection_match": selection_match,
            "observed_ndcg_at_10": observed.macro_ndcg_at_k,
            "observed_mrr": observed.mean_reciprocal_rank,
            "expected_ndcg_at_10": ndcg,
            "expected_mrr": mrr,
        }
        if not metrics_match or selection_match is False:
            raise AssertionError(f"{system} did not reproduce its merged reference")
    return output


def _control_run_parity(
    result: SourceQueryExperimentResult,
    destination: Path,
) -> dict[str, Any]:
    """Verify exact official control-run bytes against the merged PR #8 receipt."""

    if (len(result.query_ids), len(result.model_ids)) != (322, 150):
        return {
            "official_reference_applicable": False,
            "reason": "non_official_fixture_shape",
        }
    checks = {
        "response_sumscore_nested": (
            destination / "sumscore_control.trec",
            SUMSCORE_REFERENCE_RUN_SHA256,
        ),
        "response_top10sum_nested": (
            destination / "top10sum_control.trec",
            TOP10SUM_REFERENCE_RUN_SHA256,
        ),
    }
    output: dict[str, Any] = {"official_reference_applicable": True}
    for system, (path, expected) in checks.items():
        observed = _sha256(path)
        matches = observed == expected
        output[system] = {
            "identical_rankings_scores_and_run_id": matches,
            "observed_sha256": observed,
            "expected_sha256": expected,
        }
        if not matches:
            raise AssertionError(f"{system} run did not reproduce merged PR #8 bytes")
    return output


def _build_summary(
    *,
    config: SourceQueryExperimentConfig,
    data: DevelopmentData,
    query_ids: Sequence[str],
    candidates: Sequence[Any],
    source_metadata: Mapping[str, Any],
    inner_ndcg_by_candidate: Mapping[str, NDArray[np.float64]],
    inner_mrr_by_candidate: Mapping[str, NDArray[np.float64]],
    selected_config_by_outer: Mapping[str, str],
    evaluations: Mapping[str, EvaluationResult],
    comparisons: Mapping[str, Mapping[str, Any]],
    fixed_candidate_metrics: Mapping[str, Mapping[str, float]],
    deployment_config: str,
    diagnostics: Mapping[str, Any],
    decision: Mapping[str, Any],
    controls: Mapping[str, Any],
    elapsed_seconds: float,
    cpu_seconds: float,
) -> dict[str, Any]:
    selected_inner_ndcg = [
        float(inner_ndcg_by_candidate[selected_config_by_outer[query_id]][index])
        for index, query_id in enumerate(query_ids)
    ]
    selected_inner_mrr = [
        float(inner_mrr_by_candidate[selected_config_by_outer[query_id]][index])
        for index, query_id in enumerate(query_ids)
    ]
    return {
        "protocol": {
            "name": "source_query_first_answer_alignment",
            "base_revision": "81d4aa50812864133f7066ee5b698fd48ea20503",
            "preregistration_revision": "459e545",
            "outer_split": "strict_leave_one_query_out",
            "inner_split": "strict_leave_one_query_out_within_outer_training_rows",
            "retrieval_and_signal_tensors_are_qrel_free": True,
            "selection_metric": "inner_macro_ndcg_at_10",
            "selection_tie_break": "declared_k10_then_k50_then_k100",
            "mrr_used_for_selection": False,
            "ranking_tie_break": "fold_local_global_prior_then_model_id",
            "target_response_role": "same_selected_k_diagnostic_only",
            "run_classification": "internal_automatic",
            "grid_selection_provenance": config.grid_selection_provenance,
            "bootstrap": {
                "resamples": config.bootstrap_resamples,
                "base_seed": config.seed,
                "baseline_order": list(HEADLINE_BASELINES),
                "ndcg_seed_by_baseline": {
                    baseline: config.seed + offset
                    for offset, baseline in enumerate(HEADLINE_BASELINES)
                },
                "mrr_seed_by_baseline": {
                    baseline: config.seed + offset + 10_000
                    for offset, baseline in enumerate(HEADLINE_BASELINES)
                },
                "target_response_vs_primary_ndcg_seed": config.seed + 100,
                "target_response_vs_primary_mrr_seed": config.seed + 10_100,
                "selection_rerun_inside_resample": False,
            },
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
            {"key": source_candidate_key(value), "k": source_candidate_k(value)}
            for value in candidates
        ],
        "retrieval_reproducibility": dict(source_metadata),
        "metrics": {
            system: {
                "ndcg_at_10": evaluation.macro_ndcg_at_k,
                "mrr": evaluation.mean_reciprocal_rank,
            }
            for system, evaluation in evaluations.items()
        },
        "comparisons": dict(comparisons),
        "selection": {
            "configuration_frequency": dict(
                sorted(Counter(selected_config_by_outer.values()).items())
            ),
            "selected_inner_ndcg_mean": statistics.fmean(selected_inner_ndcg),
            "selected_inner_ndcg_min": min(selected_inner_ndcg),
            "selected_inner_ndcg_max": max(selected_inner_ndcg),
            "selected_inner_mrr_mean": statistics.fmean(selected_inner_mrr),
            "inner_ndcg_winner_margin": diagnostics["selection"][
                "inner_ndcg_winner_margin"
            ],
        },
        "deployment": {
            "configuration": deployment_config,
            "criterion": "full_development_fixed_candidate_ndcg_at_10",
            "development_score_is_non_nested_and_optimistic": True,
            "purpose": "future_hidden_test_queries_and_preregistered_volume_gates",
            "behavior": diagnostics["deployment_behavior"],
        },
        "fixed_candidate_development_metrics": dict(fixed_candidate_metrics),
        "diagnostics": dict(diagnostics),
        "decision": dict(decision),
        "controls": dict(controls),
        "caveats": {
            "answer_alignment_is_not_correctness": True,
            "target_response_is_nonselecting": True,
            "deployment_effectiveness_is_optimistic": True,
            "bootstrap_conditions_on_completed_nested_selection": True,
            "inventory_association_is_not_causal": True,
        },
        "runtime": {
            "experiment_seconds_through_summary": elapsed_seconds,
            "cpu_seconds_through_summary": cpu_seconds,
            "peak_rss_bytes_at_summary": _peak_rss_bytes(),
        },
    }


def _write_selected_sources(result: SourceQueryExperimentResult, path: Path) -> None:
    raw = result.source_evidence_tensors.get("source_relevance_raw")
    weights = result.source_evidence_tensors.get("source_weights")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "source_rank",
                "source_query_id",
                "source_query_key",
                "source_relevance_raw",
                "weight_k10",
                "weight_k50",
                "weight_k100",
            )
        )
        for query_index, query_id in enumerate(result.query_ids):
            for source_index, (source_id, source_key) in enumerate(
                zip(
                    result.selected_source_ids[query_index],
                    result.selected_source_keys[query_index],
                    strict=True,
                ),
                start=1,
            ):
                writer.writerow(
                    (
                        query_id,
                        source_index,
                        source_id,
                        source_key,
                        f"{raw[query_index, source_index - 1]:.12g}"
                        if raw is not None
                        else "",
                        *(
                            f"{weights[c, query_index, source_index - 1]:.12g}"
                            if weights is not None
                            else ""
                            for c in range(len(result.candidates))
                        ),
                    )
                )


def _write_inner_scores(result: SourceQueryExperimentResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "system",
                "outer_query_id",
                "configuration",
                "inner_macro_ndcg_at_10",
                "inner_mrr",
                "selected",
            )
        )
        systems = (
            (
                "source_alignment_nested",
                result.candidates,
                result.inner_ndcg_by_candidate,
                result.inner_mrr_by_candidate,
                result.selected_config_by_outer,
            ),
            (
                "response_sumscore_nested",
                result.sumscore_control.candidates,
                result.sumscore_control.inner_ndcg_by_candidate,
                result.sumscore_control.inner_mrr_by_candidate,
                result.sumscore_control.selected_config_by_outer,
            ),
            (
                "response_top10sum_nested",
                result.top10sum_control.candidates,
                result.top10sum_control.inner_ndcg_by_candidate,
                result.top10sum_control.inner_mrr_by_candidate,
                result.top10sum_control.selected_config_by_outer,
            ),
        )
        for system, candidates, ndcg, mrr, selections in systems:
            for query_index, query_id in enumerate(result.query_ids):
                for candidate in candidates:
                    key = (
                        source_candidate_key(candidate)
                        if system == "source_alignment_nested"
                        else candidate.key
                    )
                    writer.writerow(
                        (
                            system,
                            query_id,
                            key,
                            f"{ndcg[key][query_index]:.12g}",
                            f"{mrr[key][query_index]:.12g}",
                            int(selections[query_id] == key),
                        )
                    )


def _write_selections(result: SourceQueryExperimentResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "outer_query_id",
                "source_configuration",
                "source_inner_ndcg_at_10",
                "source_inner_mrr",
                "sumscore_configuration",
                "top10sum_configuration",
            )
        )
        for index, query_id in enumerate(result.query_ids):
            key = result.selected_config_by_outer[query_id]
            writer.writerow(
                (
                    query_id,
                    key,
                    f"{result.inner_ndcg_by_candidate[key][index]:.12g}",
                    f"{result.inner_mrr_by_candidate[key][index]:.12g}",
                    result.sumscore_control.selected_config_by_outer[query_id],
                    result.top10sum_control.selected_config_by_outer[query_id],
                )
            )


def _write_predictions(result: SourceQueryExperimentResult, path: Path) -> None:
    model_position = {value: index for index, value in enumerate(result.model_ids)}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "selected_configuration",
                "model_id",
                "primary_score",
                "diagnostic_score",
                "global_prior",
                "primary_rank",
                "diagnostic_rank",
                "sumscore_rank",
                "top10sum_rank",
            )
        )
        for query_id in result.query_ids:
            diagnostic_rank = {
                model_id: index
                for index, model_id in enumerate(
                    result.rankings["target_response_same_k"][query_id], start=1
                )
            }
            sumscore_rank = {
                model_id: index
                for index, model_id in enumerate(
                    result.rankings["response_sumscore_nested"][query_id], start=1
                )
            }
            top10sum_rank = {
                model_id: index
                for index, model_id in enumerate(
                    result.rankings["response_top10sum_nested"][query_id], start=1
                )
            }
            for primary_rank, model_id in enumerate(
                result.rankings["source_alignment_nested"][query_id], start=1
            ):
                _ = model_position[model_id]
                writer.writerow(
                    (
                        query_id,
                        result.selected_config_by_outer[query_id],
                        model_id,
                        f"{result.selected_primary_scores[query_id][model_id]:.12g}",
                        f"{result.selected_diagnostic_scores[query_id][model_id]:.12g}",
                        f"{result.selected_priors[query_id][model_id]:.12g}",
                        primary_rank,
                        diagnostic_rank[model_id],
                        sumscore_rank[model_id],
                        top10sum_rank[model_id],
                    )
                )


def _write_per_query(result: SourceQueryExperimentResult, path: Path) -> None:
    systems = (
        "source_alignment_nested",
        "target_response_same_k",
        "response_sumscore_nested",
        "response_top10sum_nested",
        "dense_nested_reference",
        "tfidf_fixed_k10",
        "global_prior",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "selected_configuration",
                *(
                    value
                    for system in systems
                    for value in (f"{system}_ndcg_at_10", f"{system}_reciprocal_rank")
                ),
            )
        )
        for query_id in result.query_ids:
            writer.writerow(
                (
                    query_id,
                    result.selected_config_by_outer[query_id],
                    *(
                        f"{value:.12g}"
                        for system in systems
                        for value in (
                            result.evaluations[system].per_query[query_id].ndcg_at_k,
                            result.evaluations[system]
                            .per_query[query_id]
                            .reciprocal_rank,
                        )
                    ),
                )
            )


def _write_query_diagnostics(result: SourceQueryExperimentResult, path: Path) -> None:
    rows = result.diagnostics["per_query_selected_k"]
    columns = (
        "query_id",
        "selected_configuration",
        "selected_k",
        "selected_inner_ndcg_at_10",
        "selected_inner_mrr",
        "inner_ndcg_winner_margin",
        "invalid_cell_count",
        "invalid_cell_share",
        "primary_zero_score_model_count",
        "diagnostic_zero_score_model_count",
        "primary_top10_prior_tie_position_count",
        "primary_top10_zero_score_position_count",
        "source_weight_sum",
        "positive_retrieval_count",
        "exact_mask_canonical_count",
        "exact_mask_source_id_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((*columns, "validity_reason_counts_json"))
        for row in rows:
            writer.writerow(
                (
                    *(row[column] for column in columns),
                    json.dumps(
                        row["validity_reason_counts"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )


def _selected_zero_summary(
    scores: NDArray[np.float64], selected_indices: NDArray[np.int64]
) -> dict[str, float | int]:
    values = [
        int(np.count_nonzero(scores[int(selected), index] == 0.0))
        for index, selected in enumerate(selected_indices)
    ]
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _array_summary(array: NDArray[Any] | None) -> dict[str, float | int] | None:
    if array is None:
        return None
    values = np.asarray(array)
    return {
        "count": int(values.size),
        "sum": float(np.sum(values, dtype=np.float64)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _mean_and_range(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


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
    left: NDArray[np.floating[Any]], right: NDArray[np.floating[Any]]
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
    groups = np.array_split(np.asarray(order, dtype=np.int64), min(4, len(order)))
    total = float(np.sum(top10_counts, dtype=np.float64))
    output: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        appearances = float(np.sum(top10_counts[group], dtype=np.float64))
        output.append(
            {
                "quartile": index,
                "model_count": int(len(group)),
                "inventory_min": float(np.min(inventory[group])),
                "inventory_max": float(np.max(inventory[group])),
                "top10_appearances": int(appearances),
                "top10_appearance_share": appearances / total if total else 0.0,
            }
        )
    return output


def _execution_provenance(result: SourceQueryExperimentResult) -> dict[str, Any]:
    paths = [
        Path(result.config.query_path),
        Path(result.config.qrels_path),
        Path(result.config.discovery_path),
    ]
    if result.config.dense_per_query_path:
        paths.append(Path(result.config.dense_per_query_path))
    code_paths = (
        Path(__file__),
        Path(__file__).with_name("source_query_first.py"),
        Path(__file__).with_name("response_experiment.py"),
        Path(__file__).with_name("response_retrieval.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("runio.py"),
    )
    return {
        "input_sha256": {str(path): _sha256(path) for path in paths if path.is_file()},
        "code_revision": _git_revision(),
        "git_worktree_status": _git_worktree_status(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "code_sha256": {
            path.name: _sha256(path) for path in code_paths if path.is_file()
        },
    }


def _peak_rss_bytes() -> int:
    """Return the process peak RSS in bytes on the supported Unix platforms."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux and the BSDs report KiB; macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def _decision_receipt(result: SourceQueryExperimentResult) -> str:
    payload = {
        "candidate_keys": [source_candidate_key(value) for value in result.candidates],
        "source_score_hash": _array_sha256(result.model_scores),
        "diagnostic_score_hash": _array_sha256(result.diagnostic_model_scores),
        "selections": result.selected_config_by_outer,
        "rankings": result.rankings,
        "metrics": result.summary["metrics"],
        "comparisons": result.comparisons,
        "deployment": result.deployment_config,
        "decision": result.decision,
    }
    return _hash_json(payload)


def _array_sha256(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _raw_arrays_sha256(*arrays: NDArray[Any]) -> str:
    """Match the retrieval layer's hash over an ordered group of raw arrays."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_status() -> str | None:
    try:
        return subprocess.run(
            ("git", "status", "--short"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RESPONSE_CONTROL_CANDIDATES",
    "SOURCE_QUERY_K_VALUES",
    "TOP10SUM_RESPONSE_CANDIDATES",
    "SourceQueryExperimentConfig",
    "SourceQueryExperimentResult",
    "run_source_query_experiment",
    "source_candidate_key",
    "write_source_query_experiment_artifacts",
]
