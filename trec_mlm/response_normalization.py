"""Preregistered response-volume normalization experiment.

The retrieval layer computes the frozen SumScore and six normalized aggregations
from one set of response-document hits.  This module then runs two independent
nested selectors over that shared tensor: the 144-candidate normalized primary
system and the original 24-candidate SumScore control.
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
import statistics
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from trec_mlm.metrics import EvaluationResult, paired_bootstrap_ci
from trec_mlm.query_knn import load_development_data
from trec_mlm.response_experiment import (
    ResponseExperimentConfig,
    ResponseExperimentResult,
    _quartile_diagnostics,
    _spearman,
    run_nested_response_experiment,
    write_response_experiment_artifacts,
)
from trec_mlm.response_retrieval import (
    DEFAULT_RESPONSE_CANDIDATES,
    NORMALIZED_RESPONSE_CANDIDATES,
    NORMALIZED_RESPONSE_AGGREGATIONS,
    ResponseCandidate,
    load_discovery_records,
    score_response_candidates,
)
from trec_mlm.runio import validate_trec_run, write_trec_run


SUMSCORE_REFERENCE_RESULTS = "reports/trec_2026_response_retrieval_results.json"
SUMSCORE_REFERENCE_RUN = "artifacts/response_retrieval_baseline/run.trec"
SUMSCORE_REFERENCE_NDCG_AT_10 = 0.2773577839029814
SUMSCORE_REFERENCE_MRR = 0.31753289707886323
SUMSCORE_REFERENCE_SELECTION = {
    "bm25_rm3__query_response__basic__d500": 6,
    "bm25_rm3__response_only__aggressive__d500": 2,
    "bm25_rm3__response_only__basic__d500": 314,
}
SUMSCORE_REFERENCE_DEPLOYMENT = "bm25_rm3__response_only__basic__d500"

INVENTORY_RHO_REFERENCE = 0.8388
INVENTORY_RHO_GATE = 0.7388
LARGEST_QUARTILE_REFERENCE = 0.466
LARGEST_QUARTILE_MODEL_SHARE = 0.247
LARGEST_QUARTILE_GATE = 0.411
MAX_NDCG_DROP_FROM_SUMSCORE = 0.010


@dataclass(frozen=True, slots=True)
class ResponseNormalizationConfig:
    """Inputs and locked settings for the normalization follow-up."""

    query_path: str
    qrels_path: str
    discovery_path: str
    dense_per_query_path: str | None = (
        "reports/trec_2026_dense_query_knn_per_query.tsv"
    )
    normalized_candidates: tuple[ResponseCandidate, ...] = (
        NORMALIZED_RESPONSE_CANDIDATES
    )
    sumscore_candidates: tuple[ResponseCandidate, ...] = DEFAULT_RESPONSE_CANDIDATES
    cutoff: int = 10
    normalized_run_id: str = "response_volume_normalized_nested_loo"
    sumscore_run_id: str = "response_sumscore_nested_loo_control"
    tfidf_run_id: str = "qknn_tfidf_w12_k10_loo_control"
    prior_run_id: str = "global_prior_loo_control"
    bootstrap_resamples: int = 10_000
    seed: int = 2026
    show_progress: bool = False
    grid_selection_provenance: str = (
        "advisor_approved_protocol_b6cfa6c_locked_before_effectiveness_scores"
    )
    sumscore_reference_results_path: str | None = SUMSCORE_REFERENCE_RESULTS
    sumscore_reference_run_path: str | None = SUMSCORE_REFERENCE_RUN


@dataclass(slots=True)
class ResponseNormalizationResult:
    """Both nested systems plus their direct comparisons and decision gates."""

    config: ResponseNormalizationConfig
    score_result: Any
    normalized: ResponseExperimentResult
    sumscore: ResponseExperimentResult
    rankings: dict[str, dict[str, list[str]]]
    evaluations: dict[str, EvaluationResult]
    comparisons: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    decision: dict[str, Any]
    summary: dict[str, Any]


def run_response_normalization_experiment(
    config: ResponseNormalizationConfig,
    *,
    score_result: Any | None = None,
    dense_per_query_reference: Mapping[str, Any] | EvaluationResult | None = None,
) -> ResponseNormalizationResult:
    """Run the locked normalized primary and SumScore control from one tensor."""

    _validate_config(config)
    started = time.perf_counter()
    data = load_development_data(config.query_path, config.qrels_path)
    query_ids = tuple(sorted(data.queries))
    model_ids = data.models

    combined_candidates = config.sumscore_candidates + config.normalized_candidates
    if score_result is None:
        corpus = load_discovery_records(config.discovery_path, model_ids)
        score_result = score_response_candidates(
            corpus,
            query_ids=query_ids,
            query_texts=[data.queries[query_id] for query_id in query_ids],
            expected_model_ids=model_ids,
            candidates=combined_candidates,
            show_progress=config.show_progress,
        )
    _validate_combined_score_result(
        score_result,
        candidates=combined_candidates,
        query_ids=query_ids,
        model_ids=model_ids,
    )

    sumscore_count = len(config.sumscore_candidates)
    sumscore_scores = _slice_score_result(score_result, range(sumscore_count))
    normalized_scores = _slice_score_result(
        score_result,
        range(sumscore_count, len(combined_candidates)),
    )

    common = {
        "query_path": config.query_path,
        "qrels_path": config.qrels_path,
        "discovery_path": config.discovery_path,
        "dense_per_query_path": config.dense_per_query_path,
        "cutoff": config.cutoff,
        "tfidf_run_id": config.tfidf_run_id,
        "prior_run_id": config.prior_run_id,
        "bootstrap_resamples": config.bootstrap_resamples,
        "seed": config.seed,
        "show_progress": config.show_progress,
    }
    normalized = run_nested_response_experiment(
        ResponseExperimentConfig(
            **common,
            candidates=config.normalized_candidates,
            run_id=config.normalized_run_id,
            grid_selection_provenance=config.grid_selection_provenance,
        ),
        score_result=normalized_scores,
        dense_per_query_reference=dense_per_query_reference,
    )
    sumscore = run_nested_response_experiment(
        ResponseExperimentConfig(
            **common,
            candidates=config.sumscore_candidates,
            run_id=config.sumscore_run_id,
            grid_selection_provenance="frozen_baseline3_sumscore_control",
        ),
        score_result=sumscore_scores,
        dense_per_query_reference=dense_per_query_reference,
    )
    _assert_shared_controls(normalized, sumscore)

    rankings = {
        "response_normalized_nested": normalized.rankings["response_nested"],
        "response_sumscore_nested": sumscore.rankings["response_nested"],
        "dense_nested_reference": normalized.rankings.get(
            "dense_nested_reference", {}
        ),
        "tfidf_fixed_k10": normalized.rankings["tfidf_fixed_k10"],
        "global_prior": normalized.rankings["global_prior"],
    }
    evaluations = {
        "response_normalized_nested": normalized.evaluations["response_nested"],
        "response_sumscore_nested": sumscore.evaluations["response_nested"],
        "dense_nested_reference": normalized.evaluations["dense_nested_reference"],
        "tfidf_fixed_k10": normalized.evaluations["tfidf_fixed_k10"],
        "global_prior": normalized.evaluations["global_prior"],
    }
    comparisons = {
        baseline: _paired_comparison(
            baseline_name=baseline,
            candidate_name="response_normalized_nested",
            baseline=evaluations[baseline],
            candidate=evaluations["response_normalized_nested"],
            n_resamples=config.bootstrap_resamples,
            seed=config.seed + offset,
        )
        for offset, baseline in enumerate(
            (
                "response_sumscore_nested",
                "dense_nested_reference",
                "tfidf_fixed_k10",
                "global_prior",
            )
        )
    }

    diagnostics = _build_normalization_diagnostics(normalized, sumscore)
    decision = _build_decision(evaluations, comparisons, diagnostics)
    reference_validation = _validate_sumscore_reference(config, sumscore)
    summary = _build_summary(
        config=config,
        score_result=score_result,
        normalized=normalized,
        sumscore=sumscore,
        evaluations=evaluations,
        comparisons=comparisons,
        diagnostics=diagnostics,
        decision=decision,
        reference_validation=reference_validation,
        elapsed_seconds=time.perf_counter() - started,
    )
    return ResponseNormalizationResult(
        config=config,
        score_result=score_result,
        normalized=normalized,
        sumscore=sumscore,
        rankings=rankings,
        evaluations=evaluations,
        comparisons=comparisons,
        diagnostics=diagnostics,
        decision=decision,
        summary=summary,
    )


def write_response_normalization_artifacts(
    result: ResponseNormalizationResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write complete evidence for both selectors and a combined QA receipt."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalized_paths = write_response_experiment_artifacts(
        result.normalized,
        destination / "normalized",
    )
    sumscore_paths = write_response_experiment_artifacts(
        result.sumscore,
        destination / "sumscore",
    )

    normalized_execution_config = json.loads(
        Path(normalized_paths["config.json"]).read_text(encoding="utf-8")
    )
    code_sha256 = dict(normalized_execution_config.get("code_sha256") or {})
    code_sha256[Path(__file__).name] = _sha256(Path(__file__))
    root_config = {
        **asdict(result.config),
        "execution_provenance": {
            key: normalized_execution_config.get(key)
            for key in (
                "input_sha256",
                "code_revision",
                "git_worktree_status",
                "python_version",
                "platform",
            )
        },
        "combined_retrieval_hashes": _combined_retrieval_hashes(
            result.score_result
        ),
    }
    root_config["execution_provenance"]["code_sha256"] = code_sha256
    _write_json(destination / "config.json", root_config)
    _write_json(destination / "summary.json", result.summary)
    _write_json(destination / "diagnostics.json", result.diagnostics)
    _write_json(destination / "decision.json", result.decision)

    query_ids = result.normalized.query_ids
    with (destination / "per_query.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "normalized_configuration",
                "sumscore_configuration",
                "normalized_ndcg_at_10",
                "normalized_reciprocal_rank",
                "sumscore_ndcg_at_10",
                "sumscore_reciprocal_rank",
                "dense_ndcg_at_10",
                "dense_reciprocal_rank",
                "tfidf_ndcg_at_10",
                "tfidf_reciprocal_rank",
                "prior_ndcg_at_10",
                "prior_reciprocal_rank",
            )
        )
        for query_id in query_ids:
            values = {
                name: evaluation.per_query[query_id]
                for name, evaluation in result.evaluations.items()
            }
            writer.writerow(
                (
                    query_id,
                    result.normalized.selected_config_by_outer[query_id],
                    result.sumscore.selected_config_by_outer[query_id],
                    f"{values['response_normalized_nested'].ndcg_at_k:.12g}",
                    f"{values['response_normalized_nested'].reciprocal_rank:.12g}",
                    f"{values['response_sumscore_nested'].ndcg_at_k:.12g}",
                    f"{values['response_sumscore_nested'].reciprocal_rank:.12g}",
                    f"{values['dense_nested_reference'].ndcg_at_k:.12g}",
                    f"{values['dense_nested_reference'].reciprocal_rank:.12g}",
                    f"{values['tfidf_fixed_k10'].ndcg_at_k:.12g}",
                    f"{values['tfidf_fixed_k10'].reciprocal_rank:.12g}",
                    f"{values['global_prior'].ndcg_at_k:.12g}",
                    f"{values['global_prior'].reciprocal_rank:.12g}",
                )
            )

    with (destination / "selections.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "outer_query_id",
                "normalized_configuration",
                "normalized_aggregation",
                "sumscore_configuration",
            )
        )
        normalized_by_key = {
            candidate.key: candidate for candidate in result.normalized.candidates
        }
        for query_id in query_ids:
            normalized_key = result.normalized.selected_config_by_outer[query_id]
            writer.writerow(
                (
                    query_id,
                    normalized_key,
                    normalized_by_key[normalized_key].aggregation,
                    result.sumscore.selected_config_by_outer[query_id],
                )
            )

    run_specs = {
        "response_normalized_nested": (
            "normalized_run.trec",
            result.config.normalized_run_id,
        ),
        "response_sumscore_nested": (
            "sumscore_control.trec",
            result.config.sumscore_run_id,
        ),
        "tfidf_fixed_k10": ("tfidf_control.trec", result.config.tfidf_run_id),
        "global_prior": ("global_prior_control.trec", result.config.prior_run_id),
    }
    for system, (filename, run_id) in run_specs.items():
        write_trec_run(destination / filename, result.rankings[system], run_id=run_id)

    validations = {
        system: validate_trec_run(
            destination / filename,
            expected_query_ids=result.normalized.query_ids,
            expected_model_ids=result.normalized.model_ids,
        )
        for system, (filename, _run_id) in run_specs.items()
    }
    run_parity = _sumscore_run_parity(
        destination / "sumscore_control.trec",
        (
            result.config.sumscore_reference_run_path
            if (
                len(result.normalized.query_ids),
                len(result.normalized.model_ids),
            )
            == (322, 150)
            else None
        ),
        expected_run_id=result.config.sumscore_run_id,
    )
    validation = {
        "inputs": {
            "query_count": len(result.normalized.query_ids),
            "candidate_model_count": len(result.normalized.model_ids),
            "normalized_candidate_count": len(result.normalized.candidates),
            "sumscore_candidate_count": len(result.sumscore.candidates),
            "combined_candidate_count": len(result.score_result.candidates),
        },
        "protocol": {
            "score_tensor_shape": list(result.score_result.model_scores.shape),
            "normalized_sumscore_selected_independently": True,
            "normalized_selector_excludes_sumscore": True,
            "retrieval_shared_across_aggregations": True,
            "outer_qrels_excluded_from_selection": True,
            "decision_gate_count": len(result.decision["gates"]),
        },
        "runs": validations,
        "sumscore_reference": result.summary["sumscore_reference_validation"],
        "sumscore_run_parity": run_parity,
        "execution_provenance": root_config["execution_provenance"],
        "sub_artifacts": {
            "normalized": normalized_paths,
            "sumscore": sumscore_paths,
        },
    }
    validation["artifact_sha256"] = {
        name: _sha256(destination / name)
        for name in (
            "config.json",
            "summary.json",
            "diagnostics.json",
            "decision.json",
            "per_query.tsv",
            "selections.tsv",
            "normalized_run.trec",
            "sumscore_control.trec",
            "tfidf_control.trec",
            "global_prior_control.trec",
        )
    }
    _write_json(destination / "validation.json", validation)

    paths = {
        name: str(destination / name)
        for name in (
            "config.json",
            "summary.json",
            "diagnostics.json",
            "decision.json",
            "per_query.tsv",
            "selections.tsv",
            "normalized_run.trec",
            "sumscore_control.trec",
            "tfidf_control.trec",
            "global_prior_control.trec",
            "validation.json",
        )
    }
    paths["normalized_artifacts"] = str(destination / "normalized")
    paths["sumscore_artifacts"] = str(destination / "sumscore")
    return paths


def _validate_config(config: ResponseNormalizationConfig) -> None:
    if tuple(candidate.key for candidate in config.sumscore_candidates) != tuple(
        candidate.key for candidate in DEFAULT_RESPONSE_CANDIDATES
    ):
        raise ValueError("SumScore candidates must match the frozen 24-candidate grid")
    if tuple(candidate.key for candidate in config.normalized_candidates) != tuple(
        candidate.key for candidate in NORMALIZED_RESPONSE_CANDIDATES
    ):
        raise ValueError("normalized candidates must match the frozen 144-candidate grid")
    if config.cutoff != 10:
        raise ValueError("response normalization is frozen at cutoff=10")
    if (
        isinstance(config.bootstrap_resamples, bool)
        or not isinstance(config.bootstrap_resamples, int)
        or config.bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")


def _validate_combined_score_result(
    score_result: Any,
    *,
    candidates: Sequence[ResponseCandidate],
    query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> None:
    if tuple(candidate.key for candidate in score_result.candidates) != tuple(
        candidate.key for candidate in candidates
    ):
        raise ValueError("combined score candidate order does not match the protocol")
    if tuple(score_result.query_ids) != tuple(query_ids):
        raise ValueError("combined score query order does not match development data")
    if tuple(score_result.model_ids) != tuple(model_ids):
        raise ValueError("combined score model order does not match development data")
    shape = (len(candidates), len(query_ids), len(model_ids))
    if np.asarray(score_result.model_scores).shape != shape:
        raise ValueError(f"combined score tensor must have shape {shape}")


def _slice_score_result(score_result: Any, indices: Sequence[int]) -> Any:
    position_tuple = tuple(int(index) for index in indices)
    if not position_tuple:
        raise ValueError("a score-result slice cannot be empty")
    expected_contiguous = tuple(range(position_tuple[0], position_tuple[-1] + 1))
    selector: slice | np.ndarray
    if position_tuple == expected_contiguous:
        # Both locked grids occupy contiguous blocks. Preserve views here so the
        # 168-candidate official tensor is not duplicated in Ryzen-class RAM.
        selector = slice(position_tuple[0], position_tuple[-1] + 1)
    else:
        selector = np.asarray(position_tuple, dtype=np.int64)
    attributes: dict[str, Any] = {
        "candidates": tuple(
            score_result.candidates[index] for index in position_tuple
        ),
        "query_ids": tuple(score_result.query_ids),
        "model_ids": tuple(score_result.model_ids),
        "model_scores": np.asarray(score_result.model_scores)[selector],
    }
    for name in (
        "hit_counts",
        "mean_hit_scores",
        "overlap_document_counts",
    ):
        if hasattr(score_result, name) and getattr(score_result, name) is not None:
            attributes[name] = np.asarray(getattr(score_result, name))[selector]
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
        "configuration_sha256",
    ):
        if hasattr(score_result, name):
            attributes[name] = getattr(score_result, name)
    complete_arrays = [attributes["model_scores"]]
    complete_arrays.extend(
        attributes[name]
        for name in (
            "hit_counts",
            "mean_hit_scores",
            "overlap_document_counts",
        )
        if name in attributes
    )
    attributes["score_tensor_sha256"] = _hash_arrays(*complete_arrays)
    return SimpleNamespace(**attributes)


def _assert_shared_controls(
    normalized: ResponseExperimentResult,
    sumscore: ResponseExperimentResult,
) -> None:
    for system in ("tfidf_fixed_k10", "global_prior"):
        if normalized.rankings[system] != sumscore.rankings[system]:
            raise AssertionError(f"{system} changed between the two nested selectors")
    for system in ("dense_nested_reference", "tfidf_fixed_k10", "global_prior"):
        left = normalized.evaluations[system]
        right = sumscore.evaluations[system]
        if left.per_query != right.per_query:
            raise AssertionError(f"{system} per-query reference changed")


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
        query_id: value.ndcg_at_k for query_id, value in baseline.per_query.items()
    }
    candidate_ndcg = {
        query_id: value.ndcg_at_k for query_id, value in candidate.per_query.items()
    }
    baseline_mrr = {
        query_id: value.reciprocal_rank
        for query_id, value in baseline.per_query.items()
    }
    candidate_mrr = {
        query_id: value.reciprocal_rank
        for query_id, value in candidate.per_query.items()
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
        "ndcg_at_10_win_tie_loss": _win_tie_loss(
            baseline_ndcg, candidate_ndcg
        ),
        "mrr_win_tie_loss": _win_tie_loss(baseline_mrr, candidate_mrr),
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


def _build_normalization_diagnostics(
    normalized: ResponseExperimentResult,
    sumscore: ResponseExperimentResult,
) -> dict[str, Any]:
    normalized_deployment = _deployment_bias(normalized)
    sumscore_deployment = _deployment_bias(sumscore)
    normalized_oof = _oof_bias_and_ties(normalized)
    sumscore_oof = _oof_bias_and_ties(sumscore)
    return {
        "deployment_behavior": {
            "normalized": normalized_deployment,
            "sumscore": sumscore_deployment,
            "gating_scope": "fixed_full_development_deployment_configurations",
            "effectiveness_estimate": "optimistic_not_oof",
        },
        "stitched_oof_robustness": {
            "normalized": normalized_oof,
            "sumscore": sumscore_oof,
            "gating": False,
        },
        "selection": {
            "normalized_configuration_frequency": dict(
                sorted(Counter(normalized.selected_config_by_outer.values()).items())
            ),
            "normalized_aggregation_frequency": dict(
                sorted(
                    Counter(
                        _candidate_by_key(normalized)[key].aggregation
                        for key in normalized.selected_config_by_outer.values()
                    ).items()
                )
            ),
            "sumscore_configuration_frequency": dict(
                sorted(Counter(sumscore.selected_config_by_outer.values()).items())
            ),
        },
    }


def _deployment_bias(result: ResponseExperimentResult) -> dict[str, Any]:
    key = result.deployment_config
    value = result.diagnostics["volume_bias_by_candidate"][key]
    spearman = value["spearman"].get(
        "inventory_vs_aggregation_score",
        value["spearman"].get("inventory_vs_sum_score"),
    )
    quartiles = value["top10_representation_by_inventory_quartile"]
    return {
        "configuration": key,
        "aggregation": _candidate_by_key(result)[key].aggregation,
        "inventory_vs_aggregation_score_spearman": spearman,
        "largest_inventory_quartile_top10_share": quartiles[-1][
            "top10_appearance_share"
        ],
        "quartiles": quartiles,
        "fixed_candidate_development_metrics": result.fixed_candidate_metrics[key],
    }


def _oof_bias_and_ties(result: ResponseExperimentResult) -> dict[str, Any]:
    inventory_names = tuple(result.retrieval_metadata["inventory_filter_names"])
    inventory_values = np.asarray(
        result.retrieval_metadata["retained_document_counts"], dtype=np.float64
    )
    inventory_by_filter = {
        name: inventory_values[index] for index, name in enumerate(inventory_names)
    }
    candidate_by_key = _candidate_by_key(result)
    candidate_position = {
        candidate.key: index for index, candidate in enumerate(result.candidates)
    }
    model_position = {
        model_id: index for index, model_id in enumerate(result.model_ids)
    }

    rho_values: list[float] = []
    largest_quartile_shares: list[float] = []
    prior_tie_positions = 0
    zero_score_positions = 0
    top_positions = 0
    for query_index, query_id in enumerate(result.query_ids):
        key = result.selected_config_by_outer[query_id]
        candidate = candidate_by_key[key]
        inventory = inventory_by_filter[candidate.filter_name]
        raw_scores = result.model_scores[candidate_position[key], query_index]
        rho = _spearman(inventory, raw_scores)
        if rho is not None:
            rho_values.append(float(rho))

        ranking = result.rankings["response_nested"][query_id]
        top = ranking[: min(10, len(ranking))]
        top10_counts = np.zeros(len(result.model_ids), dtype=np.float64)
        for model_id in top:
            top10_counts[model_position[model_id]] += 1.0
        quartiles = _quartile_diagnostics(
            inventory,
            top10_counts,
            result.model_ids,
        )
        largest_quartile_shares.append(
            float(quartiles[-1]["top10_appearance_share"])
        )

        priors = np.asarray(
            [result.selected_priors[query_id][model_id] for model_id in result.model_ids],
            dtype=np.float64,
        )
        for model_id in top:
            position = model_position[model_id]
            score = raw_scores[position]
            score_tie = np.flatnonzero(raw_scores == score)
            if len(score_tie) > 1 and np.any(priors[score_tie] != priors[position]):
                prior_tie_positions += 1
            if score == 0.0:
                zero_score_positions += 1
            top_positions += 1

    return {
        "query_count": len(result.query_ids),
        "fold_inventory_vs_score_spearman": _mean_and_range(rho_values),
        "fold_largest_quartile_top10_share": _mean_and_range(
            largest_quartile_shares
        ),
        "top10_position_count": top_positions,
        "aggregate_tie_resolved_by_prior_fraction": (
            prior_tie_positions / top_positions if top_positions else 0.0
        ),
        "zero_aggregate_score_fraction": (
            zero_score_positions / top_positions if top_positions else 0.0
        ),
        "prior_tie_position_count": prior_tie_positions,
        "zero_score_position_count": zero_score_positions,
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


def _candidate_by_key(
    result: ResponseExperimentResult,
) -> dict[str, ResponseCandidate]:
    return {candidate.key: candidate for candidate in result.candidates}


def _build_decision(
    evaluations: Mapping[str, EvaluationResult],
    comparisons: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_ndcg = evaluations["response_normalized_nested"].macro_ndcg_at_k
    sumscore_ndcg = evaluations["response_sumscore_nested"].macro_ndcg_at_k
    tfidf_interval = comparisons["tfidf_fixed_k10"]["ndcg_at_10_delta"]
    deployment = diagnostics["deployment_behavior"]["normalized"]
    rho = deployment["inventory_vs_aggregation_score_spearman"]
    quartile_share = deployment["largest_inventory_quartile_top10_share"]
    gates = {
        "ndcg_ci_above_tfidf": {
            "passed": bool(tfidf_interval["lower"] > 0.0),
            "observed": tfidf_interval["lower"],
            "operator": ">",
            "threshold": 0.0,
            "measure": "paired_95pct_ci_lower_bound_normalized_minus_tfidf_ndcg_at_10",
        },
        "ndcg_within_0_010_of_sumscore": {
            "passed": bool(normalized_ndcg >= sumscore_ndcg - MAX_NDCG_DROP_FROM_SUMSCORE),
            "observed": normalized_ndcg - sumscore_ndcg,
            "operator": ">=",
            "threshold": -MAX_NDCG_DROP_FROM_SUMSCORE,
            "measure": "normalized_minus_sumscore_oof_ndcg_at_10",
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
            "advance_normalized_aggregation"
            if passed
            else "retain_sumscore_and_stop_this_aggregation_branch"
        ),
        "strong_success": normalized_ndcg >= sumscore_ndcg,
        "gates": gates,
        "mrr_is_non_gating": True,
        "dense_comparison_is_non_gating": True,
    }


def _validate_sumscore_reference(
    config: ResponseNormalizationConfig,
    sumscore: ResponseExperimentResult,
) -> dict[str, Any]:
    official_shape = (322, 150)
    if (len(sumscore.query_ids), len(sumscore.model_ids)) != official_shape:
        return {
            "applicable": False,
            "reason": "non_official_fixture_shape",
        }
    observed = sumscore.evaluations["response_nested"]
    metric_match = math.isclose(
        observed.macro_ndcg_at_k,
        SUMSCORE_REFERENCE_NDCG_AT_10,
        rel_tol=0.0,
        abs_tol=5e-12,
    ) and math.isclose(
        observed.mean_reciprocal_rank,
        SUMSCORE_REFERENCE_MRR,
        rel_tol=0.0,
        abs_tol=5e-12,
    )
    selection = dict(Counter(sumscore.selected_config_by_outer.values()))
    selection_match = selection == SUMSCORE_REFERENCE_SELECTION
    deployment_match = sumscore.deployment_config == SUMSCORE_REFERENCE_DEPLOYMENT

    tracked_match: bool | None = None
    path = (
        Path(config.sumscore_reference_results_path)
        if config.sumscore_reference_results_path
        else None
    )
    if path is not None and path.is_file():
        tracked = json.loads(path.read_text(encoding="utf-8"))
        tracked_metrics = tracked["metrics"]["response_nested"]
        tracked_match = math.isclose(
            observed.macro_ndcg_at_k,
            float(tracked_metrics["ndcg_at_10"]),
            rel_tol=0.0,
            abs_tol=5e-12,
        ) and math.isclose(
            observed.mean_reciprocal_rank,
            float(tracked_metrics["mrr"]),
            rel_tol=0.0,
            abs_tol=5e-12,
        )
    if not (metric_match and selection_match and deployment_match):
        raise AssertionError("SumScore nested control did not reproduce Baseline 3")
    if tracked_match is False:
        raise AssertionError("SumScore control disagrees with tracked Baseline 3 results")
    return {
        "applicable": True,
        "metric_match": metric_match,
        "selection_frequency_match": selection_match,
        "deployment_configuration_match": deployment_match,
        "tracked_results_match": tracked_match,
        "observed_ndcg_at_10": observed.macro_ndcg_at_k,
        "observed_mrr": observed.mean_reciprocal_rank,
    }


def _build_summary(
    *,
    config: ResponseNormalizationConfig,
    score_result: Any,
    normalized: ResponseExperimentResult,
    sumscore: ResponseExperimentResult,
    evaluations: Mapping[str, EvaluationResult],
    comparisons: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    decision: Mapping[str, Any],
    reference_validation: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "protocol": {
            "name": "response_volume_normalization",
            "status": "locked_before_effectiveness_scoring",
            "advisor_approved": True,
            "outer_split": "strict_leave_one_query_out",
            "inner_split": "strict_leave_one_query_out_within_outer_training_rows",
            "selection_metric": "inner_macro_ndcg_at_10",
            "ranking_tie_break": "aggregate_score_then_fold_local_prior_then_model_id",
            "normalized_selection_candidate_count": len(normalized.candidates),
            "sumscore_control_candidate_count": len(sumscore.candidates),
            "sumscore_excluded_from_primary_selector": True,
            "logprob_features": False,
            "fusion": False,
            "run_classification": "internal_automatic",
            "grid_selection_provenance": config.grid_selection_provenance,
        },
        "data": {
            "query_count": len(normalized.query_ids),
            "candidate_model_count": len(normalized.model_ids),
            "combined_score_tensor_shape": list(score_result.model_scores.shape),
        },
        "aggregation_grid": {
            "normalized_order": list(NORMALIZED_RESPONSE_AGGREGATIONS),
            "normalized_candidate_count": len(normalized.candidates),
            "sumscore_candidate_count": len(sumscore.candidates),
            "combined_candidate_count": len(score_result.candidates),
        },
        "metrics": {
            name: {
                "ndcg_at_10": evaluation.macro_ndcg_at_k,
                "mrr": evaluation.mean_reciprocal_rank,
            }
            for name, evaluation in evaluations.items()
        },
        "comparisons": dict(comparisons),
        "selection": diagnostics["selection"],
        "deployment": {
            "normalized": diagnostics["deployment_behavior"]["normalized"],
            "sumscore": diagnostics["deployment_behavior"]["sumscore"],
            "development_effectiveness_is_optimistic": True,
        },
        "fixed_candidate_development_metrics": {
            "normalized": normalized.fixed_candidate_metrics,
            "sumscore": sumscore.fixed_candidate_metrics,
        },
        "diagnostics": dict(diagnostics),
        "decision": dict(decision),
        "sumscore_reference_validation": dict(reference_validation),
        "retrieval_reproducibility": {
            "combined": _combined_retrieval_hashes(score_result),
            "normalized_slice": dict(normalized.retrieval_metadata),
            "sumscore_slice": dict(sumscore.retrieval_metadata),
        },
        "runtime": {"experiment_seconds": elapsed_seconds},
    }


def _sumscore_run_parity(
    generated: Path,
    reference_path: str | None,
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    if reference_path is None or not Path(reference_path).is_file():
        return {"applicable": False, "reason": "reference_run_not_available"}
    reference = Path(reference_path)
    generated_rows = [
        line.rsplit(maxsplit=1)[0] for line in generated.read_text(encoding="utf-8").splitlines()
    ]
    reference_rows = [
        line.rsplit(maxsplit=1)[0] for line in reference.read_text(encoding="utf-8").splitlines()
    ]
    identical_rankings_and_scores = generated_rows == reference_rows
    if not identical_rankings_and_scores:
        raise AssertionError("generated SumScore run differs from Baseline 3 run")
    return {
        "applicable": True,
        "identical_rankings_and_scores": True,
        "run_id_ignored": True,
        "generated_run_id": expected_run_id,
        "reference_sha256": _sha256(reference),
        "generated_sha256": _sha256(generated),
    }


def _combined_retrieval_hashes(score_result: Any) -> dict[str, Any]:
    return {
        name: getattr(score_result, name)
        for name in (
            "source_sha256",
            "corpus_sha256",
            "configuration_sha256",
            "score_tensor_sha256",
        )
        if hasattr(score_result, name)
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_arrays(*arrays: np.ndarray) -> str:
    """Match the retrieval layer's deterministic multi-array hash."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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


__all__ = [
    "ResponseNormalizationConfig",
    "ResponseNormalizationResult",
    "run_response_normalization_experiment",
    "write_response_normalization_artifacts",
]
