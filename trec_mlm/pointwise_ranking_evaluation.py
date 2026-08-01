"""Leakage-safe evaluation and deterministic receipts for pointwise Phase 0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import io
import json
from math import isclose
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from trec_mlm.artifact_io import (
    manifest_aggregate_sha256,
    manifest_entries,
    sha256_file,
    write_pretty_json,
)
from trec_mlm.metrics import (
    EvaluationResult,
    evaluate_rankings,
    paired_bootstrap_ci,
)
from trec_mlm.pointwise_ranking import (
    FROZEN_DEVELOPMENT_QUERIES_SHA256,
    FROZEN_REFINEMENT_AGGREGATE_SHA256,
    FROZEN_SELECTED_RECORD_IDS_SHA256,
    FROZEN_TOP_RECORD_INDICES_SHA256,
    FROZEN_TOP_RECORD_SIMILARITIES_SHA256,
    PairSelectionReceipt,
    PointwiseEvidence,
    PointwisePredictions,
    RRF_K,
    construct_pointwise_rankings,
    load_pointwise_evidence,
    select_pairwise_candidates,
)
from trec_mlm.query_knn import load_complete_qrels
from trec_mlm.runio import rankings_from_run, write_trec_run


BASELINE_TOLERANCE = 5e-12
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 31
WIN_TIE_LOSS_TOLERANCE = 1e-12
SYSTEMS = ("control", "prior_only", "routing_only", "pointwise_only", "combined")
FROZEN_QRELS_SHA256 = (
    "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35"
)
FROZEN_DENSE_RANKINGS_SHA256 = (
    "734d96fd42d9cbff940c9c29acb6efe6459a0452d9cadd24a6fa3074beb96fb8"
)
FROZEN_DENSE_RUN_SHA256 = (
    "ddb4f088de919d54e72b059fc5c62b598c6c8c937ceac2e30ae99417ddb96287"
)
FROZEN_DENSE_RESULTS_SHA256 = (
    "3ca4252cb37f12e9272561422fd2a45d0dcbf741770d0f544efda8acf4ddf902"
)
FROZEN_DENSE_PER_QUERY_SHA256 = (
    "ae4a0cf8bc7eb0642e1c0d315cafa3a56cf60b3d6bfb6ab63e80fb2d0de56437"
)
FROZEN_DENSE_NDCG_AT_10 = 0.35985474409504775
FROZEN_DENSE_MRR = 0.40406330811168817
FROZEN_DENSE_ZERO_NDCG = 116
OFFICIAL_QUERY_COUNT = 322
OFFICIAL_CANDIDATE_COUNT = 150


@dataclass(frozen=True, slots=True)
class FrozenEvaluation:
    """Complete in-memory result prior to deterministic serialization."""

    rankings: dict[str, dict[str, list[str]]]
    evaluations: dict[str, EvaluationResult]
    comparisons: dict[str, dict[str, Any]]
    query_slices: list[dict[str, Any]]
    candidate_slices: list[dict[str, Any]]
    per_query: list[dict[str, Any]]
    decision: dict[str, Any]


def _metric_maps(result: EvaluationResult) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {query_id: row.ndcg_at_k for query_id, row in result.per_query.items()},
        {query_id: row.reciprocal_rank for query_id, row in result.per_query.items()},
    )


def _metrics_summary(result: EvaluationResult) -> dict[str, float | int]:
    return {
        "query_count": len(result.per_query),
        "ndcg_at_10": result.macro_ndcg_at_k,
        "mrr": result.mean_reciprocal_rank,
        "zero_ndcg_queries": sum(
            row.ndcg_at_k == 0.0 for row in result.per_query.values()
        ),
    }


def _validate_pair(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> tuple[str, ...]:
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired metrics require identical non-empty query IDs")
    return tuple(sorted(baseline))


def _win_tie_loss(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, int | float]:
    query_ids = _validate_pair(baseline, candidate)
    wins = ties = losses = 0
    for query_id in query_ids:
        difference = float(candidate[query_id]) - float(baseline[query_id])
        if abs(difference) <= WIN_TIE_LOSS_TOLERANCE:
            ties += 1
        elif difference > 0.0:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "tolerance": WIN_TIE_LOSS_TOLERANCE,
    }


def _zero_transitions(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, int]:
    query_ids = _validate_pair(baseline, candidate)
    counts = {
        "zero_both": 0,
        "dense_zero_candidate_nonzero": 0,
        "dense_nonzero_candidate_zero": 0,
        "nonzero_both": 0,
    }
    for query_id in query_ids:
        baseline_zero = float(baseline[query_id]) == 0.0
        candidate_zero = float(candidate[query_id]) == 0.0
        if baseline_zero and candidate_zero:
            counts["zero_both"] += 1
        elif baseline_zero:
            counts["dense_zero_candidate_nonzero"] += 1
        elif candidate_zero:
            counts["dense_nonzero_candidate_zero"] += 1
        else:
            counts["nonzero_both"] += 1
    return {**counts, "query_count": len(query_ids)}


def comparison_receipt(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Return paired query-grain deltas, intervals, outcomes, and transitions."""

    baseline_ndcg, baseline_mrr = _metric_maps(baseline)
    candidate_ndcg, candidate_mrr = _metric_maps(candidate)

    def interval(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, Any]:
        payload = asdict(
            paired_bootstrap_ci(
                left,
                right,
                confidence_level=0.95,
                n_resamples=n_resamples,
                seed=BOOTSTRAP_SEED,
            )
        )
        payload.update({"seed": BOOTSTRAP_SEED, "generator": "python_random_mt19937"})
        return payload

    return {
        "candidate_minus_control": True,
        "ndcg_at_10_delta": interval(baseline_ndcg, candidate_ndcg),
        "mrr_delta": interval(baseline_mrr, candidate_mrr),
        "ndcg_at_10_win_tie_loss": _win_tie_loss(baseline_ndcg, candidate_ndcg),
        "mrr_win_tie_loss": _win_tie_loss(baseline_mrr, candidate_mrr),
        "zero_transitions": _zero_transitions(baseline_ndcg, candidate_ndcg),
    }


def reproduce_dense_control(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    *,
    result_receipt_path: str | Path,
    per_query_receipt_path: str | Path,
    expected_result_sha256: str,
    expected_per_query_sha256: str,
    expected_ndcg: float,
    expected_mrr: float,
    expected_zero_ndcg: int,
) -> dict[str, Any]:
    """Require exact frozen receipt identities and metric reproduction."""

    result_path = Path(result_receipt_path)
    per_query_path = Path(per_query_receipt_path)
    observed_result_sha256 = sha256_file(result_path)
    observed_per_query_sha256 = sha256_file(per_query_path)
    if observed_result_sha256 != expected_result_sha256:
        raise ValueError("dense aggregate result receipt SHA-256 differs")
    if observed_per_query_sha256 != expected_per_query_sha256:
        raise ValueError("dense per-query result receipt SHA-256 differs")

    try:
        tracked = json.loads(result_path.read_text(encoding="utf-8"))["metrics"][
            "dense_nested"
        ]
        tracked_ndcg = float(tracked["ndcg_at_10"])
        tracked_mrr = float(tracked["mrr"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("dense aggregate result receipt schema differs") from error
    for label, observed, frozen in (
        ("tracked nDCG@10", tracked_ndcg, expected_ndcg),
        ("tracked MRR", tracked_mrr, expected_mrr),
    ):
        if not isclose(observed, frozen, rel_tol=0.0, abs_tol=BASELINE_TOLERANCE):
            raise ValueError(f"dense aggregate {label} differs")

    evaluation = evaluate_rankings(rankings, qrels, k=10)
    zero_count = sum(row.ndcg_at_k == 0.0 for row in evaluation.per_query.values())
    for label, observed, frozen in (
        ("nDCG@10", evaluation.macro_ndcg_at_k, expected_ndcg),
        ("MRR", evaluation.mean_reciprocal_rank, expected_mrr),
    ):
        if not isclose(observed, frozen, rel_tol=0.0, abs_tol=BASELINE_TOLERANCE):
            raise ValueError(f"dense reproduced {label} differs")
    if zero_count != expected_zero_ndcg:
        raise ValueError("dense reproduced zero-nDCG query count differs")

    with per_query_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = {
            "query_id",
            "dense_ndcg_at_10",
            "dense_reciprocal_rank",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError("dense per-query result receipt schema differs")
        tracked_rows: dict[str, tuple[float, float]] = {}
        for row in reader:
            query_id = row["query_id"]
            if not query_id or query_id in tracked_rows:
                raise ValueError("dense per-query result receipt IDs differ")
            try:
                tracked_rows[query_id] = (
                    float(row["dense_ndcg_at_10"]),
                    float(row["dense_reciprocal_rank"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError("dense per-query result receipt values differ") from error
    if set(tracked_rows) != set(evaluation.per_query):
        raise ValueError("dense per-query result receipt IDs differ")
    for query_id, metric in evaluation.per_query.items():
        tracked_query_ndcg, tracked_query_mrr = tracked_rows[query_id]
        if not isclose(
            tracked_query_ndcg,
            metric.ndcg_at_k,
            rel_tol=0.0,
            abs_tol=BASELINE_TOLERANCE,
        ) or not isclose(
            tracked_query_mrr,
            metric.reciprocal_rank,
            rel_tol=0.0,
            abs_tol=BASELINE_TOLERANCE,
        ):
            raise ValueError(f"dense per-query metric differs for {query_id!r}")
    return {
        "status": "passed",
        "absolute_tolerance": BASELINE_TOLERANCE,
        "result_receipt_sha256": observed_result_sha256,
        "per_query_receipt_sha256": observed_per_query_sha256,
        "per_query_values_checked": len(tracked_rows),
        **_metrics_summary(evaluation),
    }


def _validate_complete_rankings(
    rankings: Mapping[str, Sequence[str]],
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> dict[str, list[str]]:
    if set(rankings) != set(query_ids):
        raise ValueError("ranking query inventory changed")
    expected_candidates = set(candidate_ids)
    output: dict[str, list[str]] = {}
    for query_id in query_ids:
        ranking = list(rankings[query_id])
        if len(ranking) != len(candidate_ids) or set(ranking) != expected_candidates:
            raise ValueError(f"ranking candidate inventory changed for {query_id!r}")
        output[query_id] = ranking
    return output


def validate_qrel_matrix(
    qrels: Mapping[str, Mapping[str, int | float]],
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    expected_query_count: int = OFFICIAL_QUERY_COUNT,
    expected_candidate_count: int = OFFICIAL_CANDIDATE_COUNT,
) -> None:
    """Require the exact frozen complete query-by-candidate qrel matrix."""

    if len(query_ids) != expected_query_count:
        raise ValueError(
            f"expected {expected_query_count} queries, observed {len(query_ids)}"
        )
    if len(candidate_ids) != expected_candidate_count:
        raise ValueError(
            f"expected {expected_candidate_count} candidates, observed {len(candidate_ids)}"
        )
    if len(set(query_ids)) != len(query_ids) or set(qrels) != set(query_ids):
        raise ValueError("qrel query inventory differs from frozen predictions")
    expected_candidates = set(candidate_ids)
    if len(expected_candidates) != len(candidate_ids) or any(
        len(values) != len(candidate_ids) or set(values) != expected_candidates
        for values in qrels.values()
    ):
        raise ValueError("qrel candidate inventory differs from frozen predictions")


def _combined_rankings(
    pointwise: Mapping[str, Sequence[str]],
    dense: Mapping[str, Sequence[str]],
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for query_id in query_ids:
        pointwise_ranks = {
            candidate_id: rank
            for rank, candidate_id in enumerate(pointwise[query_id], start=1)
        }
        dense_ranks = {
            candidate_id: rank
            for rank, candidate_id in enumerate(dense[query_id], start=1)
        }
        output[query_id] = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -(
                    1.0 / (RRF_K + pointwise_ranks[candidate_id])
                    + 1.0 / (RRF_K + dense_ranks[candidate_id])
                ),
                dense_ranks[candidate_id],
                pointwise_ranks[candidate_id],
                candidate_id,
            ),
        )
    return output


def _slice_metrics(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    qrels: Mapping[str, Mapping[str, int | float]],
    query_ids: Sequence[str],
) -> dict[str, dict[str, float | int] | None]:
    if not query_ids:
        return {system: None for system in SYSTEMS}
    subset_qrels = {query_id: qrels[query_id] for query_id in query_ids}
    return {
        system: _metrics_summary(evaluate_rankings(system_rankings, subset_qrels, k=10))
        for system, system_rankings in rankings.items()
    }


def _query_slice_rows(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    evaluations: Mapping[str, EvaluationResult],
    qrels: Mapping[str, Mapping[str, int | float]],
    predictions: PointwisePredictions,
) -> list[dict[str, Any]]:
    density = {
        query_id: sum(float(value) > 0.0 for value in values.values())
        for query_id, values in qrels.items()
    }
    dense_metrics = evaluations["control"].per_query
    specifications: list[tuple[str, list[tuple[str, list[str]]]]] = [
        (
            "qrel_positive_density",
            [
                ("le_5", [query_id for query_id in predictions.query_ids if density[query_id] <= 5]),
                (
                    "6_20",
                    [query_id for query_id in predictions.query_ids if 6 <= density[query_id] <= 20],
                ),
                ("gt_20", [query_id for query_id in predictions.query_ids if density[query_id] > 20]),
            ],
        ),
        (
            "dense_ndcg_status",
            [
                (
                    "zero",
                    [query_id for query_id in predictions.query_ids if dense_metrics[query_id].ndcg_at_k == 0.0],
                ),
                (
                    "nonzero",
                    [query_id for query_id in predictions.query_ids if dense_metrics[query_id].ndcg_at_k != 0.0],
                ),
            ],
        ),
        (
            "query_similarity_quartile",
            [
                (
                    str(quartile),
                    [
                        query_id
                        for query_id in predictions.query_ids
                        if predictions.query_similarity_quartiles[query_id] == quartile
                    ],
                )
                for quartile in range(1, 5)
            ],
        ),
    ]
    return [
        {
            "family": family,
            "label": label,
            "query_count": len(query_ids),
            "query_ids": query_ids,
            "metrics": _slice_metrics(rankings, qrels, query_ids),
        }
        for family, groups in specifications
        for label, query_ids in groups
    ]


def _candidate_slice_rows(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    qrels: Mapping[str, Mapping[str, int | float]],
    predictions: PointwisePredictions,
) -> list[dict[str, Any]]:
    families = (
        ("audited_count_quartile", predictions.candidate_audited_quartiles),
        ("smoothed_prior_quartile", predictions.candidate_prior_quartiles),
    )
    rows: list[dict[str, Any]] = []
    for family, labels in families:
        for quartile in range(1, 5):
            candidate_ids = [
                candidate_id
                for candidate_id in predictions.candidate_ids
                if labels[candidate_id] == quartile
            ]
            if not candidate_ids:
                metrics: dict[str, dict[str, float | int] | None] = {
                    system: None for system in SYSTEMS
                }
            else:
                candidate_set = set(candidate_ids)
                filtered_qrels = {
                    query_id: {
                        candidate_id: value
                        for candidate_id, value in values.items()
                        if candidate_id in candidate_set
                    }
                    for query_id, values in qrels.items()
                }
                metrics = {
                    system: _metrics_summary(
                        evaluate_rankings(
                            {
                                query_id: [
                                    candidate_id
                                    for candidate_id in ranking
                                    if candidate_id in candidate_set
                                ]
                                for query_id, ranking in system_rankings.items()
                            },
                            filtered_qrels,
                            k=10,
                        )
                    )
                    for system, system_rankings in rankings.items()
                }
            rows.append(
                {
                    "family": family,
                    "label": str(quartile),
                    "candidate_count": len(candidate_ids),
                    "candidate_ids": candidate_ids,
                    "metrics": metrics,
                }
            )
    return rows


def phase0_decision(
    *,
    combined_ndcg_delta: float,
    combined_ndcg_lower: float,
    combined_mrr_delta: float,
    combined_zero_ndcg: int,
    routing_ndcg: float,
    prior_ndcg: float,
    selector_receipt: PairSelectionReceipt,
) -> dict[str, Any]:
    """Apply the exact effectiveness gate and separate selector availability."""

    checks = {
        "combined_ndcg_point_positive": combined_ndcg_delta > 0.0,
        "combined_ndcg_interval_lower_above_zero": combined_ndcg_lower > 0.0,
        "combined_mrr_delta_at_least_minus_0_005": combined_mrr_delta >= -0.005,
        "combined_zero_ndcg_below_116": combined_zero_ndcg < 116,
        "routing_ndcg_above_prior_ndcg": routing_ndcg > prior_ndcg,
    }
    decision = "pass" if all(checks.values()) else "fail"
    selector_available = (
        selector_receipt.selected_pair_count >= 500
        and selector_receipt.source_question_count >= 100
    )
    return {
        "decision": decision,
        "checks": checks,
        "values": {
            "combined_ndcg_delta": combined_ndcg_delta,
            "combined_ndcg_lower": combined_ndcg_lower,
            "combined_mrr_delta": combined_mrr_delta,
            "combined_zero_ndcg": combined_zero_ndcg,
            "routing_ndcg": routing_ndcg,
            "prior_ndcg": prior_ndcg,
        },
        "selector_availability": {
            "available": selector_available,
            "selected_pair_count": selector_receipt.selected_pair_count,
            "source_question_count": selector_receipt.source_question_count,
            "minimum_pair_count": 500,
            "minimum_source_question_count": 100,
            "selector_version": selector_receipt.selector_version,
        },
        "recommendation": (
            "recommend_500_pair_pilot"
            if decision == "pass" and selector_available
            else "stop_pairwise_work"
        ),
    }


def evaluate_frozen_rankings(
    evidence: PointwiseEvidence,
    predictions: PointwisePredictions,
    dense_rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    selector_receipt: PairSelectionReceipt,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> FrozenEvaluation:
    """Evaluate the five frozen systems and all prespecified slices."""

    if tuple(evidence.candidate_ids) != tuple(predictions.candidate_ids):
        raise ValueError("pointwise evidence and prediction candidates differ")
    if set(qrels) != set(predictions.query_ids):
        raise ValueError("qrel query inventory differs from pointwise predictions")
    expected_candidates = set(predictions.candidate_ids)
    if any(set(values) != expected_candidates for values in qrels.values()):
        raise ValueError("qrel candidate inventory differs from pointwise predictions")

    control = _validate_complete_rankings(
        dense_rankings,
        query_ids=predictions.query_ids,
        candidate_ids=predictions.candidate_ids,
    )
    prior = _validate_complete_rankings(
        predictions.prior_rankings,
        query_ids=predictions.query_ids,
        candidate_ids=predictions.candidate_ids,
    )
    routing = _validate_complete_rankings(
        predictions.routing_rankings,
        query_ids=predictions.query_ids,
        candidate_ids=predictions.candidate_ids,
    )
    pointwise = _validate_complete_rankings(
        predictions.pointwise_rankings,
        query_ids=predictions.query_ids,
        candidate_ids=predictions.candidate_ids,
    )
    rankings = {
        "control": control,
        "prior_only": prior,
        "routing_only": routing,
        "pointwise_only": pointwise,
        "combined": _combined_rankings(
            pointwise,
            control,
            query_ids=predictions.query_ids,
            candidate_ids=predictions.candidate_ids,
        ),
    }
    evaluations = {
        system: evaluate_rankings(system_rankings, qrels, k=10)
        for system, system_rankings in rankings.items()
    }
    comparisons = {
        system: comparison_receipt(
            evaluations["control"], evaluations[system], n_resamples=n_resamples
        )
        for system in ("pointwise_only", "combined")
    }
    combined_comparison = comparisons["combined"]
    decision = phase0_decision(
        combined_ndcg_delta=float(
            combined_comparison["ndcg_at_10_delta"]["mean_difference"]
        ),
        combined_ndcg_lower=float(combined_comparison["ndcg_at_10_delta"]["lower"]),
        combined_mrr_delta=float(combined_comparison["mrr_delta"]["mean_difference"]),
        combined_zero_ndcg=int(_metrics_summary(evaluations["combined"])["zero_ndcg_queries"]),
        routing_ndcg=evaluations["routing_only"].macro_ndcg_at_k,
        prior_ndcg=evaluations["prior_only"].macro_ndcg_at_k,
        selector_receipt=selector_receipt,
    )
    per_query = [
        {
            "query_id": query_id,
            "positive_candidate_count": sum(
                float(value) > 0.0 for value in qrels[query_id].values()
            ),
            "dense_ndcg_status": (
                "zero"
                if evaluations["control"].per_query[query_id].ndcg_at_k == 0.0
                else "nonzero"
            ),
            "query_similarity_mean": predictions.query_similarity_means[query_id],
            "query_similarity_quartile": predictions.query_similarity_quartiles[query_id],
            "metrics": {
                system: asdict(evaluation.per_query[query_id])
                for system, evaluation in evaluations.items()
            },
        }
        for query_id in predictions.query_ids
    ]
    return FrozenEvaluation(
        rankings=rankings,
        evaluations=evaluations,
        comparisons=comparisons,
        query_slices=_query_slice_rows(rankings, evaluations, qrels, predictions),
        candidate_slices=_candidate_slice_rows(rankings, qrels, predictions),
        per_query=per_query,
        decision=decision,
    )


def _result_payload(
    evaluation: FrozenEvaluation,
    evidence: PointwiseEvidence,
    selector_receipt: PairSelectionReceipt,
    *,
    input_receipt: Mapping[str, Any],
    baseline_reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "trec_2026_pointwise_ranking_phase0_evaluation",
        "version": 1,
        "status": "evaluated",
        "development_queries_previously_seen": True,
        "independent_validation": False,
        "qrel_free_ranking_construction": True,
        "inputs": dict(input_receipt),
        "baseline_reproduction": dict(baseline_reproduction),
        "protocol": {
            "ndcg_gain": "raw_positive_qrel_grade",
            "ndcg_cutoff": 10,
            "mrr_relevance": "qrel_grade_greater_than_zero",
            "rrf_k": RRF_K,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "baseline_absolute_tolerance": BASELINE_TOLERANCE,
        },
        "systems": {
            system: _metrics_summary(result)
            for system, result in evaluation.evaluations.items()
        },
        "diagnostics": {
            "candidate_audited_counts": {
                candidate_id: evidence.audited_counts[candidate_id]
                for candidate_id in evidence.candidate_ids
            },
            "candidate_supported_counts": {
                candidate_id: evidence.supported_counts[candidate_id]
                for candidate_id in evidence.candidate_ids
            },
            "candidate_audited_question_coverage": {
                candidate_id: evidence.audited_question_coverage[candidate_id]
                for candidate_id in evidence.candidate_ids
            },
            "candidate_smoothed_priors": {
                candidate_id: evidence.candidate_priors[candidate_id]
                for candidate_id in evidence.candidate_ids
            },
        },
        "comparisons_vs_control": evaluation.comparisons,
        "query_slices": evaluation.query_slices,
        "candidate_slices": evaluation.candidate_slices,
        "decision": evaluation.decision,
        "selector_receipt": {
            "selector_version": selector_receipt.selector_version,
            "target_pair_count": selector_receipt.target_pair_count,
            "eligible_pair_count": selector_receipt.eligible_pair_count,
            "eligible_source_question_count": (
                selector_receipt.eligible_source_question_count
            ),
            "selected_pair_count": selector_receipt.selected_pair_count,
            "source_question_count": selector_receipt.source_question_count,
            "initial_candidate_use_cap": selector_receipt.initial_candidate_use_cap,
            "final_candidate_use_cap": selector_receipt.final_candidate_use_cap,
            "candidate_use_counts": selector_receipt.candidate_use_counts,
        },
    }


def _per_query_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    fields = [
        "query_id",
        "positive_candidate_count",
        "dense_ndcg_status",
        "query_similarity_mean",
        "query_similarity_quartile",
    ] + [
        f"{system}_{metric}"
        for system in SYSTEMS
        for metric in ("ndcg_at_10", "reciprocal_rank")
    ]
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        flattened: dict[str, Any] = {
            "query_id": row["query_id"],
            "positive_candidate_count": row["positive_candidate_count"],
            "dense_ndcg_status": row["dense_ndcg_status"],
            "query_similarity_mean": format(float(row["query_similarity_mean"]), ".17g"),
            "query_similarity_quartile": row["query_similarity_quartile"],
        }
        metrics = row["metrics"]
        for system in SYSTEMS:
            flattened[f"{system}_ndcg_at_10"] = format(
                float(metrics[system]["ndcg_at_k"]), ".17g"
            )
            flattened[f"{system}_reciprocal_rank"] = format(
                float(metrics[system]["reciprocal_rank"]), ".17g"
            )
        writer.writerow(flattened)
    return handle.getvalue()


def _candidate_slice_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    fields = [
        "slice_family",
        "slice_label",
        "candidate_count",
        "candidate_ids_json",
        "system",
        "query_count",
        "ndcg_at_10",
        "mrr",
        "zero_ndcg_queries",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        for system in SYSTEMS:
            metric = row["metrics"][system]
            writer.writerow(
                {
                    "slice_family": row["family"],
                    "slice_label": row["label"],
                    "candidate_count": row["candidate_count"],
                    "candidate_ids_json": json.dumps(
                        row["candidate_ids"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "system": system,
                    "query_count": "" if metric is None else metric["query_count"],
                    "ndcg_at_10": (
                        "" if metric is None else format(float(metric["ndcg_at_10"]), ".17g")
                    ),
                    "mrr": "" if metric is None else format(float(metric["mrr"]), ".17g"),
                    "zero_ndcg_queries": (
                        "" if metric is None else metric["zero_ndcg_queries"]
                    ),
                }
            )
    return handle.getvalue()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_evaluation_outputs(
    evaluation: FrozenEvaluation,
    evidence: PointwiseEvidence,
    predictions: PointwisePredictions,
    selector_receipt: PairSelectionReceipt,
    *,
    artifact_dir: str | Path,
    result_export_path: str | Path,
    per_query_export_path: str | Path,
    candidate_slice_export_path: str | Path,
    input_receipt: Mapping[str, Any],
    baseline_reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically emit a complete ignored bundle and sanitized tracked exports."""

    destination = Path(artifact_dir)
    if destination.exists():
        raise FileExistsError(f"artifact destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    result_payload = _result_payload(
        evaluation,
        evidence,
        selector_receipt,
        input_receipt=input_receipt,
        baseline_reproduction=baseline_reproduction,
    )
    per_query_payload = _per_query_tsv(evaluation.per_query)
    candidate_slice_payload = _candidate_slice_tsv(evaluation.candidate_slices)
    try:
        rankings_dir = staging / "rankings"
        runs_dir = staging / "runs"
        rankings_dir.mkdir()
        runs_dir.mkdir()
        for system in SYSTEMS:
            write_pretty_json(rankings_dir / f"{system}.json", evaluation.rankings[system])
            write_trec_run(
                runs_dir / f"{system}.trec",
                evaluation.rankings[system],
                run_id=f"phase0_{system}",
            )
        arrays = staging / "pointwise_scores"
        arrays.mkdir()
        np.save(arrays / "candidate_priors.npy", predictions.candidate_priors, allow_pickle=False)
        np.save(arrays / "local_scores.npy", predictions.local_scores, allow_pickle=False)
        np.save(arrays / "residual_scores.npy", predictions.residual_scores, allow_pickle=False)
        np.save(
            arrays / "pointwise_rrf_scores.npy",
            predictions.pointwise_rrf_scores,
            allow_pickle=False,
        )
        write_pretty_json(staging / "results.json", result_payload)
        (staging / "per_query.tsv").write_text(per_query_payload, encoding="utf-8")
        (staging / "candidate_slices.tsv").write_text(
            candidate_slice_payload, encoding="utf-8"
        )
        write_pretty_json(
            staging / "selector_receipt.json",
            {
                **asdict(selector_receipt),
                "pairs": [asdict(pair) for pair in selector_receipt.pairs],
            },
        )
        entries = manifest_entries(staging)
        write_pretty_json(
            staging / "manifest.json",
            {
                "artifact_type": "trec_2026_pointwise_ranking_phase0_bundle",
                "version": 1,
                "files": entries,
                "aggregate_sha256": manifest_aggregate_sha256(entries),
            },
        )
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    exports = {
        "results": (Path(result_export_path), (json.dumps(
            result_payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n").encode("utf-8")),
        "per_query": (Path(per_query_export_path), per_query_payload.encode("utf-8")),
        "candidate_slices": (
            Path(candidate_slice_export_path),
            candidate_slice_payload.encode("utf-8"),
        ),
    }
    for path, payload in exports.values():
        _atomic_write_bytes(path, payload)
    return {
        "artifact_manifest_sha256": sha256_file(destination / "manifest.json"),
        "artifact_aggregate_sha256": json.loads(
            (destination / "manifest.json").read_text(encoding="utf-8")
        )["aggregate_sha256"],
        "export_sha256": {
            name: sha256_file(path) for name, (path, _) in exports.items()
        },
    }


def _load_dense_rankings(path: Path, expected_sha256: str) -> dict[str, list[str]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("frozen dense ranking JSON SHA-256 differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("frozen dense ranking JSON is malformed") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError("frozen dense ranking JSON schema differs")
    output: dict[str, list[str]] = {}
    for query_id, ranking in payload.items():
        if (
            not isinstance(query_id, str)
            or not query_id
            or not isinstance(ranking, list)
            or not ranking
            or any(not isinstance(item, str) or not item for item in ranking)
            or len(set(ranking)) != len(ranking)
        ):
            raise ValueError("frozen dense ranking JSON schema differs")
        output[query_id] = list(ranking)
    return output


def run_pointwise_phase0_evaluation(
    *,
    labels_path: str | Path,
    records_path: str | Path,
    refinement_root: str | Path,
    development_queries_path: str | Path,
    qrels_path: str | Path,
    dense_rankings_path: str | Path,
    dense_run_path: str | Path,
    dense_result_receipt_path: str | Path,
    dense_per_query_receipt_path: str | Path,
    artifact_dir: str | Path,
    result_export_path: str | Path,
    per_query_export_path: str | Path,
    candidate_slice_export_path: str | Path,
    expected_qrels_sha256: str = FROZEN_QRELS_SHA256,
    expected_dense_rankings_sha256: str = FROZEN_DENSE_RANKINGS_SHA256,
    expected_dense_run_sha256: str = FROZEN_DENSE_RUN_SHA256,
    expected_dense_results_sha256: str = FROZEN_DENSE_RESULTS_SHA256,
    expected_dense_per_query_sha256: str = FROZEN_DENSE_PER_QUERY_SHA256,
    expected_query_count: int = OFFICIAL_QUERY_COUNT,
    expected_candidate_count: int = OFFICIAL_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Run the frozen zero-call construction, verify control, then open qrels."""

    labels_file = Path(labels_path)
    records_file = Path(records_path)
    refinement = Path(refinement_root)
    development_queries = Path(development_queries_path)
    qrels_file = Path(qrels_path)
    dense_rankings_file = Path(dense_rankings_path)
    dense_run_file = Path(dense_run_path)

    # Every ranking input and qrel-free derived signal is closed before qrels
    # are hashed or opened.
    dense_rankings = _load_dense_rankings(
        dense_rankings_file, expected_dense_rankings_sha256
    )
    if sha256_file(dense_run_file) != expected_dense_run_sha256:
        raise ValueError("frozen dense TREC run SHA-256 differs")
    if rankings_from_run(dense_run_file) != dense_rankings:
        raise ValueError("frozen dense JSON and TREC run rankings differ")
    evidence = load_pointwise_evidence(labels_file, records_file)
    predictions = construct_pointwise_rankings(
        evidence,
        refinement,
        development_queries_path=development_queries,
    )
    selector_receipt = select_pairwise_candidates(evidence)
    dense_rankings = _validate_complete_rankings(
        dense_rankings,
        query_ids=predictions.query_ids,
        candidate_ids=predictions.candidate_ids,
    )

    qrels_sha256 = sha256_file(qrels_file)
    if qrels_sha256 != expected_qrels_sha256:
        raise ValueError("development qrels SHA-256 differs")
    qrels = load_complete_qrels(qrels_file, expected_query_ids=predictions.query_ids)
    if sha256_file(qrels_file) != qrels_sha256:
        raise RuntimeError("development qrels changed while being loaded")
    validate_qrel_matrix(
        qrels,
        query_ids=predictions.query_ids,
        candidate_ids=predictions.candidate_ids,
        expected_query_count=expected_query_count,
        expected_candidate_count=expected_candidate_count,
    )
    baseline_reproduction = reproduce_dense_control(
        dense_rankings,
        qrels,
        result_receipt_path=dense_result_receipt_path,
        per_query_receipt_path=dense_per_query_receipt_path,
        expected_result_sha256=expected_dense_results_sha256,
        expected_per_query_sha256=expected_dense_per_query_sha256,
        expected_ndcg=FROZEN_DENSE_NDCG_AT_10,
        expected_mrr=FROZEN_DENSE_MRR,
        expected_zero_ndcg=FROZEN_DENSE_ZERO_NDCG,
    )
    evaluation = evaluate_frozen_rankings(
        evidence,
        predictions,
        dense_rankings,
        qrels,
        selector_receipt,
    )
    input_receipt = {
        "labels_sha256": evidence.labels_sha256,
        "records_sha256": evidence.records_sha256,
        "development_queries_sha256": FROZEN_DEVELOPMENT_QUERIES_SHA256,
        "qrels_sha256": qrels_sha256,
        "refinement_aggregate_sha256": FROZEN_REFINEMENT_AGGREGATE_SHA256,
        "selected_record_ids_sha256": FROZEN_SELECTED_RECORD_IDS_SHA256,
        "top_record_indices_sha256": FROZEN_TOP_RECORD_INDICES_SHA256,
        "top_record_similarities_sha256": FROZEN_TOP_RECORD_SIMILARITIES_SHA256,
        "dense_rankings_sha256": expected_dense_rankings_sha256,
        "dense_run_sha256": expected_dense_run_sha256,
        "dense_result_receipt_sha256": expected_dense_results_sha256,
        "dense_per_query_receipt_sha256": expected_dense_per_query_sha256,
        "query_count": len(predictions.query_ids),
        "candidate_count": len(predictions.candidate_ids),
        "qrel_count": sum(len(values) for values in qrels.values()),
    }
    output_receipt = write_evaluation_outputs(
        evaluation,
        evidence,
        predictions,
        selector_receipt,
        artifact_dir=artifact_dir,
        result_export_path=result_export_path,
        per_query_export_path=per_query_export_path,
        candidate_slice_export_path=candidate_slice_export_path,
        input_receipt=input_receipt,
        baseline_reproduction=baseline_reproduction,
    )
    return {
        **output_receipt,
        "systems": {
            system: _metrics_summary(result)
            for system, result in evaluation.evaluations.items()
        },
        "decision": evaluation.decision,
        "baseline_reproduction": baseline_reproduction,
    }


__all__ = [
    "BASELINE_TOLERANCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "FrozenEvaluation",
    "comparison_receipt",
    "evaluate_frozen_rankings",
    "phase0_decision",
    "reproduce_dense_control",
    "run_pointwise_phase0_evaluation",
    "validate_qrel_matrix",
    "write_evaluation_outputs",
]
