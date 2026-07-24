"""Sealed evaluation for the frozen raw-response refinement experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from trec_mlm.metrics import EvaluationResult, evaluate_rankings
from trec_mlm.profile_evaluation import paired_interval, win_tie_loss, zero_transitions
from trec_mlm.query_knn import load_complete_qrels
from trec_mlm.raw_response_refinement import (
    RawRefinementResult,
    load_raw_response_refinement_artifact,
)


PARENT_TOLERANCE = 1e-12
LOCKED_QRELS_SHA256 = (
    "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35"
)
MATCHED_PARENTS = {
    "primary": {
        "baseline": "dense_profile_equal",
        "candidate": "raw_hierarchical_dense_equal",
    },
    "secondary": {
        "baseline": "profile_only",
        "candidate": "raw_hierarchical",
    },
}
MATCHED_PARENT_METRICS: dict[str, dict[str, float | int]] = {
    "dense_profile_equal": {
        "ndcg_at_10": 0.4202139921941393,
        "mrr": 0.45301739378333744,
        "zero_ndcg_queries": 99,
    },
    "profile_only": {
        "ndcg_at_10": 0.41492445997856287,
        "mrr": 0.441697813928065,
        "zero_ndcg_queries": 92,
    },
}
_REPEAT_RECEIPT_KEYS = {
    "status",
    "selected_record_ids_identical",
    "rankings_identical",
    "maximum_absolute_score_difference",
    "tolerance",
    "first_aggregate_sha256",
    "second_aggregate_sha256",
    "first_execution_id",
    "second_execution_id",
}


@dataclass(frozen=True, slots=True)
class RawRefinementEvaluationConfig:
    """Paths crossing the qrel-free/evaluation boundary."""

    artifact_path: str
    repeat_receipt_path: str
    qrels_path: str


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: RawRefinementEvaluationConfig) -> None:
    for name in ("artifact_path", "repeat_receipt_path", "qrels_path"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name):
            raise ValueError(f"{name} must be a non-empty path")


def primary_decision(
    *,
    ndcg_interval_lower: float,
    ndcg_point_delta: float,
    mrr_delta: float,
    zero_count: int,
    zero_rescues: int,
    integrity_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the frozen advance, diagnostic, stop, or invalid rule."""

    values = (ndcg_interval_lower, ndcg_point_delta, mrr_delta)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("decision metric values must be finite")
    if isinstance(zero_count, bool) or not isinstance(zero_count, int) or zero_count < 0:
        raise ValueError("zero_count must be a non-negative integer")
    if isinstance(zero_rescues, bool) or not isinstance(zero_rescues, int) or zero_rescues < 0:
        raise ValueError("zero_rescues must be a non-negative integer")
    integrity_passes = bool(integrity_checks) and all(
        value is True for value in integrity_checks.values()
    )
    gates = {
        "ndcg_interval_lower_above_zero": ndcg_interval_lower > 0.0,
        "mrr_not_materially_lower": mrr_delta >= -0.005,
        "zero_count_reduced": zero_count < 99,
        "integrity_passes": integrity_passes,
    }
    if not integrity_passes:
        decision = "invalid"
    elif all(gates.values()):
        decision = "advance"
    elif ndcg_point_delta > 0.0 or zero_rescues > 0:
        decision = "diagnostic"
    else:
        decision = "stop"
    return {
        "decision": decision,
        "gates": gates,
        "integrity_checks": dict(integrity_checks),
        "values": {
            "ndcg_interval_lower": ndcg_interval_lower,
            "ndcg_point_delta": ndcg_point_delta,
            "mrr_delta": mrr_delta,
            "zero_count": zero_count,
            "zero_rescues": zero_rescues,
        },
    }


def validate_refinement_repeat_receipt(
    payload: Mapping[str, Any],
    construction_sha256: str,
    construction_execution_id: str,
) -> dict[str, Any]:
    """Require the exact passing receipt emitted by the production comparator."""

    if set(payload) != _REPEAT_RECEIPT_KEYS:
        raise ValueError("repeat receipt schema changed")
    first_hash = payload.get("first_aggregate_sha256")
    second_hash = payload.get("second_aggregate_sha256")
    if construction_sha256 not in {first_hash, second_hash}:
        raise ValueError("repeat receipt does not bind the evaluated construction")
    if (
        not _is_sha256(first_hash)
        or not _is_sha256(second_hash)
        or first_hash == second_hash
    ):
        raise ValueError("repeat receipt construction identities changed")
    first_execution = payload.get("first_execution_id")
    second_execution = payload.get("second_execution_id")
    try:
        uuid.UUID(first_execution)
        uuid.UUID(second_execution)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("repeat receipt execution identities changed") from error
    bound_execution = (
        first_execution
        if construction_sha256 == first_hash
        else second_execution
    )
    if construction_execution_id != bound_execution:
        raise ValueError("repeat receipt execution does not bind the evaluated construction")
    maximum = payload.get("maximum_absolute_score_difference")
    if (
        first_execution == second_execution
        or payload.get("status") != "passed"
        or payload.get("selected_record_ids_identical") is not True
        or payload.get("rankings_identical") is not True
        or payload.get("tolerance") != 1e-12
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not isfinite(float(maximum))
        or maximum < 0.0
        or maximum > 1e-12
    ):
        raise ValueError("repeat receipt did not prove frozen repeatability")
    return dict(payload)


def _validate_qrel_inventories(
    qrels: Mapping[str, Mapping[str, float]],
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> None:
    if set(qrels) != set(query_ids):
        raise ValueError("qrel query inventory differs from frozen construction")
    candidates = set(candidate_ids)
    if any(set(values) != candidates for values in qrels.values()):
        raise ValueError("qrel candidate inventory differs from frozen construction")


def _metric_maps(result: EvaluationResult) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {query_id: metric.ndcg_at_k for query_id, metric in result.per_query.items()},
        {
            query_id: metric.reciprocal_rank
            for query_id, metric in result.per_query.items()
        },
    )


def _metrics_payload(result: EvaluationResult) -> dict[str, float | int]:
    return {
        "ndcg_at_10": result.macro_ndcg_at_k,
        "mrr": result.mean_reciprocal_rank,
        "zero_ndcg_queries": sum(
            metric.ndcg_at_k == 0.0 for metric in result.per_query.values()
        ),
    }


def _validate_parent_metrics(evaluations: Mapping[str, EvaluationResult]) -> None:
    for name, expected in MATCHED_PARENT_METRICS.items():
        actual = _metrics_payload(evaluations[name])
        for metric_name in ("ndcg_at_10", "mrr"):
            if abs(float(actual[metric_name]) - float(expected[metric_name])) > PARENT_TOLERANCE:
                raise ValueError(
                    f"matched parent {name} {metric_name} differs from frozen identity"
                )
        if actual["zero_ndcg_queries"] != expected["zero_ndcg_queries"]:
            raise ValueError(
                f"matched parent {name} zero-nDCG count differs from frozen identity"
            )


def _comparison(
    baseline_name: str,
    candidate_name: str,
    evaluations: Mapping[str, EvaluationResult],
) -> dict[str, Any]:
    baseline_ndcg, baseline_rr = _metric_maps(evaluations[baseline_name])
    candidate_ndcg, candidate_rr = _metric_maps(evaluations[candidate_name])
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "ndcg_at_10_delta": paired_interval(baseline_ndcg, candidate_ndcg),
        "mrr_delta": paired_interval(baseline_rr, candidate_rr),
        "ndcg_at_10_win_tie_loss": win_tie_loss(baseline_ndcg, candidate_ndcg),
        "mrr_win_tie_loss": win_tie_loss(baseline_rr, candidate_rr),
        "zero_transitions": zero_transitions(baseline_ndcg, candidate_ndcg),
    }


def _parent_rankings(artifact: RawRefinementResult) -> dict[str, dict[str, list[str]]]:
    construction = artifact._dependencies.construction
    rankings = construction.rankings
    if not all(name in rankings for name in MATCHED_PARENT_METRICS):
        raise ValueError("matched parent ranking identities are absent")
    construction_hash = artifact.metadata["dependency_aggregate_sha256"]["construction"]
    if construction.manifest.get("aggregate_sha256") != construction_hash:
        raise ValueError("construction dependency hash binding changed")
    return {name: rankings[name] for name in MATCHED_PARENT_METRICS}


def _top_ten_changed(
    baseline: Mapping[str, Sequence[str]], candidate: Mapping[str, Sequence[str]]
) -> bool:
    return any(
        tuple(candidate[query_id][:10]) != tuple(baseline[query_id][:10])
        for query_id in baseline
    )


def _evaluate_raw_response_refinement(
    config: RawRefinementEvaluationConfig,
) -> dict[str, Any]:
    """Validate the sealed artifact and repeat, then open qrels exactly once."""

    _validate_config(config)
    artifact = load_raw_response_refinement_artifact(config.artifact_path)
    if artifact.manifest is None or artifact.validation is None or artifact.root is None:
        raise ValueError("raw refinement artifact is not finalized")
    aggregate = artifact.manifest["aggregate_sha256"]
    construction_hash = artifact.metadata["dependency_aggregate_sha256"][
        "construction"
    ]
    before_identity = (aggregate, construction_hash)
    parent_rankings = _parent_rankings(artifact)
    receipt = validate_refinement_repeat_receipt(
        json.loads(Path(config.repeat_receipt_path).read_text(encoding="utf-8")),
        aggregate,
        artifact.execution_receipt["execution_id"],
    )

    primary_unchanged = not _top_ten_changed(
        parent_rankings["dense_profile_equal"],
        artifact.rankings["raw_hierarchical_dense_equal"],
    )
    secondary_unchanged = not _top_ten_changed(
        parent_rankings["profile_only"], artifact.rankings["raw_hierarchical"]
    )
    if primary_unchanged and secondary_unchanged:
        after = load_raw_response_refinement_artifact(config.artifact_path)
        after_identity = (
            after.manifest["aggregate_sha256"],
            after.metadata["dependency_aggregate_sha256"]["construction"],
        )
        if after_identity != before_identity:
            raise RuntimeError("construction changed during evaluation")
        return {
            "status": "stop-no-op",
            "construction_aggregate_sha256": aggregate,
            "construction_dependency_sha256": construction_hash,
            "construction_unchanged": True,
            "repeat_receipt": receipt,
            "top_10_changed": {"primary": False, "secondary": False},
            "decision": {
                "decision": "stop",
                "reason": "both frozen systems have unchanged top-10 rankings",
            },
        }

    qrels_sha256 = _sha256(config.qrels_path)
    if qrels_sha256 != LOCKED_QRELS_SHA256:
        raise ValueError("qrel SHA-256 differs from the locked sealed identity")
    qrels = load_complete_qrels(config.qrels_path)
    if _sha256(config.qrels_path) != qrels_sha256:
        raise RuntimeError("qrel file changed while it was being loaded")
    _validate_qrel_inventories(qrels, artifact.query_ids, artifact.candidate_ids)
    systems = {
        "dense_reference": artifact.dense_rankings,
        **parent_rankings,
        **artifact.rankings,
    }
    evaluations = {
        name: evaluate_rankings(rankings, qrels, k=10)
        for name, rankings in systems.items()
    }
    _validate_parent_metrics(evaluations)
    comparisons = {
        name: _comparison(spec["baseline"], spec["candidate"], evaluations)
        for name, spec in MATCHED_PARENTS.items()
    }
    comparisons["dense_diagnostics"] = {
        name: _comparison("dense_reference", name, evaluations)
        for name in ("raw_hierarchical", "raw_hierarchical_dense_equal")
    }
    primary = comparisons["primary"]
    candidate_metrics = _metrics_payload(
        evaluations["raw_hierarchical_dense_equal"]
    )
    integrity_checks = {
        **artifact.validation["integrity_checks"],
        "artifact_validation": artifact.validation.get("status") == "passed",
        "repeatability": True,
        "parent_metrics": True,
        "construction_hash_binding": True,
    }
    decision = primary_decision(
        ndcg_interval_lower=primary["ndcg_at_10_delta"]["lower"],
        ndcg_point_delta=primary["ndcg_at_10_delta"]["mean_difference"],
        mrr_delta=primary["mrr_delta"]["mean_difference"],
        zero_count=candidate_metrics["zero_ndcg_queries"],
        zero_rescues=primary["zero_transitions"][
            "dense_zero_candidate_nonzero"
        ],
        integrity_checks=integrity_checks,
    )
    per_query_rows = [
        {
            "query_id": query_id,
            "system": name,
            "ndcg_at_10": metric.ndcg_at_k,
            "reciprocal_rank": metric.reciprocal_rank,
        }
        for query_id in artifact.query_ids
        for name, result in evaluations.items()
        for metric in (result.per_query[query_id],)
    ]
    after = load_raw_response_refinement_artifact(config.artifact_path)
    after_identity = (
        after.manifest["aggregate_sha256"],
        after.metadata["dependency_aggregate_sha256"]["construction"],
    )
    if after_identity != before_identity:
        raise RuntimeError("construction changed during evaluation")
    return {
        "status": "evaluated",
        "construction_aggregate_sha256": aggregate,
        "construction_dependency_sha256": construction_hash,
        "construction_unchanged": True,
        "qrels_sha256": qrels_sha256,
        "repeat_receipt": receipt,
        "matched_parent_metrics": {
            name: dict(metrics) for name, metrics in MATCHED_PARENT_METRICS.items()
        },
        "metrics": {
            name: _metrics_payload(result) for name, result in evaluations.items()
        },
        "comparisons": comparisons,
        "per_query_rows": per_query_rows,
        "top_10_changed": {
            "primary": not primary_unchanged,
            "secondary": not secondary_unchanged,
        },
        "decision": decision,
    }


def evaluate_raw_response_refinement(
    config: RawRefinementEvaluationConfig,
) -> dict[str, Any]:
    """Return a decision receipt, including explicit invalid classifications."""

    try:
        return _evaluate_raw_response_refinement(config)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "decision": primary_decision(
                ndcg_interval_lower=0.0,
                ndcg_point_delta=0.0,
                mrr_delta=0.0,
                zero_count=99,
                zero_rescues=0,
                integrity_checks={"evaluation_valid": False},
            ),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
