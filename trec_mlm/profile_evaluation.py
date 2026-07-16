"""Read-only evaluation and decision receipts for frozen profile rankings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import fsum, isfinite
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import unicodedata

import numpy as np

from trec_mlm.metrics import EvaluationResult, evaluate_rankings
from trec_mlm.profile_experiment import (
    CONSTRUCTION_REPEAT_CHECKS,
    INFERENCE_REPEAT_CHECKS,
    FROZEN_RANKING_KEYS,
    load_profile_artifacts,
)
from trec_mlm.query_knn import load_complete_qrels


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026
EQUALITY_TOLERANCE = 1e-12
_QUESTION_TYPE = re.compile(r"^([A-Za-z]+)")
_QUESTION_WORDS = frozenset({"what", "who", "when", "where", "which", "how", "why"})


@dataclass(frozen=True, slots=True)
class ProfileEvaluationConfig:
    construction_path: str
    qrels_path: str
    repeat_receipt_path: str | None = None


def _validate_pair(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> tuple[str, ...]:
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired metric inputs must have identical non-empty query IDs")
    query_ids = tuple(sorted(baseline))
    for query_id in query_ids:
        for value in (baseline[query_id], candidate[query_id]):
            if not isfinite(float(value)):
                raise ValueError("paired metric values must be finite")
    return query_ids


def validate_qrel_inventories(
    qrels: Mapping[str, Mapping[str, float]],
    *,
    query_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> None:
    """Validate identities without treating irrelevant file order as data."""

    if set(qrels) != set(query_ids):
        raise ValueError("qrel query inventory differs from frozen construction")
    expected_candidates = set(candidate_ids)
    if any(set(values) != expected_candidates for values in qrels.values()):
        raise ValueError("qrel candidate inventory differs from frozen construction")


def paired_interval(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    """Return the frozen PCG64 paired percentile interval."""

    query_ids = _validate_pair(baseline, candidate)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    differences = np.asarray(
        [float(candidate[item]) - float(baseline[item]) for item in query_ids],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(
        0, len(differences), size=(n_resamples, len(differences)), endpoint=False
    )
    means = np.mean(differences[indices], axis=1, dtype=np.float64)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean_difference": float(np.mean(differences, dtype=np.float64)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": 0.95,
        "n_resamples": n_resamples,
        "seed": BOOTSTRAP_SEED,
        "generator": "PCG64",
    }


def win_tie_loss(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, int | float]:
    """Count paired outcomes using the preregistered absolute tolerance."""

    query_ids = _validate_pair(baseline, candidate)
    wins = ties = losses = 0
    for query_id in query_ids:
        difference = float(candidate[query_id]) - float(baseline[query_id])
        if abs(difference) <= EQUALITY_TOLERANCE:
            ties += 1
        elif difference > 0.0:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "tolerance": EQUALITY_TOLERANCE,
    }


def zero_transitions(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, int]:
    """Return the exhaustive four-way zero-nDCG transition table."""

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


def question_type(text: str) -> str:
    """Classify by the frozen first casefolded ASCII question word."""

    if not isinstance(text, str):
        raise TypeError("question text must be a string")
    stripped = text.lstrip()
    while stripped and unicodedata.category(stripped[0]).startswith("P"):
        stripped = stripped[1:].lstrip()
    match = _QUESTION_TYPE.match(stripped)
    if match is None:
        return "other"
    word = match.group(1).casefold()
    return word if word in _QUESTION_WORDS else "other"


def phase0_decision(*, ndcg_delta: float, mrr_delta: float, zero_count: int) -> dict[str, Any]:
    """Apply the full-precision fixed dense-plus-E5 diagnostic gate."""

    checks = {
        "ndcg_point_improves": ndcg_delta > 0.0,
        "mrr_not_materially_lower": mrr_delta >= -0.005,
        "zero_count_reduced": zero_count < 116,
    }
    decision = "pass" if all(checks.values()) else "fail"
    return {
        "decision": decision,
        "reasons": [
            name for name, passed in checks.items() if passed is (decision == "pass")
        ],
        "checks": checks,
        "values": {
            "ndcg_delta": ndcg_delta,
            "mrr_delta": mrr_delta,
            "zero_count": zero_count,
        },
    }


def profile_decision(
    *,
    ndcg_interval_lower: float,
    ndcg_point_delta: float,
    mrr_delta: float,
    zero_count: int,
    low_density_ndcg_delta: float,
    dense_zero_rescues: int,
    integrity_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Emit the exact promote/diagnostic/reject receipt classification."""

    integrity_passes = bool(integrity_checks) and all(integrity_checks.values())
    gates = {
        "ndcg_interval_lower_above_zero": ndcg_interval_lower > 0.0,
        "mrr_not_materially_lower": mrr_delta >= -0.005,
        "zero_count_reduced": zero_count < 116,
        "low_density_ndcg_improves": low_density_ndcg_delta > 0.0,
        "integrity_passes": integrity_passes,
    }
    if not integrity_passes:
        decision = "reject-invalid"
    elif all(gates.values()):
        decision = "promote"
    elif ndcg_point_delta > 0.0 or zero_count < 116 or dense_zero_rescues > 0:
        decision = "diagnostic"
    else:
        decision = "reject"
    return {
        "decision": decision,
        "reasons": [
            name
            for name, passed in gates.items()
            if (passed if decision == "promote" else not passed)
        ],
        "gates": gates,
        "integrity_checks": dict(integrity_checks),
        "values": {
            "ndcg_interval_lower": ndcg_interval_lower,
            "ndcg_point_delta": ndcg_point_delta,
            "mrr_delta": mrr_delta,
            "zero_count": zero_count,
            "low_density_ndcg_delta": low_density_ndcg_delta,
            "dense_zero_rescues": dense_zero_rescues,
        },
    }


def _metric_maps(result: EvaluationResult) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {item: metric.ndcg_at_k for item, metric in result.per_query.items()},
        {item: metric.reciprocal_rank for item, metric in result.per_query.items()},
    )


def evaluation_systems(construction: Any) -> dict[str, dict[str, list[str]]]:
    """Return only rankings locked inside the validated construction."""

    return {**construction.source_rankings, **construction.rankings}


def validate_repeat_receipt(
    payload: dict[str, Any], construction_sha256: str
) -> dict[str, Any]:
    """Require a passing receipt cryptographically bound to this construction."""

    if payload.get("status") != "passed":
        raise ValueError("repeat receipt did not pass")
    if construction_sha256 not in {
        payload.get("first_aggregate_sha256"),
        payload.get("second_aggregate_sha256"),
    }:
        raise ValueError("repeat receipt does not bind the evaluated construction")
    checks = payload.get("integrity_checks")
    if not isinstance(checks, dict) or set(checks) != CONSTRUCTION_REPEAT_CHECKS or not all(checks.values()):
        raise ValueError("repeat receipt has failed integrity checks")
    first_sha = payload.get("first_aggregate_sha256")
    second_sha = payload.get("second_aggregate_sha256")
    if not isinstance(first_sha, str) or not isinstance(second_sha, str) or first_sha == second_sha:
        raise ValueError("repeat receipt does not prove distinct construction repeats")
    maximum = payload.get("maximum_absolute_score_difference")
    if payload.get("tolerance") != 1e-5 or not isinstance(maximum, (int, float)) or maximum > 1e-5 or payload.get("rankings_identical") is not True or payload.get("canonical_profile_files_identical") is not True:
        raise ValueError("repeat receipt numeric or canonical contract changed")
    inference = payload.get("inference_repeat_receipt")
    if not isinstance(inference, dict) or inference.get("status") != "passed" or set(inference.get("integrity_checks", {})) != INFERENCE_REPEAT_CHECKS or not all(inference["integrity_checks"].values()) or inference.get("first_aggregate_sha256") == inference.get("second_aggregate_sha256") or inference.get("tolerance") != 1e-5 or not isinstance(inference.get("maximum_absolute_embedding_difference"), (int, float)) or inference["maximum_absolute_embedding_difference"] > 1e-5:
        raise ValueError("repeat receipt lacks a valid Stage 1 repeat gate")
    return payload


def _mean_difference(
    baseline: Mapping[str, float], candidate: Mapping[str, float], query_ids: list[str]
) -> float:
    return fsum(candidate[item] - baseline[item] for item in query_ids) / len(query_ids)


def summarize_slice(
    metrics: Mapping[
        str, tuple[Mapping[str, float], Mapping[str, float]]
    ],
    query_ids: list[str],
) -> dict[str, Any]:
    """Summarize effectiveness and dense-relative deltas for one fixed slice."""

    if not query_ids:
        return {"query_count": 0, "query_ids": [], "systems": {}}
    dense_ndcg, dense_mrr = metrics["dense_reference"]
    systems: dict[str, dict[str, float]] = {}
    for name, (ndcg, mrr) in metrics.items():
        systems[name] = {
            "ndcg_at_10": fsum(ndcg[item] for item in query_ids) / len(query_ids),
            "mrr": fsum(mrr[item] for item in query_ids) / len(query_ids),
            "ndcg_at_10_delta": _mean_difference(dense_ndcg, ndcg, query_ids),
            "mrr_delta": _mean_difference(dense_mrr, mrr, query_ids),
        }
    return {"query_count": len(query_ids), "query_ids": query_ids, "systems": systems}


def reportable_system_names(
    systems: Any, phase0_decision_value: str
) -> tuple[str, ...]:
    """Suppress the preregistered dormant three-way result after Phase 0 failure."""

    return tuple(
        name
        for name in systems
        if phase0_decision_value == "pass" or name != "dense_profile_e5_equal"
    )


def evaluate_profile_artifacts(config: ProfileEvaluationConfig) -> dict[str, Any]:
    """Open qrels only after validating the immutable construction manifest."""

    if config.repeat_receipt_path is None:
        raise ValueError("repeat receipt is required before qrels may be loaded")
    construction = load_profile_artifacts(config.construction_path)
    before_manifest = json.dumps(construction.manifest, sort_keys=True)
    construction_metadata = json.loads(
        (construction.root / "metadata.json").read_text(encoding="utf-8")
    )
    construction_profiles = json.loads(
        (construction.root / "profiles.json").read_text(encoding="utf-8")
    )
    profile_encoder = construction_metadata["profile_encoder"]
    construction_summary = {
        "record_count": construction_metadata["record_count"],
        "available_candidate_count": sum(
            value.get("available") is True for value in construction_profiles.values()
        ),
        "partial_response_truncation": sum(
            value.get("audit", {}).get("partial_response_truncation", 0)
            for value in construction_profiles.values()
        ),
        "full_response_truncation": sum(
            value.get("audit", {}).get("full_response_truncation", 0)
            for value in construction_profiles.values()
        ),
        "encoding_seconds": profile_encoder["encoding_seconds"],
        "peak_memory_allocated_bytes": profile_encoder[
            "peak_memory_allocated_bytes"
        ],
        "discovery_sha256": construction_metadata["source_sha256"],
        "dense_run_sha256": construction_metadata["dense_run_sha256"],
        "e5_validation_sha256": construction_metadata["e5_validation_sha256"],
        "model_revision": profile_encoder["resolved_model_revision"],
        "container_image": profile_encoder["required_container_image"],
    }
    repeat_payload = validate_repeat_receipt(
        json.loads(Path(config.repeat_receipt_path).read_text(encoding="utf-8")),
        construction.manifest["aggregate_sha256"],
    )
    qrels = load_complete_qrels(config.qrels_path)
    validate_qrel_inventories(
        qrels,
        query_ids=construction.query_ids,
        candidate_ids=construction.candidate_ids,
    )
    systems = evaluation_systems(construction)
    evaluations = {name: evaluate_rankings(ranking, qrels, k=10) for name, ranking in systems.items()}
    metric_maps = {name: _metric_maps(value) for name, value in evaluations.items()}
    dense_ndcg, dense_rr = _metric_maps(evaluations["dense_reference"])
    comparisons: dict[str, Any] = {}
    for name in FROZEN_RANKING_KEYS:
        candidate_ndcg, candidate_rr = _metric_maps(evaluations[name])
        comparisons[name] = {
            "ndcg_at_10_delta": paired_interval(dense_ndcg, candidate_ndcg),
            "mrr_delta": paired_interval(dense_rr, candidate_rr),
            "ndcg_at_10_win_tie_loss": win_tie_loss(dense_ndcg, candidate_ndcg),
            "mrr_win_tie_loss": win_tie_loss(dense_rr, candidate_rr),
            "zero_transitions": zero_transitions(dense_ndcg, candidate_ndcg),
        }
    formatted_inputs = json.loads(
        (construction.root / "formatted_inputs.json").read_text(encoding="utf-8")
    )
    target_texts = formatted_inputs["target_queries"]
    if len(target_texts) != len(construction.query_ids) or any(
        "\nQuery: " not in text for text in target_texts
    ):
        raise ValueError("locked target-query text inventory changed")
    queries = {
        query_id: text.split("\nQuery: ", maxsplit=1)[1]
        for query_id, text in zip(construction.query_ids, target_texts, strict=True)
    }
    positive_counts = {
        query_id: sum(float(value) > 0.0 for value in query_qrels.values())
        for query_id, query_qrels in qrels.items()
    }
    density_groups = {
        "at_most_5": [item for item in construction.query_ids if positive_counts[item] <= 5],
        "6_to_20": [item for item in construction.query_ids if 6 <= positive_counts[item] <= 20],
        "more_than_20": [item for item in construction.query_ids if positive_counts[item] > 20],
    }
    type_groups = {
        key: [item for item in construction.query_ids if question_type(queries[item]) == key]
        for key in (*sorted(_QUESTION_WORDS), "other")
    }
    primary_ndcg, _ = _metric_maps(evaluations["dense_profile_equal"])
    low_density_ids = [item for item in construction.query_ids if positive_counts[item] <= 20]
    phase_comparison = comparisons["dense_e5_equal"]
    phase = phase0_decision(
        ndcg_delta=phase_comparison["ndcg_at_10_delta"]["mean_difference"],
        mrr_delta=phase_comparison["mrr_delta"]["mean_difference"],
        zero_count=sum(value == 0.0 for value in _metric_maps(evaluations["dense_e5_equal"])[0].values()),
    )
    primary_comparison = comparisons["dense_profile_equal"]
    receipt = profile_decision(
        ndcg_interval_lower=primary_comparison["ndcg_at_10_delta"]["lower"],
        ndcg_point_delta=primary_comparison["ndcg_at_10_delta"]["mean_difference"],
        mrr_delta=primary_comparison["mrr_delta"]["mean_difference"],
        zero_count=sum(value == 0.0 for value in primary_ndcg.values()),
        low_density_ndcg_delta=_mean_difference(dense_ndcg, primary_ndcg, low_density_ids),
        dense_zero_rescues=primary_comparison["zero_transitions"]["dense_zero_candidate_nonzero"],
        integrity_checks={
            **construction.validation["integrity_checks"],
            "repeatability": True,
        },
    )
    reportable_names = reportable_system_names(evaluations, phase["decision"])
    reportable_metric_maps = {
        name: metric_maps[name] for name in reportable_names
    }
    after = load_profile_artifacts(config.construction_path)
    construction_unchanged = before_manifest == json.dumps(after.manifest, sort_keys=True)
    if not construction_unchanged:
        raise RuntimeError("evaluation changed the construction manifest")
    return {
        "status": "evaluated",
        "construction_aggregate_sha256": construction.manifest["aggregate_sha256"],
        "construction_unchanged": True,
        "metrics": {
            name: {
                "ndcg_at_10": value.macro_ndcg_at_k,
                "mrr": value.mean_reciprocal_rank,
                "zero_ndcg_queries": sum(metric.ndcg_at_k == 0.0 for metric in value.per_query.values()),
            }
            for name, value in evaluations.items()
            if name in reportable_names
        },
        "per_query": {
            name: {item: asdict(metric) for item, metric in value.per_query.items()}
            for name, value in evaluations.items()
            if name in reportable_names
        },
        "comparisons": {
            name: value
            for name, value in comparisons.items()
            if name in reportable_names
        },
        "density_slices": {
            key: summarize_slice(reportable_metric_maps, items)
            for key, items in density_groups.items()
        },
        "question_type_slices": {
            key: summarize_slice(reportable_metric_maps, items)
            for key, items in type_groups.items()
        },
        "phase0_receipt": phase,
        "profile_receipt": receipt,
        "three_way_reportable": phase["decision"] == "pass",
        "repeat_receipt": repeat_payload,
        "construction_summary": construction_summary,
    }


def write_profile_evaluation(
    payload: Mapping[str, Any],
    output_dir: str | Path,
    *,
    construction_path: str | Path,
) -> dict[str, str]:
    """Atomically write evaluation and decision JSON outside construction."""

    destination = Path(output_dir)
    construction = Path(construction_path).resolve()
    resolved_destination = destination.resolve()
    if resolved_destination == construction or construction in resolved_destination.parents:
        raise ValueError("evaluation output must be outside construction")
    if destination.exists():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        files = {
            "evaluation.json": payload,
            "phase0_receipt.json": payload["phase0_receipt"],
            "profile_receipt.json": payload["profile_receipt"],
            "comparisons.json": payload["comparisons"],
            "zero_transitions.json": {
                name: value["zero_transitions"]
                for name, value in payload["comparisons"].items()
            },
            "construction_integrity_receipt.json": {
                "status": "passed" if payload["construction_unchanged"] else "failed",
                "before_aggregate_sha256": payload["construction_aggregate_sha256"],
                "after_aggregate_sha256": payload["construction_aggregate_sha256"],
                "unchanged": payload["construction_unchanged"],
            },
        }
        for name, value in files.items():
            (staging / name).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        per_query_lines = ["system\tquery_id\tndcg_at_10\treciprocal_rank"]
        for system in sorted(payload["per_query"]):
            for query_id, metric in payload["per_query"][system].items():
                per_query_lines.append(
                    "\t".join(
                        (
                            system,
                            query_id,
                            format(float(metric["ndcg_at_k"]), ".17g"),
                            format(float(metric["reciprocal_rank"]), ".17g"),
                        )
                    )
                )
        (staging / "per_query.tsv").write_text(
            "\n".join(per_query_lines) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {path.name: str(path) for path in destination.iterdir() if path.is_file()}


__all__ = [
    "ProfileEvaluationConfig",
    "evaluate_profile_artifacts",
    "paired_interval",
    "phase0_decision",
    "profile_decision",
    "question_type",
    "win_tie_loss",
    "write_profile_evaluation",
    "zero_transitions",
]
