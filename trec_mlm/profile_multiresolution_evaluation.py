"""Sealed post-freeze diagnostic evaluation for issue-18 profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from trec_mlm.artifact_io import sha256_file
from trec_mlm.metrics import EvaluationResult, evaluate_rankings
from trec_mlm.profile_evaluation import paired_interval, win_tie_loss, zero_transitions
from trec_mlm.profile_multiresolution import (
    EXPECTED_SYSTEMS,
    ISSUE18_FROZEN_AGGREGATE_SHA256,
    ISSUE18_FROZEN_ARTIFACT_PATH,
    LegacyViabilityEvidence,
    load_issue18_frozen_snapshot,
    load_legacy_viability_evidence,
    validate_frozen_tree_token,
    validate_issue12_source_snapshot,
    validate_legacy_viability_evidence,
    validate_multiresolution_viability_receipt,
)
from trec_mlm.query_knn import load_complete_qrels


EVALUATION_VERSION = 1
TRACK_A_SYSTEM = "dense_profile_profile_heavy"
DIAGNOSTIC_EVALUATION_DECISIONS = frozenset(
    {"continue", "stop-query-sensitivity", "stop-no-incremental-value"}
)
CANONICAL_VIABILITY_RECEIPT_PATH = Path(
    "reports/trec_2026_issue18_pr17head_viability.json"
)
CANONICAL_QRELS_PATH = Path("data/dev/llm_dev_qrels.txt")
LOCKED_QRELS_SHA256 = (
    "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35"
)
CANONICAL_EVALUATION_OUTPUT_PATH = Path(
    "reports/trec_2026_issue18_pr17head_evaluation.json"
)


_sha256 = sha256_file


def validate_multiresolution_qrel_inventories(
    qrels: Mapping[str, Mapping[str, int | float]],
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> None:
    """Require the exact frozen query-by-candidate judgment matrix."""

    expected_queries = set(query_ids)
    if set(qrels) != expected_queries or len(qrels) != len(tuple(query_ids)):
        raise ValueError("qrel query inventory differs from frozen construction")
    expected_candidates = set(candidate_ids)
    for query_id in query_ids:
        values = qrels[query_id]
        if set(values) != expected_candidates or len(values) != len(tuple(candidate_ids)):
            raise ValueError(
                f"qrel candidate inventory differs from frozen construction for {query_id!r}"
            )


def validate_viability_receipt(
    payload: Mapping[str, Any],
    artifact_path: str | Path,
    *,
    frozen_snapshot: Any | None = None,
    legacy_evidence: LegacyViabilityEvidence | None = None,
) -> Mapping[str, Any]:
    """Expose the centralized strict receipt validator for evaluation callers."""

    return validate_multiresolution_viability_receipt(
        payload,
        artifact_path,
        frozen_snapshot=frozen_snapshot,
        legacy_evidence=legacy_evidence,
    )


def _authorize_diagnostic_evaluation(receipt: Mapping[str, Any]) -> str:
    decision = receipt.get("decision")
    if decision not in DIAGNOSTIC_EVALUATION_DECISIONS:
        raise ValueError("viability receipt does not authorize diagnostic evaluation")
    return str(decision)


def _metrics_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "query_count": len(result.per_query),
        "ndcg_at_10": result.macro_ndcg_at_k,
        "mrr": result.mean_reciprocal_rank,
        "zero_ndcg_queries": sum(
            metric.ndcg_at_k == 0.0 for metric in result.per_query.values()
        ),
        "per_query": {
            query_id: asdict(metric) for query_id, metric in result.per_query.items()
        },
    }


def _metric_maps(result: EvaluationResult) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {query_id: value.ndcg_at_k for query_id, value in result.per_query.items()},
        {query_id: value.reciprocal_rank for query_id, value in result.per_query.items()},
    )


def _paired_payload(
    baseline: EvaluationResult, candidate: EvaluationResult
) -> dict[str, Any]:
    baseline_ndcg, baseline_mrr = _metric_maps(baseline)
    candidate_ndcg, candidate_mrr = _metric_maps(candidate)
    return {
        "baseline": "query_knn",
        "candidate_minus_baseline": True,
        "ndcg_at_10_delta": paired_interval(baseline_ndcg, candidate_ndcg),
        "mrr_delta": paired_interval(baseline_mrr, candidate_mrr),
        "ndcg_at_10_win_tie_loss": win_tie_loss(baseline_ndcg, candidate_ndcg),
        "mrr_win_tie_loss": win_tie_loss(baseline_mrr, candidate_mrr),
        "zero_transitions": zero_transitions(baseline_ndcg, candidate_ndcg),
    }


def _track_a_rankings(construction: Any, source: Any) -> dict[str, list[str]]:
    if (
        source.query_ids != construction.query_ids
        or source.candidate_ids != construction.candidate_ids
        or source.source_rankings["dense_reference"] != construction.dense_rankings
        or source.manifest.get("aggregate_sha256")
        != construction.config.source_aggregate_sha256
    ):
        raise ValueError("issue-12 source identities differ from frozen construction")
    ranking = source.rankings[TRACK_A_SYSTEM]
    if set(ranking) != set(construction.query_ids) or any(
        len(ranking[query_id]) != len(construction.candidate_ids)
        or set(ranking[query_id]) != set(construction.candidate_ids)
        for query_id in construction.query_ids
    ):
        raise ValueError("stored Track A ranking inventory differs from frozen issue-12")
    return ranking


def _validate_sealed_paths(
    artifact_path: Path,
    receipt_path: Path,
    qrels_path: Path,
    output_path: Path,
) -> None:
    identities = (
        (artifact_path, Path(ISSUE18_FROZEN_ARTIFACT_PATH), "artifact"),
        (receipt_path, CANONICAL_VIABILITY_RECEIPT_PATH, "viability receipt"),
        (qrels_path, CANONICAL_QRELS_PATH, "qrels"),
        (output_path, CANONICAL_EVALUATION_OUTPUT_PATH, "evaluation output"),
    )
    for observed, expected, label in identities:
        if observed.resolve() != expected.resolve():
            raise ValueError(f"{label} must use the canonical sealed path")
    if output_path.exists():
        raise FileExistsError(f"evaluation output already exists: {output_path}")


def _write_evaluation(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.link(temporary, output_path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evaluate_multiresolution(
    artifact_path: str | Path,
    viability_receipt_path: str | Path,
    qrels_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate only immutable, frozen rankings on the already-seen dev queries."""

    artifact_root = Path(artifact_path)
    receipt_path = Path(viability_receipt_path)
    qrels_file = Path(qrels_path)
    destination = Path(output_path)
    _validate_sealed_paths(artifact_root, receipt_path, qrels_file, destination)
    construction = load_issue18_frozen_snapshot(artifact_root)
    artifact_token = construction.artifact_token
    if artifact_token is None:
        raise ValueError("frozen construction lacks its immutable artifact token")

    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid viability receipt") from error
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if _sha256(receipt_path) != receipt_sha256:
        raise RuntimeError("viability receipt changed while it was being loaded")
    legacy_evidence = load_legacy_viability_evidence()
    validated_receipt = validate_viability_receipt(
        receipt,
        artifact_root,
        frozen_snapshot=construction,
        legacy_evidence=legacy_evidence,
    )
    viability_decision = _authorize_diagnostic_evaluation(validated_receipt)

    source = construction.source_snapshot
    if source is None:
        raise ValueError("frozen construction lacks its strict issue-12 source snapshot")
    track_a_rankings = _track_a_rankings(construction, source)

    validate_issue12_source_snapshot(source)
    validate_frozen_tree_token(
        artifact_token,
        expected_aggregate_sha256=ISSUE18_FROZEN_AGGREGATE_SHA256,
    )
    validate_legacy_viability_evidence(legacy_evidence)
    if _sha256(receipt_path) != receipt_sha256:
        raise RuntimeError("viability receipt changed before qrels were opened")
    qrels_sha256 = _sha256(qrels_file)
    if qrels_sha256 != LOCKED_QRELS_SHA256:
        raise ValueError("qrel SHA-256 differs from the locked sealed identity")
    qrels = load_complete_qrels(qrels_file, expected_query_ids=construction.query_ids)
    if _sha256(qrels_file) != qrels_sha256:
        raise RuntimeError("qrel file changed while it was being loaded")
    validate_multiresolution_qrel_inventories(
        qrels,
        query_ids=construction.query_ids,
        candidate_ids=construction.candidate_ids,
    )

    baseline = evaluate_rankings(construction.dense_rankings, qrels, k=10)
    contextual_e5 = evaluate_rankings(
        source.source_rankings["contextual_e5_control"], qrels, k=10
    )
    evaluations = {
        system: evaluate_rankings(construction.rankings[system], qrels, k=10)
        for system in EXPECTED_SYSTEMS
    }
    track_a_evaluation = evaluate_rankings(track_a_rankings, qrels, k=10)

    payload: dict[str, Any] = {
        "artifact_type": "issue18_multiresolution_diagnostic_evaluation",
        "version": EVALUATION_VERSION,
        "status": "evaluated",
        "diagnostic_only": True,
        "independent_validation": False,
        "source_query_set_previously_seen": True,
        "promotion_permitted": False,
        "construction_aggregate_sha256": artifact_token.aggregate_sha256,
        "source_issue12_aggregate_sha256": construction.config.source_aggregate_sha256,
        "viability_decision": viability_decision,
        "viability_receipt_sha256": receipt_sha256,
        "qrels_sha256": qrels_sha256,
        "query_count": len(construction.query_ids),
        "candidate_count": len(construction.candidate_ids),
        "controls": {
            "query_knn": _metrics_payload(baseline),
            "contextual_e5": _metrics_payload(contextual_e5),
        },
        "systems": {
            system: _metrics_payload(evaluations[system]) for system in EXPECTED_SYSTEMS
        },
        "paired_comparisons_vs_query_knn": {
            system: _paired_payload(baseline, evaluations[system])
            for system in EXPECTED_SYSTEMS
        },
        "track_a": {
            "system": TRACK_A_SYSTEM,
            "label": "previously_seen_322_query_diagnostic_reproduction",
            "diagnostic_only": True,
            "independent_validation": False,
            "source_query_set_previously_seen": True,
            "dense_weight": 1.0,
            "profile_weight": 2.0,
            "rrf_k": 60,
            "stored_ranking_matches_issue12_frozen": True,
            "metrics": _metrics_payload(track_a_evaluation),
            "paired_comparison_vs_query_knn": _paired_payload(
                baseline, track_a_evaluation
            ),
        },
        "qrel_identity_validation": {
            "status": "passed",
            "exact_query_inventory": True,
            "exact_candidate_inventory_per_query": True,
        },
        "construction_integrity": {
            "status": "pending_post_evaluation_check",
            "before_aggregate_sha256": artifact_token.aggregate_sha256,
            "after_aggregate_sha256": None,
            "complete_tree_unchanged": False,
        },
    }

    try:
        validate_frozen_tree_token(
            artifact_token,
            expected_aggregate_sha256=ISSUE18_FROZEN_AGGREGATE_SHA256,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("construction changed during evaluation") from error
    if _sha256(receipt_path) != receipt_sha256:
        raise RuntimeError("viability receipt changed during evaluation")
    try:
        validate_issue12_source_snapshot(source)
    except (OSError, ValueError) as error:
        raise RuntimeError("issue-12 source changed during evaluation") from error
    try:
        validate_legacy_viability_evidence(legacy_evidence)
    except (OSError, ValueError) as error:
        raise RuntimeError("legacy viability evidence changed during evaluation") from error
    payload["construction_integrity"] = {
        "status": "passed",
        "before_aggregate_sha256": artifact_token.aggregate_sha256,
        "after_aggregate_sha256": artifact_token.aggregate_sha256,
        "complete_tree_unchanged": True,
        "file_count": len(artifact_token.entries) + 1,
    }
    _write_evaluation(payload, destination)
    return payload


__all__ = [
    "evaluate_multiresolution",
    "validate_multiresolution_qrel_inventories",
    "validate_viability_receipt",
]
