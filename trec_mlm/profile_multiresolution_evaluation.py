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

from trec_mlm.metrics import EvaluationResult, evaluate_rankings
from trec_mlm.profile_evaluation import paired_interval, win_tie_loss, zero_transitions
from trec_mlm.profile_experiment import load_profile_artifacts
from trec_mlm.profile_multiresolution import (
    EXPECTED_SYSTEMS,
    load_multiresolution_artifact,
    validate_multiresolution_viability_receipt,
)
from trec_mlm.profile_retrieval import reciprocal_rank_fusion
from trec_mlm.query_knn import load_complete_qrels


EVALUATION_VERSION = 1
TRACK_A_SYSTEM = "dense_profile_profile_heavy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _construction_snapshot(root: Path) -> dict[str, dict[str, int | str]]:
    if not root.is_dir():
        raise ValueError(f"construction artifact is not a directory: {root}")
    return {
        path.relative_to(root).as_posix(): {
            "byte_size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_manifest(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid frozen construction manifest") from error
    aggregate = payload.get("aggregate_sha256")
    if not isinstance(aggregate, str) or len(aggregate) != 64:
        raise ValueError("invalid frozen construction aggregate hash")
    return payload


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
    payload: Mapping[str, Any], artifact_aggregate_sha256: str
) -> Mapping[str, Any]:
    """Expose the centralized strict receipt validator for evaluation callers."""

    return validate_multiresolution_viability_receipt(
        payload, artifact_aggregate_sha256
    )


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
    try:
        profiles = json.loads((source.root / "profiles.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid issue-12 profile availability inventory") from error
    if set(profiles) != set(construction.candidate_ids):
        raise ValueError("issue-12 profile candidate inventory changed")
    unavailable = frozenset(
        candidate_id
        for candidate_id in construction.candidate_ids
        if profiles[candidate_id].get("available") is False
    )
    reproduced = reciprocal_rank_fusion(
        query_ids=construction.query_ids,
        candidate_ids=construction.candidate_ids,
        rankings_by_signal={
            "dense": source.source_rankings["dense_reference"],
            "profile": source.rankings["profile_only"],
        },
        weights={"dense": 1.0, "profile": 2.0},
        unavailable_profile_ids=unavailable,
        k=60,
    ).rankings
    if reproduced != source.rankings[TRACK_A_SYSTEM]:
        raise ValueError("Track A reproduction differs from frozen issue-12 ranking")
    return reproduced


def _validate_output_path(artifact_path: Path, output_path: Path) -> None:
    construction = artifact_path.resolve()
    destination = output_path.resolve()
    if destination == construction or construction in destination.parents:
        raise ValueError("evaluation output must be outside construction")
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
        os.replace(temporary, output_path)
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
    destination = Path(output_path)
    construction = load_multiresolution_artifact(artifact_root)
    _validate_output_path(artifact_root, destination)
    manifest = _read_manifest(artifact_root)
    before = _construction_snapshot(artifact_root)

    receipt_path = Path(viability_receipt_path)
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid viability receipt") from error
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if _sha256(receipt_path) != receipt_sha256:
        raise RuntimeError("viability receipt changed while it was being loaded")
    validate_viability_receipt(receipt, manifest["aggregate_sha256"])

    source = load_profile_artifacts(construction.config.construction_path)
    track_a_rankings = _track_a_rankings(construction, source)

    if _sha256(receipt_path) != receipt_sha256:
        raise RuntimeError("viability receipt changed before qrels were opened")
    qrels_file = Path(qrels_path)
    qrels_sha256 = _sha256(qrels_file)
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
        "construction_aggregate_sha256": manifest["aggregate_sha256"],
        "source_issue12_aggregate_sha256": construction.config.source_aggregate_sha256,
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
            "reconstruction_matches_issue12_frozen": True,
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
            "before_aggregate_sha256": manifest["aggregate_sha256"],
            "after_aggregate_sha256": None,
            "complete_tree_unchanged": False,
        },
    }

    try:
        load_multiresolution_artifact(artifact_root)
        after_manifest = _read_manifest(artifact_root)
        after = _construction_snapshot(artifact_root)
    except (OSError, ValueError) as error:
        raise RuntimeError("construction changed during evaluation") from error
    if before != after or manifest != after_manifest:
        raise RuntimeError("construction changed during evaluation")
    if _sha256(receipt_path) != receipt_sha256:
        raise RuntimeError("viability receipt changed during evaluation")
    payload["construction_integrity"] = {
        "status": "passed",
        "before_aggregate_sha256": manifest["aggregate_sha256"],
        "after_aggregate_sha256": after_manifest["aggregate_sha256"],
        "complete_tree_unchanged": True,
        "file_count": len(before),
    }
    _write_evaluation(payload, destination)
    return payload


__all__ = [
    "evaluate_multiresolution",
    "validate_multiresolution_qrel_inventories",
    "validate_viability_receipt",
]
