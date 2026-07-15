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
from trec_mlm.profile_multiresolution import (
    EVALUATED_CONSTRUCTION_AGGREGATE_SHA256,
    EVALUATED_CONSTRUCTION_PATH,
    EXPECTED_SYSTEMS,
    ISSUE18_FROZEN_ARTIFACT_PATH,
    load_issue18_frozen_snapshot,
    validate_multiresolution_viability_receipt,
)
from trec_mlm.runio import rankings_from_run
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
    payload: Mapping[str, Any], artifact_path: str | Path
) -> Mapping[str, Any]:
    """Expose the centralized strict receipt validator for evaluation callers."""

    return validate_multiresolution_viability_receipt(payload, artifact_path)


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


def _load_issue12_evaluation_rankings(construction: Any) -> Any:
    root = Path(EVALUATED_CONSTRUCTION_PATH)
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    if (
        tuple(inputs.get("query_ids", ())) != construction.query_ids
        or tuple(inputs.get("candidate_ids", ())) != construction.candidate_ids
    ):
        raise ValueError("issue-12 evaluation inventories changed")
    source_rankings = {
        key: json.loads(
            (root / "rankings" / f"{key}.json").read_text(encoding="utf-8")
        )
        for key in ("dense_reference", "contextual_e5_control")
    }
    rankings = {
        key: json.loads(
            (root / "rankings" / f"{key}.json").read_text(encoding="utf-8")
        )
        for key in ("profile_only", TRACK_A_SYSTEM)
    }
    for key, ranking in {**source_rankings, **rankings}.items():
        if (
            set(ranking) != set(construction.query_ids)
            or any(
                len(ranking[query_id]) != len(construction.candidate_ids)
                or set(ranking[query_id]) != set(construction.candidate_ids)
                for query_id in construction.query_ids
            )
            or rankings_from_run(root / "runs" / f"{key}.trec") != ranking
        ):
            raise ValueError(f"stored issue-12 ranking or run changed: {key}")
    return type(
        "FrozenIssue12Rankings",
        (),
        {
            "query_ids": construction.query_ids,
            "candidate_ids": construction.candidate_ids,
            "source_rankings": source_rankings,
            "rankings": rankings,
            "manifest": {"aggregate_sha256": EVALUATED_CONSTRUCTION_AGGREGATE_SHA256},
        },
    )()


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
    receipt_path = Path(viability_receipt_path)
    qrels_file = Path(qrels_path)
    destination = Path(output_path)
    _validate_sealed_paths(artifact_root, receipt_path, qrels_file, destination)
    construction = load_issue18_frozen_snapshot(artifact_root)
    manifest = _read_manifest(artifact_root)
    before = _construction_snapshot(artifact_root)

    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid viability receipt") from error
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if _sha256(receipt_path) != receipt_sha256:
        raise RuntimeError("viability receipt changed while it was being loaded")
    validated_receipt = validate_viability_receipt(receipt, artifact_root)
    viability_decision = _authorize_diagnostic_evaluation(validated_receipt)

    source = _load_issue12_evaluation_rankings(construction)
    track_a_rankings = _track_a_rankings(construction, source)

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
        "construction_aggregate_sha256": manifest["aggregate_sha256"],
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
            "before_aggregate_sha256": manifest["aggregate_sha256"],
            "after_aggregate_sha256": None,
            "complete_tree_unchanged": False,
        },
    }

    try:
        load_issue18_frozen_snapshot(artifact_root)
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
