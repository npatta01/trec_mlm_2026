"""Validated build input for deterministic research-hub views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
import json
import math
from pathlib import Path

from trec_mlm.experiment_ledger import LedgerBuild, build_ledger

from .load import load_hub_sources
from .model import CompletedFollowUpEvidence, HubSources, PointwisePhase0Evidence
from .pointwise_phase0 import load_pointwise_phase0_evidence
from .validate import ValidationSummary, validate_hub_sources


@dataclass(frozen=True)
class HubBuild:
    sources: HubSources
    ledger: LedgerBuild
    validation: ValidationSummary
    failure_diagnostic: Mapping[str, object] = field(default_factory=dict)
    completed_follow_up: CompletedFollowUpEvidence | None = None
    pointwise_phase0: PointwisePhase0Evidence | None = None


_COMPLETED_FOLLOW_UP_ROLE = "strongest_completed_follow_up"
_COMPLETED_FOLLOW_UP_RECEIPT = (
    "reports/trec_2026_e5_raw_response_refinement_results.json"
)
_COMPLETED_FOLLOW_UP_BASELINE = "dense_profile_equal"
_COMPLETED_FOLLOW_UP_CANDIDATE = "raw_hierarchical_dense_equal"
_COMPLETED_FOLLOW_UP_GATE_KEYS = frozenset(
    {
        "integrity_passes",
        "mrr_not_materially_lower",
        "ndcg_interval_lower_above_zero",
        "zero_count_reduced",
    }
)
_COMPLETED_FOLLOW_UP_INTEGRITY_KEYS = frozenset(
    {
        "artifact_validation",
        "construction_hash_binding",
        "dependency_bindings",
        "embedding_contract",
        "inventory",
        "parent_metrics",
        "qrel_independence",
        "repeatability",
        "semantic_reconstruction",
        "trec_completeness",
    }
)


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _finite_number(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{description} must be a finite number")
    return float(value)


def _unit_interval_number(value: object, description: str) -> float:
    result = _finite_number(value, description)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must be between 0 and 1")
    return result


def _nonnegative_count(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def _delta_with_confidence_interval(
    comparison: Mapping[str, object],
    field: str,
) -> tuple[float, float, float]:
    delta = _mapping(comparison.get(field), field)
    lower = _finite_number(delta.get("lower"), f"{field}.lower")
    mean = _finite_number(delta.get("mean_difference"), f"{field}.mean_difference")
    upper = _finite_number(delta.get("upper"), f"{field}.upper")
    if any(abs(value) > 1.0 for value in (lower, mean, upper)):
        raise ValueError(f"{field} values must be between -1 and 1")
    if not lower <= mean <= upper:
        raise ValueError(f"{field} bounds must contain mean_difference")
    return mean, lower, upper


def _require_exact_true_checks(
    value: object,
    description: str,
    required_keys: frozenset[str],
) -> None:
    checks = _mapping(value, description)
    if set(checks) != required_keys:
        raise ValueError(f"{description} keys must match frozen receipt contract")
    if any(result is not True for result in checks.values()):
        raise ValueError(f"{description} must all be true")


def _load_completed_follow_up(
    root: Path,
    sources: HubSources,
    ledger: LedgerBuild,
) -> CompletedFollowUpEvidence | None:
    studies = tuple(
        study
        for study in sources.studies
        if study.metadata.get("current_state_role") == _COMPLETED_FOLLOW_UP_ROLE
    )
    if not studies:
        return None
    if len(studies) != 1:
        raise ValueError("exactly one strongest completed follow-up study is required")
    study = studies[0]
    if study.metadata.get("status") != "concluded":
        raise ValueError(f"{study.source_path}: study status must be concluded")
    for metadata_field in (
        "system_ids",
        "representative_system_ids",
        "comparison_system_ids",
    ):
        linked_ids = study.metadata.get(metadata_field, [])
        if not isinstance(linked_ids, (list, tuple)):
            raise ValueError(f"{study.source_path}: {metadata_field} must be a list")
        if linked_ids:
            raise ValueError(f"{study.source_path}: must not link ledger system IDs")
    evidence_refs = study.metadata.get("evidence_refs", [])
    if (
        not isinstance(evidence_refs, (list, tuple))
        or _COMPLETED_FOLLOW_UP_RECEIPT not in evidence_refs
    ):
        raise ValueError(
            f"{study.source_path}: must reference {_COMPLETED_FOLLOW_UP_RECEIPT}"
        )

    payload = _mapping(
        json.loads((root / _COMPLETED_FOLLOW_UP_RECEIPT).read_text(encoding="utf-8")),
        _COMPLETED_FOLLOW_UP_RECEIPT,
    )
    if payload.get("artifact_type") != "raw_response_refinement_results":
        raise ValueError(
            f"{_COMPLETED_FOLLOW_UP_RECEIPT}: artifact_type must be "
            "raw_response_refinement_results"
        )
    evaluation = _mapping(payload.get("evaluation"), "evaluation")
    comparisons = _mapping(evaluation.get("comparisons"), "evaluation.comparisons")
    primary = _mapping(comparisons.get("primary"), "evaluation.comparisons.primary")
    if primary.get("baseline") != _COMPLETED_FOLLOW_UP_BASELINE:
        raise ValueError("primary baseline must be dense_profile_equal")
    if primary.get("candidate") != _COMPLETED_FOLLOW_UP_CANDIDATE:
        raise ValueError("primary candidate must be raw_hierarchical_dense_equal")
    ndcg_delta, ndcg_ci_lower, ndcg_ci_upper = _delta_with_confidence_interval(
        primary, "ndcg_at_10_delta"
    )
    mrr_delta, mrr_ci_lower, mrr_ci_upper = _delta_with_confidence_interval(
        primary, "mrr_delta"
    )

    metrics = _mapping(evaluation.get("metrics"), "evaluation.metrics")
    candidate_metrics = _mapping(
        metrics.get(_COMPLETED_FOLLOW_UP_CANDIDATE),
        f"evaluation.metrics.{_COMPLETED_FOLLOW_UP_CANDIDATE}",
    )
    matched_parent_metrics = _mapping(
        evaluation.get("matched_parent_metrics"),
        "evaluation.matched_parent_metrics",
    )
    parent_metrics = _mapping(
        matched_parent_metrics.get(_COMPLETED_FOLLOW_UP_BASELINE),
        f"evaluation.matched_parent_metrics.{_COMPLETED_FOLLOW_UP_BASELINE}",
    )
    ndcg_at_10 = _unit_interval_number(
        candidate_metrics.get("ndcg_at_10"), "ndcg_at_10"
    )
    mrr = _unit_interval_number(candidate_metrics.get("mrr"), "mrr")
    zero_ndcg_queries = _nonnegative_count(
        candidate_metrics.get("zero_ndcg_queries"), "zero_ndcg_queries"
    )

    decision = _mapping(evaluation.get("decision"), "evaluation.decision")
    if decision.get("decision") != "advance":
        raise ValueError("decision must be advance")
    _require_exact_true_checks(
        decision.get("gates"),
        "decision gates",
        _COMPLETED_FOLLOW_UP_GATE_KEYS,
    )
    _require_exact_true_checks(
        decision.get("integrity_checks"),
        "decision integrity checks",
        _COMPLETED_FOLLOW_UP_INTEGRITY_KEYS,
    )

    parent = next(
        (
            system
            for system in ledger.systems
            if system.system_id == _COMPLETED_FOLLOW_UP_BASELINE
        ),
        None,
    )
    if parent is None:
        raise ValueError("dense_profile_equal must be present in the ledger")
    receipt_parent = (
        _unit_interval_number(parent_metrics.get("ndcg_at_10"), "parent ndcg_at_10"),
        _unit_interval_number(parent_metrics.get("mrr"), "parent mrr"),
        _nonnegative_count(
            parent_metrics.get("zero_ndcg_queries"), "parent zero_ndcg_queries"
        ),
    )
    ledger_parent = (parent.ndcg_at_10, parent.mrr, parent.zero_ndcg_queries)
    if receipt_parent != ledger_parent:
        raise ValueError("parent metrics do not match ledger dense_profile_equal")
    if zero_ndcg_queries > parent.query_count:
        raise ValueError("zero_ndcg_queries exceeds query count")
    if receipt_parent[2] > parent.query_count:
        raise ValueError("parent zero_ndcg_queries exceeds query count")
    if not math.isclose(
        ndcg_at_10 - receipt_parent[0],
        ndcg_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("nDCG delta does not match candidate and parent metrics")
    if not math.isclose(
        mrr - receipt_parent[1],
        mrr_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("MRR delta does not match candidate and parent metrics")

    return CompletedFollowUpEvidence(
        study_id=study.record_id,
        source_path=_COMPLETED_FOLLOW_UP_RECEIPT,
        baseline_id=_COMPLETED_FOLLOW_UP_BASELINE,
        candidate_id=_COMPLETED_FOLLOW_UP_CANDIDATE,
        decision="advance",
        ndcg_at_10=ndcg_at_10,
        mrr=mrr,
        zero_ndcg_queries=zero_ndcg_queries,
        ndcg_delta=ndcg_delta,
        ndcg_ci_lower=ndcg_ci_lower,
        ndcg_ci_upper=ndcg_ci_upper,
        mrr_delta=mrr_delta,
        mrr_ci_lower=mrr_ci_lower,
        mrr_ci_upper=mrr_ci_upper,
    )


def _load_failure_diagnostic(
    root: Path,
    sources: HubSources,
) -> Mapping[str, object]:
    study = next(
        (
            record
            for record in sources.studies
            if record.record_id == "sparse-failure-analogue-diagnosis"
        ),
        None,
    )
    if study is None:
        return {}
    if study.metadata.get("status") != "concluded":
        raise ValueError(
            f"{study.source_path}: failure diagnostic study status must be concluded"
        )
    source_path = "reports/trec_2026_sparse_failure_cause_diagnostic.json"
    evidence_refs = study.metadata.get("evidence_refs", [])
    if source_path not in evidence_refs:
        raise ValueError(
            f"{study.source_path}: failure diagnostic must reference {source_path}"
        )
    payload = json.loads((root / source_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"{source_path}: receipt status must be complete")
    authored = study.metadata.get("failure_cohorts")
    if not isinstance(authored, Mapping):
        raise ValueError(
            f"{study.source_path}: failure_cohorts must be an authored mapping"
        )
    split: dict[str, object] = {}
    for key in ("fusion_21", "no_signal_54"):
        authored_bucket = authored.get(key)
        if not isinstance(authored_bucket, Mapping):
            raise ValueError(
                f"{study.source_path}: failure_cohorts.{key} must be a mapping"
            )
        authored_count = authored_bucket.get("count")
        if (
            not isinstance(authored_count, int)
            or isinstance(authored_count, bool)
            or authored_count < 0
        ):
            raise ValueError(
                f"{study.source_path}: failure_cohorts.{key}.count must be a "
                "nonnegative integer"
            )
        label = authored_bucket.get("label")
        definition = authored_bucket.get("definition")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"{study.source_path}: failure_cohorts.{key}.label must be nonempty"
            )
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(
                f"{study.source_path}: failure_cohorts.{key}.definition must be nonempty"
            )
        receipt_bucket = payload.get(key)
        receipt_count = (
            receipt_bucket.get("count") if isinstance(receipt_bucket, Mapping) else None
        )
        if (
            not isinstance(receipt_count, int)
            or isinstance(receipt_count, bool)
            or receipt_count < 0
        ):
            raise ValueError(
                f"{source_path}: {key}.count must be a nonnegative integer"
            )
        if receipt_count != authored_count:
            raise ValueError(
                f"{source_path}: {key}.count {receipt_count} does not match authored "
                f"count {authored_count} in {study.source_path}"
            )
        split[key] = {
            "count": authored_count,
            "label": label.strip(),
            "definition": definition.strip(),
        }
    return split


def build_research_hub(root: Path) -> HubBuild:
    """Load and validate every canonical input used by generated hub views."""

    sources = load_hub_sources(root)
    ledger = build_ledger(root)
    validation = validate_hub_sources(sources, ledger, root)
    return HubBuild(
        sources=sources,
        ledger=ledger,
        validation=validation,
        failure_diagnostic=_load_failure_diagnostic(root, sources),
        completed_follow_up=_load_completed_follow_up(root, sources, ledger),
        pointwise_phase0=load_pointwise_phase0_evidence(root, sources, ledger),
    )
