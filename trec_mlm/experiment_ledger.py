from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from trec_mlm.query_knn import load_development_data


EVALUATION_SCOPES = frozenset(
    {
        "dev_322x150",
        "restricted_12x20",
        "restricted_6",
        "invalid_unscored",
        "constructed_not_evaluated",
    }
)
EVIDENCE_STATUSES = frozenset(
    {
        "stored_per_query",
        "recomputed_from_complete_ranking",
        "aggregate_only",
        "restricted_diagnostic",
        "invalid",
        "unscored",
    }
)
PROMOTION_STATUSES = frozenset(
    {
        "promoted_development",
        "stable_control",
        "valid_negative",
        "diagnostic_only",
        "restricted_pilot",
        "invalid",
        "stopped_unscored",
        "constructed_not_evaluated",
    }
)
PROMOTION_BLOCKING_CAVEAT_IDS = frozenset(
    {
        "aggregate_per_query_missing",
        "behavior_not_effectiveness",
        "hard_cluster_posthoc_preflight",
        "hard_cluster_preflight_not_promotable",
        "hard_cluster_valid_negative",
        "initial_embedding_prompt_contract_superseded",
        "invalid_output_contract",
        "issue18_query_sensitivity_stop",
        "profile_posthoc_diagnostic",
        "restricted_not_comparable",
        "restricted_qrel_assisted",
        "response_normalization_stopped",
        "six_query_signal_did_not_scale",
        "source_query_stopped",
    }
)
PROMOTABLE_PER_QUERY_SOURCE_KINDS = frozenset({"issue18_per_query_map"})
ARTIFACT_STATES = frozenset(
    {
        "tracked_complete",
        "tracked_receipt_only",
        "tracked_report_snapshot",
        "local_artifact_missing",
        "construction_artifact_missing",
    }
)
MISSING_ARTIFACT_STATES = frozenset(
    {"local_artifact_missing", "construction_artifact_missing"}
)
QREL_USAGE_STATUSES = frozenset(
    {
        "training_and_evaluation",
        "evaluation_only",
        "restricted_selection_and_evaluation",
        "not_evaluated",
    }
)

LEDGER_OUTPUT_FILENAMES = (
    "trec_2026_experiment_density_slices.tsv",
    "trec_2026_experiment_ledger.json",
    "trec_2026_experiment_ledger.md",
    "trec_2026_experiment_per_query.tsv",
    "trec_2026_experiment_systems.tsv",
    "trec_2026_experiment_topic_slices.tsv",
)
TOPIC_SLICE_AGGREGATION_WARNING = (
    "overlapping_relevant_candidate_slices_are_not_official_topics_and_"
    "means_cannot_be_summed_or_averaged_as_disjoint"
)
EVIDENCE_SCOPE_CONTRACTS = {
    "stored_per_query": frozenset({"dev_322x150"}),
    "recomputed_from_complete_ranking": frozenset({"dev_322x150"}),
    "aggregate_only": frozenset({"dev_322x150"}),
    "restricted_diagnostic": frozenset({"restricted_12x20", "restricted_6"}),
    "invalid": frozenset({"invalid_unscored"}),
    "unscored": frozenset({"constructed_not_evaluated"}),
}
EVIDENCE_SCOPE_PROMOTION_CONTRACTS = frozenset(
    {
        ("stored_per_query", "dev_322x150", "promoted_development"),
        ("stored_per_query", "dev_322x150", "stable_control"),
        ("stored_per_query", "dev_322x150", "valid_negative"),
        ("stored_per_query", "dev_322x150", "diagnostic_only"),
        ("aggregate_only", "dev_322x150", "stable_control"),
        ("aggregate_only", "dev_322x150", "valid_negative"),
        ("aggregate_only", "dev_322x150", "diagnostic_only"),
        ("restricted_diagnostic", "restricted_12x20", "restricted_pilot"),
        ("restricted_diagnostic", "restricted_12x20", "diagnostic_only"),
        ("restricted_diagnostic", "restricted_6", "restricted_pilot"),
        ("restricted_diagnostic", "restricted_6", "diagnostic_only"),
        ("invalid", "invalid_unscored", "invalid"),
        ("invalid", "invalid_unscored", "stopped_unscored"),
        (
            "unscored",
            "constructed_not_evaluated",
            "constructed_not_evaluated",
        ),
        ("unscored", "constructed_not_evaluated", "diagnostic_only"),
    }
)
EVALUATION_SCOPE_COUNT_CONTRACTS = {
    "dev_322x150": frozenset({(322, 150)}),
    "restricted_12x20": frozenset({(12, 20)}),
    "restricted_6": frozenset({(6, None)}),
    "invalid_unscored": frozenset({(322, 150), (12, 20), (None, None)}),
    "constructed_not_evaluated": frozenset({(322, 150), (7759, 150)}),
}
QUALITY_METRIC_FIELDS = frozenset(
    {
        "mrr",
        "mrr_delta_vs_dense",
        "ndcg_at_10",
        "ndcg_at_k",
        "ndcg_delta_vs_dense",
        "reciprocal_rank",
        "relevant_in_top_3",
        "zero_ndcg_count",
        "zero_ndcg_queries",
    }
)
EXACT_SOURCE_POINTER_CONTRACTS = {
    ("discovery_behavior_topic_v1", "aggregate_source"): (
        "discovery_behavior_profiles",
        "/scope",
        "/profile_version",
    )
}
LITERAL_METRIC_SELECTORS = {
    ("pointwise_qwen_feasibility_6", "relevant_in_top_3"): (
        "expert_label_ranking_study",
        "What we tried",
        "Step | Scope | Outcome | What it established",
        "Six-question LLM pilot",
        "Pointwise {value} top-three",
    ),
    ("generated_listwise_feasibility_6", "relevant_in_top_3"): (
        "expert_label_ranking_study",
        "What we tried",
        "Step | Scope | Outcome | What it established",
        "Six-question LLM pilot",
        "listwise {value}",
    ),
}


@dataclass(frozen=True, slots=True)
class SystemRecord:
    system_id: str
    source_system_id: str
    aliases: tuple[str, ...]
    experiment_id: str
    experiment_family: str
    method_summary: str
    evaluation_scope: str
    evidence_status: str
    promotion_status: str
    query_count: int | None
    candidate_count: int | None
    ndcg_at_10: float | None
    mrr: float | None
    zero_ndcg_queries: int | None
    recomputed_ndcg_at_10: float | None
    recomputed_mrr: float | None
    recomputed_zero_ndcg_queries: int | None
    aggregate_payload_json: str
    aggregate_source_json: str
    per_query_source_json: str | None
    per_query_available: bool
    ranking_coverage: str
    native_scores_available: bool
    artifact_state: str
    qrel_usage: str
    run_classification: str
    source_path: str
    source_sha256: str
    blocker: str | None
    caveat_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerQueryRecord:
    system_id: str
    source_system_id: str
    experiment_family: str
    evaluation_scope: str
    query_id: str
    query_text: str
    ndcg_at_10: float
    reciprocal_rank: float
    grade_1_count: int
    grade_2_count: int
    positive_count: int
    relevant_candidates: tuple[tuple[str, int], ...]
    density_bucket: str
    zero_ndcg: bool
    metric_provenance: str
    source_path: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class SliceRecord:
    system_id: str
    slice_type: str
    slice_key: str
    query_count: int
    ndcg_at_10: float
    mrr: float
    zero_ndcg_count: int
    zero_ndcg_rate: float
    grade_1_memberships: int
    grade_2_memberships: int
    overlapping: bool
    membership_policy: str
    metric_source_path: str
    metric_source_sha256: str
    query_source_path: str
    query_source_sha256: str
    qrel_source_path: str
    qrel_source_sha256: str


@dataclass(frozen=True, slots=True)
class LedgerBuild:
    systems: tuple[SystemRecord, ...]
    per_query: tuple[PerQueryRecord, ...]
    topic_slices: tuple[SliceRecord, ...]
    density_slices: tuple[SliceRecord, ...]
    sources: tuple[dict[str, str], ...]
    caveats: tuple[dict[str, str], ...]
    validation: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def load_registry(root: Path, path: Path) -> dict[str, Any]:
    registry = json.loads((root / path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("ledger registry must be a JSON object")
    validate_registry(registry, root)
    return registry


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _validate_evidence_contract(
    system: Mapping[str, Any], *, system_id: str
) -> None:
    evidence_status = system.get("evidence_status")
    if not isinstance(evidence_status, str):
        raise ValueError(f"invalid evidence_status for {system_id}")
    if evidence_status == "recomputed_from_complete_ranking":
        raise ValueError(
            f"recomputed_from_complete_ranking evidence for {system_id} is "
            "unsupported until a complete-ranking loader exists"
        )
    if not isinstance(system.get("aggregate_source"), Mapping):
        raise ValueError(
            f"{evidence_status} evidence for {system_id} requires aggregate_source"
        )
    allowed_scopes = EVIDENCE_SCOPE_CONTRACTS[evidence_status]
    evaluation_scope = system.get("evaluation_scope")
    if evaluation_scope not in allowed_scopes:
        raise ValueError(
            f"{evidence_status} evidence for {system_id} has incompatible "
            "evaluation_scope"
        )
    allowed_counts = EVALUATION_SCOPE_COUNT_CONTRACTS[evaluation_scope]
    actual_counts = (system.get("query_count"), system.get("candidate_count"))
    if actual_counts not in allowed_counts:
        raise ValueError(
            f"{system_id} has incompatible scope counts for {evaluation_scope}; "
            f"allowed counts are {sorted(map(repr, allowed_counts))}"
        )

    per_query_source = system.get("per_query_source")
    blocker = system.get("per_query_blocker")
    ranking_coverage = system.get("ranking_coverage")
    native_scores_available = system.get("native_scores_available")
    if not isinstance(native_scores_available, bool):
        raise ValueError(
            f"{evidence_status} evidence for {system_id} requires boolean "
            "native_scores_available"
        )

    if evidence_status == "stored_per_query":
        if not isinstance(per_query_source, Mapping):
            raise ValueError(
                f"stored_per_query evidence for {system_id} requires "
                "per_query_source"
            )
        if blocker is not None:
            raise ValueError(
                f"stored_per_query evidence for {system_id} must not have a "
                "per_query_blocker"
            )
        if ranking_coverage != "per_query_metrics_only":
            raise ValueError(
                f"stored_per_query evidence for {system_id} requires "
                "ranking_coverage=per_query_metrics_only"
            )
        if native_scores_available:
            raise ValueError(
                f"stored_per_query evidence for {system_id} requires "
                "native_scores_available=false"
            )
        return

    if per_query_source is not None:
        raise ValueError(
            f"{evidence_status} evidence for {system_id} forbids per_query_source"
        )
    if not isinstance(blocker, str) or not blocker.strip():
        raise ValueError(
            f"{evidence_status} evidence for {system_id} requires a concrete "
            "per_query_blocker string"
        )
    if ranking_coverage != "none":
        raise ValueError(
            f"{evidence_status} evidence for {system_id} requires "
            "ranking_coverage=none"
        )
    if native_scores_available:
        raise ValueError(
            f"{evidence_status} evidence for {system_id} requires "
            "native_scores_available=false"
        )


def _validate_evidence_scope_promotion_contract(
    system: Mapping[str, Any], *, system_id: str
) -> None:
    contract = (
        system.get("evidence_status"),
        system.get("evaluation_scope"),
        system.get("promotion_status"),
    )
    if contract not in EVIDENCE_SCOPE_PROMOTION_CONTRACTS:
        raise ValueError(
            f"{system_id} has an incompatible evidence/scope/promotion contract: "
            f"{contract!r}"
        )


def _validate_grid_contracts(
    experiment: Mapping[str, Any],
    sources: Mapping[str, Any],
    source_values: Mapping[str, Any],
) -> None:
    contracts = experiment.get("grid_contracts", [])
    if not isinstance(contracts, list):
        raise ValueError("grid_contracts must be a list")
    for contract in contracts:
        contract = _require_mapping(contract, "grid contract")
        source_id = contract.get("source_id")
        pointer = contract.get("pointer")
        allowed_keys = contract.get("allowed_keys")
        inherited = contract.get("inherited_metadata")
        if source_id not in sources:
            raise ValueError(f"unknown grid source ID: {source_id}")
        if not isinstance(pointer, str):
            raise ValueError(f"grid pointer must be a string: {source_id}")
        if (
            not isinstance(allowed_keys, list)
            or not all(isinstance(key, str) for key in allowed_keys)
            or len(set(allowed_keys)) != len(allowed_keys)
        ):
            raise ValueError(f"grid allowed_keys must be unique strings: {source_id}")
        if inherited != {"promotion_status": "diagnostic_only"}:
            raise ValueError(
                "grid expansions may inherit only promotion_status=diagnostic_only"
            )
        grid = resolve_json_pointer(source_values[source_id], pointer)
        if isinstance(grid, Mapping):
            actual = set(grid)
        elif isinstance(grid, list) and all(
            isinstance(item, Mapping) and isinstance(item.get("key"), str)
            for item in grid
        ):
            actual = {item["key"] for item in grid}
            if len(actual) != len(grid):
                raise ValueError(f"duplicate grid keys for {source_id}: {pointer}")
        else:
            raise ValueError(
                f"grid pointer must resolve to an object or keyed list: {source_id}{pointer}"
            )
        allowed = set(allowed_keys)
        unexpected = actual - allowed
        if unexpected:
            raise ValueError(
                f"unexpected grid keys for {source_id}: {sorted(unexpected)}"
            )
        missing = allowed - actual
        if missing:
            raise ValueError(
                f"missing reviewed grid keys for {source_id}: {sorted(missing)}"
            )


def validate_registry(registry: Mapping[str, Any], root: Path) -> None:
    if registry.get("version") != 1:
        raise ValueError("ledger registry version must be 1")
    sources = registry.get("sources")
    caveats = registry.get("caveats")
    experiments = registry.get("experiments")
    if (
        not isinstance(sources, Mapping)
        or not isinstance(caveats, Mapping)
        or not isinstance(experiments, list)
    ):
        raise ValueError(
            "registry sources, caveats, and experiments have invalid shapes"
        )

    experiment_ids: set[str] = set()
    system_ids: set[str] = set()
    identity_owners: dict[str, str] = {}
    pointer_sources: list[
        tuple[str, str, Mapping[str, Any], set[str], bool]
    ] = []
    literal_snapshots: list[tuple[str, Mapping[str, Any], set[str]]] = []
    referenced_sources: set[str] = set()
    promoted_systems: list[Mapping[str, Any]] = []

    def claim_identity(identity: str, owner: str) -> None:
        existing_owner = identity_owners.get(identity)
        if existing_owner is not None and existing_owner != owner:
            raise ValueError(f"duplicate identity or alias: {identity}")
        identity_owners[identity] = owner

    for raw_experiment in experiments:
        experiment = _require_mapping(raw_experiment, "experiment")
        experiment_id = experiment.get("experiment_id")
        if (
            not isinstance(experiment_id, str)
            or not experiment_id
            or experiment_id in experiment_ids
        ):
            raise ValueError(f"duplicate experiment_id: {experiment_id}")
        experiment_ids.add(experiment_id)
        systems = experiment.get("systems")
        if not isinstance(systems, list):
            raise ValueError(f"systems must be a list for {experiment_id}")
        for raw_system in systems:
            system = _require_mapping(raw_system, "system")
            system_id = system.get("system_id")
            if not isinstance(system_id, str) or not system_id:
                raise ValueError(f"invalid system_id: {system_id}")
            if system_id in system_ids:
                raise ValueError(f"duplicate system_id: {system_id}")
            system_ids.add(system_id)
            claim_identity(system_id, system_id)
            source_system_id = system.get("source_system_id")
            if not isinstance(source_system_id, str) or not source_system_id:
                raise ValueError(
                    f"source_system_id must be a nonempty string for {system_id}"
                )
            claim_identity(source_system_id, system_id)
            aliases = system.get("aliases")
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) and bool(alias) for alias in aliases
            ):
                raise ValueError(
                    f"aliases must be a list of nonempty strings for {system_id}"
                )
            seen_aliases: set[str] = set()
            for alias in aliases:
                if alias in seen_aliases:
                    raise ValueError(f"duplicate identity or alias: {alias}")
                seen_aliases.add(alias)
                claim_identity(alias, system_id)
            system_identifiers = {
                system_id,
                source_system_id,
                *aliases,
            }
            if system.get("evaluation_scope") not in EVALUATION_SCOPES:
                raise ValueError(f"unknown evaluation_scope for {system_id}")
            if system.get("evidence_status") not in EVIDENCE_STATUSES:
                raise ValueError(f"unknown evidence_status for {system_id}")
            if system.get("promotion_status") not in PROMOTION_STATUSES:
                raise ValueError(f"unknown promotion_status for {system_id}")
            is_promoted = system.get("promotion_status") == "promoted_development"
            if is_promoted:
                promoted_systems.append(system)
            if system.get("artifact_state") not in ARTIFACT_STATES:
                raise ValueError(f"unknown artifact_state for {system_id}")
            if system.get("qrel_usage") not in QREL_USAGE_STATUSES:
                raise ValueError(f"unknown qrel_usage for {system_id}")
            if not is_promoted:
                _validate_evidence_contract(system, system_id=system_id)
            if not (
                is_promoted
                and system.get("evidence_status")
                == "recomputed_from_complete_ranking"
            ):
                _validate_evidence_scope_promotion_contract(
                    system, system_id=system_id
                )
            for source_field in ("aggregate_source", "per_query_source"):
                source_ref = system.get(source_field)
                if source_ref is not None:
                    source_ref = _require_mapping(
                        source_ref, f"{source_field} for {system_id}"
                    )
                    source_id = source_ref.get("source_id")
                    if not isinstance(source_id, str):
                        raise ValueError(
                            f"source_id required in {source_field} for {system_id}"
                        )
                    referenced_sources.add(source_id)
                    source_kind = source_ref.get("kind")
                    pointer_backed = source_kind in {
                        "grid_member",
                        "issue18_per_query_map",
                        "json_pointer",
                    }
                    if pointer_backed:
                        pointer_sources.append(
                            (
                                system_id,
                                source_field,
                                source_ref,
                                system_identifiers,
                                system.get("promotion_status")
                                == "promoted_development",
                            )
                        )
                    elif source_kind == "literal_snapshot":
                        literal_snapshots.append(
                            (system_id, source_ref, system_identifiers)
                        )
            caveat_ids = system.get("caveat_ids", [])
            if not isinstance(caveat_ids, list) or not all(
                isinstance(caveat_id, str) for caveat_id in caveat_ids
            ):
                raise ValueError(f"caveat_ids must be strings for {system_id}")
            unknown_caveats = set(caveat_ids) - set(caveats)
            if unknown_caveats:
                raise ValueError(
                    f"unknown caveat IDs for {system_id}: {sorted(unknown_caveats)}"
                )

    if len(promoted_systems) != 1:
        raise ValueError(
            "registry must contain exactly one promoted_development system; "
            f"found {len(promoted_systems)}"
        )
    promoted = promoted_systems[0]
    promoted_id = promoted["system_id"]
    if registry.get("promoted_development_system_id") != promoted_id:
        raise ValueError(
            "reviewed promoted development system ID must name the exactly-one "
            f"promoted_development row: {promoted_id}"
        )
    if promoted.get("evaluation_scope") != "dev_322x150":
        raise ValueError(
            f"promoted_development system {promoted_id} must use dev_322x150 scope"
        )
    if promoted.get("query_count") != 322 or promoted.get("candidate_count") != 150:
        raise ValueError(
            f"promoted_development system {promoted_id} must have 322x150 counts"
        )
    if promoted.get("evidence_status") != "stored_per_query":
        raise ValueError(
            f"promoted_development system {promoted_id} requires stored_per_query "
            "evidence; complete-ranking promotion has no supported custody contract"
        )
    promoted_per_query_source = promoted.get("per_query_source")
    if not isinstance(promoted_per_query_source, Mapping):
        raise ValueError(
            f"promoted_development system {promoted_id} requires a non-null valid "
            "per_query_source"
        )
    promoted_source_kind = promoted_per_query_source.get("kind")
    if promoted_source_kind not in PROMOTABLE_PER_QUERY_SOURCE_KINDS:
        raise ValueError(
            f"promoted_development system {promoted_id} has unsupported "
            f"per_query_source kind: {promoted_source_kind}"
        )
    promoted_pointer = promoted_per_query_source.get("pointer")
    if not isinstance(promoted_pointer, str) or not promoted_pointer.startswith("/"):
        raise ValueError(
            f"promoted_development system {promoted_id} requires a per_query_source "
            "JSON pointer"
        )
    promoted_aliases = promoted.get("aliases", [])
    if not isinstance(promoted_aliases, list):
        promoted_aliases = []
    promoted_identifiers = {
        value
        for value in (
            promoted.get("system_id"),
            promoted.get("source_system_id"),
            *promoted_aliases,
        )
        if isinstance(value, str) and value
    }
    aggregate_source = promoted.get("aggregate_source")
    if not isinstance(aggregate_source, Mapping):
        raise ValueError(
            f"promoted_development system {promoted_id} requires aggregate_source "
            "identity custody"
        )
    aggregate_pointer = aggregate_source.get("pointer")
    if aggregate_source.get("kind") != "json_pointer" or not _pointer_identifies(
        aggregate_pointer, promoted_identifiers
    ):
        raise ValueError(
            f"promoted_development system {promoted_id} aggregate_source pointer "
            "does not establish aggregate_source identity custody"
        )
    if not _pointer_identifies(promoted_pointer, promoted_identifiers):
        raise ValueError(
            f"promoted_development system {promoted_id} per_query_source pointer "
            "does not establish per_query_source identity custody"
        )
    if promoted.get("ranking_coverage") != "per_query_metrics_only":
        raise ValueError(
            f"promoted_development system {promoted_id} requires compatible non-none "
            "ranking_coverage=per_query_metrics_only"
        )
    if promoted.get("per_query_blocker") is not None:
        raise ValueError(
            f"promoted_development system {promoted_id} must not have a "
            "per_query_blocker"
        )
    blocking_caveats = set(promoted.get("caveat_ids", [])) & (
        PROMOTION_BLOCKING_CAVEAT_IDS
    )
    if blocking_caveats:
        raise ValueError(
            f"promoted_development system {promoted_id} has promotion-blocking "
            f"caveats: {sorted(blocking_caveats)}"
        )
    _validate_evidence_contract(promoted, system_id=promoted_id)

    unknown_sources = referenced_sources - set(sources)
    if unknown_sources:
        raise ValueError(f"unknown source IDs: {sorted(unknown_sources)}")

    source_values: dict[str, Any] = {}
    for source_id, raw_source in sources.items():
        source = _require_mapping(raw_source, f"source {source_id}")
        path_value = source.get("path")
        if not isinstance(path_value, str):
            raise ValueError(f"source path must be repository-relative: {source_id}")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source path must be repository-relative: {source_id}")
        expected = source.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"invalid source SHA-256: {source_id}")
        full_path = root / relative
        if not full_path.is_file():
            raise ValueError(f"missing controlling source: {relative}")
        if _sha256(full_path) != expected:
            raise ValueError(f"source digest changed: {relative}")
        if full_path.suffix == ".json":
            source_values[source_id] = json.loads(full_path.read_text(encoding="utf-8"))

    for system_id, source_field, source_ref, identifiers, promoted_status in (
        pointer_sources
    ):
        source_id = source_ref["source_id"]
        identity_pointer = source_ref.get("identity_pointer")
        exact_contract = EXACT_SOURCE_POINTER_CONTRACTS.get(
            (system_id, source_field)
        )
        if exact_contract is None:
            if identity_pointer is None and _source_reference_identifies(
                source_ref, identifiers
            ):
                continue
        elif (
            source_id,
            source_ref.get("pointer"),
            identity_pointer,
        ) != exact_contract:
            exact_contract = None
        identity_value: Any = None
        if exact_contract is not None and source_id in source_values:
            try:
                identity_value = resolve_json_pointer(
                    source_values[source_id], identity_pointer
                )
            except (IndexError, KeyError, TypeError, ValueError):
                identity_value = None
        if (
            exact_contract is not None
            and isinstance(identity_value, str)
            and identity_value in identifiers
        ):
            continue
        promotion_prefix = (
            "promoted_development system " if promoted_status else "system "
        )
        raise ValueError(
            f"{promotion_prefix}{system_id} {source_field} identity does not match "
            "its canonical, source, or alias namespace"
        )

    for system_id, source_ref, identifiers in literal_snapshots:
        _validate_literal_snapshot(
            root,
            sources,
            source_ref,
            system_id=system_id,
            identifiers=identifiers,
        )

    for raw_experiment in experiments:
        experiment = _require_mapping(raw_experiment, "experiment")
        _validate_grid_contracts(experiment, sources, source_values)


def _pointer_identifies(pointer: Any, identifiers: set[str]) -> bool:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return False
    ordered_tokens = [
        raw_token.replace("~1", "/").replace("~0", "~")
        for raw_token in pointer[1:].split("/")
    ]
    tokens = set(ordered_tokens)
    pointer_paths = {"/".join(ordered_tokens), ".".join(ordered_tokens)}
    return bool(tokens & identifiers or pointer_paths & identifiers)


def _source_reference_identifies(
    source_ref: Mapping[str, Any], identifiers: set[str]
) -> bool:
    key = source_ref.get("key")
    return _pointer_identifies(source_ref.get("pointer"), identifiers) or (
        isinstance(key, str) and key in identifiers
    )


def _markdown_cell_text(value: str) -> str:
    return value.strip().replace("**", "").replace("__", "").strip("` ")


def _markdown_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [_markdown_cell_text(cell) for cell in stripped[1:-1].split("|")]


def _markdown_key(value: str) -> str:
    return "_".join(
        re.findall(r"[a-z0-9]+", _markdown_cell_text(value).casefold().replace("@", " at "))
    )


def _is_markdown_divider(cells: list[str]) -> bool:
    return bool(cells) and all(
        bool(cell) and set(cell) <= {"-", ":"} and "-" in cell for cell in cells
    )


def _literal_value_matches(expected: Any, cell: str) -> bool:
    actual = _markdown_cell_text(cell)
    if isinstance(expected, bool):
        return False
    if isinstance(expected, int | float):
        try:
            actual_number = float(actual.replace(",", ""))
        except ValueError:
            return False
        return math.isclose(
            float(expected), actual_number, rel_tol=0.0, abs_tol=1e-12
        )
    if not isinstance(expected, str):
        return False
    return expected == actual


def _parse_declared_markdown_table(
    text: str,
    *,
    section: str,
    table: str,
    system_id: str,
) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    section_line = f"## {section}"
    try:
        section_start = next(
            index for index, line in enumerate(lines) if line.strip() == section_line
        )
    except StopIteration as error:
        raise ValueError(
            f"literal_snapshot for {system_id} is missing section {section!r}"
        ) from error
    section_end = next(
        (
            index
            for index in range(section_start + 1, len(lines))
            if lines[index].startswith("# ") or lines[index].startswith("## ")
        ),
        len(lines),
    )
    expected_header = [_markdown_cell_text(cell) for cell in table.split("|")]
    header_index: int | None = None
    for index in range(section_start + 1, section_end):
        cells = _markdown_cells(lines[index])
        if cells == expected_header:
            header_index = index
            break
    if header_index is None:
        raise ValueError(
            f"literal_snapshot for {system_id} is missing table {table!r}"
        )
    divider_index = header_index + 1
    if divider_index >= section_end:
        raise ValueError(f"literal_snapshot for {system_id} has no table divider")
    divider = _markdown_cells(lines[divider_index])
    if divider is None or not _is_markdown_divider(divider):
        raise ValueError(f"literal_snapshot for {system_id} has invalid table divider")
    rows: list[list[str]] = []
    for index in range(divider_index + 1, section_end):
        cells = _markdown_cells(lines[index])
        if cells is None:
            break
        if len(cells) != len(expected_header):
            raise ValueError(f"literal_snapshot for {system_id} has malformed row")
        rows.append(cells)
    return expected_header, rows


def _validate_literal_snapshot(
    root: Path,
    sources: Mapping[str, Any],
    source_ref: Mapping[str, Any],
    *,
    system_id: str,
    identifiers: set[str],
) -> None:
    source_id = source_ref.get("source_id")
    section = source_ref.get("section")
    table = source_ref.get("table")
    row_name = source_ref.get("row")
    metrics = source_ref.get("metrics")
    if not all(
        isinstance(value, str) and value
        for value in (source_id, section, table, row_name)
    ):
        raise ValueError(
            f"literal_snapshot for {system_id} requires source_id, section, table, "
            "and row strings"
        )
    metrics = _require_mapping(metrics, f"literal_snapshot metrics for {system_id}")
    source_path, _source_sha256 = _source_details(sources, source_id)
    path = root / source_path
    if path.suffix.casefold() not in {".md", ".markdown"}:
        raise ValueError(f"literal_snapshot for {system_id} requires Markdown source")
    headers, rows = _parse_declared_markdown_table(
        path.read_text(encoding="utf-8"),
        section=section,
        table=table,
        system_id=system_id,
    )
    header_keys = [_markdown_key(header) for header in headers]
    row_match = next((row for row in rows if row[0] == row_name), None)
    column_index = next(
        (index for index, header in enumerate(headers) if header == row_name), None
    )
    if row_match is None and column_index is None:
        raise ValueError(
            f"literal_snapshot for {system_id} is missing row or column {row_name!r}"
        )
    if row_match is not None:
        identity_matches = row_name in identifiers
        if not identity_matches:
            identity_tokens = source_ref.get("identity_tokens")
            if not isinstance(identity_tokens, list) or not identity_tokens or not all(
                isinstance(token, str) and token for token in identity_tokens
            ):
                raise ValueError(
                    f"literal_snapshot for {system_id} identity requires explicit "
                    "identity_tokens for a shared Markdown row"
                )
            row_text = " ".join(row_match).casefold()
            identifier_texts = [identifier.casefold() for identifier in identifiers]
            identity_matches = all(
                token.casefold() in row_text
                and any(token.casefold() in identifier for identifier in identifier_texts)
                for token in identity_tokens
            )
    else:
        identity_matches = row_name in identifiers
    if not identity_matches:
        raise ValueError(
            f"literal_snapshot for {system_id} identity does not match its "
            "declared Markdown row or column"
        )

    header_aliases = {"zero_ndcg_queries": "zero_ndcg"}
    column_metric_rows = {
        "ndcg_at_10": ("ndcg_at_10", _markdown_key(row_name)),
        "mrr": ("mrr", _markdown_key(row_name)),
        "zero_ndcg_queries": ("zero_ndcg_queries", _markdown_key(row_name)),
        "ndcg_delta_vs_dense": ("ndcg_at_10", "candidate_minus_baseline"),
        "mrr_delta_vs_dense": ("mrr", "candidate_minus_baseline"),
    }
    if "metric_templates" in source_ref:
        metric_templates = source_ref["metric_templates"]
        metric_name = (
            next(iter(metric_templates), "metric_templates")
            if isinstance(metric_templates, Mapping)
            else "metric_templates"
        )
        raise ValueError(
            f"literal_snapshot for {system_id} metric {metric_name!r} forbids "
            "registry-controlled metric_templates"
        )
    for metric_name, expected in metrics.items():
        matched = False
        if row_match is not None:
            header_key = header_aliases.get(metric_name, metric_name)
            if header_key in header_keys:
                matched = _literal_value_matches(
                    expected, row_match[header_keys.index(header_key)]
                )
            elif isinstance(expected, str):
                selector = LITERAL_METRIC_SELECTORS.get((system_id, metric_name))
                declared_location = (source_id, section, table, row_name)
                if selector is None or declared_location != selector[:4]:
                    selector = None
                rendered = (
                    selector[4].replace("{value}", expected)
                    if selector is not None
                    else None
                )
                components = {
                    component.strip()
                    for cell in row_match
                    for component in cell.split(";")
                }
                matched = rendered is not None and rendered in components
        else:
            coordinates = column_metric_rows.get(metric_name)
            if coordinates is not None:
                metric_row_key, target_column_key = coordinates
                target_row = next(
                    (row for row in rows if _markdown_key(row[0]) == metric_row_key),
                    None,
                )
                if target_row is not None and target_column_key in header_keys:
                    matched = _literal_value_matches(
                        expected, target_row[header_keys.index(target_column_key)]
                    )
        if not matched:
            raise ValueError(
                f"literal_snapshot for {system_id} metric {metric_name!r} does not "
                "match its declared Markdown table"
            )


def density_bucket(positive_count: int) -> str:
    if positive_count <= 5:
        return "at_most_5_positives"
    if positive_count <= 20:
        return "6_to_20_positives"
    return "more_than_20_positives"


def _source_details(sources: Mapping[str, Any], source_id: str) -> tuple[str, str]:
    source = _require_mapping(sources.get(source_id), f"source {source_id}")
    path = source.get("path")
    sha256 = source.get("sha256")
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise ValueError(f"invalid source metadata: {source_id}")
    return path, sha256


def _load_json_source(
    root: Path,
    sources: Mapping[str, Any],
    source_id: str,
    cache: dict[str, Any],
) -> Any:
    if source_id not in cache:
        source_path, _source_sha256 = _source_details(sources, source_id)
        path = root / source_path
        if path.suffix != ".json":
            raise ValueError(f"source is not JSON: {source_path}")
        cache[source_id] = json.loads(path.read_text(encoding="utf-8"))
    return cache[source_id]


def _resolve_source_reference(
    root: Path,
    sources: Mapping[str, Any],
    source_ref: Mapping[str, Any],
    cache: dict[str, Any],
) -> Mapping[str, Any]:
    kind = source_ref.get("kind")
    source_id = source_ref.get("source_id")
    if not isinstance(source_id, str):
        raise ValueError("source reference requires source_id")
    if kind == "literal_snapshot":
        return _require_mapping(source_ref.get("metrics"), "literal metrics")
    if kind not in {"json_pointer", "grid_member"}:
        raise ValueError(f"unsupported aggregate source kind: {kind}")
    pointer = source_ref.get("pointer")
    if not isinstance(pointer, str):
        raise ValueError(f"aggregate source pointer must be a string: {source_id}")
    payload = resolve_json_pointer(
        _load_json_source(root, sources, source_id, cache), pointer
    )
    payload = _require_mapping(payload, f"aggregate source {source_id}{pointer}")
    if kind == "grid_member" and payload.get("key") != source_ref.get("key"):
        raise ValueError(f"grid member key drifted: {source_id}{pointer}")
    return payload


def _canonical_payload_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("aggregate payload must be finite JSON data") from error


def _quality_metric(
    payload: Mapping[str, Any],
    *field_names: str,
) -> float | None:
    value: Any = None
    for field_name in field_names:
        if field_name in payload:
            value = payload[field_name]
            break
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"quality metric {field_names[0]} must be numeric")
    metric = float(value)
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise ValueError(f"quality metric {field_names[0]} must be in [0, 1]")
    return metric


def _zero_ndcg_queries(payload: Mapping[str, Any]) -> int | None:
    plural_value = payload.get("zero_ndcg_queries")
    count_value = payload.get("zero_ndcg_count")
    if (
        plural_value is not None
        and count_value is not None
        and plural_value != count_value
    ):
        raise ValueError("zero nDCG count fields disagree")
    value = plural_value if plural_value is not None else count_value
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("zero nDCG count must be a nonnegative integer")
    return value


def _validate_per_query_metric(
    value: Any,
    *,
    field_name: str,
    system_id: str,
    query_id: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"per-query {field_name} must be numeric for {system_id}/{query_id}"
        )
    metric = float(value)
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise ValueError(
            f"per-query {field_name} must be in [0, 1] for {system_id}/{query_id}"
        )
    return metric


def _load_per_query_metrics(
    root: Path,
    sources: Mapping[str, Any],
    source_ref: Mapping[str, Any],
    cache: dict[str, Any],
    *,
    system_id: str,
) -> tuple[dict[str, tuple[float, float]], str, str, str]:
    source_id = source_ref.get("source_id")
    if not isinstance(source_id, str):
        raise ValueError(f"per-query source_id required for {system_id}")
    source_path, source_sha256 = _source_details(sources, source_id)
    kind = source_ref.get("kind")
    rows: dict[str, tuple[float, float]] = {}
    if kind == "issue18_per_query_map":
        pointer = source_ref.get("pointer")
        if not isinstance(pointer, str):
            raise ValueError(f"per-query pointer must be a string for {system_id}")
        raw_rows = resolve_json_pointer(
            _load_json_source(root, sources, source_id, cache), pointer
        )
        raw_rows = _require_mapping(raw_rows, f"per-query map for {system_id}")
        for query_id, raw_metrics in raw_rows.items():
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"invalid per-query query ID for {system_id}")
            metrics = _require_mapping(
                raw_metrics, f"per-query metrics for {system_id}/{query_id}"
            )
            rows[query_id] = (
                _validate_per_query_metric(
                    metrics.get("ndcg_at_k", metrics.get("ndcg_at_10")),
                    field_name="ndcg_at_10",
                    system_id=system_id,
                    query_id=query_id,
                ),
                _validate_per_query_metric(
                    metrics.get("reciprocal_rank", metrics.get("mrr")),
                    field_name="reciprocal_rank",
                    system_id=system_id,
                    query_id=query_id,
                ),
            )
    elif kind in {"dense_per_query_tsv", "wide_per_query_tsv"}:
        query_column = source_ref.get("query_id_column")
        ndcg_column = source_ref.get("ndcg_column")
        rr_column = source_ref.get("reciprocal_rank_column")
        if not all(
            isinstance(column, str) for column in (query_column, ndcg_column, rr_column)
        ):
            raise ValueError(f"invalid TSV per-query columns for {system_id}")
        with (root / source_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required_columns = {query_column, ndcg_column, rr_column}
            if reader.fieldnames is None or not required_columns <= set(
                reader.fieldnames
            ):
                raise ValueError(f"missing TSV per-query columns for {system_id}")
            for line_number, raw_row in enumerate(reader, start=2):
                query_id = raw_row[query_column]
                if not query_id:
                    raise ValueError(
                        f"empty per-query query ID for {system_id} at line {line_number}"
                    )
                if query_id in rows:
                    raise ValueError(
                        f"duplicate per-query query ID for {system_id}: {query_id}"
                    )
                try:
                    ndcg_value = float(raw_row[ndcg_column])
                    rr_value = float(raw_row[rr_column])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid TSV per-query metric for {system_id}/{query_id}"
                    ) from error
                rows[query_id] = (
                    _validate_per_query_metric(
                        ndcg_value,
                        field_name="ndcg_at_10",
                        system_id=system_id,
                        query_id=query_id,
                    ),
                    _validate_per_query_metric(
                        rr_value,
                        field_name="reciprocal_rank",
                        system_id=system_id,
                        query_id=query_id,
                    ),
                )
    else:
        raise ValueError(f"unsupported per-query source kind for {system_id}: {kind}")
    return rows, "stored", source_path, source_sha256


def _make_slice(
    system_id: str,
    *,
    slice_type: str,
    slice_key: str,
    rows: list[PerQueryRecord],
    grade_1_memberships: int,
    grade_2_memberships: int,
    overlapping: bool,
    membership_policy: str,
    query_source_path: str,
    query_source_sha256: str,
    qrel_source_path: str,
    qrel_source_sha256: str,
) -> SliceRecord:
    query_count = len(rows)
    if query_count == 0:
        raise ValueError(f"slice {slice_type}/{slice_key} has no queries")
    metric_sources = {(row.source_path, row.source_sha256) for row in rows}
    if len(metric_sources) != 1:
        raise ValueError(f"slice {slice_type}/{slice_key} mixes metric sources")
    metric_source_path, metric_source_sha256 = next(iter(metric_sources))
    zero_ndcg_count = sum(row.zero_ndcg for row in rows)
    return SliceRecord(
        system_id=system_id,
        slice_type=slice_type,
        slice_key=slice_key,
        query_count=query_count,
        ndcg_at_10=sum(row.ndcg_at_10 for row in rows) / query_count,
        mrr=sum(row.reciprocal_rank for row in rows) / query_count,
        zero_ndcg_count=zero_ndcg_count,
        zero_ndcg_rate=zero_ndcg_count / query_count,
        grade_1_memberships=grade_1_memberships,
        grade_2_memberships=grade_2_memberships,
        overlapping=overlapping,
        membership_policy=membership_policy,
        metric_source_path=metric_source_path,
        metric_source_sha256=metric_source_sha256,
        query_source_path=query_source_path,
        query_source_sha256=query_source_sha256,
        qrel_source_path=qrel_source_path,
        qrel_source_sha256=qrel_source_sha256,
    )


def _build_slices(
    system_id: str,
    rows: list[PerQueryRecord],
    *,
    query_source_path: str,
    query_source_sha256: str,
    qrel_source_path: str,
    qrel_source_sha256: str,
) -> tuple[list[SliceRecord], list[SliceRecord]]:
    topic_members: dict[str, list[tuple[PerQueryRecord, int]]] = defaultdict(list)
    density_members: dict[str, list[PerQueryRecord]] = defaultdict(list)
    for row in rows:
        for candidate_id, grade in row.relevant_candidates:
            topic_members[candidate_id].append((row, grade))
        density_members[row.density_bucket].append(row)

    topic_slices: list[SliceRecord] = []
    for candidate_id in sorted(topic_members):
        members = topic_members[candidate_id]
        topic_slices.append(
            _make_slice(
                system_id,
                slice_type="relevant_candidate_slice",
                slice_key=candidate_id,
                rows=[row for row, _grade in members],
                grade_1_memberships=sum(grade == 1 for _row, grade in members),
                grade_2_memberships=sum(grade == 2 for _row, grade in members),
                overlapping=True,
                membership_policy="qrel_grade_gt_zero",
                query_source_path=query_source_path,
                query_source_sha256=query_source_sha256,
                qrel_source_path=qrel_source_path,
                qrel_source_sha256=qrel_source_sha256,
            )
        )

    density_slices: list[SliceRecord] = []
    for bucket in (
        "at_most_5_positives",
        "6_to_20_positives",
        "more_than_20_positives",
    ):
        members = density_members.get(bucket, [])
        if not members:
            continue
        density_slices.append(
            _make_slice(
                system_id,
                slice_type="positive_density_slice",
                slice_key=bucket,
                rows=members,
                grade_1_memberships=sum(row.grade_1_count for row in members),
                grade_2_memberships=sum(row.grade_2_count for row in members),
                overlapping=False,
                membership_policy="positive_candidate_count",
                query_source_path=query_source_path,
                query_source_sha256=query_source_sha256,
                qrel_source_path=qrel_source_path,
                qrel_source_sha256=qrel_source_sha256,
            )
        )
    return topic_slices, density_slices


def _verify_dense_alias_vector(
    root: Path,
    sources: Mapping[str, Any],
    cache: dict[str, Any],
    dense_rows: Mapping[str, tuple[float, float]] | None,
) -> float | None:
    if "issue18_evaluation" not in sources:
        return None
    issue18 = _load_json_source(root, sources, "issue18_evaluation", cache)
    try:
        alias = resolve_json_pointer(issue18, "/controls/query_knn/per_query")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "declared issue18_evaluation source lacks /controls/query_knn/per_query"
        ) from error
    alias = _require_mapping(alias, "issue-18 dense alias per-query map")
    if dense_rows is None:
        raise ValueError(
            "declared issue18_evaluation alias requires a canonical "
            "dense_nested per-query vector"
        )
    if set(alias) != set(dense_rows):
        raise ValueError("dense alias vectors have different query IDs")
    maximum_delta = 0.0
    for query_id, (ndcg, reciprocal_rank) in dense_rows.items():
        metrics = _require_mapping(alias[query_id], f"dense alias metrics/{query_id}")
        alias_ndcg = _validate_per_query_metric(
            metrics.get("ndcg_at_k"),
            field_name="ndcg_at_10",
            system_id="query_knn",
            query_id=query_id,
        )
        alias_rr = _validate_per_query_metric(
            metrics.get("reciprocal_rank"),
            field_name="reciprocal_rank",
            system_id="query_knn",
            query_id=query_id,
        )
        maximum_delta = max(
            maximum_delta,
            abs(ndcg - alias_ndcg),
            abs(reciprocal_rank - alias_rr),
        )
    if maximum_delta > 1e-10:
        raise ValueError("dense alias vectors disagree beyond 1e-10")
    return maximum_delta


def build_ledger(
    root: Path,
    registry_path: Path = Path("experiments/ledger.json"),
) -> LedgerBuild:
    root = Path(root)
    registry = load_registry(root, registry_path)
    sources = _require_mapping(registry["sources"], "registry sources")
    experiments = registry["experiments"]
    if not isinstance(experiments, list):
        raise ValueError("registry experiments must be a list")

    query_source_path, query_source_sha256 = _source_details(
        sources, "development_queries"
    )
    qrel_source_path, qrel_source_sha256 = _source_details(sources, "development_qrels")
    development = load_development_data(
        root / query_source_path,
        root / qrel_source_path,
    )
    if len(development.models) != 150:
        raise ValueError(
            "development qrels must contain exactly 150 unique candidates per query"
        )
    for query_id, candidate_grades in development.qrels.items():
        if len(candidate_grades) != 150:
            raise ValueError(
                f"development qrels must contain exactly 150 candidates for {query_id}"
            )
        invalid_grades = {
            grade for grade in candidate_grades.values() if grade not in {0.0, 1.0, 2.0}
        }
        if invalid_grades:
            raise ValueError(
                f"qrel grades must be in {{0, 1, 2}} for {query_id}: "
                f"{sorted(invalid_grades)}"
            )

    query_summaries: dict[
        str, tuple[tuple[tuple[str, int], ...], int, int, int, str]
    ] = {}
    for query_id in sorted(development.queries):
        relevant = tuple(
            (candidate_id, int(development.qrels[query_id][candidate_id]))
            for candidate_id in development.models
            if development.qrels[query_id][candidate_id] > 0
        )
        grade_1_count = sum(grade == 1 for _candidate_id, grade in relevant)
        grade_2_count = sum(grade == 2 for _candidate_id, grade in relevant)
        positive_count = len(relevant)
        query_summaries[query_id] = (
            relevant,
            grade_1_count,
            grade_2_count,
            positive_count,
            density_bucket(positive_count),
        )

    json_cache: dict[str, Any] = {}
    system_records: list[SystemRecord] = []
    per_query_records: list[PerQueryRecord] = []
    topic_slices: list[SliceRecord] = []
    density_slices: list[SliceRecord] = []
    dense_rows: dict[str, tuple[float, float]] | None = None

    for raw_experiment in experiments:
        experiment = _require_mapping(raw_experiment, "experiment")
        experiment_id = str(experiment["experiment_id"])
        experiment_family = str(experiment["experiment_family"])
        raw_systems = experiment.get("systems")
        if not isinstance(raw_systems, list):
            raise ValueError(f"systems must be a list for {experiment_id}")
        for raw_system in raw_systems:
            system = _require_mapping(raw_system, "system")
            system_id = str(system["system_id"])
            source_system_id = str(system["source_system_id"])
            aliases = system.get("aliases")
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) for alias in aliases
            ):
                raise ValueError(f"aliases must be a list of strings for {system_id}")
            method_summary = system.get("method_summary")
            if not isinstance(method_summary, str) or not method_summary.strip():
                raise ValueError(
                    f"method_summary must be a nonempty string for {system_id}"
                )
            native_scores_available = system.get("native_scores_available")
            if not isinstance(native_scores_available, bool):
                raise ValueError(
                    f"native_scores_available must be boolean for {system_id}"
                )
            qrel_usage = system.get("qrel_usage")
            if not isinstance(qrel_usage, str):
                raise ValueError(f"qrel_usage must be a string for {system_id}")
            run_classification = system.get("run_classification")
            if not isinstance(run_classification, str):
                raise ValueError(f"run_classification must be a string for {system_id}")
            aggregate_source = _require_mapping(
                system.get("aggregate_source"),
                f"aggregate source for {system_id}",
            )
            aggregate_source_json = _canonical_payload_json(aggregate_source)
            aggregate_source_id = aggregate_source.get("source_id")
            if not isinstance(aggregate_source_id, str):
                raise ValueError(f"aggregate source_id required for {system_id}")
            source_path, source_sha256 = _source_details(sources, aggregate_source_id)
            aggregate = _resolve_source_reference(
                root, sources, aggregate_source, json_cache
            )
            evidence_status = str(system["evidence_status"])
            if evidence_status in {"invalid", "unscored"}:
                quality_fields = sorted(QUALITY_METRIC_FIELDS & set(aggregate))
                if quality_fields:
                    raise ValueError(
                        f"{evidence_status} evidence for {system_id} forbids quality "
                        f"metrics: {quality_fields}"
                    )
            aggregate_payload_json = _canonical_payload_json(aggregate)
            reported_ndcg = _quality_metric(aggregate, "ndcg_at_10", "ndcg_at_k")
            reported_mrr = _quality_metric(aggregate, "mrr", "reciprocal_rank")
            reported_zero = _zero_ndcg_queries(aggregate)
            recomputed_ndcg: float | None = None
            recomputed_mrr: float | None = None
            recomputed_zero: int | None = None

            system_rows: list[PerQueryRecord] = []
            per_query_source = system.get("per_query_source")
            per_query_source_json: str | None = None
            if per_query_source is not None:
                per_query_source = _require_mapping(
                    per_query_source, f"per-query source for {system_id}"
                )
                per_query_source_json = _canonical_payload_json(per_query_source)
                metrics, provenance, metric_path, metric_sha256 = (
                    _load_per_query_metrics(
                        root,
                        sources,
                        per_query_source,
                        json_cache,
                        system_id=system_id,
                    )
                )
                evaluation_scope = str(system["evaluation_scope"])
                if evaluation_scope == "dev_322x150" and len(metrics) != 322:
                    raise ValueError(
                        f"full-scope per-query vector for {system_id} must have "
                        "exactly 322 rows"
                    )
                declared_query_count = system.get("query_count")
                if (
                    isinstance(declared_query_count, int)
                    and len(metrics) != declared_query_count
                ):
                    raise ValueError(
                        f"per-query row count differs from registry for {system_id}"
                    )
                expected_query_ids = set(development.queries)
                if set(metrics) != expected_query_ids:
                    missing = sorted(expected_query_ids - set(metrics))
                    extra = sorted(set(metrics) - expected_query_ids)
                    raise ValueError(
                        f"per-query query IDs differ from development data for "
                        f"{system_id}: missing={missing!r}, extra={extra!r}"
                    )

                recomputed_ndcg = sum(row[0] for row in metrics.values()) / len(metrics)
                recomputed_mrr = sum(row[1] for row in metrics.values()) / len(metrics)
                recomputed_zero = sum(row[0] == 0.0 for row in metrics.values())
                for metric_name, recomputed, reported in (
                    ("ndcg_at_10", recomputed_ndcg, reported_ndcg),
                    ("mrr", recomputed_mrr, reported_mrr),
                ):
                    if reported is None or abs(recomputed - reported) > 1e-12:
                        raise ValueError(
                            f"recomputed macro metric {metric_name} differs from "
                            f"receipt for {system_id}"
                        )
                if reported_zero is not None and recomputed_zero != reported_zero:
                    raise ValueError(
                        f"recomputed zero-nDCG count differs from receipt for {system_id}"
                    )

                for query_id in sorted(metrics):
                    ndcg_at_10, reciprocal_rank = metrics[query_id]
                    (
                        relevant_candidates,
                        grade_1_count,
                        grade_2_count,
                        positive_count,
                        bucket,
                    ) = query_summaries[query_id]
                    system_rows.append(
                        PerQueryRecord(
                            system_id=system_id,
                            source_system_id=source_system_id,
                            experiment_family=experiment_family,
                            evaluation_scope=evaluation_scope,
                            query_id=query_id,
                            query_text=development.queries[query_id],
                            ndcg_at_10=ndcg_at_10,
                            reciprocal_rank=reciprocal_rank,
                            grade_1_count=grade_1_count,
                            grade_2_count=grade_2_count,
                            positive_count=positive_count,
                            relevant_candidates=relevant_candidates,
                            density_bucket=bucket,
                            zero_ndcg=ndcg_at_10 == 0.0,
                            metric_provenance=provenance,
                            source_path=metric_path,
                            source_sha256=metric_sha256,
                        )
                    )
                if system_id == "dense_nested":
                    dense_rows = metrics

                system_topic_slices, system_density_slices = _build_slices(
                    system_id,
                    system_rows,
                    query_source_path=query_source_path,
                    query_source_sha256=query_source_sha256,
                    qrel_source_path=qrel_source_path,
                    qrel_source_sha256=qrel_source_sha256,
                )
                topic_slices.extend(system_topic_slices)
                density_slices.extend(system_density_slices)
                per_query_records.extend(system_rows)

            blocker = system.get("per_query_blocker")
            if blocker is not None and not isinstance(blocker, str):
                raise ValueError(f"invalid per-query blocker for {system_id}")
            query_count = system.get("query_count")
            candidate_count = system.get("candidate_count")
            if query_count is not None and not isinstance(query_count, int):
                raise ValueError(
                    f"query_count must be an integer or null for {system_id}"
                )
            if candidate_count is not None and not isinstance(candidate_count, int):
                raise ValueError(
                    f"candidate_count must be an integer or null for {system_id}"
                )
            system_records.append(
                SystemRecord(
                    system_id=system_id,
                    source_system_id=source_system_id,
                    aliases=tuple(aliases),
                    experiment_id=experiment_id,
                    experiment_family=experiment_family,
                    method_summary=method_summary,
                    evaluation_scope=str(system["evaluation_scope"]),
                    evidence_status=str(system["evidence_status"]),
                    promotion_status=str(system["promotion_status"]),
                    query_count=query_count,
                    candidate_count=candidate_count,
                    ndcg_at_10=reported_ndcg,
                    mrr=reported_mrr,
                    zero_ndcg_queries=reported_zero,
                    recomputed_ndcg_at_10=recomputed_ndcg,
                    recomputed_mrr=recomputed_mrr,
                    recomputed_zero_ndcg_queries=recomputed_zero,
                    aggregate_payload_json=aggregate_payload_json,
                    aggregate_source_json=aggregate_source_json,
                    per_query_source_json=per_query_source_json,
                    per_query_available=bool(system_rows),
                    ranking_coverage=str(system["ranking_coverage"]),
                    native_scores_available=native_scores_available,
                    artifact_state=str(system["artifact_state"]),
                    qrel_usage=qrel_usage,
                    run_classification=run_classification,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    blocker=blocker,
                    caveat_ids=tuple(str(value) for value in system["caveat_ids"]),
                )
            )

    dense_alias_max_delta = _verify_dense_alias_vector(
        root, sources, json_cache, dense_rows
    )

    density_supports: dict[str, int] = defaultdict(int)
    for _query_id, summary in query_summaries.items():
        density_supports[summary[4]] += 1
    relevant_memberships = sum(summary[3] for summary in query_summaries.values())
    if len(development.queries) == 322:
        expected_density_supports = {
            "at_most_5_positives": 248,
            "6_to_20_positives": 30,
            "more_than_20_positives": 44,
        }
        if dict(density_supports) != expected_density_supports:
            raise ValueError(
                "real development density supports must be 248, 30, and 44"
            )
        if sum(density_supports.values()) != 322:
            raise ValueError("density buckets must cover all 322 queries exactly once")
        if relevant_memberships != 4378:
            raise ValueError("relevant-candidate qrel support must total 4,378")

    normalized_sources = tuple(
        {
            "source_id": source_id,
            "path": _source_details(sources, source_id)[0],
            "sha256": _source_details(sources, source_id)[1],
        }
        for source_id in sorted(sources)
    )
    raw_caveats = _require_mapping(registry["caveats"], "registry caveats")
    normalized_caveats: list[dict[str, str]] = []
    for caveat_id in sorted(raw_caveats):
        caveat = _require_mapping(raw_caveats[caveat_id], f"caveat {caveat_id}")
        if not all(isinstance(value, str) for value in caveat.values()):
            raise ValueError(f"caveat values must be strings: {caveat_id}")
        normalized_caveats.append(
            {
                "caveat_id": caveat_id,
                **{str(key): value for key, value in caveat.items()},
            }
        )

    return LedgerBuild(
        systems=tuple(system_records),
        per_query=tuple(per_query_records),
        topic_slices=tuple(topic_slices),
        density_slices=tuple(density_slices),
        sources=normalized_sources,
        caveats=tuple(normalized_caveats),
        validation={
            "system_count": len(system_records),
            "per_query_system_count": len({row.system_id for row in per_query_records}),
            "per_query_record_count": len(per_query_records),
            "query_count": len(development.queries),
            "candidate_count": len(development.models),
            "qrel_row_count": len(development.queries) * len(development.models),
            "relevant_candidate_memberships": relevant_memberships,
            "density_supports": dict(density_supports),
            "dense_alias_max_abs_delta": dense_alias_max_delta,
        },
    )


def _system_sort_key(system: SystemRecord) -> tuple[str, str, str]:
    return (
        system.evaluation_scope,
        system.experiment_family,
        system.system_id,
    )


def _per_query_sort_key(row: PerQueryRecord) -> tuple[str, str]:
    return row.system_id, row.query_id


def _slice_sort_key(row: SliceRecord) -> tuple[str, str, str]:
    return row.system_id, row.slice_type, row.slice_key


def _slice_aggregation_warning(row: SliceRecord) -> str | None:
    return TOPIC_SLICE_AGGREGATION_WARNING if row.overlapping else None


def _validate_repository_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"ledger output path must be repository-relative: {value}")
    return value


def _recomputed_provenance_by_system(
    build: LedgerBuild,
) -> dict[str, dict[str, str]]:
    grouped: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in build.per_query:
        grouped[row.system_id].add(
            (
                row.metric_provenance,
                _validate_repository_relative_path(row.source_path),
                row.source_sha256,
            )
        )

    provenance: dict[str, dict[str, str]] = {}
    for system_id, values in grouped.items():
        if len(values) != 1:
            raise ValueError(
                f"per-query records for {system_id} have mixed metric provenance"
            )
        metric_provenance, source_path, source_sha256 = next(iter(values))
        provenance[system_id] = {
            "derivation": "macro_from_per_query",
            "per_query_metric_provenance": metric_provenance,
            "source_path": source_path,
            "source_sha256": source_sha256,
        }
    return provenance


def _system_json(
    system: SystemRecord,
    recomputed_provenance: dict[str, str] | None,
) -> dict[str, object]:
    source_path = _validate_repository_relative_path(system.source_path)
    return {
        "system_id": system.system_id,
        "source_system_id": system.source_system_id,
        "aliases": list(system.aliases),
        "experiment_id": system.experiment_id,
        "experiment_family": system.experiment_family,
        "method_summary": system.method_summary,
        "evaluation_scope": system.evaluation_scope,
        "evidence_status": system.evidence_status,
        "promotion_status": system.promotion_status,
        "query_count": system.query_count,
        "candidate_count": system.candidate_count,
        "receipt_metrics": {
            "ndcg_at_10": system.ndcg_at_10,
            "mrr": system.mrr,
            "zero_ndcg_queries": system.zero_ndcg_queries,
        },
        "recomputed_metrics": (
            {
                "ndcg_at_10": system.recomputed_ndcg_at_10,
                "mrr": system.recomputed_mrr,
                "zero_ndcg_queries": system.recomputed_zero_ndcg_queries,
            }
            if recomputed_provenance is not None
            else None
        ),
        "aggregate_payload": json.loads(system.aggregate_payload_json),
        "aggregate_source": json.loads(system.aggregate_source_json),
        "per_query_source": (
            json.loads(system.per_query_source_json)
            if system.per_query_source_json is not None
            else None
        ),
        "receipt_provenance": {
            "source_path": source_path,
            "source_sha256": system.source_sha256,
        },
        "recomputed_provenance": recomputed_provenance,
        "per_query_available": system.per_query_available,
        "ranking_coverage": system.ranking_coverage,
        "native_scores_available": system.native_scores_available,
        "artifact_state": system.artifact_state,
        "qrel_usage": system.qrel_usage,
        "run_classification": system.run_classification,
        "blocker": system.blocker,
        "caveat_ids": list(system.caveat_ids),
    }


def _per_query_json(row: PerQueryRecord) -> dict[str, object]:
    return {
        "system_id": row.system_id,
        "source_system_id": row.source_system_id,
        "experiment_family": row.experiment_family,
        "evaluation_scope": row.evaluation_scope,
        "query_id": row.query_id,
        "query_text": row.query_text,
        "ndcg_at_10": row.ndcg_at_10,
        "reciprocal_rank": row.reciprocal_rank,
        "grade_1_count": row.grade_1_count,
        "grade_2_count": row.grade_2_count,
        "positive_count": row.positive_count,
        "relevant_candidates": [list(value) for value in row.relevant_candidates],
        "density_bucket": row.density_bucket,
        "zero_ndcg": row.zero_ndcg,
        "metric_provenance": row.metric_provenance,
        "source_path": _validate_repository_relative_path(row.source_path),
        "source_sha256": row.source_sha256,
    }


def _slice_json(row: SliceRecord) -> dict[str, object]:
    return {
        "system_id": row.system_id,
        "slice_type": row.slice_type,
        "slice_key": row.slice_key,
        "query_count": row.query_count,
        "ndcg_at_10": row.ndcg_at_10,
        "mrr": row.mrr,
        "zero_ndcg_count": row.zero_ndcg_count,
        "zero_ndcg_rate": row.zero_ndcg_rate,
        "grade_1_memberships": row.grade_1_memberships,
        "grade_2_memberships": row.grade_2_memberships,
        "overlapping": row.overlapping,
        "membership_policy": row.membership_policy,
        "aggregation_warning": _slice_aggregation_warning(row),
        "metric_source_path": _validate_repository_relative_path(
            row.metric_source_path
        ),
        "metric_source_sha256": row.metric_source_sha256,
        "query_source_path": _validate_repository_relative_path(row.query_source_path),
        "query_source_sha256": row.query_source_sha256,
        "qrel_source_path": _validate_repository_relative_path(row.qrel_source_path),
        "qrel_source_sha256": row.qrel_source_sha256,
    }


def render_ledger_json(build: LedgerBuild) -> str:
    """Render the complete deterministic machine-readable ledger."""

    recomputed_provenance = _recomputed_provenance_by_system(build)
    payload = {
        "version": 1,
        "systems": [
            _system_json(system, recomputed_provenance.get(system.system_id))
            for system in sorted(build.systems, key=_system_sort_key)
        ],
        "per_query": [
            _per_query_json(row)
            for row in sorted(build.per_query, key=_per_query_sort_key)
        ],
        "topic_slices": [
            _slice_json(row) for row in sorted(build.topic_slices, key=_slice_sort_key)
        ],
        "density_slices": [
            _slice_json(row)
            for row in sorted(build.density_slices, key=_slice_sort_key)
        ],
        "sources": [
            {
                **source,
                "path": _validate_repository_relative_path(source["path"]),
            }
            for source in sorted(build.sources, key=lambda row: row["source_id"])
        ],
        "caveats": sorted(build.caveats, key=lambda row: row["caveat_id"]),
        "validation": build.validation,
    }
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _tsv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _render_tsv(
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(
        {field: _tsv_value(row[field]) for field in fieldnames} for row in rows
    )
    return output.getvalue()


def render_systems_tsv(build: LedgerBuild) -> str:
    recomputed_provenance = _recomputed_provenance_by_system(build)
    fieldnames = (
        "system_id",
        "source_system_id",
        "aliases",
        "experiment_id",
        "experiment_family",
        "method_summary",
        "evaluation_scope",
        "evidence_status",
        "promotion_status",
        "query_count",
        "candidate_count",
        "ndcg_at_10",
        "mrr",
        "zero_ndcg_queries",
        "aggregate_payload_json",
        "aggregate_source_json",
        "per_query_source_json",
        "recomputed_ndcg_at_10",
        "recomputed_mrr",
        "recomputed_zero_ndcg_queries",
        "recomputed_metric_derivation",
        "per_query_metric_provenance",
        "recomputed_metric_source_path",
        "recomputed_metric_source_sha256",
        "per_query_available",
        "ranking_coverage",
        "native_scores_available",
        "artifact_state",
        "qrel_usage",
        "run_classification",
        "source_path",
        "source_sha256",
        "blocker",
        "caveat_ids",
    )
    rows: list[dict[str, object]] = []
    for system in sorted(build.systems, key=_system_sort_key):
        provenance = recomputed_provenance.get(system.system_id)
        rows.append(
            {
                "system_id": system.system_id,
                "source_system_id": system.source_system_id,
                "aliases": json.dumps(
                    system.aliases,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "experiment_id": system.experiment_id,
                "experiment_family": system.experiment_family,
                "method_summary": system.method_summary,
                "evaluation_scope": system.evaluation_scope,
                "evidence_status": system.evidence_status,
                "promotion_status": system.promotion_status,
                "query_count": system.query_count,
                "candidate_count": system.candidate_count,
                "ndcg_at_10": system.ndcg_at_10,
                "mrr": system.mrr,
                "zero_ndcg_queries": system.zero_ndcg_queries,
                "aggregate_payload_json": system.aggregate_payload_json,
                "aggregate_source_json": system.aggregate_source_json,
                "per_query_source_json": system.per_query_source_json,
                "recomputed_ndcg_at_10": system.recomputed_ndcg_at_10,
                "recomputed_mrr": system.recomputed_mrr,
                "recomputed_zero_ndcg_queries": (system.recomputed_zero_ndcg_queries),
                "recomputed_metric_derivation": (
                    provenance["derivation"] if provenance else None
                ),
                "per_query_metric_provenance": (
                    provenance["per_query_metric_provenance"] if provenance else None
                ),
                "recomputed_metric_source_path": (
                    provenance["source_path"] if provenance else None
                ),
                "recomputed_metric_source_sha256": (
                    provenance["source_sha256"] if provenance else None
                ),
                "per_query_available": system.per_query_available,
                "ranking_coverage": system.ranking_coverage,
                "native_scores_available": system.native_scores_available,
                "artifact_state": system.artifact_state,
                "qrel_usage": system.qrel_usage,
                "run_classification": system.run_classification,
                "source_path": _validate_repository_relative_path(system.source_path),
                "source_sha256": system.source_sha256,
                "blocker": system.blocker,
                "caveat_ids": json.dumps(
                    system.caveat_ids,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return _render_tsv(fieldnames, rows)


def render_per_query_tsv(build: LedgerBuild) -> str:
    fieldnames = (
        "system_id",
        "source_system_id",
        "experiment_family",
        "evaluation_scope",
        "query_id",
        "query_text",
        "ndcg_at_10",
        "reciprocal_rank",
        "grade_1_count",
        "grade_2_count",
        "positive_count",
        "relevant_candidates",
        "density_bucket",
        "zero_ndcg",
        "metric_provenance",
        "source_path",
        "source_sha256",
    )
    rows = [
        {
            "system_id": row.system_id,
            "source_system_id": row.source_system_id,
            "experiment_family": row.experiment_family,
            "evaluation_scope": row.evaluation_scope,
            "query_id": row.query_id,
            "query_text": row.query_text,
            "ndcg_at_10": row.ndcg_at_10,
            "reciprocal_rank": row.reciprocal_rank,
            "grade_1_count": row.grade_1_count,
            "grade_2_count": row.grade_2_count,
            "positive_count": row.positive_count,
            "relevant_candidates": json.dumps(
                row.relevant_candidates,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "density_bucket": row.density_bucket,
            "zero_ndcg": row.zero_ndcg,
            "metric_provenance": row.metric_provenance,
            "source_path": _validate_repository_relative_path(row.source_path),
            "source_sha256": row.source_sha256,
        }
        for row in sorted(build.per_query, key=_per_query_sort_key)
    ]
    return _render_tsv(fieldnames, rows)


def _render_slices_tsv(rows: tuple[SliceRecord, ...]) -> str:
    fieldnames = (
        "system_id",
        "slice_type",
        "slice_key",
        "query_count",
        "ndcg_at_10",
        "mrr",
        "zero_ndcg_count",
        "zero_ndcg_rate",
        "grade_1_memberships",
        "grade_2_memberships",
        "overlapping",
        "membership_policy",
        "aggregation_warning",
        "metric_source_path",
        "metric_source_sha256",
        "query_source_path",
        "query_source_sha256",
        "qrel_source_path",
        "qrel_source_sha256",
    )
    rendered_rows = [
        {
            "system_id": row.system_id,
            "slice_type": row.slice_type,
            "slice_key": row.slice_key,
            "query_count": row.query_count,
            "ndcg_at_10": row.ndcg_at_10,
            "mrr": row.mrr,
            "zero_ndcg_count": row.zero_ndcg_count,
            "zero_ndcg_rate": row.zero_ndcg_rate,
            "grade_1_memberships": row.grade_1_memberships,
            "grade_2_memberships": row.grade_2_memberships,
            "overlapping": row.overlapping,
            "membership_policy": row.membership_policy,
            "aggregation_warning": _slice_aggregation_warning(row),
            "metric_source_path": _validate_repository_relative_path(
                row.metric_source_path
            ),
            "metric_source_sha256": row.metric_source_sha256,
            "query_source_path": _validate_repository_relative_path(
                row.query_source_path
            ),
            "query_source_sha256": row.query_source_sha256,
            "qrel_source_path": _validate_repository_relative_path(
                row.qrel_source_path
            ),
            "qrel_source_sha256": row.qrel_source_sha256,
        }
        for row in sorted(rows, key=_slice_sort_key)
    ]
    return _render_tsv(fieldnames, rendered_rows)


def render_topic_slices_tsv(build: LedgerBuild) -> str:
    return _render_slices_tsv(build.topic_slices)


def render_density_slices_tsv(build: LedgerBuild) -> str:
    return _render_slices_tsv(build.density_slices)


def _markdown_cell(value: object) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _headline_metric(system: SystemRecord | None) -> str:
    if system is None:
        return "none recorded"
    return f"`{system.system_id}` (nDCG@10 {_markdown_metric(system.ndcg_at_10)})"


def _best_system(
    systems: list[SystemRecord],
) -> SystemRecord | None:
    scored = [system for system in systems if system.ndcg_at_10 is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: (row.ndcg_at_10 or 0.0, row.system_id))


def _markdown_system_table(systems: list[SystemRecord]) -> list[str]:
    lines = [
        "| System | Scope | Queries | Candidates | Family | Promotion | Evidence | nDCG@10 | MRR | Per-query | Artifact |",
        "|---|---|---:|---:|---|---|---|---:|---:|---|---|",
    ]
    if not systems:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — |")
        return lines
    for system in systems:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_markdown_cell(system.system_id)}`",
                    f"`{_markdown_cell(system.evaluation_scope)}`",
                    _markdown_cell(system.query_count),
                    _markdown_cell(system.candidate_count),
                    _markdown_cell(system.experiment_family),
                    f"`{_markdown_cell(system.promotion_status)}`",
                    f"`{_markdown_cell(system.evidence_status)}`",
                    _markdown_metric(system.ndcg_at_10),
                    _markdown_metric(system.mrr),
                    "yes" if system.per_query_available else "no",
                    f"`{_markdown_cell(system.artifact_state)}`",
                )
            )
            + " |"
        )
    return lines


def render_ledger_markdown(build: LedgerBuild) -> str:
    """Render the concise scope-safe human ledger index."""

    systems = list(build.systems)
    promoted = _best_system(
        [
            system
            for system in systems
            if system.evaluation_scope == "dev_322x150"
            and system.promotion_status == "promoted_development"
        ]
    )
    diagnostic = _best_system(
        [
            system
            for system in systems
            if system.evaluation_scope == "dev_322x150"
            and system.promotion_status == "diagnostic_only"
        ]
    )
    qrel_free_control = _best_system(
        [
            system
            for system in systems
            if system.evaluation_scope == "dev_322x150"
            and system.promotion_status == "stable_control"
            and system.qrel_usage == "evaluation_only"
        ]
    )
    per_query_system_count = len(
        {system.system_id for system in systems if system.per_query_available}
    )
    aggregate_only_count = sum(
        system.evidence_status == "aggregate_only" for system in systems
    )
    missing_artifact_systems = sorted(
        (
            system
            for system in systems
            if system.artifact_state in MISSING_ARTIFACT_STATES
        ),
        key=_system_sort_key,
    )

    lines = [
        "# TREC 2026 experiment ledger",
        "",
        f"- Promoted development result: {_headline_metric(promoted)}.",
        (
            "- Best full-development diagnostic: "
            f"{_headline_metric(diagnostic)} — `diagnostic_only`, not promoted."
        ),
        (f"- Best stable qrel-free control: {_headline_metric(qrel_free_control)}."),
        f"- Systems with complete per-query evidence: {per_query_system_count}.",
        f"- aggregate-only systems: {aggregate_only_count}.",
        (f"- Unresolved artifact blockers: {len(missing_artifact_systems)} systems."),
        (
            "- Artifact blocker count includes only `local_artifact_missing` and "
            "`construction_artifact_missing` states."
        ),
        "- All effectiveness scores are development-only evidence; the hidden test is not available.",
        "",
        "## Full development results",
        "",
        (
            "Only `dev_322x150` systems appear here. `diagnostic_only` and "
            "`aggregate_only` remain visible in the table."
        ),
        "",
    ]
    full_scope = [
        system for system in systems if system.evaluation_scope == "dev_322x150"
    ]
    full_scope.sort(
        key=lambda row: (
            row.ndcg_at_10 is None,
            -(row.ndcg_at_10 or 0.0),
            row.system_id,
        )
    )
    lines.extend(_markdown_system_table(full_scope))

    lines.extend(
        [
            "",
            "## Restricted pilots and invalid/unscored work",
            "",
            (
                "These rows are not numerically comparable with the complete "
                "322-query, 150-candidate development evaluation."
            ),
            "",
        ]
    )
    scope_sections = (
        (
            "restricted_12x20",
            "Restricted 12x20 (`restricted_12x20`)",
            "Qrel-assisted 12-query, 20-candidate diagnostics only.",
        ),
        (
            "restricted_6",
            "Restricted 6-query (`restricted_6`)",
            "Six-query feasibility evidence only; candidate counts may be unknown.",
        ),
        (
            "invalid_unscored",
            "Invalid/unscored (`invalid_unscored`)",
            "No valid numerical quality result is available for these systems.",
        ),
        (
            "constructed_not_evaluated",
            "Constructed, not evaluated (`constructed_not_evaluated`)",
            "A representation or profile exists without a complete ranking evaluation.",
        ),
    )
    for scope, heading, explanation in scope_sections:
        scoped_systems = [
            system for system in systems if system.evaluation_scope == scope
        ]
        scoped_systems.sort(key=_system_sort_key)
        lines.extend(["", f"### {heading}", "", explanation, ""])
        lines.extend(_markdown_system_table(scoped_systems))

    lines.extend(
        [
            "",
            "## Per-query evidence coverage",
            "",
            "| System | Evidence | Per-query | Blocker / custody caveat |",
            "|---|---|---|---|",
        ]
    )
    for system in sorted(systems, key=_system_sort_key):
        blocker = system.blocker
        if blocker is None and system.artifact_state in MISSING_ARTIFACT_STATES:
            blocker = f"Artifact custody: {system.artifact_state}."
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_markdown_cell(system.system_id)}`",
                    f"`{_markdown_cell(system.evidence_status)}`",
                    "yes" if system.per_query_available else "no",
                    _markdown_cell(blocker),
                )
            )
            + " |"
        )

    density_supports = build.validation.get("density_supports", {})
    density_text = (
        ", ".join(f"`{key}`={value}" for key, value in sorted(density_supports.items()))
        if isinstance(density_supports, Mapping)
        else "see the density TSV"
    )
    lines.extend(
        [
            "",
            "## Topic and density slices",
            "",
            (
                "Relevant-candidate slices use `qrel_grade_gt_zero`. They overlap, "
                "and candidate labels are analysis handles, not official topics. "
                "Do not sum or average the slice means as if they were disjoint."
            ),
            "",
            (
                f"The topic TSV contains {len(build.topic_slices)} system/slice rows; "
                f"the density TSV contains {len(build.density_slices)} rows."
            ),
            "",
            f"Disjoint density support: {density_text}.",
            "",
            "- [Per-query scores](trec_2026_experiment_per_query.tsv)",
            "- [Relevant-candidate slices](trec_2026_experiment_topic_slices.tsv)",
            "- [Positive-density slices](trec_2026_experiment_density_slices.tsv)",
            "",
            "## Program-wide caveats",
            "",
        ]
    )
    program_caveats = sorted(
        (caveat for caveat in build.caveats if caveat.get("scope") == "program"),
        key=lambda row: row["caveat_id"],
    )
    for caveat in program_caveats:
        lines.append(
            f"- **`{_markdown_cell(caveat['caveat_id'])}`:** "
            f"{_markdown_cell(caveat.get('summary'))}"
        )

    lines.extend(
        [
            "",
            "## Source receipts",
            "",
            "| Source | Repository path | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for source in sorted(build.sources, key=lambda row: row["source_id"]):
        source_path = _validate_repository_relative_path(source["path"])
        lines.append(
            f"| `{_markdown_cell(source['source_id'])}` | "
            f"[{_markdown_cell(source_path)}](../{source_path}) | "
            f"`{_markdown_cell(source['sha256'])}` |"
        )
    lines.extend(
        [
            "",
            "## Rebuild command",
            "",
            "```bash",
            "uv run python scripts/build_experiment_ledger.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _render_output_payloads(build: LedgerBuild) -> dict[str, str]:
    payloads = {
        "trec_2026_experiment_density_slices.tsv": render_density_slices_tsv(build),
        "trec_2026_experiment_ledger.json": render_ledger_json(build),
        "trec_2026_experiment_ledger.md": render_ledger_markdown(build),
        "trec_2026_experiment_per_query.tsv": render_per_query_tsv(build),
        "trec_2026_experiment_systems.tsv": render_systems_tsv(build),
        "trec_2026_experiment_topic_slices.tsv": render_topic_slices_tsv(build),
    }
    if tuple(sorted(payloads)) != LEDGER_OUTPUT_FILENAMES:
        raise AssertionError("ledger output filename contract drifted")
    return payloads


def write_ledger_outputs(
    build: LedgerBuild,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write all ledger views only after every renderer and stage write succeeds."""

    payloads = _render_output_payloads(build)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = tuple(output_dir / name for name in LEDGER_OUTPUT_FILENAMES)

    with tempfile.TemporaryDirectory(
        prefix=".experiment-ledger-",
        dir=output_dir.parent,
    ) as temporary:
        staging_dir = Path(temporary)
        staged_paths: dict[str, Path] = {}
        for name in LEDGER_OUTPUT_FILENAMES:
            staged = staging_dir / name
            staged.write_text(payloads[name], encoding="utf-8", newline="")
            staged_paths[name] = staged

        backup_dir = staging_dir / "previous"
        backup_dir.mkdir()
        installed: list[str] = []
        try:
            for name in LEDGER_OUTPUT_FILENAMES:
                destination = output_dir / name
                backup = backup_dir / name
                if destination.exists():
                    destination.replace(backup)
                try:
                    staged_paths[name].replace(destination)
                except Exception:
                    if backup.exists():
                        backup.replace(destination)
                    raise
                installed.append(name)
        except Exception:
            for name in reversed(installed):
                destination = output_dir / name
                destination.unlink(missing_ok=True)
                backup = backup_dir / name
                if backup.exists():
                    backup.replace(destination)
            raise
    return destinations


def assert_outputs_current(build: LedgerBuild, output_dir: Path) -> None:
    """Raise with every missing or stale output without changing the filesystem."""

    payloads = _render_output_payloads(build)
    output_dir = Path(output_dir)
    missing: list[str] = []
    stale: list[str] = []
    for name in LEDGER_OUTPUT_FILENAMES:
        path = output_dir / name
        if not path.is_file():
            missing.append(name)
        elif path.read_text(encoding="utf-8") != payloads[name]:
            stale.append(name)
    if missing or stale:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if stale:
            details.append(f"stale: {', '.join(stale)}")
        raise ValueError(
            "experiment ledger outputs are not current; " + "; ".join(details)
        )
