from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


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
ARTIFACT_STATES = frozenset(
    {
        "tracked_complete",
        "tracked_receipt_only",
        "tracked_report_snapshot",
        "local_artifact_missing",
        "construction_artifact_missing",
    }
)
QREL_USAGE_STATUSES = frozenset(
    {
        "training_and_evaluation",
        "evaluation_only",
        "restricted_selection_and_evaluation",
        "not_evaluated",
    }
)


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
    referenced_sources: set[str] = set()
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
            if (
                not isinstance(system_id, str)
                or not system_id
                or system_id in system_ids
            ):
                raise ValueError(f"duplicate system_id: {system_id}")
            system_ids.add(system_id)
            if system.get("evaluation_scope") not in EVALUATION_SCOPES:
                raise ValueError(f"unknown evaluation_scope for {system_id}")
            if system.get("evidence_status") not in EVIDENCE_STATUSES:
                raise ValueError(f"unknown evidence_status for {system_id}")
            if system.get("promotion_status") not in PROMOTION_STATUSES:
                raise ValueError(f"unknown promotion_status for {system_id}")
            if system.get("artifact_state") not in ARTIFACT_STATES:
                raise ValueError(f"unknown artifact_state for {system_id}")
            if system.get("qrel_usage") not in QREL_USAGE_STATUSES:
                raise ValueError(f"unknown qrel_usage for {system_id}")
            blocker = system.get("per_query_blocker")
            if system.get("evidence_status") == "aggregate_only" and (
                not isinstance(blocker, str) or not blocker.strip()
            ):
                raise ValueError(
                    f"concrete per_query_blocker string required for {system_id}"
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

    for raw_experiment in experiments:
        experiment = _require_mapping(raw_experiment, "experiment")
        _validate_grid_contracts(experiment, sources, source_values)
