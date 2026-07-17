from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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


@dataclass(frozen=True, slots=True)
class SystemRecord:
    system_id: str
    source_system_id: str
    experiment_id: str
    experiment_family: str
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
    per_query_available: bool
    ranking_coverage: str
    artifact_state: str
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
    value = payload.get("zero_ndcg_queries")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("zero_ndcg_queries must be a nonnegative integer")
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
    elif kind == "dense_per_query_tsv":
        query_column = source_ref.get("query_id_column")
        ndcg_column = source_ref.get("ndcg_column")
        rr_column = source_ref.get("reciprocal_rank_column")
        if not all(
            isinstance(column, str) for column in (query_column, ndcg_column, rr_column)
        ):
            raise ValueError(f"invalid dense per-query columns for {system_id}")
        with (root / source_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required_columns = {query_column, ndcg_column, rr_column}
            if reader.fieldnames is None or not required_columns <= set(
                reader.fieldnames
            ):
                raise ValueError(f"missing dense per-query columns for {system_id}")
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
                        f"invalid dense per-query metric for {system_id}/{query_id}"
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
            aggregate_source = _require_mapping(
                system.get("aggregate_source"),
                f"aggregate source for {system_id}",
            )
            aggregate_source_id = aggregate_source.get("source_id")
            if not isinstance(aggregate_source_id, str):
                raise ValueError(f"aggregate source_id required for {system_id}")
            source_path, source_sha256 = _source_details(sources, aggregate_source_id)
            aggregate = _resolve_source_reference(
                root, sources, aggregate_source, json_cache
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
            if per_query_source is not None:
                per_query_source = _require_mapping(
                    per_query_source, f"per-query source for {system_id}"
                )
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
                    experiment_id=experiment_id,
                    experiment_family=experiment_family,
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
                    per_query_available=bool(system_rows),
                    ranking_coverage=str(system["ranking_coverage"]),
                    artifact_state=str(system["artifact_state"]),
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
