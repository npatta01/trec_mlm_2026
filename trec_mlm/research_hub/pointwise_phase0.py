"""Fail-closed loading for the pointwise-ranking Phase 0 study evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from trec_mlm.experiment_ledger import LedgerBuild, SystemRecord

from .model import (
    HubRecord,
    HubSources,
    Phase0GateRow,
    Phase0Metrics,
    Phase0PairedRow,
    Phase0Slice,
    Phase0SystemEvidence,
    Phase0ZeroTransitions,
    PointwisePhase0Evidence,
)

_STUDY_ID = "pointwise-ranking-phase0"
_PROTOCOL_PATH = "protocols/trec_2026_pointwise_ranking_phase0.md"
_RESULTS_PATH = "reports/trec_2026_pointwise_ranking_phase0_results.json"
_PER_QUERY_PATH = "reports/trec_2026_pointwise_ranking_phase0_per_query.tsv"
_CANDIDATE_SLICES_PATH = (
    "reports/trec_2026_pointwise_ranking_phase0_candidate_slices.tsv"
)
_QUERY_COUNT = 322
_CANDIDATE_COUNT = 150
_SYSTEM_IDS = (
    "control",
    "prior_only",
    "routing_only",
    "pointwise_only",
    "combined",
)
_LEDGER_IDS = {
    "control": "dense_nested",
    "pointwise_only": "pointwise_phase0_pointwise_only",
    "combined": "pointwise_phase0_combined",
}
_SYSTEM_ROLES = {
    "control": "Main",
    "prior_only": "Diagnostic",
    "routing_only": "Diagnostic",
    "pointwise_only": "Main",
    "combined": "Main",
}
_QUERY_SLICE_KEYS = (
    ("qrel_positive_density", "le_5"),
    ("qrel_positive_density", "6_20"),
    ("qrel_positive_density", "gt_20"),
    ("dense_ndcg_status", "zero"),
    ("dense_ndcg_status", "nonzero"),
    ("query_similarity_quartile", "1"),
    ("query_similarity_quartile", "2"),
    ("query_similarity_quartile", "3"),
    ("query_similarity_quartile", "4"),
)
_CANDIDATE_SLICE_KEYS = tuple(
    (family, label)
    for family in ("audited_count_quartile", "smoothed_prior_quartile")
    for label in ("1", "2", "3", "4")
)
_PER_QUERY_COLUMNS = (
    "query_id",
    "positive_candidate_count",
    "dense_ndcg_status",
    "query_similarity_mean",
    "query_similarity_quartile",
    "control_ndcg_at_10",
    "control_reciprocal_rank",
    "prior_only_ndcg_at_10",
    "prior_only_reciprocal_rank",
    "routing_only_ndcg_at_10",
    "routing_only_reciprocal_rank",
    "pointwise_only_ndcg_at_10",
    "pointwise_only_reciprocal_rank",
    "combined_ndcg_at_10",
    "combined_reciprocal_rank",
)
_CANDIDATE_COLUMNS = (
    "slice_family",
    "slice_label",
    "candidate_count",
    "candidate_ids_json",
    "system",
    "query_count",
    "ndcg_at_10",
    "mrr",
    "zero_ndcg_queries",
)
_GATE_KEYS = (
    "combined_ndcg_point_positive",
    "combined_ndcg_interval_lower_above_zero",
    "combined_mrr_delta_at_least_minus_0_005",
    "combined_zero_ndcg_below_116",
    "routing_ndcg_above_prior_ndcg",
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_QUESTION_COUNT = re.compile(
    r"all [\d,]+ development query IDs and normalized texts and all "
    r"([\d,]+) pointwise source-question IDs"
)


@dataclass(frozen=True, slots=True)
class _PerQueryRow:
    query_id: str
    positive_candidate_count: int
    dense_ndcg_status: str
    similarity_quartile: str
    systems: tuple[tuple[str, Phase0Metrics], ...]


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _sequence(value: object, description: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{description} must be a list")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], description: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} keys must match the frozen contract")


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a nonempty string")
    return value


def _count(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def _number(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{description} must be a finite number")
    return float(value)


def _metric_number(value: object, description: str) -> float:
    number = _number(value, description)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{description} must be between zero and one")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_map(ledger: LedgerBuild) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for source in ledger.sources:
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or source_id in result:
            raise ValueError("ledger sources must have unique string source IDs")
        result[source_id] = source
    return result


def _require_source(
    root: Path,
    sources: Mapping[str, Mapping[str, str]],
    source_id: str,
    expected_path: str,
) -> str:
    source = sources.get(source_id)
    if source is None or source.get("path") != expected_path:
        raise ValueError(f"ledger source {source_id} must bind {expected_path}")
    expected_hash = source.get("sha256")
    if not isinstance(expected_hash, str) or not _HEX_SHA256.fullmatch(expected_hash):
        raise ValueError(f"ledger source {source_id} must have a SHA-256")
    if _sha256(root / expected_path) != expected_hash:
        raise ValueError(f"ledger source {source_id} hash does not match its report")
    return expected_hash


def _parse_metrics(value: object, description: str) -> tuple[Phase0Metrics, int]:
    row = _mapping(value, description)
    _exact_keys(
        row,
        {"ndcg_at_10", "mrr", "query_count", "zero_ndcg_queries"},
        description,
    )
    query_count = _count(row.get("query_count"), f"{description}.query_count")
    zero_count = _count(
        row.get("zero_ndcg_queries"), f"{description}.zero_ndcg_queries"
    )
    if zero_count > query_count:
        raise ValueError(f"{description}.zero_ndcg_queries exceeds query_count")
    return (
        Phase0Metrics(
            ndcg_at_10=_metric_number(
                row.get("ndcg_at_10"), f"{description}.ndcg_at_10"
            ),
            mrr=_metric_number(row.get("mrr"), f"{description}.mrr"),
            zero_ndcg_queries=zero_count,
        ),
        query_count,
    )


def _metrics_map(
    value: object, description: str, *, expected_query_count: int = _QUERY_COUNT
) -> dict[str, Phase0Metrics]:
    rows = _mapping(value, description)
    _exact_keys(rows, set(_SYSTEM_IDS), description)
    result: dict[str, Phase0Metrics] = {}
    for system_id in _SYSTEM_IDS:
        metrics, query_count = _parse_metrics(
            rows.get(system_id), f"{description}.{system_id}"
        )
        if query_count != expected_query_count:
            raise ValueError(
                f"{description}.{system_id} query count does not match its slice"
            )
        result[system_id] = metrics
    return result


def _parse_float_cell(value: str | None, description: str) -> float:
    if value is None:
        raise ValueError(f"{description} is missing")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{description} must be numeric") from exc
    return _metric_number(result, description)


def _load_per_query(root: Path) -> tuple[_PerQueryRow, ...]:
    rows: list[_PerQueryRow] = []
    seen: set[str] = set()
    with (root / _PER_QUERY_PATH).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _PER_QUERY_COLUMNS:
            raise ValueError(f"{_PER_QUERY_PATH}: columns must match frozen schema")
        for line_number, row in enumerate(reader, start=2):
            query_id = _string(row.get("query_id"), f"line {line_number} query_id")
            if query_id in seen:
                raise ValueError(f"{_PER_QUERY_PATH}: duplicate query_id {query_id}")
            seen.add(query_id)
            try:
                positive_count = int(row["positive_candidate_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{_PER_QUERY_PATH}: invalid positive count at line {line_number}"
                ) from exc
            if not 0 <= positive_count <= _CANDIDATE_COUNT:
                raise ValueError(f"{_PER_QUERY_PATH}: positive count outside scope")
            dense_status = row.get("dense_ndcg_status")
            if dense_status not in {"zero", "nonzero"}:
                raise ValueError(f"{_PER_QUERY_PATH}: invalid dense status")
            quartile = row.get("query_similarity_quartile")
            if quartile not in {"1", "2", "3", "4"}:
                raise ValueError(f"{_PER_QUERY_PATH}: invalid similarity quartile")
            _parse_float_cell(
                row.get("query_similarity_mean"), "query_similarity_mean"
            )
            metrics: list[tuple[str, Phase0Metrics]] = []
            for system_id in _SYSTEM_IDS:
                ndcg = _parse_float_cell(
                    row.get(f"{system_id}_ndcg_at_10"),
                    f"{system_id}_ndcg_at_10 line {line_number}",
                )
                mrr = _parse_float_cell(
                    row.get(f"{system_id}_reciprocal_rank"),
                    f"{system_id}_reciprocal_rank line {line_number}",
                )
                metrics.append(
                    (
                        system_id,
                        Phase0Metrics(
                            ndcg_at_10=ndcg,
                            mrr=mrr,
                            zero_ndcg_queries=int(ndcg == 0.0),
                        ),
                    )
                )
            metric_map = dict(metrics)
            if (metric_map["control"].ndcg_at_10 == 0.0) != (
                dense_status == "zero"
            ):
                raise ValueError(f"{_PER_QUERY_PATH}: dense status disagrees with nDCG")
            rows.append(
                _PerQueryRow(
                    query_id=query_id,
                    positive_candidate_count=positive_count,
                    dense_ndcg_status=dense_status,
                    similarity_quartile=quartile,
                    systems=tuple(metrics),
                )
            )
    if len(rows) != _QUERY_COUNT:
        raise ValueError(f"{_PER_QUERY_PATH}: expected 322 unique query rows")
    return tuple(rows)


def _aggregate_rows(rows: Sequence[_PerQueryRow], system_id: str) -> Phase0Metrics:
    values = [dict(row.systems)[system_id] for row in rows]
    return Phase0Metrics(
        ndcg_at_10=math.fsum(value.ndcg_at_10 for value in values) / len(values),
        mrr=math.fsum(value.mrr for value in values) / len(values),
        zero_ndcg_queries=sum(value.zero_ndcg_queries for value in values),
    )


def _same_metrics(left: Phase0Metrics, right: Phase0Metrics) -> bool:
    return (
        math.isclose(left.ndcg_at_10, right.ndcg_at_10, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(left.mrr, right.mrr, rel_tol=0.0, abs_tol=1e-12)
        and left.zero_ndcg_queries == right.zero_ndcg_queries
    )


def _slice_rows(
    rows: Sequence[_PerQueryRow], family: str, label: str
) -> tuple[_PerQueryRow, ...]:
    if family == "qrel_positive_density":
        return tuple(
            row
            for row in rows
            if (
                (label == "le_5" and row.positive_candidate_count <= 5)
                or (label == "6_20" and 6 <= row.positive_candidate_count <= 20)
                or (label == "gt_20" and row.positive_candidate_count > 20)
            )
        )
    if family == "dense_ndcg_status":
        return tuple(row for row in rows if row.dense_ndcg_status == label)
    return tuple(row for row in rows if row.similarity_quartile == label)


def _load_query_slices(
    value: object, per_query: Sequence[_PerQueryRow]
) -> tuple[Phase0Slice, ...]:
    raw_rows = _sequence(value, "query_slices")
    if len(raw_rows) != len(_QUERY_SLICE_KEYS):
        raise ValueError("query_slices must contain all nine frozen strata")
    by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for index, value_row in enumerate(raw_rows):
        row = _mapping(value_row, f"query_slices[{index}]")
        _exact_keys(row, {"family", "label", "query_count", "query_ids", "metrics"}, f"query_slices[{index}]")
        key = (
            _string(row.get("family"), "query slice family"),
            _string(row.get("label"), "query slice label"),
        )
        if key in by_key:
            raise ValueError(f"duplicate query slice {key}")
        by_key[key] = row
    if tuple(by_key) != _QUERY_SLICE_KEYS:
        raise ValueError("query_slices order and identities must match frozen contract")

    result: list[Phase0Slice] = []
    for family, label in _QUERY_SLICE_KEYS:
        row = by_key[(family, label)]
        selected = _slice_rows(per_query, family, label)
        query_ids = tuple(
            _string(item, f"{family}.{label}.query_id")
            for item in _sequence(row.get("query_ids"), f"{family}.{label}.query_ids")
        )
        if len(query_ids) != len(set(query_ids)) or set(query_ids) != {
            item.query_id for item in selected
        }:
            raise ValueError(f"query slice {family}/{label} membership drift")
        query_count = _count(row.get("query_count"), f"{family}.{label}.query_count")
        if query_count != len(selected):
            raise ValueError(f"query slice {family}/{label} count drift")
        metrics = _metrics_map(
            row.get("metrics"),
            f"query slice {family}/{label}",
            expected_query_count=len(selected),
        )
        for system_id in _SYSTEM_IDS:
            recomputed = _aggregate_rows(selected, system_id)
            if not _same_metrics(recomputed, metrics[system_id]):
                raise ValueError(
                    f"query slice {family}/{label}/{system_id} metrics drift"
                )
        result.append(
            Phase0Slice(
                family=family,
                label=label,
                member_count=query_count,
                member_ids=query_ids,
                systems=tuple((system_id, metrics[system_id]) for system_id in _SYSTEM_IDS),
            )
        )
    for family in {family for family, _label in _QUERY_SLICE_KEYS}:
        if sum(row.member_count for row in result if row.family == family) != _QUERY_COUNT:
            raise ValueError(f"query slice family {family} must partition 322 queries")
    return tuple(result)


def _load_candidate_tsv(root: Path) -> tuple[Phase0Slice, ...]:
    grouped: dict[tuple[str, str], dict[str, Phase0Metrics]] = {}
    candidate_ids: dict[tuple[str, str], tuple[str, ...]] = {}
    with (root / _CANDIDATE_SLICES_PATH).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _CANDIDATE_COLUMNS:
            raise ValueError(
                f"{_CANDIDATE_SLICES_PATH}: columns must match frozen schema"
            )
        for line_number, row in enumerate(reader, start=2):
            key = (
                _string(row.get("slice_family"), f"line {line_number} family"),
                _string(row.get("slice_label"), f"line {line_number} label"),
            )
            if key not in _CANDIDATE_SLICE_KEYS:
                raise ValueError(f"unexpected candidate slice {key}")
            system_id = _string(row.get("system"), f"line {line_number} system")
            if system_id not in _SYSTEM_IDS or system_id in grouped.setdefault(key, {}):
                raise ValueError("duplicate or unknown candidate slice system")
            try:
                ids_value = json.loads(row["candidate_ids_json"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("candidate_ids_json must be valid JSON") from exc
            ids = tuple(
                _string(item, f"line {line_number} candidate ID")
                for item in _sequence(ids_value, "candidate_ids_json")
            )
            try:
                member_count = int(row["candidate_count"])
                query_count = int(row["query_count"])
                zero_count = int(row["zero_ndcg_queries"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("candidate slice counts must be integers") from exc
            if member_count != len(ids) or len(ids) != len(set(ids)):
                raise ValueError(f"candidate slice {key} has invalid membership")
            if query_count != _QUERY_COUNT or not 0 <= zero_count <= _QUERY_COUNT:
                raise ValueError(f"candidate slice {key} has invalid query scope")
            if key in candidate_ids and candidate_ids[key] != ids:
                raise ValueError(f"candidate slice {key} membership differs by system")
            candidate_ids[key] = ids
            grouped[key][system_id] = Phase0Metrics(
                ndcg_at_10=_parse_float_cell(row.get("ndcg_at_10"), "ndcg_at_10"),
                mrr=_parse_float_cell(row.get("mrr"), "mrr"),
                zero_ndcg_queries=zero_count,
            )
    if tuple(grouped) != _CANDIDATE_SLICE_KEYS:
        raise ValueError("candidate slice rows must match all eight frozen strata")
    result: list[Phase0Slice] = []
    family_unions: dict[str, set[str]] = {}
    for key in _CANDIDATE_SLICE_KEYS:
        metrics = grouped[key]
        if set(metrics) != set(_SYSTEM_IDS):
            raise ValueError(f"candidate slice {key} must contain all five systems")
        ids = candidate_ids[key]
        family_members = family_unions.setdefault(key[0], set())
        if family_members.intersection(ids):
            raise ValueError(f"candidate slice family {key[0]} overlaps")
        family_members.update(ids)
        result.append(
            Phase0Slice(
                family=key[0],
                label=key[1],
                member_count=len(ids),
                member_ids=ids,
                systems=tuple((system_id, metrics[system_id]) for system_id in _SYSTEM_IDS),
            )
        )
    if any(len(ids) != _CANDIDATE_COUNT for ids in family_unions.values()):
        raise ValueError("each candidate slice family must partition 150 candidates")
    if len({frozenset(ids) for ids in family_unions.values()}) != 1:
        raise ValueError("candidate slice families must cover the same candidates")
    return tuple(result)


def _reconcile_candidate_json(
    value: object, candidate_slices: Sequence[Phase0Slice]
) -> None:
    raw_rows = _sequence(value, "candidate_slices")
    if len(raw_rows) != len(_CANDIDATE_SLICE_KEYS):
        raise ValueError("candidate_slices must contain eight strata")
    for raw, typed, expected_key in zip(
        raw_rows, candidate_slices, _CANDIDATE_SLICE_KEYS, strict=True
    ):
        row = _mapping(raw, f"candidate_slices.{expected_key}")
        _exact_keys(
            row,
            {"family", "label", "candidate_count", "candidate_ids", "metrics"},
            f"candidate_slices.{expected_key}",
        )
        if (row.get("family"), row.get("label")) != expected_key:
            raise ValueError("candidate_slices order and identities drifted")
        if row.get("candidate_count") != typed.member_count:
            raise ValueError(f"candidate slice {expected_key} count drift")
        json_ids = tuple(
            _string(item, f"candidate slice {expected_key} member")
            for item in _sequence(
                row.get("candidate_ids"),
                f"candidate slice {expected_key}.candidate_ids",
            )
        )
        if json_ids != typed.member_ids:
            raise ValueError(f"candidate slice {expected_key} membership drift")
        metrics = _metrics_map(row.get("metrics"), f"candidate slice {expected_key}")
        for system_id, tsv_metrics in typed.systems:
            if not _same_metrics(metrics[system_id], tsv_metrics):
                raise ValueError(
                    f"candidate slice {expected_key}/{system_id} JSON/TSV drift"
                )


def _ledger_systems(
    ledger: LedgerBuild, metrics: Mapping[str, Phase0Metrics]
) -> tuple[Phase0SystemEvidence, ...]:
    by_id = {system.system_id: system for system in ledger.systems}
    result: list[Phase0SystemEvidence] = []
    expected_state = {
        "control": ("stable_control", "training_and_evaluation"),
        "pointwise_only": ("valid_negative", "evaluation_only"),
        "combined": ("valid_negative", "evaluation_only"),
    }
    for source_system_id in _SYSTEM_IDS:
        ledger_id = _LEDGER_IDS.get(source_system_id)
        if ledger_id is None:
            result.append(
                Phase0SystemEvidence(
                    source_system_id=source_system_id,
                    ledger_system_id=None,
                    role=_SYSTEM_ROLES[source_system_id],
                    metrics=metrics[source_system_id],
                    query_count=_QUERY_COUNT,
                    candidate_count=_CANDIDATE_COUNT,
                    evaluation_scope="dev_322x150",
                    evidence_status="validated_result_diagnostic",
                    promotion_status="not_registered",
                    artifact_state="tracked_result_diagnostic",
                    qrel_usage="evaluation_only",
                )
            )
            continue
        system = by_id.get(ledger_id)
        if system is None:
            raise ValueError(f"Phase 0 ledger system missing: {ledger_id}")
        _validate_ledger_system(system, source_system_id, metrics[source_system_id], expected_state[source_system_id])
        result.append(
            Phase0SystemEvidence(
                source_system_id=source_system_id,
                ledger_system_id=ledger_id,
                role=_SYSTEM_ROLES[source_system_id],
                metrics=metrics[source_system_id],
                query_count=system.query_count or 0,
                candidate_count=system.candidate_count or 0,
                evaluation_scope=system.evaluation_scope,
                evidence_status=system.evidence_status,
                promotion_status=system.promotion_status,
                artifact_state=system.artifact_state,
                qrel_usage=system.qrel_usage,
            )
        )
    return tuple(result)


def _validate_ledger_system(
    system: SystemRecord,
    source_system_id: str,
    metrics: Phase0Metrics,
    expected_state: tuple[str, str],
) -> None:
    expected_source_id = source_system_id if source_system_id != "control" else "dense_nested"
    if system.source_system_id != expected_source_id:
        raise ValueError(f"{system.system_id}: source system identity drift")
    if (
        system.evaluation_scope != "dev_322x150"
        or system.evidence_status != "stored_per_query"
        or system.query_count != _QUERY_COUNT
        or system.candidate_count != _CANDIDATE_COUNT
        or system.artifact_state != "tracked_complete"
        or (system.promotion_status, system.qrel_usage) != expected_state
    ):
        raise ValueError(f"{system.system_id}: ledger scope/status/custody drift")
    if system.ndcg_at_10 is None or system.mrr is None:
        raise ValueError(f"{system.system_id}: ledger aggregate metrics missing")
    if not math.isclose(system.ndcg_at_10, metrics.ndcg_at_10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{system.system_id}: ledger nDCG does not match receipt")
    if not math.isclose(system.mrr, metrics.mrr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{system.system_id}: ledger MRR does not match receipt")
    if system.zero_ndcg_queries is not None and system.zero_ndcg_queries != metrics.zero_ndcg_queries:
        raise ValueError(f"{system.system_id}: ledger zero count does not match receipt")


def _paired_rows(
    value: object,
    metrics: Mapping[str, Phase0Metrics],
    per_query: Sequence[_PerQueryRow],
) -> tuple[tuple[Phase0PairedRow, ...], tuple[Phase0ZeroTransitions, ...]]:
    comparisons = _mapping(value, "comparisons_vs_control")
    _exact_keys(comparisons, {"pointwise_only", "combined"}, "comparisons_vs_control")
    paired: list[Phase0PairedRow] = []
    transitions: list[Phase0ZeroTransitions] = []
    for system_id in ("pointwise_only", "combined"):
        comparison = _mapping(comparisons.get(system_id), f"comparison {system_id}")
        _exact_keys(
            comparison,
            {
                "candidate_minus_control",
                "ndcg_at_10_delta",
                "mrr_delta",
                "ndcg_at_10_win_tie_loss",
                "mrr_win_tie_loss",
                "zero_transitions",
            },
            f"comparison {system_id}",
        )
        if comparison.get("candidate_minus_control") is not True:
            raise ValueError(f"comparison {system_id} must be candidate minus control")
        for metric_key, metric_label, value_field in (
            ("ndcg_at_10", "nDCG@10", "ndcg_at_10"),
            ("mrr", "MRR", "mrr"),
        ):
            delta = _mapping(comparison.get(f"{metric_key}_delta"), f"{system_id} {metric_key} delta")
            _exact_keys(delta, {"confidence_level", "generator", "lower", "mean_difference", "n_resamples", "seed", "upper"}, f"{system_id} {metric_key} delta")
            lower = _number(delta.get("lower"), "CI lower")
            mean = _number(delta.get("mean_difference"), "mean difference")
            upper = _number(delta.get("upper"), "CI upper")
            if not lower <= mean <= upper:
                raise ValueError(f"{system_id} {metric_key} CI does not contain mean")
            if (
                delta.get("confidence_level") != 0.95
                or delta.get("generator") != "python_random_mt19937"
                or delta.get("n_resamples") != 10000
                or delta.get("seed") != 31
            ):
                raise ValueError(f"{system_id} {metric_key} bootstrap contract drift")
            expected_delta = getattr(metrics[system_id], value_field) - getattr(metrics["control"], value_field)
            if not math.isclose(mean, expected_delta, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{system_id} {metric_key} mean difference drift")
            wtl = _mapping(comparison.get(f"{metric_key}_win_tie_loss"), f"{system_id} {metric_key} WTL")
            _exact_keys(wtl, {"wins", "ties", "losses", "tolerance"}, f"{system_id} {metric_key} WTL")
            wins = _count(wtl.get("wins"), "wins")
            ties = _count(wtl.get("ties"), "ties")
            losses = _count(wtl.get("losses"), "losses")
            tolerance = _number(wtl.get("tolerance"), "WTL tolerance")
            if tolerance != 1e-12 or wins + ties + losses != _QUERY_COUNT:
                raise ValueError(f"{system_id} {metric_key} WTL contract drift")
            differences = [
                getattr(dict(row.systems)[system_id], value_field)
                - getattr(dict(row.systems)["control"], value_field)
                for row in per_query
            ]
            recomputed = (
                sum(value > tolerance for value in differences),
                sum(abs(value) <= tolerance for value in differences),
                sum(value < -tolerance for value in differences),
            )
            if recomputed != (wins, ties, losses):
                raise ValueError(f"{system_id} {metric_key} WTL does not match per-query TSV")
            paired.append(
                Phase0PairedRow(
                    source_system_id=system_id,
                    metric=metric_label,
                    difference=mean,
                    ci_lower=lower,
                    ci_upper=upper,
                    wins=wins,
                    ties=ties,
                    losses=losses,
                )
            )
        raw_transitions = _mapping(comparison.get("zero_transitions"), f"{system_id} transitions")
        _exact_keys(raw_transitions, {"zero_both", "dense_zero_candidate_nonzero", "dense_nonzero_candidate_zero", "nonzero_both", "query_count"}, f"{system_id} transitions")
        counts = tuple(
            _count(raw_transitions.get(key), f"{system_id}.{key}")
            for key in (
                "zero_both",
                "dense_zero_candidate_nonzero",
                "dense_nonzero_candidate_zero",
                "nonzero_both",
            )
        )
        if raw_transitions.get("query_count") != _QUERY_COUNT or sum(counts) != _QUERY_COUNT:
            raise ValueError(f"{system_id} transition count drift")
        recomputed_counts = [0, 0, 0, 0]
        for row in per_query:
            row_metrics = dict(row.systems)
            control_zero = row_metrics["control"].ndcg_at_10 == 0.0
            candidate_zero = row_metrics[system_id].ndcg_at_10 == 0.0
            index = {(True, True): 0, (True, False): 1, (False, True): 2, (False, False): 3}[(control_zero, candidate_zero)]
            recomputed_counts[index] += 1
        if tuple(recomputed_counts) != counts:
            raise ValueError(f"{system_id} transitions do not match per-query TSV")
        transitions.append(Phase0ZeroTransitions(system_id, *counts))
    return tuple(paired), tuple(transitions)


def _decision(
    value: object,
    metrics: Mapping[str, Phase0Metrics],
    paired: Sequence[Phase0PairedRow],
    selector_receipt: object,
) -> tuple[tuple[Phase0GateRow, ...], str, str, int, int, int]:
    decision = _mapping(value, "decision")
    _exact_keys(decision, {"checks", "decision", "recommendation", "selector_availability", "values"}, "decision")
    checks = _mapping(decision.get("checks"), "decision.checks")
    _exact_keys(checks, set(_GATE_KEYS), "decision.checks")
    paired_by_key = {(row.source_system_id, row.metric): row for row in paired}
    combined_ndcg = paired_by_key[("combined", "nDCG@10")]
    combined_mrr = paired_by_key[("combined", "MRR")]
    expected = {
        "combined_ndcg_point_positive": combined_ndcg.difference > 0.0,
        "combined_ndcg_interval_lower_above_zero": combined_ndcg.ci_lower > 0.0,
        "combined_mrr_delta_at_least_minus_0_005": combined_mrr.difference >= -0.005,
        "combined_zero_ndcg_below_116": metrics["combined"].zero_ndcg_queries < metrics["control"].zero_ndcg_queries,
        "routing_ndcg_above_prior_ndcg": metrics["routing_only"].ndcg_at_10 > metrics["prior_only"].ndcg_at_10,
    }
    if any(checks.get(key) is not expected[key] for key in _GATE_KEYS):
        raise ValueError("decision checks do not match structured evidence")
    gate_rows = (
        Phase0GateRow(_GATE_KEYS[0], "Combined nDCG@10 difference", "> 0", f"{combined_ndcg.difference:.6f}", expected[_GATE_KEYS[0]]),
        Phase0GateRow(_GATE_KEYS[1], "Paired nDCG lower bound", "> 0", f"{combined_ndcg.ci_lower:.6f}", expected[_GATE_KEYS[1]]),
        Phase0GateRow(_GATE_KEYS[2], "Combined MRR difference", "≥ −0.005", f"{combined_mrr.difference:.6f}", expected[_GATE_KEYS[2]]),
        Phase0GateRow(_GATE_KEYS[3], "Combined zero-nDCG queries", f"< {metrics['control'].zero_ndcg_queries}", str(metrics["combined"].zero_ndcg_queries), expected[_GATE_KEYS[3]]),
        Phase0GateRow(_GATE_KEYS[4], "Routing-only nDCG above prior-only", ">", f"{metrics['routing_only'].ndcg_at_10:.6f} > {metrics['prior_only'].ndcg_at_10:.6f}".replace("0.", "."), expected[_GATE_KEYS[4]]),
    )
    if decision.get("decision") != "fail" or decision.get("recommendation") != "stop_pairwise_work":
        raise ValueError("Phase 0 decision must preserve the binding failed gate")
    values = _mapping(decision.get("values"), "decision.values")
    _exact_keys(values, {"combined_mrr_delta", "combined_ndcg_delta", "combined_ndcg_lower", "combined_zero_ndcg", "prior_ndcg", "routing_ndcg"}, "decision.values")
    expected_values = {
        "combined_mrr_delta": combined_mrr.difference,
        "combined_ndcg_delta": combined_ndcg.difference,
        "combined_ndcg_lower": combined_ndcg.ci_lower,
        "combined_zero_ndcg": metrics["combined"].zero_ndcg_queries,
        "prior_ndcg": metrics["prior_only"].ndcg_at_10,
        "routing_ndcg": metrics["routing_only"].ndcg_at_10,
    }
    for key, expected_value in expected_values.items():
        actual = values.get(key)
        if isinstance(expected_value, float):
            if not math.isclose(_number(actual, f"decision.values.{key}"), expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"decision.values.{key} drift")
        elif actual != expected_value:
            raise ValueError(f"decision.values.{key} drift")
    availability = _mapping(decision.get("selector_availability"), "selector availability")
    _exact_keys(availability, {"available", "minimum_pair_count", "minimum_source_question_count", "selected_pair_count", "selector_version", "source_question_count"}, "selector availability")
    if availability.get("available") is not True or availability.get("selector_version") != "pointwise-phase0-pair-selector-v1":
        raise ValueError("selector availability contract drift")
    selected = _count(availability.get("selected_pair_count"), "selected pairs")
    source_questions = _count(availability.get("source_question_count"), "source questions")
    if selected < _count(availability.get("minimum_pair_count"), "minimum pairs") or source_questions < _count(availability.get("minimum_source_question_count"), "minimum source questions"):
        raise ValueError("selector availability no longer satisfies minimums")
    receipt = _mapping(selector_receipt, "selector_receipt")
    if receipt.get("selected_pair_count") != selected or receipt.get("source_question_count") != source_questions:
        raise ValueError("selector receipt does not match decision availability")
    cap = _count(receipt.get("final_candidate_use_cap"), "candidate use cap")
    return gate_rows, "fail", "stop_pairwise_work", selected, source_questions, cap


def _validate_provenance(
    payload: Mapping[str, object],
    source_hashes: Mapping[str, str],
    candidate_hash: str,
) -> tuple[int, int, int, int]:
    if payload.get("artifact_type") != "trec_2026_pointwise_ranking_phase0_evaluation" or payload.get("version") != 1 or payload.get("status") != "evaluated":
        raise ValueError("Phase 0 results identity/status contract drift")
    if payload.get("development_queries_previously_seen") is not True or payload.get("qrel_free_ranking_construction") is not True:
        raise ValueError("Phase 0 development/leakage boundary drift")
    inputs = _mapping(payload.get("inputs"), "inputs")
    expected_inputs = {
        "query_count": _QUERY_COUNT,
        "candidate_count": _CANDIDATE_COUNT,
        "qrel_count": _QUERY_COUNT * _CANDIDATE_COUNT,
        "development_queries_sha256": source_hashes["development_queries"],
        "qrels_sha256": source_hashes["development_qrels"],
        "dense_result_receipt_sha256": source_hashes["dense_query_results"],
        "dense_per_query_receipt_sha256": source_hashes["dense_query_per_query"],
    }
    if any(inputs.get(key) != expected for key, expected in expected_inputs.items()):
        raise ValueError("Phase 0 input scope/hash provenance drift")
    baseline = _mapping(payload.get("baseline_reproduction"), "baseline_reproduction")
    if (
        baseline.get("status") != "passed"
        or baseline.get("query_count") != _QUERY_COUNT
        or baseline.get("per_query_values_checked") != _QUERY_COUNT
        or baseline.get("result_receipt_sha256") != source_hashes["dense_query_results"]
        or baseline.get("per_query_receipt_sha256") != source_hashes["dense_query_per_query"]
        or baseline.get("absolute_tolerance") != 5e-12
    ):
        raise ValueError("dense baseline reproduction provenance drift")
    validation = _mapping(payload.get("independent_validation"), "independent_validation")
    if validation.get("status") != "independently_validated" or validation.get("assessment") != "passed":
        raise ValueError("independent validation status must be passed")
    checks = _mapping(validation.get("checks"), "independent_validation.checks")
    expected_checks = {
        "aggregate_systems_recomputed": 5,
        "per_query_metric_cells_recomputed": 3220,
        "query_slices_recomputed": 9,
        "candidate_slice_rows_recomputed": 40,
        "candidate_system_metric_cells_recomputed": 40,
        "qrel_query_count": _QUERY_COUNT,
        "qrel_candidate_count": _CANDIDATE_COUNT,
        "qrel_row_count": _QUERY_COUNT * _CANDIDATE_COUNT,
        "run_count": 5,
        "run_rows_each": _QUERY_COUNT * _CANDIDATE_COUNT,
        "dense_per_query_receipt_rows_within_5e_12": _QUERY_COUNT,
        "gate_and_stop_recomputed": True,
        "selector_complete_receipt_reconciled": True,
        "win_tie_loss_and_zero_transitions_recomputed": True,
        "tracked_report_content_scan": "passed",
    }
    if any(checks.get(key) != expected for key, expected in expected_checks.items()):
        raise ValueError("independent validation completeness drift")
    provenance = _mapping(validation.get("provenance"), "independent_validation.provenance")
    if provenance.get("per_query_evaluator_sha256") != source_hashes["pointwise_phase0_per_query"] or provenance.get("candidate_slices_evaluator_sha256") != candidate_hash:
        raise ValueError("validated report hashes do not match tracked TSVs")
    for key, value in provenance.items():
        if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
            raise ValueError(f"independent validation provenance {key} is not SHA-256")
    caveats = _sequence(validation.get("caveats"), "independent_validation.caveats")
    match = next(
        (
            _SOURCE_QUESTION_COUNT.search(item)
            for item in caveats
            if isinstance(item, str) and _SOURCE_QUESTION_COUNT.search(item)
        ),
        None,
    )
    if match is None:
        raise ValueError("independent validation overlap boundary is missing")
    source_questions = int(match.group(1).replace(",", ""))
    return 3220, 40, _QUERY_COUNT * _CANDIDATE_COUNT, source_questions


def _audited_label_count(
    value: object, candidate_slices: Sequence[Phase0Slice]
) -> int:
    diagnostics = _mapping(value, "diagnostics")
    expected_fields = {
        "candidate_audited_counts",
        "candidate_audited_question_coverage",
        "candidate_smoothed_priors",
        "candidate_supported_counts",
    }
    _exact_keys(diagnostics, expected_fields, "diagnostics")
    candidate_ids = set().union(
        *(
            row.member_ids
            for row in candidate_slices
            if row.family == "audited_count_quartile"
        )
    )
    typed: dict[str, Mapping[str, object]] = {}
    for field in expected_fields:
        rows = _mapping(diagnostics.get(field), f"diagnostics.{field}")
        if set(rows) != candidate_ids or len(rows) != _CANDIDATE_COUNT:
            raise ValueError(f"diagnostics.{field} must cover all 150 candidates")
        typed[field] = rows
    total = 0
    for candidate_id in sorted(candidate_ids):
        audited = _count(
            typed["candidate_audited_counts"].get(candidate_id),
            f"audited count {candidate_id}",
        )
        coverage = _count(
            typed["candidate_audited_question_coverage"].get(candidate_id),
            f"audited coverage {candidate_id}",
        )
        supported = _count(
            typed["candidate_supported_counts"].get(candidate_id),
            f"supported count {candidate_id}",
        )
        _metric_number(
            typed["candidate_smoothed_priors"].get(candidate_id),
            f"smoothed prior {candidate_id}",
        )
        if coverage > audited or supported > audited:
            raise ValueError(f"diagnostics counts are inconsistent for {candidate_id}")
        total += audited
    return total


def _reconcile_baseline_metrics(
    value: object, control_metrics: Phase0Metrics
) -> None:
    baseline = _mapping(value, "baseline_reproduction")
    _exact_keys(
        baseline,
        {
            "absolute_tolerance",
            "mrr",
            "ndcg_at_10",
            "per_query_receipt_sha256",
            "per_query_values_checked",
            "query_count",
            "result_receipt_sha256",
            "status",
            "zero_ndcg_queries",
        },
        "baseline_reproduction",
    )
    baseline_metrics = Phase0Metrics(
        ndcg_at_10=_metric_number(
            baseline.get("ndcg_at_10"), "baseline_reproduction.ndcg_at_10"
        ),
        mrr=_metric_number(baseline.get("mrr"), "baseline_reproduction.mrr"),
        zero_ndcg_queries=_count(
            baseline.get("zero_ndcg_queries"),
            "baseline_reproduction.zero_ndcg_queries",
        ),
    )
    if not _same_metrics(baseline_metrics, control_metrics):
        raise ValueError("baseline reproduction metrics do not match control")


def load_pointwise_phase0_evidence(
    root: Path, sources: HubSources, ledger: LedgerBuild
) -> PointwisePhase0Evidence | None:
    """Load Phase 0 evidence only when its canonical study exists."""

    study = next((row for row in sources.studies if row.record_id == _STUDY_ID), None)
    if study is None:
        return None
    _validate_study_links(study)
    ledger_sources = _source_map(ledger)
    source_hashes = {
        "pointwise_phase0_results": _require_source(root, ledger_sources, "pointwise_phase0_results", _RESULTS_PATH),
        "pointwise_phase0_per_query": _require_source(root, ledger_sources, "pointwise_phase0_per_query", _PER_QUERY_PATH),
        "dense_query_results": _require_source(root, ledger_sources, "dense_query_results", "reports/trec_2026_dense_query_knn_results.json"),
        "dense_query_per_query": _require_source(root, ledger_sources, "dense_query_per_query", "reports/trec_2026_dense_query_knn_per_query.tsv"),
        "development_queries": _require_source(root, ledger_sources, "development_queries", "data/dev/llm_dev_data.tsv"),
        "development_qrels": _require_source(root, ledger_sources, "development_qrels", "data/dev/llm_dev_qrels.txt"),
    }
    payload = _mapping(json.loads((root / _RESULTS_PATH).read_text(encoding="utf-8")), _RESULTS_PATH)
    required_top_keys = {
        "artifact_type", "baseline_reproduction", "candidate_slices",
        "comparisons_vs_control", "decision", "development_queries_previously_seen",
        "diagnostics", "independent_validation", "inputs", "protocol",
        "qrel_free_ranking_construction", "query_slices", "selector_receipt",
        "status", "systems", "version",
    }
    _exact_keys(payload, required_top_keys, _RESULTS_PATH)
    candidate_hash = _sha256(root / _CANDIDATE_SLICES_PATH)
    validation_counts = _validate_provenance(payload, source_hashes, candidate_hash)
    per_query = _load_per_query(root)
    aggregate_metrics = _metrics_map(payload.get("systems"), "systems")
    _reconcile_baseline_metrics(
        payload.get("baseline_reproduction"), aggregate_metrics["control"]
    )
    for system_id in _SYSTEM_IDS:
        if not _same_metrics(_aggregate_rows(per_query, system_id), aggregate_metrics[system_id]):
            raise ValueError(f"{system_id} aggregate does not match per-query TSV")
    systems = _ledger_systems(ledger, aggregate_metrics)
    query_slices = _load_query_slices(payload.get("query_slices"), per_query)
    candidate_slices = _load_candidate_tsv(root)
    _reconcile_candidate_json(payload.get("candidate_slices"), candidate_slices)
    audited_label_count = _audited_label_count(
        payload.get("diagnostics"), candidate_slices
    )
    paired_rows, transitions = _paired_rows(payload.get("comparisons_vs_control"), aggregate_metrics, per_query)
    gate_rows, decision, recommendation, selected, source_questions, cap = _decision(
        payload.get("decision"), aggregate_metrics, paired_rows, payload.get("selector_receipt")
    )
    if sum(row.passed for row in gate_rows) != 1:
        raise ValueError("Phase 0 gate must pass exactly one of five checks")
    positive_candidate_slices = [
        row
        for row in candidate_slices
        if dict(row.systems)["combined"].ndcg_at_10
        > dict(row.systems)["control"].ndcg_at_10
    ]
    if [(row.family, row.label) for row in positive_candidate_slices] != [
        ("smoothed_prior_quartile", "2")
    ]:
        raise ValueError("candidate-slice positive context drift")
    return PointwisePhase0Evidence(
        study_id=_STUDY_ID,
        protocol_path=_PROTOCOL_PATH,
        results_path=_RESULTS_PATH,
        per_query_path=_PER_QUERY_PATH,
        candidate_slices_path=_CANDIDATE_SLICES_PATH,
        systems=systems,
        paired_rows=paired_rows,
        zero_transitions=transitions,
        query_slices=query_slices,
        candidate_slices=candidate_slices,
        gate_rows=gate_rows,
        passed_gate_count=1,
        decision=decision,
        recommendation=recommendation,
        selector_selected_pairs=selected,
        selector_source_questions=source_questions,
        selector_candidate_use_cap=cap,
        audited_label_count=audited_label_count,
        validation_metric_cells=validation_counts[0],
        validation_candidate_rows=validation_counts[1],
        validation_run_rows=validation_counts[2],
        validation_source_questions=validation_counts[3],
    )


def _validate_study_links(study: HubRecord) -> None:
    if study.metadata.get("status") != "concluded":
        raise ValueError(f"{study.source_path}: Phase 0 study must be concluded")
    expected_system_ids = tuple(_LEDGER_IDS.values())
    if tuple(study.metadata.get("system_ids", ())) != expected_system_ids:
        raise ValueError(f"{study.source_path}: Phase 0 system IDs drift")
    if tuple(study.metadata.get("comparison_system_ids", ())) != expected_system_ids:
        raise ValueError(f"{study.source_path}: Phase 0 comparison IDs drift")
    if tuple(study.metadata.get("representative_system_ids", ())) != expected_system_ids[1:]:
        raise ValueError(f"{study.source_path}: Phase 0 representative IDs drift")
    evidence_refs = study.metadata.get("evidence_refs", ())
    if not isinstance(evidence_refs, Sequence) or isinstance(evidence_refs, (str, bytes)):
        raise ValueError(f"{study.source_path}: evidence_refs must be a list")
    required = {_PROTOCOL_PATH, _RESULTS_PATH, _PER_QUERY_PATH, _CANDIDATE_SLICES_PATH}
    if set(evidence_refs) != required:
        raise ValueError(f"{study.source_path}: Phase 0 evidence references drift")
