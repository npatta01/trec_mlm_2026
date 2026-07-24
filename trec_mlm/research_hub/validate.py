"""Schema and relationship validation for canonical research-hub records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from trec_mlm.experiment_ledger import LedgerBuild

from .load import validate_record_identity
from .model import HubRecord, HubSources

QUESTION_STATUSES = frozenset({"open", "partially_answered", "answered", "parked"})
CONFIDENCE_LEVELS = frozenset({"low", "moderate", "high"})
STUDY_TYPES = frozenset(
    {
        "ranking_experiment",
        "audit",
        "data_quality",
        "reference_review",
        "feasibility",
        "evidence_synthesis",
    }
)
STUDY_STATUSES = frozenset({"planned", "running", "concluded", "blocked"})
STUDY_OUTCOMES = frozenset(
    {"positive", "negative", "mixed", "inconclusive", "invalid", "unscored", "pending"}
)
DECISIONS = frozenset(
    {
        "adopt",
        "retain_control",
        "reject",
        "stop",
        "iterate",
        "revisit_on_trigger",
        "informational",
        "pending",
    }
)
TEST_STAGES = frozenset({"idea", "ready", "running", "completed", "parked"})
RATIONALE_PROVENANCE = frozenset({"contemporaneous", "reconstructed", "unknown"})
DATA_SENSITIVITY = frozenset({"public", "project_internal", "restricted", "secret"})
DATA_SHARING = frozenset(
    {"repository", "local_only", "external_reference", "metadata_only"}
)

DATA_CATEGORIES = frozenset({"provided", "collected", "generated", "external"})
BACKGROUND_TYPES = frozenset(
    {
        "literature_review",
        "participant_review",
        "benchmark",
        "methods_landscape",
    }
)

READY_FIELDS = (
    "hypothesis",
    "rationale",
    "expected_signal",
    "decision_rule",
    "dependencies",
    "priority",
    "evaluation_boundary",
    "artifact_plan",
    "next_action",
)


@dataclass(frozen=True)
class ValidationSummary:
    record_counts: Mapping[str, int]
    linked_system_count: int


def require_string(record: HubRecord, field: str) -> str:
    value = record.metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record.source_path}: {field} must be a nonempty string")
    return value.strip()


def require_string_list(record: HubRecord, field: str) -> tuple[str, ...]:
    value = record.metadata.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{record.source_path}: {field} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{record.source_path}: {field} contains duplicates")
    return tuple(value)


def require_choice(record: HubRecord, field: str, allowed: frozenset[str]) -> str:
    value = require_string(record, field)
    if value not in allowed:
        raise ValueError(
            f"{record.source_path}: {field} must be one of {sorted(allowed)}"
        )
    return value


def require_date_or_unknown(record: HubRecord, field: str) -> date | str:
    value = record.metadata.get(field)
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value == "unknown":
            return value
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"{record.source_path}: {field} must be an ISO date or unknown")


def validate_repo_reference(root: Path, record: HubRecord, value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{record.source_path}: evidence path must be repository-relative: {value}"
        )
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{record.source_path}: evidence path escapes repository: {value}"
        ) from exc
    if not resolved.exists():
        raise ValueError(
            f"{record.source_path}: referenced evidence does not exist: {value}"
        )


def _require_references(root: Path, record: HubRecord, field: str) -> tuple[str, ...]:
    references = require_string_list(record, field)
    for reference in references:
        validate_repo_reference(root, record, reference)
    return references


def _require_known_ids(
    record: HubRecord,
    field: str,
    known_ids: set[str],
) -> tuple[str, ...]:
    values = require_string_list(record, field)
    unknown = _unknown_ids(values, known_ids)
    if unknown:
        raise ValueError(
            f"{record.source_path}: {field} references unknown id(s): {sorted(unknown)}"
        )
    return values


def _unknown_ids(values: Sequence[str], known_ids: set[str]) -> set[str]:
    return set(values) - known_ids


def validate_question(root: Path, record: HubRecord) -> None:
    require_string(record, "primary_theme_id")
    require_string_list(record, "theme_ids")
    require_string(record, "plain_english")
    require_choice(record, "status", QUESTION_STATUSES)
    require_string(record, "current_answer")
    require_choice(record, "confidence", CONFIDENCE_LEVELS)
    require_string(record, "confidence_boundary")
    require_string_list(record, "study_ids")
    require_string_list(record, "open_subquestions")
    require_string_list(record, "next_test_ids")
    require_date_or_unknown(record, "updated_on")
    evidence_refs = _require_references(root, record, "evidence_refs")
    if not evidence_refs:
        raise ValueError(f"{record.source_path}: evidence_refs must not be empty")


def validate_study(root: Path, record: HubRecord) -> None:
    require_string(record, "primary_theme_id")
    require_string_list(record, "theme_ids")
    study_type = require_choice(record, "study_type", STUDY_TYPES)
    question_ids = require_string_list(record, "question_ids")
    if not question_ids:
        raise ValueError(f"{record.source_path}: question_ids must not be empty")
    require_string_list(record, "approach_ids")
    status = require_choice(record, "status", STUDY_STATUSES)
    outcome = require_choice(record, "outcome", STUDY_OUTCOMES)
    require_string(record, "objective")
    if study_type == "ranking_experiment":
        require_string(record, "hypothesis")
    elif record.metadata.get("hypothesis") is not None:
        require_string(record, "hypothesis")
    require_string(record, "rationale")
    provenance = require_choice(record, "rationale_provenance", RATIONALE_PROVENANCE)
    require_string(record, "protocol_summary")
    require_string(record, "conclusion")
    decision = require_choice(record, "decision", DECISIONS)
    require_string(record, "decision_summary")
    require_string_list(record, "system_ids")
    require_string_list(record, "representative_system_ids")
    evidence_refs = _require_references(root, record, "evidence_refs")
    require_string_list(record, "dataset_ids")
    require_date_or_unknown(record, "started_on")
    if "concluded_on" not in record.metadata:
        raise ValueError(f"{record.source_path}: concluded_on is required")
    concluded_on = record.metadata["concluded_on"]
    if concluded_on is None:
        if status == "concluded":
            raise ValueError(
                f"{record.source_path}: concluded_on cannot be null for concluded study"
            )
    else:
        require_date_or_unknown(record, "concluded_on")
    require_string_list(record, "supersedes")
    require_string_list(record, "superseded_by")
    require_string(record, "limitations")
    if provenance == "reconstructed" and not evidence_refs:
        raise ValueError(
            f"{record.source_path}: reconstructed rationale requires evidence_refs"
        )
    if decision != "pending" and not evidence_refs:
        raise ValueError(f"{record.source_path}: decision requires evidence_refs")
    if status == "concluded":
        if outcome == "pending":
            raise ValueError(
                f"{record.source_path}: concluded study cannot have pending outcome"
            )
        if decision == "pending":
            raise ValueError(
                f"{record.source_path}: concluded study cannot have pending decision"
            )
    elif outcome != "pending" or decision != "pending":
        raise ValueError(
            f"{record.source_path}: unfinished study requires pending outcome and decision"
        )


def validate_test(record: HubRecord) -> None:
    require_string(record, "primary_theme_id")
    require_string_list(record, "theme_ids")
    stage = require_choice(record, "stage", TEST_STAGES)
    require_string(record, "summary")
    require_string_list(record, "linked_question_ids")
    missing_readiness_fields = require_string_list(record, "missing_readiness_fields")
    unknown_fields = _unknown_ids(missing_readiness_fields, set(READY_FIELDS))
    if unknown_fields:
        raise ValueError(
            f"{record.source_path}: missing_readiness_fields contains unknown field(s): "
            f"{sorted(unknown_fields)}"
        )
    if stage in {"ready", "running", "completed"}:
        for field in READY_FIELDS:
            require_string(record, field)
        if missing_readiness_fields:
            raise ValueError(
                f"{record.source_path}: ready test has missing_readiness_fields"
            )
    if stage == "completed":
        require_string(record, "resulting_study_id")


def validate_dataset(root: Path, record: HubRecord) -> None:
    require_string(record, "plain_english")
    require_choice(record, "category", DATA_CATEGORIES)
    require_string(record, "provenance")
    require_string(record, "method")
    require_string(record, "version")
    sharing = require_choice(record, "sharing", DATA_SHARING)
    repository_path = record.metadata.get("repository_path")
    if sharing == "repository":
        if not isinstance(repository_path, str) or not repository_path.strip():
            raise ValueError(
                f"{record.source_path}: repository_path must be a nonempty string"
            )
        validate_repo_reference(root, record, repository_path)
    elif repository_path is not None:
        require_string(record, "repository_path")
        validate_repo_reference(root, record, repository_path)
    require_choice(record, "sensitivity", DATA_SENSITIVITY)
    require_string(record, "leakage_constraints")
    require_string(record, "permitted_use")
    require_string_list(record, "study_ids")
    require_string(record, "availability")
    require_string(record, "reproducibility")
    if record.metadata.get("representative_sample") is not None:
        require_string(record, "representative_sample")


def validate_background(root: Path, record: HubRecord) -> None:
    require_choice(record, "resource_type", BACKGROUND_TYPES)
    require_string(record, "summary")
    source = require_string(record, "canonical_source_path")
    validate_repo_reference(root, record, source)
    require_string_list(record, "citations")
    require_string_list(record, "tags")
    require_string_list(record, "question_ids")
    require_date_or_unknown(record, "reviewed_on")


def validate_theme(root: Path, record: HubRecord) -> None:
    require_string(record, "plain_english")
    require_string(record, "current_conclusion")
    require_choice(record, "decision", DECISIONS)
    require_choice(record, "confidence", CONFIDENCE_LEVELS)
    require_string(record, "confidence_boundary")
    require_string_list(record, "next_test_ids")
    require_date_or_unknown(record, "updated_on")
    evidence_refs = _require_references(root, record, "evidence_refs")
    if not evidence_refs:
        raise ValueError(f"{record.source_path}: evidence_refs must not be empty")


def _validate_approach(record: HubRecord) -> None:
    require_string(record, "summary")
    require_string_list(record, "question_ids")
    require_string_list(record, "study_ids")


def _validate_relationships(
    sources: HubSources,
    ledger: LedgerBuild,
) -> int:
    theme_ids = {row.record_id for row in sources.themes}
    question_ids = {row.record_id for row in sources.questions}
    study_ids = {row.record_id for row in sources.studies}
    test_ids = {row.record_id for row in sources.tests}
    approach_ids = {row.record_id for row in sources.approaches}
    dataset_ids = {row.record_id for row in sources.datasets}
    system_scopes = {row.system_id: row.evaluation_scope for row in ledger.systems}
    system_ids = set(system_scopes)

    for record in sources.themes:
        _require_known_ids(record, "next_test_ids", test_ids)
    for record in (*sources.questions, *sources.studies, *sources.tests):
        primary_theme_id = require_string(record, "primary_theme_id")
        if primary_theme_id not in theme_ids:
            raise ValueError(
                f"{record.source_path}: primary_theme_id references unknown id: "
                f"{primary_theme_id}"
            )
        linked_themes = _require_known_ids(record, "theme_ids", theme_ids)
        if primary_theme_id not in linked_themes:
            raise ValueError(
                f"{record.source_path}: primary_theme_id must occur in theme_ids"
            )

    for record in sources.questions:
        _require_known_ids(record, "study_ids", study_ids)
        _require_known_ids(record, "next_test_ids", test_ids)
    for record in sources.studies:
        _require_known_ids(record, "question_ids", question_ids)
        _require_known_ids(record, "approach_ids", approach_ids)
        linked_systems = _require_known_ids(record, "system_ids", system_ids)
        representatives = _require_known_ids(
            record, "representative_system_ids", system_ids
        )
        if not set(representatives) <= set(linked_systems):
            raise ValueError(
                f"{record.source_path}: representative_system_ids must occur in system_ids"
            )
        if record.metadata.get("comparison_system_ids") is not None:
            comparison_systems = _require_known_ids(
                record, "comparison_system_ids", system_ids
            )
            scopes = {system_scopes[system_id] for system_id in comparison_systems}
            if len(scopes) > 1:
                raise ValueError(
                    f"{record.source_path}: comparison_system_ids span more than one evaluation scope"
                )
        _require_known_ids(record, "dataset_ids", dataset_ids)
        _require_known_ids(record, "supersedes", study_ids)
        _require_known_ids(record, "superseded_by", study_ids)
    question_study_links = {
        (record.record_id, study_id)
        for record in sources.questions
        for study_id in require_string_list(record, "study_ids")
    }
    study_question_links = {
        (question_id, record.record_id)
        for record in sources.studies
        for question_id in require_string_list(record, "question_ids")
    }
    if question_study_links != study_question_links:
        missing_from_questions = sorted(study_question_links - question_study_links)
        missing_from_studies = sorted(question_study_links - study_question_links)
        raise ValueError(
            "question-study relationships must be reciprocal; "
            f"missing from question study_ids: {missing_from_questions}; "
            f"missing from study question_ids: {missing_from_studies}"
        )
    for record in sources.tests:
        _require_known_ids(record, "linked_question_ids", question_ids)
        if record.metadata.get("stage") == "completed":
            resulting_study = require_string(record, "resulting_study_id")
            if resulting_study not in study_ids:
                raise ValueError(
                    f"{record.source_path}: resulting_study_id references unknown id: {resulting_study}"
                )
    for record in sources.approaches:
        _require_known_ids(record, "question_ids", question_ids)
        _require_known_ids(record, "study_ids", study_ids)
    for record in sources.datasets:
        _require_known_ids(record, "study_ids", study_ids)
    for record in sources.background:
        _require_known_ids(record, "question_ids", question_ids)
    linked_system_ids = {
        system_id
        for record in sources.studies
        for system_id in require_string_list(record, "system_ids")
    }
    orphan_system_ids = sorted(system_ids - linked_system_ids)
    if orphan_system_ids:
        raise ValueError(
            "orphan ledger run(s) must be linked from at least one canonical "
            f"study system_ids list: {orphan_system_ids}"
        )
    return len(linked_system_ids)


def validate_hub_sources(
    sources: HubSources,
    ledger: LedgerBuild,
    root: Path,
) -> ValidationSummary:
    records = sources.all_records()
    owners: dict[tuple[str, str], str] = {}
    for record in records:
        validate_record_identity(record)
        key = (record.kind, record.record_id)
        if key in owners:
            raise ValueError(
                f"duplicate {record.kind} id {record.record_id!r}: "
                f"{owners[key]} and {record.source_path}"
            )
        owners[key] = record.source_path

    for record in sources.themes:
        validate_theme(root, record)
    for record in sources.questions:
        validate_question(root, record)
    for record in sources.studies:
        validate_study(root, record)
    for record in sources.tests:
        validate_test(record)
    for record in sources.approaches:
        _validate_approach(record)
    for record in sources.datasets:
        validate_dataset(root, record)
    for record in sources.background:
        validate_background(root, record)

    return ValidationSummary(
        record_counts={
            "theme": len(sources.themes),
            "question": len(sources.questions),
            "study": len(sources.studies),
            "test": len(sources.tests),
            "approach": len(sources.approaches),
            "dataset": len(sources.datasets),
            "background": len(sources.background),
        },
        linked_system_count=_validate_relationships(sources, ledger),
    )
