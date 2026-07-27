"""Deterministic renderers and atomic publication for research-hub views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import html
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

from trec_mlm.experiment_ledger import SystemRecord

from .build import HubBuild
from .load import validate_record_identity
from .model import HubRecord
from .validate import READY_FIELDS, github_issue_number

_QUESTION_STATUS_PRIORITY = {
    "open": 0,
    "partially_answered": 1,
    "parked": 2,
    "answered": 3,
}
_TEST_STAGE_PRIORITY = {
    "running": 0,
    "ready": 1,
    "idea": 2,
    "parked": 3,
    "completed": 4,
}
_STATIC_OUTPUT_PATHS = (
    Path("reports/trec_2026_research_hub_summary.md"),
    Path("research/catalog.json"),
    Path("research/hub/generated/coverage.qmd"),
    Path("research/hub/generated/current-state.qmd"),
    Path("research/hub/generated/failure-library.qmd"),
    Path("research/hub/generated/background-index.qmd"),
    Path("research/hub/generated/data-index.qmd"),
    Path("research/hub/generated/history.qmd"),
    Path("research/hub/generated/ideas-index.qmd"),
    Path("research/hub/generated/questions-index.qmd"),
    Path("research/hub/generated/runs.qmd"),
    Path("research/hub/generated/studies-index.qmd"),
    Path("research/hub/generated/summary.qmd"),
    Path("research/hub/generated/themes-index.qmd"),
)
_RESTRICTED_SCOPES = frozenset({"restricted_12x20", "restricted_6", "invalid_unscored"})
_SAFE_SYSTEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@+-]*\Z")


@dataclass(frozen=True)
class _NextAction:
    """A context-free summary of the first deterministic linked test."""

    record_id: str | None
    title: str | None
    text: str


def _yaml_scalar(value: object) -> str:
    """Serialize a string as a quoted YAML-compatible scalar."""

    return json.dumps(str(value), ensure_ascii=False)


def _markdown_table_text(value: object) -> str:
    """Escape untrusted text for a Markdown table cell."""

    flattened = " ".join(str(value).split())
    escaped = html.escape(flattened, quote=True)
    return escaped.replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|")


def _inline_code(value: object) -> str:
    """Render text as a Markdown inline-code span with a safe delimiter."""

    flattened = " ".join(str(value).split())
    runs = [len(run) for run in re.findall(r"`+", flattened)]
    delimiter = "`" * (max(runs, default=0) + 1)
    return f"{delimiter}{flattened}{delimiter}"


def _path_code(value: object) -> str:
    """Render a repository path that can wrap safely on narrow screens."""

    flattened = " ".join(str(value).split())
    return f'<code class="hub-path">{html.escape(flattened)}</code>'


def _validate_system_id(system_id: str) -> str:
    """Require a ledger ID that is exactly one portable filename component."""

    if (
        not isinstance(system_id, str)
        or system_id in {".", ".."}
        or not _SAFE_SYSTEM_ID.fullmatch(system_id)
    ):
        raise ValueError(
            f"system_id must be one safe filename component: {system_id!r}"
        )
    return system_id


def _validate_output_path(path: Path) -> Path:
    """Require a normalized repository-relative output path."""

    path = Path(path)
    if (
        path.is_absolute()
        or not path.parts
        or path == Path(".")
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in path.as_posix()
    ):
        raise ValueError(f"output path must be normalized repository-relative: {path}")
    return path


def _confined_destination(root: Path, relative_path: Path) -> Path:
    """Resolve a destination and reject symlink or lexical repository escapes."""

    relative_path = _validate_output_path(relative_path)
    destination = root / relative_path
    try:
        destination.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"output path must remain repository-relative: {relative_path}"
        ) from exc
    return destination


def _validated_output_mapping(
    outputs: Mapping[Path, str],
) -> dict[Path, str]:
    validated: dict[Path, str] = {}
    for raw_path, content in outputs.items():
        path = _validate_output_path(raw_path)
        if path in validated:
            raise ValueError(f"duplicate normalized output path: {path}")
        if not isinstance(content, str):
            raise TypeError(f"output content must be text: {path}")
        validated[path] = content
    return validated


def _metadata_list(record: HubRecord, field: str) -> tuple[str, ...]:
    value = record.metadata.get(field, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _related_ids(record: HubRecord) -> list[str]:
    fields = (
        "theme_ids",
        "linked_question_ids",
        "question_ids",
        "study_ids",
        "approach_ids",
        "dataset_ids",
    )
    return sorted(
        {value for field in fields for value in _metadata_list(record, field)}
    )


def _catalog_related_ids(build: HubBuild, record: HubRecord) -> list[str]:
    related = set(_related_ids(record))
    if record.kind == "theme":
        related.update(
            test.record_id for test in _tests_for_theme(build, record.record_id)
        )
    elif record.kind == "question":
        related.update(
            test.record_id for test in _tests_for_questions(build, (record.record_id,))
        )
    elif record.kind == "study":
        related.update(
            test.record_id
            for test in _tests_for_questions(
                build,
                _metadata_list(record, "question_ids"),
            )
        )
    return sorted(related)


def _rendered_path(record: HubRecord) -> str:
    directory = {
        "theme": "generated/themes",
        "question": "questions",
        "study": "studies",
        "test": "generated/ideas",
        "approach": "approaches",
        "dataset": "data",
        "background": "background",
    }[record.kind]
    return f"{directory}/{record.record_id}.html"


def render_catalog(build: HubBuild) -> str:
    """Render the stable, machine-readable search catalog."""

    for record in build.sources.all_records():
        validate_record_identity(record)
    resources = [
        {
            "id": record.record_id,
            "type": record.kind,
            "title": record.title,
            "summary": str(
                record.metadata.get("plain_english")
                or record.metadata.get("summary")
                or record.metadata.get("objective")
                or ""
            ),
            "tags": sorted(_metadata_list(record, "tags")),
            "canonical_path": record.source_path,
            "rendered_path": _rendered_path(record),
            "related_ids": _catalog_related_ids(build, record),
        }
        for record in build.sources.all_records()
    ]
    for system in build.ledger.systems:
        system_id = _validate_system_id(system.system_id)
        study_ids = _linked_study_ids(build, system_id)
        resources.append(
            {
                "id": system_id,
                "type": "run",
                "title": system_id,
                "summary": system.method_summary,
                "tags": sorted(
                    {
                        system.experiment_family,
                        system.evaluation_scope,
                        system.promotion_status,
                        system.evidence_status,
                    }
                ),
                "canonical_path": "experiments/ledger.json",
                "rendered_path": f"generated/runs/{system_id}.html",
                "related_ids": study_ids,
            }
        )
    resources.sort(key=lambda row: (row["type"], row["id"]))
    return (
        json.dumps(
            {"version": 1, "resources": resources},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _summary_current_position(build: HubBuild) -> list[str]:
    lines = ["", "## Current position", ""]
    studies = sorted(
        (
            study
            for study in build.sources.studies
            if study.metadata.get("decision") != "pending"
            and not _metadata_list(study, "superseded_by")
        ),
        key=lambda row: (
            *_date_sort_value(row.metadata.get("concluded_on")),
            row.record_id,
        ),
    )[:3]
    if not studies:
        return [*lines, "No decisions have been recorded yet."]
    for study in studies:
        decision = _inline_code(study.metadata.get("decision", "unknown"))
        summary = _markdown_table_text(study.metadata.get("decision_summary", ""))
        lines.append(
            f"- **{_markdown_table_text(study.title)}** "
            f"({_inline_code(study.record_id)}; {decision}): {summary}"
        )
    return lines


def _ordered_questions(build: HubBuild) -> list[HubRecord]:
    return sorted(
        (
            question
            for question in build.sources.questions
            if question.metadata.get("status") in {"open", "partially_answered"}
        ),
        key=lambda row: (
            *_date_sort_value(row.metadata.get("updated_on")),
            _QUESTION_STATUS_PRIORITY.get(
                str(row.metadata.get("status")),
                len(_QUESTION_STATUS_PRIORITY),
            ),
            row.record_id,
        ),
    )


def _summary_open_questions(build: HubBuild) -> list[str]:
    lines = ["", "## Open questions", ""]
    questions = _ordered_questions(build)[:3]
    if not questions:
        return [*lines, "No research questions have been recorded."]
    for question in questions:
        status = _inline_code(question.metadata.get("status", "unknown"))
        confidence = _inline_code(question.metadata.get("confidence", "unknown"))
        answer = _markdown_table_text(question.metadata.get("current_answer", ""))
        boundary = _markdown_table_text(
            question.metadata.get("confidence_boundary", "")
        )
        lines.append(
            f"- **{_markdown_table_text(question.title)}** "
            f"({_inline_code(question.record_id)}; {status}; confidence {confidence}): "
            f"{answer} **Confidence boundary:** {boundary}"
        )
    return lines


def _summary_next_tests(build: HubBuild) -> list[str]:
    lines = ["", "## Next tests", ""]
    stage_priority = {"running": 0, "ready": 1, "idea": 2}
    priority = {"high": 0, "medium": 1, "low": 2}
    tests = sorted(
        (
            test
            for test in build.sources.tests
            if test.metadata.get("stage") in stage_priority
        ),
        key=lambda row: (
            priority.get(str(row.metadata.get("priority")), len(priority)),
            stage_priority[str(row.metadata.get("stage"))],
            row.record_id,
        ),
    )[:3]
    if not tests:
        return [*lines, "No active tests have been recorded."]
    for test in tests:
        stage = _inline_code(test.metadata.get("stage", "unknown"))
        test_priority = _inline_code(test.metadata.get("priority", "unspecified"))
        summary = _markdown_table_text(test.metadata.get("summary", ""))
        lines.append(
            f"- **{_markdown_table_text(test.title)}** "
            f"({_inline_code(test.record_id)}; {stage}; priority {test_priority}): "
            f"{summary}"
        )
    return lines


def _summary_navigation(*, generated_qmd: bool) -> list[str]:
    if generated_qmd:
        summary = "generated/summary.qmd"
        coverage = "evidence/coverage.qmd"
        history = "evidence/history.qmd"
        runs = "evidence/runs.qmd"
        catalog = "catalog.json"
    else:
        summary = "../research/hub/generated/summary.qmd"
        coverage = "../research/hub/generated/coverage.qmd"
        history = "../research/hub/generated/history.qmd"
        runs = "../research/hub/generated/runs.qmd"
        catalog = "../research/catalog.json"
    return [
        "",
        "## Navigation",
        "",
        f"- [Generated summary]({summary})",
        f"- [Research question coverage]({coverage})",
        f"- [Decision history]({history})",
        f"- [Experiment runs]({runs})",
        f"- [Machine-readable catalog]({catalog})",
    ]


def render_summary_markdown(build: HubBuild) -> str:
    """Render the repository-facing executive summary."""

    lines = ["# TREC 2026 Research Hub", "", "Development evidence only."]
    lines.extend(_summary_current_position(build))
    lines.extend(_summary_open_questions(build))
    lines.extend(_summary_next_tests(build))
    lines.extend(_summary_navigation(generated_qmd=False))
    return "\n".join(lines).rstrip() + "\n"


def _diagnostic_count(build: HubBuild, key: str) -> int | None:
    bucket = build.failure_diagnostic.get(key)
    if bucket is None:
        return None
    count = bucket.get("count") if isinstance(bucket, Mapping) else None
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(
            f"failure diagnostic {key}.count must be a nonnegative integer"
        )
    return count


def _failure_library_cohorts(build: HubBuild) -> tuple[tuple[int, str], tuple[int, str]]:
    """Return the validated diagnostic cohorts used by the failure library."""

    fusion = build.failure_diagnostic.get("fusion_21")
    no_signal = build.failure_diagnostic.get("no_signal_54")
    fusion_count = _diagnostic_count(build, "fusion_21")
    no_signal_count = _diagnostic_count(build, "no_signal_54")
    if (
        fusion_count is None
        or no_signal_count is None
        or not isinstance(fusion, Mapping)
        or not isinstance(no_signal, Mapping)
    ):
        raise ValueError("failure diagnostic must provide both documented cohorts")

    def definition(bucket: Mapping[str, object], key: str) -> str:
        value = bucket.get("definition")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"failure diagnostic {key}.definition must be nonempty")
        return value.strip()

    return (
        (fusion_count, definition(fusion, "fusion_21")),
        (no_signal_count, definition(no_signal, "no_signal_54")),
    )


def _render_failure_library_record(
    record_id: str,
    title: str,
    fields: Sequence[tuple[str, str]],
) -> list[str]:
    """Render one privacy-safe failure mechanism with a stable field contract."""

    lines = [
        f'<section class="hub-failure-record" data-failure-mechanism="{record_id}">',
        f"<h2>{html.escape(title)}</h2>",
        '<dl class="hub-failure-fields">',
    ]
    for label, value in fields:
        lines.extend([f"<dt>{html.escape(label)}</dt>", f"<dd>{value}</dd>"])
    lines.extend(["</dl>", "</section>"])
    return lines


def render_failure_library_qmd(build: HubBuild) -> str:
    """Render the privacy-safe failure-mechanism evidence include."""

    if not build.failure_diagnostic:
        return (
            "<!-- Generated by scripts/build_research_hub.py; do not edit. -->\n\n"
            "## Evidence-backed failure mechanisms\n\n"
            "No validated failure diagnostic is available for this build. Do not infer "
            "failure mechanisms or prevalence from this placeholder.\n"
        )
    (fusion_count, fusion_definition), (no_signal_count, no_signal_definition) = (
        _failure_library_cohorts(build)
    )
    lines = [
        "<!-- Generated by scripts/build_research_hub.py; do not edit. -->",
        "",
        "## Evidence-backed failure mechanisms",
        "",
        "These selected cases explain mechanisms, not benchmark prevalence. They do not "
        "render query text, candidate responses, private identifiers, raw artifacts, "
        "or model rationale.",
        "",
    ]
    lines.extend(
        _render_failure_library_record(
            "fusion-dilution",
            "Fusion dilution",
            (
                ("What failed", f"{fusion_count} development failures: {html.escape(fusion_definition)}"),
                (
                    "Evidence available",
                    "The canonical development-only diagnostic separates the two "
                    "promoted component rankings from their equal reciprocal-rank fusion.",
                ),
                (
                    "Response or change",
                    "Do not retune a static mixture post hoc; test a preregistered "
                    "preservation gate or a bounded adaptive rule against the frozen control.",
                ),
                (
                    "Observed outcome",
                    "A component's top-ten signal can be moved out of the fused top ten.",
                ),
                (
                    "Possible failure location",
                    "The equal reciprocal-rank fusion step after component rankings are formed.",
                ),
                (
                    "Competing explanation",
                    "The component signal may be incidental or insufficiently robust, rather than "
                    "fusion alone being the cause.",
                ),
                (
                    "Confirmed defect, audit gap, or uncertainty",
                    "Confirmed diagnostic mechanism in the selected development cohort; it is not "
                    "a hidden-test defect finding.",
                ),
                (
                    "Evidence limit",
                    "Post-hoc development audit with unavailable target responses and assessor rationale.",
                ),
                (
                    "Next falsifying test",
                    '<a href="../generated/ideas/failure-aware-fusion.qmd">Failure-aware evidence fusion</a> '
                    "must beat the frozen dense/profile control under a fixed decision rule.",
                ),
                (
                    "Source study and evidence status",
                    '<a href="../studies/sparse-failure-analogue-diagnosis.qmd">'
                    "Sparse-failure analogue diagnosis</a> — concluded development-only diagnostic.",
                ),
            ),
        )
    )
    lines.extend(
        [
            "",
            *_render_failure_library_record(
                "no-component-signal",
                "No component signal",
                (
                    (
                        "What failed",
                        f"{no_signal_count} development failures: {html.escape(no_signal_definition)}",
                    ),
                    (
                        "Evidence available",
                        "The canonical development-only diagnostic shows neither promoted component "
                        "supplied a top-ten signal.",
                    ),
                    (
                        "Response or change",
                        "Do not reweight the same lists as a recovery; introduce a new, frozen evidence "
                        "source or stop the proposed change.",
                    ),
                    (
                        "Observed outcome",
                        "Fusion cannot recover a relevant candidate absent from both component top tens.",
                    ),
                    (
                        "Possible failure location",
                        "Upstream retrieval or candidate-side evidence construction before fusion.",
                    ),
                    (
                        "Competing explanation",
                        "Historical analogues may be too weak, or the available component representations "
                        "may omit the needed expertise signal.",
                    ),
                    (
                        "Confirmed defect, audit gap, or uncertainty",
                        "Confirmed missing component signal in the selected development cohort; the missing "
                        "evidence source is still uncertain.",
                    ),
                    (
                        "Evidence limit",
                        "Development-only post-hoc diagnosis does not reveal the target response or establish "
                        "hidden-test frequency.",
                    ),
                    (
                        "Next falsifying test",
                        '<a href="../generated/ideas/semantic-failure-clusters.qmd">Semantic failure clusters</a> '
                        "must identify a reproducible new signal beyond the frozen components.",
                    ),
                    (
                        "Source study and evidence status",
                        '<a href="../studies/sparse-failure-analogue-diagnosis.qmd">'
                        "Sparse-failure analogue diagnosis</a> — concluded development-only diagnostic.",
                    ),
                ),
            ),
            "",
            *_render_failure_library_record(
                "unreliable-generated-evidence",
                "Unreliable generated evidence",
                (
                    (
                        "What failed",
                        "Generated reference evidence can be ambiguous, unsupported, or insufficiently "
                        "grounded for an automatic ranking decision.",
                    ),
                    (
                        "Evidence available",
                        "Selected grounding, corpus-quality, and blinded reference-review audits identify "
                        "uncertainty, provenance, and disagreement risks.",
                    ),
                    (
                        "Response or change",
                        "Use explicit uncertainty and review gates; develop a reproducible disagreement "
                        "trigger instead of grounding every request.",
                    ),
                    (
                        "Observed outcome",
                        "Grounding and review are useful safeguards, but not reliable blanket cures or "
                        "validated ground truth.",
                    ),
                    (
                        "Possible failure location",
                        "Reference generation, Search grounding, source support, or the trigger that decides "
                        "when review is required.",
                    ),
                    (
                        "Competing explanation",
                        "Reviewer noise or purposive sampling may explain part of the observed disagreement.",
                    ),
                    (
                        "Confirmed defect, audit gap, or uncertainty",
                        "An audit gap and reliability uncertainty remain; no universal generated-evidence defect "
                        "has been established.",
                    ),
                    (
                        "Evidence limit",
                        "No prevalence count is available. The audits use selected or purposive samples and do "
                        "not estimate a population rate.",
                    ),
                    (
                        "Next falsifying test",
                        '<a href="../generated/ideas/reference-review-trigger.qmd">Reference review trigger</a> '
                        "must show that a frozen trigger improves reliability without treating all generated "
                        "evidence as false.",
                    ),
                    (
                        "Source study and evidence status",
                        '<a href="../studies/gemini-grounding-failure-review.qmd">Gemini grounding failure review</a>, '
                        '<a href="../studies/grounded-data-quality-audit.qmd">grounded reference data-quality audit</a>, and '
                        '<a href="../studies/four-model-reference-review.qmd">four-model reference-answer review</a> '
                        "— concluded selected-audit and purposive-review evidence.",
                    ),
                ),
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


_CURRENT_STATE_LADDER_SPECS = (
    ("dense_nested", "Dense query baseline", "dev_322x150", "stable_control"),
    ("contextual_e5", "Contextual E5 control", "dev_322x150", "stable_control"),
    ("local_domain_prior", "Local-domain prior", "dev_322x150", "valid_negative"),
    (
        "response_sumscore_nested",
        "Response sum-score",
        "dev_322x150",
        "valid_negative",
    ),
    ("profile_only", "Profile-only control", "dev_322x150", "stable_control"),
    (
        "dense_profile_equal",
        "Dense/profile equal",
        "dev_322x150",
        "promoted_development",
    ),
    (
        "dense_profile_profile_heavy",
        "Profile-heavy diagnostic",
        "dev_322x150",
        "diagnostic_only",
    ),
)

_CURRENT_STATE_METHOD_FAMILY_SPECS = (
    (
        "query-baselines-and-neighbors",
        "Query baselines and neighbors",
        "Use historical requests to establish a query-only control before adding candidate-side evidence.",
        "Dense query neighbors remain the stable query-only control; tested cluster aggregation did not replace it.",
        "dense-query-knn",
        "Which request patterns need evidence beyond similar historical queries?",
        "semantic-failure-clusters",
        ("query-controls", "dense-query-knn", "query-cluster-aggregation"),
    ),
    (
        "candidate-labels-and-learned-scorers",
        "Candidate labels and learned scorers",
        "Use candidate labels as a lightweight clue about fit, and test learned scorers only under a comparable protocol.",
        "Contextual E5 remains a control; the tested learned-scorer pilots are not reusable in their tested forms.",
        "candidate-label-matching",
        "Can a redesigned learned scorer improve without inheriting the pilots' restricted or invalid scope?",
        "semantic-failure-clusters",
        ("candidate-label-matching", "label-ranker-pilots"),
    ),
    (
        "candidate-profiles-and-fusion",
        "Candidate profiles and fusion",
        "Combine qrel-free candidate profiles with a frozen query control to express complementary candidate-side evidence.",
        "Equal dense/profile fusion is the current promoted development direction; alternative weighting is diagnostic only.",
        "semantic-profiles",
        "Does the promoted direction generalize when the evaluation protocol is frozen for unseen queries?",
        "semantic-failure-clusters",
        ("behavior-profile", "multiresolution-profiles", "semantic-profiles"),
    ),
    (
        "direct-response-evidence",
        "Direct response evidence",
        "Compare bounded candidate-response evidence with the control without disclosing response content.",
        "The tested lexical and source-alignment response routes did not displace the control in their tested forms.",
        "response-retrieval",
        "What privacy-safe response signal can add evidence where both promoted components have no top-ten signal?",
        "failure-aware-fusion",
        (
            "response-retrieval",
            "response-volume-normalization",
            "source-query-first",
        ),
    ),
    (
        "references-and-bounded-refinement",
        "References and bounded refinement",
        "Use reviewed generated references and bounded refinement as separate evidence routes, not registered ranking rows.",
        "A reviewed reference default is useful, while the E5 follow-up remains informational, outside the ledger, and unpromoted.",
        "four-model-reference-review",
        "When is generated reference evidence reliable enough to influence a frozen ranking test?",
        "reference-review-trigger",
        ("four-model-reference-review", "e5-raw-response-refinement"),
    ),
    (
        "reliability-failures-and-readiness",
        "Reliability, failures, and readiness",
        "Use audits and failure diagnoses to identify when a ranking signal needs a new test, a safeguard, or a stop decision.",
        "Known failures show both fusion dilution and missing component signal; audits do not estimate hidden-test prevalence.",
        "sparse-failure-analogue-diagnosis",
        "Which failure mechanisms repeat under a frozen, privacy-safe evaluation boundary?",
        "semantic-failure-clusters",
        (
            "gemini-grounding-failure-review",
            "grounded-data-quality-audit",
            "sparse-failure-analogue-diagnosis",
        ),
    ),
)

_CURRENT_STATE_STUDY_STATUS_LABELS = {
    "planned": "Planned",
    "running": "Running",
    "concluded": "Concluded",
}

_CURRENT_STATE_PROMOTION_LABELS = {
    "stable_control": "Stable control",
    "valid_negative": "Valid negative",
    "promoted_development": "Promoted development",
    "diagnostic_only": "Diagnostic only",
}

_DECISION_LABELS = {
    "adopt": "Adopted",
    "informational": "Informational",
    "iterate": "Iterate",
    "reject": "Rejected",
    "retain_control": "Retained control",
    "stop": "Stopped",
}


def _current_state_ladder_rows(
    build: HubBuild,
) -> tuple[tuple[str, SystemRecord], ...]:
    """Return the fixed registered ladder after validating identity and status."""

    by_id = {system.system_id: system for system in build.ledger.systems}
    rows: list[tuple[str, SystemRecord]] = []
    for system_id, label, evaluation_scope, promotion_status in (
        _CURRENT_STATE_LADDER_SPECS
    ):
        system = by_id.get(system_id)
        if system is None:
            raise ValueError(f"current-state ladder requires registered {system_id}")
        if system.evaluation_scope != evaluation_scope:
            raise ValueError(
                f"{system_id} must remain {evaluation_scope}, "
                f"not {system.evaluation_scope}"
            )
        if system.promotion_status != promotion_status:
            raise ValueError(
                f"{system_id} must remain {promotion_status}, "
                f"not {system.promotion_status}"
            )
        rows.append((label, system))
    return tuple(rows)


def _current_state_theme_rows(
    build: HubBuild,
) -> tuple[
    tuple[str, str, str, str, HubRecord, str, HubRecord, tuple[HubRecord, ...]], ...
]:
    """Validate and materialize the presentation-only authored-study partition."""

    by_id = {study.record_id: study for study in build.sources.studies}
    tests_by_id = {test.record_id: test for test in build.sources.tests}
    if len(by_id) != len(build.sources.studies):
        raise ValueError("current-state reading map requires unique authored study IDs")
    specified_ids = tuple(
        study_id
        for *_fields, study_ids in _CURRENT_STATE_METHOD_FAMILY_SPECS
        for study_id in study_ids
    )
    if len(specified_ids) != len(set(specified_ids)):
        raise ValueError("current-state reading map assigns a study more than once")
    missing = sorted(set(specified_ids) - set(by_id))
    unexpected = sorted(set(by_id) - set(specified_ids))
    if missing or unexpected:
        raise ValueError(
            "current-state reading map must partition every authored study exactly once; "
            f"missing={missing}, unexpected={unexpected}"
        )

    rows: list[
        tuple[str, str, str, str, HubRecord, str, HubRecord, tuple[HubRecord, ...]]
    ] = []
    for (
        theme_id,
        title,
        purpose,
        working_answer,
        strongest_study_id,
        open_question,
        next_test_id,
        study_ids,
    ) in _CURRENT_STATE_METHOD_FAMILY_SPECS:
        if strongest_study_id not in study_ids:
            raise ValueError(
                f"current-state method family {theme_id} must name its strongest "
                "study within the family"
            )
        if strongest_study_id not in by_id:
            raise ValueError(
                f"current-state method family {theme_id} names an unknown strongest study"
            )
        next_test = tests_by_id.get(next_test_id)
        if next_test is None:
            raise ValueError(
                f"current-state method family {theme_id} names an unknown next test"
            )
        studies = tuple(by_id[study_id] for study_id in study_ids)
        for study in studies:
            status = study.metadata.get("status")
            if status not in _CURRENT_STATE_STUDY_STATUS_LABELS:
                raise ValueError(
                    f"{study.record_id} has unsupported reading-map status {status!r}"
                )
        rows.append(
            (
                theme_id,
                title,
                purpose,
                working_answer,
                by_id[strongest_study_id],
                open_question,
                next_test,
                studies,
            )
        )
    return tuple(rows)


def _visual_equivalent(
    visual_id: str,
    body: Sequence[str],
    *,
    source: str,
    status: str,
) -> list[str]:
    source_line = f"<p><strong>Source:</strong> {source}</p>"
    status_line = f"<p><strong>Status:</strong> {html.escape(status)}</p>"
    return [
        (
            '<details class="hub-visual-equivalent" '
            f'data-equivalent-for="{visual_id}">'
        ),
        "<summary>Text and data equivalent</summary>",
        *body,
        source_line,
        status_line,
        "</details>",
        (
            '<section class="hub-visual-print-equivalent" '
            f'data-print-equivalent-for="{visual_id}" '
            f'aria-labelledby="{visual_id}-print-equivalent-title">'
        ),
        (
            f'<h4 id="{visual_id}-print-equivalent-title">'
            "Text and data equivalent</h4>"
        ),
        *body,
        source_line,
        status_line,
        "</section>",
    ]


def _evidence_visual(
    visual_id: str,
    title: str,
    explanation: str,
    visual: Sequence[str],
    equivalent: Sequence[str],
    *,
    source: str,
    status: str,
    extra_class: str = "",
) -> list[str]:
    class_name = "hub-evidence-visual"
    if extra_class:
        class_name = f"{class_name} {extra_class}"
    return [
        (
            f'<figure class="{class_name}" data-visual="{visual_id}" '
            f'aria-labelledby="{visual_id}-title">'
        ),
        f'<h3 id="{visual_id}-title">{html.escape(title)}</h3>',
        f'<p class="hub-visual-status"><strong>Status:</strong> {html.escape(status)}</p>',
        f'<p class="hub-visual-explanation">{html.escape(explanation)}</p>',
        *visual,
        *_visual_equivalent(
            visual_id,
            equivalent,
            source=source,
            status=status,
        ),
        f"<figcaption><strong>Source:</strong> {source}</figcaption>",
        "</figure>",
    ]


def _render_ranking_task_flow_visual() -> list[str]:
    stages = (
        ("New request", "Receive an unseen request without its relevance labels."),
        ("Allowed evidence", "Use frozen query, profile, or response evidence."),
        ("Rank candidates", "Score and order all candidate model identifiers."),
        ("Submit and evaluate", "Write a TREC run; evaluate only when labels exist."),
    )
    visual = ['<ol class="hub-map-flow" aria-hidden="true">']
    equivalent = ["<ol>"]
    for title, description in stages:
        visual.append(
            f"<li><strong>{html.escape(title)}</strong>"
            f"<span>{html.escape(description)}</span></li>"
        )
        equivalent.append(
            f"<li><strong>{html.escape(title)}:</strong> "
            f"{html.escape(description)}</li>"
        )
    visual.append("</ol>")
    equivalent.append("</ol>")
    return _evidence_visual(
        "ranking-task-flow",
        "Ranking task flow",
        (
            "The system ranks candidate LLMs; it does not answer the request. "
            "Construction and evaluation stay separated."
        ),
        visual,
        equivalent,
        source='<a href="problem.qmd">Task explainer</a>',
        status="Benchmark workflow",
        extra_class="hub-flow-visual",
    )


def _render_source_hierarchy_visual() -> list[str]:
    rows = (
        (
            "Canonical run facts",
            "Experiment ledger",
            "Metrics, scopes, promotion decisions, artifact custody, and caveats.",
        ),
        (
            "Canonical interpretation",
            "Research Hub records",
            "Study conclusions, research questions, and prospective tests.",
        ),
        (
            "Supporting evidence",
            "Protocols and artifacts",
            "Frozen methods, receipts, reports, and permitted audit evidence.",
        ),
        (
            "Derived views",
            "Generated Hub pages",
            "Readable summaries rebuilt from the canonical sources above.",
        ),
    )
    visual = ['<div class="hub-source-hierarchy" aria-hidden="true">']
    for level, source_name, purpose in rows:
        visual.append(
            (
                '<div class="hub-source-level">'
                f"<span>{html.escape(level)}</span>"
                f"<strong>{html.escape(source_name)}</strong>"
                f"<small>{html.escape(purpose)}</small>"
                "</div>"
            )
        )
    visual.append("</div>")
    equivalent = [
        '<table class="hub-equivalent-table">',
        "<thead><tr><th>Level</th><th>Source</th><th>Responsibility</th></tr></thead>",
        "<tbody>",
    ]
    for level, source_name, purpose in rows:
        equivalent.append(
            f"<tr><td>{html.escape(level)}</td><td>{html.escape(source_name)}</td>"
            f"<td>{html.escape(purpose)}</td></tr>"
        )
    equivalent.extend(["</tbody>", "</table>"])
    return _evidence_visual(
        "canonical-source-hierarchy",
        "Canonical-source hierarchy",
        (
            "Run facts and research conclusions have different owners. Generated "
            "pages are views, never a second ledger."
        ),
        visual,
        equivalent,
        source=(
            '<a href="../evidence/runs.qmd">Experiment registry view</a> and '
            '<a href="../studies/index.qmd">authored Hub studies</a>'
        ),
        status="Canonical-source policy",
        extra_class="hub-hierarchy-visual",
    )


def _family_decision_rows(
    build: HubBuild,
) -> tuple[tuple[str, str, int, str, str], ...]:
    studies = {study.record_id: study for study in build.sources.studies}
    families = sorted({system.experiment_family for system in build.ledger.systems})
    rows: list[tuple[str, str, int, str, str]] = []
    for family in families:
        study = studies.get(family.replace("_", "-"))
        count = sum(
            system.experiment_family == family for system in build.ledger.systems
        )
        if study is None:
            rows.append(
                (
                    family,
                    family.replace("_", " ").title(),
                    count,
                    "ledger_only",
                    "Ledger evidence only",
                )
            )
            continue
        decision = str(study.metadata.get("decision", "informational"))
        rows.append(
            (
                family,
                study.title,
                count,
                decision,
                str(study.metadata.get("decision_summary", "")),
            )
        )
    return tuple(rows)


def _render_family_decision_visual(build: HubBuild) -> list[str]:
    rows = _family_decision_rows(build)
    visual = ['<div class="hub-family-map" aria-hidden="true">']
    for _family, title, count, decision, _summary in rows:
        label = _DECISION_LABELS.get(decision, decision.replace("_", " ").title())
        system_word = "system" if count == 1 else "systems"
        visual.append(
            f'<div class="hub-family-node" data-decision="{html.escape(decision, quote=True)}">'
            f'<span class="hub-family-status">{html.escape(label)}</span>'
            f"<strong>{html.escape(title)}</strong>"
            f"<small>{count} {system_word}</small></div>"
        )
    visual.append("</div>")
    equivalent = [
        '<table class="hub-equivalent-table">',
        (
            "<thead><tr><th>Experiment family</th><th>Registered systems</th>"
            "<th>Decision</th><th>Interpretation</th></tr></thead>"
        ),
        "<tbody>",
    ]
    for _family, title, count, decision, summary in rows:
        label = _DECISION_LABELS.get(decision, decision.replace("_", " ").title())
        equivalent.append(
            f"<tr><td>{html.escape(title)}</td><td>{count}</td>"
            f"<td>{html.escape(label)}</td><td>{html.escape(summary)}</td></tr>"
        )
    equivalent.extend(["</tbody>", "</table>"])
    return _evidence_visual(
        "experiment-family-decisions",
        "Experiment-family decision map",
        (
            "Each family keeps its registered system count beside the authored Hub "
            "decision. A retained diagnostic is not a promotion."
        ),
        visual,
        equivalent,
        source=(
            '<a href="../evidence/runs.qmd">Experiment ledger</a> and '
            '<a href="../studies/index.qmd">study decisions</a>'
        ),
        status="Development decisions",
        extra_class="hub-family-visual",
    )


def _render_research_theme_map_visual(build: HubBuild) -> list[str]:
    rows = _current_state_theme_rows(build)
    visual = ['<div class="hub-theme-reading-map">']
    equivalent = [
        '<table class="hub-equivalent-table">',
        "<thead><tr><th>Method family</th><th>Purpose</th><th>Working answer</th>"
        "<th>Strongest study</th><th>Open question</th><th>Next valid test</th>"
        "<th>Authored study</th><th>Study status</th></tr></thead>",
        "<tbody>",
    ]
    for (
        family_id,
        title,
        purpose,
        working_answer,
        strongest_study,
        open_question,
        next_test,
        studies,
    ) in rows:
        visual.append(
            f'<section class="hub-reading-theme" data-method-family="{family_id}" '
            f'data-reading-theme="{family_id}">'
            f"<h4>{html.escape(title)}</h4>"
            f'<p class="hub-method-purpose"><strong>Purpose:</strong> '
            f"{html.escape(purpose)}</p>"
            f'<p class="hub-method-answer"><strong>Working answer:</strong> '
            f"{html.escape(working_answer)}</p>"
            '<p class="hub-method-strongest"><strong>Strongest canonical study:</strong> '
            f'<a href="../studies/{html.escape(strongest_study.record_id, quote=True)}.qmd">'
            f"{html.escape(strongest_study.title)}</a> "
            f'({html.escape(_CURRENT_STATE_STUDY_STATUS_LABELS[strongest_study.metadata["status"]])})</p>'
            f'<p class="hub-method-open-question"><strong>Open question:</strong> '
            f"{html.escape(open_question)}</p>"
            '<p class="hub-method-next-test"><strong>Next valid test:</strong> '
            f'<a href="../generated/ideas/{html.escape(next_test.record_id, quote=True)}.qmd">'
            f"{html.escape(next_test.title)}</a></p><ol>"
        )
        for study in studies:
            status = str(study.metadata["status"])
            status_label = _CURRENT_STATE_STUDY_STATUS_LABELS[status]
            study_link = f"../studies/{study.record_id}.qmd"
            visual.append(
                f'<li data-study-status="{html.escape(status, quote=True)}">'
                f'<a href="{study_link}">{html.escape(study.title)}</a>'
                f'<span class="hub-study-status">{status_label}</span></li>'
            )
            equivalent.append(
                f'<tr data-reading-theme="{family_id}" '
                f'data-study-status="{html.escape(status, quote=True)}">'
                f"<td>{html.escape(title)}</td>"
                f"<td>{html.escape(purpose)}</td>"
                f"<td>{html.escape(working_answer)}</td>"
                f'<td><a href="../studies/{html.escape(strongest_study.record_id, quote=True)}.qmd">'
                f"{html.escape(strongest_study.title)}</a> "
                f'({html.escape(_CURRENT_STATE_STUDY_STATUS_LABELS[strongest_study.metadata["status"]])})</td>'
                f"<td>{html.escape(open_question)}</td>"
                f'<td><a href="../generated/ideas/{html.escape(next_test.record_id, quote=True)}.qmd">'
                f"{html.escape(next_test.title)}</a></td>"
                f'<td><a href="{study_link}">{html.escape(study.title)}</a></td>'
                f"<td>{status_label}</td></tr>"
            )
        visual.append("</ol></section>")
    visual.append("</div>")
    equivalent.extend(["</tbody>", "</table>"])
    return _evidence_visual(
        "research-theme-map",
        "Studies by method family",
        (
            "This is a presentation-only method-family map, not a canonical taxonomy. "
            "It groups every authored study exactly once in a fixed newcomer reading order. "
            "Study status vocabulary: Planned, Running, and Concluded."
        ),
        visual,
        equivalent,
        source='<a href="../studies/index.qmd">Authored Hub studies</a>',
        status="Presentation-only grouping of canonical study records",
        extra_class="hub-theme-map-visual",
    )


def _render_reader_action_strip(build: HubBuild) -> list[str]:
    """Render reader labels while retaining the source status on every card."""

    ladder = _current_state_ladder_rows(build)
    promoted = [system for _label, system in ladder if system.promotion_status == "promoted_development"]
    controls = [system for _label, system in ladder if system.promotion_status == "stable_control"]
    negatives = [system for _label, system in ladder if system.promotion_status == "valid_negative"]
    if len(promoted) != 1:
        raise ValueError("reader action strip requires exactly one promoted development row")
    evidence = build.completed_follow_up
    e5_study = next(
        (study for study in build.sources.studies if study.record_id == "e5-raw-response-refinement"),
        None,
    )
    if evidence is None or e5_study is None:
        raise ValueError("reader action strip requires the completed E5 follow-up")
    if e5_study.metadata.get("status") != "concluded":
        raise ValueError(
            "e5-raw-response-refinement must be concluded before it can appear "
            "as a promising follow-up"
        )
    if evidence.decision != "advance" or e5_study.metadata.get("decision") != "informational":
        raise ValueError("reader action strip requires the E5 advance/informational boundary")
    readiness = next(
        (record for record in build.sources.background if record.record_id == "pretest-readiness-plan"),
        None,
    )
    snapshot = readiness.metadata.get("dependency_snapshot") if readiness else None
    issues = snapshot.get("issues") if isinstance(snapshot, Mapping) else None
    if not isinstance(issues, Mapping):
        raise ValueError("reader action strip requires the authored pre-test dependency snapshot")
    issue_links: list[str] = []
    for issue in issues.values():
        url = issue.get("github_issue") if isinstance(issue, Mapping) else None
        match = re.fullmatch(
            r"https://github\.com/npatta01/trec_mlm_2026/issues/([1-9][0-9]*)", url
        ) if isinstance(url, str) else None
        if match is None:
            raise ValueError("reader action strip requires repository issue URLs")
        issue_links.append(
            f'<a href="{html.escape(url, quote=True)}">#{match.group(1)}</a>'
        )
    if {link.split(">", maxsplit=1)[1][:-4] for link in issue_links} != {"#39", "#40", "#41", "#42"}:
        raise ValueError("reader action strip requires issues #39 through #42")
    cards = (
        (
            "use-now",
            "Use now",
            f"<code>{html.escape(promoted[0].system_id)}</code> — canonical ledger status "
            f"<code>{html.escape(promoted[0].promotion_status)}</code>; development evidence only.",
        ),
        (
            "retain-as-controls",
            "Retain as controls",
            " ".join(
                f"<code>{html.escape(system.system_id)}</code> "
                f"(<code>{html.escape(system.promotion_status)}</code>)"
                for system in controls
            ),
        ),
        (
            "promising-follow-up",
            "Promising follow-up",
            "Completed E5 receipt gate <code>advance</code>; Hub status "
            "<code>informational</code>; outside the ledger, unregistered, and unpromoted.",
        ),
        (
            "still-required-before-test",
            "Still required before test",
            ", ".join(sorted(issue_links))
            + " — stable work contracts; GitHub remains the source for mutable issue state.",
        ),
        (
            "do-not-reuse-tested-form",
            "Do not reuse in the tested form",
            " ".join(
                f"<code>{html.escape(system.system_id)}</code> "
                f"(<code>{html.escape(system.promotion_status)}</code>)"
                for system in negatives
            )
            + "; this is a condition-bound result, not a permanent verdict.",
        ),
        (
            "unknown",
            "Unknown",
            "Hidden-test behavior and generalization remain unknown; no hidden-test evidence is available.",
        ),
    )
    lines = ['<div class="hub-reader-action-strip">']
    for action_id, label, detail in cards:
        lines.extend(
            [
                f'<section class="hub-reader-action" data-reader-action="{action_id}">',
                f"<h3>{html.escape(label)}</h3>",
                f"<p>{detail}</p>",
                "</section>",
            ]
        )
    lines.append("</div>")
    return lines


def _render_canonical_theme_cards(build: HubBuild) -> list[str]:
    """Render the durable theme answers without duplicating ledger metrics."""

    lines = ['<div class="hub-canonical-theme-cards">']
    for theme in sorted(build.sources.themes, key=lambda row: row.record_id):
        tests = _ordered_theme_tests(build, theme)
        first_test = _next_action(tests)
        remaining = max(len(tests) - 1, 0)
        evidence_refs = _metadata_list(theme, "evidence_refs")
        lines.extend(
            [
                f'<section class="hub-canonical-theme" data-canonical-theme="{html.escape(theme.record_id, quote=True)}">',
                f"<h3>{html.escape(theme.title)}</h3>",
                "<p><strong>Current working answer:</strong> "
                f"{html.escape(str(theme.metadata['current_conclusion']))}</p>",
                "<p><strong>Strongest canonical evidence:</strong> "
                f"{html.escape(str(theme.metadata['strongest_evidence_summary']))}</p>",
                "<ul class=\"hub-theme-evidence-refs\">",
                *(f"<li>{_path_code(reference)}</li>" for reference in evidence_refs),
                "</ul>",
                "<p><strong>Confidence and boundary:</strong> "
                f"{html.escape(_human_label(theme.metadata['confidence']))} confidence. "
                f"{html.escape(str(theme.metadata['confidence_boundary']))}</p>",
                "<p><strong>Open question:</strong> "
                f"{html.escape(str(theme.metadata['open_question']))}</p>",
                "<p><strong>First linked next test:</strong> "
                f"{_render_action_link(first_test, href_prefix='../generated/ideas')} "
                f"({remaining} remaining)</p>",
                "</section>",
            ]
        )
    lines.append("</div>")
    return lines


def _render_representative_results_visual(build: HubBuild) -> list[str]:
    rows = _current_state_ladder_rows(build)
    metrics = [
        system.ndcg_at_10
        for _label, system in rows
        if system.ndcg_at_10 is not None
    ]
    maximum = max(metrics, default=0.0) or 1.0
    visual = ['<div class="hub-result-bars" aria-hidden="true">']
    equivalent = [
        '<table class="hub-equivalent-table">',
        (
            "<thead><tr><th>Role</th><th>System</th><th>Evaluation scope</th>"
            "<th>Ledger status</th><th>Evidence status</th>"
            "<th>nDCG@10</th><th>MRR</th></tr></thead>"
        ),
        "<tbody>",
    ]
    for label, system in rows:
        promotion_label = _CURRENT_STATE_PROMOTION_LABELS[system.promotion_status]
        ratio = 0.0 if system.ndcg_at_10 is None else system.ndcg_at_10 / maximum
        visual.extend(
            [
                f'<div class="hub-result-row" '
                f'data-result-system="{html.escape(system.system_id, quote=True)}" '
                f'data-result-status="{html.escape(system.promotion_status, quote=True)}" '
                f'data-evidence-status="{html.escape(system.evidence_status, quote=True)}">',
                (
                    '<div class="hub-result-label">'
                    f'<span class="hub-result-code" aria-hidden="true">'
                    f"{html.escape(promotion_label[0])}</span>"
                    f"<strong>{html.escape(label)}</strong> "
                    f"<code>{html.escape(system.system_id)}</code>"
                    f"<small>{html.escape(promotion_label)} "
                    f"(<code>{html.escape(system.promotion_status)}</code>)</small></div>"
                ),
                '<div class="hub-result-track">',
                (
                    f'<span class="hub-result-bar" style="--hub-bar: {ratio * 100:.3f}%">'
                    f"{_format_metric(system.ndcg_at_10)}</span>"
                ),
                "</div>",
                "</div>",
            ]
        )
        equivalent.append(
            f'<tr data-result-system="{html.escape(system.system_id, quote=True)}" '
            f'data-result-status="{html.escape(system.promotion_status, quote=True)}" '
            f'data-evidence-status="{html.escape(system.evidence_status, quote=True)}">'
            f"<td>{html.escape(label)}</td>"
            f"<td><code>{html.escape(system.system_id)}</code></td>"
            f"<td><code>{html.escape(system.evaluation_scope)}</code></td>"
            f"<td><code>{html.escape(system.promotion_status)}</code></td>"
            f"<td><code>{html.escape(system.evidence_status)}</code></td>"
            f"<td>{_format_metric(system.ndcg_at_10)}</td>"
            f"<td>{_format_metric(system.mrr)}</td></tr>"
        )
    visual.append("</div>")
    equivalent.extend(["</tbody>", "</table>"])
    return _evidence_visual(
        "representative-development-results",
        "Fixed registered result ladder",
        (
            "Seven fixed registered rows remain in authored order; score never selects, "
            "sorts, substitutes, or promotes a row. Bar length compares nDCG@10 only "
            "within this registered scale. Aggregate-only rows do not claim per-query "
            "evidence, and every promotion status is also written as a non-color label."
        ),
        visual,
        equivalent,
        source='<a href="../evidence/runs.qmd">Experiment ledger</a>',
        status="Development evidence; hidden test unavailable",
        extra_class="hub-results-visual",
    )


def _render_completed_follow_up(build: HubBuild) -> list[str]:
    evidence = build.completed_follow_up
    study = next(
        (
            row
            for row in build.sources.studies
            if row.record_id == "e5-raw-response-refinement"
        ),
        None,
    )
    if evidence is None:
        if study is not None:
            raise ValueError(
                "e5-raw-response-refinement requires validated completed_follow_up evidence"
            )
        return []
    if study is None or evidence.study_id != study.record_id:
        raise ValueError(
            "completed_follow_up must bind to the authored e5-raw-response-refinement study"
        )
    if study.metadata.get("current_state_role") != "strongest_completed_follow_up":
        raise ValueError(
            "e5-raw-response-refinement must carry the authored strongest completed "
            "follow-up role before the current-state page makes that claim"
        )
    if study.metadata.get("status") != "concluded":
        raise ValueError(
            "e5-raw-response-refinement must be concluded before the current-state "
            "page describes it as a completed follow-up"
        )
    if study.metadata.get("decision") != "informational":
        raise ValueError(
            "e5-raw-response-refinement Hub decision must remain informational"
        )
    decision_summary = str(study.metadata.get("decision_summary", "")).strip()
    if not decision_summary:
        raise ValueError(
            "e5-raw-response-refinement must explain its receipt and Hub decision layers"
        )
    linked_system_ids = {
        *_metadata_list(study, "system_ids"),
        *_metadata_list(study, "representative_system_ids"),
    }
    if linked_system_ids:
        raise ValueError(
            "e5-raw-response-refinement must not link registered systems while it is "
            "described as outside the ledger"
        )
    if evidence.candidate_id != "raw_hierarchical_dense_equal":
        raise ValueError(
            "completed_follow_up candidate must be raw_hierarchical_dense_equal"
        )
    if any(
        system.system_id == evidence.candidate_id for system in build.ledger.systems
    ):
        raise ValueError(
            "raw_hierarchical_dense_equal must remain absent from the registered ledger"
        )

    return [
        '<section class="hub-off-ledger-follow-up" aria-labelledby="off-ledger-title">',
        '<h3 id="off-ledger-title">Completed E5 follow-up outside the ledger</h3>',
        (
            "<p>This completed aggregate follow-up is <strong>not registered</strong> "
            "and <strong>not promoted</strong>. It stays outside the registered ladder "
            "and its chart scale.</p>"
        ),
        '<table class="hub-equivalent-table">',
        "<thead><tr><th>Candidate</th><th>Baseline</th><th>Receipt gate decision</th>"
        "<th>nDCG@10</th><th>MRR</th><th>Zero-nDCG queries</th></tr></thead>",
        "<tbody><tr>",
        f"<td><code>{html.escape(evidence.candidate_id)}</code></td>",
        f"<td><code>{html.escape(evidence.baseline_id)}</code></td>",
        f"<td>{html.escape(evidence.decision)}</td>",
        f"<td>{_format_metric(evidence.ndcg_at_10)}</td>",
        f"<td>{_format_metric(evidence.mrr)}</td>",
        f"<td>{evidence.zero_ndcg_queries}</td>",
        "</tr></tbody></table>",
        (
            "<p><strong>Decision layers:</strong> "
            f"{html.escape(decision_summary)}</p>"
        ),
        (
            "<p><strong>Difference from baseline:</strong> "
            f"nDCG@10 {_format_metric(evidence.ndcg_delta)} "
            f"(95% CI {_format_metric(evidence.ndcg_ci_lower)} to "
            f"{_format_metric(evidence.ndcg_ci_upper)}); "
            f"MRR {_format_metric(evidence.mrr_delta)} "
            f"(95% CI {_format_metric(evidence.mrr_ci_lower)} to "
            f"{_format_metric(evidence.mrr_ci_upper)}).</p>"
        ),
        (
            "<p><strong>Source:</strong> "
            f"{_path_code(evidence.source_path)} and "
            f'<a href="../studies/{study.record_id}.qmd">{html.escape(study.title)}</a>'
            "</p>"
        ),
        "<p><strong>Status:</strong> Completed aggregate follow-up; outside the ledger; "
        "not registered; not promoted.</p>",
        "</section>",
    ]


def _artifact_rows(build: HubBuild) -> tuple[tuple[str, str, int], ...]:
    systems = build.ledger.systems
    return (
        (
            "complete",
            "Complete",
            sum(system.artifact_state == "tracked_complete" for system in systems),
        ),
        (
            "receipt",
            "Receipt only",
            sum(
                system.artifact_state == "tracked_receipt_only" for system in systems
            ),
        ),
        (
            "snapshot",
            "Report snapshot",
            sum(
                system.artifact_state == "tracked_report_snapshot"
                for system in systems
            ),
        ),
        (
            "blocker",
            "Artifact blocker",
            sum(
                system.artifact_state
                in {"local_artifact_missing", "construction_artifact_missing"}
                for system in systems
            ),
        ),
    )


def _render_artifact_custody_visual(build: HubBuild) -> list[str]:
    rows = _artifact_rows(build)
    total = sum(count for _key, _label, count in rows)
    denominator = total or 1
    visual = ['<div class="hub-custody-bar" aria-hidden="true">']
    for key, label, count in rows:
        visual.append(
            f'<span class="hub-custody-segment" data-custody="{key}" '
            f'style="--hub-segment: {count / denominator * 100:.3f}%">'
            f"<strong>{count}</strong><small>{html.escape(label)}</small></span>"
        )
    visual.append("</div>")
    equivalent = [
        '<table class="hub-equivalent-table">',
        "<thead><tr><th>Custody state</th><th>Registered systems</th></tr></thead>",
        "<tbody>",
        *[
            f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
            for _key, label, count in rows
        ],
        f"<tr><th>Total</th><td>{total}</td></tr>",
        "</tbody>",
        "</table>",
    ]
    return _evidence_visual(
        "artifact-custody",
        "Artifact-custody composition",
        (
            "Custody measures reproducibility evidence, not ranking quality. Blockers "
            "identify systems whose local construction or run artifacts are missing."
        ),
        visual,
        equivalent,
        source='<a href="../evidence/coverage.qmd">Ledger coverage audit</a>',
        status="Current repository custody",
        extra_class="hub-custody-visual",
    )


def _render_remaining_failures_visual(build: HubBuild) -> list[str]:
    fusion_bucket = build.failure_diagnostic.get("fusion_21")
    no_signal_bucket = build.failure_diagnostic.get("no_signal_54")
    fusion_count = _diagnostic_count(build, "fusion_21")
    no_signal_count = _diagnostic_count(build, "no_signal_54")
    if (fusion_count is None) != (no_signal_count is None):
        raise ValueError("failure diagnostic must provide both documented cohorts")
    if fusion_count is None:
        fusion_label = "Fusion dilution"
        fusion_definition = "One promoted component already has top-ten signal."
        no_signal_label = "No signal"
        no_signal_definition = "Neither promoted component has top-ten signal."
    else:
        cohort_text: list[tuple[str, str]] = []
        for key, bucket in (
            ("fusion_21", fusion_bucket),
            ("no_signal_54", no_signal_bucket),
        ):
            label = bucket.get("label") if isinstance(bucket, Mapping) else None
            definition = (
                bucket.get("definition") if isinstance(bucket, Mapping) else None
            )
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"failure diagnostic {key}.label must be nonempty")
            if not isinstance(definition, str) or not definition.strip():
                raise ValueError(
                    f"failure diagnostic {key}.definition must be nonempty"
                )
            cohort_text.append((label.strip(), definition.strip()))
        (fusion_label, fusion_definition), (
            no_signal_label,
            no_signal_definition,
        ) = cohort_text
    rows = (
        (
            "fusion",
            fusion_label,
            fusion_count,
            fusion_definition,
        ),
        (
            "no-signal",
            no_signal_label,
            no_signal_count,
            no_signal_definition,
        ),
    )
    total = sum(count or 0 for _key, _label, count, _meaning in rows)
    denominator = total or 1
    explanation = (
        f"The promoted run's {total} sole-positive top-ten failures separate into "
        "fusion dilution and cases that need genuinely new evidence."
        if fusion_count is not None
        else (
            "The documented sole-positive top-ten failures separate into fusion "
            "dilution and cases that need genuinely new evidence."
        )
    )
    visual = ['<div class="hub-failure-split" aria-hidden="true">']
    for key, label, count, _meaning in rows:
        display_count = "—" if count is None else str(count)
        visual.append(
            f'<div class="hub-failure-segment" data-failure="{key}" '
            f'style="--hub-segment: {(count or 0) / denominator * 100:.3f}%">'
            f"<strong>{display_count}</strong><span>{html.escape(label)}</span></div>"
        )
    visual.append("</div>")
    equivalent = [
        '<table class="hub-equivalent-table">',
        "<thead><tr><th>Failure cohort</th><th>Cases</th><th>Definition</th></tr></thead>",
        "<tbody>",
    ]
    for _key, label, count, meaning in rows:
        display_count = "Not available" if count is None else str(count)
        equivalent.append(
            f"<tr><td>{html.escape(label)}</td><td>{display_count}</td>"
            f"<td>{html.escape(meaning)}</td></tr>"
        )
    equivalent.extend(["</tbody>", "</table>"])
    return _evidence_visual(
        "remaining-failures",
        "Remaining-failure split",
        explanation,
        visual,
        equivalent,
        source=(
            '<a href="../studies/sparse-failure-analogue-diagnosis.qmd">'
            "Sparse-failure analogue diagnosis</a>"
        ),
        status="Concluded development-only diagnostic",
        extra_class="hub-failure-visual",
    )


def _render_pretest_dependency_visual(build: HubBuild) -> list[str]:
    plans = tuple(
        record
        for record in build.sources.background
        if record.record_id == "pretest-readiness-plan"
    )
    if not plans:
        return _evidence_visual(
            "pretest-dependencies",
            "Pre-test engineering and research dependencies",
            "This synthetic build has no authored pre-test dependency snapshot.",
            [
                '<div class="hub-dependency-map" aria-hidden="true">',
                '<div class="hub-dependency-node"><strong>Not available</strong></div>',
                "</div>",
            ],
            ["<ul><li>No dependency snapshot is available.</li></ul>"],
            source="No authored dependency snapshot",
            status="Unavailable in this build",
            extra_class="hub-dependency-visual",
        )
    if len(plans) != 1:
        raise ValueError(
            "current-state dependency map requires one authored "
            "pretest-readiness-plan background record"
        )
    plan = plans[0]
    snapshot = plan.metadata.get("dependency_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"{plan.source_path}: dependency_snapshot must be a mapping")
    as_of = snapshot.get("as_of")
    if not isinstance(as_of, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of) is None:
        raise ValueError(
            f"{plan.source_path}: dependency_snapshot.as_of must be an ISO date string"
        )
    explanation = snapshot.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError(
            f"{plan.source_path}: dependency_snapshot.explanation must be nonempty"
        )
    raw_issues = snapshot.get("issues")
    if not isinstance(raw_issues, Mapping) or not raw_issues:
        raise ValueError(
            f"{plan.source_path}: dependency_snapshot.issues must be a mapping"
        )
    issues: dict[str, tuple[str, str, str, int]] = {}
    issue_pattern = re.compile(
        r"https://github\.com/npatta01/trec_mlm_2026/issues/([1-9][0-9]*)\Z"
    )
    for raw_id, raw_issue in raw_issues.items():
        if not isinstance(raw_id, str) or re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", raw_id
        ) is None:
            raise ValueError(
                f"{plan.source_path}: dependency issue IDs must be safe slugs"
            )
        if not isinstance(raw_issue, Mapping):
            raise ValueError(
                f"{plan.source_path}: dependency issue {raw_id} must be a mapping"
            )
        issue_url = raw_issue.get("github_issue")
        label = raw_issue.get("label")
        lane = raw_issue.get("lane")
        match = issue_pattern.fullmatch(issue_url) if isinstance(issue_url, str) else None
        if match is None:
            raise ValueError(
                f"{plan.source_path}: dependency issue {raw_id} must use a repository "
                "issue URL"
            )
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"{plan.source_path}: dependency issue {raw_id}.label must be nonempty"
            )
        if lane not in {"input", "freeze", "research"}:
            raise ValueError(
                f"{plan.source_path}: dependency issue {raw_id}.lane is invalid"
            )
        issues[raw_id] = (issue_url, label.strip(), lane, int(match.group(1)))
    external = snapshot.get("external")
    if not isinstance(external, Mapping):
        raise ValueError(
            f"{plan.source_path}: dependency_snapshot.external must be a mapping"
        )
    external_id = external.get("id")
    external_label = external.get("label")
    if not isinstance(external_id, str) or re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", external_id
    ) is None:
        raise ValueError(f"{plan.source_path}: dependency external.id must be a safe slug")
    if not isinstance(external_label, str) or not external_label.strip():
        raise ValueError(
            f"{plan.source_path}: dependency external.label must be nonempty"
        )
    raw_edges = snapshot.get("edges")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raise ValueError(
            f"{plan.source_path}: dependency_snapshot.edges must be a list"
        )
    edges: list[tuple[str, str, str]] = []
    known_nodes = {*issues, external_id}
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Mapping):
            raise ValueError(f"{plan.source_path}: dependency edges must be mappings")
        source_id = raw_edge.get("from")
        target_id = raw_edge.get("to")
        relation = raw_edge.get("relation")
        if (
            not isinstance(source_id, str)
            or not isinstance(target_id, str)
            or source_id not in known_nodes
            or target_id not in known_nodes
        ):
            raise ValueError(f"{plan.source_path}: dependency edge references unknown node")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError(f"{plan.source_path}: dependency relation must be nonempty")
        edges.append((str(source_id), str(target_id), relation.strip()))
    if not edges:
        raise ValueError(f"{plan.source_path}: dependency snapshot must contain edges")

    def issue_node(issue_id: str) -> str:
        _url, label, _lane, number = issues[issue_id]
        return (
            '<div class="hub-dependency-node">'
            f"<strong>#{number}</strong><span>{html.escape(label)}</span></div>"
        )

    def linked_node(node_id: str) -> str:
        if node_id == external_id:
            return html.escape(external_label.strip())
        issue_url, label, _lane, number = issues[node_id]
        return (
            f'<a href="{html.escape(issue_url, quote=True)}">'
            f"#{number} {html.escape(label)}</a>"
        )

    input_ids = sorted(
        issue_id for issue_id, (_url, _label, lane, _number) in issues.items()
        if lane == "input"
    )
    freeze_ids = sorted(
        issue_id for issue_id, (_url, _label, lane, _number) in issues.items()
        if lane == "freeze"
    )
    research_ids = sorted(
        issue_id for issue_id, (_url, _label, lane, _number) in issues.items()
        if lane == "research"
    )
    if not input_ids or not freeze_ids or not research_ids:
        raise ValueError(
            f"{plan.source_path}: dependency snapshot requires input, freeze, and "
            "research lanes"
        )
    edge_pairs = {(source_id, target_id) for source_id, target_id, _relation in edges}
    expected_edge_pairs = {
        *((source_id, target_id) for source_id in input_ids for target_id in freeze_ids),
        *((source_id, external_id) for source_id in freeze_ids),
        *((source_id, external_id) for source_id in research_ids),
    }
    if len(edge_pairs) != len(edges) or edge_pairs != expected_edge_pairs:
        raise ValueError(
            f"{plan.source_path}: dependency edges must match the visible lane topology"
        )
    visual = [
        '<div class="hub-dependency-map" aria-hidden="true">',
        '<div class="hub-dependency-inputs">',
        *[issue_node(issue_id) for issue_id in input_ids],
        "</div>",
        '<span class="hub-dependency-arrow">feed</span>',
        *[
            issue_node(issue_id).replace(
                'class="hub-dependency-node"',
                'class="hub-dependency-node hub-dependency-freeze"',
            )
            for issue_id in freeze_ids
        ],
        *[
            issue_node(issue_id).replace(
                'class="hub-dependency-node"',
                'class="hub-dependency-node hub-dependency-research"',
            )
            for issue_id in research_ids
        ],
        '<span class="hub-dependency-arrow">then</span>',
        '<div class="hub-dependency-node hub-dependency-external"><strong>External</strong>'
        f"<span>{html.escape(external_label.strip())}</span></div>",
        "</div>",
    ]
    equivalent = [
        "<ol>",
        *[
            f"<li>{linked_node(source_id)} {html.escape(relation)} "
            f"{linked_node(target_id)}.</li>"
            for source_id, target_id, relation in edges
        ],
        "<li>No hidden-test result exists yet.</li>",
        "</ol>",
    ]
    return _evidence_visual(
        "pretest-dependencies",
        "Pre-test engineering and research dependencies",
        explanation.strip(),
        visual,
        equivalent,
        source=(
            '<a href="../background/pretest-readiness-plan.qmd">'
            f"Authored dependency snapshot ({html.escape(as_of)})</a>"
        ),
        status=(
            "Planned structure. Open GitHub for current title, status, labels, "
            "and milestone."
        ),
        extra_class="hub-dependency-visual",
    )


def render_current_state_qmd(build: HubBuild) -> str:
    """Render the newcomer evidence map from canonical build inputs."""

    systems = build.ledger.systems
    family_count = len({system.experiment_family for system in systems})
    system_count = len(systems)
    study_count = len(build.sources.studies)
    promoted_count = sum(
        system.promotion_status == "promoted_development" for system in systems
    )
    artifact_rows = _artifact_rows(build)
    artifact_counts = {key: count for key, _label, count in artifact_rows}
    fusion_count = _diagnostic_count(build, "fusion_21")
    no_signal_count = _diagnostic_count(build, "no_signal_54")
    failure_heading = "Diagnose the remaining failure split"
    if fusion_count is not None and no_signal_count is not None:
        failure_heading = (
            f"Diagnose the {fusion_count + no_signal_count}-case failure split"
        )
    has_development_only_caveat = any(
        caveat.get("caveat_id") == "development_only"
        for caveat in build.ledger.caveats
    )
    if not has_development_only_caveat:
        raise ValueError("ledger must preserve the development_only evidence boundary")
    # Validate the full study partition before optional reader-oriented surfaces
    # inspect individual studies such as the off-ledger E5 follow-up.
    _current_state_theme_rows(build)
    family_label = "experiment family" if family_count == 1 else "experiment families"
    system_label = "registered system" if system_count == 1 else "registered systems"
    study_label = "authored study" if study_count == 1 else "authored studies"
    promoted_label = (
        "promoted development system"
        if promoted_count == 1
        else "promoted development systems"
    )
    blocker_count = artifact_counts["blocker"]
    blocker_label = "blocker" if blocker_count == 1 else "blockers"
    snapshot_count = artifact_counts["snapshot"]
    snapshot_label = (
        "report snapshot" if snapshot_count == 1 else "report snapshots"
    )
    lines = [
        "<!-- Generated by scripts/build_research_hub.py; do not edit. -->",
        "",
        "## Evidence in one view",
        "",
        "::: {.hub-current-state-facts}",
        (
            f"- **Registered evidence:** {family_count} {family_label}; "
            f"{system_count} {system_label}; {study_count} {study_label}; "
            f"{promoted_count} {promoted_label}."
        ),
        (
            "- **Artifact custody:** "
            f"{artifact_counts['complete']} complete / "
            f"{artifact_counts['receipt']} receipt-only / "
            f"{snapshot_count} {snapshot_label} / "
            f"{blocker_count} {blocker_label}."
        ),
        "- **Evidence boundary:** No hidden-test evidence is available.",
        ":::",
    ]
    lines.extend(
        [
            "",
            "## What the project recommends now",
            "",
            *_render_reader_action_strip(build),
            "",
            "## Studies by method family",
            "",
            *_render_research_theme_map_visual(build),
            "",
            "## Canonical research-theme answers",
            "",
            *_render_canonical_theme_cards(build),
            "",
            "## Compare the fixed registered result ladder",
            "",
            *_render_representative_results_visual(build),
            "",
            *_render_completed_follow_up(build),
            "",
            f"## {failure_heading}",
            "",
            *_render_remaining_failures_visual(build),
            "",
            "## Audit reference",
            "",
            '<details class="hub-audit-reference">',
            "<summary>Workflow, source hierarchy, decisions, and custody</summary>",
            "",
            *_render_ranking_task_flow_visual(),
            "",
            *_render_source_hierarchy_visual(),
            "",
            *_render_family_decision_visual(build),
            "",
            *_render_artifact_custody_visual(build),
            "",
            "</details>",
            "",
            "## What must happen before the test run",
            "",
            *_render_pretest_dependency_visual(build),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _records_for_theme(
    records: Sequence[HubRecord],
    theme_id: str,
) -> list[HubRecord]:
    return sorted(
        (
            record
            for record in records
            if theme_id in _metadata_list(record, "theme_ids")
        ),
        key=lambda record: record.record_id,
    )


def _tests_for_theme(build: HubBuild, theme_id: str) -> list[HubRecord]:
    """Derive a theme's tests from canonical test-owned relationships."""

    return sorted(
        (
            test
            for test in build.sources.tests
            if theme_id in _metadata_list(test, "theme_ids")
        ),
        key=lambda test: test.record_id,
    )


def _tests_for_questions(
    build: HubBuild,
    question_ids: Sequence[str],
) -> list[HubRecord]:
    """Derive reverse question links from each canonical test record."""

    linked_ids = set(question_ids)
    return sorted(
        (
            test
            for test in build.sources.tests
            if linked_ids.intersection(_metadata_list(test, "linked_question_ids"))
        ),
        key=lambda test: test.record_id,
    )


def _ordered_theme_tests(build: HubBuild, theme: HubRecord) -> list[HubRecord]:
    """Order a theme's linked tests by its authored next-test preference."""

    tests = _tests_for_theme(build, theme.record_id)
    preferred_ids = _metadata_list(theme, "next_test_ids")
    preferred_order = {
        record_id: index for index, record_id in enumerate(preferred_ids)
    }
    return sorted(
        tests,
        key=lambda test: (
            preferred_order.get(test.record_id, len(preferred_order)),
            test.record_id,
        ),
    )


def _next_action(tests: Sequence[HubRecord]) -> _NextAction:
    """Return one deterministic action without choosing an output context."""

    if not tests:
        return _NextAction(None, None, "No next test is currently linked.")
    test = tests[0]
    return _NextAction(
        record_id=test.record_id,
        title=test.title,
        text=str(test.metadata.get("next_action", "")),
    )


def _render_action_text(action: _NextAction) -> str:
    """Render an action as plain text for callers already inside a link."""

    if action.title is None:
        return html.escape(action.text)
    return f"{html.escape(action.title)}: {html.escape(action.text)}"


def _render_action_link(action: _NextAction, *, href_prefix: str) -> str:
    """Render an action link using the caller's output-relative destination."""

    if action.record_id is None or action.title is None:
        return html.escape(action.text)
    return (
        f'<a href="{html.escape(href_prefix, quote=True)}/'
        f'{html.escape(action.record_id, quote=True)}.qmd">'
        f"{html.escape(action.title)}</a>: {html.escape(action.text)}"
    )


def _count_label(count: int, singular: str) -> str:
    if count == 1:
        return f"{count} {singular}"
    plural = f"{singular[:-1]}ies" if singular.endswith("y") else f"{singular}s"
    return f"{count} {plural}"


def _human_label(value: object) -> str:
    return str(value).replace("_", " ").strip().capitalize()


def _development_evidence_notice() -> list[str]:
    return [
        "::: {.hub-evidence-notice}",
        "**Development evidence only.** Promotion labels do not imply held-out results.",
        ":::",
    ]


def _render_theme_rows(build: HubBuild, *, href_prefix: str) -> list[str]:
    lines: list[str] = []
    for theme in sorted(build.sources.themes, key=lambda row: row.record_id):
        action = _next_action(_ordered_theme_tests(build, theme))
        lines.extend(
            [
                "",
                (
                    '<a class="hub-theme-row" '
                    f'href="{href_prefix}/'
                    f'{html.escape(theme.record_id, quote=True)}.qmd">'
                ),
                (
                    "<span>"
                    f"{html.escape(_human_label(theme.metadata.get('decision', '')))} · "
                    f"{html.escape(_human_label(theme.metadata.get('confidence', '')))} "
                    "confidence"
                    "</span>"
                ),
                f"<strong>{html.escape(theme.title)}</strong>",
                (
                    "<span>"
                    f"{html.escape(str(theme.metadata.get('current_conclusion', '')))}"
                    "</span>"
                ),
                (f'<span class="hub-next-action">{_render_action_text(action)}</span>'),
                "</a>",
            ]
        )
    return lines


def _render_summary_cards(build: HubBuild) -> str:
    lines = [
        *_development_evidence_notice(),
        "",
        "## Current position by theme",
        *_render_theme_rows(build, href_prefix="generated/themes"),
    ]
    lines.extend(["", "## Open across themes", ""])
    cross_theme_questions = [
        question
        for question in _ordered_questions(build)
        if len(_metadata_list(question, "theme_ids")) > 1
    ]
    if cross_theme_questions:
        for question in cross_theme_questions:
            lines.append(
                f"- [{_markdown_table_text(question.title)}]"
                f"(questions/{question.record_id}.qmd): "
                f"{_markdown_table_text(question.metadata.get('current_answer', ''))}"
            )
    else:
        lines.append("No active question currently crosses themes.")
    next_tests = _summary_next_tests(build)
    lines.extend(["", "## Highest-value next tests", "", *next_tests[3:]])
    lines.extend(
        [
            "",
            "## Explore",
            "",
            "- [All research questions](questions/index.qmd)",
            "- [Study library](studies/index.qmd)",
            "- [Research question coverage](evidence/coverage.qmd)",
            "- [Decision history](evidence/history.qmd)",
            "- [Experiment runs](evidence/runs.qmd)",
        ]
    )
    return "\n".join(lines)


def render_summary_qmd(build: HubBuild) -> str:
    return _render_summary_cards(build).rstrip() + "\n"


def render_themes_index_qmd(build: HubBuild) -> str:
    """Render the theme-only landing from canonical theme relationships."""

    lines = [
        *_development_evidence_notice(),
        "",
        "Open a theme for its current conclusion, questions, and evidence trail.",
        "",
        "## Themes",
        *_render_theme_rows(build, href_prefix="../generated/themes"),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_background_index_qmd(build: HubBuild) -> str:
    """Render source-backed Background navigation in an editorial reading order."""

    records = {record.record_id: record for record in build.sources.background}
    groups = (
        (
            "TREC 2025",
            "Start with the task and then inspect how participants turned "
            "historical behavior into model rankings.",
            ("trec-2025-literature", "prior-participant-approaches"),
        ),
        (
            "Wider methods and evaluations",
            "Connect expert finding, retrieval, routing, and evaluation research "
            "to this task without importing results across incompatible settings.",
            ("methods-landscape", "llm-routing-literature", "related-competitions"),
        ),
        (
            "Current model and reference planning",
            "Use the dated planning evidence to choose reference generators and "
            "review controls; it is not candidate-ranking evidence.",
            ("reference-model-planning",),
        ),
    )
    configured_ids = {
        record_id for _, _, record_ids in groups for record_id in record_ids
    }
    additional_ids = tuple(sorted(set(records) - configured_ids))
    if additional_ids:
        groups = (
            *groups,
            (
                "Additional background",
                "Other source-backed context registered by this build.",
                additional_ids,
            ),
        )

    lines = [
        "These resources synthesize the controlling reviews and preserve the "
        "boundary between development evidence, prior-track results, and "
        "adjacent research.",
    ]
    for heading, introduction, record_ids in groups:
        present_ids = tuple(
            record_id for record_id in record_ids if record_id in records
        )
        if not present_ids:
            continue
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                introduction,
                "",
                '<div class="hub-background-list">',
            ]
        )
        for record_id in present_ids:
            record = records[record_id]
            lines.extend(
                [
                    (
                        '<a class="hub-background-row" '
                        f'href="{html.escape(record.record_id, quote=True)}.qmd">'
                    ),
                    f"<strong>{html.escape(record.title)}</strong>",
                    (
                        "<span>"
                        f"{html.escape(str(record.metadata.get('summary', '')))}"
                        "</span>"
                    ),
                    "</a>",
                ]
            )
        lines.append("</div>")
    return "\n".join(lines).rstrip() + "\n"


def _dataset_availability_class(record: HubRecord) -> str:
    availability = str(record.metadata.get("availability", ""))
    sharing = str(record.metadata.get("sharing", ""))
    if availability == "tracked_complete":
        return "hub-available"
    if sharing in {"local_only", "metadata_only"}:
        return "hub-restricted"
    return "hub-mixed"


def render_data_index_qmd(build: HubBuild) -> str:
    """Render the Data catalog from canonical dataset records."""

    groups = (
        (
            "provided",
            "Organizer-provided data",
            "These records define the development task and how its rankings "
            "are evaluated.",
        ),
        (
            "collected",
            "Collected evidence",
            "Evidence gathered from candidate models under bounded project protocols.",
        ),
        (
            "generated",
            "Generated data and run evidence",
            "Artifacts produced by ranking and evaluation workflows.",
        ),
        (
            "external",
            "External data",
            "Referenced data maintained outside this repository.",
        ),
    )
    datasets = sorted(build.sources.datasets, key=lambda row: row.record_id)
    lines = [
        "The catalog is generated from canonical dataset records. It separates "
        "organizer-provided data from collected, generated, and external "
        "evidence without reproducing sensitive source content.",
        "",
        "::: {.hub-data-groups}",
    ]
    for category, heading, introduction in groups:
        records = [
            record for record in datasets if record.metadata.get("category") == category
        ]
        if not records:
            continue
        lines.extend(["", f"## {heading}", "", introduction])
        for index, record in enumerate(records):
            if index:
                lines.extend(["", "---"])
            lines.extend(
                [
                    "",
                    f"### [{_markdown_table_text(record.title)}]"
                    f"({record.record_id}.qmd)",
                    "",
                    str(record.metadata.get("plain_english", "")),
                    "",
                    (
                        '<div class="hub-badges">'
                        f'<span class="hub-badge">{html.escape(_human_label(category))}'
                        "</span>"
                        f'<span class="hub-badge {_dataset_availability_class(record)}">'
                        f"{html.escape(_human_label(record.metadata.get('availability', '')))}"
                        "</span>"
                        "</div>"
                    ),
                ]
            )
    lines.extend(["", ":::"])
    return "\n".join(lines).rstrip() + "\n"


def _render_theme_page(build: HubBuild, theme: HubRecord) -> str:
    questions = _records_for_theme(build.sources.questions, theme.record_id)
    studies = _records_for_theme(build.sources.studies, theme.record_id)
    tests = _ordered_theme_tests(build, theme)
    action = _next_action(tests)
    theme_id = quote(theme.record_id, safe="")
    status = (
        f"{_human_label(theme.metadata.get('decision', ''))} · "
        f"{_human_label(theme.metadata.get('confidence', ''))} confidence"
    )
    lines = [
        "---",
        f"title: {_yaml_scalar(theme.title)}",
        f"description: {_yaml_scalar(theme.metadata.get('plain_english', ''))}",
        "toc: false",
        "---",
        "",
        '<section class="hub-record-hero">',
        f'<p class="hub-record-status">{html.escape(status)}</p>',
        f'<p class="hub-takeaway">{html.escape(str(theme.metadata.get("current_conclusion", "")))}</p>',
        f"<p>{html.escape(str(theme.metadata.get('plain_english', '')))}</p>",
    ]
    visual = _render_record_visual(theme)
    if visual:
        lines.extend(["", *visual])
    lines.extend(
        [
            "",
            (
                '<p class="hub-next-action">'
                f"{_render_action_link(action, href_prefix='../ideas')}"
                "</p>"
            ),
            "</section>",
            "",
        ]
    )
    evidence_lines = [
        "## Why does it matter?",
        "",
        theme.body,
        "",
        "## Current working answer",
        "",
        str(theme.metadata.get("current_conclusion", "")),
        "",
        "## Strongest canonical evidence",
        "",
        str(theme.metadata.get("strongest_evidence_summary", "")),
        "",
        "Canonical evidence paths:",
        "",
        *(f"- {_path_code(reference)}" for reference in _metadata_list(theme, "evidence_refs")),
        "",
        "## Open question",
        "",
        str(theme.metadata.get("open_question", "")),
    ]
    lines.extend(_render_details("Evidence and current knowledge", evidence_lines))
    lines.extend(
        _render_details(
            "Limits and readiness",
            [
                "**Confidence and evidence boundary**",
                "",
                (
                    f"{html.escape(_human_label(theme.metadata.get('confidence', '')))} "
                    f"confidence. "
                    f"{html.escape(str(theme.metadata.get('confidence_boundary', '')))}"
                ),
                "",
                "**First linked next test**",
                "",
                _render_action_link(action, href_prefix="../ideas"),
                f" ({max(len(tests) - 1, 0)} remaining linked tests).",
            ],
        )
    )
    relationship_lines = ["## Other linked tests", ""]
    relationship_lines.extend(
        f"- {_render_action_link(_NextAction(test.record_id, test.title, str(test.metadata.get('next_action', ''))), href_prefix='../ideas')}"
        for test in tests[1:]
    )
    if len(tests) < 2:
        relationship_lines.append("- No additional linked test is currently recorded.")
    relationship_lines.extend(["", "## Questions in this theme"])
    if not questions:
        relationship_lines.extend(["", "No research question is currently linked."])
    for question in questions:
        open_subquestions = _metadata_list(question, "open_subquestions")
        relationship_lines.extend(
            [
                "",
                '<div class="hub-question-row">',
                (
                    f'<h3><a href="../../questions/'
                    f'{html.escape(question.record_id, quote=True)}.qmd">'
                    f"{html.escape(question.title)}</a></h3>"
                ),
                f"<p>{html.escape(str(question.metadata.get('plain_english', '')))}</p>",
                (
                    "<p><strong>Current bounded answer</strong><br>"
                    f"{html.escape(str(question.metadata.get('current_answer', '')))}</p>"
                ),
            ]
        )
        if open_subquestions:
            relationship_lines.extend(
                [
                    "<p><strong>Still open</strong></p>",
                    "<ul>",
                    *(
                        f"<li>{html.escape(subquestion)}</li>"
                        for subquestion in open_subquestions
                    ),
                    "</ul>",
                ]
            )
        relationship_lines.append("</div>")
    relationship_lines.extend(
        [
            "",
            "## Evidence behind this position",
            "",
            (
                '<a class="hub-evidence-line" '
                f'href="../../studies/index.html?theme={theme_id}">'
                f"Explore {_count_label(len(studies), 'study')} linked to this theme "
                "in the study library.</a>"
            ),
        ]
    )
    lines.extend(_render_details("Relationships and evidence", relationship_lines))
    return "\n".join(lines)


def render_theme_pages(build: HubBuild) -> dict[Path, str]:
    """Render one synthesized navigation page per canonical research theme."""

    return {
        _validate_output_path(
            Path(f"research/hub/generated/themes/{theme.record_id}.qmd")
        ): _render_theme_page(build, theme).rstrip() + "\n"
        for theme in sorted(build.sources.themes, key=lambda row: row.record_id)
    }


def render_questions_index_qmd(build: HubBuild) -> str:
    """Render the question library from canonical question records."""

    lines = [
        *_development_evidence_notice(),
        "",
        "Start with the question and its current bounded answer; open a row for "
        "supporting studies, limitations, and next tests.",
        "",
        '<div class="hub-question-list">',
    ]
    questions = sorted(
        build.sources.questions,
        key=lambda row: (
            _QUESTION_STATUS_PRIORITY.get(
                str(row.metadata.get("status")),
                len(_QUESTION_STATUS_PRIORITY),
            ),
            row.record_id,
        ),
    )
    for question in questions:
        lines.extend(
            [
                (
                    '<a class="hub-question-row" '
                    f'href="{html.escape(question.record_id, quote=True)}.qmd">'
                ),
                (
                    "<span>"
                    f"{html.escape(_human_label(question.metadata.get('status', '')))} · "
                    f"{html.escape(_human_label(question.metadata.get('confidence', '')))} "
                    "confidence"
                    "</span>"
                ),
                f"<strong>{html.escape(question.title)}</strong>",
                f"<span>{html.escape(str(question.metadata.get('plain_english', '')))}</span>",
                (
                    '<span class="hub-next-action">'
                    f"{_render_action_text(_next_action(_tests_for_questions(build, (question.record_id,))))}"
                    "</span>"
                ),
                "</a>",
            ]
        )
    lines.append("</div>")
    return "\n".join(lines).rstrip() + "\n"


def _body_section(record: HubRecord, heading: str) -> str:
    """Return one authored H2 section without interpreting its content."""

    lines = record.body.splitlines()
    marker = f"## {heading}"
    try:
        start = lines.index(marker) + 1
    except ValueError:
        return ""
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end]).replace("<!--", "").replace("-->", "").strip()


def _authored_body(record: HubRecord) -> str:
    """Return canonical supporting prose while hiding its source-only comment."""

    return "\n".join(
        line
        for line in record.body.replace("<!--", "").replace("-->", "").splitlines()
        if not line.strip().startswith("{{< include ")
    ).strip()


def _render_details(
    label: str,
    lines: Sequence[str],
    *,
    open: bool = False,
) -> list[str]:
    """Render supporting material behind a native progressive-disclosure control."""

    open_attribute = " open" if open else ""
    return [
        "",
        f'<details class="hub-details"{open_attribute}>',
        f"<summary>{html.escape(label)}</summary>",
        *lines,
        "</details>",
    ]


def _render_question_detail(build: HubBuild, question: HubRecord) -> str:
    studies_by_id = {study.record_id: study for study in build.sources.studies}
    themes_by_id = {theme.record_id: theme for theme in build.sources.themes}
    studies = sorted(
        (
            studies_by_id[study_id]
            for study_id in _metadata_list(question, "study_ids")
            if study_id in studies_by_id
        ),
        key=lambda study: study.record_id,
    )
    tests = _tests_for_questions(build, (question.record_id,))
    action = _next_action(tests)
    themes = sorted(
        (
            themes_by_id[theme_id]
            for theme_id in _metadata_list(question, "theme_ids")
            if theme_id in themes_by_id
        ),
        key=lambda theme: theme.record_id,
    )
    authored_motivation = _body_section(question, "In plain English") or question.body
    authored_limits = _body_section(question, "Evidence limits")
    authored_connections = _body_section(question, "Linked studies")
    status = (
        f"{_human_label(question.metadata.get('status', ''))} · "
        f"{_human_label(question.metadata.get('confidence', ''))} confidence"
    )
    lines = [
        "<!-- Generated by scripts/build_research_hub.py; do not edit. -->",
        "",
        '<section class="hub-record-hero">',
        f'<p class="hub-record-status">{html.escape(status)}</p>',
        f'<p class="hub-takeaway">{html.escape(str(question.metadata.get("plain_english", "")))}</p>',
        f"<p>{html.escape(str(question.metadata.get('current_answer', '')))}</p>",
    ]
    visual = _render_record_visual(question)
    if visual:
        lines.extend(["", *visual])
    lines.extend(
        [
            "",
            (
                '<p class="hub-next-action">'
                f"{_render_action_link(action, href_prefix='../generated/ideas')}"
                "</p>"
            ),
            "</section>",
        ]
    )
    evidence_lines = [
        "## Why does it matter?",
        "",
        html.escape(authored_motivation),
        "",
        "## Current bounded answer",
        "",
        html.escape(str(question.metadata.get("current_answer", ""))),
        "",
        (
            f"**Confidence:** "
            f"{html.escape(_human_label(question.metadata.get('confidence', '')))}"
        ),
    ]
    lines.extend(_render_details("Evidence and current knowledge", evidence_lines))
    limit_lines = [html.escape(str(question.metadata.get("confidence_boundary", "")))]
    if authored_limits:
        limit_lines.extend(["", html.escape(authored_limits)])
    open_subquestions = _metadata_list(question, "open_subquestions")
    if open_subquestions:
        limit_lines.extend(["", "**Open subquestions**", ""])
        limit_lines.extend(
            f"- {html.escape(subquestion)}" for subquestion in open_subquestions
        )
    lines.extend(_render_details("Limits and readiness", limit_lines))
    relationship_lines = ["## Linked tests", ""]
    relationship_lines.extend(
        (
            f'- <a href="../generated/ideas/{html.escape(test.record_id, quote=True)}.qmd">'
            f"{html.escape(test.title)}</a>: "
            f"{html.escape(str(test.metadata.get('next_action', '')))}"
        )
        for test in tests
    )
    if not tests:
        relationship_lines.append("- No next test is currently linked.")
    relationship_lines.extend(["", "**Themes**", ""])
    relationship_lines.extend(
        (
            f"- [{_markdown_table_text(theme.title)}]"
            f"(../generated/themes/{theme.record_id}.qmd)"
        )
        for theme in themes
    )
    if not themes:
        relationship_lines.append("- No linked theme.")
    relationship_lines.extend(["", "**Studies**", ""])
    relationship_lines.extend(
        (f"- [{_markdown_table_text(study.title)}](../studies/{study.record_id}.qmd)")
        for study in studies
    )
    if not studies:
        relationship_lines.append("- No linked study.")
    if authored_connections:
        relationship_lines.extend(["", html.escape(authored_connections)])
    relationship_lines.extend(["", "**Canonical evidence**", ""])
    evidence_refs = _metadata_list(question, "evidence_refs")
    relationship_lines.extend(
        f"- {_path_code(reference)}" for reference in evidence_refs
    )
    if not evidence_refs:
        relationship_lines.append("- No canonical evidence path is recorded.")
    lines.extend(
        _render_details("Relationships and technical details", relationship_lines)
    )
    return "\n".join(lines)


def render_question_detail_pages(build: HubBuild) -> dict[Path, str]:
    """Render source-driven explainer fragments for canonical question pages."""

    return {
        _validate_output_path(
            Path(f"research/hub/generated/questions/{question.record_id}.md")
        ): _render_question_detail(build, question).rstrip() + "\n"
        for question in sorted(
            build.sources.questions,
            key=lambda row: row.record_id,
        )
    }


def _test_readiness_count(test: HubRecord) -> str:
    complete = sum(
        isinstance(test.metadata.get(field), str)
        and bool(str(test.metadata[field]).strip())
        for field in READY_FIELDS
    )
    return f"{complete}/{len(READY_FIELDS)} readiness fields complete"


def _render_github_issue_link(test: HubRecord) -> str | None:
    issue_number = github_issue_number(test)
    issue_url = test.metadata.get("github_issue")
    if issue_number is None or not isinstance(issue_url, str):
        return None
    return (
        f'<a href="{html.escape(issue_url, quote=True)}">'
        f"Track in GitHub #{issue_number}</a>"
    )


def render_tests_index_qmd(build: HubBuild) -> str:
    """Render the prospective-work library from canonical test records."""

    themes_by_id = {theme.record_id: theme for theme in build.sources.themes}
    tests = sorted(
        build.sources.tests,
        key=lambda row: (
            str(row.metadata.get("primary_theme_id", "")),
            _TEST_STAGE_PRIORITY.get(
                str(row.metadata.get("stage")),
                len(_TEST_STAGE_PRIORITY),
            ),
            row.record_id,
        ),
    )
    lines = [
        *_development_evidence_notice(),
        "",
        "Ideas are prospective work, not results. Open a row for the full "
        "explanation, evidence boundary, and remaining readiness work.",
    ]
    current_theme_id: str | None = None
    current_stage: str | None = None
    stage_open = False
    for test in tests:
        primary_theme_id = str(test.metadata.get("primary_theme_id", ""))
        stage_value = str(test.metadata.get("stage", ""))
        if primary_theme_id != current_theme_id:
            if stage_open:
                lines.append("</div>")
                stage_open = False
            theme = themes_by_id.get(primary_theme_id)
            theme_title = theme.title if theme is not None else primary_theme_id
            lines.extend(["", f"## {theme_title}"])
            current_theme_id = primary_theme_id
            current_stage = None
        if stage_value != current_stage:
            if stage_open:
                lines.append("</div>")
            lines.extend(
                [
                    "",
                    f"### {_human_label(stage_value)}",
                    "",
                    '<div class="hub-idea-list">',
                ]
            )
            stage_open = True
            current_stage = stage_value
        stage = html.escape(_human_label(test.metadata.get("stage", "")))
        priority_value = test.metadata.get("priority")
        if isinstance(priority_value, str) and priority_value.strip():
            priority = f"{html.escape(_human_label(priority_value))} priority"
        else:
            priority = "Priority not specified"
        readiness = html.escape(_test_readiness_count(test))
        lines.extend(
            [
                (
                    '<a class="hub-idea-row" '
                    f'data-test-id="{html.escape(test.record_id, quote=True)}" '
                    f'data-themes="{html.escape(" ".join(_metadata_list(test, "theme_ids")), quote=True)}" '
                    f'href="../generated/ideas/'
                    f'{html.escape(test.record_id, quote=True)}.qmd">'
                ),
                f"<span>{stage} · {priority} · {readiness}</span>",
                f"<strong>{html.escape(test.title)}</strong>",
                f"<span>{html.escape(str(test.metadata.get('plain_english', '')))}</span>",
                (
                    '<span class="hub-next-action"><strong>Next action:</strong> '
                    f"{html.escape(str(test.metadata.get('next_action', '')))}</span>"
                ),
                "</a>",
            ]
        )
        github_issue_link = _render_github_issue_link(test)
        if github_issue_link is not None:
            lines.append(f'<p class="hub-github-issue">{github_issue_link}</p>')
    if stage_open:
        lines.append("</div>")
    if not tests:
        lines.extend(["", "No prospective work is currently recorded."])
    return "\n".join(lines).rstrip() + "\n"


def _render_test_related_research(build: HubBuild, test: HubRecord) -> list[str]:
    themes_by_id = {theme.record_id: theme for theme in build.sources.themes}
    questions_by_id = {
        question.record_id: question for question in build.sources.questions
    }
    themes = [
        themes_by_id[theme_id]
        for theme_id in _metadata_list(test, "theme_ids")
        if theme_id in themes_by_id
    ]
    questions = [
        questions_by_id[question_id]
        for question_id in _metadata_list(test, "linked_question_ids")
        if question_id in questions_by_id
    ]
    lines = ["", "## How does it connect?", "", "**Themes**", ""]
    lines.extend(
        f'- <a href="../themes/{html.escape(theme.record_id, quote=True)}.qmd">'
        f"{html.escape(theme.title)}</a>"
        for theme in sorted(themes, key=lambda row: row.record_id)
    )
    if not themes:
        lines.append("- No linked theme.")
    lines.extend(["", "**Questions**", ""])
    lines.extend(
        f'- <a href="../../questions/{html.escape(question.record_id, quote=True)}.qmd">'
        f"{html.escape(question.title)}</a>"
        for question in sorted(questions, key=lambda row: row.record_id)
    )
    if not questions:
        lines.append("- No linked research question.")
    return lines


def _render_record_visual(record: HubRecord) -> list[str]:
    """Render a record's optional, explicitly authored visual model."""

    visual = record.metadata.get("visual")
    if not isinstance(visual, Mapping) or visual.get("type") != "causal_comparison":
        return []

    current_items = visual.get("current_items")
    proposed_routes = visual.get("proposed_routes")
    if not isinstance(current_items, list) or not isinstance(proposed_routes, list):
        return []

    lines = [
        '<figure class="hub-causal-comparison">',
        '<section class="hub-causal-current">',
        f"<h3>{html.escape(str(visual.get('current_label', '')))}</h3>",
        f"<p>{html.escape(str(visual.get('current_summary', '')))}</p>",
        "<ul>",
    ]
    lines.extend(f"<li>{html.escape(str(item))}</li>" for item in current_items)
    lines.extend(
        [
            "</ul>",
            "</section>",
            '<div class="hub-causal-change">'
            f"{html.escape(str(visual.get('change_label', '')))}</div>",
            '<section class="hub-causal-proposed">',
            f"<h3>{html.escape(str(visual.get('proposed_label', '')))}</h3>",
            f"<p>{html.escape(str(visual.get('proposed_summary', '')))}</p>",
            "<dl>",
        ]
    )
    for route in proposed_routes:
        if not isinstance(route, Mapping):
            continue
        lines.extend(
            [
                "<div>",
                f"<dt>{html.escape(str(route.get('label', '')))}</dt>",
                f"<dd>{html.escape(str(route.get('action', '')))}</dd>",
                "</div>",
            ]
        )
    lines.extend(
        [
            "</dl>",
            "</section>",
            (
                '<figcaption class="hub-decision-test">'
                f"{html.escape(str(visual.get('decision_test', '')))}</figcaption>"
            ),
            "</figure>",
        ]
    )
    return lines


def _render_test_page(build: HubBuild, test: HubRecord) -> str:
    mechanism_steps = _metadata_list(test, "mechanism_steps")
    missing_fields = _metadata_list(test, "missing_readiness_fields")
    stage = html.escape(_human_label(test.metadata.get("stage", "")))
    priority_value = test.metadata.get("priority")
    if isinstance(priority_value, str) and priority_value.strip():
        priority = f"{html.escape(_human_label(priority_value))} priority"
    else:
        priority = "Priority not specified"
    readiness = html.escape(_test_readiness_count(test))
    lines = [
        "<!-- Generated by scripts/build_research_hub.py; do not edit. -->",
        "",
        f"# {html.escape(test.title)}",
        "",
        "::: {.hub-record-hero .hub-idea-hero}",
        f'<p class="hub-idea-status">{stage} · {priority} · {readiness}</p>',
        "",
        f'<p class="hub-takeaway">{html.escape(str(test.metadata.get("plain_english", "")))}</p>',
        f"<p>{html.escape(str(test.metadata.get('why_it_matters', '')))}</p>",
    ]
    github_issue_link = _render_github_issue_link(test)
    if github_issue_link is not None:
        lines.extend(["", f'<p class="hub-github-issue">{github_issue_link}</p>'])
    visual = _render_record_visual(test)
    if visual:
        lines.extend(["", *visual])
    lines.extend(
        [
            "",
            (
                '<p class="hub-next-action">'
                f"{html.escape(str(test.metadata.get('next_action', '')))}</p>"
            ),
            ":::",
            "",
        ]
    )
    evidence_lines = [
        "## How does it work?",
        "",
        "- " + "\n- ".join(html.escape(step) for step in mechanism_steps),
        "",
        "## What do we know?",
        "",
        html.escape(test.body),
    ]
    lines.extend(_render_details("Evidence and current knowledge", evidence_lines))
    readiness_lines = ["## What is missing?", ""]
    readiness_lines.extend(
        f"- {html.escape(_human_label(field))}" for field in missing_fields
    )
    if not missing_fields:
        readiness_lines.append("- No readiness fields are currently missing.")
    lines.extend(_render_details("Limits and readiness", readiness_lines))
    relationship_lines = _render_test_related_research(build, test)
    lines.extend(
        _render_details("Relationships and technical details", relationship_lines)
    )
    return "\n".join(lines)


def render_test_pages(build: HubBuild) -> dict[Path, str]:
    """Render one explanation page per canonical test record."""

    return {
        _validate_output_path(
            Path(f"research/hub/generated/ideas/{test.record_id}.qmd")
        ): _render_test_page(build, test).rstrip() + "\n"
        for test in sorted(build.sources.tests, key=lambda row: row.record_id)
    }


def _study_question_titles(build: HubBuild, study: HubRecord) -> str:
    questions_by_id = {
        question.record_id: question for question in build.sources.questions
    }
    titles = [
        questions_by_id[question_id].title
        for question_id in _metadata_list(study, "question_ids")
        if question_id in questions_by_id
    ]
    return "; ".join(titles)


def render_studies_index_qmd(build: HubBuild) -> str:
    """Render the study library grouped by primary theme and study type."""

    themes_by_id = {theme.record_id: theme for theme in build.sources.themes}
    studies = sorted(
        build.sources.studies,
        key=lambda row: (
            str(row.metadata.get("primary_theme_id", "")),
            str(row.metadata.get("study_type", "")),
            row.record_id,
        ),
    )
    lines = [
        *_development_evidence_notice(),
        "",
        "A study is one decision-relevant experiment, audit, or review. "
        "Open a row for its full evidence and limitations.",
    ]
    current_theme: str | None = None
    current_study_type: str | None = None
    for study in studies:
        primary_theme = str(study.metadata.get("primary_theme_id", ""))
        study_type = str(study.metadata.get("study_type", ""))
        if primary_theme != current_theme:
            theme = themes_by_id.get(primary_theme)
            theme_title = theme.title if theme is not None else primary_theme
            lines.extend(
                [
                    "",
                    f"## {theme_title}",
                    "",
                    f'<div id="theme-{html.escape(primary_theme, quote=True)}"></div>',
                ]
            )
            current_theme = primary_theme
            current_study_type = None
        if study_type != current_study_type:
            lines.extend(["", f"### {_human_label(study_type)}", ""])
            current_study_type = study_type
        lines.extend(
            [
                (
                    '<a class="hub-study-row" '
                    f'data-study-id="{html.escape(study.record_id, quote=True)}" '
                    f'data-themes="{html.escape(" ".join(_metadata_list(study, "theme_ids")), quote=True)}" '
                    f'href="{html.escape(study.record_id, quote=True)}.qmd">'
                ),
                (
                    "<span>"
                    f"{html.escape(_human_label(study.metadata.get('outcome', '')))} · "
                    f"{html.escape(_human_label(study.metadata.get('decision', '')))}"
                    "</span>"
                ),
                f"<strong>{html.escape(study.title)}</strong>",
                f"<span>{html.escape(str(study.metadata.get('conclusion', '')))}</span>",
                (
                    '<span class="hub-next-action"><strong>Practical decision</strong> '
                    f"{html.escape(str(study.metadata.get('decision_summary', '')))}</span>"
                ),
                "</a>",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _study_systems(
    build: HubBuild,
    study: HubRecord,
    field: str,
) -> list[SystemRecord]:
    systems_by_id = {system.system_id: system for system in build.ledger.systems}
    return [
        systems_by_id[system_id]
        for system_id in _metadata_list(study, field)
        if system_id in systems_by_id
    ]


def _render_study_representative_runs(
    build: HubBuild,
    study: HubRecord,
) -> list[str]:
    representatives = _study_systems(build, study, "representative_system_ids")
    lines = [
        "",
        "## Representative runs",
        "",
        "Rows follow the canonical representative-run order within each scope. "
        "Separate scope tables are not directly comparable with other evaluation "
        "scopes.",
    ]
    if not representatives:
        return [
            *lines,
            "",
            "This study has no representative ranking-system row in the ledger.",
        ]

    scopes = sorted({system.evaluation_scope for system in representatives})
    for scope in scopes:
        scoped = [
            system for system in representatives if system.evaluation_scope == scope
        ]
        suppress_effectiveness = scope in {
            "constructed_not_evaluated",
            "invalid_unscored",
        }
        lines.extend(
            [
                "",
                f"### Evaluation scope: {_inline_code(scope)}",
                "",
                _scope_guard(scoped[0]),
                "",
                "| Run | Method | Evidence | Promotion | nDCG@10 | MRR |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for system in scoped:
            system_id = _validate_system_id(system.system_id)
            ndcg = "—" if suppress_effectiveness else _format_metric(system.ndcg_at_10)
            mrr = "—" if suppress_effectiveness else _format_metric(system.mrr)
            lines.append(
                f"| [{_inline_code(system_id)}]"
                f"(../generated/runs/{quote(system_id, safe='@+._-')}.qmd) | "
                f"{_markdown_table_text(system.method_summary)} | "
                f"{_inline_code(system.evidence_status)} | "
                f"{_inline_code(system.promotion_status)} | {ndcg} | {mrr} |"
            )
    return lines


def _render_study_artifact_custody(
    build: HubBuild,
    study: HubRecord,
) -> list[str]:
    systems = _study_systems(build, study, "system_ids")
    lines = ["", "## Artifact custody", ""]
    if not systems:
        return [
            *lines,
            "No ranking-system row is linked. Custody is defined by the canonical "
            "evidence records below.",
        ]
    counts: dict[str, int] = {}
    for system in systems:
        counts[system.artifact_state] = counts.get(system.artifact_state, 0) + 1
    lines.extend(
        [
            f"{_count_label(len(systems), 'ledger run')} linked to this study.",
            "",
            "| Artifact state | Linked runs |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {_inline_code(state)} | {counts[state]} |" for state in sorted(counts)
    )
    return lines


def _render_study_relationships(
    build: HubBuild,
    study: HubRecord,
) -> list[str]:
    question_ids = set(_metadata_list(study, "question_ids"))
    questions = [
        question
        for question in build.sources.questions
        if question.record_id in question_ids
    ]
    related_studies = [
        candidate
        for candidate in build.sources.studies
        if candidate.record_id != study.record_id
        and question_ids.intersection(_metadata_list(candidate, "question_ids"))
    ]
    lines = ["", "**Questions**", ""]
    lines.extend(
        (
            f"- [{_markdown_table_text(question.title)}]"
            f"(../questions/{question.record_id}.qmd)"
        )
        for question in sorted(questions, key=lambda row: row.record_id)
    )
    if not questions:
        lines.append("- No linked question.")
    lines.extend(["", "**Related studies**", ""])
    lines.extend(
        (f"- [{_markdown_table_text(candidate.title)}]({candidate.record_id}.qmd)")
        for candidate in sorted(related_studies, key=lambda row: row.record_id)
    )
    if not related_studies:
        lines.append("- No other study addresses the same question.")
    return lines


def _render_study_detail(build: HubBuild, study: HubRecord) -> str:
    question_ids = _metadata_list(study, "question_ids")
    questions = [
        question
        for question in build.sources.questions
        if question.record_id in question_ids
    ]
    tests = _tests_for_questions(build, question_ids)
    action = _next_action(tests)
    question_links = "; ".join(
        f"[{_markdown_table_text(question.title)}]"
        f"(../questions/{question.record_id}.qmd)"
        for question in sorted(questions, key=lambda row: row.record_id)
    )
    status = (
        f"{_human_label(study.metadata.get('outcome', ''))} · "
        f"{_human_label(study.metadata.get('decision', ''))}"
    )
    lines = [
        "<!-- Generated by scripts/build_research_hub.py; do not edit. -->",
        "",
        '<section class="hub-record-hero">',
        f'<p class="hub-record-status">{html.escape(status)}</p>',
        f'<p class="hub-takeaway">{html.escape(str(study.metadata.get("objective", "")))}</p>',
        f"<p>{html.escape(str(study.metadata.get('conclusion', '')))}</p>",
    ]
    visual = _render_record_visual(study)
    if visual:
        lines.extend(["", *visual])
    lines.extend(
        [
            "",
            (
                '<p class="hub-next-action">'
                f"{_render_action_link(action, href_prefix='../generated/ideas')}"
                "</p>"
            ),
            "</section>",
        ]
    )
    hypothesis = study.metadata.get("hypothesis")
    evidence_lines = [
        "## Why does it matter?",
        "",
        f"**Rationale:** {study.metadata.get('rationale', '')}",
        "",
        f"**Research question:** {question_links or 'No linked question.'}",
        "",
        "## How does it work?",
        "",
    ]
    if hypothesis:
        evidence_lines.extend([f"**Hypothesis:** {hypothesis}", ""])
    evidence_lines.extend(
        [
            str(study.metadata.get("protocol_summary", "")),
            "",
            "Any run comparison below is partitioned by ledger evaluation scope.",
            "",
            "## What do we know?",
            "",
            f"**Outcome:** {_inline_code(study.metadata.get('outcome', 'unknown'))}",
            "",
            str(study.metadata.get("conclusion", "")),
            "",
            (
                f"**Decision:** "
                f"{_inline_code(study.metadata.get('decision', 'unknown'))}"
            ),
            "",
            str(study.metadata.get("decision_summary", "")),
            "",
            (
                f"Rationale provenance: "
                f"{_inline_code(study.metadata.get('rationale_provenance', 'unknown'))}."
            ),
        ]
    )
    lines.extend(_render_details("Evidence and current knowledge", evidence_lines))
    lines.extend(
        _render_details(
            "Limits and readiness",
            [
                str(study.metadata.get("limitations", "")),
                "",
                "Development results do not establish hidden-test effectiveness.",
            ],
        )
    )
    relationship_lines = [
        "## Authored decision notes",
        "",
        _authored_body(study),
        "",
        "## Next tests",
        "",
        *(
            f"- [{_markdown_table_text(test.title)}]"
            f"(../generated/ideas/{test.record_id}.qmd): "
            f"{_markdown_table_text(test.metadata.get('next_action', ''))}"
            for test in tests
        ),
        *(
            ["- No next test is linked through the study questions."]
            if not tests
            else []
        ),
        "",
        *_render_study_relationships(build, study),
        *_render_study_representative_runs(build, study),
        *_render_study_artifact_custody(build, study),
        "",
        "## Canonical evidence",
        "",
        "Repository-root evidence paths:",
    ]
    evidence_refs = _metadata_list(study, "evidence_refs")
    relationship_lines.extend(
        f"- {_path_code(reference)}" for reference in evidence_refs
    )
    if not evidence_refs:
        relationship_lines.append("- No canonical evidence path is recorded.")
    lines.extend(
        _render_details("Relationships and technical details", relationship_lines)
    )
    return "\n".join(lines)


def render_study_detail_pages(build: HubBuild) -> dict[Path, str]:
    """Render relationship- and ledger-derived fragments for study pages."""

    return {
        _validate_output_path(
            Path(f"research/hub/generated/studies/{study.record_id}.md")
        ): _render_study_detail(build, study).rstrip() + "\n"
        for study in sorted(build.sources.studies, key=lambda row: row.record_id)
    }


def _render_coverage_matrix(build: HubBuild) -> str:
    approaches = sorted(build.sources.approaches, key=lambda row: row.record_id)
    lines = [
        "---",
        'title: "Research Question Coverage"',
        "---",
        "",
        "# Research Question Coverage",
        "",
        "Development evidence only.",
        "",
        "| Question | "
        + " | ".join(_inline_code(approach.record_id) for approach in approaches)
        + " |",
        "|---|" + "---|" * len(approaches),
    ]
    questions = sorted(
        build.sources.questions,
        key=lambda row: (
            _QUESTION_STATUS_PRIORITY.get(
                str(row.metadata.get("status")),
                len(_QUESTION_STATUS_PRIORITY),
            ),
            row.record_id,
        ),
    )
    for question in questions:
        cells = [
            _coverage_cell(build, question.record_id, approach.record_id)
            for approach in approaches
        ]
        lines.append(
            f"| **{_markdown_table_text(question.title)}** "
            f"({_inline_code(question.record_id)}) | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def _study_evidence_states(build: HubBuild, study: HubRecord) -> set[str]:
    outcome = str(study.metadata.get("outcome", "unknown"))
    if outcome in {"invalid", "unscored"}:
        return {outcome}
    system_by_id = {system.system_id: system for system in build.ledger.systems}
    systems = [
        system_by_id[system_id]
        for system_id in _metadata_list(study, "system_ids")
        if system_id in system_by_id
    ]
    if not systems:
        return {outcome}
    states: set[str] = set()
    for system in systems:
        if system.evidence_status == "invalid":
            states.add("invalid")
        elif system.evidence_status == "unscored":
            states.add("unscored")
        elif system.evaluation_scope in _RESTRICTED_SCOPES:
            states.add(f"restricted {outcome}")
        else:
            states.add(outcome)
    return states


def _coverage_cell(
    build: HubBuild,
    question_id: str,
    approach_id: str,
) -> str:
    studies = [
        study
        for study in build.sources.studies
        if question_id in _metadata_list(study, "question_ids")
        and approach_id in _metadata_list(study, "approach_ids")
    ]
    if not studies:
        return "uncovered"
    states = sorted(
        {state for study in studies for state in _study_evidence_states(build, study)}
    )
    count = len(studies)
    noun = "study" if count == 1 else "studies"
    if len(states) == 1:
        return f"{_markdown_table_text(states[0])} ({count} {noun})"
    return f"mixed ({count} {noun}): " + ", ".join(
        _markdown_table_text(state) for state in states
    )


def render_coverage_qmd(build: HubBuild) -> str:
    return _render_coverage_matrix(build).rstrip() + "\n"


def _date_sort_value(value: object) -> tuple[int, int]:
    if isinstance(value, date):
        return (0, -value.toordinal())
    if isinstance(value, str) and value != "unknown":
        try:
            return (0, -date.fromisoformat(value).toordinal())
        except ValueError:
            pass
    return (1, 0)


def _render_decision_history(build: HubBuild) -> str:
    lines = [
        "---",
        'title: "Decision History"',
        "---",
        "",
        "# Decision History",
        "",
        "Development evidence only.",
    ]
    dated = sorted(
        (
            study
            for study in build.sources.studies
            if _date_sort_value(study.metadata.get("concluded_on"))[0] == 0
        ),
        key=lambda row: (
            _history_date_sort_value(row.metadata.get("concluded_on")),
            row.record_id,
        ),
    )
    unknown = sorted(
        (
            study
            for study in build.sources.studies
            if _date_sort_value(study.metadata.get("concluded_on"))[0] == 1
        ),
        key=lambda row: row.record_id,
    )
    lines.extend(_history_section(build, "Dated decisions", dated))
    lines.extend(_history_section(build, "Date unknown", unknown))
    lines.extend(_question_history_snapshot(build))
    return "\n".join(lines)


def _question_history_snapshot(build: HubBuild) -> list[str]:
    lines = [
        "",
        "## Current question snapshots",
        "",
        "These rows are regenerated from canonical question records; they are "
        "current states, not additional dated decisions.",
        "",
        "| Updated | Question | Status | Current bounded answer | Confidence boundary |",
        "|---|---|---|---|---|",
    ]
    for question in sorted(
        build.sources.questions,
        key=lambda row: (
            _history_date_sort_value(row.metadata.get("updated_on")),
            row.record_id,
        ),
    ):
        updated = question.metadata.get("updated_on", "unknown")
        if isinstance(updated, date):
            updated = updated.isoformat()
        lines.append(
            f"| {_markdown_table_text(updated)} | "
            f"**{_markdown_table_text(question.title)}** "
            f"({_inline_code(question.record_id)}) | "
            f"{_inline_code(question.metadata.get('status', 'unknown'))} | "
            f"{_markdown_table_text(question.metadata.get('current_answer', ''))} | "
            f"{_markdown_table_text(question.metadata.get('confidence_boundary', ''))} |"
        )
    if not build.sources.questions:
        lines.append("| — | — | — | — | — |")
    return lines


def _history_date_sort_value(value: object) -> int:
    if isinstance(value, date):
        return value.toordinal()
    if isinstance(value, str) and value != "unknown":
        try:
            return date.fromisoformat(value).toordinal()
        except ValueError:
            pass
    return date.max.toordinal()


def _history_section(
    build: HubBuild,
    title: str,
    studies: Sequence[HubRecord],
) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Date | Study | Active questions | What changed | Decision | Next study |",
        "|---|---|---|---|---|---|",
    ]
    for study in studies:
        concluded = study.metadata.get("concluded_on") or "—"
        if isinstance(concluded, date):
            concluded = concluded.isoformat()
        active_question_ids = {
            question.record_id
            for question in build.sources.questions
            if question.metadata.get("status") in {"open", "partially_answered"}
        }
        active = sorted(
            set(_metadata_list(study, "question_ids")) & active_question_ids
        )
        active_text = ", ".join(_inline_code(value) for value in active) or "—"
        next_text = _supported_next_studies(build, study)
        lines.append(
            f"| {_markdown_table_text(concluded)} | "
            f"**{_markdown_table_text(study.title)}** "
            f"({_inline_code(study.record_id)}) | "
            f"{active_text} | "
            f"{_markdown_table_text(study.metadata.get('conclusion', ''))} | "
            f"{_inline_code(study.metadata.get('decision', 'unknown'))}: "
            f"{_markdown_table_text(study.metadata.get('decision_summary', ''))} | "
            f"{next_text} |"
        )
    if not studies:
        lines.append("| — | — | — | — | — | — |")
    return lines


def _supported_next_studies(build: HubBuild, study: HubRecord) -> str:
    by_id = {candidate.record_id: candidate for candidate in build.sources.studies}
    supported = [
        by_id[next_id]
        for next_id in sorted(_metadata_list(study, "superseded_by"))
        if next_id in by_id
        and study.record_id in _metadata_list(by_id[next_id], "supersedes")
    ]
    if not supported:
        return "—"
    return "; ".join(
        f"Next study {_inline_code(candidate.record_id)}: "
        f"{_markdown_table_text(candidate.metadata.get('rationale', ''))}"
        for candidate in supported
    )


def render_history_qmd(build: HubBuild) -> str:
    return _render_decision_history(build).rstrip() + "\n"


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _render_run_index(build: HubBuild) -> str:
    promotions = sorted({system.promotion_status for system in build.ledger.systems})
    scopes = sorted({system.evaluation_scope for system in build.ledger.systems})
    lines = [
        "---",
        'title: "Experiment Runs"',
        "---",
        "",
        "# Experiment Runs",
        "",
        "Development evidence only. Compare metrics only within the same evaluation scope.",
        "",
        '::: {data-hub-filter-group=""}',
        '<div class="hub-filter-controls">',
        '<label>Search runs<input type="search" data-hub-filter '
        'placeholder="System, family, or method"></label>',
        '<label>Promotion<select data-hub-status><option value="">All</option>',
        *(
            f'<option value="{html.escape(value, quote=True)}">'
            f"{html.escape(value)}</option>"
            for value in promotions
        ),
        "</select></label>",
        '<label>Evaluation scope<select data-hub-scope><option value="">All</option>',
        *(
            f'<option value="{html.escape(value, quote=True)}">'
            f"{html.escape(value)}</option>"
            for value in scopes
        ),
        "</select></label>",
        "</div>",
        '<p class="visually-hidden" aria-live="polite" data-hub-filter-live></p>',
    ]
    for scope in scopes:
        lines.extend(["", f"## Evaluation scope: {_inline_code(scope)}"])
        evidence_classes = sorted(
            {
                system.evidence_status
                for system in build.ledger.systems
                if system.evaluation_scope == scope
            }
        )
        for evidence_status in evidence_classes:
            systems = sorted(
                (
                    system
                    for system in build.ledger.systems
                    if system.evaluation_scope == scope
                    and system.evidence_status == evidence_status
                ),
                key=lambda row: row.system_id,
            )
            lines.extend(
                [
                    "",
                    f"### Evidence: {_inline_code(evidence_status)}",
                    "",
                    '<div class="hub-grid">',
                ]
            )
            suppress_effectiveness = evidence_status in {"invalid", "unscored"}
            for system in systems:
                system_id = _validate_system_id(system.system_id)
                search_text = " ".join(
                    (
                        system_id,
                        system.experiment_family,
                        system.method_summary,
                    )
                ).lower()
                lines.extend(
                    [
                        (
                            '<article class="hub-card hub-filter-row" '
                            "data-hub-filter-row "
                            f'data-status="{html.escape(system.promotion_status, quote=True)}" '
                            f'data-scope="{html.escape(system.evaluation_scope, quote=True)}" '
                            f'data-search="{html.escape(search_text, quote=True)}">'
                        ),
                        (
                            f'<h4><a href="../generated/runs/'
                            f'{html.escape(system_id, quote=True)}.html">'
                            f"<code>{html.escape(system_id)}</code></a></h4>"
                        ),
                        (
                            f"<p>{html.escape(system.promotion_status)} · "
                            f"{html.escape(system.evaluation_scope)} · "
                            f"{html.escape(system.evidence_status)}</p>"
                        ),
                        f"<p>{html.escape(system.method_summary)}</p>",
                    ]
                )
                if suppress_effectiveness:
                    lines.append(
                        '<p class="hub-muted">Effectiveness suppressed for '
                        "invalid or unscored evidence.</p>"
                    )
                else:
                    lines.append(
                        f"<p>nDCG@10 {_format_metric(system.ndcg_at_10)} · "
                        f"MRR {_format_metric(system.mrr)}</p>"
                    )
                lines.append("</article>")
            lines.append("</div>")
    lines.append(":::")
    return "\n".join(lines)


def render_runs_qmd(build: HubBuild) -> str:
    for system in build.ledger.systems:
        _linked_study_ids(build, system.system_id)
    return _render_run_index(build).rstrip() + "\n"


def _linked_study_ids(build: HubBuild, system_id: str) -> list[str]:
    study_ids = sorted(
        study.record_id
        for study in build.sources.studies
        if system_id in _metadata_list(study, "system_ids")
    )
    if not study_ids:
        raise ValueError(
            "orphan ledger run must be linked from at least one canonical "
            f"study system_ids list: {system_id!r}"
        )
    return study_ids


def _scope_guard(system: SystemRecord) -> str:
    if system.evaluation_scope in _RESTRICTED_SCOPES:
        return (
            "This restricted or invalid evaluation scope is not directly comparable "
            "with full development runs."
        )
    if system.evaluation_scope == "constructed_not_evaluated":
        return "This run was constructed but not evaluated; no effectiveness claim is supported."
    return "Development evidence only. No held-out effectiveness claim is supported."


def _render_run_page(build: HubBuild, system: SystemRecord) -> str:
    system_id = _validate_system_id(system.system_id)
    lines = [
        "---",
        f"title: {_yaml_scalar(system_id)}",
        "---",
        "",
        f"# {_inline_code(system_id)}",
        "",
        _scope_guard(system),
        "",
        "## Evidence contract",
        "",
        f"- **Method:** {_markdown_table_text(system.method_summary)}",
        f"- **source_system_id:** {_inline_code(system.source_system_id)}",
        f"- **Experiment family:** {_inline_code(system.experiment_family)}",
        f"- **Evaluation scope:** {_inline_code(system.evaluation_scope)}",
        f"- **Evidence status:** {_inline_code(system.evidence_status)}",
        f"- **Promotion status:** {_inline_code(system.promotion_status)}",
        f"- **Artifact state:** {_inline_code(system.artifact_state)}",
        f"- **Qrel usage:** {_inline_code(system.qrel_usage)}",
        (
            "- **per_query_available:** "
            f"{_inline_code(str(system.per_query_available).lower())}"
        ),
        "",
        "## Evaluation",
        "",
        f"- **Queries:** {system.query_count if system.query_count is not None else '—'}",
        (
            "- **Candidates per query:** "
            f"{system.candidate_count if system.candidate_count is not None else '—'}"
        ),
        f"- **nDCG@10:** {_format_metric(system.ndcg_at_10)}",
        f"- **MRR:** {_format_metric(system.mrr)}",
        (
            "- **Zero-nDCG queries:** "
            f"{system.zero_ndcg_queries if system.zero_ndcg_queries is not None else '—'}"
        ),
        "",
        "## Linked studies",
        "",
    ]
    linked_studies = _linked_study_ids(build, system_id)
    lines.extend(f"- {_inline_code(study_id)}" for study_id in linked_studies)
    lines.extend(
        [
            "",
            "## Evidence receipt",
            "",
            f"- **Source:** {_inline_code(system.source_path)}",
            f"- **SHA-256:** {_inline_code(system.source_sha256)}",
            f"- **Ranking coverage:** {_inline_code(system.ranking_coverage)}",
        ]
    )
    if system.blocker:
        lines.append(f"- **Blocker:** {_markdown_table_text(system.blocker)}")
    evidence_links = _run_evidence_links(build, system)
    if evidence_links:
        lines.extend(["", "## Evidence indexes", "", *evidence_links])
    lines.extend(["", "## Caveats", ""])
    if system.caveat_ids:
        lines.extend(
            f"- {_inline_code(caveat_id)}" for caveat_id in sorted(system.caveat_ids)
        )
    else:
        lines.append("No ledger caveats are attached.")
    return "\n".join(lines)


def _run_evidence_links(build: HubBuild, system: SystemRecord) -> list[str]:
    links: list[str] = []
    if system.per_query_available:
        links.append(
            "- **Per-query evidence:** `reports/trec_2026_experiment_per_query.tsv`"
        )
    if any(row.system_id == system.system_id for row in build.ledger.topic_slices):
        links.append(
            "- **Relevant-candidate slice evidence:** "
            "`reports/trec_2026_experiment_topic_slices.tsv`"
        )
    if any(row.system_id == system.system_id for row in build.ledger.density_slices):
        links.append(
            "- **Positive-density slice evidence:** "
            "`reports/trec_2026_experiment_density_slices.tsv`"
        )
    return links


def render_run_pages(build: HubBuild) -> dict[Path, str]:
    pages: dict[Path, str] = {}
    for system in sorted(build.ledger.systems, key=lambda row: row.system_id):
        system_id = _validate_system_id(system.system_id)
        path = _validate_output_path(
            Path(f"research/hub/generated/runs/{system_id}.qmd")
        )
        pages[path] = _render_run_page(build, system).rstrip() + "\n"
    return pages


def render_hub_outputs(build: HubBuild) -> Mapping[Path, str]:
    """Render every tracked generated output without touching the filesystem."""

    for record in build.sources.all_records():
        validate_record_identity(record)
    return _validated_output_mapping(
        {
            Path("research/catalog.json"): render_catalog(build),
            Path("research/hub/catalog.json"): render_catalog(build),
            Path("reports/trec_2026_research_hub_summary.md"): render_summary_markdown(
                build
            ),
            Path("research/hub/generated/summary.qmd"): render_summary_qmd(build),
            Path("research/hub/generated/current-state.qmd"): (
                render_current_state_qmd(build)
            ),
            Path("research/hub/generated/failure-library.qmd"): (
                render_failure_library_qmd(build)
            ),
            Path(
                "research/hub/generated/background-index.qmd"
            ): render_background_index_qmd(build),
            Path("research/hub/generated/data-index.qmd"): render_data_index_qmd(build),
            Path("research/hub/generated/ideas-index.qmd"): render_tests_index_qmd(
                build
            ),
            Path("research/hub/generated/themes-index.qmd"): render_themes_index_qmd(
                build
            ),
            Path(
                "research/hub/generated/questions-index.qmd"
            ): render_questions_index_qmd(build),
            Path("research/hub/generated/studies-index.qmd"): render_studies_index_qmd(
                build
            ),
            Path("research/hub/generated/coverage.qmd"): render_coverage_qmd(build),
            Path("research/hub/generated/history.qmd"): render_history_qmd(build),
            Path("research/hub/generated/runs.qmd"): render_runs_qmd(build),
            **render_theme_pages(build),
            **render_test_pages(build),
            **render_question_detail_pages(build),
            **render_study_detail_pages(build),
            **render_run_pages(build),
        }
    )


def _verified_staged_outputs(
    outputs: Mapping[Path, str],
    staging_dir: Path,
) -> dict[Path, Path]:
    staged_paths: dict[Path, Path] = {}
    validated = _validated_output_mapping(outputs)
    for relative_path in sorted(validated):
        content = validated[relative_path]
        staged = staging_dir / relative_path
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(content, encoding="utf-8", newline="")
        if staged.read_bytes() != content.encode("utf-8"):
            raise OSError(f"staged output verification failed: {relative_path}")
        staged_paths[relative_path] = staged
    return staged_paths


def write_hub_outputs(build: HubBuild, root: Path) -> tuple[Path, ...]:
    """Atomically publish all hub outputs, rolling back a partial promotion."""

    outputs = _validated_output_mapping(render_hub_outputs(build))
    root = Path(root)
    relative_paths = tuple(sorted(outputs))
    destinations = tuple(_confined_destination(root, path) for path in relative_paths)
    unexpected_paths = tuple(sorted(_tracked_output_paths(root) - set(relative_paths)))
    for path in unexpected_paths:
        _confined_destination(root, path)

    with tempfile.TemporaryDirectory(prefix=".research-hub-", dir=root) as temporary:
        staging_dir = Path(temporary)
        staged_paths = _verified_staged_outputs(outputs, staging_dir)
        backup_dir = staging_dir / "previous"
        installed: list[Path] = []
        removed: list[Path] = []
        try:
            for relative_path in unexpected_paths:
                source = _confined_destination(root, relative_path)
                backup = backup_dir / relative_path
                backup.parent.mkdir(parents=True, exist_ok=True)
                source.replace(backup)
                removed.append(relative_path)
            for relative_path in relative_paths:
                destination = _confined_destination(root, relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup = backup_dir / relative_path
                if destination.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(backup)
                try:
                    staged_paths[relative_path].replace(destination)
                except Exception:
                    if backup.exists():
                        backup.replace(destination)
                    raise
                installed.append(relative_path)
        except Exception:
            for relative_path in reversed(installed):
                destination = _confined_destination(root, relative_path)
                destination.unlink(missing_ok=True)
                backup = backup_dir / relative_path
                if backup.exists():
                    backup.replace(destination)
            for relative_path in reversed(removed):
                backup = backup_dir / relative_path
                destination = _confined_destination(root, relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup.replace(destination)
            raise
    return destinations


def _tracked_output_paths(root: Path) -> set[Path]:
    paths = set(_STATIC_OUTPUT_PATHS)
    generated = root / "research/hub/generated"
    if generated.is_dir():
        paths.update(
            path.relative_to(root)
            for path in generated.rglob("*.qmd")
            if path.is_file()
        )
        paths.update(
            path.relative_to(root) for path in generated.rglob("*.md") if path.is_file()
        )
    return paths


def assert_hub_outputs_current(build: HubBuild, root: Path) -> None:
    """Report every missing, unexpected, or byte-different generated output."""

    outputs = _validated_output_mapping(render_hub_outputs(build))
    root = Path(root)
    for path in outputs:
        _confined_destination(root, path)
    expected_paths = set(outputs)
    tracked_paths = _tracked_output_paths(root)
    for path in tracked_paths:
        _confined_destination(root, path)
    missing = sorted(path for path in expected_paths if not (root / path).is_file())
    unexpected = sorted(tracked_paths - expected_paths)
    different = sorted(
        path
        for path in expected_paths - set(missing)
        if (root / path).read_bytes() != outputs[path].encode("utf-8")
    )
    if missing or unexpected or different:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(map(str, missing))}")
        if unexpected:
            details.append(f"unexpected: {', '.join(map(str, unexpected))}")
        if different:
            details.append(f"different: {', '.join(map(str, different))}")
        raise ValueError("research hub outputs are not current; " + "; ".join(details))
