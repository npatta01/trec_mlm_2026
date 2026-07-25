"""Deterministic renderers and atomic publication for research-hub views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from .validate import READY_FIELDS

_QUESTION_STATUS_PRIORITY = {
    "open": 0,
    "partially_answered": 1,
    "parked": 2,
    "answered": 3,
}
_STATIC_OUTPUT_PATHS = (
    Path("reports/trec_2026_research_hub_summary.md"),
    Path("research/catalog.json"),
    Path("research/hub/generated/coverage.qmd"),
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
        "question_ids",
        "study_ids",
        "next_test_ids",
        "approach_ids",
        "dataset_ids",
    )
    return sorted(
        {value for field in fields for value in _metadata_list(record, field)}
    )


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
            "related_ids": _related_ids(record),
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
        questions = _records_for_theme(build.sources.questions, theme.record_id)
        studies = _records_for_theme(build.sources.studies, theme.record_id)
        lines.extend(
            [
                "",
                (
                    '<a class="hub-theme-row" '
                    f'href="{href_prefix}/'
                    f'{html.escape(theme.record_id, quote=True)}.qmd">'
                ),
                f"<strong>{html.escape(theme.title)}</strong>",
                (
                    "<span>"
                    f"{html.escape(str(theme.metadata.get('current_conclusion', '')))}"
                    "</span>"
                ),
                (
                    "<span>"
                    f"{html.escape(_human_label(theme.metadata.get('decision', '')))}"
                    " · "
                    f"{html.escape(_human_label(theme.metadata.get('confidence', '')))} "
                    "confidence · "
                    f"{_count_label(len(questions), 'question')} · "
                    f"{_count_label(len(studies), 'study')}"
                    "</span>"
                ),
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
        present_ids = tuple(record_id for record_id in record_ids if record_id in records)
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
            "Evidence gathered from candidate models under bounded project "
            "protocols.",
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
            record
            for record in datasets
            if record.metadata.get("category") == category
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


def _theme_next_move(build: HubBuild, theme: HubRecord) -> str:
    tests_by_id = {test.record_id: test for test in build.sources.tests}
    tests = [
        tests_by_id[test_id]
        for test_id in _metadata_list(theme, "next_test_ids")
        if test_id in tests_by_id
    ]
    if not tests:
        return "No next test is currently linked."
    return "; ".join(
        f'<a href="../../ideas/{html.escape(test.record_id, quote=True)}.qmd">'
        f"{html.escape(test.title)}</a>: "
        f"{html.escape(str(test.metadata.get('summary', '')))}"
        for test in tests
    )


def _render_theme_page(build: HubBuild, theme: HubRecord) -> str:
    questions = _records_for_theme(build.sources.questions, theme.record_id)
    studies = _records_for_theme(build.sources.studies, theme.record_id)
    theme_id = quote(theme.record_id, safe="")
    lines = [
        "---",
        f"title: {_yaml_scalar(theme.title)}",
        f"description: {_yaml_scalar(theme.metadata.get('plain_english', ''))}",
        "toc: false",
        "---",
        "",
        "::: {.hub-conclusion}",
        "## Current conclusion",
        "",
        str(theme.metadata.get("current_conclusion", "")),
        ":::",
        "",
        '<div class="hub-decision-strip">',
        '<div><strong>Practical decision</strong><br>',
        html.escape(_human_label(theme.metadata.get("decision", ""))),
        "</div>",
        '<div><strong>Confidence and evidence boundary</strong><br>',
        (
            f"{html.escape(_human_label(theme.metadata.get('confidence', '')))} "
            f"confidence. "
            f"{html.escape(str(theme.metadata.get('confidence_boundary', '')))}"
        ),
        "</div>",
        '<div><strong>Next move</strong><br>',
        _theme_next_move(build, theme),
        "</div>",
        "</div>",
        "",
        "## Questions in this theme",
    ]
    if not questions:
        lines.extend(["", "No research question is currently linked."])
    for question in questions:
        open_subquestions = _metadata_list(question, "open_subquestions")
        lines.extend(
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
            lines.extend(
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
        lines.append("</div>")
    lines.extend(
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
                f"<strong>{html.escape(question.title)}</strong>",
                f"<span>{html.escape(str(question.metadata.get('plain_english', '')))}</span>",
                (
                    "<span><strong>Current bounded answer</strong> "
                    f"{html.escape(str(question.metadata.get('current_answer', '')))}</span>"
                ),
                (
                    "<span>"
                    f"{html.escape(_human_label(question.metadata.get('status', '')))} · "
                    f"{html.escape(_human_label(question.metadata.get('confidence', '')))} "
                    "confidence"
                    "</span>"
                ),
                "</a>",
            ]
        )
    lines.append("</div>")
    return "\n".join(lines).rstrip() + "\n"


def _test_readiness_count(test: HubRecord) -> str:
    missing = set(_metadata_list(test, "missing_readiness_fields"))
    complete = len(set(READY_FIELDS) - missing)
    return f"{complete}/{len(READY_FIELDS)} readiness fields complete"


def render_tests_index_qmd(build: HubBuild) -> str:
    """Render the prospective-work library from canonical test records."""

    lines = [
        *_development_evidence_notice(),
        "",
        "Ideas are prospective work, not results. Open a row for the full "
        "explanation, evidence boundary, and remaining readiness work.",
        "",
        '<div class="hub-idea-list">',
    ]
    for test in sorted(build.sources.tests, key=lambda row: row.record_id):
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
                f"<strong>{html.escape(test.title)}</strong>",
                f"<span>{html.escape(str(test.metadata.get('plain_english', '')))}</span>",
                f"<span>{html.escape(str(test.metadata.get('summary', '')))}</span>",
                f"<span>{stage} · {priority} · {readiness}</span>",
                "</a>",
            ]
        )
    lines.append("</div>")
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
    lines = ["", "## Related research", "", "**Themes**", ""]
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


def _render_test_page(build: HubBuild, test: HubRecord) -> str:
    mechanism_steps = _metadata_list(test, "mechanism_steps")
    missing_fields = _metadata_list(test, "missing_readiness_fields")
    lines = [
        "<!-- Generated by scripts/build_research_hub.py; do not edit. -->",
        "",
        f"# {html.escape(test.title)}",
        "",
        "## What this idea means",
        "",
        html.escape(str(test.metadata.get("plain_english", ""))),
        "",
        "## Why it matters",
        "",
        html.escape(str(test.metadata.get("why_it_matters", ""))),
        "",
        "## How it would work",
        "",
    ]
    lines.extend(
        f"{index}. {html.escape(step)}"
        for index, step in enumerate(mechanism_steps, start=1)
    )
    lines.extend(
        [
            "",
            "## Current evidence",
            "",
            html.escape(test.body),
            "",
            "## What is missing",
            "",
        ]
    )
    lines.extend(
        f"- {html.escape(_human_label(field))}" for field in missing_fields
    )
    if not missing_fields:
        lines.append("- No readiness fields are currently missing.")
    lines.extend(
        [
            "",
            "## What happens next",
            "",
            html.escape(str(test.metadata.get("next_action", ""))),
        ]
    )
    lines.extend(_render_test_related_research(build, test))
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
        "Open a row for its full evidence and limitations."
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
        question_titles = _study_question_titles(build, study)
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
                (
                    "<span><strong>Pursuit question</strong> "
                    f"{html.escape(question_titles)}</span>"
                ),
                (
                    "<span><strong>Current result</strong> "
                    f"{html.escape(str(study.metadata.get('conclusion', '')))}</span>"
                ),
                (
                    "<span><strong>Practical decision</strong> "
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
            system
            for system in representatives
            if system.evaluation_scope == scope
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
        f"| {_inline_code(state)} | {counts[state]} |"
        for state in sorted(counts)
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
    next_test_ids = {
        test_id
        for question in questions
        for test_id in _metadata_list(question, "next_test_ids")
    }
    tests = [
        test
        for test in build.sources.tests
        if test.record_id in next_test_ids
    ]
    lines = ["", "## Related questions, studies, and next tests", ""]
    lines.extend(["**Questions**", ""])
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
        (
            f"- [{_markdown_table_text(candidate.title)}]"
            f"({candidate.record_id}.qmd)"
        )
        for candidate in sorted(related_studies, key=lambda row: row.record_id)
    )
    if not related_studies:
        lines.append("- No other study addresses the same question.")
    lines.extend(["", "**Next tests**", ""])
    lines.extend(
        (
            f"- [{_markdown_table_text(test.title)}]"
            f"(../ideas/{test.record_id}.qmd): "
            f"{_markdown_table_text(test.metadata.get('summary', ''))}"
        )
        for test in sorted(tests, key=lambda row: row.record_id)
    )
    if not tests:
        lines.append("- No next test is linked through the study questions.")
    return lines


def _render_study_detail(build: HubBuild, study: HubRecord) -> str:
    questions = [
        question
        for question in build.sources.questions
        if question.record_id in _metadata_list(study, "question_ids")
    ]
    question_links = "; ".join(
        f"[{_markdown_table_text(question.title)}]"
        f"(../questions/{question.record_id}.qmd)"
        for question in sorted(questions, key=lambda row: row.record_id)
    )
    lines = [
        "<!-- Generated by scripts/build_research_hub.py; do not edit. -->",
        "",
        "## Why this study was run",
        "",
        f"**Research question:** {question_links or 'No linked question.'}",
        "",
        str(study.metadata.get("objective", "")),
        "",
        f"**Rationale:** {study.metadata.get('rationale', '')}",
        "",
        "## Protocol and comparison",
        "",
    ]
    hypothesis = study.metadata.get("hypothesis")
    if hypothesis:
        lines.extend([f"**Hypothesis:** {hypothesis}", ""])
    lines.extend(
        [
            str(study.metadata.get("protocol_summary", "")),
            "",
            "Any run comparison below is partitioned by ledger evaluation scope.",
            "",
            "## Current result",
            "",
            f"**Outcome:** {_inline_code(study.metadata.get('outcome', 'unknown'))}",
            "",
            str(study.metadata.get("conclusion", "")),
            "",
            "## Decision and rationale",
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
            "",
            "## Evidence limits",
            "",
            str(study.metadata.get("limitations", "")),
            "",
            "Development results do not establish hidden-test effectiveness.",
            *_render_study_representative_runs(build, study),
            *_render_study_artifact_custody(build, study),
            "",
            "## Canonical evidence",
            "",
            "Repository-root evidence paths:",
        ]
    )
    evidence_refs = _metadata_list(study, "evidence_refs")
    lines.extend(f"- {_inline_code(reference)}" for reference in evidence_refs)
    if not evidence_refs:
        lines.append("- No canonical evidence path is recorded.")
    lines.extend(_render_study_relationships(build, study))
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
    promotions = sorted(
        {system.promotion_status for system in build.ledger.systems}
    )
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
        '<p class="visually-hidden" aria-live="polite" '
        "data-hub-filter-live></p>",
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
            "- **Per-query evidence:** "
            "`reports/trec_2026_experiment_per_query.tsv`"
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
            Path(
                "research/hub/generated/background-index.qmd"
            ): render_background_index_qmd(build),
            Path(
                "research/hub/generated/data-index.qmd"
            ): render_data_index_qmd(build),
            Path(
                "research/hub/generated/ideas-index.qmd"
            ): render_tests_index_qmd(build),
            Path(
                "research/hub/generated/themes-index.qmd"
            ): render_themes_index_qmd(build),
            Path(
                "research/hub/generated/questions-index.qmd"
            ): render_questions_index_qmd(build),
            Path(
                "research/hub/generated/studies-index.qmd"
            ): render_studies_index_qmd(build),
            Path("research/hub/generated/coverage.qmd"): render_coverage_qmd(build),
            Path("research/hub/generated/history.qmd"): render_history_qmd(build),
            Path("research/hub/generated/runs.qmd"): render_runs_qmd(build),
            **render_theme_pages(build),
            **render_test_pages(build),
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
        study_fragments = generated / "studies"
        if study_fragments.is_dir():
            paths.update(
                path.relative_to(root)
                for path in study_fragments.glob("*.md")
                if path.is_file()
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
