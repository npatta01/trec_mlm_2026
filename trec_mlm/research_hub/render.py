"""Deterministic renderers and atomic publication for research-hub views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import html
import json
from pathlib import Path
import tempfile

from trec_mlm.experiment_ledger import SystemRecord

from .build import HubBuild
from .model import HubRecord

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
    Path("research/hub/generated/history.qmd"),
    Path("research/hub/generated/runs.qmd"),
    Path("research/hub/generated/summary.qmd"),
)
_RESTRICTED_SCOPES = frozenset(
    {"restricted_12x20", "restricted_6", "invalid_unscored"}
)


def _markdown_text(value: object) -> str:
    """Render front-matter text safely inside a Markdown table cell."""

    flattened = " ".join(str(value).split())
    return html.escape(flattened, quote=False).replace("|", "\\|")


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
        {
            value
            for field in fields
            for value in _metadata_list(record, field)
        }
    )


def _rendered_path(record: HubRecord) -> str:
    directory = {
        "question": "questions",
        "study": "studies",
        "test": "ideas",
        "approach": "approaches",
        "dataset": "data",
        "background": "background",
    }[record.kind]
    return f"{directory}/{record.record_id}.html"


def render_catalog(build: HubBuild) -> str:
    """Render the stable, machine-readable search catalog."""

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
        study_ids = sorted(
            study.record_id
            for study in build.sources.studies
            if system.system_id in _metadata_list(study, "system_ids")
        )
        resources.append(
            {
                "id": system.system_id,
                "type": "run",
                "title": system.system_id,
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
                "rendered_path": f"generated/runs/{system.system_id}.html",
                "related_ids": study_ids,
            }
        )
    resources.sort(key=lambda row: (row["type"], row["id"]))
    return json.dumps(
        {"version": 1, "resources": resources},
        indent=2,
        sort_keys=True,
    ) + "\n"


def _summary_current_position(build: HubBuild) -> list[str]:
    lines = ["", "## Current position", ""]
    studies = sorted(
        (
            study
            for study in build.sources.studies
            if study.metadata.get("decision") != "pending"
        ),
        key=lambda row: row.record_id,
    )
    if not studies:
        return [*lines, "No decisions have been recorded yet."]
    for study in studies:
        decision = _markdown_text(study.metadata.get("decision", "unknown"))
        summary = _markdown_text(study.metadata.get("decision_summary", ""))
        lines.append(
            f"- **{_markdown_text(study.title)}** "
            f"(`{_markdown_text(study.record_id)}`; `{decision}`): {summary}"
        )
    return lines


def _ordered_questions(build: HubBuild) -> list[HubRecord]:
    return sorted(
        build.sources.questions,
        key=lambda row: (
            _QUESTION_STATUS_PRIORITY.get(
                str(row.metadata.get("status")),
                len(_QUESTION_STATUS_PRIORITY),
            ),
            row.record_id,
        ),
    )


def _summary_open_questions(build: HubBuild) -> list[str]:
    lines = ["", "## Open questions", ""]
    questions = _ordered_questions(build)
    if not questions:
        return [*lines, "No research questions have been recorded."]
    for question in questions:
        status = _markdown_text(question.metadata.get("status", "unknown"))
        answer = _markdown_text(question.metadata.get("current_answer", ""))
        lines.append(
            f"- **{_markdown_text(question.title)}** "
            f"(`{_markdown_text(question.record_id)}`; `{status}`): {answer}"
        )
    return lines


def _summary_next_tests(build: HubBuild) -> list[str]:
    lines = ["", "## Next tests", ""]
    tests = sorted(
        (
            test
            for test in build.sources.tests
            if test.metadata.get("stage") not in {"completed", "parked"}
        ),
        key=lambda row: (
            0 if row.metadata.get("stage") == "running" else
            1 if row.metadata.get("stage") == "ready" else
            2,
            row.record_id,
        ),
    )
    if not tests:
        return [*lines, "No active tests have been recorded."]
    for test in tests:
        stage = _markdown_text(test.metadata.get("stage", "unknown"))
        summary = _markdown_text(test.metadata.get("summary", ""))
        lines.append(
            f"- **{_markdown_text(test.title)}** "
            f"(`{_markdown_text(test.record_id)}`; `{stage}`): {summary}"
        )
    return lines


def _summary_navigation() -> list[str]:
    return [
        "",
        "## Navigation",
        "",
        "- [Research question coverage](research/hub/generated/coverage.qmd)",
        "- [Decision history](research/hub/generated/history.qmd)",
        "- [Experiment runs](research/hub/generated/runs.qmd)",
        "- [Machine-readable catalog](research/catalog.json)",
    ]


def render_summary_markdown(build: HubBuild) -> str:
    """Render the repository-facing executive summary."""

    lines = ["# TREC 2026 Research Hub", "", "Development evidence only."]
    lines.extend(_summary_current_position(build))
    lines.extend(_summary_open_questions(build))
    lines.extend(_summary_next_tests(build))
    lines.extend(_summary_navigation())
    return "\n".join(lines).rstrip() + "\n"


def _render_summary_cards(build: HubBuild) -> str:
    lines = [
        "---",
        'title: "Research Hub Summary"',
        "toc: true",
        "---",
        "",
        "# TREC 2026 Research Hub",
        "",
        "> Development evidence only. Promotion labels do not imply held-out results.",
    ]
    lines.extend(_summary_current_position(build))
    lines.extend(_summary_open_questions(build))
    lines.extend(_summary_next_tests(build))
    return "\n".join(lines)


def render_summary_qmd(build: HubBuild) -> str:
    return _render_summary_cards(build).rstrip() + "\n"


def _render_coverage_matrix(build: HubBuild) -> str:
    lines = [
        "---",
        'title: "Research Question Coverage"',
        "---",
        "",
        "# Research Question Coverage",
        "",
        "Development evidence only.",
        "",
        "| Question | Status | Confidence | Studies | Next tests |",
        "|---|---|---|---|---|",
    ]
    for question in _ordered_questions(build):
        studies = ", ".join(
            f"`{_markdown_text(value)}`"
            for value in sorted(_metadata_list(question, "study_ids"))
        ) or "—"
        next_tests = ", ".join(
            f"`{_markdown_text(value)}`"
            for value in sorted(_metadata_list(question, "next_test_ids"))
        ) or "—"
        lines.append(
            f"| **{_markdown_text(question.title)}** "
            f"(`{_markdown_text(question.record_id)}`) | "
            f"`{_markdown_text(question.metadata.get('status', 'unknown'))}` | "
            f"`{_markdown_text(question.metadata.get('confidence', 'unknown'))}` | "
            f"{studies} | {next_tests} |"
        )
    return "\n".join(lines)


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
        "",
        "| Date | Study | Outcome | Decision | Rationale provenance |",
        "|---|---|---|---|---|",
    ]
    studies = sorted(
        build.sources.studies,
        key=lambda row: (*_date_sort_value(row.metadata.get("concluded_on")), row.record_id),
    )
    for study in studies:
        concluded = study.metadata.get("concluded_on") or "—"
        if isinstance(concluded, date):
            concluded = concluded.isoformat()
        lines.append(
            f"| {_markdown_text(concluded)} | "
            f"**{_markdown_text(study.title)}** "
            f"(`{_markdown_text(study.record_id)}`) | "
            f"`{_markdown_text(study.metadata.get('outcome', 'unknown'))}` | "
            f"`{_markdown_text(study.metadata.get('decision', 'unknown'))}`: "
            f"{_markdown_text(study.metadata.get('decision_summary', ''))} | "
            f"`{_markdown_text(study.metadata.get('rationale_provenance', 'unknown'))}` |"
        )
    return "\n".join(lines)


def render_history_qmd(build: HubBuild) -> str:
    return _render_decision_history(build).rstrip() + "\n"


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _render_run_index(build: HubBuild) -> str:
    lines = [
        "---",
        'title: "Experiment Runs"',
        "---",
        "",
        "# Experiment Runs",
        "",
        "Development evidence only. Compare metrics only within the same evaluation scope.",
        "",
        "| Run | Family | Evaluation scope | Promotion | nDCG@10 | MRR |",
        "|---|---|---|---|---:|---:|",
    ]
    for system in sorted(build.ledger.systems, key=lambda row: row.system_id):
        run_path = f"runs/{system.system_id}.qmd"
        lines.append(
            f"| [`{_markdown_text(system.system_id)}`]({run_path}) | "
            f"`{_markdown_text(system.experiment_family)}` | "
            f"`{_markdown_text(system.evaluation_scope)}` | "
            f"`{_markdown_text(system.promotion_status)}` | "
            f"{_format_metric(system.ndcg_at_10)} | "
            f"{_format_metric(system.mrr)} |"
        )
    return "\n".join(lines)


def render_runs_qmd(build: HubBuild) -> str:
    return _render_run_index(build).rstrip() + "\n"


def _linked_study_ids(build: HubBuild, system_id: str) -> list[str]:
    return sorted(
        study.record_id
        for study in build.sources.studies
        if system_id in _metadata_list(study, "system_ids")
    )


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
    lines = [
        "---",
        f'title: "{_markdown_text(system.system_id)}"',
        "---",
        "",
        f"# `{_markdown_text(system.system_id)}`",
        "",
        _scope_guard(system),
        "",
        "## Evidence contract",
        "",
        f"- **Method:** {_markdown_text(system.method_summary)}",
        f"- **Experiment family:** `{_markdown_text(system.experiment_family)}`",
        f"- **Evaluation scope:** `{_markdown_text(system.evaluation_scope)}`",
        f"- **Evidence status:** `{_markdown_text(system.evidence_status)}`",
        f"- **Promotion status:** `{_markdown_text(system.promotion_status)}`",
        f"- **Artifact state:** `{_markdown_text(system.artifact_state)}`",
        f"- **Qrel usage:** `{_markdown_text(system.qrel_usage)}`",
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
    linked_studies = _linked_study_ids(build, system.system_id)
    if linked_studies:
        lines.extend(f"- `{_markdown_text(study_id)}`" for study_id in linked_studies)
    else:
        lines.append("No canonical study currently links this run.")
    lines.extend(
        [
            "",
            "## Evidence receipt",
            "",
            f"- **Source:** `{_markdown_text(system.source_path)}`",
            f"- **SHA-256:** `{_markdown_text(system.source_sha256)}`",
            f"- **Ranking coverage:** `{_markdown_text(system.ranking_coverage)}`",
        ]
    )
    if system.blocker:
        lines.append(f"- **Blocker:** {_markdown_text(system.blocker)}")
    lines.extend(["", "## Caveats", ""])
    if system.caveat_ids:
        lines.extend(
            f"- `{_markdown_text(caveat_id)}`"
            for caveat_id in sorted(system.caveat_ids)
        )
    else:
        lines.append("No ledger caveats are attached.")
    return "\n".join(lines)


def render_run_pages(build: HubBuild) -> dict[Path, str]:
    return {
        Path(f"research/hub/generated/runs/{system.system_id}.qmd"):
            _render_run_page(build, system).rstrip() + "\n"
        for system in sorted(build.ledger.systems, key=lambda row: row.system_id)
    }


def render_hub_outputs(build: HubBuild) -> Mapping[Path, str]:
    """Render every tracked generated output without touching the filesystem."""

    return {
        Path("research/catalog.json"): render_catalog(build),
        Path("reports/trec_2026_research_hub_summary.md"):
            render_summary_markdown(build),
        Path("research/hub/generated/summary.qmd"): render_summary_qmd(build),
        Path("research/hub/generated/coverage.qmd"): render_coverage_qmd(build),
        Path("research/hub/generated/history.qmd"): render_history_qmd(build),
        Path("research/hub/generated/runs.qmd"): render_runs_qmd(build),
        **render_run_pages(build),
    }


def _verified_staged_outputs(
    outputs: Mapping[Path, str],
    staging_dir: Path,
) -> dict[Path, Path]:
    staged_paths: dict[Path, Path] = {}
    for relative_path in sorted(outputs):
        content = outputs[relative_path]
        staged = staging_dir / relative_path
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(content, encoding="utf-8", newline="")
        if staged.read_bytes() != content.encode("utf-8"):
            raise OSError(f"staged output verification failed: {relative_path}")
        staged_paths[relative_path] = staged
    return staged_paths


def write_hub_outputs(build: HubBuild, root: Path) -> tuple[Path, ...]:
    """Atomically publish all hub outputs, rolling back a partial promotion."""

    outputs = render_hub_outputs(build)
    root = Path(root)
    relative_paths = tuple(sorted(outputs))
    destinations = tuple(root / path for path in relative_paths)
    unexpected_paths = tuple(
        sorted(_tracked_output_paths(root) - set(relative_paths))
    )

    with tempfile.TemporaryDirectory(prefix=".research-hub-", dir=root) as temporary:
        staging_dir = Path(temporary)
        staged_paths = _verified_staged_outputs(outputs, staging_dir)
        backup_dir = staging_dir / "previous"
        installed: list[Path] = []
        removed: list[Path] = []
        try:
            for relative_path in unexpected_paths:
                source = root / relative_path
                backup = backup_dir / relative_path
                backup.parent.mkdir(parents=True, exist_ok=True)
                source.replace(backup)
                removed.append(relative_path)
            for relative_path in relative_paths:
                destination = root / relative_path
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
                destination = root / relative_path
                destination.unlink(missing_ok=True)
                backup = backup_dir / relative_path
                if backup.exists():
                    backup.replace(destination)
            for relative_path in reversed(removed):
                backup = backup_dir / relative_path
                destination = root / relative_path
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
    return paths


def assert_hub_outputs_current(build: HubBuild, root: Path) -> None:
    """Report every missing, unexpected, or byte-different generated output."""

    outputs = render_hub_outputs(build)
    root = Path(root)
    expected_paths = set(outputs)
    tracked_paths = _tracked_output_paths(root)
    missing = sorted(
        path for path in expected_paths if not (root / path).is_file()
    )
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
        raise ValueError(
            "research hub outputs are not current; " + "; ".join(details)
        )
