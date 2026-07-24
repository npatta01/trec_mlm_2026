from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from trec_mlm.experiment_ledger import LedgerBuild
from trec_mlm.research_hub import render as module
from trec_mlm.research_hub.build import HubBuild
from trec_mlm.research_hub.model import HubSources
from trec_mlm.research_hub.render import (
    _inline_code,
    _markdown_table_text,
    _render_coverage_matrix,
    _render_decision_history,
    _validate_system_id,
    _yaml_scalar,
    render_catalog,
    render_run_pages,
    render_runs_qmd,
    render_summary_markdown,
    render_summary_qmd,
)
from trec_mlm.research_hub.validate import ValidationSummary


@pytest.fixture
def sample_build(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
) -> HubBuild:
    return HubBuild(
        sources=minimal_sources,
        ledger=ledger_build,
        validation=ValidationSummary(
            record_counts={
                "question": 1,
                "study": 1,
                "test": 1,
                "approach": 1,
                "dataset": 1,
                "background": 1,
            },
            linked_system_count=1,
        ),
    )


def test_render_catalog_is_deterministic(sample_build: HubBuild) -> None:
    first = render_catalog(sample_build)
    second = render_catalog(sample_build)

    assert first == second
    assert first.endswith("\n")
    payload = json.loads(first)
    assert payload["resources"][0]["canonical_path"].startswith("research/hub/")
    assert "generated_at" not in payload
    assert payload["resources"] == sorted(
        payload["resources"], key=lambda row: (row["type"], row["id"])
    )


def test_catalog_collects_sorted_related_ids_and_run_study_links(
    sample_build: HubBuild,
) -> None:
    payload = json.loads(render_catalog(sample_build))
    resources = {row["id"]: row for row in payload["resources"]}

    assert resources["dense-baseline"]["related_ids"] == [
        "candidate-evidence",
        "dense-retrieval",
        "development-set",
    ]
    assert resources["dense_nested"]["related_ids"] == ["dense-baseline"]
    assert resources["candidate-evidence"]["rendered_path"] == (
        "questions/candidate-evidence.html"
    )


def test_summary_leads_with_decision_not_system_count(
    sample_build: HubBuild,
) -> None:
    summary = render_summary_markdown(sample_build)

    assert summary.index("## Current position") < summary.index("## Open questions")
    assert "98 systems tested" not in summary
    assert "Keep the established baseline." in summary
    assert summary.endswith("\n")


def test_summary_is_a_bounded_current_decision_brief(
    sample_build: HubBuild,
) -> None:
    study = sample_build.sources.studies[0]
    studies = tuple(
        replace(
            study,
            record_id=f"decision-{index}",
            metadata={
                **study.metadata,
                "study_id": f"decision-{index}",
                "concluded_on": f"2026-07-0{index}",
                "superseded_by": ["decision-5"] if index == 1 else [],
            },
        )
        for index in range(1, 6)
    )
    question = sample_build.sources.questions[0]
    questions = tuple(
        replace(
            question,
            record_id=f"question-{index}",
            metadata={
                **question.metadata,
                "question_id": f"question-{index}",
                "updated_on": f"2026-07-0{index}",
                "confidence_boundary": f"Boundary {index}.",
            },
        )
        for index in range(1, 5)
    )
    idea = sample_build.sources.tests[0]
    tests = tuple(
        replace(
            idea,
            record_id=f"test-{index}",
            metadata={
                **idea.metadata,
                "test_id": f"test-{index}",
                "stage": "running" if index == 4 else "ready" if index == 3 else "idea",
                "priority": "high" if index in {2, 3} else "low",
            },
        )
        for index in range(1, 5)
    )
    build = replace(
        sample_build,
        sources=replace(
            sample_build.sources,
            studies=studies,
            questions=questions,
            tests=tests,
        ),
    )

    summary = render_summary_markdown(build)

    assert "`decision-5`" in summary
    assert "`decision-4`" in summary
    assert "`decision-3`" in summary
    assert "`decision-2`" not in summary
    assert "`decision-1`" not in summary
    assert "`question-4`" in summary
    assert "`question-3`" in summary
    assert "`question-2`" in summary
    assert "`question-1`" not in summary
    assert "Boundary 4." in summary
    assert "Boundary 3." in summary
    assert "Boundary 2." in summary
    assert summary.count("**Confidence boundary:**") == 3
    assert "`test-4`" in summary
    assert "`test-3`" in summary
    assert "`test-2`" in summary
    assert "`test-1`" not in summary


def test_summary_selects_next_tests_by_declared_priority_before_stage(
    sample_build: HubBuild,
) -> None:
    original = sample_build.sources.tests[0]
    specifications = (
        ("low-running", "running", "low"),
        ("high-idea", "idea", "high"),
        ("high-ready", "ready", "high"),
        ("medium-idea", "idea", "medium"),
    )
    tests = tuple(
        replace(
            original,
            record_id=record_id,
            metadata={
                **original.metadata,
                "test_id": record_id,
                "stage": stage,
                "priority": priority,
            },
        )
        for record_id, stage, priority in specifications
    )
    build = replace(
        sample_build,
        sources=replace(sample_build.sources, tests=tests),
    )

    summary = render_summary_markdown(build)

    assert "`high-idea`" in summary
    assert "`high-ready`" in summary
    assert "`medium-idea`" in summary
    assert "`low-running`" not in summary


def test_escaping_is_specific_to_yaml_tables_and_inline_code() -> None:
    unsafe = '"quoted" `tick` | <script>\nnext'

    assert yaml.safe_load(_yaml_scalar(unsafe)) == unsafe
    table_text = _markdown_table_text(unsafe)
    assert "&quot;quoted&quot;" in table_text
    assert "\\`tick\\`" in table_text
    assert "\\|" in table_text
    assert "&lt;script&gt;" in table_text
    assert "\n" not in table_text
    inline = _inline_code(unsafe)
    assert inline.startswith("``")
    assert inline.endswith("``")
    assert "\n" not in inline


@pytest.mark.parametrize(
    "system_id",
    [".", "..", "../outside", "nested/run", r"nested\run", "/absolute", "C:\\run"],
)
def test_run_page_rejects_unsafe_system_filename_component(system_id: str) -> None:
    with pytest.raises(ValueError, match="safe filename component"):
        _validate_system_id(system_id)


def test_decision_history_orders_known_dates_ascending_and_labels_unknown(
    sample_build: HubBuild,
) -> None:
    original = sample_build.sources.studies[0]
    older = replace(
        original,
        record_id="older",
        metadata={
            **original.metadata,
            "study_id": "older",
            "concluded_on": "2026-06-01",
        },
    )
    newer = replace(
        original,
        record_id="newer",
        metadata={
            **original.metadata,
            "study_id": "newer",
            "concluded_on": "2026-07-01",
        },
    )
    unknown = replace(
        original,
        record_id="unknown-date",
        metadata={
            **original.metadata,
            "study_id": "unknown-date",
            "concluded_on": "unknown",
        },
    )
    build = replace(
        sample_build,
        sources=replace(sample_build.sources, studies=(older, unknown, newer)),
    )

    history = _render_decision_history(build)

    assert "## Dated decisions" in history
    assert "## Date unknown" in history
    assert history.index("`older`") < history.index("`newer`")
    assert history.index("`newer`") < history.index("`unknown-date`")


def test_decision_history_shows_questions_change_and_supported_causal_next_study(
    sample_build: HubBuild,
) -> None:
    original = sample_build.sources.studies[0]
    older = replace(
        original,
        record_id="older",
        metadata={
            **original.metadata,
            "study_id": "older",
            "conclusion": "The original assumption did not hold.",
            "superseded_by": ["newer"],
        },
    )
    newer = replace(
        original,
        record_id="newer",
        metadata={
            **original.metadata,
            "study_id": "newer",
            "rationale": "Resolve the failure found by the older study.",
            "supersedes": ["older"],
        },
    )
    unrelated = replace(
        original,
        record_id="unrelated",
        metadata={
            **original.metadata,
            "study_id": "unrelated",
            "rationale": "This must not be presented as caused by older.",
        },
    )
    build = replace(
        sample_build,
        sources=replace(
            sample_build.sources,
            studies=(older, newer, unrelated),
        ),
    )

    history = _render_decision_history(build)

    assert "Active questions" in history
    assert "`candidate-evidence`" in history
    assert "The original assumption did not hold." in history
    assert "Next study" in history
    assert "`newer`" in history
    assert "Resolve the failure found by the older study." in history
    older_row = next(line for line in history.splitlines() if "`older`" in line)
    assert "unrelated" not in older_row


def test_coverage_is_question_by_approach_with_counted_evidence_states(
    sample_build: HubBuild,
) -> None:
    question = sample_build.sources.questions[0]
    study = sample_build.sources.studies[0]
    approach = sample_build.sources.approaches[0]
    uncovered = replace(
        approach,
        record_id="uncovered-approach",
        metadata={
            **approach.metadata,
            "approach_id": "uncovered-approach",
            "study_ids": [],
        },
    )
    mixed = replace(
        approach,
        record_id="mixed-approach",
        metadata={
            **approach.metadata,
            "approach_id": "mixed-approach",
            "study_ids": ["positive-study", "invalid-study"],
        },
    )
    positive = replace(
        study,
        record_id="positive-study",
        metadata={
            **study.metadata,
            "study_id": "positive-study",
            "approach_ids": ["mixed-approach"],
        },
    )
    invalid = replace(
        study,
        record_id="invalid-study",
        metadata={
            **study.metadata,
            "study_id": "invalid-study",
            "approach_ids": ["mixed-approach"],
            "outcome": "invalid",
            "system_ids": ["qwen8b_initial_terminal_token_invalid"],
            "representative_system_ids": ["qwen8b_initial_terminal_token_invalid"],
            "comparison_system_ids": ["qwen8b_initial_terminal_token_invalid"],
        },
    )
    build = replace(
        sample_build,
        sources=replace(
            sample_build.sources,
            questions=(question,),
            approaches=(approach, mixed, uncovered),
            studies=(study, positive, invalid),
        ),
    )

    coverage = _render_coverage_matrix(build)

    assert (
        "| Question | `dense-retrieval` | `mixed-approach` | `uncovered-approach` |"
        in coverage
    )
    row = next(line for line in coverage.splitlines() if "`candidate-evidence`" in line)
    assert "positive (1 study)" in row
    assert "mixed (2 studies): invalid, positive" in row
    assert "uncovered" in row


def test_coverage_preserves_restricted_and_unscored_evidence_states(
    sample_build: HubBuild,
) -> None:
    study = sample_build.sources.studies[0]
    studies = (
        study,
        replace(
            study,
            record_id="restricted-study",
            metadata={
                **study.metadata,
                "study_id": "restricted-study",
                "system_ids": ["pointwise_qwen_feasibility_6"],
                "representative_system_ids": ["pointwise_qwen_feasibility_6"],
                "comparison_system_ids": ["pointwise_qwen_feasibility_6"],
            },
        ),
        replace(
            study,
            record_id="unscored-study",
            metadata={
                **study.metadata,
                "study_id": "unscored-study",
                "system_ids": ["source_alignment__k10"],
                "representative_system_ids": ["source_alignment__k10"],
                "comparison_system_ids": ["source_alignment__k10"],
            },
        ),
    )
    build = replace(
        sample_build,
        sources=replace(sample_build.sources, studies=studies),
    )

    coverage = _render_coverage_matrix(build)
    row = next(line for line in coverage.splitlines() if "`candidate-evidence`" in line)

    assert "mixed (3 studies): positive, restricted positive, unscored" in row


def test_run_index_partitions_scopes_and_evidence_classes(
    sample_build: HubBuild,
) -> None:
    index = render_runs_qmd(sample_build)

    assert "## Evaluation scope: `dev_322x150`" in index
    assert "## Evaluation scope: `restricted_6`" in index
    assert "### Evidence: `stored_per_query`" in index
    assert "### Evidence: `aggregate_only`" in index
    assert "data-hub-filter" in index
    assert "data-hub-filter-row" in index
    assert "dense_profile_equal.html" in index


def test_run_index_suppresses_effectiveness_values_for_invalid_and_unscored(
    sample_build: HubBuild,
) -> None:
    index = render_runs_qmd(sample_build)
    invalid = index.split("## Evaluation scope: `invalid_unscored`", 1)[1].split(
        "## Evaluation scope:", 1
    )[0]
    unscored = index.split("## Evaluation scope: `constructed_not_evaluated`", 1)[
        1
    ].split("## Evaluation scope:", 1)[0]

    assert "Effectiveness suppressed" in invalid
    assert "Effectiveness suppressed" in unscored
    assert "qwen8b_initial_terminal_token_invalid" in invalid
    assert "source_alignment__k10" in unscored


def test_run_pages_preserve_evaluation_scope(sample_build: HubBuild) -> None:
    pages = render_run_pages(sample_build)
    page = pages[Path("research/hub/generated/runs/dense_profile_equal.qmd")]

    assert "dev_322x150" in page
    assert "promoted_development" in page
    assert "Development evidence only." in page
    assert "source_system_id" in page
    assert "per_query_available" in page
    assert "trec_2026_experiment_per_query.tsv" in page
    assert "trec_2026_experiment_topic_slices.tsv" in page
    assert "trec_2026_experiment_density_slices.tsv" in page


def test_restricted_run_page_warns_against_cross_scope_comparison(
    sample_build: HubBuild,
) -> None:
    page = render_run_pages(sample_build)[
        Path("research/hub/generated/runs/pointwise_qwen_feasibility_6.qmd")
    ]

    assert "restricted_6" in page
    assert "restricted_pilot" in page
    assert "not directly comparable" in page


def test_write_is_atomic_on_renderer_failure(
    sample_build: HubBuild,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "research/catalog.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("preserve me\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "render_hub_outputs",
        lambda build: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        module.write_hub_outputs(sample_build, tmp_path)

    assert existing.read_text(encoding="utf-8") == "preserve me\n"


def test_summary_navigation_resolves_from_each_output_location(
    sample_build: HubBuild,
) -> None:
    report = render_summary_markdown(sample_build)
    generated = render_summary_qmd(sample_build)

    assert "(../research/hub/generated/coverage.qmd)" in report
    assert "(../research/hub/generated/summary.qmd)" in report
    assert "(evidence/coverage.qmd)" in generated
    assert "(../catalog.json)" in generated
