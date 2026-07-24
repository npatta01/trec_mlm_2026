from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from trec_mlm.experiment_ledger import LedgerBuild
from trec_mlm.research_hub import render as module
from trec_mlm.research_hub.build import HubBuild
from trec_mlm.research_hub.model import HubSources
from trec_mlm.research_hub.render import (
    _render_decision_history,
    _markdown_text,
    render_catalog,
    render_run_pages,
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


def test_summary_questions_sort_by_status_priority_then_id(
    sample_build: HubBuild,
) -> None:
    original = sample_build.sources.questions[0]
    answered = replace(
        original,
        record_id="alpha-answered",
        title="Answered question",
        metadata={
            **original.metadata,
            "question_id": "alpha-answered",
            "status": "answered",
        },
    )
    open_question = replace(
        original,
        record_id="zeta-open",
        title="Open question",
        metadata={
            **original.metadata,
            "question_id": "zeta-open",
            "status": "open",
        },
    )
    build = replace(
        sample_build,
        sources=replace(
            sample_build.sources,
            questions=(answered, open_question),
        ),
    )

    summary = render_summary_markdown(build)

    assert summary.index("`zeta-open`") < summary.index("`alpha-answered`")


def test_summary_qmd_escapes_untrusted_markdown_table_text(
    sample_build: HubBuild,
) -> None:
    study = sample_build.sources.studies[0]
    unsafe = replace(
        study,
        metadata={
            **study.metadata,
            "decision_summary": "Keep | verify\nagain.",
        },
    )
    build = replace(
        sample_build,
        sources=replace(sample_build.sources, studies=(unsafe,)),
    )

    rendered = render_summary_qmd(build)

    assert "Keep \\| verify again." in rendered
    assert _markdown_text("<script>|x\nnext") == (
        "&lt;script&gt;\\|x next"
    )


def test_decision_history_orders_newest_known_date_first_and_unknown_last(
    sample_build: HubBuild,
) -> None:
    original = sample_build.sources.studies[0]
    older = replace(
        original,
        record_id="older",
        metadata={**original.metadata, "study_id": "older", "concluded_on": "2026-06-01"},
    )
    newer = replace(
        original,
        record_id="newer",
        metadata={**original.metadata, "study_id": "newer", "concluded_on": "2026-07-01"},
    )
    unknown = replace(
        original,
        record_id="unknown-date",
        metadata={**original.metadata, "study_id": "unknown-date", "concluded_on": "unknown"},
    )
    build = replace(
        sample_build,
        sources=replace(sample_build.sources, studies=(older, unknown, newer)),
    )

    history = _render_decision_history(build)

    assert history.index("`newer`") < history.index("`older`")
    assert history.index("`older`") < history.index("`unknown-date`")


def test_run_pages_preserve_evaluation_scope(sample_build: HubBuild) -> None:
    pages = render_run_pages(sample_build)
    page = pages[
        Path("research/hub/generated/runs/dense_profile_equal.qmd")
    ]

    assert "dev_322x150" in page
    assert "promoted_development" in page
    assert "Development evidence only." in page


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
