from __future__ import annotations

from pathlib import Path

import yaml

from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.render import render_hub_outputs


ROOT = Path(__file__).resolve().parents[2]


def test_complete_research_hub_content_and_static_shell() -> None:
    build = build_research_hub(ROOT)
    config = yaml.safe_load((ROOT / "research/hub/_quarto.yml").read_text())

    assert config["quarto-required"] == "1.9.38"
    assert config["project"]["type"] == "website"
    assert config["project"]["output-dir"] == "_site"
    assert config["project"]["resources"] == ["assets/ledger.js", "catalog.json"]
    assert "generated/runs/*.qmd" in config["project"]["render"]
    assert "generated/themes/*.qmd" in config["project"]["render"]
    assert {
        row["text"] for row in config["website"]["navbar"]["left"]
    } >= {"Themes"}
    themes_nav = next(
        row
        for row in config["website"]["navbar"]["left"]
        if row["text"] == "Themes"
    )
    assert themes_nav["href"] == "themes/index.qmd"
    assert config["execute"]["enabled"] is False
    assert build.validation.record_counts == {
        "theme": 4,
        "question": 5,
        "study": 15,
        "test": 3,
        "approach": 5,
        "dataset": 6,
        "background": 5,
    }
    assert build.validation.linked_system_count == len(build.ledger.systems) == 98

    problem = (ROOT / "research/hub/start-here/problem.qmd").read_text()
    assert "## What are we building?" in problem
    assert "150 candidate LLMs" in problem
    assert "hidden test" in problem.lower()

    protocols = (ROOT / "research/hub/protocols/index.qmd").read_text()
    for path in sorted((ROOT / "protocols").glob("*.md")):
        assert path.relative_to(ROOT).as_posix() in protocols

    assert {
        "provided",
        "collected",
        "generated",
    } <= {row.metadata["category"] for row in build.sources.datasets}
    qrels = next(
        row for row in build.sources.datasets if row.record_id == "dev-qrels"
    )
    assert qrels.metadata["permitted_use"] == "development_evaluation"
    assert all(
        len(row.metadata["representative_system_ids"]) <= 8
        for row in build.sources.studies
    )
    by_study = {row.record_id: row for row in build.sources.studies}
    assert by_study["four-model-reference-review"].metadata["system_ids"] == []
    assert (
        by_study["grounded-data-quality-audit"].metadata["study_type"]
        == "data_quality"
    )


def test_explainer_and_index_pages_use_readable_custom_layouts() -> None:
    problem = (ROOT / "research/hub/start-here/problem.qmd").read_text()
    questions = (ROOT / "research/hub/questions/index.qmd").read_text()
    data = (ROOT / "research/hub/data/index.qmd").read_text()

    assert "hub-task-flow" in problem
    assert "hub-example" in problem
    studies = (ROOT / "research/hub/studies/index.qmd").read_text()
    themes = (ROOT / "research/hub/themes/index.qmd").read_text()
    summary = (ROOT / "research/hub/index.qmd").read_text()
    ledger_script = (ROOT / "research/hub/assets/ledger.js").read_text()

    assert "{{< include ../generated/questions-index.qmd >}}" in questions
    assert "{{< include ../generated/studies-index.qmd >}}" in studies
    assert "{{< include ../generated/themes-index.qmd >}}" in themes
    assert "research summary" not in themes.lower()
    assert "include-after-body: ../assets/ledger-script.html" in studies
    assert "{{< include generated/summary.qmd >}}" in summary
    assert 'searchParams.get("theme")' in ledger_script
    assert "[data-themes]" in ledger_script
    assert "hub-data-groups" in data
    assert "Organizer-provided" in data
    assert "listing:" not in questions
    assert "listing:" not in studies
    assert "listing:" not in data


def test_generated_navigation_contains_every_canonical_record() -> None:
    build = build_research_hub(ROOT)
    outputs = render_hub_outputs(build)

    summary = outputs[Path("research/hub/generated/summary.qmd")]
    themes = outputs[Path("research/hub/generated/themes-index.qmd")]
    questions = outputs[Path("research/hub/generated/questions-index.qmd")]
    studies = outputs[Path("research/hub/generated/studies-index.qmd")]

    for theme in build.sources.themes:
        assert theme.record_id in summary
        assert theme.record_id in themes
        assert (
            Path(f"research/hub/generated/themes/{theme.record_id}.qmd") in outputs
        )
    for question in build.sources.questions:
        assert question.record_id in questions
    for study in build.sources.studies:
        assert study.record_id in studies
