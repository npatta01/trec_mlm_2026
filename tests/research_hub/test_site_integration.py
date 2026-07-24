from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from scripts.normalize_research_hub_links import normalize_links
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
    assert "library/*.qmd" in config["project"]["render"]
    assert config["project"]["pre-render"] == [
        "uv run --project ../.. python ../../scripts/build_research_hub.py "
        "--root ../.."
    ]
    assert config["project"]["post-render"] == [
        "uv run python ../../scripts/normalize_research_hub_links.py "
        "--root ../.. --output-dir _site",
    ]
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
        "background": 6,
    }
    assert build.validation.linked_system_count == len(build.ledger.systems) == 98

    problem = (ROOT / "research/hub/start-here/problem.qmd").read_text()
    assert "## What are we building?" in problem
    assert "150 candidate LLMs" in problem
    assert "hidden test" in problem.lower()
    assert "Who became the second Karmapa Lama?" in problem
    for candidate in (
        "Cultures",
        "People_by_time",
        "Scholars_and_academics_by_subject",
    ):
        assert candidate in problem
    assert "Candidate 37" not in problem
    assert "does not claim" in problem

    protocols = (ROOT / "research/hub/protocols/index.qmd").read_text()
    assert "../../../protocols/" not in protocols
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
    background = (ROOT / "research/hub/background/index.qmd").read_text()

    assert "hub-task-flow" in problem
    assert "hub-example" in problem
    studies = (ROOT / "research/hub/studies/index.qmd").read_text()
    themes = (ROOT / "research/hub/themes/index.qmd").read_text()
    summary = (ROOT / "research/hub/index.qmd").read_text()
    ledger_script = (ROOT / "research/hub/assets/ledger.js").read_text()

    assert "{{< include ../generated/questions-index.qmd >}}" in questions
    assert "{{< include ../generated/background-index.qmd >}}" in background
    assert data.strip().endswith(
        "{{< include ../generated/data-index.qmd >}}"
    )
    assert "Candidate inventory" not in data
    assert "{{< include ../generated/studies-index.qmd >}}" in studies
    assert "{{< include ../generated/themes-index.qmd >}}" in themes
    assert "research summary" not in themes.lower()
    assert "include-after-body: ../assets/ledger-script.html" in studies
    assert "{{< include generated/summary.qmd >}}" in summary
    assert 'searchParams.get("theme")' in ledger_script
    assert "[data-themes]" in ledger_script
    assert "listing:" not in questions
    assert "listing:" not in studies
    assert "listing:" not in data
    assert "listing:" not in background


def test_generated_pages_and_theme_use_the_open_visual_system() -> None:
    build = build_research_hub(ROOT)
    outputs = render_hub_outputs(build)
    generated = "\n".join(outputs.values())
    theme = (ROOT / "research/hub/assets/theme.scss").read_text()

    for class_name in (
        "hub-theme-row",
        "hub-conclusion",
        "hub-question-row",
        "hub-evidence-line",
        "hub-study-row",
        "hub-background-row",
    ):
        assert class_name in generated
        assert f".{class_name}" in theme

    for index_path in (
        "research/hub/generated/summary.qmd",
        "research/hub/generated/themes-index.qmd",
        "research/hub/generated/questions-index.qmd",
        "research/hub/generated/studies-index.qmd",
        "research/hub/generated/background-index.qmd",
        "research/hub/generated/data-index.qmd",
    ):
        source = outputs[Path(index_path)]
        assert "listing:" not in source
        assert "line count" not in source.lower()
        assert "line-count" not in source.lower()
        assert "study_type" not in source

    assert ".hub-theme-row:hover" in theme
    assert ".hub-theme-row:focus-visible" in theme
    assert ".hub-decision-strip" in theme
    assert "@media (forced-colors: active)" in theme
    assert "@media (max-width: 768px)" in theme
    assert ".content table" in theme
    assert "overflow-x: auto" in theme
    assert "$web-font-path: false" in theme


def test_post_render_normalizes_only_local_index_links(tmp_path: Path) -> None:
    output = tmp_path / "research/hub/_site"
    output.mkdir(parents=True)
    page = output / "page.html"
    page.write_text(
        '<a href="../background/index.html">Background</a>'
        '<a href="index.html?view=all">Home</a>'
        '<a href="https://example.test/index.html">External</a>'
        '<a href="&sol;&sol;evil.example/index.html">Encoded external</a>'
        '<a href="https&colon;//evil.example/index.html">Encoded scheme</a>'
        r'<a href="\\evil.example\foo/index.html">Backslash external</a>'
    )

    changed = normalize_links(tmp_path, output)

    assert changed == (page,)
    rendered = page.read_text()
    assert 'href="../background/"' in rendered
    assert 'href="./?view=all"' in rendered
    assert 'href="https://example.test/index.html"' in rendered
    assert 'href="&sol;&sol;evil.example/index.html"' in rendered
    assert 'href="https&colon;//evil.example/index.html"' in rendered
    assert r'href="\\evil.example\foo/index.html"' in rendered


def test_post_render_rejects_symlinked_output(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    expected = tmp_path / "research/hub"
    expected.mkdir(parents=True)
    (expected / "_site").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        normalize_links(tmp_path, expected / "_site")


def test_generated_navigation_contains_every_canonical_record() -> None:
    build = build_research_hub(ROOT)
    outputs = render_hub_outputs(build)

    summary = outputs[Path("research/hub/generated/summary.qmd")]
    themes = outputs[Path("research/hub/generated/themes-index.qmd")]
    questions = outputs[Path("research/hub/generated/questions-index.qmd")]
    studies = outputs[Path("research/hub/generated/studies-index.qmd")]
    background = outputs[Path("research/hub/generated/background-index.qmd")]
    data = outputs[Path("research/hub/generated/data-index.qmd")]

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
        detail_path = Path(
            f"research/hub/generated/studies/{study.record_id}.md"
        )
        assert detail_path in outputs
        canonical = (ROOT / study.source_path).read_text()
        assert (
            "{{< include ../generated/studies/"
            f"{study.record_id}.md >}}"
        ) in canonical
        assert "%40" not in outputs[detail_path]
    for resource in build.sources.background:
        assert resource.record_id in background
    for dataset in build.sources.datasets:
        assert dataset.record_id in data


def test_contributor_guide_and_complete_record_templates() -> None:
    guide = (ROOT / "research/hub/CONTRIBUTING.md").read_text()
    rendered_guide = (ROOT / "research/hub/contributing.qmd").read_text()
    config = yaml.safe_load((ROOT / "research/hub/_quarto.yml").read_text())

    for phrase in (
        "Add a question",
        "Add a study",
        "Add a dataset",
        "Link a run",
        "Build generated outputs",
        "Check generated outputs",
    ):
        assert phrase in guide
    for command in (
        "uv run python scripts/build_research_hub.py",
        "quarto render research/hub",
        "uv run python scripts/build_research_hub.py --check",
    ):
        assert command in guide
    assert "construction" in guide.lower()
    assert "evaluation scope" in guide.lower()
    assert "confidence boundary" in guide.lower()
    assert "canonical" in guide.lower()
    assert "data index" in guide.lower()
    assert "machine-readable catalog" in guide.lower()
    assert "CONTRIBUTING.md" in rendered_guide
    assert any(
        row.get("text") == "Contribute"
        and row.get("href") == "contributing.qmd"
        for row in config["website"]["navbar"]["left"]
    )

    required_fields = {
        "theme.qmd.example": {
            "theme_id",
            "title",
            "plain_english",
            "current_conclusion",
            "decision",
            "confidence",
            "confidence_boundary",
            "next_test_ids",
            "updated_on",
            "evidence_refs",
        },
        "question.qmd.example": {
            "question_id",
            "title",
            "primary_theme_id",
            "theme_ids",
            "plain_english",
            "status",
            "current_answer",
            "confidence",
            "confidence_boundary",
            "study_ids",
            "open_subquestions",
            "next_test_ids",
            "updated_on",
            "evidence_refs",
        },
        "study.qmd.example": {
            "study_id",
            "title",
            "primary_theme_id",
            "theme_ids",
            "study_type",
            "question_ids",
            "approach_ids",
            "status",
            "outcome",
            "objective",
            "hypothesis",
            "rationale",
            "rationale_provenance",
            "protocol_summary",
            "conclusion",
            "decision",
            "decision_summary",
            "system_ids",
            "representative_system_ids",
            "comparison_system_ids",
            "evidence_refs",
            "dataset_ids",
            "started_on",
            "concluded_on",
            "supersedes",
            "superseded_by",
            "limitations",
        },
        "test.qmd.example": {
            "test_id",
            "title",
            "primary_theme_id",
            "theme_ids",
            "stage",
            "summary",
            "linked_question_ids",
            "missing_readiness_fields",
            "hypothesis",
            "rationale",
            "expected_signal",
            "decision_rule",
            "dependencies",
            "priority",
            "evaluation_boundary",
            "artifact_plan",
            "next_action",
        },
        "dataset.qmd.example": {
            "dataset_id",
            "title",
            "plain_english",
            "category",
            "provenance",
            "method",
            "version",
            "repository_path",
            "sensitivity",
            "sharing",
            "leakage_constraints",
            "permitted_use",
            "study_ids",
            "availability",
            "reproducibility",
        },
        "background.qmd.example": {
            "resource_id",
            "title",
            "resource_type",
            "summary",
            "canonical_source_path",
            "citations",
            "tags",
            "question_ids",
            "reviewed_on",
        },
    }
    for filename, required in required_fields.items():
        path = ROOT / "research/hub/templates" / filename
        assert path.is_file()
        lines = path.read_text().splitlines()
        closing = lines.index("---", 1)
        metadata = yaml.safe_load("\n".join(lines[1:closing]))
        assert required <= set(metadata), filename


def test_quarto_pre_render_reflects_an_added_question_from_another_cwd(
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "repository-copy"
    copied_root.mkdir()
    for name in (
        "configs",
        "data",
        "docs",
        "experiments",
        "protocols",
        "reports",
        "research",
        "scripts",
        "trec_mlm",
    ):
        shutil.copytree(ROOT / name, copied_root / name)
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, copied_root / name)

    implementation_sentinel = "COPIED_REPOSITORY_RENDERER_EXECUTED"
    copied_renderer = copied_root / "trec_mlm/research_hub/render.py"
    copied_renderer.write_text(
        copied_renderer.read_text(encoding="utf-8")
        + f"""

_copied_render_questions_index_qmd = render_questions_index_qmd


def render_questions_index_qmd(build: HubBuild) -> str:
    return (
        _copied_render_questions_index_qmd(build)
        + "\\n{implementation_sentinel}\\n"
    )
""",
        encoding="utf-8",
    )

    title = "Can automatic contribution propagation be trusted?"
    question_id = "automatic-propagation-proof"
    (copied_root / f"research/hub/questions/{question_id}.qmd").write_text(
        f"""---
question_id: {question_id}
title: "{title}"
primary_theme_id: query-only-routing
theme_ids: [query-only-routing]
plain_english: "Does one valid question update every derived reader surface?"
status: open
current_answer: "This temporary record exists only to test automatic propagation."
confidence: low
confidence_boundary: "Build integration only; this is not research evidence."
study_ids: []
open_subquestions: ["Will all generated and rendered surfaces agree?"]
next_test_ids: []
evidence_refs: [reports/trec_2026_experiment_ledger.md]
updated_on: 2026-07-24
---

Temporary integration-proof record.
""",
        encoding="utf-8",
    )

    copied_pythonpath = os.pathsep.join(
        filter(None, (str(copied_root), os.environ.get("PYTHONPATH")))
    )
    env = {
        **os.environ,
        "PYTHONPATH": copied_pythonpath,
        "UV_NO_SYNC": "1",
        "UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv"),
    }
    completed = subprocess.run(
        [
            "quarto",
            "render",
            str(copied_root / "research/hub/questions/index.qmd"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    for path in (
        "research/hub/generated/questions-index.qmd",
        "research/hub/generated/themes/query-only-routing.qmd",
        "research/hub/generated/history.qmd",
        "research/catalog.json",
        "research/hub/_site/questions/index.html",
    ):
        assert title in (copied_root / path).read_text(), path
    assert implementation_sentinel in (
        copied_root / "research/hub/generated/questions-index.qmd"
    ).read_text()
    assert implementation_sentinel in (
        copied_root / "research/hub/_site/questions/index.html"
    ).read_text()
