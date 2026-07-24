from __future__ import annotations

from pathlib import Path

import yaml

from trec_mlm.research_hub.build import build_research_hub


ROOT = Path(__file__).resolve().parents[2]


def test_complete_research_hub_content_and_static_shell() -> None:
    build = build_research_hub(ROOT)
    config = yaml.safe_load((ROOT / "research/hub/_quarto.yml").read_text())

    assert config["quarto-required"] == "1.9.38"
    assert config["project"]["type"] == "website"
    assert config["project"]["output-dir"] == "_site"
    assert config["project"]["resources"] == ["assets/ledger.js", "catalog.json"]
    assert "generated/runs/*.qmd" in config["project"]["render"]
    assert config["execute"]["enabled"] is False
    assert build.validation.record_counts == {
        "question": 5,
        "study": 15,
        "test": 3,
        "approach": 5,
        "dataset": 6,
        "background": 5,
    }
    assert build.validation.linked_system_count == len(build.ledger.systems) == 98

    problem = (ROOT / "research/hub/start-here/problem.qmd").read_text()
    assert "## The problem in one sentence" in problem
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
