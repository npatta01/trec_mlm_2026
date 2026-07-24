from __future__ import annotations

from pathlib import Path

from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.render import render_hub_outputs


ROOT = Path(__file__).resolve().parents[2]


def test_background_records_use_audited_sources_and_named_evidence() -> None:
    build = build_research_hub(ROOT)
    background = {
        record.record_id: record for record in build.sources.background
    }

    assert (
        background["llm-routing-literature"].metadata["canonical_source_path"]
        == "reports/trec_2026_approaches_guide.artifact.json"
    )
    assert (
        background["methods-landscape"].metadata["canonical_source_path"]
        == "reports/trec_2026_approaches_guide.artifact.json"
    )
    assert (
        background["reference-model-planning"].metadata[
            "canonical_source_path"
        ]
        == "reports/trec_2026_reference_model_landscape_2026-07-21.md"
    )
    assert "Gemini 3.6 Flash" not in background["methods-landscape"].body
    reference_body = " ".join(
        background["reference-model-planning"].body.split()
    )
    assert "Gemini 3.6 Flash" in reference_body
    assert (
        "reference-model-landscape.qmd"
        not in background["methods-landscape"].body
    )
    assert (
        "../library/reference-model-landscape.qmd"
        in background["reference-model-planning"].body
    )
    assert "competition" not in background["related-competitions"].title.lower()
    assert "RouterBench" in background["llm-routing-literature"].body
    assert "SRCB" in background["trec-2025-literature"].body
    assert all(
        "placeholder" not in record.body.lower()
        for record in background.values()
    )
    assert all(
        "deep dive" in record.body.lower() for record in background.values()
    )
    assert all(
        "staged" not in record.body.lower() for record in background.values()
    )


def test_background_index_is_generated_and_grouped() -> None:
    build = build_research_hub(ROOT)
    output = render_hub_outputs(build)[
        Path("research/hub/generated/background-index.qmd")
    ]

    assert "## TREC 2025" in output
    assert "## Wider methods and evaluations" in output
    assert "## Current model and reference planning" in output
    current_planning = output.split(
        "## Current model and reference planning", maxsplit=1
    )[1].split("## ", maxsplit=1)[0]
    assert 'reference-model-planning.qmd"' in current_planning
    assert 'methods-landscape.qmd"' not in current_planning
    wider_methods = output.split(
        "## Wider methods and evaluations", maxsplit=1
    )[1].split("## ", maxsplit=1)[0]
    assert 'methods-landscape.qmd"' in wider_methods
    assert "hub-background-row" in output
    assert "listing:" not in output
    for record in build.sources.background:
        assert f'{record.record_id}.qmd"' in output
