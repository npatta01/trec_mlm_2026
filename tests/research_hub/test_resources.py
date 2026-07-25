from __future__ import annotations

from pathlib import Path
import re

import yaml

from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.render import render_hub_outputs


ROOT = Path(__file__).resolve().parents[2]


def test_hub_theme_uses_local_inter_and_grayscale_safe_idea_visuals() -> None:
    theme = (ROOT / "research/hub/assets/theme.scss").read_text()

    assert "@font-face" in theme
    assert "Inter-Regular.otf" in theme
    assert "Inter-SemiBold.otf" in theme
    for token in (
        "--hub-paper: #f2efe7;",
        "--hub-ink: #171713;",
        "--hub-risk: #a64b21;",
        "--hub-proposal: #007b72;",
    ):
        assert token in theme
    for selector in (
        ".hub-record-hero",
        ".hub-takeaway",
        ".hub-next-action",
        ".hub-details",
        ".hub-causal-comparison",
        ".hub-causal-current",
        ".hub-causal-proposed",
    ):
        assert selector in theme
    assert ".hub-record-path" not in theme
    assert ".hub-path" in theme
    mobile = theme.split("@media (max-width: 768px)", maxsplit=1)[1]
    assert ".hub-causal-comparison" in mobile
    assert "grid-template-columns: 1fr;" in mobile
    assert "details[open]" in theme
    assert "summary:focus-visible" in theme
    assert "@media (forced-colors: active)" in theme
    assert "@media (prefers-reduced-motion: reduce)" in theme
    dark = theme.split("body.quarto-dark", maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "--hub-paper: #1d1d1b;" in dark


def test_background_records_use_audited_sources_and_named_evidence() -> None:
    build = build_research_hub(ROOT)
    background = {record.record_id: record for record in build.sources.background}

    assert (
        background["llm-routing-literature"].metadata["canonical_source_path"]
        == "reports/trec_2026_approaches_guide.artifact.json"
    )
    assert (
        background["methods-landscape"].metadata["canonical_source_path"]
        == "reports/trec_2026_approaches_guide.artifact.json"
    )
    assert (
        background["reference-model-planning"].metadata["canonical_source_path"]
        == "reports/trec_2026_reference_model_landscape_2026-07-21.md"
    )
    assert "Gemini 3.6 Flash" not in background["methods-landscape"].body
    reference_body = " ".join(background["reference-model-planning"].body.split())
    assert "Gemini 3.6 Flash" in reference_body
    assert "reference-model-landscape.qmd" not in background["methods-landscape"].body
    assert (
        "../library/reference-model-landscape.qmd"
        in background["reference-model-planning"].body
    )
    semantic_frames = background["semantic-demand-frames"]
    assert semantic_frames.metadata["canonical_source_path"] == (
        "reports/trec_2025_team_method_granularity_review_2026-07-24.md"
    )
    normalized = " ".join(semantic_frames.body.split()).lower()
    assert "no reviewed trec 2025 team used this frame" in normalized
    assert "not a validated router" in normalized
    assert "response behavior" in normalized
    assert "never hard-filter" in normalized
    yaml_blocks = re.findall(
        r"```yaml\n(.*?)\n```", semantic_frames.body, flags=re.DOTALL
    )
    assert len(yaml_blocks) == 1
    example = yaml.safe_load(yaml_blocks[0])
    assert "provenance" not in example
    extracted_fields = [
        example["semantic_graph"],
        example["semantic_graph"]["predicate"],
        example["semantic_graph"]["arguments"],
        *example["semantic_graph"]["arguments"]["values"].values(),
        example["requested_slot"],
        example["computation_steps"],
        example["answer_contract"],
        *example["answer_contract"]["facets"].values(),
        example["context_facets"],
        *example["context_facets"]["facets"].values(),
    ]
    for field in extracted_fields:
        metadata = field["field_metadata"]
        assert {
            "confidence",
            "uncertainty",
            "alternatives",
            "extraction",
            "evidence",
        } <= metadata.keys()
        assert 0 <= metadata["confidence"] <= 1
        assert metadata["uncertainty"]
        assert isinstance(metadata["alternatives"], list)
        assert {"mode", "version"} <= metadata["extraction"].keys()
        assert {"source", "span"} <= metadata["evidence"].keys()
        assert all(metadata["extraction"].values())
        assert all(metadata["evidence"].values())
    assert "competition" not in background["related-competitions"].title.lower()
    assert "RouterBench" in background["llm-routing-literature"].body
    assert "SRCB" in background["trec-2025-literature"].body
    assert all(
        "placeholder" not in record.body.lower() for record in background.values()
    )
    assert all("deep dive" in record.body.lower() for record in background.values())
    assert all("staged" not in record.body.lower() for record in background.values())


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
    wider_methods = output.split("## Wider methods and evaluations", maxsplit=1)[
        1
    ].split("## ", maxsplit=1)[0]
    assert 'methods-landscape.qmd"' in wider_methods
    assert "hub-background-row" in output
    assert "listing:" not in output
    for record in build.sources.background:
        assert f'{record.record_id}.qmd"' in output
