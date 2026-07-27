from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest

from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.render import render_failure_library_qmd


ROOT = Path(__file__).resolve().parents[2]


def _record(rendered: str, record_id: str) -> str:
    match = re.search(
        rf'<section class="hub-failure-record" data-failure-mechanism="{record_id}">'
        r"(?P<body>.*?)</section>",
        rendered,
        flags=re.DOTALL,
    )
    assert match is not None, record_id
    return match.group(0)


def test_failure_library_derives_diagnostic_counts_and_keeps_generated_evidence_unquantified() -> None:
    """A changed diagnostic receipt must change its two displayed cohorts, not invent a third rate."""

    build = build_research_hub(ROOT)
    rendered = render_failure_library_qmd(build)

    fusion = _record(rendered, "fusion-dilution")
    no_signal = _record(rendered, "no-component-signal")
    unreliable = _record(rendered, "unreliable-generated-evidence")

    assert "21 development failures" in fusion
    assert "54 development failures" in no_signal
    assert "No prevalence count is available" in unreliable
    assert not re.search(r"\b(?:\d+|one|two|three) (?:cases|failures)\b", unreliable)

    for record in (fusion, no_signal, unreliable):
        for field in (
            "What failed",
            "Evidence available",
            "Response or change",
            "Observed outcome",
            "Possible failure location",
            "Competing explanation",
            "Confirmed defect, audit gap, or uncertainty",
            "Evidence limit",
            "Next falsifying test",
            "Source study and evidence status",
        ):
            assert f"<dt>{field}</dt>" in record

    changed = replace(
        build,
        failure_diagnostic={
            "fusion_21": {
                **build.failure_diagnostic["fusion_21"],
                "count": 4,
            },
            "no_signal_54": {
                **build.failure_diagnostic["no_signal_54"],
                "count": 7,
            },
        },
    )
    changed_rendered = render_failure_library_qmd(changed)
    assert "4 development failures" in changed_rendered
    assert "7 development failures" in changed_rendered
    assert "21 development failures" not in changed_rendered
    assert "54 development failures" not in changed_rendered


def test_failure_library_rejects_an_incomplete_canonical_failure_split() -> None:
    """The page must not present one diagnostic cohort without its complement."""

    build = build_research_hub(ROOT)
    with pytest.raises(ValueError, match="both documented cohorts"):
        render_failure_library_qmd(
            replace(build, failure_diagnostic={"fusion_21": build.failure_diagnostic["fusion_21"]})
        )


def test_failure_library_page_includes_the_generated_evidence_and_safe_companions() -> None:
    """Readers reach the generated library through a private-safe authored page."""

    page = (ROOT / "research/hub/start-here/failure-library.qmd").read_text(
        encoding="utf-8"
    )

    assert "{{< include ../generated/failure-library.qmd >}}" in page
    assert "cases explain mechanisms, not benchmark prevalence" in page
    assert "worked-example.qmd" in page
    assert "../background/pretest-readiness-plan.qmd" in page
