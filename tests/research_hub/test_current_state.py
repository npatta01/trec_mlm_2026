from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest
import yaml

from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.render import render_current_state_qmd


ROOT = Path(__file__).resolve().parents[2]

VISUAL_IDS = {
    "research-theme-map",
    "ranking-task-flow",
    "canonical-source-hierarchy",
    "experiment-family-decisions",
    "representative-development-results",
    "artifact-custody",
    "remaining-failures",
    "pretest-dependencies",
}

EXPECTED_READING_THEMES = (
    (
        "query-baselines-and-neighbors",
        (
            "query-controls",
            "dense-query-knn",
            "query-cluster-aggregation",
        ),
    ),
    (
        "candidate-labels-and-learned-scorers",
        ("candidate-label-matching", "label-ranker-pilots"),
    ),
    (
        "candidate-profiles-and-fusion",
        ("behavior-profile", "multiresolution-profiles", "semantic-profiles"),
    ),
    (
        "direct-response-evidence",
        (
            "response-retrieval",
            "response-volume-normalization",
            "source-query-first",
        ),
    ),
    (
        "references-and-bounded-refinement",
        ("four-model-reference-review", "e5-raw-response-refinement"),
    ),
    (
        "reliability-failures-and-readiness",
        (
            "gemini-grounding-failure-review",
            "grounded-data-quality-audit",
            "sparse-failure-analogue-diagnosis",
        ),
    ),
)

EXPECTED_METHOD_FAMILY_ANSWERS = (
    (
        "query-baselines-and-neighbors",
        "dense-query-knn",
        "semantic-failure-clusters",
    ),
    (
        "candidate-labels-and-learned-scorers",
        "candidate-label-matching",
        "semantic-failure-clusters",
    ),
    (
        "candidate-profiles-and-fusion",
        "semantic-profiles",
        "semantic-failure-clusters",
    ),
    (
        "direct-response-evidence",
        "response-retrieval",
        "failure-aware-fusion",
    ),
    (
        "references-and-bounded-refinement",
        "four-model-reference-review",
        "reference-review-trigger",
    ),
    (
        "reliability-failures-and-readiness",
        "sparse-failure-analogue-diagnosis",
        "semantic-failure-clusters",
    ),
)

EXPECTED_LADDER = (
    ("dense_nested", "dev_322x150", "stable_control", "stored_per_query"),
    ("contextual_e5", "dev_322x150", "stable_control", "stored_per_query"),
    ("local_domain_prior", "dev_322x150", "valid_negative", "aggregate_only"),
    (
        "response_sumscore_nested",
        "dev_322x150",
        "valid_negative",
        "aggregate_only",
    ),
    ("profile_only", "dev_322x150", "stable_control", "stored_per_query"),
    (
        "dense_profile_equal",
        "dev_322x150",
        "promoted_development",
        "stored_per_query",
    ),
    (
        "dense_profile_profile_heavy",
        "dev_322x150",
        "diagnostic_only",
        "stored_per_query",
    ),
)


def _visual_block(source: str, visual_id: str) -> str:
    match = re.search(
        rf'<figure\b[^>]*data-visual="{re.escape(visual_id)}"[^>]*>'
        rf"(?P<body>.*?)</figure>",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, visual_id
    return match.group(0)


def test_current_state_route_leads_with_boundary_and_pretest_actions() -> None:
    page = (ROOT / "research/hub/start-here/current-state.qmd").read_text()
    page_metadata = yaml.safe_load(page.split("---", maxsplit=2)[1])
    config = yaml.safe_load((ROOT / "research/hub/_quarto.yml").read_text())
    start_here = (ROOT / "research/hub/start-here/index.qmd").read_text()

    assert "hub-current-state-hero" in page
    assert "Current conclusion" in page
    assert "Development evidence only" in page
    assert "No hidden-test evidence" in page
    assert page_metadata["toc"] is False
    assert "{{< include ../generated/current-state.qmd >}}" in page
    assert (
        ":::: {.hub-current-state-page}\n"
        "{{< include ../generated/current-state.qmd >}}\n"
        "::::"
    ) in page
    assert page.index("hub-current-state-hero") < page.index(
        "{{< include ../generated/current-state.qmd >}}"
    )
    for issue_number in (39, 40, 41):
        assert (
            f"https://github.com/npatta01/trec_mlm_2026/issues/{issue_number}"
            in page
        )
    assert "#42" in page
    assert (
        "https://github.com/npatta01/trec_mlm_2026/issues/42" in page
    )
    assert "current-state.qmd" in start_here

    start_nav = next(
        row
        for row in config["website"]["navbar"]["left"]
        if row["text"] == "Start Here"
    )
    assert start_nav["href"] == "start-here/current-state.qmd"


def test_generated_current_state_has_eight_accessible_evidence_visuals() -> None:
    rendered = render_current_state_qmd(build_research_hub(ROOT))
    found_ids = set(re.findall(r'data-visual="([a-z-]+)"', rendered))

    assert found_ids == VISUAL_IDS
    assert "<img" not in rendered
    assert "<canvas" not in rendered
    assert "<script" not in rendered
    for visual_id in VISUAL_IDS:
        visual = _visual_block(rendered, visual_id)
        title_id = f"{visual_id}-title"
        assert f'aria-labelledby="{title_id}"' in visual
        assert f'id="{title_id}"' in visual
        assert 'class="hub-visual-explanation"' in visual
        assert (
            f'<details class="hub-visual-equivalent" '
            f'data-equivalent-for="{visual_id}">' in visual
        )
        assert (
            f'<section class="hub-visual-print-equivalent" '
            f'data-print-equivalent-for="{visual_id}"' in visual
        )
        equivalent = visual.split(
            f'data-equivalent-for="{visual_id}">', maxsplit=1
        )[1]
        assert "Source:" in equivalent
        assert "Status:" in equivalent
        assert "<table" in equivalent or "<ol" in equivalent or "<ul" in equivalent


def test_reading_map_partitions_every_authored_study_in_fixed_presentation_order() -> None:
    build = build_research_hub(ROOT)
    rendered = render_current_state_qmd(build)
    visual = _visual_block(rendered, "research-theme-map")
    primary = visual.split('<details class="hub-visual-equivalent"', maxsplit=1)[0]

    theme_ids = re.findall(r'data-reading-theme="([a-z-]+)"', primary)
    assert theme_ids == [theme_id for theme_id, _study_ids in EXPECTED_READING_THEMES]
    assert len(theme_ids) == len(set(theme_ids)) == 6

    expected_study_ids = [
        study_id
        for _theme_id, study_ids in EXPECTED_READING_THEMES
        for study_id in study_ids
    ]
    linked_study_ids = re.findall(
        r'<li data-study-status="[a-z]+"><a href="\.\./studies/([a-z0-9-]+)\.qmd"',
        primary,
    )
    assert linked_study_ids == expected_study_ids
    assert set(expected_study_ids) == {
        study.record_id for study in build.sources.studies
    }
    assert len(expected_study_ids) == len(set(expected_study_ids))
    assert "method-family map, not a canonical taxonomy" in visual
    assert "Study status vocabulary: Planned, Running, and Concluded." in visual
    statuses = re.findall(r'data-study-status="([a-z]+)"', primary)
    assert len(statuses) == len(expected_study_ids)
    assert set(statuses) <= {"planned", "running", "concluded"}


def test_method_family_cards_answer_reader_questions_with_member_evidence() -> None:
    rendered = render_current_state_qmd(build_research_hub(ROOT))
    visual = _visual_block(rendered, "research-theme-map")

    assert "Studies by method family" in visual
    assert "presentation-only method-family map, not a canonical taxonomy" in visual
    cards = re.findall(
        r'<section class="hub-reading-theme" data-method-family="([a-z-]+)" '
        r'data-reading-theme="[a-z-]+">'
        r"(?P<body>.*?)</section>",
        visual,
        flags=re.DOTALL,
    )
    assert [family_id for family_id, _body in cards] == [
        family_id for family_id, _strongest, _next_test in EXPECTED_METHOD_FAMILY_ANSWERS
    ]
    for (family_id, strongest_study, next_test), (_rendered_id, card) in zip(
        EXPECTED_METHOD_FAMILY_ANSWERS, cards, strict=True
    ):
        assert 'class="hub-method-purpose"' in card
        assert 'class="hub-method-answer"' in card
        assert 'class="hub-method-open-question"' in card
        assert f'href="../studies/{strongest_study}.qmd"' in card
        assert f'href="../generated/ideas/{next_test}.qmd"' in card
        member_ids = re.findall(r'href="\.\./studies/([a-z0-9-]+)\.qmd"', card)
        assert strongest_study in member_ids, family_id

    equivalent = visual.split('<details class="hub-visual-equivalent"', maxsplit=1)[1]
    for family_id, strongest_study, _next_test in EXPECTED_METHOD_FAMILY_ANSWERS:
        family_rows = re.findall(
            rf'<tr data-reading-theme="{family_id}".*?</tr>',
            equivalent,
            flags=re.DOTALL,
        )
        assert family_rows
        assert all(
            f'href="../studies/{strongest_study}.qmd"' in row for row in family_rows
        )


def test_reader_action_strip_keeps_each_label_bound_to_canonical_status() -> None:
    rendered = render_current_state_qmd(build_research_hub(ROOT))
    cards = dict(
        re.findall(
            r'<section class="hub-reader-action" data-reader-action="([a-z-]+)">'
            r"(?P<body>.*?)</section>",
            rendered,
            flags=re.DOTALL,
        )
    )

    assert set(cards) == {
        "use-now",
        "retain-as-controls",
        "promising-follow-up",
        "still-required-before-test",
        "do-not-reuse-tested-form",
        "unknown",
    }
    assert "dense_profile_equal" in cards["use-now"]
    assert "promoted_development" in cards["use-now"]
    for system_id in ("dense_nested", "contextual_e5", "profile_only"):
        assert system_id in cards["retain-as-controls"]
    assert "stable_control" in cards["retain-as-controls"]
    assert "advance" in cards["promising-follow-up"]
    assert "informational" in cards["promising-follow-up"]
    assert "outside the ledger" in cards["promising-follow-up"]
    for issue_number in (39, 40, 41, 42):
        assert f"#{issue_number}" in cards["still-required-before-test"]
    assert "valid_negative" in cards["do-not-reuse-tested-form"]
    assert "condition-bound" in cards["do-not-reuse-tested-form"]
    assert "hidden-test" in cards["unknown"]

    system_ids = {
        action: set(re.findall(r"<code>([a-z0-9_]+)</code>", body))
        for action, body in cards.items()
    }
    assert system_ids == {
        "use-now": {"dense_profile_equal", "promoted_development"},
        "retain-as-controls": {
            "dense_nested",
            "contextual_e5",
            "profile_only",
            "stable_control",
        },
        "promising-follow-up": {"advance", "informational"},
        "still-required-before-test": set(),
        "do-not-reuse-tested-form": {
            "local_domain_prior",
            "response_sumscore_nested",
            "valid_negative",
        },
        "unknown": set(),
    }
    issue_numbers = {
        action: set(re.findall(r">#(\d+)</a>", body))
        for action, body in cards.items()
    }
    assert issue_numbers["still-required-before-test"] == {"39", "40", "41", "42"}
    assert not any(
        numbers
        for action, numbers in issue_numbers.items()
        if action != "still-required-before-test"
    )


def test_canonical_theme_cards_render_authored_evidence_and_open_questions() -> None:
    build = build_research_hub(ROOT)
    rendered = render_current_state_qmd(build)
    cards = dict(
        re.findall(
            r'<section class="hub-canonical-theme" data-canonical-theme="([a-z-]+)">'
            r"(?P<body>.*?)</section>",
            rendered,
            flags=re.DOTALL,
        )
    )

    assert set(cards) == {theme.record_id for theme in build.sources.themes}
    for theme in build.sources.themes:
        card = cards[theme.record_id]
        assert str(theme.metadata["current_conclusion"]) in card
        assert str(theme.metadata["strongest_evidence_summary"]) in card
        assert str(theme.metadata["open_question"]) in card
        assert str(theme.metadata["confidence_boundary"]) in card
        assert "First linked next test" in card


def test_reading_map_order_does_not_depend_on_source_loading_order() -> None:
    build = build_research_hub(ROOT)
    reordered = replace(
        build,
        sources=replace(build.sources, studies=tuple(reversed(build.sources.studies))),
    )

    original = _visual_block(render_current_state_qmd(build), "research-theme-map")
    changed = _visual_block(render_current_state_qmd(reordered), "research-theme-map")

    assert changed == original


def test_reading_map_rejects_a_wholly_unrelated_current_state_study_set() -> None:
    build = build_research_hub(ROOT)
    studies = tuple(
        replace(
            study,
            record_id=f"unrelated-study-{index}",
            source_path=f"research/hub/studies/unrelated-study-{index}.qmd",
        )
        for index, study in enumerate(build.sources.studies)
    )

    with pytest.raises(
        ValueError,
        match="must partition every authored study exactly once",
    ):
        render_current_state_qmd(
            replace(build, sources=replace(build.sources, studies=studies))
        )


def test_reading_map_rejects_unrelated_studies_without_follow_up_evidence() -> None:
    build = build_research_hub(ROOT)
    studies = tuple(
        replace(
            study,
            record_id=f"unrelated-study-{index}",
            source_path=f"research/hub/studies/unrelated-study-{index}.qmd",
        )
        for index, study in enumerate(build.sources.studies)
    )

    with pytest.raises(
        ValueError,
        match="must partition every authored study exactly once",
    ):
        render_current_state_qmd(
            replace(
                build,
                sources=replace(build.sources, studies=studies),
                completed_follow_up=None,
            )
        )


def test_registered_result_ladder_has_fixed_identity_scope_status_and_order() -> None:
    rendered = render_current_state_qmd(build_research_hub(ROOT))
    visual = _visual_block(rendered, "representative-development-results")
    rows = re.findall(
        r'<tr data-result-system="([a-z0-9_]+)"[^>]*>'
        r"<td>.*?</td><td><code>.*?</code></td>"
        r"<td><code>(.*?)</code></td><td><code>(.*?)</code></td>"
        r"<td><code>(.*?)</code></td>",
        visual,
        flags=re.DOTALL,
    )

    assert rows == list(EXPECTED_LADDER) * 2
    assert "Seven fixed registered rows" in visual
    assert "Aggregate-only rows do not claim per-query evidence" in visual
    assert visual.count('data-evidence-status="aggregate_only"') == 6
    for status in {row[2] for row in EXPECTED_LADDER}:
        assert f">{status}<" in visual


@pytest.mark.parametrize("diagnostic_score", [0.0, 0.999999])
def test_ladder_position_and_status_do_not_change_with_diagnostic_score(
    diagnostic_score: float,
) -> None:
    build = build_research_hub(ROOT)
    systems = tuple(
        replace(system, ndcg_at_10=diagnostic_score)
        if system.system_id == "dense_profile_profile_heavy"
        else system
        for system in build.ledger.systems
    )

    visual = _visual_block(
        render_current_state_qmd(
            replace(build, ledger=replace(build.ledger, systems=systems))
        ),
        "representative-development-results",
    )

    assert re.findall(r'data-result-system="([a-z0-9_]+)"', visual) == [
        row[0] for row in EXPECTED_LADDER
    ] * 3
    assert visual.count(
        'data-result-system="dense_profile_profile_heavy" '
        'data-result-status="diagnostic_only"'
    ) == 3


def test_completed_follow_up_is_a_separate_normalized_off_ledger_block() -> None:
    build = build_research_hub(ROOT)
    assert build.completed_follow_up is not None

    rendered = render_current_state_qmd(build)
    ladder = _visual_block(rendered, "representative-development-results")
    off_ledger = re.search(
        r'<section class="hub-off-ledger-follow-up".*?</section>',
        rendered,
        flags=re.DOTALL,
    )

    assert off_ledger is not None
    card = off_ledger.group(0)
    assert "raw_hierarchical_dense_equal" in card
    assert "outside the ledger" in card
    assert "not registered" in card
    assert "not promoted" in card
    assert "Receipt gate decision" in card
    assert "Decision layers" in card
    assert "advance gate" in card
    assert "Hub decision remains informational" in card
    assert "0.435625" in card
    assert "0.465458" in card
    assert "raw_hierarchical_dense_equal" not in ladder
    assert rendered.index(ladder) < rendered.index(card)
    assert rendered.index(card) < rendered.index('data-visual="remaining-failures"')


def test_completed_follow_up_rejects_candidate_registered_in_the_ledger() -> None:
    build = build_research_hub(ROOT)
    registered_candidate = replace(
        build.ledger.systems[0],
        system_id="raw_hierarchical_dense_equal",
    )

    with pytest.raises(
        ValueError,
        match="raw_hierarchical_dense_equal must remain absent from the registered ledger",
    ):
        render_current_state_qmd(
            replace(
                build,
                ledger=replace(
                    build.ledger,
                    systems=(*build.ledger.systems, registered_candidate),
                ),
            )
        )


def test_current_state_quantities_come_from_canonical_build_inputs() -> None:
    build = build_research_hub(ROOT)
    rendered = render_current_state_qmd(build)
    systems = build.ledger.systems

    family_count = len({system.experiment_family for system in systems})
    promoted_count = sum(
        system.promotion_status == "promoted_development" for system in systems
    )
    states = {
        name: sum(system.artifact_state == name for system in systems)
        for name in (
            "tracked_complete",
            "tracked_receipt_only",
            "tracked_report_snapshot",
        )
    }
    blockers = sum(
        system.artifact_state
        in {"local_artifact_missing", "construction_artifact_missing"}
        for system in systems
    )

    assert f"{family_count} experiment families" in rendered
    assert f"{len(systems)} registered systems" in rendered
    assert f"{len(build.sources.studies)} authored studies" in rendered
    assert f"{promoted_count} promoted development system" in rendered
    assert ">{}<".format(states["tracked_complete"]) in rendered
    assert ">{}<".format(states["tracked_receipt_only"]) in rendered
    assert ">{}<".format(states["tracked_report_snapshot"]) in rendered
    assert f"{states['tracked_report_snapshot']} report snapshots" in rendered
    assert f">{blockers}<" in rendered
    assert ">21<" in rendered
    assert ">54<" in rendered
    fusion = build.failure_diagnostic["fusion_21"]["count"]
    no_signal = build.failure_diagnostic["no_signal_54"]["count"]
    assert f"{fusion + no_signal} sole-positive top-ten failures" in rendered


def test_result_status_is_not_inferred_from_score() -> None:
    build = build_research_hub(ROOT)
    rendered = render_current_state_qmd(build)
    systems = {system.system_id: system for system in build.ledger.systems}

    promoted = systems["dense_profile_equal"]
    diagnostic = systems["dense_profile_profile_heavy"]
    rejected = systems["local_domain_prior"]
    assert diagnostic.ndcg_at_10 is not None
    assert rejected.ndcg_at_10 is not None
    assert promoted.ndcg_at_10 is not None
    assert diagnostic.ndcg_at_10 > promoted.ndcg_at_10

    diagnostic_row = re.search(
        r'<tr[^>]*data-result-system="dense_profile_profile_heavy".*?</tr>',
        rendered,
        flags=re.DOTALL,
    )
    rejected_row = re.search(
        r'<tr[^>]*data-result-system="local_domain_prior".*?</tr>',
        rendered,
        flags=re.DOTALL,
    )
    assert diagnostic_row is not None
    assert rejected_row is not None
    assert "Promoted" not in diagnostic_row.group(0)
    assert "Promoted" not in rejected_row.group(0)
    assert 'data-result-status="promoted_development"' in rendered


def test_current_state_rejects_registered_ladder_promotion_drift() -> None:
    build = build_research_hub(ROOT)
    changed = tuple(
        replace(system, promotion_status="diagnostic_only")
        if system.system_id == "profile_only"
        else system
        for system in build.ledger.systems
    )

    with pytest.raises(
        ValueError,
        match="profile_only must remain stable_control",
    ):
        render_current_state_qmd(
            replace(build, ledger=replace(build.ledger, systems=changed))
        )


def test_failure_total_is_derived_from_both_diagnostic_cohorts() -> None:
    build = replace(
        build_research_hub(ROOT),
        failure_diagnostic={
            "fusion_21": {
                "count": 4,
                "label": "Fusion dilution",
                "definition": "One promoted component has top-ten signal.",
            },
            "no_signal_54": {
                "count": 7,
                "label": "No signal",
                "definition": "Neither promoted component has top-ten signal.",
            },
        },
    )

    rendered = render_current_state_qmd(build)

    assert "11 sole-positive top-ten failures" in rendered
    assert "75 sole-positive top-ten failures" not in rendered
    assert "11-case failure split" in rendered
    assert "75-case" not in rendered


def test_current_state_scoped_style_selectors_exist() -> None:
    theme = (ROOT / "research/hub/assets/theme.scss").read_text()

    for selector in (
        ".hub-current-state-page",
        ".hub-evidence-visual",
        ".hub-result-bar",
        ".hub-result-label > code",
        ".hub-custody-segment",
        ".hub-dependency-map",
        ".hub-visual-equivalent",
        ".hub-visual-print-equivalent",
        ".hub-current-state-tutorial",
        ".hub-example-thread",
        ".hub-term-decoder",
        ".hub-theme-map",
        ".hub-theme-node",
        ".hub-reading-theme > ol > li",
        ".hub-study-status",
        ".hub-result-followup",
        ".hub-audit-reference",
        '.hub-result-row[data-result-status="promoted_development"]',
        '.hub-result-row[data-result-status="stable_control"]',
        '.hub-result-row[data-result-status="diagnostic_only"]',
        '.hub-result-row[data-result-status="valid_negative"]',
    ):
        assert selector in theme


def test_authored_tutorial_keeps_the_synthetic_and_evidence_boundaries() -> None:
    page = (ROOT / "research/hub/start-here/current-state.qmd").read_text()
    lower_page = " ".join(page.casefold().split())

    assert page.count("{{< include ../generated/current-state.qmd >}}") == 1
    assert "Explain why my balcony basil leaves are turning yellow" in page
    for candidate in ("GardenGuide", "QuickFacts", "CodeTutor"):
        assert candidate in page
    assert "fabricated" in lower_page
    assert "no private response text" in lower_page
    assert (
        "combine what worked for similar past requests with a summary of what "
        "each candidate's past answers suggest it is good at"
    ) in lower_page
    assert "query-cluster aggregation" in lower_page
    assert "rejected" in lower_page
    assert "development evidence only" in lower_page
    assert "no hidden-test evidence" in lower_page
    assert "not registered" in lower_page
    assert "not promoted" in lower_page
    for issue_number in (39, 40, 41, 42):
        assert f"issues/{issue_number}" in page


def test_authored_tutorial_preserves_method_cardinality_and_e5_mechanism() -> None:
    page = (ROOT / "research/hub/start-here/current-state.qmd").read_text()
    lower_page = " ".join(page.casefold().split())

    assert "find similar past requests" in lower_page
    assert "find one similar past request" not in lower_page
    assert "neighborhood of released development requests" in lower_page
    assert "candidate-relevance histories" in lower_page

    assert "10, 20, or 40 related historical requests" in lower_page
    assert "all 150 candidate responses" in lower_page
    assert "weights each answer-alignment score" in lower_page

    assert "10 closest candidate-local historical records" in lower_page
    assert "80% broad profile" in lower_page
    assert "20% raw-response refinement" in lower_page
    assert "equal reciprocal-rank fusion with the dense ranking" in lower_page
    assert "../studies/e5-raw-response-refinement.qmd" in page

    glossary = " ".join(
        (ROOT / "research/hub/start-here/glossary.qmd")
        .read_text()
        .casefold()
        .split()
    )
    assert "10, 20, or 40 historical requests" in glossary
    assert "selected request is query-side evidence" in glossary
    assert "candidate responses are candidate-side evidence" in glossary


def test_authored_page_does_not_duplicate_generated_evidence_counts() -> None:
    page = (ROOT / "research/hub/start-here/current-state.qmd").read_text()

    for generated_fact in (
        "98 registered systems",
        "11 experiment families",
        "43 blockers",
        "21 fusion",
        "54 no signal",
        "0.420214",
    ):
        assert generated_fact not in page
    assert "raw artifact" not in page.lower()
    assert "98-row" not in page
