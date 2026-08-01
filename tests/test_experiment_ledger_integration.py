from pathlib import Path
from collections import Counter

from trec_mlm.experiment_ledger import assert_outputs_current, build_ledger


def test_checked_in_experiment_ledger_is_current() -> None:
    root = Path(".")
    build = build_ledger(root)
    assert build.validation["system_count"] == 100
    assert len(build.per_query) == 4186
    assert len(build.topic_slices) == 1950
    assert len(build.density_slices) == 39
    assert (
        sum(
            system.artifact_state
            in {"local_artifact_missing", "construction_artifact_missing"}
            for system in build.systems
        )
        == 43
    )
    assert_outputs_current(build, root / "reports")


def test_real_ledger_has_required_program_caveats() -> None:
    build = build_ledger(Path("."))
    caveat_ids = {row["caveat_id"] for row in build.caveats}
    assert {
        "development_only",
        "adaptive_program_overfit",
        "restricted_not_comparable",
        "topic_slices_overlap",
        "missing_raw_artifacts",
        "issue18_not_restorable",
        "confidence_not_correctness",
        "bootstrap_protocols_differ",
    } <= caveat_ids


def test_pointwise_phase0_registers_exactly_two_complete_valid_negatives() -> None:
    build = build_ledger(Path("."))
    family = [
        system
        for system in build.systems
        if system.experiment_family == "pointwise_ranking_phase0"
    ]

    assert {system.system_id for system in family} == {
        "pointwise_phase0_pointwise_only",
        "pointwise_phase0_combined",
    }
    assert all(system.evidence_status == "stored_per_query" for system in family)
    assert all(system.promotion_status == "valid_negative" for system in family)
    assert all(system.evaluation_scope == "dev_322x150" for system in family)
    assert all(system.query_count == 322 for system in family)
    assert all(system.candidate_count == 150 for system in family)
    assert all(system.per_query_available for system in family)
    assert [
        system.system_id
        for system in build.systems
        if system.promotion_status == "promoted_development"
    ] == ["dense_profile_equal"]

    counts = Counter(row.system_id for row in build.per_query)
    assert counts["pointwise_phase0_pointwise_only"] == 322
    assert counts["pointwise_phase0_combined"] == 322
    assert next(
        row for row in build.systems if row.system_id == "dense_nested"
    ).promotion_status == "stable_control"

    by_id = {system.system_id: system for system in family}
    assert by_id["pointwise_phase0_pointwise_only"].ndcg_at_10 == (
        0.11882482433832028
    )
    assert by_id["pointwise_phase0_combined"].ndcg_at_10 == 0.2113240933733129
    for system in family:
        assert {
            "development_only",
            "pointwise_phase0_failed_gate",
            "pointwise_support_not_gold",
        } <= set(system.caveat_ids)
