from pathlib import Path

from trec_mlm.experiment_ledger import assert_outputs_current, build_ledger


def test_checked_in_experiment_ledger_is_current() -> None:
    root = Path(".")
    build = build_ledger(root)
    assert build.validation["system_count"] == 98
    assert len(build.per_query) == 3542
    assert len(build.topic_slices) == 1650
    assert len(build.density_slices) == 33
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
