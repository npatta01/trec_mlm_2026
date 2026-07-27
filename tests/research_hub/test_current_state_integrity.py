from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from trec_mlm.research_hub import build as build_module
from trec_mlm.experiment_ledger import LedgerBuild
from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.model import HubRecord, HubSources
from trec_mlm.research_hub.render import render_current_state_qmd, render_catalog


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = "reports/trec_2026_sparse_failure_cause_diagnostic.json"
FOLLOW_UP_REPORT_PATH = "reports/trec_2026_e5_raw_response_refinement_results.json"


def _completed_follow_up_sources(**overrides: object) -> HubSources:
    metadata = {
        "status": "concluded",
        "current_state_role": "strongest_completed_follow_up",
        "system_ids": [],
        "representative_system_ids": [],
        "evidence_refs": [FOLLOW_UP_REPORT_PATH],
        **overrides,
    }
    return HubSources(
        studies=(
            HubRecord(
                kind="study",
                record_id="e5-raw-response-refinement",
                title="E5 raw-response refinement",
                source_path="research/hub/studies/e5-raw-response-refinement.qmd",
                metadata=metadata,
                body="",
            ),
        )
    )


def _completed_follow_up_receipt() -> dict[str, object]:
    return {
        "artifact_type": "raw_response_refinement_results",
        "evaluation": {
            "comparisons": {
                "primary": {
                    "baseline": "dense_profile_equal",
                    "candidate": "raw_hierarchical_dense_equal",
                    "ndcg_at_10_delta": {
                        "lower": 0.006964609514709062,
                        "mean_difference": 0.015411280824014868,
                        "upper": 0.024355856293844506,
                    },
                    "mrr_delta": {
                        "lower": 0.0014759731217427877,
                        "mean_difference": 0.012440358541930006,
                        "upper": 0.023749239191402996,
                    },
                },
            },
            "metrics": {
                "dense_profile_equal": {
                    "ndcg_at_10": 0.4202139921941393,
                    "mrr": 0.45301739378333744,
                    "zero_ndcg_queries": 99,
                },
                "raw_hierarchical_dense_equal": {
                    "ndcg_at_10": 0.4356252730181542,
                    "mrr": 0.46545775232526737,
                    "zero_ndcg_queries": 92,
                },
            },
            "matched_parent_metrics": {
                "dense_profile_equal": {
                    "ndcg_at_10": 0.4202139921941393,
                    "mrr": 0.45301739378333744,
                    "zero_ndcg_queries": 99,
                },
            },
            "decision": {
                "decision": "advance",
                "gates": {
                    "integrity_passes": True,
                    "mrr_not_materially_lower": True,
                    "ndcg_interval_lower_above_zero": True,
                    "zero_count_reduced": True,
                },
                "integrity_checks": {
                    "artifact_validation": True,
                    "construction_hash_binding": True,
                    "dependency_bindings": True,
                    "embedding_contract": True,
                    "inventory": True,
                    "parent_metrics": True,
                    "qrel_independence": True,
                    "repeatability": True,
                    "semantic_reconstruction": True,
                    "trec_completeness": True,
                },
            },
        },
    }


def _write_completed_follow_up_receipt(root: Path, payload: dict[str, object]) -> None:
    path = root / FOLLOW_UP_REPORT_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ledger_with_parent_metric_drift(ledger: LedgerBuild) -> LedgerBuild:
    systems = tuple(
        replace(system, ndcg_at_10=0.0)
        if system.system_id == "dense_profile_equal"
        else system
        for system in ledger.systems
    )
    return replace(ledger, systems=systems)


def test_completed_follow_up_loader_normalizes_the_prespecified_receipt(
    tmp_path: Path,
    ledger_build: LedgerBuild,
) -> None:
    _write_completed_follow_up_receipt(tmp_path, _completed_follow_up_receipt())

    evidence = build_module._load_completed_follow_up(
        tmp_path,
        _completed_follow_up_sources(),
        ledger_build,
    )

    assert evidence is not None
    assert evidence.study_id == "e5-raw-response-refinement"
    assert evidence.source_path == FOLLOW_UP_REPORT_PATH
    assert evidence.baseline_id == "dense_profile_equal"
    assert evidence.candidate_id == "raw_hierarchical_dense_equal"
    assert evidence.decision == "advance"
    assert evidence.ndcg_at_10 == pytest.approx(0.4356252730181542)
    assert evidence.ndcg_delta == pytest.approx(0.015411280824014868)
    assert evidence.ndcg_ci_lower == pytest.approx(0.006964609514709062)
    assert evidence.mrr == pytest.approx(0.46545775232526737)
    assert evidence.zero_ndcg_queries == 92
    assert not hasattr(evidence, "per_query_rows")


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_metric", "ndcg_at_10 must be a finite number"),
        ("nonfinite_metric", "mrr must be a finite number"),
        ("out_of_range_metric", "ndcg_at_10 must be between 0 and 1"),
        ("missing_matched_parent", "evaluation.matched_parent_metrics must be an object"),
        ("drifted_matched_parent", "parent metrics do not match ledger"),
        ("reversed_bounds", "ndcg_at_10_delta bounds must contain"),
        ("invalid_bounds", "mrr_delta.lower must be a finite number"),
        ("wrong_primary_baseline", "primary baseline must be dense_profile_equal"),
        (
            "wrong_primary_candidate",
            "primary candidate must be raw_hierarchical_dense_equal",
        ),
        (
            "wrong_artifact_type",
            "artifact_type must be raw_response_refinement_results",
        ),
        ("not_advanced", "decision must be advance"),
        ("false_gate", "decision gates must all be true"),
        ("missing_gate", "decision gates keys must match frozen receipt contract"),
        ("unexpected_gate", "decision gates keys must match frozen receipt contract"),
        (
            "missing_integrity_check",
            "decision integrity checks keys must match frozen receipt contract",
        ),
        (
            "unexpected_integrity_check",
            "decision integrity checks keys must match frozen receipt contract",
        ),
        ("zero_count_exceeds_queries", "zero_ndcg_queries exceeds query count"),
        ("delta_metric_mismatch", "nDCG delta does not match candidate and parent"),
    ],
)
def test_completed_follow_up_loader_rejects_corrupt_receipt_fields(
    tmp_path: Path,
    ledger_build: LedgerBuild,
    corruption: str,
    message: str,
) -> None:
    payload = _completed_follow_up_receipt()
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    metrics = evaluation["metrics"]
    comparisons = evaluation["comparisons"]
    decision = evaluation["decision"]
    assert isinstance(metrics, dict)
    assert isinstance(comparisons, dict)
    assert isinstance(decision, dict)
    primary = comparisons["primary"]
    candidate_metrics = metrics["raw_hierarchical_dense_equal"]
    assert isinstance(primary, dict)
    assert isinstance(candidate_metrics, dict)
    if corruption == "missing_metric":
        candidate_metrics.pop("ndcg_at_10")
    elif corruption == "nonfinite_metric":
        candidate_metrics["mrr"] = math.inf
    elif corruption == "out_of_range_metric":
        candidate_metrics["ndcg_at_10"] = 1.1
    elif corruption == "missing_matched_parent":
        evaluation.pop("matched_parent_metrics")
    elif corruption == "drifted_matched_parent":
        matched_parent_metrics = evaluation["matched_parent_metrics"]
        assert isinstance(matched_parent_metrics, dict)
        matched_parent = matched_parent_metrics["dense_profile_equal"]
        assert isinstance(matched_parent, dict)
        matched_parent["ndcg_at_10"] = 0.0
    elif corruption == "reversed_bounds":
        primary["ndcg_at_10_delta"] = {
            "lower": 0.02,
            "mean_difference": 0.015,
            "upper": 0.01,
        }
    elif corruption == "invalid_bounds":
        primary["mrr_delta"] = {
            "lower": math.nan,
            "mean_difference": 0.012,
            "upper": 0.02,
        }
    elif corruption == "wrong_primary_baseline":
        primary["baseline"] = "dense_reference"
    elif corruption == "wrong_primary_candidate":
        primary["candidate"] = "raw_hierarchical"
    elif corruption == "wrong_artifact_type":
        payload["artifact_type"] = "wrong_type"
    elif corruption == "not_advanced":
        decision["decision"] = "informational"
    elif corruption == "false_gate":
        gates = decision["gates"]
        assert isinstance(gates, dict)
        gates["integrity_passes"] = False
    elif corruption == "missing_gate":
        gates = decision["gates"]
        assert isinstance(gates, dict)
        gates.pop("zero_count_reduced")
    elif corruption == "unexpected_gate":
        gates = decision["gates"]
        assert isinstance(gates, dict)
        gates["new_unfrozen_gate"] = True
    elif corruption == "missing_integrity_check":
        checks = decision["integrity_checks"]
        assert isinstance(checks, dict)
        checks.pop("trec_completeness")
    elif corruption == "unexpected_integrity_check":
        checks = decision["integrity_checks"]
        assert isinstance(checks, dict)
        checks["new_unfrozen_check"] = True
    elif corruption == "zero_count_exceeds_queries":
        candidate_metrics["zero_ndcg_queries"] = 323
    elif corruption == "delta_metric_mismatch":
        candidate_metrics["ndcg_at_10"] = 0.5
    else:
        raise AssertionError(f"unhandled corruption: {corruption}")
    _write_completed_follow_up_receipt(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        build_module._load_completed_follow_up(
            tmp_path,
            _completed_follow_up_sources(),
            ledger_build,
        )


@pytest.mark.parametrize(
    ("sources", "ledger", "message"),
    [
        (
            _completed_follow_up_sources(status="running"),
            None,
            "study status must be concluded",
        ),
        (
            _completed_follow_up_sources(system_ids=["dense_profile_equal"]),
            None,
            "must not link ledger system IDs",
        ),
        (
            _completed_follow_up_sources(evidence_refs=[]),
            None,
            "must reference",
        ),
        (
            _completed_follow_up_sources(),
            "drifted",
            "parent metrics do not match ledger",
        ),
    ],
)
def test_completed_follow_up_loader_rejects_invalid_study_or_parent_binding(
    tmp_path: Path,
    ledger_build: LedgerBuild,
    sources: HubSources,
    ledger: str | None,
    message: str,
) -> None:
    _write_completed_follow_up_receipt(tmp_path, _completed_follow_up_receipt())
    bound_ledger = (
        _ledger_with_parent_metric_drift(ledger_build)
        if ledger == "drifted"
        else ledger_build
    )

    with pytest.raises(ValueError, match=message):
        build_module._load_completed_follow_up(tmp_path, sources, bound_ledger)


def _failure_sources(
    *,
    study_status: str = "concluded",
    fusion_count: int = 4,
    no_signal_count: int = 7,
) -> HubSources:
    study = HubRecord(
        kind="study",
        record_id="sparse-failure-analogue-diagnosis",
        title="Sparse-failure analogue diagnosis",
        source_path="research/hub/studies/sparse-failure-analogue-diagnosis.qmd",
        metadata={
            "status": study_status,
            "evidence_refs": [REPORT_PATH],
            "failure_cohorts": {
                "fusion_21": {
                    "count": fusion_count,
                    "label": "Fusion dilution",
                    "definition": "One component retains top-ten signal.",
                },
                "no_signal_54": {
                    "count": no_signal_count,
                    "label": "No signal",
                    "definition": "Neither component retains top-ten signal.",
                },
            },
        },
        body="",
    )
    return HubSources(studies=(study,))


def _write_failure_receipt(
    root: Path,
    *,
    status: str = "complete",
    fusion_count: int = 4,
    no_signal_count: int = 7,
) -> None:
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "fusion_21": {"count": fusion_count},
                "no_signal_54": {"count": no_signal_count},
            }
        ),
        encoding="utf-8",
    )


def test_failure_split_is_authored_metadata_verified_against_complete_receipt(
    tmp_path: Path,
) -> None:
    _write_failure_receipt(tmp_path)

    result = build_module._load_failure_diagnostic(tmp_path, _failure_sources())

    assert result == {
        "fusion_21": {
            "count": 4,
            "label": "Fusion dilution",
            "definition": "One component retains top-ten signal.",
        },
        "no_signal_54": {
            "count": 7,
            "label": "No signal",
            "definition": "Neither component retains top-ten signal.",
        },
    }


@pytest.mark.parametrize(
    ("receipt_status", "fusion_count", "message"),
    [
        ("running", 4, "status must be complete"),
        ("complete", 5, "does not match authored count"),
    ],
)
def test_failure_split_rejects_incomplete_or_drifted_receipt(
    tmp_path: Path,
    receipt_status: str,
    fusion_count: int,
    message: str,
) -> None:
    _write_failure_receipt(
        tmp_path,
        status=receipt_status,
        fusion_count=fusion_count,
    )

    with pytest.raises(ValueError, match=message):
        build_module._load_failure_diagnostic(tmp_path, _failure_sources())


def test_failure_split_rejects_unconcluded_authored_study(tmp_path: Path) -> None:
    _write_failure_receipt(tmp_path)

    with pytest.raises(ValueError, match="study status must be concluded"):
        build_module._load_failure_diagnostic(
            tmp_path,
            _failure_sources(study_status="running"),
        )


def test_failure_split_loader_accepts_zero_sized_nonnegative_cohorts(
    tmp_path: Path,
) -> None:
    _write_failure_receipt(tmp_path, fusion_count=0, no_signal_count=0)

    result = build_module._load_failure_diagnostic(
        tmp_path,
        _failure_sources(fusion_count=0, no_signal_count=0),
    )

    assert result["fusion_21"]["count"] == 0
    assert result["no_signal_54"]["count"] == 0


def test_all_zero_representative_metrics_render_without_division_error() -> None:
    build = build_research_hub(ROOT)
    representative_ids = {
        "dense_nested",
        "contextual_e5",
        "local_domain_prior",
        "response_sumscore_nested",
        "dense_profile_equal",
        "profile_only",
        "dense_profile_profile_heavy",
    }
    systems = tuple(
        replace(system, ndcg_at_10=0.0)
        if system.system_id in representative_ids
        else system
        for system in build.ledger.systems
    )

    rendered = render_current_state_qmd(
        replace(build, ledger=replace(build.ledger, systems=systems))
    )

    assert rendered.count("--hub-bar: 0.000%") == 7


def test_zero_sized_failure_cohorts_are_valid_nonnegative_evidence() -> None:
    build = replace(
        build_research_hub(ROOT),
        failure_diagnostic={
            "fusion_21": {
                "count": 0,
                "label": "Fusion dilution",
                "definition": "One component retains top-ten signal.",
            },
            "no_signal_54": {
                "count": 0,
                "label": "No signal",
                "definition": "Neither component retains top-ten signal.",
            },
        },
    )

    rendered = render_current_state_qmd(build)

    assert rendered.count("<strong>0</strong>") >= 2
    assert "0 sole-positive top-ten failures" in rendered


def test_dependency_map_uses_authored_snapshot_not_mutable_github_state() -> None:
    build = build_research_hub(ROOT)
    plan = next(
        record
        for record in build.sources.background
        if record.record_id == "pretest-readiness-plan"
    )
    snapshot = dict(plan.metadata["dependency_snapshot"])
    issues = dict(snapshot["issues"])
    issues["inference"] = {
        **issues["inference"],
        "label": "Authored pipeline boundary",
    }
    snapshot["issues"] = issues
    changed_plan = replace(
        plan,
        metadata={**plan.metadata, "dependency_snapshot": snapshot},
    )
    backgrounds = tuple(
        changed_plan if record.record_id == plan.record_id else record
        for record in build.sources.background
    )

    rendered = render_current_state_qmd(
        replace(build, sources=replace(build.sources, background=backgrounds))
    )

    assert "Authored pipeline boundary" in rendered
    assert "Authored dependency snapshot" in rendered
    assert "Live GitHub issues" not in rendered
    assert "Open GitHub for current title, status, labels, and milestone" in rendered


def test_dependency_map_rejects_edges_that_disagree_with_visible_lanes() -> None:
    build = build_research_hub(ROOT)
    plan = next(
        record
        for record in build.sources.background
        if record.record_id == "pretest-readiness-plan"
    )
    snapshot = dict(plan.metadata["dependency_snapshot"])
    edges = [dict(edge) for edge in snapshot["edges"]]
    edges[0]["to"] = "organizer-release"
    snapshot["edges"] = edges
    changed_plan = replace(
        plan,
        metadata={**plan.metadata, "dependency_snapshot": snapshot},
    )
    backgrounds = tuple(
        changed_plan if record.record_id == plan.record_id else record
        for record in build.sources.background
    )

    with pytest.raises(ValueError, match="edges must match the visible lane topology"):
        render_current_state_qmd(
            replace(build, sources=replace(build.sources, background=backgrounds))
        )


def test_strongest_followup_claim_requires_authored_study_role() -> None:
    build = build_research_hub(ROOT)
    e5 = next(
        study
        for study in build.sources.studies
        if study.record_id == "e5-raw-response-refinement"
    )
    metadata = dict(e5.metadata)
    metadata.pop("current_state_role", None)
    studies = tuple(
        replace(e5, metadata=metadata) if study.record_id == e5.record_id else study
        for study in build.sources.studies
    )

    with pytest.raises(ValueError, match="authored strongest completed follow-up"):
        render_current_state_qmd(
            replace(build, sources=replace(build.sources, studies=studies))
        )


def test_strongest_followup_claim_rejects_an_unfinished_e5_study() -> None:
    build = build_research_hub(ROOT)
    e5 = next(
        study
        for study in build.sources.studies
        if study.record_id == "e5-raw-response-refinement"
    )
    running = replace(
        e5,
        metadata={
            **e5.metadata,
            "status": "running",
            "outcome": "pending",
            "decision": "pending",
            "concluded_on": None,
        },
    )
    studies = tuple(
        running if study.record_id == e5.record_id else study
        for study in build.sources.studies
    )

    with pytest.raises(ValueError, match="must be concluded"):
        render_current_state_qmd(
            replace(build, sources=replace(build.sources, studies=studies))
        )


def test_agent_readme_names_the_catalogs_canonical_path_field() -> None:
    build = build_research_hub(ROOT)
    readme = (ROOT / "research/README.md").read_text(encoding="utf-8")
    catalog = json.loads(render_catalog(build))

    assert all("canonical_path" in resource for resource in catalog["resources"])
    assert "Follow each `canonical_path`" in readme
    assert "Follow each `source_path`" not in readme
