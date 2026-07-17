from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from trec_mlm.experiment_ledger import (
    ARTIFACT_STATES,
    EVALUATION_SCOPES,
    EVIDENCE_STATUSES,
    PROMOTION_STATUSES,
    QREL_USAGE_STATUSES,
    LedgerBuild,
    build_ledger,
    density_bucket,
    load_registry,
    resolve_json_pointer,
    validate_registry,
)


def minimal_registry() -> dict[str, object]:
    return {
        "version": 1,
        "sources": {
            "receipt": {
                "path": "reports/result.json",
                "sha256": "0" * 64,
            }
        },
        "caveats": {
            "development_only": {
                "scope": "program",
                "summary": "Development evidence only.",
            }
        },
        "experiments": [
            {
                "experiment_id": "example",
                "experiment_family": "control",
                "systems": [
                    {
                        "system_id": "example_system",
                        "source_system_id": "example_source",
                        "aliases": [],
                        "method_summary": "Example.",
                        "evaluation_scope": "dev_322x150",
                        "evidence_status": "aggregate_only",
                        "promotion_status": "valid_negative",
                        "query_count": 322,
                        "candidate_count": 150,
                        "aggregate_source": {
                            "source_id": "receipt",
                            "pointer": "/metrics/example",
                        },
                        "per_query_source": None,
                        "per_query_blocker": "Tracked source stores aggregates only.",
                        "ranking_coverage": "none",
                        "native_scores_available": False,
                        "artifact_state": "tracked_receipt_only",
                        "qrel_usage": "evaluation_only",
                        "run_classification": "internal_automatic",
                        "caveat_ids": ["development_only"],
                    },
                    {
                        "system_id": "example_primary",
                        "source_system_id": "example_primary_source",
                        "aliases": [],
                        "method_summary": "Example promoted primary.",
                        "evaluation_scope": "dev_322x150",
                        "evidence_status": "stored_per_query",
                        "promotion_status": "promoted_development",
                        "query_count": 322,
                        "candidate_count": 150,
                        "aggregate_source": {
                            "source_id": "receipt",
                            "pointer": "/metrics/example",
                        },
                        "per_query_source": {
                            "source_id": "receipt",
                            "pointer": "/metrics/example/per_query",
                        },
                        "per_query_blocker": None,
                        "ranking_coverage": "per_query_metrics_only",
                        "native_scores_available": False,
                        "artifact_state": "tracked_complete",
                        "qrel_usage": "evaluation_only",
                        "run_classification": "internal_automatic",
                        "caveat_ids": ["development_only"],
                    },
                ],
            }
        ],
    }


def write_declared_source(
    tmp_path: Path,
    registry: dict[str, object],
    payload: dict[str, Any] | None = None,
) -> Path:
    source_path = tmp_path / "reports" / "result.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps(payload or {"metrics": {"example": {}}}), encoding="utf-8"
    )
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    registry["sources"]["receipt"]["sha256"] = digest  # type: ignore[index]
    return source_path


def build_fixture_ledger(
    tmp_path: Path,
    *,
    per_query_count: int = 2,
    reported_ndcg: object | None = None,
    declare_missing_issue18_alias: bool = False,
    declare_issue18_alias_without_dense: bool = False,
) -> LedgerBuild:
    query_ids = [f"q{index}" for index in range(1, per_query_count + 1)]
    candidate_ids = ["A", "B", *(f"filler_{index:03d}" for index in range(148))]
    metrics = {
        query_id: {
            "ndcg_at_k": 0.5 if index == 1 else 0.0,
            "reciprocal_rank": 1.0 if index == 1 else (0.25 if index == 2 else 0.0),
        }
        for index, query_id in enumerate(query_ids, start=1)
    }
    macro_ndcg = sum(row["ndcg_at_k"] for row in metrics.values()) / len(metrics)
    macro_mrr = sum(row["reciprocal_rank"] for row in metrics.values()) / len(metrics)

    reports = tmp_path / "reports"
    data = tmp_path / "data" / "dev"
    experiments = tmp_path / "experiments"
    reports.mkdir(parents=True)
    data.mkdir(parents=True)
    experiments.mkdir(parents=True)

    receipt_path = reports / "result.json"
    receipt_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "example": {
                        "ndcg_at_k": (
                            macro_ndcg if reported_ndcg is None else reported_ndcg
                        ),
                        "reciprocal_rank": macro_mrr,
                        "zero_ndcg_queries": sum(
                            row["ndcg_at_k"] == 0.0 for row in metrics.values()
                        ),
                        "per_query": metrics,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    query_path = data / "queries.tsv"
    query_path.write_text(
        "".join(f"{query_id}\tQuestion {query_id}?\n" for query_id in query_ids),
        encoding="utf-8",
    )
    qrels_path = data / "qrels.txt"
    qrel_lines: list[str] = []
    for index, query_id in enumerate(query_ids, start=1):
        for candidate_id in candidate_ids:
            grade = 0
            if query_id == "q1" and candidate_id == "A":
                grade = 1
            elif query_id == "q1" and candidate_id == "B":
                grade = 2
            elif query_id == "q2" and candidate_id == "B":
                grade = 1
            elif index >= 3 and candidate_id == "A":
                grade = 1
            qrel_lines.append(f"{query_id} 0 {candidate_id} {grade}\n")
    qrels_path.write_text("".join(qrel_lines), encoding="utf-8")

    def source(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    full_scope = per_query_count != 2
    sources = {
        "receipt": source(receipt_path),
        "development_queries": source(query_path),
        "development_qrels": source(qrels_path),
    }
    if declare_missing_issue18_alias or declare_issue18_alias_without_dense:
        issue18_path = reports / "issue18.json"
        issue18_payload = (
            {"controls": {"query_knn": {"per_query": metrics}}}
            if declare_issue18_alias_without_dense
            else {}
        )
        issue18_path.write_text(json.dumps(issue18_payload), encoding="utf-8")
        sources["issue18_evaluation"] = source(issue18_path)

    registry = {
        "version": 1,
        "sources": sources,
        "caveats": {
            "development_only": {
                "scope": "program",
                "summary": "Development evidence only.",
            }
        },
        "experiments": [
            {
                "experiment_id": "example",
                "experiment_family": "control",
                "systems": [
                    {
                        "system_id": "example_system",
                        "source_system_id": "example_source",
                        "aliases": [],
                        "method_summary": "Example.",
                        "evaluation_scope": (
                            "dev_322x150" if full_scope else "restricted_6"
                        ),
                        "evidence_status": "stored_per_query",
                        "promotion_status": (
                            "promoted_development" if full_scope else "restricted_pilot"
                        ),
                        "query_count": 322 if full_scope else per_query_count,
                        "candidate_count": 150,
                        "aggregate_source": {
                            "kind": "json_pointer",
                            "source_id": "receipt",
                            "pointer": "/metrics/example",
                        },
                        "per_query_source": {
                            "kind": "issue18_per_query_map",
                            "source_id": "receipt",
                            "pointer": "/metrics/example/per_query",
                        },
                        "per_query_blocker": None,
                        "ranking_coverage": "per_query_metrics_only",
                        "native_scores_available": False,
                        "artifact_state": "tracked_complete",
                        "qrel_usage": "evaluation_only",
                        "run_classification": "internal_automatic",
                        "caveat_ids": ["development_only"],
                    }
                ],
            }
        ],
    }
    if not full_scope:
        promoted_fixture = deepcopy(registry["experiments"][0]["systems"][0])
        promoted_fixture.update(
            {
                "system_id": "fixture_promoted_primary",
                "source_system_id": "fixture_promoted_primary",
                "method_summary": "Synthetic full-scope primary for registry validation.",
                "evaluation_scope": "dev_322x150",
                "evidence_status": "recomputed_from_complete_ranking",
                "promotion_status": "promoted_development",
                "query_count": 322,
                "per_query_source": None,
                "per_query_blocker": "Synthetic fixture has no materialized vector.",
            }
        )
        registry["experiments"][0]["systems"].append(promoted_fixture)
    registry_path = experiments / "ledger.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return build_ledger(tmp_path)


def test_resolve_json_pointer_handles_escaped_tokens() -> None:
    value = {"a/b": {"~key": 7}}
    assert resolve_json_pointer(value, "/a~1b/~0key") == 7


def test_validate_registry_rejects_duplicate_system_ids(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["experiments"].append(registry["experiments"][0])  # type: ignore[union-attr,index]
    with pytest.raises(ValueError, match="duplicate experiment_id"):
        validate_registry(registry, tmp_path)


def test_validate_registry_requires_aggregate_only_blocker(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["experiments"][0]["systems"][0]["per_query_blocker"] = ""  # type: ignore[index]
    with pytest.raises(ValueError, match="per_query_blocker"):
        validate_registry(registry, tmp_path)


@pytest.mark.parametrize("blocker", ["   ", 7])
def test_validate_registry_requires_concrete_string_blocker(
    tmp_path: Path, blocker: object
) -> None:
    registry = minimal_registry()
    registry["experiments"][0]["systems"][0]["per_query_blocker"] = blocker  # type: ignore[index]
    with pytest.raises(ValueError, match="per_query_blocker"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_absolute_source_path(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["sources"]["receipt"]["path"] = "/tmp/result.json"  # type: ignore[index]
    with pytest.raises(ValueError, match="repository-relative"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_duplicate_canonical_system_id(
    tmp_path: Path,
) -> None:
    registry = minimal_registry()
    duplicate = deepcopy(registry["experiments"][0])  # type: ignore[index]
    duplicate["experiment_id"] = "other"
    registry["experiments"].append(duplicate)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="duplicate system_id"):
        validate_registry(registry, tmp_path)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("evaluation_scope", "test_2026", "unknown evaluation_scope"),
        ("evidence_status", "probably_complete", "unknown evidence_status"),
        ("promotion_status", "winner", "unknown promotion_status"),
        ("artifact_state", "somewhere_local", "unknown artifact_state"),
        ("qrel_usage", "unknown", "unknown qrel_usage"),
    ],
)
def test_validate_registry_rejects_unknown_vocabulary(
    tmp_path: Path,
    field: str,
    bad_value: str,
    message: str,
) -> None:
    registry = minimal_registry()
    registry["experiments"][0]["systems"][0][field] = bad_value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_unknown_source_reference(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["experiments"][0]["systems"][0]["aggregate_source"]["source_id"] = (  # type: ignore[index]
        "unknown"
    )
    with pytest.raises(ValueError, match="unknown source IDs"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_unknown_caveat(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["experiments"][0]["systems"][0]["caveat_ids"] = ["unknown"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown caveat IDs"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_parent_source_path(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["sources"]["receipt"]["path"] = "../result.json"  # type: ignore[index]
    with pytest.raises(ValueError, match="repository-relative"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_changed_source_digest(tmp_path: Path) -> None:
    registry = minimal_registry()
    source_path = write_declared_source(tmp_path, registry)
    source_path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source digest changed"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_unreviewed_grid_key(tmp_path: Path) -> None:
    registry = minimal_registry()
    write_declared_source(
        tmp_path,
        registry,
        {"fixed_metrics": {"approved": {}, "new_unreviewed": {}}},
    )
    registry["experiments"][0]["grid_contracts"] = [  # type: ignore[index]
        {
            "source_id": "receipt",
            "pointer": "/fixed_metrics",
            "allowed_keys": ["approved"],
            "inherited_metadata": {"promotion_status": "diagnostic_only"},
        }
    ]
    with pytest.raises(ValueError, match="unexpected grid keys"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_duplicate_promoted_development_systems(
    tmp_path: Path,
) -> None:
    registry = minimal_registry()
    registry["experiments"][0]["systems"][0]["promotion_status"] = (  # type: ignore[index]
        "promoted_development"
    )
    with pytest.raises(ValueError, match="exactly one promoted_development"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_diagnostic_as_only_promoted_system() -> None:
    registry = json.loads(Path("experiments/ledger.json").read_text(encoding="utf-8"))
    systems = {
        system["system_id"]: system
        for experiment in registry["experiments"]
        for system in experiment["systems"]
    }
    systems["dense_profile_equal"]["promotion_status"] = "diagnostic_only"
    systems["dense_profile_profile_heavy"]["promotion_status"] = "promoted_development"
    with pytest.raises(ValueError, match="promotion-blocking caveats"):
        validate_registry(registry, Path("."))


@pytest.mark.parametrize("system_id", ["pointwise_qwen", "generated_listwise"])
def test_validate_registry_rejects_restricted_or_invalid_promoted_system(
    system_id: str,
) -> None:
    registry = json.loads(Path("experiments/ledger.json").read_text(encoding="utf-8"))
    systems = {
        system["system_id"]: system
        for experiment in registry["experiments"]
        for system in experiment["systems"]
    }
    systems["dense_profile_equal"]["promotion_status"] = "diagnostic_only"
    systems[system_id]["promotion_status"] = "promoted_development"
    with pytest.raises(ValueError, match="promoted_development.*dev_322x150"):
        validate_registry(registry, Path("."))


def test_reviewed_registry_has_required_families_and_systems() -> None:
    registry = load_registry(Path("."), Path("experiments/ledger.json"))
    experiments = {
        experiment["experiment_family"]: experiment
        for experiment in registry["experiments"]
    }
    assert set(experiments) == {
        "behavior_profile",
        "candidate_label_matching",
        "dense_query_knn",
        "label_ranker_pilots",
        "multiresolution_profiles",
        "query_cluster_aggregation",
        "query_controls",
        "response_retrieval",
        "response_volume_normalization",
        "semantic_profiles",
        "source_query_first",
    }

    system_ids = {
        system["system_id"]
        for experiment in registry["experiments"]
        for system in experiment["systems"]
    }
    assert {
        "global_prior",
        "tfidf_fixed_k10",
        "dense_nested",
        "mpnet_raw_label",
        "contextual_e5",
        "pointwise_qwen",
        "generated_listwise",
        "grouprank_7b",
        "firstmistral_listwise",
        "response_sumscore_nested",
        "response_top10sum_nested",
        "source_alignment_nested",
        "target_response_same_k",
        "discovery_behavior_topic_v1",
        "profile_only",
        "dense_profile_equal",
        "dense_profile_profile_heavy",
        "a1_top2_k10",
        "x1_multires_top2_dense_equal",
        "local_domain_prior",
    } <= system_ids
    assert len(system_ids) == 98

    assert (
        len(experiments["dense_query_knn"]["grid_contracts"][0]["allowed_keys"]) == 28
    )
    assert (
        len(experiments["response_retrieval"]["grid_contracts"][0]["allowed_keys"])
        == 24
    )
    assert (
        len(experiments["source_query_first"]["grid_contracts"][0]["allowed_keys"]) == 3
    )
    assert experiments["response_volume_normalization"]["metadata"][
        "tested_aggregation_order"
    ] == ["best_hit", "top3_sum", "top10_sum", "top3_mean", "top10_mean", "mean_all"]


def test_reviewed_registry_preserves_six_query_feasibility_history() -> None:
    registry = load_registry(Path("."), Path("experiments/ledger.json"))
    experiment = next(
        row
        for row in registry["experiments"]
        if row["experiment_family"] == "label_ranker_pilots"
    )
    systems = {row["system_id"]: row for row in experiment["systems"]}

    pointwise = systems["pointwise_qwen_feasibility_6"]
    generated = systems["generated_listwise_feasibility_6"]
    assert pointwise["evaluation_scope"] == "restricted_6"
    assert generated["evaluation_scope"] == "restricted_6"
    assert pointwise["evidence_status"] == "restricted_diagnostic"
    assert generated["evidence_status"] == "restricted_diagnostic"
    assert pointwise["promotion_status"] == "restricted_pilot"
    assert generated["promotion_status"] == "diagnostic_only"
    assert pointwise["query_count"] == generated["query_count"] == 6
    assert pointwise["candidate_count"] is None
    assert generated["candidate_count"] is None
    assert pointwise["aggregate_source"]["row"] == "Six-question LLM pilot"
    assert generated["aggregate_source"]["row"] == "Six-question LLM pilot"
    assert pointwise["aggregate_source"]["metrics"]["relevant_in_top_3"] == "5/6"
    assert generated["aggregate_source"]["metrics"]["relevant_in_top_3"] == "3/6"
    assert pointwise["per_query_blocker"].strip()
    assert generated["per_query_blocker"].strip()
    assert "six_query_signal_did_not_scale" in pointwise["caveat_ids"]
    assert "six_query_signal_did_not_scale" in generated["caveat_ids"]

    later_pointwise = systems["pointwise_qwen"]
    assert later_pointwise["evaluation_scope"] == "restricted_12x20"
    assert later_pointwise["aggregate_source"]["metrics"]["relevant_in_top_3"] == "5/12"
    assert "six_query_signal_did_not_scale" in later_pointwise["caveat_ids"]


def test_reviewed_registry_preserves_superseded_initial_embedding_sweep() -> None:
    registry = load_registry(Path("."), Path("experiments/ledger.json"))
    systems = {
        system["system_id"]: system
        for experiment in registry["experiments"]
        for system in experiment["systems"]
    }
    initial = [
        systems["qwen8b_initial_terminal_token_invalid"],
        systems["e5_initial_terminal_token_invalid"],
    ]

    assert initial[0]["aggregate_source"] == initial[1]["aggregate_source"]
    assert initial[0]["aggregate_source"] == {
        "kind": "literal_snapshot",
        "source_id": "expert_label_ranking_study",
        "section": "What we tried",
        "table": "Step | Scope | Outcome | What it established",
        "row": "Initial Qwen8B and E5 sweep",
        "metrics": {
            "outcome": "Invalid low scores",
            "scope": "322 × 150",
            "what_it_established": (
                "Required model-specific terminal-token behavior was omitted. "
                "The runtime does not invent each embedding checkpoint's "
                "asymmetric contract."
            ),
        },
    }
    for system in initial:
        assert system["evaluation_scope"] == "invalid_unscored"
        assert system["evidence_status"] == "invalid"
        assert system["promotion_status"] == "invalid"
        assert system["query_count"] == 322
        assert system["candidate_count"] == 150
        assert "initial_embedding_prompt_contract_superseded" in system["caveat_ids"]
        assert (
            not {
                "ndcg_at_10",
                "ndcg_at_k",
                "mrr",
                "reciprocal_rank",
                "zero_ndcg_queries",
                "zero_ndcg_count",
            }
            & system["aggregate_source"]["metrics"].keys()
        )


def test_reviewed_registry_preserves_matched_support_preflight_history() -> None:
    registry = load_registry(Path("."), Path("experiments/ledger.json"))
    systems = {
        system["system_id"]: system
        for experiment in registry["experiments"]
        for system in experiment["systems"]
    }
    expected = {
        "hard_cluster_k20_preflight": "/cluster_counts/20/cluster_mean",
        "matched_neighbor_k20_preflight": ("/cluster_counts/20/matched_neighbor_mean"),
        "hard_cluster_k40_preflight": "/cluster_counts/40/cluster_mean",
        "matched_neighbor_k40_preflight": ("/cluster_counts/40/matched_neighbor_mean"),
    }
    for system_id, pointer in expected.items():
        system = systems[system_id]
        assert system["aggregate_source"] == {
            "kind": "json_pointer",
            "source_id": "issue13_matched_support_preflight",
            "pointer": pointer,
        }
        assert system["evaluation_scope"] == "dev_322x150"
        assert system["evidence_status"] == "aggregate_only"
        assert system["promotion_status"] == "diagnostic_only"
        assert system["query_count"] == 322
        assert system["candidate_count"] == 150
        assert system["artifact_state"] == "tracked_receipt_only"
        assert system["qrel_usage"] == "training_and_evaluation"
        assert {
            "qrel_transferred_training",
            "hard_cluster_posthoc_preflight",
            "hard_cluster_preflight_not_promotable",
            "aggregate_per_query_missing",
        } <= set(system["caveat_ids"])


def test_reviewed_registry_does_not_infer_grouprank_scope_counts() -> None:
    registry = load_registry(Path("."), Path("experiments/ledger.json"))
    grouprank = next(
        system
        for experiment in registry["experiments"]
        for system in experiment["systems"]
        if system["system_id"] == "grouprank_7b"
    )

    assert grouprank["query_count"] is None
    assert grouprank["candidate_count"] is None
    assert grouprank["qrel_usage"] == "not_evaluated"
    assert "restricted_not_comparable" not in grouprank["caveat_ids"]
    assert "restricted_qrel_assisted" not in grouprank["caveat_ids"]


def test_frozen_vocabularies_match_approved_design() -> None:
    assert EVALUATION_SCOPES == {
        "dev_322x150",
        "restricted_12x20",
        "restricted_6",
        "invalid_unscored",
        "constructed_not_evaluated",
    }
    assert EVIDENCE_STATUSES == {
        "stored_per_query",
        "recomputed_from_complete_ranking",
        "aggregate_only",
        "restricted_diagnostic",
        "invalid",
        "unscored",
    }
    assert PROMOTION_STATUSES == {
        "promoted_development",
        "stable_control",
        "valid_negative",
        "diagnostic_only",
        "restricted_pilot",
        "invalid",
        "stopped_unscored",
        "constructed_not_evaluated",
    }
    assert ARTIFACT_STATES == {
        "tracked_complete",
        "tracked_receipt_only",
        "tracked_report_snapshot",
        "local_artifact_missing",
        "construction_artifact_missing",
    }
    assert QREL_USAGE_STATUSES == {
        "training_and_evaluation",
        "evaluation_only",
        "restricted_selection_and_evaluation",
        "not_evaluated",
    }


def test_build_ledger_recomputes_macro_and_overlapping_slices(
    tmp_path: Path,
) -> None:
    build = build_fixture_ledger(tmp_path)
    system = next(row for row in build.systems if row.system_id == "example_system")
    assert system.ndcg_at_10 == pytest.approx(0.25)
    assert system.mrr == pytest.approx(0.625)
    slices = {(row.slice_key, row.query_count): row for row in build.topic_slices}
    assert slices[("A", 1)].ndcg_at_10 == pytest.approx(0.5)
    assert slices[("B", 2)].ndcg_at_10 == pytest.approx(0.25)
    assert all(row.overlapping for row in build.topic_slices)


def test_build_ledger_requires_322_rows_for_full_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 322"):
        build_fixture_ledger(tmp_path, per_query_count=321)


def test_build_ledger_rejects_macro_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="macro metric"):
        build_fixture_ledger(tmp_path, reported_ndcg=0.9)


def test_build_ledger_keeps_numeric_metric_validation_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="quality metric ndcg_at_10 must be numeric"):
        build_fixture_ledger(tmp_path, reported_ndcg="five sixths")


def test_build_ledger_rejects_declared_issue18_source_without_dense_alias(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="query_knn"):
        build_fixture_ledger(tmp_path, declare_missing_issue18_alias=True)


def test_build_ledger_rejects_issue18_alias_without_canonical_dense_vector(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="dense_nested"):
        build_fixture_ledger(tmp_path, declare_issue18_alias_without_dense=True)


def test_density_slices_are_disjoint() -> None:
    assert density_bucket(1) == "at_most_5_positives"
    assert density_bucket(5) == "at_most_5_positives"
    assert density_bucket(6) == "6_to_20_positives"
    assert density_bucket(20) == "6_to_20_positives"
    assert density_bucket(21) == "more_than_20_positives"


def test_real_ledger_extracts_complete_recoverable_vectors() -> None:
    build = build_ledger(Path("."))
    counts = Counter(row.system_id for row in build.per_query)
    recoverable_systems = {
        "dense_nested",
        "contextual_e5",
        "profile_only",
        "dense_profile_equal",
        "dense_profile_profile_heavy",
        "a1_top2_k10",
        "a1_top2_k10_dense_equal",
        "r1_multires_max",
        "r1_multires_max_dense_equal",
        "x1_multires_top2",
        "x1_multires_top2_dense_equal",
    }
    assert set(counts) == recoverable_systems
    assert set(counts.values()) == {322}
    assert (
        sum(
            row.query_count
            for row in build.density_slices
            if row.system_id == "dense_nested"
        )
        == 322
    )
    dense_supports = {
        row.slice_key: row.query_count
        for row in build.density_slices
        if row.system_id == "dense_nested"
    }
    assert dense_supports == {
        "at_most_5_positives": 248,
        "6_to_20_positives": 30,
        "more_than_20_positives": 44,
    }

    systems = {row.system_id: row for row in build.systems}
    for system_id in (
        "response_sumscore_nested",
        "source_alignment_nested",
        "mpnet_raw_label",
    ):
        assert not systems[system_id].per_query_available
        assert systems[system_id].blocker

    assert systems["grouprank_7b"].query_count is None
    assert systems["grouprank_7b"].candidate_count is None

    dense_receipt = json.loads(
        Path("reports/trec_2026_dense_query_knn_results.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]["dense_nested"]
    dense = systems["dense_nested"]
    assert dense.ndcg_at_10 == dense_receipt["ndcg_at_10"]
    assert dense.mrr == dense_receipt["mrr"]
    assert dense.zero_ndcg_queries is None
    assert dense.recomputed_ndcg_at_10 == pytest.approx(dense.ndcg_at_10, abs=1e-12)
    assert dense.recomputed_mrr == pytest.approx(dense.mrr, abs=1e-12)
    assert dense.recomputed_zero_ndcg_queries == 116
    assert dense.recomputed_ndcg_at_10 != dense.ndcg_at_10

    aggregate_payloads = {
        system_id: json.loads(systems[system_id].aggregate_payload_json)
        for system_id in (
            "pointwise_qwen_feasibility_6",
            "generated_listwise_feasibility_6",
            "qwen4b_contextual",
            "local_domain_prior",
        )
    }
    reviewed_registry = load_registry(Path("."), Path("experiments/ledger.json"))
    reviewed_systems = {
        system["system_id"]: system
        for experiment in reviewed_registry["experiments"]
        for system in experiment["systems"]
    }
    for system_id, payload in aggregate_payloads.items():
        assert payload == reviewed_systems[system_id]["aggregate_source"]["metrics"]
        assert systems[system_id].aggregate_payload_json == json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    assert (
        aggregate_payloads["pointwise_qwen_feasibility_6"]["relevant_in_top_3"] == "5/6"
    )
    assert (
        aggregate_payloads["generated_listwise_feasibility_6"]["relevant_in_top_3"]
        == "3/6"
    )
    assert aggregate_payloads["qwen4b_contextual"]["strict_repeat"].startswith("Fail")
    assert aggregate_payloads["local_domain_prior"][
        "ndcg_delta_vs_dense"
    ] == pytest.approx(-0.0111738962)
    assert systems["hard_cluster_k20_preflight"].zero_ndcg_queries == 148
    assert systems["matched_neighbor_k20_preflight"].zero_ndcg_queries == 124
    assert systems["hard_cluster_k40_preflight"].zero_ndcg_queries == 154
    assert systems["matched_neighbor_k40_preflight"].zero_ndcg_queries == 136
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

    dense_topic = next(
        row
        for row in build.topic_slices
        if row.system_id == "dense_nested" and row.slice_key == "Anthropology"
    )
    source_registry = reviewed_registry["sources"]
    assert (
        dense_topic.metric_source_path
        == (source_registry["dense_query_per_query"]["path"])
    )
    assert (
        dense_topic.metric_source_sha256
        == (source_registry["dense_query_per_query"]["sha256"])
    )
    assert (
        dense_topic.query_source_path == source_registry["development_queries"]["path"]
    )
    assert (
        dense_topic.query_source_sha256
        == source_registry["development_queries"]["sha256"]
    )
    assert dense_topic.qrel_source_path == source_registry["development_qrels"]["path"]
    assert (
        dense_topic.qrel_source_sha256 == source_registry["development_qrels"]["sha256"]
    )
    assert build.validation["dense_alias_max_abs_delta"] is not None
