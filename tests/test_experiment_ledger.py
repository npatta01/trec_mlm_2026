from __future__ import annotations

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
                    }
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
    assert len(system_ids) == 90

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
