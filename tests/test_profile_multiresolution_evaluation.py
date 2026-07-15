from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_mlm.profile_multiresolution import (
    EXPECTED_SYSTEMS,
    NEW_PROFILE_SYSTEMS,
    VIABILITY_THRESHOLDS,
    MultiResolutionConfig,
)
from trec_mlm.profile_multiresolution_evaluation import (
    evaluate_multiresolution,
    validate_multiresolution_qrel_inventories,
    validate_viability_receipt,
)


def _rankings() -> dict[str, dict[str, list[str]]]:
    rows = {
        "forward": {"q1": ["A", "B"], "q2": ["A", "B"]},
        "reverse": {"q1": ["B", "A"], "q2": ["B", "A"]},
    }
    return {
        name: rows["forward" if index % 2 == 0 else "reverse"]
        for index, name in enumerate(EXPECTED_SYSTEMS)
    }


def _valid_viability_receipt() -> dict:
    cluster_values = {
        "candidate_occupied_cluster_retention": 1.0,
        "largest_cluster_share": 0.2,
        "median_normalized_occupancy_entropy": 0.9,
        "minimum_normalized_occupancy_entropy": 0.8,
        "singleton_cluster_share": 0.1,
        "exact_duplicate_centroids": 0,
        "within_candidate_near_duplicate_centroid_share": 0.0,
        "median_centroid_separation": 0.1,
    }
    base_ranking = {
        "unique_top10_rankings": 300,
        "maximum_top1_candidate_share": 0.05,
        "top10_candidate_coverage": 100,
        "maximum_candidate_top10_query_share": 0.2,
    }
    incremental = {
        "top10_difference_query_share": 0.2,
        "mean_top10_jaccard": 0.8,
        "unique_top10_retention": 1.0,
        "candidate_coverage_retention": 1.0,
    }
    component_systems = {
        f"{label}_k{resolution}"
        for resolution in (5, 10, 20)
        for label in ("max", "top2")
    }
    diagnostic_systems = component_systems | set(EXPECTED_SYSTEMS)
    thresholds = {}
    for name, (operator, threshold) in VIABILITY_THRESHOLDS.items():
        if name in cluster_values:
            values = {f"k{k}": cluster_values[name] for k in (5, 10, 20)}
        elif name.startswith("incremental_"):
            field = name.removeprefix("incremental_")
            values = {system: incremental[field] for system in NEW_PROFILE_SYSTEMS}
        else:
            values = {system: base_ranking[name] for system in diagnostic_systems}
        thresholds[name] = {
            "operator": operator,
            "threshold": threshold,
            "values": values,
            "passed": True,
        }
    integrity = {
        "source_binding": True,
        "query_identities": True,
        "candidate_identities": True,
        "official_322_by_150_coverage": True,
        "complete_rankings": True,
        "component_score_schema": True,
        "system_score_schema": True,
        "contributor_schema": True,
        "finite_scores": True,
        "ranking_score_consistency": True,
        "normalized_centroids": True,
    }
    return {
        "artifact_type": "issue18_multiresolution_viability_receipt",
        "version": 1,
        "qrel_free": True,
        "artifact_aggregate_sha256": "a" * 64,
        "decision": "continue",
        "thresholds": thresholds,
        "integrity_checks": integrity,
        "failed_thresholds": [],
        "failed_integrity_thresholds": [],
        "failed_incremental_thresholds": [],
        "cluster_diagnostics": {
            f"k{k}": {"candidate_count": 150, **cluster_values} for k in (5, 10, 20)
        },
        "ranking_diagnostics": {
            system: {
                **base_ranking,
                **(incremental if system in NEW_PROFILE_SYSTEMS else {}),
            }
            for system in diagnostic_systems
        },
        "cost_evidence": {
            "issue12_construction_storage_bytes": 100,
            "multiresolution_artifact_storage_bytes": 200,
            "storage_ratio": 2.0,
            "storage_at_most_four_times_issue12": True,
            "issue12_construction_elapsed_seconds": 10.0,
            "multiresolution_construction_elapsed_seconds": 20.0,
            "wall_time_ratio": 2.0,
            "wall_time_at_most_four_times_issue12": True,
            "repeat_cost_authorized": True,
        },
        "repeat_authorized": True,
    }


def _construction(tmp_path: Path) -> tuple[Path, SimpleNamespace, SimpleNamespace]:
    root = tmp_path / "construction"
    root.mkdir()
    manifest = {
        "artifact_type": "issue18_multiresolution_profile_manifest",
        "version": 1,
        "qrel_free": True,
        "files": {},
        "aggregate_sha256": "a" * 64,
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    multi = SimpleNamespace(
        config=replace(MultiResolutionConfig(), construction_path=str(tmp_path / "issue12")),
        query_ids=("q1", "q2"),
        candidate_ids=("A", "B"),
        dense_rankings={"q1": ["A", "B"], "q2": ["B", "A"]},
        rankings=_rankings(),
    )
    source = SimpleNamespace(
        query_ids=multi.query_ids,
        candidate_ids=multi.candidate_ids,
        source_rankings={
            "dense_reference": multi.dense_rankings,
            "contextual_e5_control": {"q1": ["B", "A"], "q2": ["A", "B"]},
        },
        rankings={
            "profile_only": {"q1": ["B", "A"], "q2": ["B", "A"]},
            "dense_profile_profile_heavy": {"q1": ["B", "A"], "q2": ["B", "A"]},
        },
        root=tmp_path / "issue12",
        manifest={"aggregate_sha256": multi.config.source_aggregate_sha256},
    )
    source.root.mkdir()
    (source.root / "profiles.json").write_text(
        json.dumps({"A": {"available": True}, "B": {"available": True}}) + "\n"
    )
    (tmp_path / "qrels").write_text("q1 0 A 0\n", encoding="utf-8")
    (tmp_path / "viability.json").write_text(
        json.dumps(_valid_viability_receipt(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, multi, source


def _patch_loaders(monkeypatch: pytest.MonkeyPatch, multi, source) -> None:
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_multiresolution_artifact",
        lambda _: multi,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_profile_artifacts",
        lambda _: source,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_complete_qrels",
        lambda _, **__: {
            "q1": {"A": 0.0, "B": 1.0},
            "q2": {"A": 1.0, "B": 0.0},
        },
    )


def test_invalid_or_unfrozen_artifact_is_rejected_before_qrels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def open_qrels(*_args, **_kwargs):
        nonlocal opened
        opened = True
        return {}

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_complete_qrels", open_qrels
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        evaluate_multiresolution(
            tmp_path / "missing",
            tmp_path / "viability.json",
            tmp_path / "qrels",
            tmp_path / "out.json",
        )
    assert opened is False


@pytest.mark.parametrize("receipt_payload", (None, {"decision": "continue"}))
def test_missing_or_forged_receipt_cannot_touch_qrels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_payload,
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    receipt_path = tmp_path / "missing.json"
    if receipt_payload is not None:
        receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")
    qrels_path = tmp_path / "qrels"
    original_sha256 = __import__(
        "trec_mlm.profile_multiresolution_evaluation", fromlist=["_sha256"]
    )._sha256

    def guarded_sha256(path):
        if Path(path) == qrels_path:
            raise AssertionError("qrels were hashed before receipt authorization")
        return original_sha256(path)

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation._sha256", guarded_sha256
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_complete_qrels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("qrels were opened before receipt authorization")
        ),
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        evaluate_multiresolution(
            root, receipt_path, qrels_path, tmp_path / "evaluation.json"
        )


def test_qrel_inventory_validation_requires_exact_queries_and_candidates() -> None:
    qrels = {"q1": {"A": 1.0, "B": 0.0}, "q2": {"A": 0.0, "B": 1.0}}
    validate_multiresolution_qrel_inventories(
        qrels, query_ids=("q1", "q2"), candidate_ids=("A", "B")
    )
    with pytest.raises(ValueError, match="query inventory"):
        validate_multiresolution_qrel_inventories(
            {"q1": qrels["q1"]}, query_ids=("q1", "q2"), candidate_ids=("A", "B")
        )
    with pytest.raises(ValueError, match="candidate inventory"):
        validate_multiresolution_qrel_inventories(
            {**qrels, "q2": {"A": 0.0}},
            query_ids=("q1", "q2"),
            candidate_ids=("A", "B"),
        )


def test_evaluation_rejects_qrel_file_change_while_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)

    def changing_qrels(path, **_kwargs):
        Path(path).write_text("q1 0 A 1\n", encoding="utf-8")
        return {
            "q1": {"A": 0.0, "B": 1.0},
            "q2": {"A": 1.0, "B": 0.0},
        }

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_complete_qrels",
        changing_qrels,
    )
    with pytest.raises(RuntimeError, match="qrel file changed"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            tmp_path / "qrels",
            tmp_path / "output.json",
        )


def test_evaluation_reports_every_frozen_system_paired_evidence_and_track_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    output = tmp_path / "evaluation.json"

    payload = evaluate_multiresolution(
        root, tmp_path / "viability.json", tmp_path / "qrels", output
    )

    assert set(payload["systems"]) == set(EXPECTED_SYSTEMS)
    assert set(payload["paired_comparisons_vs_query_knn"]) == set(EXPECTED_SYSTEMS)
    for comparison in payload["paired_comparisons_vs_query_knn"].values():
        assert comparison["ndcg_at_10_delta"]["n_resamples"] == 10_000
        assert comparison["mrr_delta"]["confidence_level"] == 0.95
        assert sum(comparison["ndcg_at_10_win_tie_loss"][key] for key in ("wins", "ties", "losses")) == 2
        assert comparison["zero_transitions"]["query_count"] == 2
    assert payload["track_a"] == {
        **payload["track_a"],
        "diagnostic_only": True,
        "independent_validation": False,
        "source_query_set_previously_seen": True,
        "dense_weight": 1.0,
        "profile_weight": 2.0,
        "rrf_k": 60,
        "reconstruction_matches_issue12_frozen": True,
    }
    assert payload["controls"]["query_knn"]["query_count"] == 2
    assert payload["controls"]["contextual_e5"]["query_count"] == 2
    assert payload["qrels_sha256"] == hashlib.sha256(
        (tmp_path / "qrels").read_bytes()
    ).hexdigest()
    assert payload["viability_receipt_sha256"] == hashlib.sha256(
        (tmp_path / "viability.json").read_bytes()
    ).hexdigest()
    assert json.loads(output.read_text()) == payload


def test_evaluation_detects_any_construction_change_and_keeps_output_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)

    def original_loader(_):
        return multi

    calls = 0

    def mutate_after_load(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "manifest.json").write_text("changed\n")
        return multi

    _patch_loaders(monkeypatch, multi, source)
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_multiresolution_artifact",
        mutate_after_load,
    )
    with pytest.raises(RuntimeError, match="construction changed"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            tmp_path / "qrels",
            tmp_path / "out.json",
        )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_multiresolution_artifact",
        original_loader,
    )
    with pytest.raises(ValueError, match="outside construction"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            tmp_path / "qrels",
            root / "evaluation.json",
        )


def test_viability_receipt_must_authorize_and_bind_evaluation() -> None:
    receipt = _valid_viability_receipt()
    assert validate_viability_receipt(receipt, "a" * 64) == receipt
    with pytest.raises(ValueError, match="authorize"):
        validate_viability_receipt({**receipt, "decision": "reject-invalid"}, "a" * 64)
    with pytest.raises(ValueError, match="bind"):
        validate_viability_receipt(receipt, "b" * 64)
    with pytest.raises(ValueError, match="schema"):
        validate_viability_receipt(
            {
                "artifact_type": "issue18_multiresolution_viability_receipt",
                "version": 1,
                "qrel_free": True,
                "artifact_aggregate_sha256": "a" * 64,
                "decision": "continue",
            },
            "a" * 64,
        )
    changed = json.loads(json.dumps(receipt))
    changed["thresholds"]["largest_cluster_share"]["passed"] = False
    with pytest.raises(ValueError, match="inconsistent"):
        validate_viability_receipt(changed, "a" * 64)


def test_cli_has_only_sealed_paths_and_no_tuning_options() -> None:
    from scripts.evaluate_multiresolution_profiles import _parser

    parser = _parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {
        "help",
        "artifact",
        "qrels",
        "viability_receipt",
        "output",
    }
    assert not any(
        name in destinations for name in ("weight", "rrf_k", "top_count", "resolution", "seed")
    )
