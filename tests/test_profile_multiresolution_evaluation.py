from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
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


def _valid_viability_receipt(*, decision: str = "continue") -> dict:
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
    failed_query_sensitivity = (
        ["maximum_top1_candidate_share"]
        if decision == "stop-query-sensitivity"
        else []
    )
    failed_incremental = (
        ["incremental_mean_top10_jaccard"]
        if decision == "stop-no-incremental-value"
        else []
    )
    failed_integrity = ["finite_scores"] if decision == "reject-invalid" else []
    receipt = {
        "artifact_type": "issue18_multiresolution_viability_receipt",
        "version": 1,
        "qrel_free": True,
        "artifact_aggregate_sha256": "a" * 64,
        "decision": decision,
        "thresholds": thresholds,
        "integrity_checks": integrity,
        "failed_thresholds": [
            *failed_integrity,
            *failed_query_sensitivity,
            *failed_incremental,
        ],
        "failed_integrity_thresholds": failed_integrity,
        "failed_query_sensitivity_thresholds": failed_query_sensitivity,
        "failed_incremental_thresholds": failed_incremental,
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
        "control_sources": {
            name: {
                "ranking_name": ranking_name,
                "ranking_file": f"rankings/{ranking_name}.json",
                "ranking_sha256": character * 64,
                "source_aggregate_sha256": MultiResolutionConfig().source_aggregate_sha256,
                "query_count": 2,
                "candidate_count": 2,
                "query_inventory_sha256": "d" * 64,
                "candidate_inventory_sha256": "e" * 64,
            }
            for name, ranking_name, character in (
                ("query_knn", "dense_reference", "b"),
                ("contextual_e5", "contextual_e5_control", "c"),
            )
        },
        "control_comparisons": {
            system: {
                control: {
                    "mean_top10_jaccard": 1.0,
                    "top10_difference_query_share": 0.0,
                    "mean_top10_overlap_count": 2.0,
                    "mean_common_top10_absolute_rank_displacement": 0.0,
                }
                for control in ("query_knn", "contextual_e5")
            }
            for system in EXPECTED_SYSTEMS
        },
        "execution_provenance": {
            "execution_code_commit": "f" * 40,
            "command": "locked synthetic construction",
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "dependency_versions": {
                "python": "3.13.9",
                "numpy": "2.5.1",
                "scipy": "1.18.0",
                "scikit-learn": "1.9.0",
            },
            "construction_completed_at": "2026-07-15T12:41:18.692102441-04:00",
            "artifact_aggregate_sha256": "a" * 64,
            "receipt_generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "repeat_authorized": decision == "continue",
    }
    return receipt


def _construction(
    tmp_path: Path, *, decision: str = "continue"
) -> tuple[Path, SimpleNamespace, SimpleNamespace]:
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
        json.dumps(_valid_viability_receipt(decision=decision), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, multi, source


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    multi,
    source,
    *,
    authoritative_receipt: dict | None = None,
) -> None:
    canonical = copy.deepcopy(
        authoritative_receipt
        if authoritative_receipt is not None
        else json.loads((source.root.parent / "viability.json").read_text())
    )
    multi.source_snapshot = source
    if not hasattr(multi, "artifact_token"):
        multi.artifact_token = SimpleNamespace(
            aggregate_sha256="a" * 64,
            entries=(),
        )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_issue18_frozen_snapshot",
        lambda _: multi,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_issue12_source_snapshot",
        lambda _: None,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_frozen_tree_token",
        lambda *_args, **_kwargs: None,
    )
    legacy_token = SimpleNamespace(path=Path("legacy.json"), sha256="d" * 64)
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_legacy_viability_evidence",
        lambda: legacy_token,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_legacy_viability_evidence",
        lambda _token: None,
    )

    def validate_receipt(payload, _artifact, **_kwargs):
        if payload != canonical:
            raise ValueError("viability receipt differs from authoritative artifact evidence")
        return payload

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_viability_receipt",
        validate_receipt,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation._validate_sealed_paths",
        lambda *_: None,
    )
    qrels_path = source.root.parent / "qrels"
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.LOCKED_QRELS_SHA256",
        hashlib.sha256(qrels_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_complete_qrels",
        lambda _, **__: {
            "q1": {"A": 0.0, "B": 1.0},
            "q2": {"A": 1.0, "B": 0.0},
        },
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution._manifest_completion_timestamp",
        lambda _root: "2026-07-15T12:41:18.692102441-04:00",
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution.assess_multiresolution_viability",
        lambda _root, *, receipt_generated_at=None: copy.deepcopy(canonical),
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


def test_artifact_mutation_after_loader_return_is_rejected_before_qrels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trec_mlm import profile_multiresolution as multiresolution
    from trec_mlm import profile_multiresolution_evaluation as module

    root, multi, source = _construction(tmp_path)
    empty_aggregate = multiresolution._aggregate_hash({})
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "issue18_multiresolution_profile_manifest",
                "version": 1,
                "qrel_free": True,
                "files": {},
                "aggregate_sha256": empty_aggregate,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    validated_manifest = json.loads((root / "manifest.json").read_text())
    multi.artifact_token = multiresolution._capture_frozen_tree_token(
        root,
        expected_aggregate_sha256=empty_aggregate,
        validated_manifest=validated_manifest,
        validated_entries={},
    )
    original_validate_token = multiresolution.validate_frozen_tree_token
    _patch_loaders(monkeypatch, multi, source)
    monkeypatch.setattr(module, "ISSUE18_FROZEN_AGGREGATE_SHA256", empty_aggregate)
    monkeypatch.setattr(module, "validate_frozen_tree_token", original_validate_token)

    def load_then_mutate(_path):
        (root / "manifest.json").write_text("changed after validation\n")
        return multi

    monkeypatch.setattr(module, "load_issue18_frozen_snapshot", load_then_mutate)
    qrels_path = tmp_path / "qrels"
    original_sha256 = module._sha256

    def guard_qrels(path):
        if Path(path) == qrels_path:
            raise AssertionError("mutated artifact reached qrel hashing")
        return original_sha256(path)

    monkeypatch.setattr(module, "_sha256", guard_qrels)
    with pytest.raises(ValueError, match="tree token|manifest token"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            qrels_path,
            tmp_path / "evaluation.json",
        )


@pytest.mark.skipif(
    not Path("artifacts/multiresolution_profiles/issue18_pr17head/manifest.json").is_file(),
    reason=(
        "requires the ignored local issue-18 scientific artifact; "
        "see reports/trec_2026_issue18_pr17head_artifact_custody.json"
    ),
)
def test_real_evaluation_pre_qrel_path_avoids_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import heapq

    from trec_mlm import profile_experiment
    from trec_mlm import profile_multiresolution as multiresolution
    from trec_mlm import profile_multiresolution_evaluation as module
    from trec_mlm import profile_retrieval

    artifact = Path("artifacts/multiresolution_profiles/issue18_pr17head")
    receipt = Path("reports/trec_2026_issue18_pr17head_viability.json")
    legacy_receipt = Path(
        "reports/trec_2026_issue18_pr17head_viability_legacy.json"
    )
    qrels = Path("data/dev/llm_dev_qrels.txt")
    output = tmp_path / "evaluation.json"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("evaluation invoked reconstruction")

    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(legacy_receipt.read_bytes())
    monkeypatch.setattr(
        multiresolution,
        "LEGACY_VIABILITY_RECEIPT_PATH",
        legacy,
    )
    monkeypatch.setattr(heapq, "nsmallest", forbidden)
    monkeypatch.setattr(
        multiresolution, "_top10_score_rows", forbidden, raising=False
    )

    for owner, names in (
        (
            multiresolution,
            (
                "construct_multiresolution",
                "canonical_kmeans",
                "score_profiles",
                "_freeze_systems",
                "reciprocal_rank_fusion",
                "load_profile_artifacts",
                "_recompute_profile_centroid",
                "_rank_scores",
            ),
        ),
        (
            profile_experiment,
            ("load_profile_artifacts", "score_profiles", "_recompute_profile_centroid"),
        ),
        (profile_retrieval, ("canonical_kmeans", "score_profiles", "reciprocal_rank_fusion")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden, raising=False)
    monkeypatch.setattr(
        module, "_load_issue12_evaluation_rankings", forbidden, raising=False
    )
    monkeypatch.setattr(module, "CANONICAL_EVALUATION_OUTPUT_PATH", output)

    authoritative = multiresolution.assess_multiresolution_viability(artifact)
    receipt_bytes = (
        json.dumps(authoritative, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    original_read_bytes = Path.read_bytes

    def read_authoritative(path: Path) -> bytes:
        if path.resolve() == receipt.resolve():
            return receipt_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_authoritative)

    original_sha256 = module._sha256

    def stop_at_qrels(path: Path) -> str:
        if Path(path).resolve() == qrels.resolve():
            raise RuntimeError("authorized qrel boundary reached")
        if Path(path).resolve() == receipt.resolve():
            return hashlib.sha256(receipt_bytes).hexdigest()
        return original_sha256(path)

    monkeypatch.setattr(module, "_sha256", stop_at_qrels)
    monkeypatch.setattr(module, "load_complete_qrels", forbidden)

    with pytest.raises(RuntimeError, match="authorized qrel boundary reached"):
        evaluate_multiresolution(artifact, receipt, qrels, output)


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


@pytest.mark.parametrize(
    "decision", ("stop-query-sensitivity", "stop-no-incremental-value")
)
def test_valid_stop_receipt_authorizes_one_diagnostic_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    root, multi, source = _construction(tmp_path, decision=decision)
    receipt = json.loads((tmp_path / "viability.json").read_text())
    _patch_loaders(
        monkeypatch, multi, source, authoritative_receipt=receipt
    )

    payload = evaluate_multiresolution(
        root,
        tmp_path / "viability.json",
        tmp_path / "qrels",
        tmp_path / "evaluation.json",
    )

    assert payload["viability_decision"] == decision
    assert payload["diagnostic_only"] is True
    assert receipt["repeat_authorized"] is False


def test_tampered_legacy_evidence_cannot_touch_qrels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(
        tmp_path, decision="stop-no-incremental-value"
    )
    receipt = json.loads((tmp_path / "viability.json").read_text())
    _patch_loaders(monkeypatch, multi, source, authoritative_receipt=receipt)
    touched = False

    def reject_legacy(_token) -> None:
        raise ValueError("legacy viability evidence changed after validation")

    def touch_qrels(*_args, **_kwargs):
        nonlocal touched
        touched = True
        return {}

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_legacy_viability_evidence",
        reject_legacy,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_complete_qrels", touch_qrels
    )
    with pytest.raises(ValueError, match="legacy viability evidence"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            tmp_path / "qrels",
            tmp_path / "evaluation.json",
        )
    assert touched is False


def test_authoritative_reject_invalid_receipt_cannot_touch_qrels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path, decision="reject-invalid")
    receipt = json.loads((tmp_path / "viability.json").read_text())
    _patch_loaders(
        monkeypatch, multi, source, authoritative_receipt=receipt
    )
    qrels_path = tmp_path / "qrels"
    original_sha256 = __import__(
        "trec_mlm.profile_multiresolution_evaluation", fromlist=["_sha256"]
    )._sha256

    def guarded_sha256(path):
        if Path(path) == qrels_path:
            raise AssertionError("reject-invalid receipt touched qrels")
        return original_sha256(path)

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation._sha256", guarded_sha256
    )
    with pytest.raises(ValueError, match="authorize diagnostic evaluation"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            qrels_path,
            tmp_path / "evaluation.json",
        )


def test_receipt_bytes_are_rechecked_after_validation_before_qrel_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    receipt_path = tmp_path / "viability.json"
    qrels_path = tmp_path / "qrels"
    original_validator = validate_viability_receipt
    original_sha256 = __import__(
        "trec_mlm.profile_multiresolution_evaluation", fromlist=["_sha256"]
    )._sha256

    def validate_then_change(payload, artifact_path, **_kwargs):
        validated = original_validator(payload, artifact_path)
        receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
        return validated

    def guarded_sha256(path):
        if Path(path) == qrels_path:
            raise AssertionError("qrels were hashed before receipt-byte recheck")
        return original_sha256(path)

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_viability_receipt",
        validate_then_change,
    )
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation._sha256", guarded_sha256
    )
    with pytest.raises(RuntimeError, match="receipt changed before qrels"):
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


def test_track_a_uses_stored_issue12_ranking_without_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trec_mlm import profile_multiresolution_evaluation as module
    from trec_mlm import profile_retrieval

    _root, multi, source = _construction(tmp_path)
    monkeypatch.setattr(
        profile_retrieval,
        "reciprocal_rank_fusion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Track A ranking was reconstructed")
        ),
    )

    assert module._track_a_rankings(multi, source) is source.rankings[
        module.TRACK_A_SYSTEM
    ]


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


def test_evaluation_requires_locked_qrel_sha_after_receipt_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.LOCKED_QRELS_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="qrel SHA-256"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            tmp_path / "qrels",
            tmp_path / "output.json",
        )


def test_evaluation_rechecks_receipt_bytes_after_diagnostic_computation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    receipt_path = tmp_path / "viability.json"
    evaluation_module = __import__(
        "trec_mlm.profile_multiresolution_evaluation", fromlist=["evaluate_rankings"]
    )
    original_evaluate = evaluation_module.evaluate_rankings
    changed = False

    def evaluate_then_change(*args, **kwargs):
        nonlocal changed
        result = original_evaluate(*args, **kwargs)
        if not changed:
            receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
            changed = True
        return result

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.evaluate_rankings",
        evaluate_then_change,
    )
    output = tmp_path / "evaluation.json"
    with pytest.raises(RuntimeError, match="receipt changed during evaluation"):
        evaluate_multiresolution(
            root, receipt_path, tmp_path / "qrels", output
        )
    assert not output.exists()


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
        "stored_ranking_matches_issue12_frozen": True,
    }
    assert "reconstruction_matches_issue12_frozen" not in payload["track_a"]
    assert payload["controls"]["query_knn"]["query_count"] == 2
    assert payload["controls"]["contextual_e5"]["query_count"] == 2
    assert payload["qrels_sha256"] == hashlib.sha256(
        (tmp_path / "qrels").read_bytes()
    ).hexdigest()
    assert payload["viability_receipt_sha256"] == hashlib.sha256(
        (tmp_path / "viability.json").read_bytes()
    ).hexdigest()
    assert json.loads(output.read_text()) == payload


def test_evaluation_revalidates_one_source_snapshot_before_and_after_qrels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    validations: list[object] = []
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_issue12_source_snapshot",
        lambda snapshot: validations.append(snapshot),
    )

    evaluate_multiresolution(
        root,
        tmp_path / "viability.json",
        tmp_path / "qrels",
        tmp_path / "evaluation.json",
    )

    assert validations == [source, source]


def test_post_evaluation_source_token_failure_prevents_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    calls = 0

    def fail_second_validation(_snapshot):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("source token changed")

    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_issue12_source_snapshot",
        fail_second_validation,
    )
    output = tmp_path / "evaluation.json"

    with pytest.raises(RuntimeError, match="source changed during evaluation"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            tmp_path / "qrels",
            output,
        )
    assert not output.exists()


def test_evaluation_detects_any_construction_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)

    calls = 0

    def fail_second_validation(_token, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("artifact token changed")

    _patch_loaders(monkeypatch, multi, source)
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.validate_frozen_tree_token",
        fail_second_validation,
    )
    with pytest.raises(RuntimeError, match="construction changed"):
        evaluate_multiresolution(
            root,
            tmp_path / "viability.json",
            tmp_path / "qrels",
            tmp_path / "out.json",
        )


def test_viability_receipt_must_strictly_bind_persisted_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    receipt = json.loads((tmp_path / "viability.json").read_text())
    _patch_loaders(monkeypatch, multi, source, authoritative_receipt=receipt)
    assert validate_viability_receipt(receipt, root) == receipt
    with pytest.raises(ValueError, match="authoritative"):
        validate_viability_receipt(
            {**receipt, "artifact_aggregate_sha256": "b" * 64}, root
        )
    with pytest.raises(ValueError, match="authoritative"):
        validate_viability_receipt(
            {
                "artifact_type": "issue18_multiresolution_viability_receipt",
                "version": 1,
                "qrel_free": True,
                "artifact_aggregate_sha256": "a" * 64,
                "decision": "continue",
            },
            root,
        )
    changed = json.loads(json.dumps(receipt))
    changed["thresholds"]["largest_cluster_share"]["passed"] = False
    with pytest.raises(ValueError, match="authoritative"):
        validate_viability_receipt(changed, root)


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


def test_evaluation_rejects_every_alternate_sealed_identity_before_qrels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trec_mlm import profile_multiresolution_evaluation as module

    qrel_touched = False

    def forbidden_qrel_hash(_path: Path) -> str:
        nonlocal qrel_touched
        qrel_touched = True
        raise AssertionError("alternate paths reached qrel hashing")

    monkeypatch.setattr(module, "_sha256", forbidden_qrel_hash)
    canonical = (
        Path("artifacts/multiresolution_profiles/issue18_pr17head"),
        Path("reports/trec_2026_issue18_pr17head_viability.json"),
        Path("data/dev/llm_dev_qrels.txt"),
        Path("reports/trec_2026_issue18_pr17head_evaluation.json"),
    )
    alternates = (
        (tmp_path / "artifact", *canonical[1:]),
        (canonical[0], tmp_path / "receipt.json", *canonical[2:]),
        (*canonical[:2], tmp_path / "qrels", canonical[3]),
        (*canonical[:3], tmp_path / "evaluation.json"),
    )
    for paths in alternates:
        with pytest.raises(ValueError, match="canonical"):
            module._validate_sealed_paths(*paths)
    assert qrel_touched is False


def test_evaluation_refuses_existing_canonical_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trec_mlm import profile_multiresolution_evaluation as module

    paths = tuple(tmp_path / name for name in ("artifact", "receipt", "qrels", "output"))
    monkeypatch.setattr(module, "ISSUE18_FROZEN_ARTIFACT_PATH", str(paths[0]))
    monkeypatch.setattr(module, "CANONICAL_VIABILITY_RECEIPT_PATH", paths[1])
    monkeypatch.setattr(module, "CANONICAL_QRELS_PATH", paths[2])
    monkeypatch.setattr(module, "CANONICAL_EVALUATION_OUTPUT_PATH", paths[3])
    paths[3].write_text("sealed\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        module._validate_sealed_paths(*paths)


def test_evaluation_atomic_commit_preserves_file_created_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trec_mlm import profile_multiresolution_evaluation as module

    destination = tmp_path / "evaluation.json"
    attacker = b"attacker-owned\n"
    original_replace = module.os.replace
    original_link = module.os.link

    def race_replace(source, target):
        Path(target).write_bytes(attacker)
        return original_replace(source, target)

    def race_link(source, target):
        Path(target).write_bytes(attacker)
        return original_link(source, target)

    monkeypatch.setattr(module.os, "replace", race_replace)
    monkeypatch.setattr(module.os, "link", race_link)

    with pytest.raises(FileExistsError):
        module._write_evaluation({"status": "valid"}, destination)
    assert destination.read_bytes() == attacker
