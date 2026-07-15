from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_mlm.profile_multiresolution import EXPECTED_SYSTEMS, MultiResolutionConfig
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
        evaluate_multiresolution(tmp_path / "missing", tmp_path / "qrels", tmp_path / "out.json")
    assert opened is False


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
        evaluate_multiresolution(root, tmp_path / "qrels", tmp_path / "output.json")


def test_evaluation_reports_every_frozen_system_paired_evidence_and_track_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, multi, source = _construction(tmp_path)
    _patch_loaders(monkeypatch, multi, source)
    output = tmp_path / "evaluation.json"

    payload = evaluate_multiresolution(root, tmp_path / "qrels", output)

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
        evaluate_multiresolution(root, tmp_path / "qrels", tmp_path / "out.json")
    monkeypatch.setattr(
        "trec_mlm.profile_multiresolution_evaluation.load_multiresolution_artifact",
        original_loader,
    )
    with pytest.raises(ValueError, match="outside construction"):
        evaluate_multiresolution(root, tmp_path / "qrels", root / "evaluation.json")


def test_viability_receipt_must_authorize_and_bind_evaluation() -> None:
    receipt = {
        "artifact_type": "issue18_multiresolution_viability_receipt",
        "version": 1,
        "qrel_free": True,
        "artifact_aggregate_sha256": "a" * 64,
        "decision": "continue",
        "integrity_checks": {"source_binding": True, "complete_rankings": True},
        "failed_integrity_thresholds": [],
    }
    assert validate_viability_receipt(receipt, "a" * 64) == receipt
    with pytest.raises(ValueError, match="authorize"):
        validate_viability_receipt({**receipt, "decision": "reject-invalid"}, "a" * 64)
    with pytest.raises(ValueError, match="bind"):
        validate_viability_receipt(receipt, "b" * 64)


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
