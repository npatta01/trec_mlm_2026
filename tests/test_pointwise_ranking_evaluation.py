from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import trec_mlm.pointwise_ranking_evaluation as evaluation_module
from trec_mlm.metrics import evaluate_rankings
from trec_mlm.pointwise_ranking import (
    PairSelectionReceipt,
    PointwiseEvidence,
    PointwisePredictions,
)
from trec_mlm.pointwise_ranking_evaluation import (
    BASELINE_TOLERANCE,
    _evaluate_frozen_rankings,
    comparison_receipt,
    evaluate_frozen_rankings,
    phase0_decision,
    reproduce_dense_control,
    run_pointwise_phase0_evaluation,
    validate_qrel_matrix,
    write_evaluation_outputs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predictions() -> PointwisePredictions:
    query_ids = ("q1", "q2", "q3", "q4", "q5")
    candidate_ids = ("A", "B", "C", "D")
    prior = {query_id: ["A", "B", "C", "D"] for query_id in query_ids}
    routing = {
        "q1": ["B", "A", "D", "C"],
        "q2": ["C", "D", "B", "A"],
        "q3": ["D", "C", "B", "A"],
        "q4": ["A", "C", "B", "D"],
        "q5": ["B", "D", "A", "C"],
    }
    pointwise = {
        "q1": ["A", "B", "D", "C"],
        "q2": ["C", "A", "B", "D"],
        "q3": ["A", "D", "B", "C"],
        "q4": ["A", "B", "C", "D"],
        "q5": ["B", "A", "D", "C"],
    }
    return PointwisePredictions(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        candidate_priors=np.asarray([0.8, 0.6, 0.4, 0.2]),
        local_scores=np.zeros((5, 4), dtype=np.float64),
        residual_scores=np.zeros((5, 4), dtype=np.float64),
        pointwise_rrf_scores=np.zeros((5, 4), dtype=np.float64),
        prior_rankings=prior,
        routing_rankings=routing,
        pointwise_rankings=pointwise,
        query_similarity_means={
            "q1": 0.1,
            "q2": 0.2,
            "q3": 0.3,
            "q4": 0.4,
            "q5": 0.5,
        },
        query_similarity_quartiles={"q1": 1, "q2": 1, "q3": 2, "q4": 3, "q5": 4},
        candidate_audited_quartiles={"A": 1, "B": 1, "C": 2, "D": 3},
        candidate_prior_quartiles={"A": 4, "B": 3, "C": 2, "D": 1},
    )


def _evidence() -> PointwiseEvidence:
    return PointwiseEvidence(
        labels_sha256="1" * 64,
        records_sha256="2" * 64,
        labels_by_record_id={},
        record_ids=(),
        question_ids=(),
        source_question_fingerprints={},
        candidate_ids=("A", "B", "C", "D"),
        audited_counts={"A": 1, "B": 2, "C": 3, "D": 4},
        supported_counts={"A": 1, "B": 1, "C": 1, "D": 1},
        audited_question_coverage={"A": 1, "B": 2, "C": 3, "D": 4},
        candidate_priors={"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2},
    )


def _qrels() -> dict[str, dict[str, float]]:
    return {
        "q1": {"A": 3, "B": 1, "C": 0, "D": 0},
        "q2": {"A": 0, "B": 0, "C": 2, "D": 0},
        "q3": {"A": 0, "B": 0, "C": 0, "D": 1},
        "q4": {"A": 1, "B": 1, "C": 1, "D": 1},
        "q5": {"A": 0, "B": 0, "C": 0, "D": 0},
    }


def _selector(*, selected: int = 500, questions: int = 100) -> PairSelectionReceipt:
    return PairSelectionReceipt(
        selector_version="pointwise-phase0-pair-selector-v1",
        target_pair_count=500,
        eligible_pair_count=700,
        eligible_source_question_count=300,
        selected_pair_count=selected,
        source_question_count=questions,
        initial_candidate_use_cap=20,
        final_candidate_use_cap=20,
        candidate_use_counts={"A": 20},
        pairs=(),
    )


def test_reproduce_dense_control_checks_aggregate_and_each_query(tmp_path: Path) -> None:
    rankings = {"q1": ["A", "B", "C"], "q2": ["B", "C", "A"]}
    qrels = {
        "q1": {"A": 2, "B": 1, "C": 0},
        "q2": {"A": 0, "B": 0, "C": 1},
    }
    evaluated = evaluate_rankings(rankings, qrels, k=10)
    result_path = tmp_path / "dense.json"
    result_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "dense_nested": {
                        "ndcg_at_10": evaluated.macro_ndcg_at_k,
                        "mrr": evaluated.mean_reciprocal_rank,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    per_query_path = tmp_path / "dense.tsv"
    per_query_path.write_text(
        "query_id\tdense_ndcg_at_10\tdense_reciprocal_rank\n"
        + "".join(
            f"{query_id}\t{metric.ndcg_at_k:.13g}\t{metric.reciprocal_rank:.13g}\n"
            for query_id, metric in evaluated.per_query.items()
        ),
        encoding="utf-8",
    )

    receipt = reproduce_dense_control(
        rankings,
        qrels,
        result_receipt_path=result_path,
        per_query_receipt_path=per_query_path,
        expected_result_sha256=_sha256(result_path),
        expected_per_query_sha256=_sha256(per_query_path),
        expected_ndcg=evaluated.macro_ndcg_at_k,
        expected_mrr=evaluated.mean_reciprocal_rank,
        expected_zero_ndcg=0,
    )

    assert receipt["status"] == "passed"
    assert receipt["per_query_values_checked"] == 2
    assert receipt["absolute_tolerance"] == BASELINE_TOLERANCE
    changed = per_query_path.read_text().replace("0.6309297535715", "0.6309297535815")
    per_query_path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="per-query.*differs"):
        reproduce_dense_control(
            rankings,
            qrels,
            result_receipt_path=result_path,
            per_query_receipt_path=per_query_path,
            expected_result_sha256=_sha256(result_path),
            expected_per_query_sha256=_sha256(per_query_path),
            expected_ndcg=evaluated.macro_ndcg_at_k,
            expected_mrr=evaluated.mean_reciprocal_rank,
            expected_zero_ndcg=0,
        )


def test_comparison_uses_query_grain_and_reports_all_zero_transitions() -> None:
    baseline = evaluate_rankings(
        {"q1": [], "q2": [], "q3": [], "q4": ["A", "B"]},
        {
            "q1": {"A": 2, "B": 0},
            "q2": {"A": 0, "B": 1},
            "q3": {"A": 0, "B": 0},
            "q4": {"A": 1, "B": 0},
        },
    )
    candidate = evaluate_rankings(
        {"q1": ["A", "B"], "q2": ["B", "A"], "q3": ["B", "A"], "q4": []},
        {
            "q1": {"A": 2, "B": 0},
            "q2": {"A": 0, "B": 1},
            "q3": {"A": 0, "B": 0},
            "q4": {"A": 1, "B": 0},
        },
    )

    result = comparison_receipt(baseline, candidate, n_resamples=100)

    assert result["ndcg_at_10_delta"]["n_resamples"] == 100
    assert result["ndcg_at_10_delta"]["seed"] == 31
    assert result["ndcg_at_10_win_tie_loss"] == {
        "wins": 2,
        "ties": 1,
        "losses": 1,
        "tolerance": 1e-12,
    }
    assert result["zero_transitions"] == {
        "zero_both": 1,
        "dense_zero_candidate_nonzero": 2,
        "dense_nonzero_candidate_zero": 1,
        "nonzero_both": 0,
        "query_count": 4,
    }


def test_evaluate_emits_frozen_rrf_query_slices_and_empty_candidate_bins() -> None:
    predictions = _predictions()
    dense = {
        "q1": ["B", "A", "C", "D"],
        "q2": ["A", "B", "C", "D"],
        "q3": ["A", "B", "C", "D"],
        "q4": ["D", "C", "B", "A"],
        "q5": ["A", "B", "C", "D"],
    }

    result = _evaluate_frozen_rankings(
        _evidence(),
        predictions,
        dense,
        _qrels(),
        _selector(),
        n_resamples=100,
    )

    # q1 pointwise and dense invert their first two ranks. Equal RRF ties and
    # therefore falls back to dense rank before pointwise rank.
    assert result.rankings["combined"]["q1"][:2] == ["B", "A"]
    assert [row["label"] for row in result.query_slices if row["family"] == "qrel_positive_density"] == [
        "le_5",
        "6_20",
        "gt_20",
    ]
    density_counts = {
        row["label"]: row["query_count"]
        for row in result.query_slices
        if row["family"] == "qrel_positive_density"
    }
    assert density_counts == {"le_5": 5, "6_20": 0, "gt_20": 0}
    assert {
        row["label"]: row["query_count"]
        for row in result.query_slices
        if row["family"] == "query_similarity_quartile"
    } == {"1": 2, "2": 1, "3": 1, "4": 1}

    audited_rows = [
        row for row in result.candidate_slices if row["family"] == "audited_count_quartile"
    ]
    assert [row["label"] for row in audited_rows] == ["1", "2", "3", "4"]
    assert audited_rows[-1]["candidate_count"] == 0
    assert audited_rows[-1]["candidate_ids"] == []
    assert all(value is None for value in audited_rows[-1]["metrics"].values())
    # Candidate filtering recomputes the metric rather than selecting rows from
    # the all-candidate evaluation.
    first_bin = audited_rows[0]
    assert first_bin["candidate_ids"] == ["A", "B"]
    filtered = {query_id: {item: grades[item] for item in ("A", "B")} for query_id, grades in _qrels().items()}
    expected = evaluate_rankings(
        {query_id: [item for item in dense[query_id] if item in {"A", "B"}] for query_id in dense},
        filtered,
    )
    assert first_bin["metrics"]["control"]["ndcg_at_10"] == pytest.approx(
        expected.macro_ndcg_at_k
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "pass"),
        ({"combined_ndcg_delta": 0.0}, "fail"),
        ({"combined_ndcg_lower": 0.0}, "fail"),
        ({"combined_mrr_delta": -0.0050000001}, "fail"),
        ({"combined_zero_ndcg": 116}, "fail"),
        ({"routing_ndcg": 0.2, "prior_ndcg": 0.2}, "fail"),
    ],
)
def test_phase0_decision_requires_every_exact_gate_condition(
    changes: dict[str, float | int], expected: str
) -> None:
    values: dict[str, float | int] = {
        "combined_ndcg_delta": 0.01,
        "combined_ndcg_lower": 0.001,
        "combined_mrr_delta": -0.005,
        "combined_zero_ndcg": 115,
        "routing_ndcg": 0.21,
        "prior_ndcg": 0.20,
    }
    values.update(changes)

    receipt = phase0_decision(
        **values,
        selector_receipt=_selector(),
    )

    assert receipt["decision"] == expected
    assert receipt["selector_availability"]["available"] is True
    assert receipt["recommendation"] == (
        "recommend_500_pair_pilot" if expected == "pass" else "stop_pairwise_work"
    )


def test_selector_availability_is_separate_from_effectiveness_gate() -> None:
    receipt = phase0_decision(
        combined_ndcg_delta=0.01,
        combined_ndcg_lower=0.001,
        combined_mrr_delta=0.0,
        combined_zero_ndcg=115,
        routing_ndcg=0.3,
        prior_ndcg=0.2,
        selector_receipt=_selector(selected=499, questions=99),
    )

    assert receipt["decision"] == "pass"
    assert receipt["selector_availability"]["available"] is False
    assert receipt["recommendation"] == "stop_pairwise_work"


def test_writer_emits_atomic_complete_bundle_and_byte_identical_exports(
    tmp_path: Path,
) -> None:
    predictions = _predictions()
    evidence = _evidence()
    dense = {
        query_id: ["A", "B", "C", "D"] for query_id in predictions.query_ids
    }
    evaluation = _evaluate_frozen_rankings(
        evidence,
        predictions,
        dense,
        _qrels(),
        _selector(),
        n_resamples=100,
    )
    input_receipt = {
        "qrels_sha256": "3" * 64,
        "dense_rankings_sha256": "4" * 64,
    }
    baseline_receipt = {"status": "passed", "per_query_values_checked": 5}
    first_artifact = tmp_path / "artifact-1"
    second_artifact = tmp_path / "artifact-2"

    first = write_evaluation_outputs(
        evaluation,
        evidence,
        predictions,
        _selector(),
        artifact_dir=first_artifact,
        input_receipt=input_receipt,
        baseline_reproduction=baseline_receipt,
    )
    second = write_evaluation_outputs(
        evaluation,
        evidence,
        predictions,
        _selector(),
        artifact_dir=second_artifact,
        input_receipt=input_receipt,
        baseline_reproduction=baseline_receipt,
    )

    assert first["export_sha256"] == second["export_sha256"]
    assert sorted(path.name for path in (first_artifact / "runs").iterdir()) == [
        "combined.trec",
        "control.trec",
        "pointwise_only.trec",
        "prior_only.trec",
        "routing_only.trec",
    ]
    for path in (first_artifact / "runs").iterdir():
        assert len(path.read_text().splitlines()) == 20
    assert (first_artifact / "manifest.json").is_file()
    assert not any(path.name.startswith(".") for path in tmp_path.rglob("*"))
    sanitized_text = "\n".join(
        (first_artifact / filename).read_text()
        for filename in ("results.json", "per_query.tsv", "candidate_slices.tsv")
    )
    assert "/home/" not in sanitized_text
    assert "raw private question" not in sanitized_text

    result_payload = json.loads((first_artifact / "results.json").read_text())
    assert evaluation.bootstrap_resamples == 100
    assert evaluation.bootstrap_seed == 31
    assert result_payload["protocol"]["bootstrap_resamples"] == 100
    assert result_payload["protocol"]["bootstrap_seed"] == 31
    export_hashes = first["export_sha256"].copy()
    with pytest.raises(FileExistsError, match="already exists"):
        write_evaluation_outputs(
            evaluation,
            evidence,
            predictions,
            _selector(),
            artifact_dir=first_artifact,
            input_receipt=input_receipt,
            baseline_reproduction=baseline_receipt,
        )
    assert {
        name: _sha256(first_artifact / filename)
        for name, filename in (
            ("results", "results.json"),
            ("per_query", "per_query.tsv"),
            ("candidate_slices", "candidate_slices.tsv"),
        )
    } == export_hashes


def test_writer_rejects_bootstrap_metadata_that_disagrees_with_computation(
    tmp_path: Path,
) -> None:
    predictions = _predictions()
    evidence = _evidence()
    dense = {
        query_id: ["A", "B", "C", "D"] for query_id in predictions.query_ids
    }
    evaluation = _evaluate_frozen_rankings(
        evidence,
        predictions,
        dense,
        _qrels(),
        _selector(),
        n_resamples=100,
    )
    comparisons = deepcopy(evaluation.comparisons)
    comparisons["combined"]["ndcg_at_10_delta"]["n_resamples"] = 10_000
    inconsistent = replace(evaluation, comparisons=comparisons)

    with pytest.raises(ValueError, match="bootstrap metadata"):
        write_evaluation_outputs(
            inconsistent,
            evidence,
            predictions,
            _selector(),
            artifact_dir=tmp_path / "artifact",
            input_receipt={},
            baseline_reproduction={"status": "passed"},
        )


def test_public_production_entrypoint_has_no_frozen_integrity_overrides() -> None:
    parameters = inspect.signature(run_pointwise_phase0_evaluation).parameters
    forbidden = {
        "expected_qrels_sha256",
        "expected_dense_rankings_sha256",
        "expected_dense_run_sha256",
        "expected_dense_results_sha256",
        "expected_dense_per_query_sha256",
        "expected_query_count",
        "expected_candidate_count",
        "bootstrap_seed",
        "bootstrap_resamples",
        "n_resamples",
        "expected_ndcg",
        "expected_mrr",
        "expected_zero_ndcg",
        "baseline_tolerance",
        "result_export_path",
        "per_query_export_path",
        "candidate_slice_export_path",
    }

    assert forbidden.isdisjoint(parameters)
    assert forbidden.isdisjoint(inspect.signature(evaluate_frozen_rankings).parameters)
    assert forbidden.isdisjoint(inspect.signature(write_evaluation_outputs).parameters)


def test_cli_exposes_only_the_single_artifact_publication_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_pointwise_ranking_phase0.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--artifact-dir" in result.stdout
    assert "--result-export" not in result.stdout
    assert "--per-query-export" not in result.stdout
    assert "--candidate-slice-export" not in result.stdout


def test_failure_before_single_rename_leaves_no_publish_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = _predictions()
    evidence = _evidence()
    dense = {
        query_id: ["A", "B", "C", "D"] for query_id in predictions.query_ids
    }
    evaluation = _evaluate_frozen_rankings(
        evidence,
        predictions,
        dense,
        _qrels(),
        _selector(),
        n_resamples=100,
    )
    artifact = tmp_path / "artifact"
    exports = [
        tmp_path / "tracked-results.json",
        tmp_path / "tracked-per-query.tsv",
        tmp_path / "tracked-candidate.tsv",
    ]
    old_payloads = {}
    for index, path in enumerate(exports):
        payload = f"old-{index}\n".encode()
        path.write_bytes(payload)
        old_payloads[path] = payload
    original_rename = evaluation_module.os.rename
    calls = 0

    def fail_before_publish(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected pre-rename failure")
        original_rename(source, target)

    monkeypatch.setattr(evaluation_module.os, "rename", fail_before_publish)

    with pytest.raises(OSError, match="pre-rename"):
        write_evaluation_outputs(
            evaluation,
            evidence,
            predictions,
            _selector(),
            artifact_dir=artifact,
            input_receipt={},
            baseline_reproduction={"status": "passed"},
        )

    assert not artifact.exists()
    assert {path: path.read_bytes() for path in exports} == old_payloads
    assert not any(path.name.startswith(".") for path in tmp_path.rglob("*"))

    receipt = write_evaluation_outputs(
        evaluation,
        evidence,
        predictions,
        _selector(),
        artifact_dir=artifact,
        input_receipt={},
        baseline_reproduction={"status": "passed"},
    )
    assert artifact.is_dir()
    assert calls == 2
    assert {path: path.read_bytes() for path in exports} == old_payloads
    assert set(receipt["export_sha256"]) == {"results", "per_query", "candidate_slices"}


def test_single_directory_publish_fsyncs_staged_tree_then_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = _predictions()
    evidence = _evidence()
    evaluation = _evaluate_frozen_rankings(
        evidence,
        predictions,
        {query_id: ["A", "B", "C", "D"] for query_id in predictions.query_ids},
        _qrels(),
        _selector(),
        n_resamples=100,
    )
    artifact = tmp_path / "artifact"
    fsynced_directories: list[Path] = []
    original_fsync_directory = evaluation_module._fsync_directory

    def record_fsync(path: Path) -> None:
        fsynced_directories.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(evaluation_module, "_fsync_directory", record_fsync)

    write_evaluation_outputs(
        evaluation,
        evidence,
        predictions,
        _selector(),
        artifact_dir=artifact,
        input_receipt={},
        baseline_reproduction={"status": "passed"},
    )

    assert artifact.is_dir()
    assert fsynced_directories[-1] == artifact.parent
    assert len(fsynced_directories) >= 5


def test_qrel_matrix_requires_exact_dimensions_and_identities() -> None:
    validate_qrel_matrix(
        {"q1": {"A": 1, "B": 0}, "q2": {"A": 0, "B": 1}},
        query_ids=("q1", "q2"),
        candidate_ids=("A", "B"),
        expected_query_count=2,
        expected_candidate_count=2,
    )

    with pytest.raises(ValueError, match="expected 3 queries"):
        validate_qrel_matrix(
            {"q1": {"A": 1, "B": 0}, "q2": {"A": 0, "B": 1}},
            query_ids=("q1", "q2"),
            candidate_ids=("A", "B"),
            expected_query_count=3,
            expected_candidate_count=2,
        )
    with pytest.raises(ValueError, match="candidate inventory"):
        validate_qrel_matrix(
            {"q1": {"A": 1}, "q2": {"A": 0, "B": 1}},
            query_ids=("q1", "q2"),
            candidate_ids=("A", "B"),
            expected_query_count=2,
            expected_candidate_count=2,
        )
