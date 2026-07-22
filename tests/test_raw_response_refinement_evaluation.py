from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_mlm.raw_response_refinement_evaluation as evaluation
from trec_mlm.metrics import evaluate_rankings
from trec_mlm.raw_response_refinement_evaluation import (
    RawRefinementEvaluationConfig,
    evaluate_raw_response_refinement,
    primary_decision,
    validate_refinement_repeat_receipt,
)


def test_advance_requires_all_primary_gates() -> None:
    decision = primary_decision(
        ndcg_interval_lower=0.001,
        ndcg_point_delta=0.01,
        mrr_delta=-0.004,
        zero_count=98,
        zero_rescues=1,
        integrity_checks={"manifest": True},
    )
    assert decision["decision"] == "advance"


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"zero_count": 99}, "diagnostic"),
        ({"ndcg_interval_lower": 0.0}, "diagnostic"),
        ({"mrr_delta": -0.0050000001}, "diagnostic"),
        ({"mrr_delta": -0.005}, "advance"),
        ({"integrity_checks": {"manifest": False}}, "invalid"),
        (
            {
                "ndcg_interval_lower": -0.01,
                "ndcg_point_delta": 0.0,
                "zero_count": 99,
                "zero_rescues": 1,
            },
            "diagnostic",
        ),
        (
            {
                "ndcg_interval_lower": -0.01,
                "ndcg_point_delta": 0.0,
                "zero_count": 99,
                "zero_rescues": 0,
            },
            "stop",
        ),
    ],
)
def test_primary_decision_boundaries(changed: dict[str, object], expected: str) -> None:
    values: dict[str, object] = {
        "ndcg_interval_lower": 0.001,
        "ndcg_point_delta": 0.01,
        "mrr_delta": -0.004,
        "zero_count": 98,
        "zero_rescues": 1,
        "integrity_checks": {"manifest": True},
    }
    values.update(changed)
    assert primary_decision(**values)["decision"] == expected


def _repeat_receipt(aggregate: str) -> dict[str, object]:
    return {
        "status": "passed",
        "selected_record_ids_identical": True,
        "rankings_identical": True,
        "maximum_absolute_score_difference": 0.0,
        "tolerance": 1e-12,
        "first_aggregate_sha256": aggregate,
        "second_aggregate_sha256": "b" * 64,
        "first_execution_id": "b8e5992f-1d7b-43b5-974a-c35d419af023",
        "second_execution_id": "ad806ca9-5bca-4344-bbe4-271639ab255d",
    }


def test_repeat_receipt_is_exact_and_bound_to_evaluated_artifact() -> None:
    aggregate = "a" * 64
    receipt = _repeat_receipt(aggregate)
    execution_id = "b8e5992f-1d7b-43b5-974a-c35d419af023"
    assert (
        validate_refinement_repeat_receipt(receipt, aggregate, execution_id) == receipt
    )
    with pytest.raises(ValueError, match="does not bind"):
        validate_refinement_repeat_receipt(receipt, "c" * 64, execution_id)
    with pytest.raises(ValueError, match="execution does not bind"):
        validate_refinement_repeat_receipt(
            receipt, aggregate, "f9311443-2eb2-4d68-93a0-e953bc4857f5"
        )
    with pytest.raises(ValueError, match="schema"):
        validate_refinement_repeat_receipt(
            {**receipt, "fixture": True}, aggregate, execution_id
        )


def _artifact(root: Path) -> SimpleNamespace:
    query_ids = ("q1", "q2")
    candidate_ids = ("A", "B")
    profile_only = {"q1": ["A", "B"], "q2": ["B", "A"]}
    dense_profile_equal = {"q1": ["B", "A"], "q2": ["A", "B"]}
    return SimpleNamespace(
        root=root,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        rankings={
            "raw_hierarchical": {"q1": ["B", "A"], "q2": ["B", "A"]},
            "raw_hierarchical_dense_equal": {
                "q1": ["A", "B"],
                "q2": ["A", "B"],
            },
        },
        dense_rankings={"q1": ["A", "B"], "q2": ["B", "A"]},
        manifest={"aggregate_sha256": "a" * 64},
        execution_receipt={
            "execution_id": "b8e5992f-1d7b-43b5-974a-c35d419af023"
        },
        validation={
            "status": "passed",
            "qrel_free": True,
            "integrity_checks": {"manifest": True, "qrel_independence": True},
        },
        metadata={"dependency_aggregate_sha256": {"construction": "c" * 64}},
        _dependencies=SimpleNamespace(
            construction=SimpleNamespace(
                rankings={
                    "profile_only": profile_only,
                    "dense_profile_equal": dense_profile_equal,
                },
                manifest={"aggregate_sha256": "c" * 64},
            )
        ),
    )


def _install_fixture_loaders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mutate_on_reload: bool = False,
) -> tuple[RawRefinementEvaluationConfig, dict[str, dict[str, float]]]:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = _artifact(artifact_root)
    calls = 0

    def load_artifact(_path: str) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if mutate_on_reload and calls == 2:
            artifact.metadata["dependency_aggregate_sha256"]["construction"] = (
                "d" * 64
            )
        return artifact

    monkeypatch.setattr(
        evaluation, "load_raw_response_refinement_artifact", load_artifact
    )
    qrels = {"q1": {"A": 1.0, "B": 0.0}, "q2": {"A": 0.0, "B": 1.0}}
    monkeypatch.setattr(evaluation, "load_complete_qrels", lambda _path: qrels)
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("sealed fixture qrels\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluation,
        "LOCKED_QRELS_SHA256",
        evaluation._sha256(qrels_path),
    )
    monkeypatch.setattr(
        evaluation,
        "MATCHED_PARENT_METRICS",
        {
            name: {
                "ndcg_at_10": result.macro_ndcg_at_k,
                "mrr": result.mean_reciprocal_rank,
                "zero_ndcg_queries": sum(
                    metric.ndcg_at_k == 0.0 for metric in result.per_query.values()
                ),
            }
            for name, result in {
                name: evaluate_rankings(ranking, qrels, k=10)
                for name, ranking in artifact._dependencies.construction.rankings.items()
            }.items()
        },
    )
    receipt_path = tmp_path / "repeat.json"
    receipt_path.write_text(json.dumps(_repeat_receipt("a" * 64)), encoding="utf-8")
    return (
        RawRefinementEvaluationConfig(
            artifact_path=str(artifact_root),
            repeat_receipt_path=str(receipt_path),
            qrels_path=str(qrels_path),
        ),
        qrels,
    )


def test_evaluator_requires_exact_qrel_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, qrels = _install_fixture_loaders(monkeypatch, tmp_path)
    qrels["q1"].pop("B")
    result = evaluate_raw_response_refinement(config)
    assert result["status"] == "invalid"
    assert result["decision"]["decision"] == "invalid"
    assert "candidate inventory" in result["error"]["message"]


def test_evaluator_rejects_construction_hash_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = _install_fixture_loaders(monkeypatch, tmp_path, mutate_on_reload=True)
    result = evaluate_raw_response_refinement(config)
    assert result["decision"]["decision"] == "invalid"
    assert "changed during evaluation" in result["error"]["message"]


def test_parent_metrics_must_reproduce_hard_coded_values() -> None:
    qrels = {"q1": {"A": 1.0, "B": 0.0}}
    result = evaluate_rankings({"q1": ["A", "B"]}, qrels, k=10)
    with pytest.raises(ValueError, match="matched parent"):
        evaluation._validate_parent_metrics(
            {"profile_only": result, "dense_profile_equal": result}
        )


def test_evaluator_locks_and_rechecks_qrel_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, qrels = _install_fixture_loaders(monkeypatch, tmp_path)

    def mutate_qrels(path: str) -> dict[str, dict[str, float]]:
        Path(path).write_text("mutated qrels\n", encoding="utf-8")
        return qrels

    monkeypatch.setattr(evaluation, "load_complete_qrels", mutate_qrels)
    result = evaluate_raw_response_refinement(config)
    assert result["decision"]["decision"] == "invalid"
    assert "changed while it was being loaded" in result["error"]["message"]


def test_evaluator_emits_frozen_comparisons_and_per_query_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = _install_fixture_loaders(monkeypatch, tmp_path)
    result = evaluate_raw_response_refinement(config)
    assert result["construction_aggregate_sha256"] == "a" * 64
    assert result["construction_dependency_sha256"] == "c" * 64
    assert result["qrels_sha256"] == evaluation.LOCKED_QRELS_SHA256
    assert result["construction_unchanged"] is True
    assert set(result["metrics"]) == {
        "dense_reference",
        "profile_only",
        "dense_profile_equal",
        "raw_hierarchical",
        "raw_hierarchical_dense_equal",
    }
    assert set(result["comparisons"]) == {"primary", "secondary", "dense_diagnostics"}
    assert result["comparisons"]["primary"]["candidate"] == (
        "raw_hierarchical_dense_equal"
    )
    assert result["comparisons"]["primary"]["ndcg_at_10_delta"]["n_resamples"] == 10_000
    assert result["comparisons"]["primary"]["ndcg_at_10_delta"]["seed"] == 2026
    assert len(result["per_query_rows"]) == 10
    assert set(result["per_query_rows"][0]) == {
        "query_id",
        "system",
        "ndcg_at_10",
        "reciprocal_rank",
    }
    frozen_ndcg = evaluation.MATCHED_PARENT_METRICS["profile_only"]["ndcg_at_10"]
    result["matched_parent_metrics"]["profile_only"]["ndcg_at_10"] = -1.0
    assert (
        evaluation.MATCHED_PARENT_METRICS["profile_only"]["ndcg_at_10"]
        == frozen_ndcg
    )


def test_no_op_stops_before_qrels_are_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = _install_fixture_loaders(monkeypatch, tmp_path)
    artifact = _artifact(tmp_path / "artifact")
    artifact.rankings["raw_hierarchical"] = artifact._dependencies.construction.rankings[
        "profile_only"
    ]
    artifact.rankings["raw_hierarchical_dense_equal"] = (
        artifact._dependencies.construction.rankings["dense_profile_equal"]
    )
    monkeypatch.setattr(
        evaluation, "load_raw_response_refinement_artifact", lambda _path: artifact
    )

    def fail_if_opened(_path: str) -> None:
        raise AssertionError("qrels were opened")

    monkeypatch.setattr(evaluation, "load_complete_qrels", fail_if_opened)
    result = evaluate_raw_response_refinement(config)
    assert result["status"] == "stop-no-op"
    assert result["decision"]["decision"] == "stop"
