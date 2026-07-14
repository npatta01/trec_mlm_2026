from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from trec_mlm.profile_evaluation import (
    ProfileEvaluationConfig,
    evaluate_profile_artifacts,
    paired_interval,
    phase0_decision,
    profile_decision,
    question_type,
    win_tie_loss,
    zero_transitions,
    write_profile_evaluation,
    evaluation_systems,
)


def test_evaluation_refuses_to_open_construction_without_repeat_receipt(
    tmp_path,
) -> None:
    config = ProfileEvaluationConfig(
        construction_path=str(tmp_path / "construction"),
        qrels_path=str(tmp_path / "qrels.txt"),
        repeat_receipt_path=None,
    )
    with pytest.raises(ValueError, match="repeat receipt is required"):
        evaluate_profile_artifacts(config)


def test_evaluation_writer_refuses_output_inside_construction(tmp_path) -> None:
    construction = tmp_path / "construction"
    construction.mkdir()
    with pytest.raises(ValueError, match="outside construction"):
        write_profile_evaluation(
            {"status": "evaluated"},
            construction / "evaluation",
            construction_path=construction,
        )


def test_evaluation_systems_use_only_manifest_locked_rankings() -> None:
    construction = SimpleNamespace(
        source_rankings={
            "dense_reference": {"q": ["A", "B"]},
            "contextual_e5_control": {"q": ["B", "A"]},
        },
        rankings={"profile_only": {"q": ["A", "B"]}},
    )
    assert evaluation_systems(construction) == {
        **construction.source_rankings,
        **construction.rankings,
    }


def test_paired_interval_uses_pcg64_2026_percentiles() -> None:
    baseline = {"a": 0.0, "b": 0.5, "c": 1.0}
    candidate = {"a": 0.3, "b": 0.4, "c": 1.0}
    actual = paired_interval(baseline, candidate, n_resamples=10_000)

    differences = np.asarray([0.3, -0.1, 0.0], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(2026))
    indices = rng.integers(0, 3, size=(10_000, 3))
    expected = np.mean(differences[indices], axis=1)
    assert actual["mean_difference"] == pytest.approx(np.mean(differences))
    assert actual["lower"] == pytest.approx(np.percentile(expected, 2.5))
    assert actual["upper"] == pytest.approx(np.percentile(expected, 97.5))
    assert actual["seed"] == 2026


def test_win_tie_loss_uses_absolute_tolerance_1e12() -> None:
    baseline = {"a": 0.0, "b": 1.0, "c": 0.5}
    candidate = {"a": 2e-12, "b": 1.0 + 5e-13, "c": 0.4}
    assert win_tie_loss(baseline, candidate) == {
        "wins": 1,
        "ties": 1,
        "losses": 1,
        "tolerance": 1e-12,
    }


def test_zero_transitions_are_exhaustive() -> None:
    baseline = {"both": 0.0, "rescued": 0.0, "harmed": 0.2, "neither": 0.3}
    candidate = {"both": 0.0, "rescued": 0.1, "harmed": 0.0, "neither": 0.4}
    assert zero_transitions(baseline, candidate) == {
        "zero_both": 1,
        "dense_zero_candidate_nonzero": 1,
        "dense_nonzero_candidate_zero": 1,
        "nonzero_both": 1,
        "query_count": 4,
    }


@pytest.mark.parametrize(
    ("ndcg", "mrr", "zeros", "expected"),
    [
        (1e-15, -0.005, 115, "pass"),
        (0.0, -0.005, 115, "fail"),
        (0.1, -0.0050000001, 115, "fail"),
        (0.1, 0.0, 116, "fail"),
    ],
)
def test_phase0_full_precision_gate(
    ndcg: float, mrr: float, zeros: int, expected: str
) -> None:
    result = phase0_decision(ndcg_delta=ndcg, mrr_delta=mrr, zero_count=zeros)
    assert result["decision"] == expected


def test_profile_decision_promote_diagnostic_reject_and_invalid() -> None:
    passing = dict(
        ndcg_interval_lower=1e-15,
        ndcg_point_delta=0.01,
        mrr_delta=-0.005,
        zero_count=115,
        low_density_ndcg_delta=1e-15,
        dense_zero_rescues=1,
        integrity_checks={"repeat": True, "leakage": True},
    )
    assert profile_decision(**passing)["decision"] == "promote"
    assert profile_decision(
        **{**passing, "ndcg_interval_lower": -0.01}
    )["decision"] == "diagnostic"
    assert profile_decision(
        **{
            **passing,
            "ndcg_interval_lower": -0.01,
            "ndcg_point_delta": 0.0,
            "zero_count": 116,
            "dense_zero_rescues": 0,
        }
    )["decision"] == "reject"
    assert profile_decision(
        **{**passing, "integrity_checks": {"repeat": False}}
    )["decision"] == "reject-invalid"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" What happened?", "what"),
        ("...WHO wrote it?", "who"),
        ("¿Dónde está?", "other"),
        ("Name the author", "other"),
    ],
)
def test_question_type_classifier_is_frozen(text: str, expected: str) -> None:
    assert question_type(text) == expected
