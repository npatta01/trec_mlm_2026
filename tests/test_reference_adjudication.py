import json
from pathlib import Path
import stat

import pytest

from scripts.build_reference_comparisons import write_sealed_map
from scripts.build_reference_review_report import ranking_rows
from trec_mlm.reference_adjudication import (
    SYSTEMS,
    aggregate_rankings,
    apply_review_corrections,
    blind_system_order,
    build_blind_packet,
    merge_reviews,
    unblind_review,
    validate_review,
    validate_review_coverage,
    validate_sealed_map,
)


def test_blind_order_is_repeatable_and_varies_by_question() -> None:
    first = blind_system_order("q1", SYSTEMS, seed=2026)
    assert first == blind_system_order("q1", SYSTEMS, seed=2026)
    assert first != blind_system_order("q2", SYSTEMS, seed=2026)
    assert set(first) == set(SYSTEMS)


def test_build_packet_hides_system_names_and_writes_private_map(tmp_path: Path) -> None:
    sample = {
        "schema_version": "reference-validation-sample-v1",
        "sample_size": 1,
        "questions": [
            {
                "question_id": "q1",
                "question": "Question?",
                "cluster_id": 7,
                "strata": ["factoid"],
                "challenge_indicators": {"numeric": False},
            }
        ],
    }
    answers = {
        "perplexity": {"q1": "Alpha answer"},
        "gemini": {"q1": "Beta answer"},
        "qwen27": {"q1": "Gamma answer"},
        "gemma31": {"q1": "Delta answer"},
    }
    map_path = tmp_path / "sealed.json"
    packet, sealed = build_blind_packet(sample, answers, seed=2026)
    map_path.write_text(json.dumps(sealed), encoding="utf-8")
    map_path.chmod(0o600)

    encoded = json.dumps(packet)
    assert "Perplexity" not in encoded
    assert "Gemini" not in encoded
    assert "Qwen" not in encoded
    assert "Gemma" not in encoded
    assert set(packet[0]["answers"]) == {
        "response_a",
        "response_b",
        "response_c",
        "response_d",
    }
    assert stat.S_IMODE(map_path.stat().st_mode) == 0o600


def test_write_sealed_map_creates_private_file(tmp_path: Path) -> None:
    destination = tmp_path / "sealed.json"
    write_sealed_map(destination, {"seed": 2026})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"seed": 2026}
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not destination.with_suffix(".json.tmp").exists()


def test_build_packet_preserves_explicit_generation_failure() -> None:
    sample = {
        "questions": [
            {"question_id": "q1", "question": "Question?", "cluster_id": 1}
        ]
    }
    answers = {system: {"q1": f"answer-{index}"} for index, system in enumerate(SYSTEMS)}
    answers["qwen27"]["q1"] = None
    packet, _ = build_blind_packet(sample, answers, seed=2026)
    failed = packet[0]["generation_failures"]
    assert len(failed) == 1
    assert packet[0]["answers"][failed[0]] is None


def test_build_packet_rejects_duplicate_question_ids() -> None:
    sample = {
        "questions": [
            {"question_id": "q1", "question": "First?"},
            {"question_id": "q1", "question": "Duplicate?"},
        ]
    }
    answers = {system: {"q1": "answer"} for system in SYSTEMS}

    with pytest.raises(ValueError, match="duplicate sample question ID: q1"):
        build_blind_packet(sample, answers, seed=2026)


def _review(question_id: str) -> dict:
    labels = ["response_a", "response_b", "response_c", "response_d"]
    return {
        "question_id": question_id,
        "pointwise": {
            label: {
                "relevance": 2,
                "correctness": 4,
                "completeness": 2,
                "uncertainty": 1,
                "usefulness": 1,
                "severity": "none",
                "flags": [],
            }
            for label in labels
        },
        "pairwise": [
            {"left": left, "right": right, "outcome": "tie"}
            for i, left in enumerate(labels)
            for right in labels[i + 1 :]
        ],
        "tiers": [labels],
        "confidence": "high",
        "rationale": "All answers agree and directly answer the question.",
        "evidence_checks": [],
        "review_status": "locked_blind",
    }


def test_review_requires_all_six_pairs_and_valid_tiers() -> None:
    review = _review("q1")
    validate_review(review)
    review["pairwise"].pop()
    with pytest.raises(ValueError, match="six pairs"):
        validate_review(review)


def test_review_rejects_pairwise_result_that_contradicts_tiers() -> None:
    review = _review("q1")
    review["tiers"] = [
        ["response_a"],
        ["response_b"],
        ["response_c"],
        ["response_d"],
    ]
    review["pairwise"][0]["outcome"] = "right_better"

    with pytest.raises(ValueError, match="pairwise result contradicts tiers"):
        validate_review(review)


def test_review_correction_overlay_repairs_only_expected_pairwise_outcome() -> None:
    review = _review("q1")
    review["tiers"] = [
        ["response_a"],
        ["response_b"],
        ["response_c"],
        ["response_d"],
    ]
    for result in review["pairwise"]:
        result["outcome"] = "left_better"
    review["pairwise"][0]["outcome"] = "tie"
    corrected = apply_review_corrections(
        [review],
        [
            {
                "question_id": "q1",
                "left": "response_a",
                "right": "response_b",
                "from_outcome": "tie",
                "to_outcome": "left_better",
                "reason": "Tier order and rationale both place A above B.",
            }
        ],
    )

    assert corrected[0]["pairwise"][0]["outcome"] == "left_better"
    assert review["pairwise"][0]["outcome"] == "tie"
    validate_review(corrected[0])


def test_review_coverage_requires_every_id_once() -> None:
    validate_review_coverage(["q1", "q2"], [_review("q1"), _review("q2")])
    with pytest.raises(ValueError, match="duplicate expected review IDs.*q1"):
        validate_review_coverage(["q1", "q1"], [_review("q1")])
    with pytest.raises(ValueError, match="duplicate.*q1"):
        validate_review_coverage(
            ["q1", "q2"], [_review("q1"), _review("q1"), _review("q2")]
        )
    with pytest.raises(ValueError, match="missing.*q2"):
        validate_review_coverage(["q1", "q2"], [_review("q1")])


def test_merge_reviews_is_append_safe_and_requires_explicit_replace() -> None:
    first = _review("q1")
    assert merge_reviews([], [first]) == [first]
    with pytest.raises(ValueError, match="already reviewed.*q1"):
        merge_reviews([first], [_review("q1")])
    replacement = _review("q1")
    replacement["confidence"] = "medium"
    assert merge_reviews([first], [replacement], replace=True) == [replacement]


def test_unblind_review_preserves_tied_ranking_and_maps_every_score() -> None:
    review = _review("q1")
    review["tiers"] = [
        ["response_a", "response_c"],
        ["response_b"],
        ["response_d"],
    ]
    outcomes = {
        frozenset(("response_a", "response_b")): "left_better",
        frozenset(("response_a", "response_c")): "tie",
        frozenset(("response_a", "response_d")): "left_better",
        frozenset(("response_b", "response_c")): "right_better",
        frozenset(("response_b", "response_d")): "left_better",
        frozenset(("response_c", "response_d")): "left_better",
    }
    for result in review["pairwise"]:
        result["outcome"] = outcomes[frozenset((result["left"], result["right"]))]
    system_map = {
        "response_a": "gemini",
        "response_b": "qwen27",
        "response_c": "perplexity",
        "response_d": "gemma31",
    }
    result = unblind_review(review, system_map)
    assert result["ranking"] == [
        ["gemini", "perplexity"],
        ["qwen27"],
        ["gemma31"],
    ]
    assert set(result["pointwise"]) == set(SYSTEMS)
    assert len(result["pairwise"]) == 6


def test_validate_sealed_map_rejects_tampered_bijective_mapping() -> None:
    sample = {"questions": [{"question_id": "q1", "question": "Question?"}]}
    answers = {system: {"q1": "answer"} for system in SYSTEMS}
    _, sealed = build_blind_packet(sample, answers, seed=2026)
    mapping = sealed["maps"]["q1"]
    mapping["response_a"], mapping["response_b"] = (
        mapping["response_b"],
        mapping["response_a"],
    )

    with pytest.raises(ValueError, match="does not match deterministic blind order"):
        validate_sealed_map(sealed, ["q1"])


def test_aggregate_rankings_gives_fractional_credit_for_top_ties() -> None:
    row = {
        "question_id": "q1",
        "ranking": [["gemini", "perplexity"], ["qwen27"], ["gemma31"]],
        "pointwise": {
            system: {"severity": "none", "correctness": 4} for system in SYSTEMS
        },
        "pairwise": [],
    }
    summary = aggregate_rankings([row])
    assert summary["systems"]["gemini"]["top_tier_share"] == 1.0
    assert summary["systems"]["gemini"]["fractional_top_credit"] == 0.5
    assert summary["systems"]["perplexity"]["fractional_top_credit"] == 0.5
    assert summary["systems"]["qwen27"]["mean_tier"] == 2.0


def test_frozen_reference_report_rejects_partial_inputs(tmp_path: Path) -> None:
    packet = tmp_path / "packet.jsonl"
    rankings = tmp_path / "rankings.jsonl"
    packet.write_text(
        json.dumps({"question_id": "q1", "cluster_id": 1}) + "\n",
        encoding="utf-8",
    )
    rankings.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "ranking": [["gemini", "perplexity", "qwen27", "gemma31"]],
                "pointwise": {
                    system: {"severity": "none", "correctness": 4}
                    for system in SYSTEMS
                },
                "pairwise": [],
                "confidence": "high",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 400 unique questions"):
        ranking_rows(packet, rankings)
